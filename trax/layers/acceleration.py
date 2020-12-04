# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Accelerate layer that allows for fast computation on accelerators."""

import jax
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base


class Accelerate(base.Layer):
  """Accelerates a layer, running in data-parallel way on multiple devices.

  By default it uses all available accelerators, splits the input on the
  first (batch) axis, and runs each part on the corresponding accelerator.
  If only one accelerator is available, this layer JIT-compiles the underlying
  layer and in this way makes it run faster.

  The output is guaranteed to be the same as the output of the original layer
  if the batch dimension is divisible by the number of devices. If it is not,
  then 0-padding is added to make it divisible and the output may be affected
  if it relies on layers like batch normalization.

  This layer does not require calling `init` if the underlying layer has
  already been initialized, so it can be used as follows:
  ```
  layer = tl.Serial(...)
  layer.init(...)
  fast_layer = tl.Accelerate(layer)
  y = fast_layer(x)  # Computes y splitting x on batch and running data-parallel
  ```

  In case the weights of this layer need to be set using the weights of
  the sublayer, use the `replicate_weights` function:
  ```
  # Instead of layer.weights = new_weights:
  fast_layer.replicate_weights(new_weights)
  ```
  """

  def __init__(self, layer, n_devices=None):
    super().__init__(n_in=layer.n_in, n_out=layer.n_out)
    self._sublayers = [layer]
    self._n_devices = n_devices or fastmath.device_count()
    self._jit_pure_fn = jit_forward(
        layer.pure_fn, self._n_devices, do_mean=False)

  @property
  def sublayer(self):
    """Returns the unique sublayer managed by this layer."""
    return self._sublayers[0]

  def pure_fn(self, x, weights, state, rng, use_cache=False):
    """Calls self.sublayer.pure_fn in an accelerated way."""
    # Check if we can divide x evenly across devices.
    # Note: x can be a list/tuple because the underlying layer may take
    # its input as a list/tuple, ex: (inputs, targets, weight).
    if isinstance(x, (list, tuple)):
      remainder = x[0].shape[0] % self._n_devices
    else:
      remainder = x.shape[0] % self._n_devices
    if remainder == 0:  # If yes, run the accelerated sublayer.pure_fn.
      return self._jit_pure_fn(x, weights, state, rng)
    # If not, pad first.
    def pad(z):
      pad_widths = [(0, 0)] * len(z.shape)
      pad_widths[0] = (0, self._n_devices - remainder)
      return jnp.pad(z, pad_widths, mode='constant',
                     constant_values=z.dtype.type(0))
    padded_x = [pad(z) for z in x] if isinstance(x, (list, tuple)) else pad(x)
    # Run and un-pad.
    padded_y, state = self._jit_pure_fn(padded_x, weights, state, rng)
    if isinstance(x, (list, tuple)):
      y = tuple(padded_z[:z.shape[0]] for (padded_z, z) in zip(padded_y, x))
      y = list(y) if isinstance(x, list) else y
    else:
      y = padded_y[:x.shape[0]]
    return y, state

  def init(self, input_signature):
    """Calls self.sublayer.init and replicated on multiple devices."""
    weights, state = self.sublayer.init(input_signature, use_cache=True)
    self._weights = for_n_devices(weights, self._n_devices)
    self._state = for_n_devices(state, self._n_devices)

  def replicate_weights(self, weights):
    """Sets the weights of the sublayer and replicates them for this layer."""
    self.sublayer.weights = weights
    self._weights = for_n_devices(weights, self._n_devices)

  def replicate_state(self, state):
    """Sets the state of the sublayer and replicates it for this layer."""
    self.sublayer.state = state
    self._state = for_n_devices(state, self._n_devices)

  def _unreplicate(self, x):
    """Return a single-device version of x using the first component only."""
    if self._n_devices < 2:
      return x
    return fastmath.nested_map(lambda y: y[0], x)

  @property
  def weights(self):
    # Override the getter so it works even if only sublayer is initialized.
    if self._weights is base.EMPTY_WEIGHTS:
      self._weights = for_n_devices(self.sublayer.weights, self._n_devices)
    return self._weights

  @weights.setter
  def weights(self, weights):
    self._weights = weights
    self.sublayer.weights = self._unreplicate(weights)

  @property
  def state(self):
    # Override the getter so it works even if only sublayer is initialized.
    if self._state is base.EMPTY_STATE:
      self._state = for_n_devices(self.sublayer.state, self._n_devices)
    return self._state

  @state.setter
  def state(self, state):
    self._state = state
    self.sublayer.state = self._unreplicate(state)


def mean_or_pmean(n_devices, x, axis=None):
  """jnp.mean or pmean.

  `x` is a distributed value. Directly calling jnp.mean on `x` means stacking
  x's components together to form a large array and then doing jnp.mean on
  it. In TF, stacking `x` will introduce D2H copy, so we use a collective
  (pmean) here instead of directly calling jnp.mean for TF.

  Args:
    n_devices: number of devices.
    x: a distributed array.
    axis: the axis to reduce. Can only be 0 or None.

  Returns:
    A local array.
  """
  if fastmath.backend_name() == 'tensorflow-numpy' and n_devices > 1:
    if axis not in (None, 0):
      raise ValueError('axis can only be None or 0')
    x = fastmath.pmap(fastmath.psum)(x)[0] / n_devices
    if axis is None:
      x = jnp.mean(x)
    return x
  else:
    return jnp.mean(x, axis=axis)


def jit_forward(forward, n_devices, do_mean=True):
  """Returns a JIT-compiled forward function running on `n_devices`."""
  model_predict = _accelerate(forward, n_devices)
  # n_devices == 0 => CPU
  if n_devices < 2:
    return model_predict

  def predict(x, weights, state, rng):
    """Predict function JIT-compiled and parallelized as requested."""
    res, state = model_predict(
        reshape_by_device(x, n_devices),
        weights,
        state,
        jnp.stack(fastmath.random.split(rng, n_devices)))
    res = _combine_devices(res)
    if do_mean:
      return fastmath.nested_map(
          lambda y: mean_or_pmean(n_devices, y, axis=0), res), state
    else:
      return res, state

  return predict


def _combine_devices(x_tuple):
  """Combines multi-device tensors into a single batch."""
  def f(x):
    if len(x.shape) < 2:
      return x  # No extra batch dimension: use devices as batch, so return.
    batch_size = x.shape[0] * x.shape[1]
    return jnp.reshape(x, [batch_size] + list(x.shape[2:]))
  return fastmath.nested_map(f, x_tuple)


def _accelerate(f, n_devices):
  """JIT-compiled version of `f` running on `n_devices`."""
  if n_devices == 0:  # no accelerators - run on CPU
    return fastmath.jit(f, device=jax.devices('cpu')[0])

  if n_devices == 1:
    return fastmath.jit(f)

  return fastmath.pmap(f, axis_name='batch')


def reshape_by_device(x, n_devices):
  """Reshapes possibly nested `x` into a shape `(n_devices, ...)`."""
  def f(x):
    x_shape = list(x.shape)
    batch_size = x_shape[0]
    batch_size_per_device = batch_size // n_devices
    if batch_size_per_device * n_devices != batch_size:
      raise ValueError(f'Number of devices ({n_devices}) does not evenly '
                       f'divide batch size ({batch_size}).')
    new_shape_prefix = [n_devices, batch_size_per_device]
    return jnp.reshape(x, new_shape_prefix + x_shape[1:])
  return fastmath.nested_map(f, x)


def for_n_devices(x, n_devices):
  """Replicates/broadcasts `x` for `n_devices`."""
  def f(x):
    if n_devices > 1 and fastmath.is_backend(fastmath.Backend.JAX):
      return _multi_device_put(x)
    elif n_devices > 1:
      return jnp.broadcast_to(x, (n_devices,) + jnp.asarray(x).shape)
    else:
      return x
  return fastmath.nested_map(f, x)


def on_cpu(x):
  """Put the tensor x in CPU memory in JAX."""
  return jax.device_put(x, jax.devices('cpu')[0])


def on_accelerator(x):
  """Put the tensor x in (single) accelerator memory in JAX."""
  try:
    accelerator_devices = jax.devices('gpu')
  except RuntimeError:
    try:
      accelerator_devices = jax.devices('tpu')
    except RuntimeError:
      accelerator_devices = []
  if not accelerator_devices:
    return x
  assert len(accelerator_devices) == 1
  return jax.device_put(x, accelerator_devices[0])


def _multi_device_put(x, devices=None):
  """Memory efficient multi-device replication / broadcast in JAX.

  JAX uses a ShardedDeviceArray class that holds a list of device buffers
  on separate devices for use with pmap'd computations.  Sharded arrays
  are explicitly used to eliminate unnecessary inter-device transfer of
  memory buffers between use in pmap'd computations.  The JAX API currently
  does not have a multi-device 'put' function that copies a buffer onto
  N devices in a memory-efficient fashion, so we implement our own here.

  Args:
    x: jax DeviceArray or numpy ndarray to be replicated.
    devices: a jax.devices() list or subset thereof of devices to
      replicate onto.  Should match the list passed to any pmaps
      ingesting the replicated array.

  Returns:
    A ShardedDeviceArray with
    dtype = x.dtype and shape = (n_devices,) + x.shape
    that's backed by replicated device_buffers on each local device.
  """
  # Calculate the abstract shape of the replicated array.
  if not devices:
    devices = jax.local_devices()
  # The code below is equivalent to:
  #   jax.api.device_put_sharded(len(devices) * [x], devices)
  # but it does one PCI transfer and later uses ICI.
  # TODO(lukaszkaiser): remove once JAX has a core function to do the same.
  aval = jax.core.unmapped_aval(len(devices), 0,
                                jax.core.raise_to_shaped(jax.core.get_aval(x)))
  buf, = jax.xla.device_put(x, devices[0])  # assuming single-buf repr
  rest_bufs = [buf.copy_to_device(d) for d in devices[1:]]
  return jax.pxla.ShardedDeviceArray(aval, [buf, *rest_bufs])
