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
"""Trainer class that accelerates running optimizers on layers."""
import functools
from absl import logging
from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp


class Trainer(object):
  """Accelerates running an optimizer on a Trax layer returning a scalar loss.

  By default it uses all available accelerators, runs an accelerated version
  of the loss layer forward and then updates its weights using the optimizer.
  If only one accelerator is available, this JIT-compiles the underlying
  computation and in this way makes it run faster.

  We assume that both the loss layer (usually model with loss) and the optimizer
  have already been initialized.

  The output after running the `one_step` function is just the loss from the
  loss layer and optimizer statisics but, as a side effect, it also updates
  the weights of the loss layer and the slots of the optimizer.
  """

  def __init__(self, loss_layer, optimizer, n_devices=None):
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._n_devices = n_devices or fastmath.device_count()

    # optimizer slots and opt_params may need to be replicated
    self._slots, self._opt_params = tl.for_n_devices(
        (self._optimizer.slots, self._optimizer.opt_params), self._n_devices)

    # accelerated version of loss layer to replicate weights and state
    self._accelerated_loss_layer = tl.Accelerate(
        loss_layer, n_devices=n_devices)

    # Signature:
    # (batch, weights, state, rng) -> ((loss, state), gradients)
    self._forward_and_backward_fn = (
        fastmath.value_and_grad(
            loss_layer.pure_fn,
            argnums=1,  # arg1 of pure_fn: weights
            has_aux=True))  # return (loss, state), gradients

    # Signature:
    # (weights, slots), step, opt_params, batch, state, rng ->
    # (weights, slots), state, stats
    self._accelerated_update_fn = (
        _accelerate_update_fn(
            self._forward_and_backward_fn,
            self._optimizer,
            n_devices=self._n_devices,
            accelerate=True,
        )
    )

  @property
  def loss_layer(self):
    """Returns the loss layer used to initialize this class."""
    return self._loss_layer

  @property
  def accelerated_loss_layer(self):
    """Returns the accelerated loss layer managed by this class."""
    return self._accelerated_loss_layer

  @property
  def optimizer(self):
    """Returns the optimizer used to initialize this class."""
    return self._optimizer

  def one_step(self, batch, rng, step=0, learning_rate=None):
    """Updates loss layer weights/state and optimizer slots by running one step.

    Args:
      batch: Batch of data to use for optimization.
      rng: Random number generator to use for running this step.
      step: Which step of the training are we running.
      learning_rate: Learning rate to use instead of the default one.

    Returns:
      Tuple (loss, stats) with new values from one step
      of training, where stats are current optimizer statistics.
    """
    # Update the learning rate if needed.
    if learning_rate is not None:
      self._opt_params['learning_rate'] = tl.for_n_devices(
          learning_rate, self._n_devices)

    # batch needs to be split across the local devices -- the difference
    # between _for_n_devices and _reshape_by_device is that the latter splits
    # the batch dim to batch // n_devices, vs _for_n_devices
    # broadcasts/replicates to n_devices dimension.
    if self._n_devices > 1:
      batch = tl.reshape_by_device(batch, self._n_devices)

    # separate rng needs to be created for each device
    if self._n_devices > 1:
      rng = jnp.stack(fastmath.random.split(rng, self._n_devices))

    weights = self._accelerated_loss_layer.weights
    state = self._accelerated_loss_layer.state
    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      # Prints every power of two, if debugging is enabled.
      logging.info('step[%d]', step)
      logging.info('opt_params[%s]', self._opt_params)
      logging.info('slots[%s]', self._slots)
      logging.info('weights[%s]', weights)
      logging.info('state[%s]', state)

    # NOTE: stats is a replicated dictionary of key to jnp arrays.
    (new_weights, new_slots), new_state, stats = self._accelerated_update_fn(
        (weights, self._slots), step, self._opt_params, batch, state, rng)

    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      logging.info('updated weights[%s]', new_weights)
      logging.info('stats[%s]', stats)

    self._accelerated_loss_layer.weights = new_weights
    self._accelerated_loss_layer.state = new_state
    self._slots = new_slots
    self._optimizer.slots = self._unreplicate(self._slots)
    return stats['loss'], stats

  def _unreplicate(self, x):
    if self._n_devices == 1:
      return x
    return fastmath.nested_map(lambda x: x[0], x)


# Returns a function with the following signature:
# (weights, slots), step, opt_params, batch, state, rng ->
# (weights, slots), state, stats
def _accelerate_update_fn(forward_and_backward_fn,
                          optimizer,
                          n_devices,
                          accelerate=True):
  """Accelerate the given forward_and_backward_fn function."""
  if n_devices == 1:
    def single_device_update_fn(
        weights_and_slots, step, opt_params, batch, state, rng):
      step = jnp.array(step, dtype=jnp.int32)  # Needed in TFNP backend.
      weights, slots = weights_and_slots
      (loss, state), gradients = forward_and_backward_fn(
          batch, weights, state, rng)
      weights, slots, stats = optimizer.tree_update(
          step, gradients, weights, slots, opt_params)
      stats['loss'] = loss
      return (weights, slots), state, stats
    if accelerate:
      # TODO(afrozm): Find out the status of buffer donation on GPUs, then do
      #  donate_argnums=(0,).
      single_device_update_fn = fastmath.jit(single_device_update_fn)
    return single_device_update_fn

  # More than one device (core), i.e. all of TPU configurations etc.
  assert n_devices > 1, f'{n_devices} should be greater than 1.'

  @functools.partial(fastmath.pmap, axis_name='batch', donate_argnums=(0,))
  def _multi_device_update_fn(
      weights_and_slots, step, opt_params, batch, state, rng):
    # We assume all tensors have the first dimension = n_devices.
    weights, slots = weights_and_slots
    (loss, state), gradients = forward_and_backward_fn(
        batch, weights, state, rng)

    # gradients now need to be summed over all the devices across different host
    # machines, n_devices is only the number of devices on *this* host machine.
    gradients = fastmath.psum(gradients, 'batch')
    n_devices_total = fastmath.psum(jnp.array(1.0), 'batch')
    # Average across hosts.
    gradients = fastmath.nested_map(lambda g: g / n_devices_total, gradients)

    weights, slots, stats = optimizer.tree_update(
        step, gradients, weights, slots, opt_params)
    stats['loss'] = loss
    return (weights, slots), state, stats

  def multi_device_update_fn(
      weights_and_slots, step, opt_params, batch, state, rng):
    # Need to replicate step to n_devices leading dimension.
    return _multi_device_update_fn(weights_and_slots,
                                   jnp.repeat(step, n_devices), opt_params,
                                   batch, state, rng)

  return multi_device_update_fn


