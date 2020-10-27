# coding=utf-8
# Copyright 2021 The Trax Authors.
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
"""Core layer types and key functions used by various layers."""

from absl import logging
import numpy as np
import tensorflow as tf

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn


# The output tensor has the same shape as the input tensor, except for the size
# of the last dimension.
@assert_shape('...a->...b')
class Dense(base.Layer):
  """A dense (a.k.a. fully-connected, affine) layer.

  Dense layers are the prototypical example of a trainable layer, i.e., a layer
  with trainable weights. Each node in a dense layer computes a weighted sum of
  all node values from the preceding layer and adds to that sum a node-specific
  bias term. The full layer computation is expressed compactly in linear
  algebra as an affine map `y = Wx + b`, where `W` is a matrix and `y`, `x`,
  and `b` are vectors. The layer is trained, or "learns", by updating the
  values in `W` and `b`.

  Less commonly, a dense layer can omit the bias term and be a pure linear map:
  `y = Wx`.
  """

  def __init__(self,
               n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True,
               use_bfloat16=False):
    """Returns a dense (fully connected) layer of width `n_units`.

    A dense layer maps collections of `R^m` vectors to `R^n`, where `n`
    (`= n_units`) is fixed at layer creation time, and `m` is set at layer
    initialization time.

    Args:
      n_units: Number of nodes in the layer, also known as the width of the
          layer.
      kernel_initializer: Function that creates a matrix of (random) initial
          connection weights `W` for the layer.
      bias_initializer: Function that creates a vector of (random) initial
          bias weights `b` for the layer.
      use_bias: If `True`, compute an affine map `y = Wx + b`; else compute
          a linear map `y = Wx`.
      use_bfloat16: If `True`, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
    """
    super().__init__(name=f'Dense_{n_units}')
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._use_bias = use_bias
    self._use_bfloat16 = use_bfloat16

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input, except the final dimension
      is the layer's `n_units` value.
    """
    if self._use_bias:
      if not isinstance(self.weights, (tuple, list)):
        raise ValueError(f'Weights should be a (w, b) tuple or list; '
                         f'instead got: {self.weights}')
      w, b = self.weights
      return jnp.dot(x, w) + b  # Affine map.
    else:
      w = self.weights
      return jnp.dot(x, w)  # Linear map.

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights.

    Weights are a `(w, b)` tuple for layers created with `use_bias=True` (the
    default case), or a `w` tensor for layers created with `use_bias=False`.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on.
    """
    shape_w = (input_signature.shape[-1], self._n_units)
    shape_b = (self._n_units,)
    rng_w, rng_b = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(shape_w, rng_w)
    if self._use_bfloat16:
      w = w.astype(jnp.bfloat16)

    if self._use_bias:
      b = self._bias_initializer(shape_b, rng_b)
      if self._use_bfloat16:
        b = b.astype(jnp.bfloat16)
      self.weights = (w, b)
    else:
      self.weights = w


# The output tensor has the same shape as the input tensor, but with added
# dimension at the end. This dimension size corresponds to embedding depth.
@assert_shape('...->...d')
class Embedding(base.Layer):
  """Trainable layer that maps discrete tokens/IDs to vectors.

  Embedding layers are commonly used to map discrete data, like words in NLP,
  into vectors. Here is a canonical example::

      vocab_size = 5
      word_ids = np.array([1, 2, 3, 4], dtype=np.int32)  # word_ids < vocab_size
      embedding_layer = tl.Embedding(vocab_size, 32)
      embedding_layer.init(trax.shapes.signature(word_ids))
      embedded = embedding_layer(word_ids)  # embedded.shape = (4, 32)
  """

  def __init__(self,
               vocab_size,
               d_feature,
               use_bfloat16=False,
               kernel_initializer=init.ScaledInitializer(
                   out_dim=-1, in_dim=-2, scale=1., mode='fan_out',
                   distribution='uniform')):
    """Returns an embedding layer with given vocabulary size and vector size.

    The layer clips input values (token IDs) to the range `[0, vocab_size)`.
    That is, negative token IDs all clip to `0` before being mapped to a
    vector, and token IDs with value `vocab_size` or greater all clip to
    `vocab_size - 1` before being mapped to a vector.

    Args:
      vocab_size: Size of the input vocabulary. The layer will assign a unique
        vector to each id in `range(vocab_size)`.
      d_feature: Dimensionality/depth of the output vectors.
      use_bfloat16: If `True`, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
      kernel_initializer: Function that creates (random) initial vectors for
        the embedding.
    """
    # TODO(jonni): is the clipping behavior what we want going forward?
    super().__init__(name=f'Embedding_{vocab_size}_{d_feature}')
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._use_bfloat16 = use_bfloat16
    self._kernel_initializer = kernel_initializer

  def forward(self, x):
    """Returns embedding vectors corresponding to input token IDs.

    Args:
      x: Tensor of token IDs.

    Returns:
      Tensor of embedding vectors.
    """
    embedded = jnp.take(self.weights, x, axis=0)
    if self._use_bfloat16:  # Return float32 activations w/ bfloat16 weights.
      embedded = embedded.astype(jnp.float32)
    return embedded

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    del input_signature
    shape_w = (self._vocab_size, self._d_feature)
    # TODO(lukaszkaiser): do we split self.rng for consistency? Add a method?
    w = self._kernel_initializer(shape_w, self.rng)
    if self._use_bfloat16:
      w = w.astype(jnp.bfloat16)
    self.weights = w


@assert_shape('...->...')  # The output and input shapes are the same.
class Dropout(base.Layer):
  """A layer that stochastically ignores a subset of inputs each training step.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  The parameter `shared_axes` allows to specify a list of axes on which
  the mask will be shared: we will use size 1 on those axes for dropout mask
  and broadcast it. Sharing reduces randomness, but can save memory.

  This layer is active only during training (`mode='train'`). In other
  circumstances it is a no-op.

  Originally introduced in the paper "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting" available under the following link:
  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
  """

  def __init__(self, rate=0.0, shared_axes=None, mode='train'):
    """Creates a dropout layer with the given target drop rate.

    Args:
      rate: Stochastic rate (probability) for dropping an activation value
          from the preceding layer (setting it to zero).
      shared_axes: List of axes on which the mask is shared.
      mode: If `'train'`, this layer will perform dropout; else, it will pass
          all values through unaltered.
    """
    super().__init__()
    self._initial_rate = rate
    self._shared_axes = [] if shared_axes is None else shared_axes
    self._mode = mode

  def init_weights_and_state(self, input_signature):
    """Sets layer-specific internal state."""
    del input_signature
    self.state = jnp.array(self._initial_rate)

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of activations.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    if self._mode != 'train':
      return x
    state, rng = self.state, self.rng
    rate = self._initial_rate
    if isinstance(state, dict) and self._name in state:
      rate = state[self._name]
    mask_shape = list(x.shape)
    for axis in self._shared_axes:
      mask_shape[axis] = 1
    keep_prob = 1.0 - rate
    keep = fastmath.random.bernoulli(rng, keep_prob, tuple(mask_shape))
    mask = keep.astype(x.dtype) / keep_prob
    return x * mask


class Weights(base.Layer):
  """Learnable weights as a layer.

  It takes no input and returns a single tensor: weights.
  """

  def __init__(self, initializer, shape=tuple(), use_bfloat16=False):
    """Returns a learnable tensor of shape `shape`.

    Args:
      initializer: Function taking shape and rng as arguments.
      shape: Shape of the learnable weights.
      use_bfloat16: If `True`, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
    """
    super().__init__(name=f'Weights_{shape}', n_in=0, n_out=1)
    self._shape = shape
    self._initializer = initializer
    self._use_bfloat16 = use_bfloat16

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor with previously specified shape and dtype.
    """
    del x  # Unused. There is no input to this layer.
    return self.weights

  def init_weights_and_state(self, input_signature):
    """Returns newly initialized weights for this layer.

    Weights is a single  `w` tensor with previously specified shape.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on. Unused.
    """
    del input_signature  # Unused. There is no input to this layer.
    self.weights = self._initializer(self._shape, self.rng)
    if self._use_bfloat16:
      self.weights = self.weights.astype(jnp.bfloat16)


def PrintShape(n_in=1, msg=''):
  """Prints the shapes of `n_in` inputs and returns then unchanged."""
  def Fwd(xs):
    shapes_and_dtypes = ', '.join([str(x.shape) + f'[{x.dtype}]' for x in xs])
    info = f'PrintShape: {msg}: [{shapes_and_dtypes}]'
    print(info)
    logging.info(info)
    return xs
  return base.PureLayer(Fwd, n_in=n_in, n_out=n_in, name=f'PrintShape_{n_in}')


class SummaryImage(base.Layer):
  """A layer receiving a tensor, and adding it to TensorBoard as an image.

  It takes an input and returns it unchanged. It stores this input as a state to
  be used as a metric in TensorBoard.
  It converts a tensor to a scalar by running a given aggregation function (mean
  by default). On TensorBoard, results for each device will be reported
  separately.
  """

  def __init__(self, name, n_in, num_summaries=5,
               recover_fn=None):
    """Takes a tensor and returns it.

    Args:
      name: Name of the metric to be reported.
      n_in: Number of inputs.
      num_summaries: Number of images to show.
      recover_fn: the function for converting a tensor to a dipslayable image.
    """
    super().__init__(name=f'Summary_{name}', n_in=n_in, n_out=n_in)
    name = 'summary_' + name
    self._name = name
    self._num_summaries = num_summaries
    self._recover_fn = recover_fn

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor with previously specified shape and dtype.
    """
    self.state = {}
    batch_size = x[0].shape[0]
    num_images = min(self._num_summaries, batch_size)
    for s in range(num_images):
      images = []
      for i in range(self._n_in):
        images.append(
            self._recover_fn(x[i][s]) if self._recover_fn else x[i][s])
      self.state[self._name + str(s)] = jnp.concatenate(images, axis=0)
    return x[:self._n_in]

  def init_weights_and_state(self, input_signature):
    """Returns newly initialized weights for this layer.

    Weights is a single  `w` tensor with previously specified shape.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on. Unused.
    """
    del input_signature  # Unused.
    self.weights = ()
    self.state = {self._name: jnp.array(0.)}


class SummaryScalar(base.Layer):
  """A layer receiving a tensor, and adding it to TensorBoard as a scalar.

  It takes an input and returns it unchanged. It stores this input as a state to
  be used as a metric in TensorBoard.
  It converts a tensor to a scalar by running a given aggregation function (mean
  by default). On TensorBoard, results for each device will be reported
  separately.
  """

  def __init__(self, name, aggregation_fun=jnp.mean):
    """Takes a tensor and returns it.

    Args:
      name: Name of the metric to be reported.
      aggregation_fun: Aggregation function to be used.
    """
    super().__init__(name=f'Summary_{name}', n_in=1, n_out=1)
    name = 'summary_' + name
    self._name = name
    self._aggregation_fun = aggregation_fun

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor with previously specified shape and dtype.
    """
    self.state = {self._name: self._aggregation_fun(x)}
    return x

  def init_weights_and_state(self, input_signature):
    """Returns newly initialized weights for this layer.

    Weights is a single  `w` tensor with previously specified shape.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on. Unused.
    """
    del input_signature  # Unused.
    self.weights = ()
    self.state = {self._name: jnp.array(0.)}


class RandomUniform(base.Layer):
  """Layer returning a tensor with random values distributed uniformly."""

  def __init__(self, min_val=0.0, max_val=1.0, shape=(), dtype=jnp.float32,
               sync=False):
    """Layer returning a tensor with random values distributed uniformly.

    Args:
      min_val: Lower end of uniform distribution.
      max_val: Upper end of uniform distribution.
      shape: Shape of the tensor to return. Values are sampled independently.
      dtype: Type of value to return.
      sync: Whether to synchronise `rng` across devices.
    """
    super().__init__(n_in=0, n_out=1)
    self._min_val = min_val
    self._max_val = max_val
    self._shape = shape
    self._dtype = dtype
    self._sync = sync

  def forward(self, xs):
    """Executes this layer as part of a forward pass through the model.

    Args:
      xs: Unused tensors.

    Returns:
      Random uniform tensor of the shape and type specified in constructor.
    """
    rng = self._get_conditionally_synced_rng()
    result = fastmath.random.uniform(
        rng, self._shape, self._dtype, self._min_val, self._max_val)
    return result

  def _get_conditionally_synced_rng(self):
    if self._sync and fastmath.device_count() > 1:
      return fastmath.psum(self.rng, 'batch')
    else:
      return self.rng


class LocallyConnected1d(base.Layer):
  """Locally-connected layer for 1D inputs.

  The LocallyConnected1d layer applies a different set of filters to each patch
  of the input. This is similar to applying a convolution layer, except that
  locally-connected layer uses a different set of weights for each patch.

  The size of patch is determined by the kernel size. The stride is currently
  not modifiable and set to one. This means for the input of shape (..., L, D)
  the output shape for paddings 'SAME' and 'WRAP' will be (..., L, filters) and
  for padding 'VALID' (..., L-kernel_size+1, filters); where L is the number of
  "pixels" or "steps" in the input, D is the size of the embedding.

  Note that, since the weights for different patches are not shared, the number
  of "pixels" or "steps" cannot change after calling init_weights_and_state.
  This is because each "pixel" is assigned its own set of weights.
  """

  def __init__(self, filters, kernel_size,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True, padding='VALID'):
    """Returns a locally-connected conv-like layer.

    Args:
      filters: Number of output filters in the convolution.
      kernel_size: A length of the convolution window. Must be an odd number.
      kernel_initializer: Function that creates a matrix of (random) initial
          connection weights `W` for the layer.
      bias_initializer: Function that creates a vector of (random) initial
          bias weights `b` for the layer.
      use_bias: If `True`, the layer uses a bias vector.
      padding: The type of padding to use; must be 'VALID', 'SAME', or 'WRAP'.
    """
    super().__init__(name=f'LocallyConnected1d_{filters}_{kernel_size}')
    self._filters = filters
    self._kernel_size = kernel_size
    assert self._kernel_size % 2 == 1  # kernel size has to be odd
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._use_bias = use_bias
    self._padding = padding

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input, except the final dimension
      is the layer's `filters` value, and the second to last dimension is
      shrinked if 'VALID' padding is used with kernel_size bigger than one.
    """
    if self._use_bias:
      if not isinstance(self.weights, (tuple, list)):
        raise ValueError(f'Weights should be a (w, b) tuple or list; '
                         f'instead got: {self.weights}')
      w, b = self.weights
    else:
      w = self.weights

    linear_results_before_shifting = jnp.einsum(
        '...lp,lkpd->...lkd', x, w)
    # TODO(jaszczur): this could be run after padding for better efficiency

    if self._kernel_size == 1:
      # With kernel size 1 we don't have to split or shift anything.
      linear_result = jnp.squeeze(linear_results_before_shifting, axis=-2)
    else:
      # We computed a result for every "pixel", but each direction from the
      # receptive field (there are 'self._kernel_size' such directions) must be
      # shifted by a different amount. The easiest way to do it is to split
      # the tensor to 'self._kernel_size' smaller tensors, shift each one
      # appropriately, and then sum them together.
      split_shifting_linear_results = jnp.split(
          linear_results_before_shifting, self._kernel_size, axis=-2)

      for i in range(self._kernel_size):
        # Each tensor has to be shifted a different amount.
        if self._padding == 'WRAP':
          # We can shift by padding and cutting. With 'wrap' padding we
          # essentially have a torus.
          padding = [(0, 0) for i in split_shifting_linear_results[i].shape]
          padding[-3] = ((self._kernel_size - 1) - i, i)
          split_shifting_linear_results[i] = jnp.pad(
              split_shifting_linear_results[i], padding, mode='wrap')
          split_shifting_linear_results[i] = split_shifting_linear_results[i][
              ..., (self._kernel_size-1)//2:-(self._kernel_size-1)//2, :, :]
        elif self._padding == 'SAME':
          # We can shift by padding and cutting.
          padding = [(0, 0) for i in split_shifting_linear_results[i].shape]
          padding[-3] = ((self._kernel_size - 1) - i, i)
          split_shifting_linear_results[i] = jnp.pad(
              split_shifting_linear_results[i], padding)
          split_shifting_linear_results[i] = split_shifting_linear_results[i][
              ..., (self._kernel_size-1)//2:-(self._kernel_size-1)//2, :, :]
          # TODO(jaszczur): improve efficiency by not padding things to cut
        elif self._padding == 'VALID':
          # We don't need to shift - just cut the leftmost and rightmost values.
          cut_left = (self._kernel_size - 1) - i
          cut_right = split_shifting_linear_results[i].shape[-3] - i
          split_shifting_linear_results[i] = split_shifting_linear_results[i][
              ..., cut_left:cut_right, :, :]
        else:
          raise ValueError(f'Invalid padding {self._padding}')
      # After shifting.
      shifted_linear_results = jnp.concatenate(split_shifting_linear_results,
                                               axis=-2)
      linear_result = jnp.sum(shifted_linear_results, axis=-2)

    if self._use_bias:
      return linear_result + b
    else:
      return linear_result

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights.

    Weights are a `(w, b)` tuple for layers created with `use_bias=True` (the
    default case), or a `w` tensor for layers created with `use_bias=False`.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on.
    """
    shape_w = (input_signature.shape[-2], self._kernel_size,
               input_signature.shape[-1], self._filters)
    if self._padding == 'VALID':
      shape_b = (input_signature.shape[-2] - self._kernel_size + 1,
                 self._filters,)
    else:
      shape_b = (input_signature.shape[-2], self._filters,)

    rng_w, rng_b = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(shape_w, rng_w, nonreceptive_dims=[0])

    if self._use_bias:
      b = self._bias_initializer(shape_b, rng_b)
      self.weights = (w, b)
    else:
      self.weights = w


def Flatten(n_axes_to_keep=1):
  """Returns a layer that combines one or more trailing axes of a tensor.

  Flattening keeps all the values of the input tensor, but reshapes it by
  collapsing one or more trailing axes into a single axis. For example, a
  `Flatten(n_axes_to_keep=2)` layer would map a tensor with shape
  `(2, 3, 5, 7, 11)` to the same values with shape `(2, 3, 385)`.

  Args:
    n_axes_to_keep: Number of leading axes to leave unchanged when reshaping;
        collapse only the axes after these.
  """
  layer_name = f'Flatten_keep{n_axes_to_keep}'
  def f(x):  # pylint: disable=invalid-name
    in_rank = len(x.shape)
    if in_rank <= n_axes_to_keep:
      raise ValueError(f'Input rank ({in_rank}) must exceed the number of '
                       f'axes to keep ({n_axes_to_keep}) after flattening.')
    shape = x.shape
    if isinstance(shape, tf.TensorShape):
      shape = tuple(shape.as_list())
    return jnp.reshape(x, (shape[:n_axes_to_keep] + (-1,)))
  return Fn(layer_name, f)


def LogSoftmax(axis=-1):
  """Returns a layer that applies log softmax along one tensor axis.

  Note that the implementation actually computes x - LogSumExp(x),
  which is mathematically equal to LogSoftmax(x).

  `LogSoftmax` acts on a group of values and normalizes them to look like a set
  of log probability values. (Probability values must be non-negative, and as
  a set must sum to 1. A group of log probability values can be seen as the
  natural logarithm function applied to a set of probability values.)

  Args:
    axis: Axis along which values are grouped for computing log softmax.
  """
  return Fn('LogSoftmax', lambda x: log_softmax(x, axis=axis))


def LogSumExp(axis=-1):
  """Returns a layer that computes log(sum(exp(x))) along one tensor axis.

  Args:
    axis: Axis along which values are grouped for computing log-sum-exp.
  """
  return Fn('LogSumExp',
            lambda x: fastmath.logsumexp(x, axis=axis, keepdims=True))


def Softmax(axis=-1):
  """Returns a layer that applies softmax along one tensor axis.

  `Softmax` acts on a group of values and normalizes them to look like a set
  of probability values. (Probability values must be non-negative, and as a
  set must sum to 1.)

  Args:
    axis: Axis along which values are grouped for computing softmax.
  """
  return Fn('Softmax',
            lambda x: jnp.exp(log_softmax(x, axis=axis)))


def ToFloat():
  """Returns a layer that changes the dtype of a tensor to `float32`."""
  return Fn('ToFloat', lambda x: x.astype(np.float32))


def Mean(axis=-1, keepdims=False):
  """Returns a layer that computes mean values using one tensor axis.

  `Mean` uses one tensor axis to form groups of values and replaces each group
  with the mean value of that group. The resulting values can either remain
  in their own size 1 axis (`keepdims=True`), or that axis can be removed from
  the overall tensor (default `keepdims=False`), lowering the rank of the
  tensor by one.

  Args:
    axis: Axis along which values are grouped for computing a mean.
    keepdims: If `True`, keep the resulting size 1 axis as a separate tensor
        axis; else, remove that axis.
  """
  return Fn('Mean', lambda x: jnp.mean(x, axis=axis, keepdims=keepdims))


def Min(axis=-1, keepdims=False):
  """Returns a layer that applies min along one tensor axis.

  Args:
    axis: Axis along which values are grouped for computing minimum.
    keepdims: If `True`, keep the resulting size 1 axis as a separate tensor
        axis; else, remove that axis.
  """
  return Fn('Min', lambda x: jnp.min(x, axis, keepdims=keepdims))


def Max(axis=-1, keepdims=False):
  """Returns a layer that applies max along one tensor axis.

  Args:
    axis: Axis along which values are grouped for computing maximum.
    keepdims: If `True`, keep the resulting size 1 axis as a separate tensor
        axis; else, remove that axis.
  """
  return Fn('Max', lambda x: jnp.max(x, axis, keepdims=keepdims))


def Sum(axis=None, keepdims=False):
  """Returns a layer that computes sums using one tensor axis.

  `Sum` uses one tensor axis to form groups of values and replaces each group
  with the sum of that group. The resulting sum values can either remain in
  their own size 1 axis (`keepdims=True`), or that axis can be removed from the
  overall tensor (default `keepdims=False`), lowering the rank of the tensor by
  one.

  Args:
    axis: Axis along which values are grouped for computing a sum; if None,
        compute sum over all elements in tensor.
    keepdims: If `True`, keep the resulting size 1 axis as a separate tensor
        axis; else, remove that axis.
  """
  return Fn('Sum', lambda x: jnp.sum(x, axis=axis, keepdims=keepdims))


def ThresholdToBinary(threshold=.5):
  """Returns a layer that thresholds inputs to yield outputs in {0, 1}."""
  def f(model_output):  # pylint: disable=invalid-name
    return (model_output > threshold).astype(jnp.int32)
  return Fn('ThresholdToBinary', f)


def ArgMax(axis=-1):
  """Returns a layer that calculates argmax along the given axis."""
  def f(model_output):  # pylint: disable=invalid-name
    return jnp.argmax(model_output, axis=axis)
  return Fn('ArgMax', f)


@assert_shape('...->...')  # The output and input shapes are the same.
def Negate():
  """Returns a layer that computes the element-wise negation of a tensor."""
  return Fn('Negate', lambda x: -x)


@assert_shape('...->...')  # The output and input shapes are the same.
def StopGradient():
  """Returns an identity layer with a stop gradient."""
  return Fn('StopGradient', lambda x: fastmath.stop_gradient(x))  # pylint: disable=unnecessary-lambda


def one_hot(x, n_categories, dtype=jnp.float32):  # pylint: disable=invalid-name
  """Makes a one-hot array (n+1 dims) from an int-categorical array (n dims)."""
  indices_less_than_n = jnp.arange(n_categories)
  return jnp.array(x[..., jnp.newaxis] == indices_less_than_n, dtype)


def log_softmax(x, axis=-1):  # pylint: disable=invalid-name
  """Transforms activation vectors to log-probability vectors.

  Log probability vectors are derived by, in effect, applying softmax to raw
  activation vectors and then applying log element-wise. The actual
  implementation uses a mathematically valid simplification of this.

  Args:
    x: An ndarray with activation vectors along the given axis.
    axis: Axis along which values are grouped for computing log softmax.

  Returns:
    An ndarray containing log-probability vectors derived from the raw
    activation vectors in `x`.
  """
  return x - fastmath.logsumexp(x, axis=axis, keepdims=True)


def log_gaussian_pdf(x, mu, sigma):  # pylint: disable=invalid-name
  """Returns `log N(x | mu, sigma)`.

  Args:
    x: <tbd>
    mu: <tbd>
    sigma: <tbd>
  """
  a = mu.shape[-1] * jnp.log(2 * jnp.pi)
  _, b = jnp.linalg.slogdet(sigma)
  y = jnp.linalg.solve(sigma, x - mu)
  y = jnp.expand_dims(y, axis=-1)
  xm = jnp.expand_dims(x - mu, axis=-2)
  c = jnp.matmul(xm, y)
  c = jnp.squeeze(jnp.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def log_gaussian_diag_pdf(x, mu, diag_sigma):  # pylint: disable=invalid-name
  """Returns `log N(x | mu, eye(diag_sigma))`.

  Args:
    x: <tbd>
    mu: <tbd>
    diag_sigma: <tbd>
  """
  a = mu.shape[-1] * jnp.log(2 * jnp.pi)
  b = jnp.sum(jnp.log(diag_sigma), axis=-1)
  y = x - mu / diag_sigma
  y = jnp.expand_dims(y, axis=-1)
  xm = jnp.expand_dims(x - mu, axis=-2)
  c = jnp.matmul(xm, y)
  c = jnp.squeeze(jnp.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def multigaussian_loss(preds, targets, ngauss=1):  # pylint: disable=invalid-name
  """Returns a mixture of gaussians loss.

  Args:
    preds: <tbd>
    targets: <tbd>
    ngauss: <tbd>
  """
  ndims = targets.shape[-1]
  logits = preds[:, :ngauss]
  mus = preds[:, ngauss:ngauss*(ndims + 1)]
  sigmas = preds[:, ngauss(ndims + 1):]
  sigmas = sigmas * sigmas + 1e-6  # Make positive.
  loglogits = logits - fastmath.logsumexp(logits, axis=-1, keepdims=True)
  mus = jnp.reshape(mus, [-1, ngauss, ndims])
  sigmas = jnp.reshape(sigmas, [-1, ngauss, ndims])
  targets = jnp.reshape(targets, [-1, 1, ndims])
  glogprobs = log_gaussian_diag_pdf(targets, mus, sigmas)
  return fastmath.logsumexp(loglogits + glogprobs, axis=-1)


# TODO(jonni): Rename to log_softmax_sample.
def logsoftmax_sample(log_probs, temperature=1.0):  # pylint: disable=invalid-name
  """Returns a sample from a log-softmax output, with temperature.

  Args:
    log_probs: Logarithms of probabilities (often coming from LogSoftmax)
    temperature: For scaling before sampling (1.0 = default, 0.0 = pick argmax)
  """
  # This is equivalent to sampling from a softmax with temperature.
  u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
  g = -np.log(-np.log(u))
  return np.argmax(log_probs + g * temperature, axis=-1)
