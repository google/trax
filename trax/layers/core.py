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
"""Trax layers library."""

import numpy as np

from trax import math
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.base import Fn
from trax.math import numpy as jnp


class Dense(base.Layer):
  """A dense (a.k.a. fully-connected, affine) layer.

  Dense layers are the prototypical example of a trainable layer, i.e., a layer
  with trainable "weights". Each node in a dense layer computes a weighted sum
  of all node values from the preceding layer and adds to that sum a
  node-specific "bias" term. The full layer computation is expressed compactly
  in linear algebra as an affine map $$y = W x + b$$, where $$W$$ is a matrix
  and $$y$$, $$x$$, and $$b$$ are vectors. The layer is trained, or "learns",
  by updating the values in $$W$$ and $$b$$.

  Less commonly, a dense layer can be used as a pure linear map, by omitting
  the bias term: $$y = W x$$.
  """

  def __init__(self,
               n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True):
    """Returns a dense / fully connected layer of width `n_units`.

    Args:
      n_units: Number of nodes in the layer, also known as the "width" of the
          layer.
      kernel_initializer: Function that creates a matrix of (random) initial
          connection weights ($$W$$) for the layer.
      bias_initializer: Function that creates a vector of (random) initial
          bias weights ($$b$$) for the layer.
      use_bias: If True, compute an affine map: $$y = W x + b$$; else compute
          a linear map: $$y = W x$$.
    """
    super().__init__(name=f'Dense_{n_units}')
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._use_bias = use_bias

  def forward(self, x, weights):
    if len(weights) != 2:
      raise ValueError(f'Weights has length {len(weights)}; should instead '
                       f'have two elements: w, b.')
    if self._use_bias:
      w, b = weights
      return jnp.matmul(x, w) + b  # Affine map.
    else:
      w = weights
      return jnp.matmul(x, w)  # Linear map.

  def new_weights(self, input_signature):
    """Returns new (randomly initialized) weights for this layer.

    In the default case (`use_bias=True`), weights are a `(w, b)` tuple.
    If `use_bias` is false, weights are a `w` tensor.

    Args:
      input_signature: ShapeDtype instance characterizing the input this layer
          should compute on.
    """
    shape_w = (input_signature.shape[-1], self._n_units)
    shape_b = (self._n_units,)
    rng_w, rng_b = math.random.split(self.rng, 2)
    w = self._kernel_initializer(shape_w, rng_w)

    if self._use_bias:
      b = self._bias_initializer(shape_b, rng_b)
      return (w, b)
    else:
      return w


class Embedding(base.Layer):
  """Trainable layer that maps discrete tokens/ids to vectors."""

  # TODO(jonni): Consider reversing param order to: vocab_size, d_feature
  def __init__(self,
               d_feature,
               vocab_size,
               kernel_initializer=init.RandomNormalInitializer(1.0)):
    """Returns an embedding layer with given vocabulary size and vector size.

    The layer clips input values (token ids) to the legal range:
    `[0, vocab_size)`. That is, negative token ids all clip to 0 before being
    mapped to a vector, and token ids with value vocab_size or greater all
    clip to vocab_size - 1 before being mapped to a vector. In effect, both
    id `0  and id `vocab_size - 1` are potentially overloaded as
    out-of-vocabulary token ids.

    TODO(jonni): Is this the behavior we want going forward?

    Args:
      d_feature: Dimensionality/depth of the output vectors.
      vocab_size: Size (number of different token ids) of the input vocabulary.
          The layer will handle tokens whose id's are in range(vocab_size).
      kernel_initializer: Function that creates (random) initial vectors for
          the embedding.
    """
    super().__init__()
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def forward(self, x, weights):
    return jnp.take(weights, x, axis=0)

  def new_weights(self, input_signature):
    del input_signature
    shape_w = (self._vocab_size, self._d_feature)
    # TODO(lukaszkaiser): do we split self.rng for consistency? Add a method?
    w = self._kernel_initializer(shape_w, self.rng)
    return w


class Dropout(base.Layer):
  """Dropout."""

  def __init__(self, rate=0.0, name='dropout', mode='train'):
    super(Dropout, self).__init__()
    self._initial_rate = rate
    # TODO(lukaszkaiser): remove the name property by the end of September'19.
    # It's only needed for a specific purpose in the short term, will go.
    self._name = 'dropout_' + name
    self._mode = mode

  def new_weights_and_state(self, input_signature):
    del input_signature
    state = {self._name: jnp.array(self._initial_rate)}
    return base.EMPTY_WEIGHTS, state

  def forward_with_state(self, x, weights, state, rng):
    """Execute dropout."""
    if self._mode != 'train':
      return x, state
    rate = self._initial_rate
    if isinstance(state, dict) and self._name in state:
      rate = state[self._name]
    if rng is None:
      msg = ('Dropout layer requires apply_fn to be called with a rng keyword '
             'argument. That is, instead of `Dropout(weights, inputs)`, call '
             'it like `Dropout(weights, inputs, rng=key)`.')
      raise ValueError(msg)
    keep = math.random.bernoulli(rng, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), jnp.zeros_like(x)), state


def Flatten(n_axes_to_keep=1):
  layer_name = f'Flatten_keep{n_axes_to_keep}'
  def f(x):  # pylint: disable=invalid-name
    in_rank = len(x.shape)
    if in_rank <= n_axes_to_keep:
      raise ValueError(f'Input rank ({in_rank}) must exceed the number of '
                       f'axes to keep ({n_axes_to_keep}) after flattening.')
    return jnp.reshape(x, (x.shape[:n_axes_to_keep] + (-1,)))
  return Fn(layer_name, f)


def Exp():
  return Fn('Exp', lambda x: jnp.exp(x))  # pylint: disable=unnecessary-lambda


def LogSoftmax(axis=-1):
  """Layer that applies log softmax: log-normalize along the given axis."""
  return Fn('LogSoftmax',
            lambda x: x - math.logsumexp(x, axis, keepdims=True))


def Softmax(axis=-1):
  """Layer that applies softmax: exponentiate and normalize along given axis."""
  return Fn('Softmax',
            lambda x: jnp.exp(x - math.logsumexp(x, axis, keepdims=True)))


def ToFloat():
  return Fn('ToFloat', lambda x: x.astype(np.float32))


def Mean(axis=-1, keepdims=False):
  return Fn('Mean', lambda x: jnp.mean(x, axis=axis, keepdims=keepdims))


def Sum(axis=-1, keepdims=False):
  return Fn('Sum', lambda x: jnp.sum(x, axis=axis, keepdims=keepdims))


def Negate():
  return Fn('Negate', lambda x: -x)


def log_gaussian_pdf(x, mu, sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, sigma)."""
  a = mu.shape[-1] * jnp.log(2 * jnp.pi)
  _, b = jnp.linalg.slogdet(sigma)
  y = jnp.linalg.solve(sigma, x - mu)
  y = jnp.expand_dims(y, axis=-1)
  xm = jnp.expand_dims(x - mu, axis=-2)
  c = jnp.matmul(xm, y)
  c = jnp.squeeze(jnp.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def log_gaussian_diag_pdf(x, mu, diag_sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, eye(diag_sigma))."""
  a = mu.shape[-1] * jnp.log(2 * jnp.pi)
  b = jnp.sum(jnp.log(diag_sigma), axis=-1)
  y = x - mu / diag_sigma
  y = jnp.expand_dims(y, axis=-1)
  xm = jnp.expand_dims(x - mu, axis=-2)
  c = jnp.matmul(xm, y)
  c = jnp.squeeze(jnp.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def multigaussian_loss(preds, targets, ngauss=1):  # pylint: disable=invalid-name
  """Compute mixture of gaussians loss."""
  ndims = targets.shape[-1]
  logits = preds[:, :ngauss]
  mus = preds[:, ngauss:ngauss*(ndims + 1)]
  sigmas = preds[:, ngauss(ndims + 1):]
  sigmas = sigmas * sigmas + 1e-6  # Make positive.
  loglogits = logits - math.logsumexp(logits, axis=-1, keepdims=True)
  mus = jnp.reshape(mus, [-1, ngauss, ndims])
  sigmas = jnp.reshape(sigmas, [-1, ngauss, ndims])
  targets = jnp.reshape(targets, [-1, 1, ndims])
  glogprobs = log_gaussian_diag_pdf(targets, mus, sigmas)
  return math.logsumexp(loglogits + glogprobs, axis=-1)


def gumbel_sample(log_probs, temperature=1.0):  # pylint: disable=invalid-name
  """Gumbel sampling from a categorical distribution, with temperature."""
  # This is equivalent to sampling from a softmax with temperature.
  u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
  g = -np.log(-np.log(u))
  return np.argmax(log_probs + g * temperature, axis=-1)
