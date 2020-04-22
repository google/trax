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
  """A dense (a.k.a. fully-connected, affine) layer."""

  def __init__(self,
               n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(Dense, self).__init__()
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, x, weights):
    w, b = weights
    return jnp.dot(x, w) + b

  def new_weights(self, input_signature):
    input_shape = input_signature.shape
    rng1, rng2 = self.new_rngs(2)
    w = self._kernel_initializer((input_shape[-1], self._n_units), rng1)
    b = self._bias_initializer((self._n_units,), rng2)
    return (w, b)


class Embedding(base.Layer):
  """Layer constructor function for an embedding layer."""

  def __init__(self,
               d_feature,
               vocab_size,
               kernel_initializer=init.RandomNormalInitializer(1.0)):
    super(Embedding, self).__init__()
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def forward(self, x, weights):
    return jnp.take(weights, x, axis=0)

  def new_weights(self, input_signature):
    del input_signature
    out_dim = (self._vocab_size, self._d_feature)
    weights = self._kernel_initializer(out_dim, self.new_rng())
    return weights


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

  def forward_with_state(self, x, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None):
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
