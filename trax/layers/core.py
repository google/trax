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

import numpy as onp

from trax import math
from trax.layers import base
from trax.layers import initializers as init
from trax.math import numpy as np


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
    return np.dot(x, w) + b

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
    return np.take(weights, x, axis=0)

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
    state = {self._name: np.array(self._initial_rate)}
    return base.EMPTY_WEIGHTS, state

  def forward_with_state(self, x, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    """Execute dropout."""
    del kwargs
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
    return np.where(keep, x / (1.0 - rate), np.zeros_like(x)), state


@base.layer()
def Flatten(x, n_axes_to_keep=1, **unused_kwargs):
  if n_axes_to_keep >= len(x.shape):
    raise ValueError("n_axes_to_keep[%d] should be less than input's rank[%d]" %
                     (n_axes_to_keep, len(x.shape)))
  return np.reshape(x, (x.shape[:n_axes_to_keep] + (-1,)))


@base.layer()
def Exp(x, **unused_kwargs):
  return np.exp(x)


@base.layer()
def LogSoftmax(x, axis=-1, **unused_kwargs):
  """Apply log softmax to x: log-normalize along the given axis."""
  return x - math.logsumexp(x, axis, keepdims=True)


@base.layer()
def Softmax(x, axis=-1, **unused_kwargs):
  """Apply softmax to x: exponentiate and normalize along the given axis."""
  return np.exp(x - math.logsumexp(x, axis, keepdims=True))


@base.layer()
def ToFloat(x, **unused_kwargs):
  return x.astype(onp.float32)


@base.layer()
def Mean(x, axis=-1, keepdims=False, **unused_kwargs):
  return np.mean(x, axis=axis, keepdims=keepdims)


@base.layer()
def Sum(x, axis=-1, keepdims=False, **unused_kwargs):
  return np.sum(x, axis=axis, keepdims=keepdims)


@base.layer()
def Negate(x, **unused_kwargs):
  return -x


def log_gaussian_pdf(x, mu, sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, sigma)."""
  a = mu.shape[-1] * np.log(2 * np.pi)
  _, b = np.linalg.slogdet(sigma)
  y = np.linalg.solve(sigma, x - mu)
  y = np.expand_dims(y, axis=-1)
  xm = np.expand_dims(x - mu, axis=-2)
  c = np.matmul(xm, y)
  c = np.squeeze(np.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def log_gaussian_diag_pdf(x, mu, diag_sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, eye(diag_sigma))."""
  a = mu.shape[-1] * np.log(2 * np.pi)
  b = np.sum(np.log(diag_sigma), axis=-1)
  y = x - mu / diag_sigma
  y = np.expand_dims(y, axis=-1)
  xm = np.expand_dims(x - mu, axis=-2)
  c = np.matmul(xm, y)
  c = np.squeeze(np.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def multigaussian_loss(preds, targets, ngauss=1):  # pylint: disable=invalid-name
  """Compute mixture of gaussians loss."""
  ndims = targets.shape[-1]
  logits = preds[:, :ngauss]
  mus = preds[:, ngauss:ngauss*(ndims + 1)]
  sigmas = preds[:, ngauss(ndims + 1):]
  sigmas = sigmas * sigmas + 1e-6  # Make positive.
  loglogits = logits - math.logsumexp(logits, axis=-1, keepdims=True)
  mus = np.reshape(mus, [-1, ngauss, ndims])
  sigmas = np.reshape(sigmas, [-1, ngauss, ndims])
  targets = np.reshape(targets, [-1, 1, ndims])
  glogprobs = log_gaussian_diag_pdf(targets, mus, sigmas)
  return math.logsumexp(loglogits + glogprobs, axis=-1)


def gumbel_sample(log_probs, temperature=1.0):  # pylint: disable=invalid-name
  """Gumbel sampling from a categorical distribution, with temperature."""
  # This is equivalent to sampling from a softmax with temperature.
  u = onp.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
  g = -onp.log(-onp.log(u))
  return onp.argmax(log_probs + g * temperature, axis=-1)
