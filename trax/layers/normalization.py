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
"""Trax normalization layers."""

from trax.fastmath import numpy as jnp
from trax.layers import base


class BatchNorm(base.Layer):
  """Layer that performs batch normalization.

  In training, batch normalization keeps smoothed cumulative statistics across
  batches of input data and modifies each new batch so that its components are
  normally distributed. In eval or inference, a `BatchNorm` instance uses its
  stored mean and variance to approximately normalize each new batch of data.

  See https://arxiv.org/abs/1502.03167 for original presentation and motivation
  of batch normalization).
  """

  def __init__(self, axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
               momentum=0.999, mode='train'):
    super().__init__()
    self._axis = axis
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._momentum = momentum
    self._mode = mode

  def forward(self, x):
    """Computes batch normalization as part of a forward pass in the model."""
    running_mean, running_var, n_batches = self.state
    if self._mode == 'train':
      n_batches += 1
      mean, var = self._fast_mean_and_variance(x)
      # Gather smoothed input statistics for later use in evals or inference.
      running_mean = _exponentially_smoothed(self._momentum, running_mean, mean)
      running_var = _exponentially_smoothed(self._momentum, running_var, var)
      self.state = (running_mean, running_var, n_batches)
    else:
      mean = running_mean
      var = running_var

    z = self._z_score(x, mean, var)
    beta, gamma = self._beta_gamma_with_correct_axes(x, self.weights)

    # Return the z rescaled by the parameters if requested.
    if self._center and self._scale:
      output = gamma * z + beta
    elif self._center:
      output = z + beta
    elif self._scale:
      output = gamma * z
    else:
      output = z
    if output.dtype != x.dtype:
      raise TypeError(f'The dtype of the output ({output.dtype}) of batch '
                      f'norm is not the same as the input ({x.dtype}). '
                      f'Batch norm should not change the dtype.')
    return output

  def init_weights_and_state(self, input_signature):
    """Helper to initialize batch norm weights and state."""
    axis = self._axis
    axis = (axis,) if jnp.isscalar(axis) else axis
    input_shape = input_signature.shape
    shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
    # TODO(jonni): Should beta and gamma match the dtype in the input signature?
    beta = jnp.zeros(shape, dtype='float32') if self._center else ()
    gamma = jnp.ones(shape, dtype='float32') if self._scale else ()
    def get_stats_axis(i, d):
      if i in axis:
        return 1
      else:
        return d
    stats_shape = tuple(get_stats_axis(i, d) for i, d in enumerate(input_shape))
    running_mean = jnp.zeros(stats_shape, dtype=jnp.float32)
    running_var = jnp.ones(stats_shape, dtype=jnp.float32)
    n_batches = jnp.zeros((), dtype=jnp.int64)
    self.weights = (beta, gamma)
    self.state = (running_mean, running_var, n_batches)

  def _fast_mean_and_variance(self, x):
    mean = jnp.mean(x, self._axis, keepdims=True)
    # Fast but less numerically-stable variance calculation than jnp.var.
    m1 = jnp.mean(x**2, self._axis, keepdims=True)
    variance = m1 - mean**2
    return mean, variance

  def _z_score(self, x, mean, variance):
    mu = mean.astype(x.dtype)
    sigma = jnp.sqrt(variance + self._epsilon).astype(x.dtype)
    return (x - mu) / sigma

  def _beta_gamma_with_correct_axes(self, x, weights):
    # Expand the parameters to have the right axes.
    beta, gamma = weights
    # TODO(phawkins): jnp.expand_dims should accept an axis tuple.
    # (https://github.com/numpy/numpy/issues/12290)
    ed = tuple(None if i in self._axis else slice(None)
               for i in range(jnp.ndim(x)))
    beta = beta[ed]
    gamma = gamma[ed]
    return beta, gamma


class LayerNorm(base.Layer):
  """Layer normalization."""

  def __init__(self, center=True, epsilon=1e-6):
    super().__init__()
    self._epsilon = epsilon
    self._center = center

  def forward(self, x):
    scale, bias = self.weights
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean if self._center else x
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    norm_inputs = centered / jnp.sqrt(variance + self._epsilon)
    scaled = norm_inputs * scale
    return scaled + bias if self._center else scaled

  def init_weights_and_state(self, input_signature):
    features = input_signature.shape[-1]
    scale = jnp.ones(features, dtype=input_signature.dtype)
    bias = jnp.zeros(features, dtype=input_signature.dtype)
    self.weights = scale, bias


class FilterResponseNorm(base.Layer):
  """Filter Response Normalization layer without Threshold Linear Unit.

  c.f. https://arxiv.org/pdf/1911.09737.pdf
  """

  def __init__(self,
               mode=None,
               learn_epsilon=False,
               init_epsilon=1e-6,
               init_learnt_epsilon=1e-4):
    super().__init__()

    del mode

    # If we learn epsilon then epsilon = init_epsilon + |learnt_value|
    # where learnt_value is initialized to init_learnt_epsilon.
    # If learn_epsilon is false then epsilon is just init_epsilon.
    #
    # NOTE: I (afrozm) haven't been able to train with `learn_epsilon = True`.
    self._learn_epsilon = learn_epsilon

    # TODO(jonni): Replace asserts with ValueError.
    assert init_epsilon > 0
    assert init_learnt_epsilon > 0

    self._init_epsilon = jnp.array(init_epsilon, dtype=jnp.float32)
    self._init_learnt_epsilon = jnp.array(init_learnt_epsilon,
                                          dtype=jnp.float32)

  def forward(self, inputs):
    gamma, beta, epsilon_l = self.weights

    epsilon = self._init_epsilon
    if epsilon_l is not base.EMPTY_WEIGHTS:
      epsilon += jnp.abs(epsilon_l[0])

    # Omit B and C
    axis = tuple(range(1, len(jnp.shape(inputs)) - 1))
    # (B, 1, 1, C)
    nu2 = jnp.mean(inputs**2, axis=axis, keepdims=True)
    # (B, W, H, C)
    xhat = inputs / jnp.sqrt(nu2 + epsilon)

    return gamma * xhat + beta

  def init_weights_and_state(self, input_signature):
    # Usually (B, W, H, C)
    shape = input_signature.shape
    num_channels = shape[-1]

    gamma = jnp.ones((num_channels,), dtype=jnp.float32)
    beta = jnp.zeros((num_channels,), dtype=jnp.float32)

    epsilon_l = base.EMPTY_WEIGHTS
    if self._learn_epsilon:
      epsilon_l = (self._init_learnt_epsilon,)

    self.weights = gamma, beta, epsilon_l


def _exponentially_smoothed(momentum, old, new):
  smoothed_value = momentum * old + (1 - momentum) * new
  return smoothed_value.astype(old.dtype)
