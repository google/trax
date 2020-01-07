# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Trax normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax.layers import base
from trax.math import numpy as np


class BatchNorm(base.Layer):
  """Batch normalization."""

  def __init__(self, axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
               momentum=0.999, mode='train'):
    super(BatchNorm, self).__init__()
    self._axis = axis
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._momentum = momentum
    self._mode = mode

  def new_weights_and_state(self, input_signature):
    """Helper to initialize batch norm weights."""
    axis = self._axis
    axis = (axis,) if np.isscalar(axis) else axis
    input_shape = input_signature.shape
    shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
    beta = np.zeros(shape, dtype='float32') if self._center else ()
    gamma = np.ones(shape, dtype='float32') if self._scale else ()
    def get_stats_axis(i, d):
      if i in axis:
        return 1
      else:
        return d
    stats_shape = tuple(get_stats_axis(i, d) for i, d in enumerate(input_shape))
    running_mean = np.zeros(stats_shape, dtype=np.float32)
    running_var = np.ones(stats_shape, dtype=np.float32)
    n_batches = np.zeros((), dtype=np.int64)
    weights = (beta, gamma)
    state = (running_mean, running_var, n_batches)
    return weights, state

  def _fast_mean_and_variance(self, x):
    mean = np.mean(x, self._axis, keepdims=True)
    # Fast but less numerically-stable variance calculation than np.var.
    m1 = np.mean(x**2, self._axis, keepdims=True)
    variance = m1 - mean**2
    return mean, variance

  def _exponential_smoothing(self, new, old):
    smoothed_value = self._momentum * old + (1 - self._momentum) * new
    return smoothed_value.astype(old.dtype)

  def _z_score(self, x, mean, variance):
    mu = mean.astype(x.dtype)
    sigma = np.sqrt(variance + self._epsilon).astype(x.dtype)
    return (x - mu) / sigma

  def _beta_gamma_with_correct_axes(self, x, weights):
    # Expand the parameters to have the right axes.
    beta, gamma = weights
    # TODO(phawkins): np.expand_dims should accept an axis tuple.
    # (https://github.com/numpy/numpy/issues/12290)
    ed = tuple(None if i in self._axis else slice(None)
               for i in range(np.ndim(x)))
    beta = beta[ed]
    gamma = gamma[ed]
    return beta, gamma

  def forward_with_state(self, x, weights, state, **unused_kwargs):
    """Computes batch normalization as part of a forward pass in the model."""

    running_mean, running_var, n_batches = state
    if self._mode == 'train':
      n_batches += 1
      mean, var = self._fast_mean_and_variance(x)
      running_mean = self._exponential_smoothing(mean, running_mean)
      running_var = self._exponential_smoothing(var, running_var)
      state = (running_mean, running_var, n_batches)
    else:
      mean = running_mean
      var = running_var

    z = self._z_score(x, mean, var)
    beta, gamma = self._beta_gamma_with_correct_axes(x, weights)

    # Return the z rescaled by the parameters if requested.
    if self._center and self._scale:
      output = gamma * z + beta
    elif self._center:
      output = z + beta
    elif self._scale:
      output = gamma * z
    else:
      output = z
    assert output.dtype == x.dtype, ('The dtype of the output (%s) of batch '
                                     'norm is not the same as the input (%s). '
                                     'Batch norm should not change the dtype' %
                                     (output.dtype, x.dtype))
    return output, state


# Layer normalization.
def _layer_norm_weights(input_signature):
  """Helper: create layer norm parameters."""
  features = input_signature.shape[-1]
  scale = np.ones(features)
  bias = np.zeros(features)
  weights = (scale, bias)
  return weights


@base.layer(new_weights_fn=_layer_norm_weights)
def LayerNorm(x, weights, epsilon=1e-6, **unused_kwargs):  # pylint: disable=invalid-name
  (scale, bias) = weights
  mean = np.mean(x, axis=-1, keepdims=True)
  variance = np.mean((x - mean)**2, axis=-1, keepdims=True)
  norm_inputs = (x - mean) / np.sqrt(variance + epsilon)
  return norm_inputs * scale + bias


class FilterResponseNorm(base.Layer):
  """Filter Response Normalization layer without Threshold Linear Unit.

  c.f. https://arxiv.org/pdf/1911.09737.pdf
  """

  def __init__(self,
               mode=None,
               learn_epsilon=False,
               init_epsilon=1e-6,
               init_learnt_epsilon=1e-4):
    super(FilterResponseNorm, self).__init__()

    del mode

    # If we learn epsilon then epsilon = init_epsilon + |learnt_value|
    # where learnt_value is initialized to init_learnt_epsilon.
    # If learn_epsilon is false then epsilon is just init_epsilon.
    #
    # NOTE: I (afrozm) haven't been able to train with `learn_epsilon = True`.
    self._learn_epsilon = learn_epsilon

    assert init_epsilon > 0
    assert init_learnt_epsilon > 0

    self._init_epsilon = np.array(init_epsilon, dtype=np.float32)
    self._init_learnt_epsilon = np.array(init_learnt_epsilon, dtype=np.float32)

  def new_weights(self, input_signature):
    # Usually (B, W, H, C)
    shape = input_signature.shape
    num_channels = shape[-1]

    gamma = np.ones((num_channels,), dtype=np.float32)
    beta = np.zeros((num_channels,), dtype=np.float32)

    epsilon_l = base.EMPTY_WEIGHTS
    if self._learn_epsilon:
      epsilon_l = (self._init_learnt_epsilon,)

    return gamma, beta, epsilon_l

  def forward(self, inputs, weights):
    gamma, beta, epsilon_l = weights

    epsilon = self._init_epsilon
    if epsilon_l is not base.EMPTY_WEIGHTS:
      epsilon += np.abs(epsilon_l[0])

    # Omit B and C
    axis = tuple(range(1, len(np.shape(inputs)) - 1))
    # (B, 1, 1, C)
    nu2 = np.mean(inputs**2, axis=axis, keepdims=True)
    # (B, W, H, C)
    xhat = inputs / np.sqrt(nu2 + epsilon)

    return gamma * xhat + beta
