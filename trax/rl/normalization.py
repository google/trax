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
"""Normalization helpers."""

import gin
import numpy as np

from trax import fastmath
from trax import layers as tl


def running_mean_init(shape, fill_value=0):
  return (np.full(shape, fill_value), np.array(0))


def running_mean_update(x, state):
  (mean, n) = state
  mean = n.astype(np.float32) / (n + 1) * mean + x / (n + 1)
  return (mean, n + 1)


def running_mean_get_mean(state):
  (mean, _) = state
  return mean


def running_mean_get_count(state):
  (_, count) = state
  return count


def running_mean_and_variance_init(shape):
  mean_state = running_mean_init(shape, fill_value=0.0)
  var_state = running_mean_init(shape, fill_value=1.0)
  return (mean_state, var_state)


def running_mean_and_variance_update(x, state):
  (mean_state, var_state) = state
  old_mean = running_mean_get_mean(mean_state)
  mean_state = running_mean_update(x, mean_state)
  new_mean = running_mean_get_mean(mean_state)

  var_state = running_mean_update((x - new_mean) * (x - old_mean), var_state)

  return (mean_state, var_state)


def running_mean_and_variance_get_mean(state):
  (mean_state, _) = state
  return running_mean_get_mean(mean_state)


def running_mean_and_variance_get_count(state):
  (mean_state, _) = state
  return running_mean_get_count(mean_state)


def running_mean_and_variance_get_variance(state):
  (_, var_state) = state
  return running_mean_get_mean(var_state)


@gin.configurable(denylist=['mode'])
class Normalize(tl.Layer):
  """Numerically stable normalization layer."""

  def __init__(self, sample_limit=float('+inf'), epsilon=1e-5, mode='train'):
    super().__init__()
    self._sample_limit = sample_limit
    self._epsilon = epsilon
    self._mode = mode

  def init_weights_and_state(self, input_signature):
    self.state = running_mean_and_variance_init(input_signature.shape[2:])

  def forward(self, inputs):
    state = self.state
    observations = inputs
    if self._mode == 'collect':
      # Accumulate statistics only in the collect mode, i.e. when collecting
      # data using the agent.
      for observation in observations[:, -1]:  # (batch_size, time, ...)
        # Update statistics for each observation separately for simplicity.
        # Currently during data collection the batch size is 1 anyway.
        count = running_mean_and_variance_get_count(state)
        state = fastmath.cond(
            count < self._sample_limit,
            true_operand=(observation, state),
            true_fun=lambda args: running_mean_and_variance_update(*args),
            false_operand=None,
            false_fun=lambda _: state,
        )

    mean = running_mean_and_variance_get_mean(state)
    var = running_mean_and_variance_get_variance(state)
    norm_observations = (observations - mean) / (var ** 0.5 + self._epsilon)
    self.state = state
    return norm_observations


@gin.configurable(denylist=['mode'])
def LayerNormSquash(mode, width=128):  # pylint: disable=invalid-name
  """Dense-LayerNorm-Tanh normalizer inspired by ACME."""
  # https://github.com/deepmind/acme/blob/master/acme/jax/networks/continuous.py#L34
  del mode
  return tl.Serial([
      tl.Dense(width),
      tl.LayerNorm(),
      tl.Tanh(),
  ])
