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
"""SM3 optimizer class."""

import enum

from trax.fastmath import numpy as jnp
from trax.optimizers import base as opt_base


class MomentumType(enum.IntEnum):
  EMA = 1
  HEAVY_BALL = 2
  NESTEROV = 3


class SM3(opt_base.Optimizer):
  """SM3 optimizer, as described in https://arxiv.org/abs/1901.11150."""

  def __init__(self,
               learning_rate=0.01,
               momentum=0.9,
               second_moment_averaging=1.0,
               weight_decay=0.0,
               momentum_type=MomentumType.EMA):  # pylint: disable=useless-super-delegation
    """Create the SM3 optimizer.

    Memory-Efficient Adaptive Optimization.
    https://arxiv.org/abs/1901.11150

    Args:
      learning_rate: a postitive scalar value for the initial learning rate.
      momentum: optional, a positive scalar value for momentum
      second_moment_averaging: averaging of second moments (if 1.0, adds from
        begining of time like AdaGrad).
      weight_decay: Weight decay for regularizing the model.
      momentum_type: Nestrov, Heavy-Ball or EMA (Default).

    """
    self._has_momentum = momentum > 0.0
    self._momentum_type = momentum_type
    self._graft = second_moment_averaging != 1.0
    super().__init__(
        learning_rate=learning_rate,
        momentum=momentum,
        second_moment_averaging=second_moment_averaging,
        weight_decay=weight_decay,
    )

  def init(self, w):
    momentum = []
    if self._has_momentum:
      momentum = jnp.zeros_like(w)
    v1s = [jnp.zeros(sz, dtype=w.dtype) for sz in w.shape]
    v2s = []
    if self._graft:
      v2s = [jnp.zeros(sz, dtype=w.dtype) for sz in w.shape]
    return (momentum, v1s, v2s)

  def _momentum_update(self, g, m, beta1):
    """Handle various types of momentum."""
    if self._momentum_type == MomentumType.EMA:
      m = (1 - beta1) * g + beta1 * m
      update = m
    elif self._momentum_type == MomentumType.HEAVY_BALL:
      m = g + beta1 * m
      update = m
    elif self._momentum_type == MomentumType.NESTEROV:
      m = g + beta1 * m
      nesterov_m = g + beta1 * m
      update = nesterov_m
    else:
      assert False, 'Unknown momentum_type.'
    return m, update

  def _update_diagonal(self, g, w, m, v1, v2, opt_params):
    learning_rate = opt_params['learning_rate']
    beta2 = opt_params['second_moment_averaging']
    weight_decay = opt_params['weight_decay']

    is_beta2_1 = (beta2 == 1).astype(g.dtype)
    one_minus_beta2_except1 = is_beta2_1  + (1.0 - beta2) * (1.0 - is_beta2_1)
    v1[0] = beta2 * v1[0] + one_minus_beta2_except1 * g * g

    preconditioner = jnp.where(v1[0] > 0, 1.0 / (jnp.sqrt(v1[0]) + 1e-16),
                               jnp.zeros_like(v1[0]))

    pg = preconditioner * g
    if self._graft:
      v2[0] += g * g
      preconditioner_graft = jnp.where(
          v2[0] > 0, 1.0 / (jnp.sqrt(v2[0]) + 1e-16), jnp.zeros_like(v2[0]))
      pg_graft = preconditioner_graft * g
      pg_norm = jnp.linalg.norm(pg)
      pg_graft_norm = jnp.linalg.norm(pg_graft)
      pg = pg * (pg_graft_norm/(pg_norm + 1e-16))

    pg = pg + w * weight_decay

    if self._has_momentum:
      m, update = self._momentum_update(pg, m, opt_params['momentum'])
    else:
      update = pg

    w = w - (update * learning_rate).astype(w.dtype)
    return w, (m, v1, v2)

  def _expanded_shape(self, shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _minimum(self, tensor_list):
    minimum = tensor_list[0]
    for i in range(1, len(tensor_list)):
      minimum = jnp.minimum(minimum, tensor_list[i])
    return minimum

  def _update_sketched(self, g, w, m, v1, v2, opt_params):
    """Update for higher-rank parameters."""
    learning_rate = opt_params['learning_rate']
    momentum = opt_params['momentum']
    beta2 = opt_params['second_moment_averaging']
    weight_decay = opt_params['weight_decay']

    shape = w.shape
    rank = len(shape)
    reshaped_accumulators = [jnp.reshape(v1[i], self._expanded_shape(shape, i))
                             for i in range(rank)]
    acc = self._minimum(reshaped_accumulators)

    is_beta2_1 = (beta2 == 1).astype(g.dtype)
    one_minus_beta2_except1 = is_beta2_1  + (1.0 - beta2) * (1.0 - is_beta2_1)
    acc = beta2 * acc + one_minus_beta2_except1 * g * g

    preconditioner = jnp.where(acc > 0.0, 1.0 / (jnp.sqrt(acc) + 1e-16),
                               jnp.zeros_like(acc))
    pg = g * preconditioner
    if self._graft:
      v2_acc = self._minimum([
          jnp.reshape(v2[i], self._expanded_shape(shape, i))
          for i in range(rank)
      ])
      v2_acc = v2_acc + g * g
      preconditioner_graft = jnp.where(v2_acc > 0.0,
                                       1.0 / (jnp.sqrt(v2_acc) + 1e-16),
                                       jnp.zeros_like(v2_acc))
      pg_graft = preconditioner_graft * g
      pg_norm = jnp.linalg.norm(pg)
      pg_graft_norm = jnp.linalg.norm(pg_graft)
      pg = pg * (pg_graft_norm/(pg_norm + 1e-16))

    pg = pg + w * weight_decay

    if self._has_momentum:
      m, update = self._momentum_update(pg, m, momentum)
    else:
      update = pg

    w = w - (learning_rate * update).astype(w.dtype)
    for i in range(len(v1)):
      axes = list(range(int(i))) + list(range(int(i) + 1, rank))
      dim_accumulator = jnp.amax(acc, axis=axes)
      v1[i] = dim_accumulator

    if self._graft:
      for i in range(len(v2)):
        axes = list(range(int(i))) + list(range(int(i) + 1, rank))
        dim_accumulator = jnp.amax(v2_acc, axis=axes)
        v2[i] = dim_accumulator
    return w, (m, v1, v2)

  def update(self, step, g, w, slots, opt_params):
    del step
    m, v1, v2 = slots
    rank = len(w.shape)
    if rank > 1:
      return self._update_sketched(g, w, m, v1, v2, opt_params)
    else:
      return self._update_diagonal(g, w, m, v1, v2, opt_params)
