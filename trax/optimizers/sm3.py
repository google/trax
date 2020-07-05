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
"""SM3 optimizer class."""

from trax.fastmath import numpy as jnp
from trax.optimizers import base as opt_base


class SM3(opt_base.Optimizer):
  """SM3 optimizer."""

  def __init__(self, learning_rate, momentum=0.9):  # pylint: disable=useless-super-delegation
    """Create the SM3 optimizer.

    Memory-Efficient Adaptive Optimization for Large-Scale Learning.
    https://arxiv.org/abs/1901.11150

    Args:
      learning_rate: a postitive scalar value for the initial learning rate.
      momentum: optional, a positive scalar value for momentum
    """
    super(SM3, self).__init__(
        learning_rate=learning_rate,
        momentum=momentum,
    )

  def init(self, weights):
    vs = [jnp.zeros(sz, dtype=weights.dtype) for sz in weights.shape]
    return (jnp.zeros_like(weights), vs)

  def _update_diagonal(self, grads, weights, m, v, opt_params):
    learning_rate = opt_params['learning_rate']
    momentum = opt_params['momentum']
    v[0] += grads * grads
    preconditioner = jnp.where(v[0] > 0, 1.0 / jnp.sqrt(v[0]),
                               jnp.zeros_like(v[0]))
    preconditioned_grads = preconditioner * grads
    m = (1 - momentum) * preconditioned_grads + momentum * m
    weights = weights - (learning_rate * m).astype(weights.dtype)
    return weights, (m, v)

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

  def _update_sketched(self, grads, weights, m, v, opt_params):
    """Update for higher-rank parameters."""
    learning_rate = opt_params['learning_rate']
    momentum = opt_params['momentum']
    shape = weights.shape
    rank = len(shape)
    reshaped_accumulators = [jnp.reshape(v[i], self._expanded_shape(shape, i))
                             for i in range(rank)]
    current_accumulator = self._minimum(reshaped_accumulators)
    current_accumulator += grads * grads
    accumulator_inv_sqrt = jnp.where(current_accumulator > 0.0,
                                     1.0 / jnp.sqrt(current_accumulator),
                                     jnp.zeros_like(current_accumulator))
    preconditioned_gradient = grads * accumulator_inv_sqrt
    m = (1.0 - momentum) * preconditioned_gradient + momentum * m
    weights = weights - (learning_rate * m).astype(weights.dtype)
    for i in range(len(v)):
      axes = list(range(int(i))) + list(range(int(i) + 1, rank))
      dim_accumulator = jnp.amax(current_accumulator, axis=axes)
      v[i] = dim_accumulator
    return weights, (m, v)

  def update(self, step, grads, weights, slots, opt_params):
    del step
    m, v = slots
    shape = weights.shape
    rank = len(shape)
    if rank > 1:
      return self._update_sketched(grads, weights, m, v, opt_params)
    else:
      return self._update_diagonal(grads, weights, m, v, opt_params)
