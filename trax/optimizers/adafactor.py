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
"""Adafactor optimizer class."""

from trax.fastmath import numpy as jnp
from trax.optimizers import base as opt_base


class Adafactor(opt_base.Optimizer):
  """Adafactor optimizer, as described in https://arxiv.org/abs/1804.04235."""

  def __init__(self,
               learning_rate=0.05,
               factored=True,
               multiply_by_parameter_scale=True,
               do_clipping=True,
               do_momentum=False,
               momentum_in_bfloat16=False,
               beta1=0.0,
               decay_rate=0.8,
               clipping_threshold=1.0,
               weight_decay_rate=1e-5,
               weight_decay_n_steps=0,
               epsilon1=1e-16,
               epsilon2=1e-3):
    """Create the Adafactor optimizer.

    Adafactor is described in https://arxiv.org/abs/1804.04235.

    Args:
      learning_rate: float: trax-provided learning rate.
      factored: boolean: whether to use factored second-moment estimator for 2d
        variables.
      multiply_by_parameter_scale: boolean: if True, then scale provided
        learning_rate by parameter norm. if False, provided learning_rate is
        absolute step size.
      do_clipping: whether to clip gradients; if True, set clipping_theshold.
      do_momentum: whether to use momentum; if True, set beta1.
      momentum_in_bfloat16: if True, store momentum in bfloat16 to save memory.
      beta1: a float value between 0 and 1, enables momentum and uses extra
        memory if nonzero!  Off by default.
      decay_rate: float: controls second-moment exponential decay schedule.
      clipping_threshold: an optional float >= 1, if None no update clipping.
      weight_decay_rate: rate at which to decay weights.
      weight_decay_n_steps: for how many steps to decay weights (always if None)
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
    """
    # These 4 parameters are not configurable once the class is created.
    self._factored = factored
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    self._do_clipping = do_clipping
    self._do_momentum = do_momentum
    self._momentum_in_bfloat16 = momentum_in_bfloat16
    # Dynamically configurable parameters will be passed to the update function.
    super().__init__(
        learning_rate=learning_rate,
        beta1=beta1,
        decay_rate=decay_rate,
        clipping_threshold=clipping_threshold,
        weight_decay_rate=weight_decay_rate,
        weight_decay_n_steps=weight_decay_n_steps,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )

  @staticmethod
  def _decay_rate_pow(i, exponent=0.8):
    """Default Adafactor second-moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return 1.0 - t**(-exponent)

  def init(self, weights):
    shape = weights.shape
    slots = []
    if self._factored and len(shape) >= 2:
      v_row = jnp.zeros(shape[:-1], dtype=jnp.float32)
      v_col = jnp.zeros(shape[:-2] + shape[-1:], dtype=jnp.float32)
      slots.extend([v_row, v_col])
    else:
      v = jnp.zeros_like(weights)
      slots.append(v)
    if self._do_momentum:
      m = jnp.zeros_like(weights)
      if self._momentum_in_bfloat16:
        m = m.astype(jnp.bfloat16)
      slots.append(m)
    return slots

  def update(self, step, grads, weights, slots, opt_params):
    updates = []
    learning_rate = opt_params['learning_rate']
    beta1 = opt_params['beta1']
    decay_rate = opt_params['decay_rate']
    clipping_threshold = opt_params['clipping_threshold']
    weight_decay_rate = opt_params['weight_decay_rate']
    weight_decay_n_steps = opt_params['weight_decay_n_steps']
    weight_decay_rate = jnp.where(
        weight_decay_n_steps < 1,  # if weight_decay_n_steps == 0, ignore it
        weight_decay_rate,
        (weight_decay_rate * jnp.maximum(weight_decay_n_steps - step, 0.0) /
         jnp.maximum(weight_decay_n_steps, 0.0)))
    epsilon1 = opt_params['epsilon1']
    epsilon2 = opt_params['epsilon2']
    decay_rate = self._decay_rate_pow(step, exponent=decay_rate)
    update_scale = learning_rate
    if self._multiply_by_parameter_scale:
      update_scale *= jnp.maximum(
          jnp.sqrt(jnp.mean(weights * weights)), epsilon2)
    mixing_rate = 1.0 - decay_rate

    grads_sqr = grads * grads
    if self._factored and len(weights.shape) >= 2:
      v_row = slots[0]  # In this case, the slots are (v_row, v_col, ...).
      v_col = slots[1]
      new_v_row = (
          decay_rate * v_row + mixing_rate * jnp.mean(grads_sqr, axis=-1))
      new_v_col = (
          decay_rate * v_col + mixing_rate * jnp.mean(grads_sqr, axis=-2))
      updates.extend([new_v_row, new_v_col])
      row_mean = jnp.mean(new_v_row, axis=-1, keepdims=True)
      row_factor = (row_mean / (new_v_row + epsilon1))**0.5
      col_factor = (new_v_col + epsilon1)**-0.5
      y = (
          grads * jnp.expand_dims(row_factor, axis=-1) *
          jnp.expand_dims(col_factor, axis=-2))
    else:
      v = slots[0]  # In this case, the slots are (v, ...)
      new_v = decay_rate * v + mixing_rate * grads_sqr
      updates.append(new_v)
      y = grads * (new_v + epsilon1)**-0.5

    if self._do_clipping:
      clipping_denom = (
          jnp.maximum(1.0, jnp.sqrt(jnp.mean(y * y)) / clipping_threshold))
      y /= clipping_denom

    subtrahend = update_scale * y
    if self._do_momentum:
      m = slots[-1]  # Momentum is always the last slot (if used).
      m = m.astype(subtrahend.dtype)  # Accumulate in subtrahend dtype.
      new_m = beta1 * m + (1.0 - beta1) * subtrahend
      subtrahend = new_m
      updates.append(new_m.astype(slots[-1].dtype))  # Back to bfloat if needed.

    new_weights = (1 - weight_decay_rate) * weights - subtrahend
    # TODO(lukaszkaiser): why is the astype needed here? Check and correct.
    return new_weights.astype(weights.dtype), updates
