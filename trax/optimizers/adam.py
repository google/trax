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
"""Adam optimizer class."""

from trax.fastmath import numpy as jnp
from trax.optimizers import base as opt_base


# pylint: disable=line-too-long
class Adam(opt_base.Optimizer):
  r"""Adam optimizer; described in https://arxiv.org/abs/1412.6980.

  The update rule for time step :math:`t`, given gradients :math:`g_t` and
  "Stepsize" :math:`\alpha`, is:

  .. math::
      \hat{m}_t &\leftarrow \big(\beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t\big)\ /\ (1 - \beta_1^t) \\
      \hat{v}_t &\leftarrow \big(\beta_2 \cdot m_{t-1} + (1 - \beta_2) \cdot g_t^2\big)\ /\ (1 - \beta_2^t) \\
      \theta_t  &\leftarrow \theta_{t-1} -\ \alpha \cdot \hat{m}_t / \big(\sqrt{\hat{v}_t} + \epsilon\big)

  """
  # pylint: enable=line-too-long

  def __init__(self, learning_rate=0.0001, weight_decay_rate=1e-5,  # pylint: disable=useless-super-delegation
               b1=0.9, b2=0.999, eps=1e-5, clip_grad_norm=None):
    r"""Creates an Adam optimizer.

    Args:
      learning_rate: Initial (unadapted) learning rate :math:`\alpha`; original
          paper calls this `Stepsize` and suggests .001 as a generally good
          value.
      weight_decay_rate: Fraction of prior weight values to subtract on each
          step; equivalent to multiplying each weight element by
          `1 - weight_decay_rate`. (This is not part of the core Adam
          algorithm.)
      b1: Exponential decay rate :math:`\beta_1` for first moment estimates.
      b2: Exponential decay rate :math:`\beta_2` for second moment estimates.
      eps: Small positive constant :math:`\epsilon` for numerical stability.
      clip_grad_norm: Threshold value above which gradient clipping occurs.
          (This is not part of the core Adam algorithm.)
    """
    super().__init__(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        clip_grad_norm=clip_grad_norm
    )

  def init(self, weights):
    m = jnp.zeros_like(weights)
    v = jnp.zeros_like(weights)
    return m, v

  def update(self, step, grads, weights, slots, opt_params):
    m, v = slots
    learning_rate = opt_params['learning_rate']
    weight_decay_rate = opt_params['weight_decay_rate']
    b1 = opt_params['b1']
    b2 = opt_params['b2']
    eps = opt_params['eps']
    m = (1 - b1) * grads + b1 * m  # First  moment estimate.
    v = (1 - b2) * (grads ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (step + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (step + 1))
    new_weights = ((1 - weight_decay_rate) * weights - (
        learning_rate * mhat / (jnp.sqrt(vhat) + eps))).astype(weights.dtype)
    return new_weights, (m, v)
