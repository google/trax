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
"""Adam optimizer class."""

from trax.math import numpy as np
from trax.optimizers import base as opt_base


class Adam(opt_base.Optimizer):
  """Adam optimizer."""

  def __init__(self, learning_rate, weight_decay_rate=1e-5,  # pylint: disable=useless-super-delegation
               b1=0.9, b2=0.999, eps=1e-5):
    """Create the Adam optimizer.

    Args:
      learning_rate: a postitive scalar value for the initial learning rate.
      weight_decay_rate: rate at which to decay weights.
      b1: optional, a positive scalar value for beta_1, the exponential decay
        rate for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay
         rate for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-5).
    """
    super(Adam, self).__init__(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        b1=b1,
        b2=b2,
        eps=eps,
    )

  def init(self, weights):
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
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
    new_weights = (1 - weight_decay_rate) * weights - (
        learning_rate * mhat / (np.sqrt(vhat) + eps)).astype(weights.dtype)
    return new_weights, (m, v)
