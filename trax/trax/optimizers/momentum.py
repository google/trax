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
"""Nesterov momentum optimizer (also known as Nesterov Accelerated Gradient)."""

from trax.fastmath import numpy as jnp
from trax.optimizers import base


# TODO(jonni): Consider renaming this class to NesterovMomentum.
class Momentum(base.Optimizer):
  r"""A momentum optimizer.

  This class implements two variants of momentum stochastic gradient descent
  (SGD): with and without the Nesterov correction. The implementation of the
  Nesterov update is based on the concepts in Sutskever et al. (2013)
  [http://jmlr.org/proceedings/papers/v28/sutskever13.pdf], reformulated in
  Bengio et al. (2012) [https://arxiv.org/abs/1212.0901], to work well with
  backpropagation (equations 6 and 7):

  .. math::
      v_t      &= \mu_{t-1}v_{t-1} - \epsilon_{t-1}\nabla f(\Theta_{t-1}) \\
      \Theta_t &= \Theta_{t-1} - \mu_{t-1} v_{t-1} + \mu_t v_t + v_t

  where :math:`\mu_{t-1}` is the momentum (decay) coefficient at time step
  :math:`t-1` and :math:`\epsilon_{t-1}` is the learning rate at :math:`t-1`.

  Note that the implementation below also includes a weight decay rate
  (:math:`\alpha`) on the parameters, independent of the Nesterov momentum.
  """

  def __init__(
      self, learning_rate, mass=0.9, weight_decay_rate=1e-5, nesterov=True
  ):  # pylint: disable=useless-super-delegation
    super(Momentum, self).__init__(
        learning_rate=learning_rate,
        mass=mass,
        weight_decay_rate=weight_decay_rate,
    )
    self._nesterov = nesterov

  def init(self, weights):
    return jnp.zeros_like(weights)

  def update(self, step, grads, weights, velocity, opt_params):
    del step
    v = velocity
    mu = opt_params['mass']
    alpha = opt_params['weight_decay_rate']
    epsilon = opt_params['learning_rate']

    new_v = mu * v + grads
    if self._nesterov:
      weight_update = mu * new_v + grads
    else:
      weight_update = new_v
    new_weights = (1 - alpha) * weights - epsilon * weight_update

    new_weights = new_weights.astype(weights.dtype)
    return (new_weights, new_v)
