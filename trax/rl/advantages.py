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
"""RL advantage estimators."""

import gin
import numpy as np

from trax import fastmath

common_args = ['gamma', 'margin']


def mask_discount(discount, discount_mask):
  """Computes a discount to apply at a given timestep, based on the mask."""
  return fastmath.numpy.where(discount_mask, discount, 1.0)


def discounted_returns(rewards, gammas):
  """Computes discounted returns for a trajectory or a batch of them."""
  returns = np.zeros_like(rewards)
  ret = 0.0
  for i in reversed(range(rewards.shape[-1])):
    ret = rewards[..., i] + gammas[..., i] * ret
    returns[..., i] = ret
  return returns


@gin.configurable(denylist=common_args)
def monte_carlo(gamma, margin):
  """Calculate Monte Carlo advantage.

  We assume the values are a tensor of shape [batch_size, length] and this
  is the same shape as rewards and returns.

  Args:
    gamma: float, gamma parameter for TD from the underlying task
    margin: number of extra steps in the sequence

  Returns:
    Function (rewards, returns, values, dones) -> advantages, where advantages
    advantages is an array of shape [batch_size, length - margin].
  """
  del gamma
  def estimator(rewards, returns, values, dones, discount_mask):
    del discount_mask
    (_, length) = returns.shape
    # Make sure that the future returns and values at "done" states are zero.
    returns[dones] = rewards[dones]
    values[dones] = 0
    return (returns - values)[:, :(length - margin)]
  return estimator


@gin.configurable(denylist=common_args)
def td_k(gamma, margin):
  """Calculate TD-k advantage.

  The k parameter is assumed to be the same as margin.

  We calculate advantage(s_i) as:

    gamma^n_steps * value(s_{i + n_steps}) - value(s_i) + discounted_rewards

  where discounted_rewards is the sum of rewards in these steps with
  discounting by powers of gamma.

  Args:
    gamma: float, gamma parameter for TD from the underlying task
    margin: number of extra steps in the sequence

  Returns:
    Function (rewards, returns, values, dones) -> advantages, where advantages
    advantages is an array of shape [batch_size, length - margin].
  """
  def estimator(rewards, returns, values, dones, discount_mask):
    del returns
    gammas = mask_discount(gamma, discount_mask)
    # Here we calculate advantage with TD-k, where k=margin.
    k = margin
    assert k > 0
    advantages = np.zeros_like(values[:, k:])
    discount = 1.0
    for i in range(margin):
      advantages += discount * rewards[:, i:-(margin - i)]
      discount *= gammas[:, i:-(margin - i)]
    advantages += discount * values[:, k:]
    # Zero out the future returns at "done" states.
    dones = dones[:, :-k]
    # TPU friendly version of the formula
    # advantages[dones] = rewards[:, :-k][dones]
    advantages = fastmath.index_update(advantages,
                                       dones,
                                       rewards[:, :-k][dones])
    # Subtract the baseline (value).
    advantages -= values[:, :-k]
    return advantages
  return estimator


@gin.configurable(denylist=common_args)
def td_lambda(gamma, margin, lambda_=0.95):
  """Calculate TD-lambda advantage.

  The estimated return is an exponentially-weighted average of different TD-k
  returns.

  Args:
    gamma: float, gamma parameter for TD from the underlying task
    margin: number of extra steps in the sequence
    lambda_: float, the lambda parameter of TD-lambda

  Returns:
    Function (rewards, returns, values, dones) -> advantages, where advantages
    advantages is an array of shape [batch_size, length - margin].
  """
  def estimator(rewards, returns, values, dones, discount_mask):
    gammas = mask_discount(gamma, discount_mask)
    lambdas = mask_discount(lambda_, discount_mask)
    td_returns = np.zeros_like(returns)
    (_, length) = returns.shape
    td_returns[:, -1] = values[:, -1]
    for i in reversed(range(length - 1)):
      lambda_i = lambdas[:, i]
      td_returns[:, i] = rewards[:, i] + (1 - dones[:, i]) * gammas[:, i] * (
          (1 - lambda_i) * values[:, i + 1] + lambda_i * td_returns[:, i + 1]
      )
    return (td_returns - values)[:, :(returns.shape[1] - margin)]
  return estimator


@gin.configurable(denylist=common_args)
def gae(gamma, margin, lambda_=0.95):
  """Calculate Generalized Advantage Estimation.

  Calculate state values bootstrapping off the following state values -
  Generalized Advantage Estimation https://arxiv.org/abs/1506.02438

  Args:
    gamma: float, gamma parameter for TD from the underlying task
    margin: number of extra steps in the sequence
    lambda_: float, the lambda parameter of GAE

  Returns:
    Function (rewards, returns, values, dones) -> advantages, where advantages
    advantages is an array of shape [batch_size, length - margin].
  """
  def estimator(rewards, returns, values, dones, discount_mask):
    del returns
    gammas = mask_discount(gamma, discount_mask)
    lambdas = mask_discount(lambda_, discount_mask)
    advantages = np.zeros_like(rewards)
    (_, length) = rewards.shape

    for i in reversed(range(length - 1)):
      bellman_delta = rewards[:, i] - values[:, i] + (1 - dones[:, i]) * (
          gammas[:, i] * values[:, i + 1]
      )
      advantages[:, i] = bellman_delta + (1 - dones[:, i]) * (
          gammas[:, i] * lambdas[:, i] * advantages[:, i + 1]
      )

    return advantages[:, :(rewards.shape[1] - margin)]
  return estimator
