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
"""RL advantage estimators."""

import gin
import numpy as np


common_args = [
    'rewards', 'returns', 'values', 'dones', 'gamma', 'n_extra_steps'
]


@gin.configurable(blacklist=common_args)
def monte_carlo(rewards, returns, values, dones, gamma, n_extra_steps):
  """Calculate Monte Carlo advantage.

  We assume the values are a tensor of shape [batch_size, length] and this
  is the same shape as rewards and returns.

  Args:
    rewards: the rewards, tensor of shape [batch_size, length]
    returns: discounted returns, tensor of shape [batch_size, length]
    values: the value function computed for this trajectory (shape as above)
    dones: trajectory termination flags
    gamma: float, gamma parameter for TD from the underlying task
    n_extra_steps: number of extra steps in the sequence

  Returns:
    the advantages, a tensor of shape [batch_size, length - n_extra_steps].
  """
  del gamma
  (_, length) = returns.shape
  # Make sure that the future returns and the values at "done" states are zero.
  returns[dones] = rewards[dones]
  values[dones] = 0
  return (returns - values)[:, :(length - n_extra_steps)]


@gin.configurable(blacklist=common_args)
def td_k(rewards, returns, values, dones, gamma, n_extra_steps):
  """Calculate TD-k advantage.

  The k parameter is assumed to be the same as n_extra_steps.

  We calculate advantage(s_i) as:

    gamma^n_steps * value(s_{i + n_steps}) - value(s_i) + discounted_rewards

  where discounted_rewards is the sum of rewards in these steps with
  discounting by powers of gamma.

  Args:
    rewards: the rewards, tensor of shape [batch_size, length]
    returns: discounted returns, tensor of shape [batch_size, length]
    values: the value function computed for this trajectory (shape as above)
    dones: trajectory termination flags
    gamma: float, gamma parameter for TD from the underlying task
    n_extra_steps: number of extra steps in the sequence, also controls the
      number of steps k

  Returns:
    the advantages, a tensor of shape [batch_size, length - n_extra_steps].
  """
  del returns
  # Here we calculate advantage with TD-k, where k=n_extra_steps.
  k = n_extra_steps
  assert k > 0
  advantages = (gamma ** k) * values[:, k:]
  discount = 1.0
  for i in range(n_extra_steps):
    advantages += discount * rewards[:, i:-(n_extra_steps - i)]
    discount *= gamma
  # Zero out the future returns at "done" states.
  dones = dones[:, :-k]
  advantages[dones] = rewards[:, :-k][dones]
  # Subtract the baseline (value).
  advantages -= values[:, :-k]
  return advantages


@gin.configurable(blacklist=common_args)
def td_lambda(
    rewards, returns, values, dones, gamma, n_extra_steps, lambda_=0.95
):
  """Calculate TD-lambda advantage.

  The estimated return is an exponentially-weighted average of different TD-k
  returns.

  Args:
    rewards: the rewards, tensor of shape [batch_size, length]
    returns: discounted returns, tensor of shape [batch_size, length]
    values: the value function computed for this trajectory (shape as above)
    dones: trajectory termination flags
    gamma: float, gamma parameter for TD from the underlying task
    n_extra_steps: number of extra steps in the sequence
    lambda_: discount parameter of the exponentially-weighted average

  Returns:
    the advantages, a tensor of shape [batch_size, length - n_extra_steps].
  """
  td_returns = np.zeros_like(returns)
  (_, length) = returns.shape
  td_returns[:, -1] = values[:, -1]
  for i in reversed(range(length - 1)):
    td_returns[:, i] = rewards[:, i] + (1 - dones[:, i]) * gamma * (
        (1 - lambda_) * values[:, i + 1] + lambda_ * td_returns[:, i + 1]
    )
  return (td_returns - values)[:, :(returns.shape[1] - n_extra_steps)]


common_args = ['rewards', 'values', 'gamma', 'gae_lambda', 'n_extra_steps']


@gin.configurable(blacklist=common_args)
def discount_gae(rewards, values, gamma, n_extra_steps, gae_lambda=0.95):
  """Calculate Generalized Advantage Estimation.

  Calculate state values bootstrapping off the following state values -
  Generalized Advantage Estimation https://arxiv.org/abs/1506.02438

  Args:
    rewards: the rewards, tensor of shape [batch_size, length]
    values: the value function computed for this trajectory (shape as above)
    gamma: float, gamma parameter for TD from the underlying task
    n_extra_steps: number of extra steps in the sequence
    gae_lambda: discount parameter of the exponentially-weighted average

  Returns:
    the advantages, a tensor of shape [batch_size, length - n_extra_steps].
  """

  advantages = np.zeros_like(rewards)
  (_, length) = rewards.shape

  # Accmulate sums
  sum_accumulator = 0

  for i in reversed(range(length-1)):
    bellman_delta = (rewards[:, i] + gamma * values[:, i + 1] - values[:, i])

    advantages[:, i] = sum_accumulator = (
        bellman_delta + gamma * gae_lambda * sum_accumulator)

  return advantages[:, :(rewards.shape[1] - n_extra_steps)]
