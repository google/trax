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
"""Functions for RL computations."""


def calculate_advantage(rewards, returns, values, gamma, n_steps):
  """Calculate advantage, the default way if n_steps=0, else using TD(gamma).

  We assume the values are a tensor of shape [batch_size, length] and this
  is the same shape as rewards and returns.

  If n_steps is 0, the advantages are defined as: returns - values.
  For larger n_steps, we use TD(gamma) and calculate advantage(s_i) as:

    gamma^n_steps * value(s_{i + n_steps}) - value(s_i) - discounted_rewards

  where discounted_rewards is the sum of rewards in these steps with
  discounting by powers of gamma.

  Args:
    rewards: the rewards, tensor of shape [batch_size, length]
    returns: discounted returns, tensor of shape [batch_size, length]
    values: the value function computed for this trajectory (shape as above)
    gamma: float, gamma parameter for TD from the underlying task
    n_steps: for how many steps to do TD (if 0, we use default advantage)

  Returns:
    the advantages, a tensor of shape [batch_size, length - n_steps].
  """
  # TODO(afrozm): split into smaller more easily understandable functions.
  if n_steps < 1:  # Default advantage: returns - values.
    return returns - values
  # Here we calculate advantage with TD(gamma) used for n steps.
  cur_advantage = (gamma**n_steps) * values[:, n_steps:] - values[:, :-n_steps]
  discount = 1.0
  for i in range(n_steps):
    cur_advantage += discount * rewards[:, i:-(n_steps - i)]
    discount *= gamma
  return cur_advantage
