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
"""Tests for trax.rl.advantages."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax.rl import advantages


def calc_bias_and_variance(x, true_mean):
  sample_mean = np.mean(x)
  bias = np.abs(sample_mean - true_mean)
  variance = np.mean((x - sample_mean) ** 2)
  return (bias, variance)


def calc_returns(rewards, gamma):
  returns = np.zeros_like(rewards)
  current_return = np.zeros_like(rewards[:, 0])
  for t in reversed(range(rewards.shape[1])):
    current_return = gamma * current_return + rewards[:, t]
    returns[:, t] = current_return
  return returns


def estimate_advantage_bias_and_variance(
    advantage_fn,
    mean_reward=1.23,
    reward_noise=0.45,
    n_samples=10000,
    length=5,
    gamma=0.9,
    margin=0,
    **advantage_kwargs
):
  advantage_fn = advantage_fn(gamma, margin, **advantage_kwargs)
  rewards = np.random.normal(
      loc=mean_reward, scale=reward_noise, size=(n_samples, length)
  )
  returns = calc_returns(rewards, gamma=gamma)
  values = np.zeros_like(returns)
  dones = np.zeros_like(returns, dtype=np.bool)
  adv = advantage_fn(rewards, returns, values, dones)
  mean_return = calc_returns(
      np.full((1, length), fill_value=mean_reward), gamma=gamma
  )[0, 0]
  return calc_bias_and_variance(adv[:, 0], mean_return)


class AdvantagesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('monte_carlo', advantages.monte_carlo),
      ('td_k', advantages.td_k),
      ('td_lambda', advantages.td_lambda),
      ('gae', advantages.gae),
  )
  def test_shapes(self, advantage_fn):
    rewards = np.array([[1, 1, 1]], dtype=np.float32)
    returns = np.array([[3, 2, 1]], dtype=np.float32)
    values = np.array([[2, 2, 2]], dtype=np.float32)
    dones = np.array([[False, False, True]])
    adv1 = advantage_fn(gamma=1, margin=1)(rewards, returns, values, dones)
    self.assertEqual(adv1.shape, (1, 2))
    adv2 = advantage_fn(gamma=1, margin=2)(rewards, returns, values, dones)
    self.assertEqual(adv2.shape, (1, 1))

  def test_monte_carlo_bias_is_zero(self):
    (bias, _) = estimate_advantage_bias_and_variance(
        advantages.monte_carlo, margin=3
    )
    np.testing.assert_allclose(bias, 0, atol=0.1)

  def test_td_k_variance_lower_than_monte_carlo(self):
    (_, var_td_3) = estimate_advantage_bias_and_variance(
        advantages.td_k, margin=3
    )
    (_, var_mc) = estimate_advantage_bias_and_variance(advantages.monte_carlo)
    self.assertLess(var_td_3, var_mc)

  @parameterized.named_parameters(('1_2', 1, 2), ('2_3', 2, 3))
  def test_td_k_bias_decreases_with_k(self, k1, k2):
    (bias1, _) = estimate_advantage_bias_and_variance(
        advantages.td_k, margin=k1
    )
    (bias2, _) = estimate_advantage_bias_and_variance(
        advantages.td_k, margin=k2
    )
    self.assertGreater(bias1, bias2)

  @parameterized.named_parameters(('1_2', 1, 2), ('2_3', 2, 3))
  def test_td_k_variance_increases_with_k(self, k1, k2):
    (_, var1) = estimate_advantage_bias_and_variance(
        advantages.td_k, margin=k1
    )
    (_, var2) = estimate_advantage_bias_and_variance(
        advantages.td_k, margin=k2
    )
    self.assertLess(var1, var2)

  def test_td_lambda_variance_lower_than_monte_carlo(self):
    (_, var_td_095) = estimate_advantage_bias_and_variance(
        advantages.td_lambda, lambda_=0.95
    )
    (_, var_mc) = estimate_advantage_bias_and_variance(advantages.monte_carlo)
    self.assertLess(var_td_095, var_mc)

  @parameterized.named_parameters(
      ('td_lambda_0.5_0.7', advantages.td_lambda, 0.5, 0.7),
      ('td_lambda_0.7_0.9', advantages.td_lambda, 0.7, 0.9),
      ('gae_0.5_0.7', advantages.gae, 0.5, 0.7),
      ('gae_0.7_0.9', advantages.gae, 0.7, 0.9),
  )
  def test_bias_decreases_with_lambda(self, advantage_fn, lambda1, lambda2):
    (bias1, _) = estimate_advantage_bias_and_variance(
        advantage_fn, lambda_=lambda1
    )
    (bias2, _) = estimate_advantage_bias_and_variance(
        advantage_fn, lambda_=lambda2
    )
    self.assertGreater(bias1, bias2)

  @parameterized.named_parameters(('0.5_0.7', 0.5, 0.7), ('0.7_0.9', 0.7, 0.9))
  def test_variance_increases_with_lambda(self, lambda1, lambda2):
    (_, var1) = estimate_advantage_bias_and_variance(
        advantages.td_lambda, lambda_=lambda1
    )
    (_, var2) = estimate_advantage_bias_and_variance(
        advantages.td_lambda, lambda_=lambda2
    )
    self.assertLess(var1, var2)

  @parameterized.named_parameters(
      ('monte_carlo', advantages.monte_carlo),
      ('td_k', advantages.td_k),
      ('td_lambda', advantages.td_lambda),
      ('gae', advantages.gae),
  )
  def test_advantage_future_return_is_zero_at_done(self, advantage_fn):
    rewards = np.array([[1, 1, 1]], dtype=np.float32)
    returns = np.array([[3, 2, 1]], dtype=np.float32)
    values = np.array([[2, 2, 2]], dtype=np.float32)
    dones = np.array([[False, True, False]])
    adv = advantage_fn(gamma=0.9, margin=1)(rewards, returns, values, dones)
    target_returns = values[:, :-1] + adv
    # Assert that in the "done" state the future return in the advantage is
    # zero, i.e. the advantage is equal to the reward.
    np.testing.assert_almost_equal(target_returns[0, 1], rewards[0, 1])


if __name__ == '__main__':
  absltest.main()
