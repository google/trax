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
"""Tests for initializers."""

from absl.testing import absltest
from absl.testing import parameterized
import gym
import numpy as np

from trax.rl import distributions


class DistributionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('discrete', gym.spaces.Discrete(n=4)),
      ('multi_discrete', gym.spaces.MultiDiscrete(nvec=[5, 5])),
      ('gaussian', gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(4, 5))),
  )
  def test_shapes(self, space):
    batch_shape = (2, 3)
    distribution = distributions.create_distribution(space)
    inputs = np.random.random(batch_shape + (distribution.n_inputs,))
    point = distribution.sample(inputs)
    self.assertEqual(point.shape, batch_shape + space.shape)
    # Check if the datatypes are compatible, i.e. either both floating or both
    # integral.
    self.assertEqual(
        isinstance(point.dtype, float), isinstance(space.dtype, float)
    )
    log_prob = distribution.log_prob(inputs, point)
    self.assertEqual(log_prob.shape, batch_shape)

  @parameterized.named_parameters(('1d', 1), ('2d', 2))
  def test_gaussian_probability_sums_to_one(self, n_dims):
    std = 1.0
    n_samples = 10000

    distribution = distributions.Gaussian(shape=(n_dims,), std=std)
    means = np.random.random((3, n_dims))
    # Monte carlo integration over [mean - 3 * std, mean + 3 * std] across
    # all dimensions.
    means = np.broadcast_to(means, (n_samples,) + means.shape)
    probs = (6 * std) ** n_dims * np.mean(
        np.exp(distribution.log_prob(
            means, np.random.uniform(means - 3 * std, means + 3 * std)
        )),
        axis=0,
    )
    # Should sum to one. High tolerance because of variance and cutting off the
    # tails.
    np.testing.assert_allclose(probs, np.ones_like(probs), atol=0.05)


if __name__ == '__main__':
  absltest.main()
