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
"""Tests for RL computation utils."""

from absl.testing import absltest
import numpy as np
from trax.rl import computation_utils


class ComputationUtilsTest(absltest.TestCase):

  def test_calculate_advantage(self):
    """Test calculating advantage."""
    rewards = np.array([[1, 1, 1]], dtype=np.float32)
    returns = np.array([[3, 2, 1]], dtype=np.float32)
    values = np.array([[2, 2, 2]], dtype=np.float32)
    adv1 = computation_utils.calculate_advantage(rewards, returns, values, 1, 0)
    self.assertEqual(adv1.shape, (1, 3))
    self.assertEqual(adv1[0, 0], 1)
    self.assertEqual(adv1[0, 1], 0)
    self.assertEqual(adv1[0, 2], -1)
    adv2 = computation_utils.calculate_advantage(rewards, returns, values, 1, 1)
    self.assertEqual(adv2.shape, (1, 2))
    self.assertEqual(adv2[0, 0], 1)
    self.assertEqual(adv2[0, 1], 1)


if __name__ == '__main__':
  absltest.main()
