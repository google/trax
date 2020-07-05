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
"""Tests for RL."""

from unittest import mock
from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models import rl


class RLTest(absltest.TestCase):

  def test_policy_forward_shape(self):
    mock_dist = mock.MagicMock()
    mock_dist.n_inputs = 4
    model = rl.Policy(policy_distribution=mock_dist)
    x = np.ones((2, 3))
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (2, 4))

  def test_value_forward_shape(self):
    model = rl.Value()
    x = np.ones((2, 3))
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (2, 1))

  def test_policy_and_value_forward_shape(self):
    mock_dist = mock.MagicMock()
    mock_dist.n_inputs = 4
    model = rl.PolicyAndValue(policy_distribution=mock_dist)
    x = np.ones((2, 3))
    _, _ = model.init(shapes.signature(x))
    ys = model(x)
    self.assertEqual([y.shape for y in ys], [(2, 4), (2, 1)])


if __name__ == '__main__':
  absltest.main()
