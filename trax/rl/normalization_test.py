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
"""Tests for trax.rl.normalization."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.rl import normalization


class NormalizationTest(absltest.TestCase):

  def test_running_mean(self):
    x = np.random.uniform(size=10)
    state = normalization.running_mean_init(shape=())
    for i in range(len(x)):
      state = normalization.running_mean_update(x[i], state)
      np.testing.assert_almost_equal(
          normalization.running_mean_get_mean(state), np.mean(x[:i + 1])
      )

  def test_running_variance(self):
    x = np.random.uniform(size=10)
    state = normalization.running_mean_and_variance_init(shape=())
    for i in range(len(x)):
      state = normalization.running_mean_and_variance_update(x[i], state)
      np.testing.assert_almost_equal(
          normalization.running_mean_and_variance_get_variance(state),
          np.var(x[:i + 1]),
      )

  def test_normalize_collect(self):
    x = np.random.uniform(size=(2, 3, 4, 5))
    normalize = normalization.Normalize(mode='collect')
    normalize.init(shapes.signature(x))
    old_state = normalize.state
    y = normalize(x)
    with self.assertRaises(AssertionError):
      np.testing.assert_equal(normalize.state, old_state)
    with self.assertRaises(AssertionError):
      np.testing.assert_almost_equal(x, y)

  def test_normalize_train(self):
    x = np.random.uniform(size=(2, 3, 4, 5))
    normalize = normalization.Normalize(mode='train', epsilon=0.0)
    normalize.init(shapes.signature(x))
    old_state = normalize.state
    y = normalize(x)
    np.testing.assert_equal(normalize.state, old_state)
    np.testing.assert_almost_equal(x, y)


if __name__ == '__main__':
  absltest.main()
