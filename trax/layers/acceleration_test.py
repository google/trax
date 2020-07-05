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
"""Tests for acceleration."""

from absl.testing import absltest

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from trax import layers as tl
from trax import shapes


class AccelerationTest(absltest.TestCase):

  def test_accelerated_same_result(self):
    layer = tl.Dense(2)
    x = np.random.uniform(size=(8, 7))
    layer.init(shapes.signature(x))
    y = layer(x)
    z = tl.Accelerate(layer)(x)
    for i in range(8):
      self.assertAlmostEqual(float(y[i, 0]), float(z[i, 0]), places=4)
      self.assertAlmostEqual(float(y[i, 1]), float(z[i, 1]), places=4)

  def test_accelerated_pad(self):
    layer = tl.Dense(2)
    x = np.random.uniform(size=(3, 7))
    layer.init(shapes.signature(x))
    y = layer(x)
    z = tl.Accelerate(layer)(x)
    self.assertEqual(z.shape, y.shape)
    for i in range(3):
      self.assertAlmostEqual(float(y[i, 0]), float(z[i, 0]), places=4)
      self.assertAlmostEqual(float(y[i, 1]), float(z[i, 1]), places=4)


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
