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
"""Tests for supervised training optimizers."""

from absl.testing import absltest

import numpy as np

from trax import optimizers
from trax.optimizers import momentum


class OptimizersTest(absltest.TestCase):

  def test_slots(self):
    weights_shape = (3, 5)
    weight_tree = np.arange(15).reshape(weights_shape)

    # SGD - an optimizer that doesn't use slots.
    opt_1 = optimizers.SGD(.01)
    self.assertIsNone(opt_1.slots)
    opt_1.tree_init(weight_tree)
    self.assertIsInstance(opt_1.slots, tuple)
    self.assertLen(opt_1.slots, 1)
    self.assertIsNone(opt_1.slots[0])

    # Momentum - an optimizer with slots
    opt_2 = momentum.Momentum(.01)
    self.assertIsNone(opt_2.slots)
    opt_2.tree_init(weight_tree)
    self.assertIsInstance(opt_2.slots, tuple)
    self.assertLen(opt_2.slots, 1)
    self.assertEqual(weights_shape, opt_2.slots[0].shape)


if __name__ == '__main__':
  absltest.main()
