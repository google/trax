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
"""Tests for trax.layers.attention."""

from absl.testing import absltest
import numpy as np

import trax.layers as tl


class AttentionTest(absltest.TestCase):

  def test_shift_right(self):
    # Test shifts right on axis=1
    layer = tl.ShiftRight()
    x = np.array([[[9, 9, 9],
                   [8, 8, 8],
                   [7, 7, 7],
                   [6, 6, 6]],
                  [[99, 98, 97],
                   [96, 95, 94],
                   [93, 92, 91],
                   [90, 89, 88]]])
    y = layer(x)
    self.assertEqual(x.shape, y.shape)
    self.assertEqual(tl.to_list(y), [[[0, 0, 0],
                                      [9, 9, 9],
                                      [8, 8, 8],
                                      [7, 7, 7]],
                                     [[0, 0, 0],
                                      [99, 98, 97],
                                      [96, 95, 94],
                                      [93, 92, 91]]])

  def test_shift_right_float(self):
    layer = tl.ShiftRight()
    x = np.array([[[9, 9, 9],
                   [8, 8, 8],
                   [7, 7, 7],
                   [6, 6, 6]],
                  [[99, 98, 97],
                   [96, 95, 94],
                   [93, 92, 91],
                   [90, 89, 88]]]).astype(np.float32)
    x /= 2.0
    self.assertEqual(x.dtype, np.float32)

    y = layer(x)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[0.0, 0.0, 0.0],
                                      [4.5, 4.5, 4.5],
                                      [4.0, 4.0, 4.0],
                                      [3.5, 3.5, 3.5]],
                                     [[0.0, 0.0, 0.0],
                                      [49.5, 49.0, 48.5],
                                      [48.0, 47.5, 47.0],
                                      [46.5, 46.0, 45.5]]])

  def test_padding_mask(self):
    layer = tl.PaddingMask()
    x = np.array([
        [1., 2., 3., 4., 0.],
        [1., 2., 3., 0., 0.],
        [1., 2., 0., 0., 0.],
    ])
    y = layer(x)
    self.assertEqual(x.shape, (3, 5))
    self.assertEqual(y.shape, (3, 1, 1, 5))
    np.testing.assert_equal(y, [[[[True, True, True, True, False]]],
                                [[[True, True, True, False, False]]],
                                [[[True, True, False, False, False]]]])


if __name__ == '__main__':
  absltest.main()
