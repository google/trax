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
"""Tests for conv layers."""

from absl.testing import absltest
import numpy as np

import trax.layers as tl


class MaxPoolTest(absltest.TestCase):

  def test_forward_shape(self):
    layer = tl.MaxPool(pool_size=(2, 2), strides=(1, 2))
    x = np.ones((11, 6, 4, 17))
    y = layer(x)
    self.assertEqual(y.shape, (11, 5, 2, 17))

  def test_forward(self):
    layer = tl.MaxPool(pool_size=(2, 2), strides=(2, 2))
    x = np.array([[
        [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]],
        [[4, 2, 3], [7, 1, 2], [40, 20, 30], [70, 10, 20]],
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[[7, 5, 6], [70, 50, 60]]]])

  def test_padding_default(self):
    layer = tl.MaxPool(pool_size=(3,), strides=(3,))

    # Discard incomplete window at end: [[3, 6], [4, 5]].
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6], [4, 5]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[2, 9]]])

  def test_padding_same(self):
    layer = tl.MaxPool(pool_size=(3,), strides=(3,), padding='SAME')

    # One padding position needed; add at end.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6], [4, 5]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[2, 9], [4, 6]]])

    # Two padding positions needed; add one at end and one at start.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[1, 9], [3, 7]]])


class SumPoolTest(absltest.TestCase):

  def test_forward_shape(self):
    layer = tl.SumPool(pool_size=(2, 2), strides=(1, 2))
    x = np.ones((11, 6, 4, 17))
    y = layer(x)
    self.assertEqual(y.shape, (11, 5, 2, 17))

  def test_forward(self):
    layer = tl.SumPool(pool_size=(2, 2), strides=(2, 2))
    x = np.array([[
        [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]],
        [[4, 2, 3], [7, 1, 2], [40, 20, 30], [70, 10, 20]],
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[[16, 10, 14], [160, 100, 140]]]])

  def test_padding_same(self):
    layer = tl.SumPool(pool_size=(3,), strides=(3,), padding='SAME')

    # One padding position needed; add at end.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6], [4, 5]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[3, 24], [7, 11]]])

    # Two padding positions needed; add one at end and one at start.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[1, 17], [5, 13]]])


class AvgPoolTest(absltest.TestCase):

  def test_forward_shape(self):
    layer = tl.AvgPool(pool_size=(2, 2), strides=(1, 2))
    x = np.ones((11, 6, 4, 17))
    y = layer(x)
    self.assertEqual(y.shape, (11, 5, 2, 17))

  def test_forward(self):
    layer = tl.AvgPool(pool_size=(2, 2), strides=(2, 2))
    x = np.array([[
        [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]],
        [[4, 2, 3], [7, 1, 2], [40, 20, 30], [70, 10, 20]],
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[[4.0, 2.5, 3.5], [40, 25, 35]]]])

  def test_padding_same(self):
    layer = tl.AvgPool(pool_size=(3,), strides=(3,), padding='SAME')

    # One padding position needed; add at end.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6], [4, 5]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[1, 8], [3.5, 5.5]]])

    # Two padding positions needed; add one at end and one at start.
    x = np.array([[
        [0, 9], [1, 8], [2, 7], [3, 6]
    ]])
    y = layer(x)
    self.assertEqual(tl.to_list(y), [[[.5, 8.5], [2.5, 6.5]]])


if __name__ == '__main__':
  absltest.main()
