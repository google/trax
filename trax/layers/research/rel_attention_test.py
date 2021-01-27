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
"""Tests for trax.layers.relattention."""

from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl
import trax.layers.research.rel_attention as ra


def _get_xs(q, k, batch=1, d_model=5):
  queries, keys = np.zeros((batch, q, d_model)), \
                  np.zeros((batch, k, d_model))
  xs = [queries, keys, keys]
  return xs


class RelAttentionTest(absltest.TestCase):

  def test_shift_right_cls(self):
    layer = ra.ShiftRightCls(5)
    x = np.array([[1, 2, 3, 4]])
    _, _ = layer.init(shapes.signature(x))
    y = layer(x)

    self.assertEqual(tl.to_list(y), [[5, 1, 2, 3]])

  def test_fast_shift_matrix_funnel_1(self):
    layer = ra._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.]]]]).astype(np.float32)

    y = layer(x, funnel_factor=1, is_upsampling=False)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[[0., 1., 2., 3.],
                                       [-1., 0., 1., 2.],
                                       [-2., -1., 0., 1.],
                                       [-3., -2., -1., 0.]]]])

  def test_fast_shift_matrix_funnel_2(self):
    layer = ra._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.]]]]).astype(np.float32)

    y = layer(x, funnel_factor=2, is_upsampling=False)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[[0., 1., 2., 3.],
                                       [-2., -1., 0., 1.]]]])

  def test_fast_shift_matrix_funnel_2_upsampling(self):
    layer = ra._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.],
                    [-3., -2., -1., 0., 1., 2., 3.], ]]]).astype(np.float32)

    y = layer(x, funnel_factor=2, is_upsampling=True)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(tl.to_list(y), [[[[0., 2.],
                                       [-1., 1.],
                                       [-2, 0.],
                                       [-3., -1.]]]])

  def test_create_mask_layer_downsample(self):
    layer = ra.CreateAttentionMaskLayer()
    xs = _get_xs(q=2, k=4)
    layer.init(shapes.signature(xs))
    _, _, _, mask = layer(xs)
    self.assertEqual(mask.shape, (1, 1, 2, 4))
    np.testing.assert_equal(tl.to_list(mask), [[[[True, True, False, False],
                                                 [True, True, True, True]]]])

  def test_create_mask_layer_upsample(self):
    layer = ra.CreateAttentionMaskLayer()
    xs = _get_xs(q=4, k=2)
    layer.init(shapes.signature(xs))
    _, _, _, mask = layer(xs)
    self.assertEqual(mask.shape, (1, 1, 4, 2))
    np.testing.assert_equal(tl.to_list(mask), [[[[True, False],
                                                 [True, False],
                                                 [True, True],
                                                 [True, True]]]])

  def test_create_mask_layer(self):
    layer = ra.CreateAttentionMaskLayer()
    xs = _get_xs(q=2, k=2)
    layer.init(shapes.signature(xs))
    _, _, _, mask = layer(xs)
    self.assertEqual(mask.shape, (1, 1, 2, 2))
    np.testing.assert_equal(tl.to_list(mask), [[[[True, False],
                                                 [True, True]]]])


if __name__ == '__main__':
  absltest.main()
