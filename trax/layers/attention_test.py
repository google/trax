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
"""Tests for trax.layers.attention."""

import functools
from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl
from trax.layers import test_utils


class AttentionTest(absltest.TestCase):

  def test_simple_call(self):
    layer = tl.CausalAttention(d_feature=4, n_heads=2)
    x = [np.array([[[2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],]]),
         np.array([[[[1, 0, 1]]]])]
    _, _ = layer.init(shapes.signature(x))

    y, mask = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))
    self.assertEqual(mask.shape, (1, 1, 1, 3))

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


class CausalAttentionTest(absltest.TestCase):

  def test_simple_call(self):
    layer = tl.CausalAttention(d_feature=4, n_heads=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))

  def test_deterministic_eval(self):
    d_model = 32
    seq_len = 3
    x_shape = (1, seq_len, d_model)
    inp = np.ones(x_shape).astype(np.float32)

    model_fn = functools.partial(
        tl.CausalAttention,
        d_feature=d_model,
        n_heads=4,
        )

    test_utils.test_eval_is_deterministic(inp, model_fn)

  def test_predict_equals_eval(self):
    d_model = 32
    seq_len = 10
    x_shape = (1, seq_len, d_model)
    inp = np.ones(x_shape).astype(np.float32)

    model_fn = functools.partial(
        tl.CausalAttention,
        d_feature=d_model,
        n_heads=4,
        )

    test_utils.test_eval_equals_predict(inp, model_fn)


class PositionalEncodingTest(absltest.TestCase):

  def test_simple_call(self):
    layer = tl.PositionalEncoding(max_len=8)
    x = np.array([[[2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]]])
    layer.init(shapes.signature(x))
    y = layer(x)
    self.assertEqual(y.shape, (1, 2, 4))

  def test_predict(self):
    layer = tl.PositionalEncoding(max_len=8)
    x = np.array([[[2.0, 3.0], [1.0, 2.0], [0.0, 1.0], [3.0, 4.0]]])
    self.assertEqual(x.shape, (1, 4, 2))
    layer.init(shapes.signature(x))
    y = layer(x)
    self.assertEqual(y.shape, (1, 4, 2))
    layer = tl.PositionalEncoding(max_len=8, mode='predict')
    layer.init(shapes.signature(x[:, :1, :]))
    y0 = layer(x[:, :1, :])   # just the first token
    self.assertEqual(y0.shape, (1, 1, 2))
    self.assertTrue(np.array_equal(y0, y[:, :1, :]))
    y1 = layer(x[:, 1:3, :])  # now the next 2 tokens
    self.assertEqual(y1.shape, (1, 2, 2))
    self.assertTrue(np.array_equal(y1, y[:, 1:3, :]))
    y2 = layer(x[:, 3:4, :])  # final one token
    self.assertEqual(y2.shape, (1, 1, 2))
    self.assertTrue(np.array_equal(y2, y[:, 3:4, :]))

  def test_predict_equals_eval(self):
    x = np.array([[[2.0, 3.0], [1.0, 2.0], [0.0, 1.0], [3.0, 4.0]]])
    self.assertEqual(x.shape, (1, 4, 2))

    layer_eval = tl.PositionalEncoding(max_len=8, d_feature=4, mode='eval')
    layer_eval.init(shapes.signature(x))

    output_eval = layer_eval(x)

    layer_predict = tl.PositionalEncoding(max_len=8, d_feature=4,
                                          mode='predict')
    layer_predict.init(shapes.signature(x))
    layer_predict.weights = layer_eval.weights

    output_predict = layer_predict(x)
    self.assertTrue(np.array_equal(output_eval, output_predict))


if __name__ == '__main__':
  absltest.main()
