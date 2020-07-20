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
from trax import shapes as trax_shapes
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

  def test_attention_identity_weights(self):
    """A unit test for the Attention layer.

    We are setting weights in
    Serial_in2_out2[
      Dup_out2
      Dup_out2
      Serial_in4_out2[
        Parallel_in3_out3[
          Dense_2
          Dense_2
          Dense_2
        ]
        PureAttention_in4_out2
        Dense_2
      ]
    ]
    and then manually compute dots and compare results
    with the Attention layer. For an elementary introduction
    to this computation see http://jalammar.github.io/illustrated-transformer/
    """

    x = np.array([[[1, 2], [2, 5]]])
    encoding = tl.PositionalEncoding(max_len=10)
    encoding.init(trax_shapes.signature(x))
    enc_x = encoding(x)
    np.testing.assert_equal(
        np.array(x[:, 0, :] + encoding.weights[:, 0, :] == enc_x[:, 0, :]),
        [[True, True]])
    np.testing.assert_equal(
        np.array(x[:, 1, :] + encoding.weights[:, 1, :] == enc_x[:, 1, :]),
        [[True, True]])

    att = tl.Attention(2)
    queries = np.eye(2)
    keys = np.eye(2)
    values = np.eye(2)
    dense = np.eye(2)
    biases = np.zeros(shape=(2))
    att.set_weights_by_index((2, 0, 0), (queries, biases))
    att.set_weights_by_index((2, 0, 1), (keys, biases))
    att.set_weights_by_index((2, 0, 2), (values, biases))
    att.set_weights_by_index((2, 2), (dense, biases))

    # Q * K^T / sqrt{d_feature}
    d_feature = 2
    dots = np.matmul(
        np.matmul(queries, enc_x), np.swapaxes(np.matmul(keys, enc_x), -1,
                                               -2)) / np.sqrt(d_feature)
    # Take softmax of the above
    dots = np.exp(dots)/np.sum(np.exp(dots), axis=-1, keepdims=True)
    # Multiply by values
    out = np.matmul(dots, np.matmul(values, enc_x))
    np.testing.assert_almost_equal(out,
                                   np.array(att((enc_x, None))[0]), decimal=5)


if __name__ == '__main__':
  absltest.main()
