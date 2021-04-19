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


class RelAttentionTest(absltest.TestCase):

  def test_fast_shift_matrix(self):
    layer = ra._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0.], [-3., -2., -1.,
                                          0.], [-3., -2., -1., 0.],
                    [-3., -2., -1., 0.]]]]).astype(np.float32)

    y = layer(x)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(
        tl.to_list(y), [[[[0., 0., -3., -2.], [-1., 0., 0., -3.],
                          [-2., -1., 0., 0.], [-3., -2., -1., 0.]]]])

  def test_create_mask_layer(self):
    layer = ra.AttentionMaskLayer()
    xs = np.zeros((1, 2, 5))
    layer.init(shapes.signature(xs))
    mask = layer(xs)
    self.assertEqual(mask.shape, (2, 2))
    np.testing.assert_equal(tl.to_list(mask), [[True, False], [True, True]])

  def test_create_mask_layer_predict(self):
    layer = ra.AttentionMaskLayer(
        total_kv_pooling=2,
        n_raw_tokens_generated=1,
        max_inference_length=3,
        mode='predict')
    xs = np.zeros((1, 1, 5))
    layer.init(shapes.signature(xs))

    for _ in range(2):
      mask = layer(xs)
      self.assertEqual(mask.shape, (1, 3))
      np.testing.assert_equal(tl.to_list(mask), [[True, False, False]])

    for _ in range(2):
      mask = layer(xs)
      self.assertEqual(mask.shape, (1, 3))
      np.testing.assert_equal(tl.to_list(mask), [[True, True, False]])

    for _ in range(2):
      mask = layer(xs)
      self.assertEqual(mask.shape, (1, 3))
      np.testing.assert_equal(tl.to_list(mask), [[True, True, True]])

  def test_positional_embeddings_predict(self):
    d_feature = 10
    total_kv_pooling = 2
    max_inference_length = 3
    batch_size = 1
    token_length = 1

    layer = ra.PositionalEmbeddings(
        d_feature=d_feature,
        total_kv_pooling=total_kv_pooling,
        n_raw_tokens_generated=1,
        max_inference_length=max_inference_length,
        mode='predict')

    xs = np.zeros((batch_size, token_length, d_feature))
    layer.init(shapes.signature(xs))

    for _ in range(2):
      pos_emb = layer(xs)
      real_positions = np.array([0, 1, 2])
      self.assertEqual(pos_emb.shape, (max_inference_length, d_feature))
      np.testing.assert_allclose(
          pos_emb,
          ra.Sinusoidal_Embeddings(real_positions, d_feature=d_feature))

    for _ in range(2):
      pos_emb = layer(xs)
      real_positions = np.array([-1, 0, 1])
      self.assertEqual(pos_emb.shape, (max_inference_length, d_feature))
      np.testing.assert_allclose(
          pos_emb,
          ra.Sinusoidal_Embeddings(real_positions, d_feature=d_feature))

    for _ in range(2):
      pos_emb = layer(xs)
      real_positions = np.array([-2, -1, 0])
      self.assertEqual(pos_emb.shape, (max_inference_length, d_feature))
      np.testing.assert_allclose(
          pos_emb,
          ra.Sinusoidal_Embeddings(real_positions, d_feature=d_feature))


if __name__ == '__main__':
  absltest.main()
