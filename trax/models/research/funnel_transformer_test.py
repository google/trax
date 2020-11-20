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
"""Tests for Funnel-Transformer models."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import layers as tl
from trax import shapes
import trax.models.research.funnel_transformer as ft


class FunnelTransformerTest(parameterized.TestCase):

  def test_mean_pool(self):
    x = np.ones((1, 4, 1))
    x[0, :3, 0] = [5., 2., 4.]

    pooling = ft.PoolLayer(tl.AvgPool, (2,), (2,))
    y = pooling(x)

    self.assertEqual(y.shape, (1, 2, 1))
    self.assertEqual(y.tolist(), [[[5.], [3.]]])

  def test_mask_pool(self):
    x = np.array([1, 0, 0, 1], dtype=bool).reshape((1, 1, 1, 4))
    pooling_cls = ft.MaskPool((2,), (2,))
    y1 = pooling_cls(x)

    self.assertEqual(y1.shape, (1, 1, 1, 2))
    self.assertEqual(y1.squeeze().tolist(), [True, False])

    pooling_without_cls = ft.MaskPool((2,), (2,), separate_cls=False)
    y2 = pooling_without_cls(x)

    self.assertEqual(y2.shape, (1, 1, 1, 2))
    self.assertEqual(y2.squeeze().tolist(), [True, True])

  def test_upsampler(self):
    long = np.ones((1, 8, 1))
    short = np.ones((1, 2, 1))
    total_pool_size = long.shape[1] // short.shape[1]
    up_cls = ft._Upsampler(total_pool_size, separate_cls=True)
    up = ft._Upsampler(total_pool_size, separate_cls=False)

    y_cls = up_cls([short, long])
    y = up((short, long))
    self.assertEqual(y_cls.shape, long.shape)
    self.assertEqual(y.shape, long.shape)

    self.assertEqual(y_cls.squeeze().tolist(), 5*[2] + 3*[1])
    self.assertEqual(y.squeeze().tolist(), 8*[2])

  def test_funnel_block_forward_shape(self):
    n_even = 4
    d_model = 8

    x = np.ones((1, n_even, d_model), dtype=np.float)
    mask = np.ones((1, n_even), dtype=np.int32)

    masker = tl.PaddingMask()
    mask = masker(mask)

    block = tl.Serial(
        ft._FunnelBlock(d_model, 8, 2, 0.1, None, 'train', tl.Relu,
                        tl.AvgPool, (2,), (2,), separate_cls=True))

    xs = [x, mask]
    _, _ = block.init(shapes.signature(xs))

    y, _ = block(xs)

    self.assertEqual(y.shape, (1, n_even // 2, d_model))

  def test_funnel_transformer_encoder_forward_shape(self):
    n_classes = 5
    model = ft.FunnelTransformerEncoder(2, n_classes=n_classes, d_model=8,
                                        d_ff=8, encoder_segment_lengths=(1, 1),
                                        n_heads=2, max_len=8)

    batch_size = 2
    n_tokens = 4
    x = np.ones((batch_size, n_tokens), dtype=np.int32)
    _ = model.init(shapes.signature(x))
    y = model(x)

    self.assertEqual(y.shape, (batch_size, n_classes))

  def test_funnel_transformer_forward_shape(self):
    d_model = 8
    vocab_size = 7
    model = ft.FunnelTransformer(7, d_model=d_model, d_ff=8,
                                 encoder_segment_lengths=(1, 1),
                                 n_decoder_blocks=1, n_heads=2, max_len=8)

    batch_size = 2
    n_tokens = 4
    x = np.ones((batch_size, n_tokens), dtype=np.int32)
    _ = model.init(shapes.signature(x))
    y = model(x)

    self.assertEqual(y.shape, (batch_size, n_tokens, vocab_size))


if __name__ == '__main__':
  absltest.main()
