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
"""Tests for Transformer models."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models.research import transformer_no_enc_dec_attention


class TransformerNoEncDecAttentionTest(absltest.TestCase):

  def test_transformer_noencdec_forward_shape(self):
    input_vocab_size = 16
    output_vocab_size = 16

    model = transformer_no_enc_dec_attention.TransformerNoEncDecAttention(
        input_vocab_size, output_vocab_size, d_model=32, d_ff=64,
        n_encoder_layers=2, n_decoder_layers=2, n_heads=2)

    enc_toks = np.array(
        [[6, 2, 0, 0, 0, 0],
         [6, 3, 7, 0, 0, 0]])
    dec_toks = np.array(
        [[4, 2, 0, 0],
         [8, 5, 0, 0]])

    xs = [enc_toks, dec_toks]
    _, _ = model.init(shapes.signature(xs))

    # decoder output, decoder mask
    ys = model(xs)

    # (B, L2, H)
    self.assertEqual(ys[0].shape,
                     (dec_toks.shape[0], dec_toks.shape[1], output_vocab_size))

    self.assertEqual(ys[1].shape, dec_toks.shape)


if __name__ == '__main__':
  absltest.main()
