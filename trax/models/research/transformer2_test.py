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
"""Tests for Transformer models."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models.research import transformer2


class Transformer2Test(absltest.TestCase):

  def test_concat_with_padding(self):
    vec_e = np.array(
        [[[7, 5, 2, 8, 8, 8, 6, 7],
          [8, 2, 6, 2, 1, 1, 4, 2],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[4, 3, 1, 7, 5, 6, 2, 1],
          [6, 9, 9, 4, 1, 3, 2, 1],
          [3, 8, 2, 4, 7, 9, 4, 1],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    # vec_e[:,:,0] != 0
    mask_e = np.array([[True, True, False, False, False, False],
                       [True, True, True, False, False, False]])

    vec_d = np.array(
        [[[4, 7, 7, 4, 8, 9, 9, 9],
          [6, 8, 2, 9, 3, 6, 6, 8],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[3, 7, 5, 6, 2, 9, 3, 1],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    layer = transformer2.ConcatWithPadding(mode='train')
    inp = (vec_e, vec_d, mask_e, vec_e, vec_d)  # tok_e = vec_e, tok_d = vec_d
    layer.init(shapes.signature(inp))
    y, _, _ = layer(inp)

    np.testing.assert_equal(
        y,
        np.array(
            [[[7, 5, 2, 8, 8, 8, 6, 7],
              [8, 2, 6, 2, 1, 1, 4, 2],
              [4, 7, 7, 4, 8, 9, 9, 9],
              [6, 8, 2, 9, 3, 6, 6, 8],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],

             [[4, 3, 1, 7, 5, 6, 2, 1],
              [6, 9, 9, 4, 1, 3, 2, 1],
              [3, 8, 2, 4, 7, 9, 4, 1],
              [3, 7, 5, 6, 2, 9, 3, 1],
              [4, 7, 3, 2, 1, 1, 1, 6],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        )
    )

  def test_concat_with_padding_predict(self):
    vec_e = np.array(
        [[[7, 5, 2, 8, 8, 8, 6, 7],
          [8, 2, 6, 2, 1, 1, 4, 2],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[4, 3, 1, 7, 5, 6, 2, 1],
          [6, 9, 9, 4, 1, 3, 2, 1],
          [3, 8, 2, 4, 7, 9, 4, 1],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    # vec_e[:,:,0] != 0
    mask_e = np.array([[True, True, False, False, False, False],
                       [True, True, True, False, False, False]])

    vec_d = np.array(
        [[[4, 7, 7, 4, 8, 9, 9, 9],
          [6, 8, 2, 9, 3, 6, 6, 8],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[3, 7, 5, 6, 2, 9, 3, 1],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    layer = transformer2.ConcatWithPadding(mode='predict')
    inp = (vec_e, vec_d, mask_e, vec_e, vec_d)  # tok_e = vec_e, tok_d = vec_d
    _, _ = layer.init(shapes.signature(inp))
    y, _, _ = layer(inp)

    np.testing.assert_equal(
        y,
        np.array(
            [[[7, 5, 2, 8, 8, 8, 6, 7],
              [8, 2, 6, 2, 1, 1, 4, 2],
              [4, 7, 7, 4, 8, 9, 9, 9],
              [6, 8, 2, 9, 3, 6, 6, 8],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],

             [[4, 3, 1, 7, 5, 6, 2, 1],
              [6, 9, 9, 4, 1, 3, 2, 1],
              [3, 8, 2, 4, 7, 9, 4, 1],
              [3, 7, 5, 6, 2, 9, 3, 1],
              [4, 7, 3, 2, 1, 1, 1, 6],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        )
    )

    # On subsequent runs however, we should get vec_d only.
    for _ in range(2):
      y, _, _ = layer(inp)
      np.testing.assert_equal(y, vec_d)

  def test_concat_with_padding2(self):
    vec_e = np.array(
        [[[7, 5, 2, 8, 8, 8, 6, 7],
          [8, 2, 6, 2, 1, 1, 4, 2],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[4, 3, 1, 7, 5, 6, 2, 1],
          [6, 9, 9, 4, 1, 3, 2, 1],
          [3, 8, 2, 4, 7, 9, 4, 1],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    # vec_e[:,:,0] != 0
    mask_e = np.array([[True, True, False, False, False, False],
                       [True, True, True, False, False, False]])

    vec_d = np.array(
        [[[4, 7, 7, 4, 8, 9, 9, 9],
          [6, 8, 2, 9, 3, 6, 6, 8],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[3, 7, 5, 6, 2, 9, 3, 1],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    layer = transformer2.ConcatWithPadding2(mode='train')
    inp = (vec_e, vec_e, vec_d, mask_e, vec_e, vec_d)
    layer.init(shapes.signature(inp))
    y1, y2, _, _ = layer(inp)

    np.testing.assert_equal(
        y1,
        np.array(
            [[[7, 5, 2, 8, 8, 8, 6, 7],
              [8, 2, 6, 2, 1, 1, 4, 2],
              [4, 7, 7, 4, 8, 9, 9, 9],
              [6, 8, 2, 9, 3, 6, 6, 8],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],

             [[4, 3, 1, 7, 5, 6, 2, 1],
              [6, 9, 9, 4, 1, 3, 2, 1],
              [3, 8, 2, 4, 7, 9, 4, 1],
              [3, 7, 5, 6, 2, 9, 3, 1],
              [4, 7, 3, 2, 1, 1, 1, 6],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        )
    )
    np.testing.assert_equal(
        y2,
        np.array(
            [[[7, 5, 2, 8, 8, 8, 6, 7],
              [8, 2, 6, 2, 1, 1, 4, 2],
              [4, 7, 7, 4, 8, 9, 9, 9],
              [6, 8, 2, 9, 3, 6, 6, 8],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],

             [[4, 3, 1, 7, 5, 6, 2, 1],
              [6, 9, 9, 4, 1, 3, 2, 1],
              [3, 8, 2, 4, 7, 9, 4, 1],
              [3, 7, 5, 6, 2, 9, 3, 1],
              [4, 7, 3, 2, 1, 1, 1, 6],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        )
    )

  def test_strip_from_concatenate_with_padding(self):
    enc_dec = np.array(
        [[[7, 5, 2, 8, 8, 8, 6, 7],
          [8, 2, 6, 2, 1, 1, 4, 2],
          [4, 7, 7, 4, 8, 9, 9, 9],
          [6, 8, 2, 9, 3, 6, 6, 8],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[4, 3, 1, 7, 5, 6, 2, 1],
          [6, 9, 9, 4, 1, 3, 2, 1],
          [3, 8, 2, 4, 7, 9, 4, 1],
          [3, 7, 5, 6, 2, 9, 3, 1],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    tok_e = np.array([[7, 8, 0, 0, 0, 0], [4, 6, 3, 0, 0, 0]])
    tok_d = np.array([[4, 6, 0, 0], [3, 4, 1, 0]])

    layer = transformer2.StripFromConcatenateWithPadding(
        mode='train')
    inp = (enc_dec, tok_e, tok_d)
    _, _ = layer.init(shapes.signature(inp))
    y = layer(inp)

    np.testing.assert_equal(
        y,
        np.array([[[4, 7, 7, 4, 8, 9, 9, 9],
                   [6, 8, 2, 9, 3, 6, 6, 8],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]],
                  [[3, 7, 5, 6, 2, 9, 3, 1],
                   [4, 7, 3, 2, 1, 1, 1, 6],
                   [4, 7, 3, 2, 1, 1, 1, 6],
                   [0, 0, 0, 0, 0, 0, 0, 0]]]))

  def test_strip_from_concatenate_with_padding_predict(self):
    enc_dec = np.array(
        [[[7, 5, 2, 8, 8, 8, 6, 7],
          [8, 2, 6, 2, 1, 1, 4, 2],
          [4, 7, 7, 4, 8, 9, 9, 9],
          [6, 8, 2, 9, 3, 6, 6, 8],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]],

         [[4, 3, 1, 7, 5, 6, 2, 1],
          [6, 9, 9, 4, 1, 3, 2, 1],
          [3, 8, 2, 4, 7, 9, 4, 1],
          [3, 7, 5, 6, 2, 9, 3, 1],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [4, 7, 3, 2, 1, 1, 1, 6],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]]
    )

    tok_e = np.array([[7, 8, 0, 0, 0, 0], [4, 6, 3, 0, 0, 0]])
    tok_d = np.array([[4, 6, 0, 0], [3, 4, 1, 0]])

    layer = transformer2.StripFromConcatenateWithPadding(
        mode='predict')
    inp = (enc_dec, tok_e, tok_d)
    _, _ = layer.init(shapes.signature(inp))
    y = layer(inp)

    np.testing.assert_equal(
        y,
        np.array([[[4, 7, 7, 4, 8, 9, 9, 9],
                   [6, 8, 2, 9, 3, 6, 6, 8],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]],
                  [[3, 7, 5, 6, 2, 9, 3, 1],
                   [4, 7, 3, 2, 1, 1, 1, 6],
                   [4, 7, 3, 2, 1, 1, 1, 6],
                   [0, 0, 0, 0, 0, 0, 0, 0]]]))

    # On subsequent runs however, we should get enc_dec only.
    for _ in range(2):
      y = layer(inp)
      np.testing.assert_equal(y, enc_dec)

  def test_transformer_noencdec_forward_shape(self):
    input_vocab_size = 16
    output_vocab_size = 16

    model = transformer2.Transformer2(
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
