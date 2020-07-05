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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import fastmath
from trax import shapes
from trax.models import transformer


class TransformerTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    vocab_size = 16
    model = transformer.TransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    x = np.ones((3, 5)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, vocab_size))

  def _test_transformer_forward_shape(self, input_vocab_size,
                                      output_vocab_size):
    model = transformer.Transformer(
        input_vocab_size, output_vocab_size, d_model=32, d_ff=64,
        n_encoder_layers=2, n_decoder_layers=2, n_heads=2)
    xs = [np.ones((3, 5)).astype(np.int32), np.ones((3, 5)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    y, _ = model(xs)

    vocab_size = output_vocab_size or input_vocab_size
    self.assertEqual(y.shape, (3, 5, vocab_size))

  def test_transformer_noencdec_forward_shape(self):
    input_vocab_size = 16
    output_vocab_size = 16

    model = transformer.TransformerNoEncDecAttention(
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

  @parameterized.named_parameters(
      ('same_vocab', 16, None),
      ('same_size', 16, 16),
      ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


  def _test_fast_inference(self, length):
    with fastmath.use_backend('jax'):
      vocab_size = 16
      model_fn = functools.partial(
          transformer.TransformerLM,
          vocab_size=vocab_size, d_model=4, d_ff=8, n_layers=2, n_heads=2,
      )
      model_slow = model_fn(mode='eval')
      model_fast = model_fn(mode='predict')
      rng = fastmath.random.get_prng(0)
      batch_size = 2
      input_signature = shapes.ShapeDtype((batch_size, 1), np.int32)
      # Given the same rng, both models initialize with the same parameters.
      model_slow.init(input_signature, rng)
      model_fast.init(input_signature, rng)

      buf = np.zeros((batch_size, length), dtype=np.int32)
      next_sym = np.zeros((batch_size, 1), dtype=np.int32)

      for index in range(length):
        logits_slow = model_slow(buf, rng=rng)
        logits_fast = model_fast(next_sym, rng=rng)
        np.testing.assert_array_almost_equal(
            logits_slow[:, index, :], logits_fast[:, 0, :],
            decimal=5,
        )
        next_sym = np.random.randint(vocab_size, size=(batch_size, 1))
        buf[:, index] = next_sym[:, 0]

  def test_dot_product_causal_attention_fast_inference(self):
    self._test_fast_inference(length=5)

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

    mask_d = np.array(
        [[True, True, False, False],
         [True, True, False, False]])

    layer = transformer._ConcatWithPadding()
    y = layer((vec_e, vec_d, mask_e, mask_d))

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

    layer = transformer._StripFromConcatenateWithPadding()
    y = layer((enc_dec, tok_e, tok_d))

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

  def test_mask_of_right_shift_unshifted(self):
    layer = transformer._MaskOfRightShiftedArray(n_shifts=0)
    x = np.array(
        [[9, 8, 7, 0],
         [1, 2, 0, 0]]
        )
    y = layer(x)
    np.testing.assert_equal(
        y,
        np.array([[True, True, True, False],
                  [True, True, False, False]]))

  def test_mask_of_right_shift(self):
    layer = transformer._MaskOfRightShiftedArray(n_shifts=2)
    x = np.array(
        [[0, 0, 9, 8, 7, 0],
         [0, 0, 1, 2, 0, 0]]
        )
    y = layer(x)
    np.testing.assert_equal(
        y,
        np.array([[True, True, True, True, True, False],
                  [True, True, True, True, False, False]]))

  def test_mask_of_right_shift_3dims(self):
    layer = transformer._MaskOfRightShiftedArray(n_shifts=2)

    # pylint: disable=bad-whitespace
    x = np.array(
        [[[ 0,  0],
          [ 0,  0],
          [ 1,  2],
          [ 3,  4],
          [ 5,  6],
          [ 7,  8],
          [ 9, 10],
          [ 0,  0],
          [ 0,  0]],

         [[ 0,  0],
          [ 0,  0],
          [11, 12],
          [13, 14],
          [15, 16],
          [17, 18],
          [19, 20],
          [ 0,  0],
          [ 0,  0]]]
    )
    # pylint: enable=bad-whitespace

    y = layer(x)
    np.testing.assert_equal(
        y,
        # pylint: disable=bad-whitespace
        np.array(
            [[[ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ False, False],
              [ False, False]],

             [[ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ True,  True],
              [ False, False],
              [ False, False]]]
        )
        # pylint: enable=bad-whitespace
    )


if __name__ == '__main__':
  absltest.main()
