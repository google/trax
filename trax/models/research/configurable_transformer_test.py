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
from trax import layers as tl
from trax import shapes
from trax.models.research import configurable_transformer as ct


class ConfigurableTransformerTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    vocab_size = 16
    model = ct.ConfigurableTransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    x = np.ones((3, 5)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, vocab_size))

  def _test_transformer_forward_shape(self, input_vocab_size,
                                      output_vocab_size):
    model = ct.ConfigurableTransformer(
        input_vocab_size,
        output_vocab_size,
        d_model=32,
        d_ff=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=2)
    xs = [np.ones((3, 5)).astype(np.int32), np.ones((3, 5)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    y, _ = model(xs)

    vocab_size = output_vocab_size or input_vocab_size
    self.assertEqual(y.shape, (3, 5, vocab_size))

  @parameterized.named_parameters(('same_vocab', 16, None),
                                  ('same_size', 16, 16),
                                  ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


  def _test_fast_inference(self, length):
    with fastmath.use_backend(fastmath.Backend.JAX):
      vocab_size = 16
      model_fn = functools.partial(
          ct.ConfigurableTransformerLM,
          vocab_size=vocab_size,
          d_model=4,
          d_ff=8,
          n_layers=2,
          n_heads=2,
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
            logits_slow[:, index, :],
            logits_fast[:, 0, :],
            decimal=5,
        )
        next_sym = np.random.randint(vocab_size, size=(batch_size, 1))
        buf[:, index] = next_sym[:, 0]

  def test_dot_product_causal_attention_fast_inference(self):
    self._test_fast_inference(length=5)

  @parameterized.named_parameters(
      ('positional_encoding', None),
      ('fixed_base_positional_encoding', 'fixed-base'),
      ('infinite_positional_encoding', 'infinite'),
      ('infinite_affine_positional_encoding', 'infinite-affine'),
      ('axial_positional_encoding', (2, 16)))
  def test_positional_encoder(self, axial_pos_shape):
    # dim should divide FixedBasePositionalEncoding.n_digits
    batch, length, dim = 2, 32, 8
    input_shape = (batch, length, dim)
    vocab_size = 32
    x = np.random.randint(0, vocab_size - 1, input_shape)
    # should sum to dim
    d_axial_pos_embs = (4, 4)

    positional_encoding = ct.PositionalEncoder(
        'train', dropout=0.1, max_len=length, axial_pos_shape=axial_pos_shape,
        d_axial_pos_embs=d_axial_pos_embs)
    _, _ = positional_encoding.init(shapes.signature(x))
    y = positional_encoding(x)
    self.assertEqual(y.shape, input_shape)

  def test_embedding_and_positional_encodings(self):
    d_model = 16
    max_len = 32
    batch = 2
    input_shape = (batch, max_len)
    input_vocab_size = 32
    x = np.random.randint(0, input_vocab_size - 1, input_shape)

    in_encoder, out_encoder, output_vocab_size = (
        ct.EmbeddingAndPositionalEncodings(
            input_vocab_size,
            d_model,
            'train',
            0.1,
            [-2],
            max_len,
            output_vocab_size=None,
            axial_pos_shape=None,
            d_axial_pos_embs=None))

    self.assertEqual(output_vocab_size, input_vocab_size)

    model_in = tl.Serial(in_encoder)
    model_out = tl.Serial(out_encoder)

    model_in.init(shapes.signature(x))
    model_out.init(shapes.signature(x))

    y = model_in(x)
    self.assertEqual(y.shape, input_shape + (d_model,))

    y = model_in(x)
    self.assertEqual(y.shape, input_shape + (d_model,))


if __name__ == '__main__':
  absltest.main()
