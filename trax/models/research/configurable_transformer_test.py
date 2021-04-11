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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.layers import test_utils
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


  def test_dot_product_causal_attention_fast_inference(self):
    self._test_fast_inference(length=5)

  def _test_fast_inference(self, length):
    with fastmath.use_backend(fastmath.Backend.JAX):
      model_fn = functools.partial(
          ct.ConfigurableTransformerLM,
          vocab_size=16,
          d_model=4,
          d_ff=8,
          n_layers=2,
          n_heads=2,
      )
      batch_size = 2
      inp = np.zeros((batch_size, length), dtype=np.int32)

      test_utils.test_eval_equals_predict(inp, model_fn)

  def test_sparse_configurable_transformer_fast_inference(self):
    self._test_sparse_fast_inference(length=5)

  def _test_sparse_fast_inference(self, length):
    with fastmath.use_backend(fastmath.Backend.JAX):
      vocab_size = 16
      d_model = 4
      batch_size = 2

      encoder_decoder_attention_type = functools.partial(
          tl.MultiplicativeConvCausalAttention,
          sparsity=2,
          length_kernel_size=1,
          )

      model_fn = functools.partial(
          ct.ConfigurableTransformer,
          input_vocab_size=vocab_size,
          d_model=d_model,
          d_ff=8,
          n_encoder_layers=2,
          n_decoder_layers=2,
          n_heads=2,
          loss_sparsity=2,
          ff_sparsity=2,
          encoder_decoder_attention_type=encoder_decoder_attention_type,
          ff_use_sru=(1, 4),
      )

      inp = np.random.randint(vocab_size, size=(batch_size, length))
      out = np.zeros((batch_size, length), dtype=np.int32)

      test_utils.test_eval_equals_predict((inp, out), model_fn, seq_tensor=1)

  @parameterized.named_parameters(
      ('positional_encoding', None),
      ('fixed_base_positional_encoding', 'fixed-base'),
      ('infinite_positional_encoding', 'infinite'),
      ('infinite_affine_positional_encoding', 'infinite-affine'),
      ('axial_positional_encoding', (2, 16)))
  def test_positional_encoder(self, pos_axial_shape):
    # dim should divide FixedBasePositionalEncoding.n_digits
    batch, length, dim = 2, 32, 8
    input_shape = (batch, length, dim)
    vocab_size = 32
    x = np.random.randint(0, vocab_size - 1, input_shape)
    # should sum to dim
    pos_d_axial_embs = (4, 4)

    positional_encoding = ct.PositionalEncoder(
        'train', dropout=0.1, max_len=length, pos_axial_shape=pos_axial_shape,
        pos_d_axial_embs=pos_d_axial_embs)
    _, _ = positional_encoding.init(shapes.signature(x))
    y = positional_encoding(x)
    self.assertEqual(y.shape, input_shape)

  @parameterized.named_parameters(
      ('input_vocab_size_only', 32, None),
      ('output_vocab_size_only', None, 32),
      ('same_input_output_vocab_size', 32, 32),
      ('different_input_output_vocab_size', 32, 16),
  )
  def test_embedding_and_positional_encodings(self, input_vocab_size,
                                              output_vocab_size):
    d_model = 16
    max_len = 32
    batch = 2
    input_shape = (batch, max_len)
    output_vocab_size_expected = output_vocab_size or input_vocab_size
    x_out = np.random.randint(0, output_vocab_size_expected - 1, input_shape)
    if input_vocab_size is None:
      x_in = np.random.uniform(size=list(input_shape) + [2])
    else:
      x_in = np.random.randint(0, input_vocab_size - 1, input_shape)

    in_encoder, out_encoder, output_vocab_size_result = (
        ct.EmbeddingAndPositionalEncodings(
            input_vocab_size,
            d_model,
            'train',
            0.1,
            [-2],
            max_len,
            output_vocab_size=output_vocab_size,
            pos_axial_shape=None,
            pos_d_axial_embs=None))

    self.assertEqual(output_vocab_size_result, output_vocab_size_expected)

    model_in = tl.Serial(in_encoder)
    model_out = tl.Serial(out_encoder)

    model_in.init(shapes.signature(x_in))
    model_out.init(shapes.signature(x_out))

    y = model_in(x_in)
    self.assertEqual(y.shape, input_shape + (d_model,))

    y = model_out(x_out)
    self.assertEqual(y.shape, input_shape + (d_model,))


if __name__ == '__main__':
  absltest.main()
