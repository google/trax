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
"""Tests for Reformer models."""

import functools

from absl.testing import absltest
import gin
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.models.reformer import reformer


class ReformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def _lsh_self_attention_fn(self):
    return functools.partial(
        tl.LSHSelfAttention,
        attention_dropout=0.0,
        chunk_len=64,
        n_buckets=[32, 32],
        n_chunks_after=0,
        n_chunks_before=1,
        n_hashes=1,
        n_parallel_heads=1,
        predict_drop_len=128,
        predict_mem_len=1024,
    )

  def _timebin_self_attention_fn(self, use_reference_code=False):
    return functools.partial(
        tl.SelfAttention,
        attention_dropout=0.05,
        chunk_len=64,
        n_chunks_before=1,
        n_parallel_heads=1,
        use_reference_code=use_reference_code
    )

  def test_reformer_lm_forward_shape(self):
    vocab_size = 16
    model = reformer.ReformerLM(
        vocab_size, d_model=32, d_ff=64, d_attention_key=16,
        d_attention_value=16, n_layers=1, n_heads=2, max_len=16)
    xs = [np.ones((1, 8)).astype(np.int32),
          np.ones((1, 8)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    ys = model(xs)
    self.assertEqual([y.shape for y in ys], [(1, 8, 16), (1, 8)])


  def test_reformer_lm_lsh(self):
    lsh_self_attention = self._lsh_self_attention_fn()
    timebin_self_attention = self._timebin_self_attention_fn()

    model = reformer.ReformerLM(
        vocab_size=256,
        d_model=256,
        d_ff=512,
        d_attention_key=64,
        d_attention_value=64,
        n_layers=2,
        n_heads=2,
        dropout=0.05,
        max_len=65536,
        attention_type=[timebin_self_attention, lsh_self_attention],
        axial_pos_shape=(256, 256),
        d_axial_pos_embs=(64, 192),
        ff_activation=tl.Relu,
        ff_use_sru=0,
        ff_chunk_size=8192,
        mode='train',
    )
    x = np.ones((1, 65536)).astype(np.int32)
    weights, state = model.init(shapes.signature(x))

    @fastmath.jit
    def mock_training_step(x, weights, state, rng):
      def compute_mock_loss(weights):
        logits, new_state = model.pure_fn(x, weights, state, rng)
        loss = fastmath.numpy.mean(logits[..., 0])
        return loss, (new_state, logits)
      gradients, (new_state, logits) = fastmath.grad(
          compute_mock_loss, has_aux=True)(weights)
      new_weights = fastmath.nested_map_multiarg(
          lambda w, g: w - 1e-4 * g, weights, gradients)
      return new_weights, new_state, logits

    weights, state, logits = mock_training_step(
        x, weights, state, fastmath.random.get_prng(0))
    self.assertEqual(logits.shape, (1, 65536, 256))

  def test_reformer_no_enc_dec_attention_one_step(self):
    vocab_size = 256
    max_len = 65536
    axial_pos = 256
    assert axial_pos * axial_pos == max_len

    chunk_len = 64

    # Since 2 * chunk_len * n_buckets should be max_len.
    n_buckets = max_len // (2 * chunk_len)

    lsh_self_attention = functools.partial(self._lsh_self_attention_fn(),
                                           chunk_len=chunk_len,
                                           n_buckets=n_buckets)

    timebin_self_attention = self._timebin_self_attention_fn()

    model = reformer.ReformerNoEncDecAttention(
        vocab_size,
        d_model=256,
        d_ff=512,
        d_attention_key=64,
        d_attention_value=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=2,
        dropout=0.05,
        max_len=max_len,
        encoder_attention_type=lsh_self_attention,
        encoder_decoder_attention_type=[timebin_self_attention,
                                        lsh_self_attention],
        axial_pos_shape=(axial_pos, axial_pos),
        d_axial_pos_embs=(64, 192),
        ff_activation=tl.Relu,
        ff_use_sru=0,
        ff_chunk_size=8192,
        mode='train',
    )

    x = [np.ones((1, max_len)).astype(np.int32),
         np.ones((1, max_len)).astype(np.int32)]
    weights, state = model.init(shapes.signature(x))

    @fastmath.jit
    def mock_training_step(x, weights, state, rng):
      def compute_mock_loss(weights):
        logits_and_dec_toks, new_state = model.pure_fn(x, weights, state, rng)
        # This returns [logits, decoder tokens]
        logits = logits_and_dec_toks[0]
        loss = fastmath.numpy.mean(logits[..., 0])
        return loss, (new_state, logits)
      gradients, (new_state, logits) = fastmath.grad(
          compute_mock_loss, has_aux=True)(weights)
      new_weights = fastmath.nested_map_multiarg(
          lambda w, g: w - 1e-4 * g, weights, gradients)
      return new_weights, new_state, logits

    weights, state, logits = mock_training_step(
        x, weights, state, fastmath.random.get_prng(0))

    self.assertEqual(logits.shape, (1, max_len, vocab_size))


if __name__ == '__main__':
  absltest.main()
