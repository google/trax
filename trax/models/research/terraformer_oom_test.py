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
"""Tests for OOM for Terraformer ."""

import functools
import operator

from absl.testing import absltest
import gin
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.models.research import terraformer


class TerraformerOOMTest(absltest.TestCase):

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

  def test_terraformer_one_step(self):
    d_model = 1024
    vocab_size = 14041
    max_len = 16384
    pos_axial = (128, 128)  # should multiply to max_len
    pos_d_axial_embs = (512, 512)  # sum to d model

    assert operator.mul(*pos_axial) == max_len
    assert sum(pos_d_axial_embs) == d_model

    d_ff = 4096
    n_heads = 8
    d_attn = d_model // n_heads

    n_buckets = 128
    encoder_chunk_len = (2 * max_len) // n_buckets  # 256
    decoder_chunk_len = 2 * encoder_chunk_len       # 512
    encoder_n_chunks_after = 1                      # since its not causal.

    lsh_self_attention = functools.partial(self._lsh_self_attention_fn(),
                                           n_buckets=n_buckets)

    encoder_lsh_self_attention = functools.partial(
        lsh_self_attention, n_chunks_after=encoder_n_chunks_after,
        chunk_len=encoder_chunk_len)

    decoder_lsh_self_attention = functools.partial(
        lsh_self_attention, n_chunks_after=0,
        chunk_len=decoder_chunk_len)

    model = terraformer.ConfigurableTerraformer(
        vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        d_attention_key=d_attn,
        d_attention_value=d_attn,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=n_heads,
        dropout=0.05,
        max_len=max_len,
        encoder_attention_type=encoder_lsh_self_attention,
        encoder_decoder_attention_type=decoder_lsh_self_attention,
        pos_axial_shape=pos_axial,
        pos_d_axial_embs=pos_d_axial_embs,
        ff_activation=tl.Relu,
        ff_use_sru=0,
        mode='train',
    )

    def random_sentence():
      return np.random.randint(low=1, high=vocab_size - 1, size=(1, max_len),
                               dtype=np.int32)

    x = [random_sentence(), random_sentence()]
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
