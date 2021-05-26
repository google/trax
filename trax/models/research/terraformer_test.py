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
"""Tests for Terraformer models."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import gin
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.layers import test_utils
from trax.models.research import terraformer


BACKENDS = [fastmath.Backend.JAX]


def short_name(b):
  if b == fastmath.Backend.JAX:
    return 'jax'
  else:
    return 'tf'


class TerraformerTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      [('_%s_efficient' % short_name(backend), backend, tl.SelfAttention, False)
       for backend in BACKENDS] +
      [('_%s_causal' % short_name(backend), backend, tl.CausalAttention, False)
       for backend in BACKENDS] +
      # NOTE: tl.SelfAttention is not currently working for this case.
      [('_%s_preembed' % short_name(backend), backend, tl.CausalAttention, True)
       for backend in BACKENDS])
  def test_terraformer_quick(self, backend, encoder_attention_type, preembed):
    with fastmath.use_backend(backend):
      vocab_size = 2
      input_vocab_size = None if preembed else vocab_size
      output_vocab_size = vocab_size if preembed else None
      max_len = 2

      model = terraformer.ConfigurableTerraformer(
          input_vocab_size,
          d_model=4,
          d_ff=4,
          n_encoder_layers=1,
          n_decoder_layers=1,
          n_heads=2,
          dropout=0.05,
          max_len=max_len,
          pos_type=None,
          ff_activation=tl.Relu,
          ff_use_sru=0,
          ff_chunk_size=2,
          mode='train',
          output_vocab_size=output_vocab_size,
          encoder_attention_type=encoder_attention_type,
      )

      if preembed:
        model_inputs = [np.ones((1, max_len, 3)).astype(np.float32),
                        np.ones((1, max_len)).astype(np.bool)]
      else:
        model_inputs = [np.ones((1, max_len)).astype(np.int32)]
      x = model_inputs + [np.ones((1, max_len)).astype(np.int32)]
      model.init(shapes.signature(x))

      logits, dec_toks = model(x)
      del dec_toks

      self.assertEqual(logits.shape, (1, max_len, vocab_size))

  def test_terraformer_deterministic_eval(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      vocab_size = 16
      d_model = 4
      batch_size = 2
      length = 5

      model_fn = functools.partial(
          terraformer.ConfigurableTerraformer,
          vocab_size,
          d_model=d_model,
          d_ff=16,
          n_encoder_layers=0,
          n_decoder_layers=1,
          n_heads=2,
          dropout=0.0,
          max_len=length*2,
          pos_type=None,
          encoder_attention_type=tl.Attention,
          encoder_decoder_attention_type=tl.CausalAttention,
      )

      inp = np.random.randint(vocab_size, size=(batch_size, length))
      out = np.zeros((batch_size, length), dtype=np.int32)

      test_utils.test_eval_is_deterministic((inp, out), model_fn)

  def test_terraformer_predict_equals_eval(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      vocab_size = 16
      d_model = 8
      batch_size = 1
      length = 5

      model_fn = functools.partial(
          terraformer.ConfigurableTerraformer,
          vocab_size,
          d_model=d_model,
          d_ff=16,
          n_encoder_layers=1,
          n_decoder_layers=1,
          n_heads=2,
          ff_use_sru=(1, 8),  # ? is SRU working?
          dropout=0.0,
          max_len=(length+7)*2,
          pos_type=None,
          reversible_encoder=True,
          n_decoder_attention_layers=1,
          encoder_attention_type=tl.Attention,
          encoder_decoder_attention_type=tl.CausalAttention,
      )

      # Token id of 0 indicates padding; and predict mode doesn't support it.
      inp = np.random.randint(1, vocab_size, size=(batch_size, length))
      inp[:, -2:] = 0
      out = np.zeros((batch_size, length), dtype=np.int32)

      test_utils.test_eval_equals_predict(
          (inp, out), model_fn, seq_axis=1, seq_tensor=-1, init_tokens=1)

  def test_terraformer_doubling(self):
    vocab_size = 2
    max_len = 2

    model = terraformer.ConfigurableTerraformer(
        vocab_size,
        d_model=8,
        d_ff=16,
        n_encoder_layers=1,
        n_decoder_layers=6,
        n_heads=2,
        dropout=0.05,
        max_len=max_len,
        pos_type=None,
        half_before_layer=2,
        double_after_layer=2,
        encoder_attention_type=tl.Attention,
        encoder_decoder_attention_type=tl.CausalAttention,
        mode='train',
    )

    x = [np.ones((1, max_len)).astype(np.int32),
         np.ones((1, max_len)).astype(np.int32)]
    model.init(shapes.signature(x))

    logits, dec_toks = model(x)
    del dec_toks

    self.assertEqual(logits.shape, (1, max_len, vocab_size))

  def test_terraformer_one_step(self):
    vocab_size = 32
    max_len = 256
    pos_axial = 16
    assert pos_axial * pos_axial == max_len

    chunk_len = 32

    # Since 2 * chunk_len * n_buckets should be max_len.
    n_buckets = max_len // (2 * chunk_len)

    lsh_self_attention = functools.partial(self._lsh_self_attention_fn(),
                                           chunk_len=chunk_len,
                                           n_buckets=n_buckets)

    timebin_self_attention = self._timebin_self_attention_fn()

    model = terraformer.ConfigurableTerraformer(
        vocab_size,
        d_model=32,
        d_ff=64,
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
        pos_axial_shape=(pos_axial, pos_axial),
        pos_d_axial_embs=(64, 192),
        ff_activation=tl.Relu,
        ff_use_sru=0,
        ff_chunk_size=64,
        ff_sparsity=8,
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
