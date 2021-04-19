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
"""Tests for Funnel-Transformer models."""

from absl.testing import absltest
from absl.testing import parameterized
import gin
import jax
import numpy as np
from trax import fastmath
from trax import layers as tl
from trax import shapes
import trax.models.research.funnel_transformer as ft
from trax.supervised import decoding


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

  def test_funnel_transformer_lm_forward_shape(self):
    d_model = 16
    vocab_size = 7
    length = 48
    batch_size = 3
    x = np.ones((batch_size, length)).astype(np.int32)

    model_chunked = ft.RelformerLM(
        vocab_size,
        shorten_factor=3,
        n_rel_layers=3,
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
        vanilla_attn_type=tl.SelfAttention,
        rel_chunk_len=4,
        vanilla_chunk_len=2,
        max_len=48)
    _, _ = model_chunked.init(shapes.signature(x))
    y = model_chunked(x)
    self.assertEqual(y.shape, (batch_size, length, vocab_size))

    model_without_chunks = ft.RelformerLM(
        vocab_size,
        shorten_factor=3,
        n_rel_layers=3,
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
        vanilla_attn_type=tl.SelfAttention,
        max_len=48)

    _, _ = model_without_chunks.init(shapes.signature(x))
    y = model_without_chunks(x)
    self.assertEqual(y.shape, (batch_size, length, vocab_size))

  def test_funnel_transformer_lm_autoregressive_property(self):
    input_shape = (1, 12)
    d_model = 8
    vocab_size = 26
    rng_1 = jax.random.PRNGKey(0)
    rng_2 = jax.random.PRNGKey(1)

    def _get_output_logits(unitialized_eval_model: tl.Layer, x):
      input_signature = shapes.signature(x)
      unitialized_eval_model.init(input_signature, rng=rng_1, use_cache=False)

      output_logits, *_ = unitialized_eval_model(x, rng=rng_1)
      return output_logits

    def test_autoregressive_property(model):
      with fastmath.use_backend(fastmath.Backend.JAX):
        x_1 = jax.random.randint(rng_1, input_shape, 0, vocab_size)
        y_1 = _get_output_logits(model, x_1)

        x_2 = jax.random.randint(rng_2, input_shape, 0, vocab_size)

        for i in range(input_shape[1]):
          masked_x_2 = np.concatenate((x_1[:, :i], x_2[:, i:]), axis=1)

          y_2 = _get_output_logits(model, masked_x_2)
          self.assertEqual(y_2.shape[0], input_shape[1])
          np.testing.assert_array_almost_equal(y_1[:i + 1], y_2[:i + 1])

    model_chunked = ft.RelformerLM(
        vocab_size,
        shorten_factor=3,
        n_rel_layers=2,
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=2,
        vanilla_attn_type=tl.SelfAttention,
        rel_chunk_len=2,
        vanilla_chunk_len=4,
    )
    test_autoregressive_property(model_chunked)

    model_without_chunks = ft.RelformerLM(
        vocab_size,
        shorten_factor=3,
        n_rel_layers=2,
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=2,
        vanilla_attn_type=tl.SelfAttention,
        rel_chunk_len=None,
        vanilla_chunk_len=None,
    )
    test_autoregressive_property(model_without_chunks)

  def test_funnel_transformer_lm_forward_shape_predict(self):
    d_model = 8
    vocab_size = 4
    batch_size = 1
    n_len_eval = 42
    attention_type = tl.SelfAttention

    shorten_factor = 3
    n_rel_layers = 2
    vanilla_layers = (1, 1)
    n_heads = 2

    rel_chunk_len, vanilla_chunk_len = 2, 6

    x = np.ones((batch_size, 1)).astype(np.int32)
    gin.bind_parameter('trax.layers.SelfAttention.chunk_len', 20)

    predict_funnel = ft.RelformerLM(
        vocab_size,
        shorten_factor=shorten_factor,
        n_rel_layers=n_rel_layers,
        vanilla_layers=vanilla_layers,
        d_model=d_model,
        d_ff=d_model,
        n_heads=n_heads,
        vanilla_attn_type=attention_type,
        rel_chunk_len=rel_chunk_len,
        vanilla_chunk_len=vanilla_chunk_len,
        max_len=n_len_eval,
        mode='predict')

    _, _ = predict_funnel.init(shapes.signature(x))

    for _ in range(5):
      y = predict_funnel(x)
      self.assertEqual(y.shape, (batch_size, 1, vocab_size))
    gin.clear_config()

  def test_funnel_transformer_lm_predict_eval_equal(self):

    def _test_for_chunk_lens(rel_chunk_len, vanilla_chunk_len):
      d_model = 8
      vocab_size = 4
      batch_size = 1
      n_len_eval = 42
      attention_type = tl.SelfAttention

      shorten_factor = 3
      n_rel_layers = 2
      vanilla_layers = (1, 1)
      n_heads = 2

      eval_funnel = ft.RelformerLM(
          vocab_size,
          shorten_factor=shorten_factor,
          n_rel_layers=n_rel_layers,
          vanilla_layers=vanilla_layers,
          d_model=d_model,
          d_ff=d_model,
          n_heads=n_heads,
          vanilla_attn_type=attention_type,
          rel_chunk_len=rel_chunk_len,
          vanilla_chunk_len=vanilla_chunk_len,
          mode='eval')

      inputs = jax.random.randint(
          key=jax.random.PRNGKey(0),
          minval=0,
          maxval=vocab_size,
          shape=(batch_size, n_len_eval)).astype(np.int32)
      _, _ = eval_funnel.init(
          shapes.signature(inputs), rng=jax.random.PRNGKey(0))
      y_eval = eval_funnel(inputs)
      self.assertEqual(y_eval.shape, (batch_size, n_len_eval, vocab_size))

      if attention_type == tl.SelfAttention:
        gin.bind_parameter('trax.layers.SelfAttention.chunk_len', n_len_eval)

      predict_funnel = ft.RelformerLM(
          vocab_size,
          shorten_factor=shorten_factor,
          n_rel_layers=n_rel_layers,
          vanilla_layers=vanilla_layers,
          d_model=d_model,
          d_ff=d_model,
          n_heads=n_heads,
          vanilla_attn_type=attention_type,
          rel_chunk_len=rel_chunk_len,
          vanilla_chunk_len=vanilla_chunk_len,
          mode='predict')

      inputs = np.concatenate(
          [np.zeros((batch_size, 1)).astype(np.int32), inputs], axis=1)
      inputs = inputs[:, :-1]

      _, _ = predict_funnel.init(
          shapes.signature(inputs[:, 0:1]),
          rng=jax.random.PRNGKey(0),
          use_cache=False)

      for i in range(n_len_eval):
        y = predict_funnel(inputs[:, i:i + 1])
        np.testing.assert_array_almost_equal(
            y, y_eval[:, i:i + 1, :], decimal=5)

    _test_for_chunk_lens(rel_chunk_len=None, vanilla_chunk_len=None)
    _test_for_chunk_lens(rel_chunk_len=2, vanilla_chunk_len=6)

  def test_autoregressive_sample_relformerlm(self):
    batch_size = 4
    max_length = 5
    model = ft.RelformerLM(
        10,
        d_model=8,
        d_ff=16,
        n_rel_layers=1,
        vanilla_layers=(1, 1),
        shorten_factor=3,
        n_heads=2,
        mode='predict')
    model.init(shapes.ShapeDtype((batch_size, 1), dtype=np.int32))
    s1 = decoding.autoregressive_sample(
        model,
        batch_size=batch_size,
        eos_id=-1,
        max_length=max_length,
        accelerate=False)
    self.assertEqual(s1.shape, (batch_size, max_length))


if __name__ == '__main__':
  absltest.main()
