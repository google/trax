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
"""Tests for Hourglass model."""

from absl.testing import absltest
from absl.testing import parameterized
import gin
import jax
import numpy as np
from trax import fastmath
from trax import layers as tl
from trax import shapes
import trax.layers.research.resampling as resampling
import trax.models.research.hourglass as hourglass


class HourglassTest(parameterized.TestCase):

  def _check_forward_shape(self, model, input_shape, output_vocab_size):
    x = np.ones(input_shape).astype(np.int32)
    model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (*input_shape, output_vocab_size))

  def test_hourglass_lm_forward_shape(self):
    d_model = 16
    vocab_size = 7
    model = hourglass.HourglassLM(
        vocab_size,
        hierarchy='2@3 2@6 2@3',
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
    )

    batch_size, seq_len = 3, 24
    self._check_forward_shape(model,
                              input_shape=(batch_size, seq_len),
                              output_vocab_size=vocab_size)

  def test_lsh_attention_in_vanilla(self):
    d_model = 16
    vocab_size = 7

    gin.bind_parameter('PureLSHSelfAttentionWrapper.pure_lsh_implementation',
                       tl.PureLSHSelfAttention)
    gin.bind_parameter('PureLSHSelfAttention.chunk_len', 2)

    model = hourglass.HourglassLM(
        vocab_size,
        hierarchy='2@3',
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
        vanilla_attn_type=tl.PureLSHSelfAttentionWrapper,
        downsampling_fn=resampling.LinearPooling,
        upsampling_fn=resampling.LinearUpsampling,
    )

    batch_size, seq_len = 3, 12
    self._check_forward_shape(
        model, input_shape=(batch_size, seq_len), output_vocab_size=vocab_size)

  def _test_autoregressive_property(self, model, input_shape,
                                    output_vocab_size):
    rng_1 = jax.random.PRNGKey(0)
    rng_2 = jax.random.PRNGKey(1)

    def _get_output_logits(unitialized_eval_model: tl.Layer, x):
      input_signature = shapes.signature(x)
      unitialized_eval_model.init(input_signature, rng=rng_1, use_cache=False)

      output_logits, *_ = unitialized_eval_model(x, rng=rng_1)
      return output_logits

    def check_autoregressive_property(model):
      with fastmath.use_backend(fastmath.Backend.JAX):
        x_1 = jax.random.randint(rng_1, input_shape, 0, output_vocab_size)
        y_1 = _get_output_logits(model, x_1)

        x_2 = jax.random.randint(rng_2, input_shape, 0, output_vocab_size)

        for i in range(input_shape[1]):
          masked_x_2 = np.concatenate((x_1[:, :i], x_2[:, i:]), axis=1)

          y_2 = _get_output_logits(model, masked_x_2)
          self.assertEqual(y_2.shape[0], input_shape[1])
          np.testing.assert_array_almost_equal(y_1[:i + 1], y_2[:i + 1])

    check_autoregressive_property(model)

  def test_hourglass_lm_autoregressive_property(self):
    d_model = 8
    vocab_size = 26

    model_single_stage = hourglass.HourglassLM(
        vocab_size,
        hierarchy='2@4',
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
    )

    model_multi_stage = hourglass.HourglassLM(
        vocab_size,
        hierarchy='2@3 2@6 2@3',
        vanilla_layers=(1, 1),
        d_model=d_model,
        d_ff=d_model,
        n_heads=2,
    )

    input_shape = (1, 12)
    self._test_autoregressive_property(model_single_stage, input_shape,
                                       output_vocab_size=vocab_size)
    self._test_autoregressive_property(model_multi_stage, input_shape,
                                       output_vocab_size=vocab_size)

  def test_parse_hourglass_hierarchy(self):
    self.assertEqual(hourglass._parse_hierarchy('6@3'), ([6], [3]))
    self.assertEqual(hourglass._parse_hierarchy('3@2 2@6 5@24 2@6 3@2'), (
        [3, 2, 5], [2, 3, 4]
    ))
    self.assertRaises(ValueError, hourglass._parse_hierarchy, '1@2 2@3 1@2')
    self.assertRaises(ValueError, hourglass._parse_hierarchy, '1@2 2@3')


if __name__ == '__main__':
  absltest.main()
