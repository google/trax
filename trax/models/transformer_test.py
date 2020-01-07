# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Tests for Transformer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
from trax import layers as tl
from trax import math
from trax.math import numpy as np
from trax.models import transformer
from trax.shapes import ShapeDtype


class TransformerTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    """Run the Transformer LM forward and check output shape."""
    vocab_size = 16
    input_signature = ShapeDtype((3, 5), onp.int32)
    model = transformer.TransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual((3, 5, vocab_size), final_shape)

  def _test_transformer_forward_shape(self, input_vocab_size,
                                      output_vocab_size):
    """Run the Transformer forward and check output shape."""
    input_sd = ShapeDtype((3, 5), onp.int32)
    input_signature = (input_sd, input_sd)
    model = transformer.Transformer(
        input_vocab_size, output_vocab_size, d_model=32, d_ff=64,
        n_encoder_layers=2, n_decoder_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(model, input_signature)
    vocab_size = output_vocab_size or input_vocab_size
    expected_shape = (3, 5, vocab_size)
    self.assertEqual(expected_shape, final_shape[0])

  @parameterized.named_parameters(
      ('same_vocab', 16, None),
      ('same_size', 16, 16),
      ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


  def _test_fast_inference(self, attention_type, length):
    with math.use_backend('jax'):
      vocab_size = 16
      model_fn = functools.partial(
          transformer.TransformerLM,
          vocab_size=vocab_size, d_model=4, d_ff=8, n_layers=2, n_heads=2,
          attention_type=attention_type,
      )
      model_slow = model_fn(mode='eval')
      model_fast = model_fn(mode='predict')
      rng = math.random.get_prng(0)
      batch_size = 2
      input_signature = ShapeDtype((batch_size, 1), np.int32)
      # Given the same rng, both models initialize with the same parameters.
      model_slow.init(input_signature)
      model_fast.init(input_signature)

      buf = onp.zeros((batch_size, length), dtype=np.int32)
      next_sym = onp.zeros((batch_size, 1), dtype=onp.int32)

      for index in range(length):
        logits_slow = model_slow(buf, rng=rng)
        logits_fast = model_fast(next_sym, rng=rng)
        onp.testing.assert_array_almost_equal(
            logits_slow[:, index, :], logits_fast[:, 0, :],
            decimal=5,
        )
        next_sym = onp.random.randint(vocab_size, size=(batch_size, 1))
        buf[:, index] = next_sym[:, 0]

  def test_dot_product_causal_attention_fast_inference(self):
    self._test_fast_inference(tl.DotProductCausalAttention, length=5)

  def test_time_bin_causal_attention_fast_inference(self):
    attention = functools.partial(tl.TimeBinCausalAttention, bin_length=2)
    self._test_fast_inference(attention, length=7)

if __name__ == '__main__':
  absltest.main()
