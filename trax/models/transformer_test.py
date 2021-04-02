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
from trax import shapes
from trax.layers import test_utils
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

  @parameterized.named_parameters(
      ('same_vocab', 16, None),
      ('same_size', 16, 16),
      ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


  def test_dot_product_causal_attention_fast_inference(self):
    model_fn = functools.partial(
        transformer.TransformerLM, d_model=4, d_ff=8, n_layers=2, n_heads=2
    )
    test_utils.test_eval_equals_predict_discrete(model_fn)


if __name__ == '__main__':
  absltest.main()
