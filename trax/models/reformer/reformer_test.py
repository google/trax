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

from absl.testing import absltest
from absl.testing import parameterized

from trax import layers as tl
from trax.math import numpy as np
from trax.models.reformer import reformer
from trax.shapes import ShapeDtype


class ReformerTest(parameterized.TestCase):

  def test_reformer_lm_forward_shape(self):
    """Run the ReformerLM forward and check output shape."""
    vocab_size = 16
    input_sd = ShapeDtype((1, 8), np.int32)
    input_signature = (input_sd, input_sd)
    model = reformer.ReformerLM(
        vocab_size, d_model=32, d_ff=64, d_attention_key=16,
        d_attention_value=16, n_layers=1, n_heads=2, max_len=16)
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual(((1, 8, 16), (1, 8)), final_shape)


if __name__ == '__main__':
  absltest.main()
