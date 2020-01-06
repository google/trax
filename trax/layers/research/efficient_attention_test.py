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

"""Tests for trax.layers.research.efficient_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from tensorflow import test
from trax.layers import attention
from trax.layers import base
from trax.layers.research import efficient_attention
from trax.shapes import ShapeDtype


class EfficientAttentionTest(test.TestCase):

  def test_memory_efficient_causal_attention(self):
    qkv_shape = ShapeDtype((3, 32, 8))
    input_signature = (qkv_shape, qkv_shape, qkv_shape)
    layer = efficient_attention.MemoryEfficientCausalAttention(
        loop_stride=16, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual((3, 32, 8), final_shape)

  def test_time_bin_causal_attention_bin_length(self):
    qkv_shape = ShapeDtype((3, 57, 8))
    input_signature = (qkv_shape, qkv_shape, qkv_shape)
    layer = efficient_attention.TimeBinCausalAttention(
        bin_length=16, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual((3, 57, 8), final_shape)

  def test_time_bin_causal_attention_n_bins(self):
    qkv_shape = ShapeDtype((3, 57, 8))
    input_signature = (qkv_shape, qkv_shape, qkv_shape)
    layer = efficient_attention.TimeBinCausalAttention(
        n_bins=4, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual((3, 57, 8), final_shape)

  def test_time_bin_and_dot_product_causal_attention_are_consistent(self):
    dot_product_layer = attention.DotProductCausalAttention(
        dropout=0.0, mode='train')
    time_bin_layer = efficient_attention.TimeBinCausalAttention(
        bin_length=4, dropout=0.0, mode='train')

    # Exactly 2 bins.
    input_shape = (3, 8, 8)
    inputs = [onp.random.uniform(size=input_shape) for _ in range(3)]

    dot_product_output = dot_product_layer(inputs)
    time_bin_output = time_bin_layer(inputs)
    onp.testing.assert_array_almost_equal(dot_product_output, time_bin_output)

  def test_lsh_causal_attention_fast_inference(self):
    qkv_shape = ShapeDtype((3, 1, 8))
    input_signature = (qkv_shape, qkv_shape, qkv_shape)
    layer = efficient_attention.LSHCausalAttention(
        n_bins=4, dropout=0.0, max_len_for_inference=128, mode='predict')
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual((3, 1, 8), final_shape)


if __name__ == '__main__':
  test.main()
