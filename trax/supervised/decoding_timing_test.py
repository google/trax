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
"""Timing tests for decoding."""

import functools
import time

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np
from tensorflow.compat.v2 import test

from trax import fastmath
from trax import layers as tl
from trax import models
from trax import shapes
from trax.supervised import decoding


def size_of_model(model):
  def _size(x):
    try:
      return x.size
    except Exception:  # pylint: disable=broad-except
      return 0
  sizes = fastmath.nested_map(_size, model.weights)
  total_size = sum(fastmath.tree_flatten(sizes))
  return total_size


class DecodingTimingTest(test.TestCase):

  def test_autoregressive_sample_reformer2_timing(self):
    max_len = 16
    all_settings = [
        {'attn_sparsity': 64, 'ff_sparsity': (256, 32),
         'attn': (tl.MultiplicativeCausalAttention, {})},
        {'attn_sparsity': 64, 'ff_sparsity': (256, 32),
         'attn': (tl.ModularCausalAttention, {})},
        {'attn_sparsity': 64, 'ff_sparsity': (256, 32),
         'attn': (tl.ConvCausalAttention, {})},
        {'attn_sparsity': 64, 'ff_sparsity': (256, 32),
         'attn': (tl.MultiplicativeConvCausalAttention,
                  {'length_kernel_size': 1})},
        {'attn_sparsity': 64, 'ff_sparsity': (256, 32),
         'attn': (tl.MultiplicativeConvCausalAttention,
                  {'length_kernel_size': 3})},
    ]
    messages = []

    for settings in all_settings:

      def _self_attention_fn():
        return functools.partial(
            tl.SelfAttention,
            predict_drop_len=2 * max_len,
            predict_mem_len=2 * max_len)

      def _causal_attention_fn():
        attn_layer, attn_kwargs = settings['attn']  # pylint: disable=cell-var-from-loop
        return functools.partial(
            attn_layer,
            sparsity=settings['attn_sparsity'],  # pylint: disable=cell-var-from-loop
            max_inference_length=2 * max_len, **attn_kwargs)

      pred_model = models.Reformer2(
          mode='predict',
          d_model=96*96,
          d_ff=64*1024,
          dropout=0.05,
          max_len=max_len,
          n_heads=64,
          n_encoder_layers=2,
          n_decoder_layers=2,
          encoder_attention_type=_self_attention_fn(),
          encoder_decoder_attention_type=_causal_attention_fn(),
          input_vocab_size=4,
          ff_sparsity=settings['ff_sparsity'],
          axial_pos_shape=None,
          use_bfloat16=True,
      )

      shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
      shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)
      pred_model.init(input_signature=(shape1l, shape11))
      inputs = np.arange(16, dtype=np.int32).reshape(1, 16)

      # This is decoding.autoregressive_sample but simplified and with timing.
      result, counter, start_time, total_time = [], 0, time.time(), 0.0
      for sample in decoding.autoregressive_sample_stream(
          pred_model, inputs, temperature=0.0):  # accelerate=False):
        elapsed_time = time.time() - start_time
        start_time = time.time()
        if counter > 3:
          total_time += elapsed_time
        result.append(sample[:, None])
        counter += 1
        if counter >= 14:
          break

      # We print 5* time for 10 tokens, @2 layers this is ~1 token @ 100 layers.
      message = (
          ('\n\nParams: %d Settings: %s\n'
           'Time for 5x10 tokens (~1tok @100): %.4f s\n\n\n')
          % (size_of_model(pred_model), settings, 5*total_time))
      messages.append(message)
      print(message)
      # self.assertLess(total_time, 20.0)  # If it's > 20s, it's some bug.
      # Check resulting shapes.
      s = np.concatenate(result, axis=1)
      self.assertEqual(s.shape[0], 1)
      self.assertEqual(s.shape[1], 14)

    print('Final results (recap):')
    for message in messages:
      print(message)

  def test_loss_layer_timing(self):
    all_settings = [
        # The first run is sometimes slower, less reliable.
        {'output': 32000, 'input': 2048, 'prob': None,
         'type': None, 'sparsity': 0, 'lowrank': 0, 'use_bias': False},

        {'output': 32000, 'input': 2048, 'prob': None,
         'type': None, 'sparsity': 0, 'lowrank': 0, 'use_bias': False},
        {'output': 32000, 'input': 2048, 'prob': None,
         'type': 'einsum', 'sparsity': 0, 'lowrank': 0, 'use_bias': False},
        {'output': 32000, 'input': 2048, 'prob': None,
         'type': 'mult', 'sparsity': 2, 'lowrank': 0, 'use_bias': False},

        {'output': 32000, 'input': 2048, 'prob': None,
         'type': None, 'sparsity': 0, 'lowrank': 0, 'use_bias': True},
        {'output': 32000, 'input': 2048, 'prob': None,
         'type': 'einsum', 'sparsity': 0, 'lowrank': 0, 'use_bias': True},
        {'output': 32000, 'input': 2048, 'prob': None,
         'type': 'mult', 'sparsity': 2, 'lowrank': 0, 'use_bias': True},
    ]

    messages = []
    for settings in all_settings:
      pred_model = tl.SparseDenseWithOptions(
          n_units=settings['output'],
          d_input=settings['input'],
          sparsity_type=settings['type'],
          sparsity=settings['sparsity'],
          d_lowrank=settings['lowrank'],
          prob_sparse=settings['prob'],
          use_bias=settings['use_bias'],
          mode='predict',
          )
      pred_model = tl.Accelerate(pred_model)

      shape1l = shapes.ShapeDtype((1, settings['input']))
      pred_model.init(input_signature=shape1l)
      inputs = np.ones((1, settings['input']))

      total_time = 0.0
      for counter in range(-50, 100):
        start_time = time.time()
        y = pred_model(inputs)
        self.assertEqual(y.shape, (1, settings['output']))
        elapsed_time = time.time() - start_time
        if counter >= 0:
          total_time += elapsed_time

      message = (
          '\n\nParams: %d Settings: %s\nTime for 100 tokens: %.4f s\n\n\n'
          % (size_of_model(pred_model), settings, total_time))
      messages.append(message)
      print(message)

    print('Final results (recap):')
    for message in messages:
      print(message)


if __name__ == '__main__':
  config.config_with_absl()
  test.main()
