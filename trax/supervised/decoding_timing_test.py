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

import copy
import functools
import os
import time
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np
import psutil
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


def memory_usage():
  return psutil.Process(os.getpid()).memory_info().rss


class DecodingTimingTest(test.TestCase):

  def _reformer2_decoding_time(self, settings):
    max_len = 16

    def _self_attention_fn():
      return functools.partial(
          tl.SelfAttention,
          predict_drop_len=2 * max_len,
          predict_mem_len=2 * max_len)

    def _causal_attention_fn():
      attn_layer, attn_kwargs = settings['attn']
      return functools.partial(
          attn_layer,
          max_inference_length=2 * max_len, **attn_kwargs)

    pred_model = models.Reformer2(
        mode='predict',
        d_model=settings['d_model'],
        d_ff=settings['d_ff'],
        dropout=0.1,
        max_len=max_len,
        n_heads=settings['n_heads'],
        n_encoder_layers=settings['encoder_layers'],
        n_decoder_layers=settings['decoder_layers'],
        encoder_attention_type=_self_attention_fn(),
        encoder_decoder_attention_type=_causal_attention_fn(),
        input_vocab_size=settings['vocab'],
        ff_sparsity=settings['ff_sparsity'],
        ff_use_sru=settings['ff_use_sru'],
        ff_dropout=0.1,
        ff_chunk_size=1024,
        attention_chunk_size=1,
        loss_sparsity=settings['loss_sparsity'],
        axial_pos_shape=None,
        use_bfloat16=True,
    )

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)
    pred_model.init(input_signature=(shape1l, shape11))
    inputs = np.arange(max_len, dtype=np.int32).reshape(1, max_len)

    # This is decoding.autoregressive_sample but simplified and with timing.
    result, start_time = [], time.time()
    elapsed_times = []
    peak_memory = 0
    for index, sample in zip(range(-4, 10),
                             decoding.autoregressive_sample_stream(
                                 pred_model, inputs, temperature=0.0)):
      peak_memory = max(peak_memory, memory_usage())
      result.append(sample[:, None])
      elapsed_time = time.time() - start_time
      if index >= 0:
        elapsed_times.append(elapsed_time)
      start_time = time.time()

    if min(elapsed_times) * 2 < max(elapsed_times):
      print('WARNING! High variance found in elapsed times! Settings: {} ; '
            'elapsed times: {} ; Probably more warm-up steps should be used, '
            'or model size should be increased.'.format(settings,
                                                        elapsed_times))
    # Check resulting shapes.
    s = np.concatenate(result, axis=1)
    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 14)
    return size_of_model(pred_model), elapsed_times, peak_memory

  def test_autoregressive_sample_reformer2_timing(self):
    # full model
    # 54B params
    # base_settings = {
    #     'encoder_layers': 6, 'decoder_layers': 36, 'vocab': 32000,
    #     'd_ff': 64*1024, 'd_model': 96*96, 'n_heads': 96,
    #     'ff_use_sru': (1, 64), 'ff_sparsity': (256, 32), 'loss_sparsity': 4,
    #     'attn': (tl.MultiplicativeConvCausalAttention,
    #              {'length_kernel_size': 1, 'sparsity': 64})}

    # 1/18 of model (1/6 of encoder, 1/18 of decoder, 1/18 of vocab)
    # 4B params
    base_settings = {
        'encoder_layers': 1, 'decoder_layers': 2, 'vocab': 7*256,
        'd_ff': 64*1024, 'd_model': 96*96, 'n_heads': 96,
        'ff_use_sru': (1, 64), 'ff_sparsity': (256, 32), 'loss_sparsity': 4,
        'attn': (tl.MultiplicativeConvCausalAttention,
                 {'length_kernel_size': 1, 'sparsity': 64})}

    all_settings = [
        # different attention layers
        {'attn': (tl.MultiplicativeConvCausalAttention,
                  {'length_kernel_size': 1, 'sparsity': 64})},
        {'attn': (tl.MultiplicativeConvCausalAttention,
                  {'length_kernel_size': 3, 'sparsity': 64})},
        {'attn': (tl.MultiplicativeModularCausalAttention,
                  {'sparsity': 64})},
        {'attn': (tl.MultiplicativeCausalAttention,
                  {'sparsity': 64})},
        {'attn': (tl.CausalAttention, {})},  # +40% params
        {'attn': (tl.CausalAttention, {}),
         'd_ff': int(5/8 * 64*1024)},        # + 0% params

        # different loss layers
        {'loss_sparsity': 8},
        {'loss_sparsity': 4},
        {'loss_sparsity': 2},
        {'loss_sparsity': 0},

        # different feed forward layers
        {'ff_use_sru': (1, 64), 'ff_sparsity': (256, 32)},
        {'ff_use_sru': 0, 'ff_sparsity': (256, 32)},
        {'ff_use_sru': (1, 64), 'ff_sparsity': 0},
        {'ff_use_sru': 0, 'ff_sparsity': 0},

        # no sparsity at all
        {'ff_use_sru': (1, 64), 'ff_sparsity': 0, 'loss_sparsity': 0,
         'attn': (tl.CausalAttention, {})},  # +40% params
        {'ff_use_sru': (1, 64), 'ff_sparsity': 0, 'loss_sparsity': 0,
         'attn': (tl.CausalAttention, {}),   # + 0% params
         'd_ff': int(5/8 * 64*1024)},
    ]

    total_times = []
    sizes = []
    messages = []
    for override_settings in all_settings:
      settings = copy.deepcopy(base_settings)
      settings.update(override_settings)

      init_memory = memory_usage()
      size, elapsed_times, peak_memory = self._reformer2_decoding_time(settings)

      total_time = sum(elapsed_times)
      time_diff = (max(elapsed_times) - min(elapsed_times)) / 2
      time_diff_percent = int(time_diff / np.mean(elapsed_times) * 100)
      message = (
          '\n\nParams: {}\nSettings: {}\nOverride: {}\n'
          'Init memory: {:.1f} GiB\nPeak memory: {:.1f} GiB\n'
          'Estimated model memory: {:.1f} GiB\n'
          'Time for 10 tokens: {:.4f} s +/- {} %\n\n\n'
          .format(size, settings, override_settings,
                  init_memory/1024**3, peak_memory/1024**3,
                  (peak_memory-init_memory)/1024**3,
                  total_time, time_diff_percent))
      print(message)
      messages.append(message)
      total_times.append(total_time)
      sizes.append(size)

    print('Final results (recap):')
    for message in messages:
      print(message)

    # This is useful for copying results into a spreadsheet etc.
    for i in range(len(all_settings)):
      print(sizes[i], total_times[i], sep='\t')

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
