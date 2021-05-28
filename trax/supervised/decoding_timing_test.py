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
"""Timing tests for decoding."""

import copy
import functools
import gc
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


def _size_of_model(model):
  def _size(x):
    try:
      return x.size
    except Exception:  # pylint: disable=broad-except
      return 0
  sizes = fastmath.nested_map(_size, model.weights)
  total_size = sum(fastmath.tree_flatten(sizes))
  return total_size


def _recurrent_delete(w):
  if 'delete' in dir(w):
    # Object has a 'delete' method, so it is a DeviceArray or something similar,
    # so we want to delete it.
    w.delete()
  elif isinstance(w, (list, tuple)):
    for x in w:
      _recurrent_delete(x)
  elif isinstance(w, dict):
    for x in w.values():
      _recurrent_delete(x)
  else:
    raise ValueError('Unknown type encountered in weights: {}'.format(type(w)))


def _memory_usage():
  gc.collect()
  return psutil.Process(os.getpid()).memory_info().rss


class DecodingTimingTest(test.TestCase):

  def _terraformer_decoding_time(self, settings):
    # Garbage collection influences the timing, so we turn it off.
    gc.disable()
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

    if settings['model'] == 'terraformer':
      pred_model = models.ConfigurableTerraformer(
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
          # ff_chunk_size=1024,
          # attention_chunk_size=1,
          n_decoder_attention_layers=settings['attention_layers'],
          loss_sparsity=settings['loss_sparsity'],
          pos_axial_shape=None,
          use_bfloat16=True,
      )
    elif settings['model'] == 'transformer':
      pred_model = models.ConfigurableTransformer(
          mode='predict',
          d_model=settings['d_model'],
          d_ff=settings['d_ff'],
          dropout=0.1,
          max_len=max_len,
          n_heads=settings['n_heads'],
          n_encoder_layers=settings['encoder_layers'],
          n_decoder_layers=settings['decoder_layers'],
          # encoder_attention_type=_self_attention_fn(),
          encoder_decoder_attention_type=_causal_attention_fn(),
          input_vocab_size=settings['vocab'],
          ff_sparsity=settings['ff_sparsity'],
          ff_use_sru=settings['ff_use_sru'],
          # ff_dropout=0.1,
          # ff_chunk_size=1024,
          # attention_chunk_size=1,
          # n_decoder_attention_layers=settings['attention_layers'],
          loss_sparsity=settings['loss_sparsity'],
          pos_axial_shape=None,
          # enc_dec_attention_sparsity=settings['enc_dec_sparsity'],
          # use_bfloat16=True,
      )
    else:
      assert False
    # We put acceleration outside of autoregressive_sample_stream, because
    # we want to have a separate run (separate input) for model compilation.
    pred_model = tl.Accelerate(pred_model)

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)
    pred_model.init(input_signature=(shape1l, shape11))
    original_state = copy.deepcopy(pred_model.state)

    inputs_warmup = np.zeros((1, max_len), dtype=np.int32)
    inputs = np.arange(max_len, dtype=np.int32).reshape(1, max_len)

    # This is a warm-up run, for compilation.
    result, current_time = [], time.time()
    elapsed_warmup_times = []
    for index, sample in zip(range(0, 4), decoding.autoregressive_sample_stream(
        pred_model, inputs_warmup, temperature=0.0, accelerate=False)):
      del index  # unused
      result.append(sample[:, None])  # to be sure that the result is computed

      current_time, start_time = time.time(), current_time
      elapsed_warmup_times.append(current_time - start_time)

    # This is a real decoding timing run that we measure.
    pred_model.state = original_state
    result, current_time = [], time.time()
    elapsed_times = []
    for index, sample in zip(range(12), decoding.autoregressive_sample_stream(
        pred_model, inputs, temperature=0.0, accelerate=False)):
      del index  # unused
      result.append(sample[:, None])  # to be sure that the result is computed

      current_time, start_time = time.time(), current_time
      elapsed_times.append(current_time - start_time)
    peak_memory = _memory_usage()

    if min(elapsed_times[2:]) * 2 < max(elapsed_times[2:]):
      print('WARNING! High variance found in elapsed times! Settings: {} ; '
            'elapsed times: {} ; Probably more warm-up steps should be used, '
            'or model size should be increased.'.format(settings,
                                                        elapsed_times))
    # Check resulting shapes.
    s = np.concatenate(result, axis=1)
    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 12)
    model_size = int(_size_of_model(pred_model))

    # We delete the model weights, because in some situations they won't be
    # deleted automatically.
    _recurrent_delete(pred_model.weights)
    gc.enable()
    return model_size, elapsed_times, peak_memory

  def test_autoregressive_sample_terraformer_timing(self):
    template_to_use = 'medium_transformer'

    settings_templates = {
        # full model
        # # 54B params
        # 'full_model': {
        #     'encoder_layers': 6, 'decoder_layers': 36, 'vocab': 32000,
        #     'attention_layers': 2,
        #     'd_ff': 64*1024, 'd_model': 96*96, 'n_heads': 96,
        #     'ff_use_sru': (1, 64), 'ff_sparsity': (256, 32),
        #     'loss_sparsity': 8,
        #     'attn': (tl.MultiplicativeConvCausalAttention,
        #              {'length_kernel_size': 3, 'sparsity': 64})},

        # 1/18 of model (1/6 of encoder, 1/18 of decoder, full vocab)
        # 4B params
        # 'big_terraformer': {
        #     'model': 'terraformer',
        #     'encoder_layers': 1, 'decoder_layers': 2, 'vocab': 32000,
        #     'attention_layers': 2,
        #     'd_ff': int(5/8 * 64*1024), 'd_model': 96*96, 'n_heads': 96,
        #     'ff_use_sru': 0, 'ff_sparsity': 0, 'loss_sparsity': 0,
        #     'attn': (tl.CausalAttention, {})},

        # 'big_transformer': {
        #     'model': 'transformer',
        #     'encoder_layers': 1, 'decoder_layers': 2, 'vocab': 32000,
        #     'attention_layers': 2,
        #     'd_ff': int(5/8 * 64*1024), 'd_model': 96*96, 'n_heads': 96,
        #     'ff_use_sru': 0, 'ff_sparsity': 0, 'loss_sparsity': 0,
        #     'attn': (tl.CausalAttention, {})},

        # medium model
        # 275M params (only decoder)
        'medium_transformer': {
            'model': 'transformer',
            'encoder_layers': 2, 'decoder_layers': 24, 'vocab': 32000,
            'attention_layers': 2,
            'd_ff': 4*1024, 'd_model': 1024, 'n_heads': 16,
            'ff_use_sru': 0, 'ff_sparsity': 0, 'loss_sparsity': 0,
            'attn': (tl.CausalAttention, {})},
        # 'medium_terraformer': {
        #     'model': 'terraformer',
        #     'encoder_layers': 2, 'decoder_layers': 24, 'vocab': 32000,
        #     'attention_layers': 2,
        #     'd_ff': 4*1024, 'd_model': 1024, 'n_heads': 16,
        #     'ff_use_sru': 0, 'ff_sparsity': 0, 'loss_sparsity': 0,
        #     'attn': (tl.CausalAttention, {})},

    }

    sweep_settings = {
        # 'big_transformer': [  # for big
        #     dict(), # baseline
        #     {'ff_sparsity': (256, 32)},  # + Sparse FF
        #     {'attn': (  # + Sparse QKV
        #         tl.MultiplicativeConvCausalAttention,
        #         {'length_kernel_size': 3, 'sparsity': 64}),
        #      'd_ff': 64*1024,
        #      },
        #     {'ff_sparsity': (256, 32),
        #      'attn': (  # + Sparse FF+QKV
        #         tl.MultiplicativeConvCausalAttention,
        #         {'length_kernel_size': 3, 'sparsity': 64}),
        #      'd_ff': 64*1024,
        #      },
        # ],

        'medium_transformer': [  # for medium
            dict(),  # baseline

            {'ff_sparsity': 64,
             'attn': (  # Sparse FF+QKV
                 tl.MultiplicativeConvCausalAttention,
                 {'length_kernel_size': 3, 'sparsity': 16}),
             'd_ff': 6*1024,
             },

            # {'ff_sparsity': 64,  # Sparse FF+QKV + Loss
            #  'attn': (
            #     tl.MultiplicativeConvCausalAttention,
            #     {'length_kernel_size': 3, 'sparsity': 16}),
            #  'd_ff': 6*1024,
            #  'loss_sparsity': 4,
            #  },

            # {'attn': (  # Sparse QKV
            #     tl.MultiplicativeConvCausalAttention,
            #     {'length_kernel_size': 3, 'sparsity': 16}),
            #  'd_ff': 6*1024,
            #  },
            # {'loss_sparsity': 4},  # Sparse Loss
            # {'ff_sparsity': 64},  # Sparse FF

            # {'ff_sparsity': 128},  # + Sparse FF 128

            # APPENDIX below

            # different loss layers
            # {'loss_sparsity': 8},
            # {'loss_sparsity': 2},
            # {'loss_sparsity': 0},
        ],

        #  'big_terraformer': [  # for big terraformer
        #      dict(), # baseline
        #      {'ff_sparsity': 64},  # + Sparse FF  / Sparse FF 64
        #      {'ff_sparsity': 64,
        #       'attn': (  # + Sparse FF+QKV
        #          tl.MultiplicativeConvCausalAttention,
        #          {'length_kernel_size': 3, 'sparsity': 16}),
        #       'd_ff': 6*1024,
        #       },
        #      {'ff_sparsity': 64,  # + Sparse FF+QKV+Loss
        #       'attn': (
        #          tl.MultiplicativeConvCausalAttention,
        #          {'length_kernel_size': 3, 'sparsity': 16}),
        #       'd_ff': 6*1024,
        #       'loss_sparsity': 4,
        #       },

        #         ],

        # 'medium_terraformer': [  # for medium terraformer
        #     {'ff_sparsity': 64,  # + Sparse FF+QKV+Loss
        #      'attn': (
        #         tl.MultiplicativeConvCausalAttention,
        #         {'length_kernel_size': 3, 'sparsity': 16}),
        #      'd_ff': 6*1024,
        #      'loss_sparsity': 4,
        #      },
        # ],
    }

    encoding_times = []
    decoding_times = []
    sizes = []
    memories = []
    messages = []
    for override_settings in sweep_settings[template_to_use]:
      settings = copy.deepcopy(settings_templates[template_to_use])
      settings.update(override_settings)

      init_memory = _memory_usage()
      size, elapsed_times, peak_memory = (
          self._terraformer_decoding_time(settings))

      # TODO(jaszczur): Why is elapsed_times[0] always small?
      encoding_time = elapsed_times[1]
      decoding_time_10 = sum(elapsed_times[2:])

      after_memory = _memory_usage()
      model_memory_gigabytes = (peak_memory-init_memory)/1024**3
      decoding_time_diff = (max(elapsed_times[2:]) - min(elapsed_times[2:])) / 2
      decoding_time_diff_percent = int(
          decoding_time_diff / np.mean(elapsed_times) * 100)
      message = (
          '\n\n'
          'Params: {}\n'
          'Settings: {}\n'
          'Override: {}\n'
          'Init memory: {:.1f} GiB\n'
          'Peak memory: {:.1f} GiB\n'
          'After memory: {:.1f} GiB\n'
          'Estimated model memory: {:.1f} GiB\n'
          'Times for each step: {}\n'
          'Time for encoding: {:.4f} s\n'
          'Time for decoding 10 tokens: {:.4f} s +/- {} %\n'
          '\n\n'
          .format(size, settings, override_settings,
                  init_memory/1024**3, peak_memory/1024**3,
                  after_memory/1024**3, model_memory_gigabytes,
                  elapsed_times, encoding_time,
                  decoding_time_10, decoding_time_diff_percent))
      print(message)
      messages.append(message)
      encoding_times.append(encoding_time)
      decoding_times.append(decoding_time_10)
      sizes.append(size)
      memories.append(model_memory_gigabytes)

    print('Final results (recap):')
    for message in messages:
      print(message)

    # This is useful for copying results into a spreadsheet etc.
    # for i in range(len(sweep_settings)):
    #   print('{}\t{}\t{}\t{:.1f}'.format(
    #       sizes[i], encoding_times[i], decoding_times[i], memories[i]))

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
          % (_size_of_model(pred_model), settings, total_time))
      messages.append(message)
      print(message)

    print('Final results (recap):')
    for message in messages:
      print(message)


if __name__ == '__main__':
  config.config_with_absl()
  test.main()
