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
"""Tests for decoding."""

import functools
import os
import time

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np
from tensorflow.compat.v2 import test

from trax import fastmath
from trax import layers
from trax import models
from trax import shapes
from trax.supervised import decoding


pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')


class DecodingTest(test.TestCase):

  def test_autoregressive_sample_transformerlm(self):
    model = models.TransformerLM(10, d_model=32, d_ff=64, n_layers=1,
                                 n_heads=2, mode='predict')
    model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
    s1 = decoding.autoregressive_sample(
        model, batch_size=1, eos_id=-1, max_length=10)
    self.assertEqual(s1.shape[0], 1)
    self.assertEqual(s1.shape[1], 10)
    batch_per_device = 2 // fastmath.device_count()
    model.init(shapes.ShapeDtype((batch_per_device, 1), dtype=np.int32))
    s2 = decoding.autoregressive_sample(
        model, batch_size=2, max_length=10)
    self.assertEqual(s2.shape[0], 2)
    self.assertLess(s2.shape[1], 11)
    model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
    prefix = np.array([[1, 2, 3]])
    s3 = decoding.autoregressive_sample(model, prefix, eos_id=-1, max_length=10,
                                        batch_size=1)
    self.assertEqual(s3.shape[0], 1)
    self.assertEqual(s3.shape[1], 10)

  def _lsh_self_attention_fn(self):
    return functools.partial(
        layers.LSHSelfAttention,
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

  def _timebin_self_attention_fn(self, use_reference_code=False, chunk_len=64):
    return functools.partial(
        layers.SelfAttention,
        attention_dropout=0.05,
        chunk_len=chunk_len,
        n_chunks_before=1,
        n_parallel_heads=1,
        use_reference_code=use_reference_code,
        predict_drop_len=128,
        predict_mem_len=1024,
    )

  def test_autoregressive_sample_reformerlm(self):
    lsh_self_attention = self._lsh_self_attention_fn()
    timebin_self_attention = self._timebin_self_attention_fn()

    model = models.ReformerLM(vocab_size=256,
                              d_model=256,
                              d_ff=512,
                              d_attention_key=128,
                              d_attention_value=128,
                              n_layers=2,
                              n_heads=2,
                              dropout=0.05,
                              max_len=65536,
                              attention_type=[timebin_self_attention,
                                              lsh_self_attention],
                              axial_pos_shape=(256, 256),
                              d_axial_pos_embs=(128, 128),
                              ff_activation=layers.Relu,
                              ff_use_sru=0,
                              mode='predict',
                              )
    model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
    s1 = decoding.autoregressive_sample(
        model, batch_size=1, eos_id=-1, max_length=10)
    self.assertEqual(s1.shape[0], 1)
    self.assertEqual(s1.shape[1], 10)

  def test_autoregressive_sample_transformer(self):
    model = models.Transformer(10, d_model=32, d_ff=64, n_encoder_layers=1,
                               n_decoder_layers=1, n_heads=2, mode='predict')
    inputs = np.ones((1, 3), dtype=np.int32)
    model.init((shapes.signature(inputs),
                shapes.ShapeDtype((1, 1), dtype=np.int32)))
    s = decoding.autoregressive_sample(model, inputs=inputs,
                                       eos_id=-1, max_length=10)
    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 10)

  def test_autoregressive_sample_transformerlm_quality(self):
    pred_model = models.TransformerLM(
        d_model=64, d_ff=128, dropout=0.05, max_len=256, n_heads=2,
        n_layers=2, vocab_size=13, mode='predict')
    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    model_path = os.path.join(_TESTDATA, 'transformerlm_copy.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=(shape11, shape11))
    inputs = np.array([[0, 3, 7, 5, 3, 2, 4, 0]], dtype=np.int32)
    s = decoding.autoregressive_sample(pred_model, inputs,
                                       max_length=6, temperature=0.0)
    self.assertEqual(str(s[0]), '[3 7 5 3 2 4]')

  def test_autoregressive_sample_reformerlm_quality(self):
    timebin_self_attention = self._timebin_self_attention_fn()
    pred_model = models.ReformerLM(
        d_model=64, d_ff=128, dropout=0.05, max_len=256, n_heads=2,
        attention_type=timebin_self_attention,
        n_layers=2, vocab_size=13, mode='predict')
    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    model_path = os.path.join(_TESTDATA, 'reformerlm_copy.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=(shape11, shape11))
    inputs = np.array([[0, 3, 7, 5, 3, 2, 4, 0]], dtype=np.int32)
    s = decoding.autoregressive_sample(pred_model, inputs,
                                       max_length=6, temperature=0.0)
    self.assertEqual(str(s[0]), '[3 7 5 3 2 4]')

  def test_autoregressive_sample_transformer_quality(self):
    pred_model = models.Transformer(
        d_model=64, d_ff=128, dropout=0.05, max_len=256, n_heads=2,
        n_encoder_layers=2, n_decoder_layers=2, input_vocab_size=13,
        mode='predict')
    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    model_path = os.path.join(_TESTDATA, 'transformer_copy.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=(shape11, shape11))
    inputs = np.array([[3, 7, 5, 3, 2, 4, 1, 8]], dtype=np.int32)
    s = decoding.autoregressive_sample(pred_model, inputs=inputs,
                                       eos_id=1, max_length=10, temperature=0.0)
    self.assertEqual(str(s[0]), '[3 7 5 3 2 4 1]')

  def test_autoregressive_sample_reformer2_lsh(self):
    max_len = 128

    pred_model = models.Reformer2(
        mode='predict',
        d_model=256,
        d_ff=512,
        dropout=0.05,
        max_len=max_len,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ff_use_sru=1,
        d_attention_key=64,
        d_attention_value=64,
        encoder_attention_type=self._lsh_self_attention_fn(),
        encoder_decoder_attention_type=self._lsh_self_attention_fn(),
        input_vocab_size=256,
        axial_pos_shape=None,
    )

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)
    pred_model.init(input_signature=(shape1l, shape11))

    # 0w0w
    inputs = np.array(
        [[0, 3, 7, 5, 3, 2, 4, 1, 8, 0, 3, 7, 5, 3, 2, 4, 1, 8]],
        dtype=np.int32)
    inputs = np.pad(inputs, [(0, 0), (0, max_len - inputs.shape[1])],
                    mode='constant', constant_values=0)
    s = decoding.autoregressive_sample(
        pred_model, inputs=inputs, eos_id=-1, max_length=10, temperature=0.0)

    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 10)

  def test_autoregressive_sample_reformer2_timing(self):
    max_len = 16

    def _self_attention_fn():
      return functools.partial(
          layers.SelfAttention,
          predict_drop_len=2 * max_len,
          predict_mem_len=2 * max_len,
      )

    pred_model = models.Reformer2(
        mode='predict',
        d_model=4*1024,
        d_ff=32*1024,
        dropout=0.05,
        max_len=max_len,
        n_heads=16,
        n_encoder_layers=3,
        n_decoder_layers=3,
        encoder_attention_type=_self_attention_fn(),
        encoder_decoder_attention_type=_self_attention_fn(),
        input_vocab_size=32,
        ff_sparsity=128,
        axial_pos_shape=None,
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

    print('\n\n\nTotal time (10 tokens): %.4fs\n\n\n' % total_time)
    self.assertLess(total_time, 10.0)  # If it's > 10s, it's some bug.
    # Check resulting shapes.
    s = np.concatenate(result, axis=1)
    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 14)

  def test_autoregressive_sample_reformer2_copy_self_attn_quality(self):
    max_len = 32

    def _self_attention_fn():
      return functools.partial(
          layers.SelfAttention,
          predict_drop_len=2 * max_len,
          predict_mem_len=2 * max_len,
      )

    pred_model = models.Reformer2(
        mode='predict',
        d_model=256,
        d_ff=512,
        dropout=0.05,
        max_len=max_len,
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=3,
        ff_use_sru=1,
        d_attention_key=64,
        d_attention_value=64,
        encoder_attention_type=_self_attention_fn(),
        encoder_decoder_attention_type=_self_attention_fn(),
        input_vocab_size=13,
        axial_pos_shape=None,
    )

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)

    model_path = os.path.join(_TESTDATA,
                              'reformer2_copy_self_attn.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=(shape1l, shape11))

    inputs = np.array([[11, 5, 1, 2, 3, 4]], dtype=np.int32)
    inp_len = inputs.shape[1]
    inputs = np.pad(inputs, [(0, 0), (0, max_len - inp_len)],
                    mode='constant', constant_values=0)
    s = decoding.autoregressive_sample(
        pred_model, inputs=inputs, eos_id=-1, max_length=inp_len,
        temperature=0.0)

    np.testing.assert_equal(s[0], inputs[0, :inp_len])


if __name__ == '__main__':
  config.config_with_absl()
  test.main()
