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
"""Tests for decoding."""

import functools
import os

import gin
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np
from tensorflow.compat.v2 import test

from trax import fastmath
from trax import layers as tl
from trax import models
from trax import shapes
from trax.supervised import decoding


pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')
_CONFIG_DIR = os.path.join(pkg_dir, 'configs/')


class DecodingTest(test.TestCase):

  def test_autoregressive_sample_transformerlm(self):
    model = models.TransformerLM(10, d_model=32, d_ff=64, n_layers=1,
                                 n_heads=2, mode='predict')
    model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
    s1 = decoding.autoregressive_sample(
        model, batch_size=1, eos_id=-1, max_length=10)
    self.assertEqual(s1.shape[0], 1)
    self.assertEqual(s1.shape[1], 10)
    batch_per_device = 2 // fastmath.local_device_count()
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

  def test_autoregressive_sample_transformerlm_tfnp(self):
    with fastmath.use_backend(fastmath.Backend.TFNP):
      model = models.TransformerLM(10, d_model=32, d_ff=64, n_layers=1,
                                   n_heads=2, mode='predict')
      model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
      s1 = decoding.autoregressive_sample(
          model, batch_size=1, eos_id=-1, max_length=10)
      self.assertEqual(s1.shape[0], 1)
      self.assertEqual(s1.shape[1], 10)
      batch_per_device = 2 // fastmath.local_device_count()
      model.init(shapes.ShapeDtype((batch_per_device, 1), dtype=np.int32))
      s2 = decoding.autoregressive_sample(
          model, batch_size=2, max_length=10)
      self.assertEqual(s2.shape[0], 2)
      self.assertLess(s2.shape[1], 11)
      model.init(shapes.ShapeDtype((1, 1), dtype=np.int32))
      prefix = np.array([[1, 2, 3]])
      s3 = decoding.autoregressive_sample(model, prefix, eos_id=-1,
                                          max_length=10, batch_size=1)
      self.assertEqual(s3.shape[0], 1)
      self.assertEqual(s3.shape[1], 10)

  def _lsh_self_attention_fn(self):
    return functools.partial(
        tl.LSHSelfAttention,
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

  def _pure_lsh_self_attention_fn(self, n_chunks_after=0):
    return functools.partial(
        tl.PureLSHSelfAttentionWrapper,
        attention_dropout=0.0,
        chunk_len=16,
        n_buckets=[32, 32],
        n_chunks_after=n_chunks_after,
        n_chunks_before=1,
        n_hashes=2,
        n_parallel_heads=1,
        max_length_for_buckets=1024,
        predict_drop_len=128,
        predict_mem_len=1024,
        num_weights=2,
        bias=False,
        pure_lsh_implementation=tl.PureLSHSelfAttention,
    )

  def _timebin_self_attention_fn(self, use_reference_code=False, chunk_len=64):
    return functools.partial(
        tl.SelfAttention,
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
                              pos_axial_shape=(256, 256),
                              pos_d_axial_embs=(128, 128),
                              ff_activation=tl.Relu,
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

  def test_autoregressive_sample_transformerlm_quality_eval(self):
    eval_model = models.TransformerLM(
        d_model=64, d_ff=128, dropout=0.05, max_len=256, n_heads=2,
        n_layers=2, vocab_size=13, mode='eval')
    model_path = os.path.join(_TESTDATA, 'transformerlm_copy.pkl.gz')
    eval_model.init_from_file(model_path)
    inputs = np.array([[0, 3, 7, 5, 3, 2, 4, 0]], dtype=np.int32)
    s = decoding.autoregressive_sample(eval_model, inputs, eval_mode=True,
                                       max_length=6, temperature=0.0)
    self.assertEqual(str(s[0]), '[3 7 5 3 2 4]')

  def test_autoregressive_sample_transformerlm_quality_beam(self):
    pred_model = models.TransformerLM(
        d_model=64, d_ff=128, dropout=0.05, max_len=256, n_heads=2,
        n_layers=2, vocab_size=13, mode='predict')
    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    model_path = os.path.join(_TESTDATA, 'transformerlm_copy.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=(shape11, shape11))
    inputs = np.array([[0, 3, 7, 5, 3, 2, 4, 0]], dtype=np.int32)
    s = decoding.beam_search(pred_model, inputs, n_beams=3, max_length=6)
    self.assertEqual(len(s), 3)  # 3 beams
    self.assertEqual(str(s[0][0][0]), '[3 7 5 3 2 4]')
    self.assertEqual(str(s[1][0][0]), '[3 7 5 3 2 2]')  # different from above
    self.assertEqual(str(s[2][0][0]), '[3 7 5 3 3 2]')  # different from above

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

  def test_autoregressive_sample_terraformer_lsh(self):
    max_len = 128

    pred_model = models.ConfigurableTerraformer(
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
        pos_axial_shape=None,
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

  def test_autoregressive_sample_terraformer_lsh_attn_quality(self):
    gin.add_config_file_search_path(_CONFIG_DIR)
    max_len = 32  # 32 is the max length we trained the checkpoint for.
    test_lengths = [8, 16, 32]
    vocab_size = 13
    # The checkpoint is correct on ~90% sequences, set random seed to deflake.
    np.random.seed(0)
    for test_len in test_lengths:
      gin.clear_config()
      gin.parse_config_file('terraformer_copy.gin')
      gin.bind_parameter('LSHSelfAttention.predict_mem_len', 2 * max_len)
      gin.bind_parameter('LSHSelfAttention.predict_drop_len', 2 * max_len)

      pred_model = models.ConfigurableTerraformer(mode='predict')

      shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
      shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)

      model_path = os.path.join(_TESTDATA, 'terraformer_copy_lsh_attn.pkl.gz')
      pred_model.init_from_file(model_path, weights_only=True,
                                input_signature=(shape1l, shape11))
      initial_state = pred_model.state

      for _ in range(2):  # Set low to make the test run reasonably fast.
        # Pick a length in [1, test_len] at random.
        inp_len = np.random.randint(low=1, high=test_len + 1)
        inputs = np.random.randint(low=1, high=vocab_size-1, size=(1, max_len))
        # TODO(jaszczur): properly fix padding in terraformer predict mode,
        # and add a test here.
        s = decoding.autoregressive_sample(
            pred_model, inputs=inputs, eos_id=-1, max_length=inp_len,
            temperature=0.0)
        np.testing.assert_equal(s[0], inputs[0, :inp_len])
        pred_model.state = initial_state
    gin.clear_config()  # Make sure to not affect other tests.

  def test_autoregressive_sample_reformerlm_lsh(self):
    max_len = 32

    pred_model = models.ReformerLM(
        mode='predict',
        d_model=256,
        d_ff=512,
        dropout=0.05,
        max_len=2 * max_len,
        n_heads=4,
        n_layers=3,
        ff_use_sru=0,
        d_attention_key=64,
        d_attention_value=64,
        attention_type=functools.partial(tl.LSHSelfAttention,
                                         chunk_len=16,
                                         n_hashes=2,
                                         n_buckets=[32, 32],
                                         predict_drop_len=max_len,
                                         predict_mem_len=max_len,
                                         max_length_for_buckets=1024),
        vocab_size=13,
        pos_type='fixed-base',
        pos_d_axial_embs=None,
    )

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
    pred_model.init(shape11)

    # 0w0
    inputs = np.array([[0, 3, 7, 5, 3, 2, 0]], dtype=np.int32)
    inputs = np.pad(inputs, [(0, 0), (0, max_len - inputs.shape[1])],
                    mode='constant', constant_values=0)
    s = decoding.autoregressive_sample(
        pred_model, inputs=inputs, eos_id=-1, max_length=10, temperature=0.0)

    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 10)

  def test_autoregressive_sample_reformerlm_lsh_quality(self):
    max_len = 32

    pred_model = models.ReformerLM(
        mode='predict',
        d_model=256,
        d_ff=512,
        dropout=0.05,
        max_len=2 * max_len,
        n_heads=4,
        n_layers=3,
        ff_use_sru=0,
        d_attention_key=64,
        d_attention_value=64,
        attention_type=functools.partial(tl.LSHSelfAttention,
                                         chunk_len=16,
                                         n_hashes=2,
                                         n_buckets=[32, 32],
                                         predict_drop_len=max_len,
                                         predict_mem_len=max_len,
                                         max_length_for_buckets=1024),
        vocab_size=13,
        pos_type='fixed-base',
        pos_d_axial_embs=None,
    )

    shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)

    model_path = os.path.join(
        _TESTDATA, 'reformerlm_copy_lsh_attn.pkl.gz')
    pred_model.init_from_file(model_path, weights_only=True,
                              input_signature=shape11)

    # 0w0
    inputs = np.array([[0, 3, 7, 5, 3, 2, 0]], dtype=np.int32)
    inp_len = inputs.shape[1]
    s = decoding.autoregressive_sample(
        pred_model, inputs=inputs, eos_id=-1, max_length=inp_len-2,
        temperature=0.0)

    np.testing.assert_equal(s[0], inputs[0, 1:inp_len-1])
    # pylint: enable=unreachable

  def test_autoregressive_sample_terraformer_pure_lsh(self):
    max_len = 128

    pred_model = models.ConfigurableTerraformer(
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
        encoder_attention_type=self._pure_lsh_self_attention_fn(
            n_chunks_after=1),
        encoder_decoder_attention_type=self._pure_lsh_self_attention_fn(),
        input_vocab_size=256,
        pos_axial_shape=None,
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

  def test_autoregressive_sample_terraformer_pure_lsh_attn_quality(self):
    gin.add_config_file_search_path(_CONFIG_DIR)
    max_len = 32  # 32 is the max length we trained the checkpoint for.
    test_lengths = [8, 16, 32]
    vocab_size = 13
    # The checkpoint is correct on ~90% sequences, set random seed to deflake.
    np.random.seed(0)
    for test_len in test_lengths:
      gin.clear_config()
      gin.parse_config_file('terraformer_purelsh_copy.gin')
      gin.bind_parameter('PureLSHSelfAttention.predict_mem_len', 2 * max_len)
      gin.bind_parameter('PureLSHSelfAttention.predict_drop_len', 2 * max_len)
      gin.bind_parameter('PureLSHSelfAttentionWrapper.bias', False)
      gin.bind_parameter('PureLSHSelfAttentionWrapper.num_weights', 2)

      pred_model = models.ConfigurableTerraformer(mode='predict')

      shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
      shape1l = shapes.ShapeDtype((1, max_len), dtype=np.int32)

      model_path = os.path.join(_TESTDATA, 'terraformer_purelsh_copy.pkl.gz')
      pred_model.init_from_file(model_path, weights_only=True,
                                input_signature=(shape1l, shape11))
      initial_state = pred_model.state

      for _ in range(2):  # Set low to make the test run reasonably fast.
        # Pick a length in [1, test_len] at random.
        inp_len = np.random.randint(low=1, high=test_len + 1)
        inputs = np.random.randint(low=1, high=vocab_size-1, size=(1, max_len))
        # TODO(jaszczur): properly fix padding in terraformer predict mode,
        # and add a test here.
        s = decoding.autoregressive_sample(
            pred_model, inputs=inputs, eos_id=-1, max_length=inp_len,
            temperature=0.0)

        np.testing.assert_equal(s[0], inputs[0, :inp_len])
        pred_model.state = initial_state
    gin.clear_config()  # Make sure to not affect other tests.


if __name__ == '__main__':
  config.config_with_absl()
  test.main()
