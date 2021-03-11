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

"""Tests for trax.supervised.trainer_lib."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
from jax.lib import xla_bridge
import tensorflow.compat.v2 as tf
from trax import fastmath
from trax import layers as tl
from trax import models
from trax import shapes as trax_shapes
from trax import test_utils
from trax.data import inputs as inputs_lib
from trax.fastmath import numpy as jnp
from trax.supervised import trainer_lib
from trax.tf_numpy import extensions as npe
from trax.tf_numpy import numpy as tf_np


def _test_inputs_lm(vocab_size, seq_len, per_device_batch_size=2):
  """Make trainer_lib.inputs.Inputs for language model."""
  batch_size = per_device_batch_size * xla_bridge.device_count()

  def input_stream(_):

    def make_batch(key):
      return fastmath.random.randint(
          key, [batch_size, seq_len],
          dtype=jnp.int32,
          minval=0,
          maxval=vocab_size)

    key = fastmath.random.get_prng(0)
    while True:
      keys = fastmath.random.split(key, 3)
      key = keys[0]
      inputs = make_batch(keys[1])
      targets = make_batch(keys[2])
      yield inputs, targets

  def input_stream_masked(n_devices):
    return inputs_lib.add_loss_weights(input_stream(n_devices))

  return inputs_lib.Inputs(input_stream_masked)


def _pure_lsh_self_attention_fn(n_chunks_after=0):
  return functools.partial(
      tl.PureLSHSelfAttentionWrapper,
      attention_dropout=0.1,
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


def _mixed_lsh_self_attention_fn(n_chunks_after=0):
  return functools.partial(
      tl.PureLSHSelfAttentionWrapper,
      attention_dropout=0.1,
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
      pure_lsh_implementation=tl.MixedLSHSelfAttention,
  )


class TrainerLibNoCpuTest(parameterized.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    super().__init__(methodName)
    if npe.tpu_devices():
      # Initialize TPU for TF
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
      tf.tpu.experimental.initialize_tpu_system(resolver)

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')
    self._old_is_allow_float64 = tf_np.is_allow_float64()
    tf_np.set_allow_float64(False)

  def tearDown(self):
    tf_np.set_allow_float64(self._old_is_allow_float64)
    super().tearDown()

  def test_train_with_pure_lsh_attention(self, backend=fastmath.Backend.JAX):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      def model(mode='train'):
        return models.Reformer2(
            mode=mode,
            d_model=16,
            d_ff=16,
            n_heads=2,
            dropout=0.05,
            n_decoder_layers=1,
            n_encoder_layers=1,
            input_vocab_size=256,
            encoder_attention_type=_pure_lsh_self_attention_fn(),
            encoder_decoder_attention_type=_pure_lsh_self_attention_fn(),
        )

      max_len = 128
      inputs = _test_inputs_lm(vocab_size=256, seq_len=max_len)

      steps = 1
      eval_steps = 1

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      trainer_lib.train(
          output_dir,
          model=model,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)

      # Read checkpoint
      model_file = os.path.join(output_dir, 'model.pkl.gz')

      shape11 = trax_shapes.ShapeDtype((1, 1), dtype=jnp.int32)
      shape1l = trax_shapes.ShapeDtype((1, max_len), dtype=jnp.int32)

      model_predict = model(mode='predict')
      model_predict.init_from_file(
          model_file, weights_only=True, input_signature=(shape1l, shape11))

  def test_train_with_mixed_lsh_attention(self, backend=fastmath.Backend.JAX):
    with fastmath.use_backend(backend):
      # Prepare model and inputs

      def model(mode='train'):
        return models.Reformer2(
            mode=mode,
            d_model=16,
            d_ff=16,
            n_heads=2,
            dropout=0.05,
            n_decoder_layers=1,
            n_encoder_layers=1,
            input_vocab_size=256,
            encoder_attention_type=_mixed_lsh_self_attention_fn(),
            encoder_decoder_attention_type=_mixed_lsh_self_attention_fn(),
        )

      max_len = 128
      inputs = _test_inputs_lm(vocab_size=256, seq_len=max_len)

      steps = 1
      eval_steps = 1

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      trainer_lib.train(
          output_dir,
          model=model,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)

      # Read checkpoint
      model_file = os.path.join(output_dir, 'model.pkl.gz')

      shape11 = trax_shapes.ShapeDtype((1, 1), dtype=jnp.int32)
      shape1l = trax_shapes.ShapeDtype((1, max_len), dtype=jnp.int32)

      model_predict = model(mode='predict')
      model_predict.init_from_file(
          model_file, weights_only=True, input_signature=(shape1l, shape11))


if __name__ == '__main__':
  config.config_with_absl()
  tf.compat.v1.enable_eager_execution()
  absltest.main()
