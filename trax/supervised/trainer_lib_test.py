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
import jax
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import tensorflow.compat.v2 as tf
from trax import fastmath
from trax import layers as tl
from trax import models
from trax import optimizers as trax_opt
from trax import shapes as trax_shapes
from trax import test_utils
from trax.data import inputs as inputs_lib
from trax.fastmath import numpy as jnp
from trax.supervised import lr_schedules as lr
from trax.supervised import trainer_lib
from trax.tf_numpy import extensions as npe
from trax.tf_numpy import numpy as tf_np



def _test_inputs(n_classes, with_weights=False, input_shape=(6, 6, 3)):
  """Make trainer_lib.inputs.Inputs."""
  batch_size = 2 * jax.device_count()

  def input_stream(n_devices):
    del n_devices
    key = fastmath.random.get_prng(0)
    while True:
      keys = fastmath.random.split(key, 4)
      key = keys[0]
      inputs = fastmath.random.uniform(
          keys[1], [batch_size] + list(input_shape))
      targets = fastmath.random.randint(
          keys[2], [batch_size], dtype=jnp.int32, minval=0, maxval=n_classes)
      weights = fastmath.random.uniform(keys[3], [batch_size])
      if with_weights:
        yield inputs, targets, weights
      else:
        yield inputs, targets

  def input_stream_masked(n_devices):
    return inputs_lib.add_loss_weights(input_stream(n_devices))

  return inputs_lib.Inputs(input_stream_masked)


def _test_inputs_lm(vocab_size, seq_len, per_device_batch_size=2):
  """Make trainer_lib.inputs.Inputs for language model."""
  batch_size = per_device_batch_size * jax.device_count()

  def input_stream(_):
    def make_batch(key):
      return fastmath.random.randint(
          key, [batch_size, seq_len], dtype=jnp.int32, minval=0,
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



BACKENDS = [fastmath.Backend.JAX]
BACKENDS_AND_CONFIGS = [(fastmath.Backend.JAX, [('Simple', None)])]


def short_name(b):
  if b == fastmath.Backend.JAX:
    return 'jax'
  else:
    return 'tf'


def opt_name(opt):
  if opt is None:
    return 'None'
  return opt.__name__


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


class TraxTest(parameterized.TestCase):

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

  def _test_train_eval_predict(self, backend, model_name='Simple',
                               optimizer=None):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      steps = 2
      eval_steps = 2

      if model_name == 'Simple':
        n_classes = 4
        # Adds Dropout and BatchNorm to test state handling.
        def model_fn(mode='train'):
          return tl.Serial(
              tl.Dropout(mode=mode, rate=0.1),
              tl.BatchNorm(mode=mode),
              models.MLP(layer_widths=(16, 16, n_classes), mode=mode))
        inputs = _test_inputs(n_classes)
        n_in = 1
      elif model_name == 'Resnet50':
        n_classes = 4
        model_fn = models.Resnet50
        inputs = _test_inputs(n_classes, input_shape=(224, 224, 3))
        n_in = 1
      elif model_name == 'Transformer':
        vocab_size = 32
        seq_len = 16
        inputs = _test_inputs_lm(vocab_size, seq_len)
        model_fn = functools.partial(
            models.Transformer,
            input_vocab_size=vocab_size)
        n_in = 2
      else:
        raise ValueError('Unrecognized model name: ' + model_name)

      kwargs = {}
      if optimizer is not None:
        kwargs['optimizer'] = optimizer

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1,  # eval at every step.
          **kwargs)

      # Assert total train steps
      self.assertEqual(steps, loop.step)

      inputs = inputs.train_stream(1)

      # Predict with final weights
      model = model_fn()
      weights = loop.model.weights
      state = loop.model.state
      model(next(inputs)[:n_in], weights=weights, state=state)

      # Predict with weights loaded from file.
      model = model_fn()
      model.init_from_file(os.path.join(output_dir, 'model.pkl.gz'))
      model(next(inputs)[:n_in])

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (short_name(backend), model_name, opt_name(opt)),  # pylint: disable=g-complex-comprehension
       backend, model_name, opt)
      for backend, configs in BACKENDS_AND_CONFIGS
      for model_name, opt in configs)
  def test_train_eval_predict(self, backend, model_name, opt):
    self._test_train_eval_predict(backend, model_name, opt)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict_sm3(self, backend):
    self._test_train_eval_predict(backend, 'Simple', trax_opt.SM3)

  @parameterized.parameters(BACKENDS)
  def test_train_restart(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)

      # Restart training
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=(2 * steps),
          eval_steps=eval_steps,
          eval_frequency=1)

      # Assert total train steps
      self.assertEqual(loop.step, 2 * steps)

  @parameterized.parameters(BACKENDS)
  def test_train_permanent_checkpoints(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 5
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path

      # Steps 1 -> 5
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1,
          permanent_checkpoint_frequency=2)

      # Steps 6 -> 10
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=(2 * steps),
          eval_steps=eval_steps,
          eval_frequency=1,
          permanent_checkpoints_at=[7, 8, 10])

      path = os.path.join(output_dir, 'model.pkl.gz')
      self.assertTrue(tf.io.gfile.exists(path))

      for step in range(11):
        filename = 'model_{}.pkl.gz'.format(step)
        path = os.path.join(output_dir, filename)
        if step in [1, 2, 4, 7, 8, 10]:
          self.assertTrue(tf.io.gfile.exists(path),
                          msg='No model for step: {} in dir {}.'.format(
                              step, tf.io.gfile.listdir(output_dir)))
        else:
          self.assertFalse(tf.io.gfile.exists(path),
                           msg='Model for step: {} in dir {}.'.format(
                               step, tf.io.gfile.listdir(output_dir)))

      # Assert total train steps
      self.assertEqual(loop.step, 10)

  @parameterized.parameters(BACKENDS)
  def test_train_restart_with_same_steps(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)

      # Restart training
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)

      # Assert total train steps
      self.assertEqual(loop.step, steps)

  def test_train_with_pure_lsh_attention(self, backend=fastmath.Backend.JAX):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      def model(mode='train'):
        return models.ConfigurableTerraformer(
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
        return models.ConfigurableTerraformer(
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
      model_predict.init_from_file(model_file, weights_only=True,
                                   input_signature=(shape1l, shape11))

  @parameterized.parameters(BACKENDS)
  def test_train_fills_in_missing_eval_metrics(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP, layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes)
      additional_eval_stream = trainer_lib.NamedStream(
          # deliberately duplicating eval data
          stream=inputs.eval_stream(1),
          name='additional_eval_task')

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1,
          additional_eval_streams=[additional_eval_stream])

      self.assertLen(loop.eval_tasks, 2)
      eval_task_1, eval_task_2 = loop.eval_tasks
      self.assertCountEqual(eval_task_1.metrics, eval_task_2.metrics)
      self.assertCountEqual(eval_task_1.metric_names, eval_task_2.metric_names)

  @parameterized.named_parameters(
      ('_%s' % short_name(backend), backend)
      for backend in BACKENDS)
  def test_train_with_weights(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes, with_weights=True)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      state = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(state.step, steps)

  @parameterized.parameters(BACKENDS)
  def test_reset_twice(self, backend):
    with fastmath.use_backend(backend):
      n_classes = 4
      model_fn = functools.partial(models.MLP,
                                   layer_widths=(16, 16, n_classes))
      inputs = _test_inputs(n_classes)

      trainer = trainer_lib.Trainer(
          model=model_fn,
          loss_fn=tl.WeightedCategoryCrossEntropy(),
          optimizer=trax_opt.SM3,
          lr_schedule=lr.multifactor(),
          inputs=inputs,
      )

      output_dir1 = self.create_tempdir(name='output_dir1').full_path
      trainer.reset(output_dir1)
      trainer.evaluate(1)
      output_dir2 = self.create_tempdir(name='output_dir2').full_path
      trainer.reset(output_dir2)
      trainer.evaluate(1)

  def test_tf_xla_forced_compile(self):
    # TODO(wangpeng): re-enable this test
    self.skipTest('Needs --config=cuda to pass this test')
    old_flag = fastmath.tf.tf_xla_forced_compile_enabled()
    fastmath.tf.set_tf_xla_forced_compile(True)
    self._test_train_eval_predict('tf')
    fastmath.tf.set_tf_xla_forced_compile(old_flag)



class EpochsTest(absltest.TestCase):

  def test_cuts_epoch_when_total_steps_reached(self):
    epoch_steps = trainer_lib.epochs(
        total_steps=5, steps_to_skip=0, epoch_steps=[1, 2, 3])
    self.assertEqual(list(epoch_steps), [1, 2, 2])

  def test_skips_full_epoch(self):
    epoch_steps = trainer_lib.epochs(
        total_steps=4, steps_to_skip=2, epoch_steps=[2, 2])
    self.assertEqual(list(epoch_steps), [2])

  def test_skips_part_of_epoch(self):
    epoch_steps = trainer_lib.epochs(
        total_steps=4, steps_to_skip=1, epoch_steps=[2, 2])
    self.assertEqual(list(epoch_steps), [1, 2])


if __name__ == '__main__':
  config.config_with_absl()
  tf.compat.v1.enable_eager_execution()
  absltest.main()
