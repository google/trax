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

"""Tests for trax.supervised.trainer_lib."""

import contextlib
import functools
import os
import tempfile
from absl.testing import parameterized

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
from jax.lib import xla_bridge

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import test
from tensorflow.compat.v2.io import gfile

from trax import fastmath
from trax import layers
from trax import models
from trax import optimizers as trax_opt
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.supervised import inputs as inputs_lib
from trax.supervised import lr_schedules as lr
from trax.supervised import trainer_lib
from trax.tf_numpy import numpy as tf_np



def _test_inputs(n_classes, with_weights=False, input_shape=(6, 6, 3)):
  """Make trainer_lib.inputs.Inputs."""
  batch_size = 2 * xla_bridge.device_count()

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



BACKENDS = ['jax', 'tf']


class TraxTest(test.TestCase, parameterized.TestCase):

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  # TODO(wangpeng): Remove `skipTest`'s when tf-numpy's `pmap` is in place

  def _test_train_eval_predict(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == 'tf':
      self.skipTest("tf-numpy backend does't support multi-devices yet.")
    with fastmath.use_backend(backend_name), self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2

      # Adds Dropout and BatchNorm to test state handling.
      def model_fn(mode='train'):
        return layers.Serial(
            layers.Dropout(mode=mode, rate=0.1), layers.BatchNorm(mode=mode),
            models.MLP(d_hidden=16, n_output_classes=n_classes, mode=mode))

      inputs = _test_inputs(n_classes)

      # Train and evaluate
      state = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(steps, state.step)

      # Assert 2 evaluations ran
      train_acc = state.history.get('train', 'metrics/accuracy')
      eval_acc = state.history.get('eval', 'metrics/accuracy')
      self.assertEqual(len(train_acc), len(eval_acc))
      self.assertLen(eval_acc, 2)

      # Predict with final weights
      inputs = inputs.train_stream(1)
      model = model_fn()
      weights = state.opt_state.weights[0]
      state = state.model_state[0]
      if xla_bridge.device_count() > 1:
        unreplicate = lambda x: x[0]
        weights = fastmath.nested_map(unreplicate, weights)
        state = fastmath.nested_map(unreplicate, state)
      model(next(inputs)[0], weights=weights, state=state)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict(self, backend_name):
    self._test_train_eval_predict(backend_name)

  def test_autoregressive_sample_transformerlm(self):
    model = models.TransformerLM(10, d_model=32, d_ff=64, n_layers=1,
                                 n_heads=2, mode='predict')
    model.init(shapes.ShapeDtype((1, 1), dtype=jnp.int32))
    s1 = trainer_lib.autoregressive_sample(
        model, batch_size=1, eos_id=-1, max_length=10)
    self.assertEqual(s1.shape[0], 1)
    self.assertEqual(s1.shape[1], 10)
    batch_per_device = 2 // fastmath.device_count()
    model.init(shapes.ShapeDtype((batch_per_device, 1), dtype=jnp.int32))
    s2 = trainer_lib.autoregressive_sample(
        model, batch_size=2, max_length=10)
    self.assertEqual(s2.shape[0], 2)
    self.assertLess(s2.shape[1], 11)
    model.init(shapes.ShapeDtype((1, 1), dtype=jnp.int32))
    prefix = jnp.array([[1, 2, 3]])
    s3 = trainer_lib.autoregressive_sample(model, eos_id=-1, max_length=10,
                                           batch_size=1, prefix=prefix)
    self.assertEqual(s3.shape[0], 1)
    self.assertEqual(int(s3[0][0]), 1)
    self.assertEqual(int(s3[0][1]), 2)
    self.assertEqual(int(s3[0][2]), 3)

  def test_autoregressive_sample_transformer(self):
    model = models.Transformer(10, d_model=32, d_ff=64, n_encoder_layers=1,
                               n_decoder_layers=1, n_heads=2, mode='predict')
    inputs = jnp.ones((1, 3), dtype=jnp.int32)
    model.init((shapes.signature(inputs),
                shapes.ShapeDtype((1, 1), dtype=jnp.int32)))
    s = trainer_lib.autoregressive_sample(model, inputs=inputs,
                                          eos_id=-1, max_length=10)
    self.assertEqual(s.shape[0], 1)
    self.assertEqual(s.shape[1], 10)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict_sm3(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == 'tf':
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with fastmath.use_backend(backend_name), self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      state = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          optimizer=trax_opt.SM3)

      # Assert total train steps
      self.assertEqual(steps, state.step)

      # Assert 2 evaluations ran
      train_acc = state.history.get('train', 'metrics/accuracy')
      eval_acc = state.history.get('eval', 'metrics/accuracy')
      self.assertEqual(len(train_acc), len(eval_acc))
      self.assertLen(eval_acc, 2)

      # Predict with weights loaded from file.
      inputs = inputs.train_stream(1)
      model = model_fn()
      model.init_from_file(os.path.join(output_dir, 'model.pkl.gz'))
      model(next(inputs)[0])

  @parameterized.parameters(BACKENDS)
  def test_train_restart(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == 'tf':
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with fastmath.use_backend(backend_name), self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps)

      # Restart training
      state = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=(2 * steps),
          eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(state.step, 2 * steps)

  @parameterized.parameters(BACKENDS)
  def test_train_with_weights(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == 'tf':
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with fastmath.use_backend(backend_name), self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes, with_weights=True)

      # Train and evaluate
      state = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(state.step, steps)

  @parameterized.parameters(BACKENDS)
  def test_reset_twice(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == 'tf':
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with fastmath.use_backend(backend_name), self.tmp_dir() as output_dir1, \
          self.tmp_dir() as output_dir2:
      n_classes = 4
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes)

      trainer = trainer_lib.Trainer(
          model=model_fn,
          loss_fn=layers.CrossEntropyLoss(),
          optimizer=trax_opt.SM3,
          lr_schedule=lr.multifactor(),
          inputs=inputs,
      )

      trainer.reset(output_dir1)
      trainer.evaluate(1)
      trainer.reset(output_dir2)
      trainer.evaluate(1)

  def test_tf_xla_forced_compile(self):
    # TODO(wangpeng): re-enable this test
    self.skipTest('Needs --config=cuda to pass this test')
    old_flag = fastmath.tf_math.tf_xla_forced_compile_enabled()
    fastmath.tf_math.set_tf_xla_forced_compile(True)
    self._test_train_eval_predict('tf')
    fastmath.tf_math.set_tf_xla_forced_compile(old_flag)

  def test_no_int32_or_uint32_returned(self):
    """Tests that Trainer._jit_update_fn doesn't return int32 or uint32.

    TF pins int32/uint32 tensors to CPU, which will cause XLA-forced-compiled
    computation to copy int32/uint32 outputs to CPU. This test makes sure that
    won't happen.
    """
    if xla_bridge.device_count() > 1:
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with fastmath.use_backend('tf'), self.tmp_dir() as output_dir:
      n_classes = 1001
      model_fn = functools.partial(models.Resnet50,
                                   n_output_classes=n_classes)
      inputs = _test_inputs(n_classes, input_shape=(224, 224, 3))
      trainer = trainer_lib.Trainer(
          model=model_fn,
          loss_fn=layers.CrossEntropyLoss(),
          optimizer=trax_opt.SM3,
          lr_schedule=lr.multifactor(),
          inputs=inputs,
      )
      trainer.reset(output_dir)
      trainer.train_epoch(1, 0)
      # Those are the things returned by Trainer._jit_update_fn
      arrays = (trainer._opt_state.weights, trainer._opt_state.slots,
                trainer._model_state, trainer._rngs)
      arrays = tf.nest.flatten(arrays)
      for x in arrays:
        if isinstance(x, jnp.ndarray) and (x.dtype == jnp.int32 or
                                           x.dtype == jnp.uint32):
          raise ValueError('Found an array of int32 or uint32: %s' % x)



class EpochsTest(test.TestCase):

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
  test.main()
