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
from trax import optimizers as trax_opt
from trax import test_utils
from trax.data import inputs as inputs_lib
from trax.fastmath import numpy as jnp
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



BACKENDS = [fastmath.Backend.JAX, fastmath.Backend.TFNP]


class TraxTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def _test_train_eval_predict(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2

      # Adds Dropout and BatchNorm to test state handling.
      def model_fn(mode='train'):
        return tl.Serial(
            tl.Dropout(mode=mode, rate=0.1), tl.BatchNorm(mode=mode),
            models.MLP(d_hidden=16, n_output_classes=n_classes, mode=mode))

      inputs = _test_inputs(n_classes)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1)  # eval at every step.

      # Assert total train steps
      self.assertEqual(steps, loop.step)

      # Predict with final weights
      inputs = inputs.train_stream(1)
      model = model_fn()
      weights = loop.model.weights
      state = loop.model.state
      model(next(inputs)[0], weights=weights, state=state)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict(self, backend):
    self._test_train_eval_predict(backend)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict_sm3(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes)

      # Train and evaluate
      output_dir = self.create_tempdir().full_path
      loop = trainer_lib.train(
          output_dir,
          model=model_fn,
          inputs=inputs,
          steps=steps,
          eval_steps=eval_steps,
          eval_frequency=1,  # eval every step.
          optimizer=trax_opt.SM3)

      # Assert total train steps
      self.assertEqual(steps, loop.step)

      # Predict with weights loaded from file.
      inputs = inputs.train_stream(1)
      model = model_fn()
      model.init_from_file(os.path.join(output_dir, 'model.pkl.gz'))
      model(next(inputs)[0])

  @parameterized.parameters(BACKENDS)
  def test_train_restart(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
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

      # Assert total train steps - with loop we don't resume, but train for as
      # many steps as given, so: steps + 2*steps = 3*steps.
      self.assertEqual(loop.step, 3 * steps)

  @parameterized.parameters(BACKENDS)
  def test_train_with_weights(self, backend):
    with fastmath.use_backend(backend):
      # Prepare model and inputs
      n_classes = 4
      steps = 2
      eval_steps = 2
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
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
      model_fn = functools.partial(
          models.MLP, d_hidden=16, n_output_classes=n_classes)
      inputs = _test_inputs(n_classes)

      trainer = trainer_lib.Trainer(
          model=model_fn,
          loss_fn=tl.CrossEntropyLoss(),
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
    with fastmath.use_backend(fastmath.Backend.TFNP):
      n_classes = 1001
      model_fn = functools.partial(models.Resnet50,
                                   n_output_classes=n_classes)
      inputs = _test_inputs(n_classes, input_shape=(224, 224, 3))
      trainer = trainer_lib.Trainer(
          model=model_fn,
          loss_fn=tl.CrossEntropyLoss(),
          optimizer=trax_opt.SM3,
          lr_schedule=lr.multifactor(),
          inputs=inputs,
      )
      output_dir = self.create_tempdir().full_path
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
