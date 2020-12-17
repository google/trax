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
"""Tests for supervised training: core classes and flows."""

import os
import time

from absl.testing import absltest
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import optimizers
from trax import shapes
from trax import test_utils
from trax.models import transformer
from trax.supervised import callbacks
from trax.supervised import training


class TrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_loop_no_eval_task(self):
    """Runs a training loop with no eval task(s)."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    training_session = training.Loop(model, [task])
    # Loop should initialize and run successfully, even with no eval task.
    training_session.run(n_steps=5)

  def test_loop_no_eval_task_tfnp(self):
    """Runs a training loop with no eval task(s), TFNP backend."""
    with fastmath.use_backend(fastmath.Backend.TFNP):
      model = tl.Serial(tl.Dense(1))
      task = training.TrainTask(
          _very_simple_data(), tl.L2Loss(), optimizers.Adam(.01))
      training_session = training.Loop(model, [task])
      # Loop should initialize and run successfully, even with no eval task.
      training_session.run(n_steps=5)

  def test_train_dense_layer(self):
    """Trains a very simple network on a very simple task."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    training_session = training.Loop(model, [task], eval_tasks=[eval_task],
                                     eval_at=lambda step_n: step_n % 2 == 0)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=15)
    self.assertEqual(15, training_session.step)
    training_session.run(n_steps=5)
    self.assertEqual(20, training_session.step)

  def test_loop_with_initialized_model(self):
    """Check that loop does not re-initialize an already initialized model."""
    model = tl.Serial(tl.Dense(1))
    example_data = next(_very_simple_data())
    model.init(example_data)
    w = model.weights[0][0]
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    loop = training.Loop(model, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0)
    self.assertEqual(0, loop.step)
    self.assertEqual(loop.model.weights[0][0], w)

  def test_train_save_restore_dense(self):
    """Saves and restores a checkpoint to check for equivalence."""
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path

    def _make_model_and_session():
      m = tl.Serial(tl.Dense(1))
      ts = training.Loop(m, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
      return m, ts

    model, training_session = _make_model_and_session()
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=1)
    training_session.save_checkpoint()
    model2, training_session2 = _make_model_and_session()

    x = np.ones((8, 1))
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertEqual(str(y1), str(y2))

    training_session2.run(n_steps=1)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertNotEqual(str(y1), str(y2))

  def test_train_save_restore_transformer(self):
    """Saves and restores a checkpoint to check for equivalence."""
    vocab_size = 8
    task = training.TrainTask(
        _very_simple_transformer_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_transformer_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path

    def _make_model_and_session():
      m = transformer.TransformerLM(
          vocab_size, d_model=4, d_ff=4, n_layers=1, n_heads=2, dropout=0.)
      ts = training.Loop(m, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
      return m, ts

    model, training_session = _make_model_and_session()
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=1)
    training_session.save_checkpoint()
    model2, training_session2 = _make_model_and_session()

    x = np.ones((2, 2)).astype(np.int32)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertEqual(str(y1), str(y2))

    training_session2.run(n_steps=1)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertNotEqual(str(y1), str(y2))

  def test_train_dense_layer_with_momentum(self):
    """Trains with an optimizer that has slots / requires initialization."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.Momentum(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['Momentum.L2Loss'])
    training_session = training.Loop(model, [task], eval_tasks=[eval_task],
                                     eval_at=lambda step_n: step_n % 2 == 0)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=20)
    self.assertEqual(20, training_session.step)

  def test_train_dense_layer_evals(self):
    """Trains a very simple network on a very simple task, 2 epochs."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()])
    training_session = training.Loop(model, [task], eval_tasks=[eval_task],
                                     eval_at=lambda step_n: False)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=10)
    self.assertEqual(10, training_session.step)
    training_session.run_evals()
    self.assertEqual(10, training_session.step)  # Unchanged

  def test_summaries_are_written(self):
    """Training writes down metrics when writing is turned on."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path
    training_session = training.Loop(model, [task], eval_tasks=[eval_task],
                                     eval_at=lambda step_n: step_n % 2 == 0,
                                     output_dir=tmp_dir)
    expected_train_metric_dir = os.path.join(tmp_dir, 'train')
    expected_eval_metric_dir = os.path.join(tmp_dir, 'eval')
    for directory in [expected_train_metric_dir, expected_eval_metric_dir]:
      self.assertFalse(
          os.path.isdir(directory), 'Failed for directory %s.' % directory)
    training_session.run(n_steps=15)
    time.sleep(1)  # wait for the files to be closed
    for directory in [expected_train_metric_dir, expected_eval_metric_dir]:
      self.assertTrue(
          os.path.isdir(directory), 'Failed for directory %s.' % directory)
      self.assertEqual(
          1, _count_files(directory), 'Failed for directory %s.' % directory)
    training_session.run(n_steps=5)
    time.sleep(1)  # wait for the files to be closed
    for directory in [expected_train_metric_dir, expected_eval_metric_dir]:
      self.assertEqual(
          2, _count_files(directory), 'Failed for directory %s.' % directory)

  def test_restores_step(self):
    """Training restores step from directory where it saved it."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    tmp_dir = self.create_tempdir().full_path
    loop = training.Loop(model, [task],
                         checkpoint_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
    loop.run(4)
    loop2 = training.Loop(model, [task], output_dir=tmp_dir)
    self.assertEqual(4, loop2.step)

  def test_restores_step_bfloat16(self):
    """Training restores step from directory where it saved it, w/ bfloat16."""
    model = tl.Serial(tl.Dense(1, use_bfloat16=True))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    tmp_dir = self.create_tempdir().full_path
    loop = training.Loop(model, [task],
                         checkpoint_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
    loop.run(4)
    loop2 = training.Loop(model, [task], output_dir=tmp_dir)
    self.assertEqual(4, loop2.step)
    loop2.run(2)  # check that continued training works
    self.assertEqual(6, loop2.step)

  def test_restores_step_sharded(self):
    """Training restores step from directory where it saved it, sharded."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD)
    tmp_dir = self.create_tempdir().full_path
    loop = training.Loop(model, [task],
                         checkpoint_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir, use_memory_efficient_trainer=True)
    loop.run(4)
    loop2 = training.Loop(model, [task],
                          output_dir=tmp_dir, use_memory_efficient_trainer=True)
    self.assertEqual(4, loop2.step)

  def test_restores_step_sharded_bfloat16(self):
    """Training restores step from where it saved it, sharded and bfloat16."""
    model = tl.Serial(tl.Dense(1, use_bfloat16=True))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD)
    tmp_dir = self.create_tempdir().full_path
    loop = training.Loop(model, [task],
                         checkpoint_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir, use_memory_efficient_trainer=True)
    loop.run(4)
    loop2 = training.Loop(model, [task],
                          output_dir=tmp_dir, use_memory_efficient_trainer=True)
    self.assertEqual(4, loop2.step)
    loop2.run(2)  # check that continued training works
    self.assertEqual(6, loop2.step)

  def test_trains_on_two_tasks(self):
    """Trains a very simple network on two very simple tasks."""
    model = tl.Serial(tl.Dense(3), tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(),
        tl.L2Loss(),
        optimizers.SGD(.01)
    )
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
    )
    training_session = training.Loop(
        model,
        tasks=(task, task),
        eval_tasks=(eval_task, eval_task),
        which_task=lambda step_n: step_n % 2,
    )
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=15)
    self.assertEqual(15, training_session.step)
    training_session.run(n_steps=5)
    self.assertEqual(20, training_session.step)

  def test_train_one_task_eval_two_tasks(self):
    """Trains a very simple network on one task and evaluates on two tasks."""
    model = tl.Serial(tl.Dense(3), tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(),
        tl.L2Loss(),
        optimizers.SGD(.01)
    )
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
    )
    training_session = training.Loop(
        model,
        tasks=(task,),
        eval_tasks=(eval_task, eval_task),
    )
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=5)
    self.assertEqual(5, training_session.step)

  def test_can_predict_with_trained_model(self):
    model = tl.Serial(tl.Dense(3), tl.Branch(tl.Dense(1), tl.Dense(2)))
    train_tasks, eval_tasks = [], []
    for output_dim in [1, 2]:
      # The head we select from the model: 0 for output_dim 1 and 1 for 2.
      head_index = output_dim - 1
      train_tasks.append(training.TrainTask(
          _very_simple_data(output_dim),
          tl.Serial(tl.Select([head_index], n_in=2), tl.L2Loss()),
          optimizers.SGD(.01)
      ))
      eval_tasks.append(training.EvalTask(
          _very_simple_data(output_dim),  # deliberately re-use training data
          [tl.Serial(tl.Select([head_index], n_in=2), tl.L2Loss())]
      ))
    tmp_dir = self.create_tempdir().full_path
    training_session = training.Loop(
        model,
        tasks=train_tasks,
        eval_tasks=eval_tasks,
        checkpoint_at=lambda step_n: step_n == 1,
        output_dir=tmp_dir,
        which_task=lambda step_n: step_n % 2,
    )
    training_session.run(n_steps=2)

    trained_model = training_session.eval_model
    inp = next(_very_simple_data())[0]
    out = trained_model(inp)
    self.assertEqual(
        shapes.signature(out),
        (shapes.ShapeDtype((8, 1)), shapes.ShapeDtype((8, 2))),
    )

  def test_train_memory_efficient(self):
    """Trains a large network in a memory-efficient way."""
    # This test requires > 16GB RAM, only run on TPUs. It does pass on GPU
    # and CPU when you run it locally, but it's too big for unit-testing.
    ram_limited = True  # Set to False to run this test locally.
    if fastmath.device_count() == 1 and ram_limited:
      return

    # Create the model.
    n_layers = 16  # 16 layers each 16K x 16K = 256M weights ~= 1GB, 16GB ram
    model = tl.Serial(
        tl.Embedding(9, 16*1024),
        tl.Dup(),
        [[tl.ReversibleHalfResidual(tl.Dense(16*1024)), tl.ReversibleSwap()]
         for _ in range(n_layers)],
        tl.Concatenate(),
        tl.Dense(9),
    )

    # Create inputs.
    inputs_batch = np.arange(8).reshape((2, 4))
    targets_batch = inputs_batch
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    def _data_gen():
      while True:
        yield labeled_batch

    # Run training.
    loss_layer = tl.WeightedCategoryCrossEntropy()
    task = training.TrainTask(_data_gen(), loss_layer, optimizers.Adafactor)
    eval_task = training.EvalTask(_data_gen(),
                                  [tl.WeightedCategoryCrossEntropy()])
    loop = training.Loop(model, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n == 2,
                         use_memory_efficient_trainer=True)
    self.assertEqual(0, loop.step)
    loop.run(n_steps=2)
    self.assertEqual(2, loop.step)

  def test_initializes_step_callbacks_with_loop_instance(self):
    """Runs a training loop, asserting that callbacks are initialized."""

    class ActualLoop:
      # Wrapper object to make the Loop reference mutable.
      loop = None

    class TestCallback(callbacks.TrainingStepCallback):

      def __init__(self, loop):
        super().__init__(loop)
        ActualLoop.loop = loop

      def call_at(self, step):
        return False

      def on_step_begin(self, step):
        del step

      def on_step_end(self, step):
        del step

    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01)
    )
    expected_loop = training.Loop(
        model, [task], callbacks=[TestCallback]
    )
    self.assertIs(ActualLoop.loop, expected_loop)

  def test_calls_step_callbacks(self):
    """Runs a training loop, asserting that callbacks are called."""
    call_at_steps = [1, 3, 4]
    begin_steps = []
    end_steps = []
    test_case = self

    class TestCallback(callbacks.TrainingStepCallback):

      def call_at(self, step):
        return step in call_at_steps

      def on_step_begin(self, step):
        begin_steps.append(step)

      def on_step_end(self, step):
        # Assert that on_step_begin() was called before.
        test_case.assertIn(step, begin_steps)
        end_steps.append(step)

    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01)
    )
    loop = training.Loop(model, [task], callbacks=[TestCallback])
    loop.run(n_steps=5)

    # Assert that the callback has been called at the appropriate steps.
    self.assertEqual(begin_steps, call_at_steps)
    self.assertEqual(end_steps, call_at_steps)


def _very_simple_data(output_dim=1):
  """"Returns stream of labeled data that maps small integers to constant pi."""
  inputs_batch = np.arange(8).reshape((8, 1))  # 8 items per batch
  targets_batch = np.pi * np.ones((8, output_dim))
  labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
  while True:
    yield labeled_batch


def _very_simple_transformer_data():
  """"Returns stream of labeled data that maps small integers to constant pi."""
  inputs_batch = np.ones((2, 2)).astype(np.int32)
  targets_batch = np.ones((2, 2, 8)).astype(np.int32)
  labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
  while True:
    yield labeled_batch


def _count_files(path):
  """Returns number of files in a given directory."""
  return len([filename for filename in os.listdir(path)
              if os.path.isfile(os.path.join(path, filename))])


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
