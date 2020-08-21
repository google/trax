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
    training_session = training.Loop(model, [task], eval_tasks=[[eval_task]],
                                     eval_at=lambda step_n: step_n % 2 == 0)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=15)
    self.assertEqual(15, training_session.step)
    training_session.run(n_steps=5)
    self.assertEqual(20, training_session.step)

  def test_train_dense_layer_with_momentum(self):
    """Trains with an optimizer that has slots / requires initialization."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.Momentum(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['Momentum.L2Loss'])
    training_session = training.Loop(model, [task], eval_tasks=[[eval_task]],
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
    training_session = training.Loop(model, [task], eval_tasks=[[eval_task]],
                                     eval_at=lambda step_n: False)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=10)
    self.assertEqual(10, training_session.step)
    training_session.run_evals()
    self.assertEqual(10, training_session.step)  # Unchanged

  def test_summaries_are_written(self):
    """Training writes down metrics when writting is turned on."""
    model = tl.Serial(tl.Dense(1))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path
    training_session = training.Loop(model, [task], eval_tasks=[[eval_task]],
                                     eval_at=lambda step_n: step_n % 2 == 0,
                                     output_dir=tmp_dir)
    expected_train_metric_dir = os.path.join(tmp_dir, '0', 'train')
    expected_eval_metric_dir = os.path.join(tmp_dir, '0', 'eval')
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

  def test_trains_on_two_tasks(self):
    """Trains a very simple network on two very simple tasks."""
    model = tl.Serial(tl.Dense(3), tl.Branch(tl.Dense(1), tl.Dense(1)))
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
    )
    training_session = training.Loop(
        model,
        tasks=(task, task),
        eval_tasks=([eval_task], [eval_task]),
        which_task=lambda step_n: step_n % 2,
    )
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=15)
    self.assertEqual(15, training_session.step)
    training_session.run(n_steps=5)
    self.assertEqual(20, training_session.step)

  def test_can_predict_with_trained_model(self):
    model = tl.Serial(tl.Dense(3), tl.Branch(tl.Dense(1), tl.Dense(2)))
    tasks = tuple(
        training.TrainTask(  # pylint: disable=g-complex-comprehension
            _very_simple_data(output_dim),
            tl.L2Loss(),
            optimizers.SGD(.01),
        )
        for output_dim in (1, 2)
    )
    eval_tasks = tuple(
        [training.EvalTask(  # pylint: disable=g-complex-comprehension
            # deliberately re-using training data
            _very_simple_data(output_dim),
            [tl.L2Loss()],
        )]
        for output_dim in (1, 2)
    )
    tmp_dir = self.create_tempdir().full_path
    training_session = training.Loop(
        model,
        tasks=tasks,
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


def _very_simple_data(output_dim=1):
  """"Returns stream of labeled data that maps small integers to constant pi."""
  inputs_batch = np.arange(8).reshape((8, 1))  # 8 items per batch
  targets_batch = np.pi * np.ones((8, output_dim))
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
