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

from absl.testing import absltest

import numpy as np

from trax import layers as tl
from trax.optimizers import momentum
from trax.optimizers import sgd
from trax.supervised import training


class TrainingTest(absltest.TestCase):

  def test_train_dense_layer(self):
    """Trains a very simple network on a very simple task."""
    model = tl.Dense(1)
    task = training.TrainTask(_very_simple_data(), tl.L2Loss(), sgd.SGD(.01))
    evals = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        names=['SGD.L2Loss'],
        eval_at=lambda step_n: step_n % 2 == 0,
        eval_N=1)
    training_session = training.Loop(model, task, evals=evals)
    self.assertIsNone(training_session.current_step)
    training_session.run(n_steps=10)
    self.assertEqual(training_session.current_step, 10)

  def test_train_dense_layer_with_momentum(self):
    """Trains with an optimizer that has slots / requires initialization."""
    model = tl.Dense(1)
    task = training.TrainTask(
        _very_simple_data(), tl.L2Loss(), momentum.Momentum(.01))
    evals = training.EvalTask(
        _very_simple_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        names=['Momentum.L2Loss'],
        eval_at=lambda step_n: step_n % 2 == 0,
        eval_N=1)
    training_session = training.Loop(model, task, evals=evals)
    self.assertIsNone(training_session.current_step)
    training_session.run(n_steps=10)
    self.assertEqual(training_session.current_step, 10)


def _very_simple_data():
  """"Returns stream of labeled data that maps small integers to constant pi."""
  inputs_batch = np.arange(7).reshape((7, 1))  # 7 items per batch
  targets_batch = np.pi * np.ones_like(inputs_batch)
  labeled_batch = (inputs_batch, targets_batch)
  while True:
    yield labeled_batch


if __name__ == '__main__':
  absltest.main()
