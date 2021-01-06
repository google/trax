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
"""Test training an MNIST model 100 steps (saves time vs. 2000 steps)."""

import io
import itertools
from unittest import mock

from absl.testing import absltest

from trax import layers as tl
from trax.data import inputs
from trax.data import tf_inputs
from trax.optimizers import adam
from trax.supervised import training


class MnistTest(absltest.TestCase):

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_train_mnist_single_task(self, mock_stdout):
    """Train MNIST model a bit, to compare to other implementations."""
    mnist_model = _build_model(two_heads=False)
    (task, eval_task) = _mnist_tasks()
    training_session = training.Loop(
        mnist_model,
        tasks=[task],
        eval_tasks=[eval_task],
        eval_at=lambda step_n: step_n % 20 == 0,
    )

    training_session.run(n_steps=100)
    self.assertEqual(training_session.step, 100)

    # Assert that we reach at least 80% eval accuracy.
    self.assertGreater(_read_metric('WeightedCategoryAccuracy', mock_stdout),
                       0.8)

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_train_mnist_multitask(self, mock_stdout):
    """Train two-head MNIST model a bit, to compare to other implementations."""
    mnist_model = _build_model(two_heads=True)
    # MNIST classification task.
    (cls_task, cls_eval_task) = _mnist_tasks(head=tl.Select([0], n_in=2))
    # Auxiliary brightness prediction task.
    reg_task = training.TrainTask(
        itertools.cycle(_mnist_brightness_dataset().train_stream(1)),
        tl.Serial(tl.Select([1]), tl.L2Loss()),
        adam.Adam(0.001),
    )
    reg_eval_task = training.EvalTask(
        itertools.cycle(_mnist_brightness_dataset().eval_stream(1)),
        [tl.Serial(tl.Select([1]), tl.L2Loss())],
        n_eval_batches=1,
        metric_names=['L2'],
    )
    training_session = training.Loop(
        mnist_model,
        tasks=[cls_task, reg_task],
        eval_tasks=[cls_eval_task, reg_eval_task],
        eval_at=lambda step_n: step_n % 20 == 0,
        which_task=lambda step_n: step_n % 2,
    )

    training_session.run(n_steps=100)
    self.assertEqual(training_session.step, 100)

    # Assert that we reach at least 80% eval accuracy on MNIST.
    self.assertGreater(_read_metric('WeightedCategoryAccuracy', mock_stdout),
                       0.8)
    # Assert that we get below 0.03 brightness prediction error.
    self.assertLess(_read_metric('L2', mock_stdout), 0.03)


def _build_model(two_heads):
  cls_head = tl.Dense(10)
  if two_heads:
    reg_head = tl.Dense(1)
    heads = tl.Branch(cls_head, reg_head)
  else:
    heads = cls_head
  return tl.Serial(
      tl.Fn('ScaleInput', lambda x: x / 255),
      tl.Flatten(),
      tl.Dense(512),
      tl.Relu(),
      tl.Dense(512),
      tl.Relu(),
      heads,
  )


def _mnist_dataset():
  """Loads (and caches) the standard MNIST data set."""
  streams = tf_inputs.data_streams('mnist')
  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=256,
                        eval_batch_size=256)


def _mnist_brightness_dataset():
  """Loads (and caches) a MNIST mean brightness data set."""
  def preprocess_stream(stream):
    def new_stream():
      for (image, _) in stream():
        yield (image, (image / 255).mean()[None])
    return new_stream

  streams = tuple(map(preprocess_stream, tf_inputs.data_streams('mnist')))
  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=256,
                        eval_batch_size=256)


def _mnist_tasks(head=None):
  """Creates MNIST training and evaluation tasks.

  Args:
    head: Adaptor layer to put before loss and accuracy layers in the tasks.

  Returns:
    A pair (train_task, eval_task) consisting of the MNIST training task and the
    MNIST evaluation task using cross-entropy as loss and accuracy as metric.
  """
  loss = tl.WeightedCategoryCrossEntropy()
  accuracy = tl.WeightedCategoryAccuracy()
  if head is not None:
    loss = tl.Serial(head, loss)
    accuracy = tl.Serial(head, accuracy)
  task = training.TrainTask(
      itertools.cycle(_mnist_dataset().train_stream(1)),
      loss,
      adam.Adam(0.001),
  )
  eval_task = training.EvalTask(
      itertools.cycle(_mnist_dataset().eval_stream(1)),
      [loss, accuracy],
      n_eval_batches=10,
      metric_names=['CrossEntropy', 'WeightedCategoryAccuracy'],
  )
  return (task, eval_task)


def _read_metric(metric_name, stdout):
  log = stdout.getvalue()
  metric_log = [line for line in log.split('\n') if metric_name in line][-1]
  return float(metric_log.strip().split(' ')[-1])


if __name__ == '__main__':
  absltest.main()
