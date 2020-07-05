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
"""Test training an MNIST model 1000 steps (saves time vs. 2000 steps)."""

import itertools

from absl.testing import absltest

from trax import layers as tl
from trax.optimizers import adafactor
from trax.supervised import inputs
from trax.supervised import tf_inputs
from trax.supervised import training


class MnistTest(absltest.TestCase):

  def test_train_mnist(self):
    """Train MNIST model (almost) fully, to compare to other implementations.

    Evals for cross-entropy loss and accuracy are run every 50 steps;
    their values are visible in the test log.
    """
    mnist_model = tl.Serial(
        tl.Flatten(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(10),
        tl.LogSoftmax(),
    )
    task = training.TrainTask(
        itertools.cycle(_mnist_dataset().train_stream(1)),
        tl.CrossEntropyLoss(),
        adafactor.Adafactor(.02))
    eval_task = training.EvalTask(
        itertools.cycle(_mnist_dataset().eval_stream(1)),
        [tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=10)

    training_session = training.Loop(mnist_model, task, eval_task=eval_task,
                                     eval_at=lambda step_n: step_n % 50 == 0)

    training_session.run(n_steps=1000)
    self.assertEqual(training_session.current_step, 1000)


def _mnist_dataset():
  """Loads (and caches) the standard MNIST data set."""
  streams = tf_inputs.data_streams('mnist')
  return inputs.batcher(streams, variable_shapes=False,
                        batch_size_per_device=256,
                        eval_batch_size=256)


if __name__ == '__main__':
  absltest.main()
