# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Test that the example training script works on fake data."""
import mock
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.examples.mnist import dataset
from trax.tf_numpy.examples.mnist import train


class TFNumpyMnistExampleTest(tf.test.TestCase):

  def testRuns(self):
    with mock.patch.object(dataset, 'load', new=fake_mnist_data):
      train.train(
          batch_size=1,
          learning_rate=0.1,
          num_training_iters=10,
          validation_steps=5)
      train.train(
          batch_size=2,
          learning_rate=0.1,
          num_training_iters=5,
          validation_steps=2)
      train.train(
          batch_size=10,
          learning_rate=0.1,
          num_training_iters=1,
          validation_steps=1)


def fake_mnist_data():

  def gen_examples(num_examples):
    x = np.array(
        np.random.randn(num_examples, 784), copy=False, dtype=np.float32)
    y = np.zeros((num_examples, 10), dtype=np.float32)
    y[:][0] = 1.
    return (x, y)

  return (gen_examples(100), gen_examples(10), gen_examples(10))


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
