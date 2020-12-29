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

"""Perform training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from six.moves import range
import tensorflow.compat.v2 as tf

from trax.tf_numpy.examples.mnist import dataset
from trax.tf_numpy.examples.mnist import model as model_lib

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 50, 'Batch size.')
flags.DEFINE_integer('num_training_iters', 10000,
                     'Number of iterations to train for.')
flags.DEFINE_integer(
    'validation_steps', 100,
    'Validation is performed every these many training steps.')
flags.DEFINE_float('learning_rate', 5.0, 'Learning rate.')


def train(batch_size, learning_rate, num_training_iters, validation_steps):
  """Runs the training."""
  print('Loading data')
  training_data, validation_data, test_data = dataset.load()
  print('Loaded dataset with {} training, {} validation and {} test examples.'.
        format(
            len(training_data[0]), len(validation_data[0]), len(test_data[0])))

  assert len(training_data[0]) % batch_size == 0
  assert len(validation_data[0]) % batch_size == 0
  assert len(test_data[0]) % batch_size == 0

  def build_iterator(data, infinite=True):
    """Build the iterator for inputs."""
    index = 0
    size = len(data[0])
    while True:
      if index + batch_size > size:
        if infinite:
          index = 0
        else:
          return
      yield data[0][index:index + batch_size], data[1][index:index + batch_size]
      index += batch_size

  train_iter = build_iterator(training_data)
  model = model_lib.Model([30])

  for i in range(num_training_iters):
    train_x, train_y = next(train_iter)
    model.train(train_x, train_y, learning_rate)
    if (i + 1) % validation_steps == 0:
      validation_iter = build_iterator(validation_data, infinite=False)
      correct_predictions = 0
      for valid_x, valid_y in validation_iter:
        correct_predictions += model.evaluate(valid_x, valid_y)
      print('{}/{} correct validation predictions.'.format(
          correct_predictions, len(validation_data[0])))


def main(unused_argv):
  train(FLAGS.batch_size, FLAGS.learning_rate, FLAGS.num_training_iters,
        FLAGS.validation_steps)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  app.run(main)
