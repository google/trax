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

from tensorflow.python.ops import numpy_ops as np

from dataset import MNIST
from model import Model

FLAGS = flags.FLAGS

flags.DEFINE_integer('BATCH_SIZE', 100, 'batch size')
flags.DEFINE_float('LEARNING_RATE', 9, 'learning rate')
flags.DEFINE_integer('TRAINING_ITERS', 50000,
                     'training will be performed for this many iterations')
flags.DEFINE_integer('VALIDATION_STEPS', 100,
                     'validation is performed every this many training steps')

def train(batch_size, learning_rate, num_training_iters, validation_steps):
    """ training loop """
    mnist_dataset = MNIST(batch_size)
    train_iter = mnist_dataset.iterator('train', infinite=True)

    model = Model([728, 728], learning_rate=learning_rate)

    for i in range(num_training_iters):
        train_x, train_y = next(train_iter)
        loss = model.train(train_x, train_y)

        # Calculate and print the train and test accuracy
        if not (i + 1) % validation_steps:
            training_iter = mnist_dataset.iterator('train', infinite=False)
            validation_iter = mnist_dataset.iterator('test', infinite=False)

            correct_train_predictions = 0
            for train_x, train_y in training_iter:
                correct_train_predictions += model.evaluate(train_x, train_y)

            correct_val_predictions = 0
            for valid_x, valid_y in validation_iter:
                correct_val_predictions += model.evaluate(valid_x, valid_y)

            print('[{}] Loss: {}, train acc: {}, test acc: {}'.format(
                i+1,
                loss.data,
                round(correct_train_predictions/mnist_dataset.train_len, 4),
                round(correct_val_predictions/mnist_dataset.val_len, 4)))


def main(unused_argv):
    """main"""
    train(FLAGS.BATCH_SIZE, FLAGS.LEARNING_RATE, FLAGS.TRAINING_ITERS,
          FLAGS.VALIDATION_STEPS)


if __name__ == '__main__':
    np.random.seed(0)
    app.run(main)
