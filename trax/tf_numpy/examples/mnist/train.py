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

import tensorflow as tf
from tensorflow.python.ops import numpy_ops as np

from model import Model

FLAGS = flags.FLAGS
flags.DEFINE_integer('BATCH_SIZE', 32, 'batch size')
flags.DEFINE_float('LEARNING_RATE', 0.1, 'learning rate')
flags.DEFINE_integer('TRAINING_ITERS', 50000,
                     'training will be performed for this many iterations')
flags.DEFINE_integer('VALIDATION_STEPS', 5,
                     'validation is performed every this many training steps')

def train(batch_size, learning_rate, num_training_iters, validation_steps):
    """ training loop """
    # Loading the MNIST Dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test/255.0

    x_train = np.asarray(tf.reshape(x_train, (-1, 784))).astype(np.float32)
    y_train = np.asarray(tf.one_hot(y_train, 10)).astype(np.float32)
    x_test = np.asarray(tf.reshape(x_test, (-1, 784))).astype(np.float32)
    y_test = np.asarray(tf.one_hot(y_test, 10)).astype(np.float32)

    mnist_train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
    mnist_test = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    print('Initialized MNIST with {} training and {} test examples.'.format(
        x_train.shape[0],
        x_test.shape[0]))

    model = Model([128, 32], learning_rate=learning_rate)

    # The training loop
    loss = 0
    for i in range(num_training_iters):
        for x, y in mnist_train:
            loss += model.train(x, y)

        # Calculate and print the train and test accuracy
        if not (i + 1) % validation_steps:
            correct_train_predictions = 0
            for train_x, train_y in mnist_train:
                correct_train_predictions += model.evaluate(train_x, train_y)

            correct_test_predictions = 0
            for test_x, test_y in mnist_test:
                correct_test_predictions += model.evaluate(test_x, test_y)

            print('[{}] Loss: {}, train acc: {}, test acc: {}'.format(
                i+1,
                round(float(loss.data / validation_steps), 4),
                round(correct_train_predictions/x_train.shape[0], 4),
                round(correct_test_predictions/x_test.shape[0], 4)))

            loss = 0


def main(unused_argv):
    """main"""
    train(FLAGS.BATCH_SIZE, FLAGS.LEARNING_RATE, FLAGS.TRAINING_ITERS,
          FLAGS.VALIDATION_STEPS)


if __name__ == '__main__':
    np.random.seed(0)
    app.run(main)
