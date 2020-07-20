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

from six.moves import range
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds

from dataset import MNIST
from tensorflow.python.ops import numpy_ops as np

BATCH_SIZE = 50
LEARNING_RATE = 5.0
NUM_TRAINING_ITERS = 10000
VALIDATION_STEPS = 100


def train(batch_size, learning_rate, num_training_iters, validation_steps):

    mnist_dataset = MNIST(batch_size)
    train_iter = mnist_dataset.iterator('train', True)

    model = model_lib.Model([30])

    for i in range(num_training_iters):
            train_x, train_y = next(train_iter)
            model.train(train_x, train_y, learning_rate)
            if not (i + 1) % validation_steps:
                validation_iter = build_iterator(validation_data, infinite=False)
                correct_predictions = 0
                for valid_x, valid_y in validation_iter:
                    correct_predictions += model.evaluate(valid_x, valid_y)
                print('{}/{} correct validation predictions.'.format(
                  correct_predictions, len(validation_data[0])))


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  train(BATCH_SIZE, LEARNING_RATE, NUM_TRAINING_ITERS, VALIDATION_STEPS)
