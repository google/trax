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

""" Load MNIST Data Iterator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
from tensorflow.python.ops import numpy_ops as np


class MNIST():

    def __init__(self, batch_size):

        """Initializes the MNIST dataset """
        self.batch_size = batch_size

        print('Loading data')
        mnist_train = tfds.load('mnist', split='train', shuffle_files=True, batch_size=-1)
        mnist_test = tfds.load('mnist', split='test', shuffle_files=True, batch_size=-1)


        self.mnist_train_images, self.mnist_train_labels = mnist_train['image'], mnist_train['label']

        self.mnist_test_images, self.mnist_test_labels = mnist_test['image'], mnist_test['label']

        print('Loaded dataset with {} training and {} test examples.'.format(
                self.mnist_train_images.shape[0], self.mnist_test_images.shape[0]))

        assert(self.mnist_train_images.shape[0] == self.mnist_train_labels.shape[0])
        assert(self.mnist_test_images.shape[0] == self.mnist_test_labels.shape[0])
        assert(self.mnist_train_images.shape[0] % batch_size == 0)
        assert(self.mnist_test_images.shape[0] % batch_size == 0)


    def iterator(self, split, infinite=True):
        """Build the iterator for inputs."""
        if split=='train':
            mnist_images, mnist_labels = self.mnist_train_images, self.mnist_train_labels
        elif split=='test':
            mnist_images, mnist_labels = self.mnist_test_labels, self.mnist_test_labels

        index = 0
        size = mnist_images.shape[0]
        while True:
            if index + self.batch_size > size:
                if infinite:
                    index = 0
                else:
                    return

            yield mnist_images[index:index + self.batch_size], mnist_labels[index:index + self.batch_size]
            index += self.batch_size


    def shuffle(self, data):
        zipped = zip(*data)
        random.shuffle(zipped)
        return zip(*zipped)


    def to_one_hot(self, label, num_classes=10):
        vec = np.zeros(num_classes, dtype=np.float32)
        vec[label] = 1.
        return vec
