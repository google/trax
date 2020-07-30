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

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import numpy_ops as np

# current convention is to import original numpy as onp
import numpy as onp

class MNIST():
    """A simple wrapper around the MNIST tensorflow dataset to make it into
    a NumPy iterable object.

    Attributes:
        batch_size: an int defining how many examples a single batch contains

    Methods:
        iterator: can be used to create an iterator over the dataset.
        _shuffle: shuffles the data points and labels, and returns the result.
    """

    def __init__(self, batch_size):
        """Initializes the MNIST dataset in a NumPy format

        Args:
            batch_size: int defining how many examples a single batch contains.
        """
        self.mnist_train = tfds.load('mnist', split='train',
                                     shuffle_files=True, batch_size=-1)
        self.mnist_test = tfds.load('mnist', split='test',
                                     shuffle_files=True, batch_size=-1)
        self.batch_size = batch_size
        self.train_len = self.mnist_train['image'].shape[0]
        self.test_len = self.mnist_test['image'].shape[0]

        print('Initialized MNIST with {} training and {} test examples.'.format(
            self.train_len,
            self.test_len))

    def iterator(self, split, infinite=True):
        """Build and returns an iterator of the data.

        Args:
            split: a string that is either 'train' or 'test'.
            infinite: boolean that determines whether iterable object is finite.

        Returns:
            An iterable object over the MNIST dataset.
        """
        if split == 'train':
            mnist_images = self.mnist_train['image']
            mnist_labels = self.mnist_train['label']
        elif split == 'test':
            mnist_images = self.mnist_test['image']
            mnist_labels = self.mnist_test['label']
        else:
            raise ValueError("split has to be either 'train' or 'test'")

        len_dim_0 = mnist_images.shape[0]
        mnist_images = np.reshape(mnist_images, (len_dim_0, 784))
        mnist_labels = np.asarray(tf.one_hot(mnist_labels, 10))

        index = 0
        size = mnist_images.shape[0]
        while True:
            if index + self.batch_size > size:
                if infinite:
                    index = 0
                    mnist_images,
                    mnist_labels = self._shuffle(mnist_images, mnist_labels)
                else:
                    return

            yield (mnist_images[index:index + self.batch_size],
                   mnist_labels[index:index + self.batch_size])
            index += self.batch_size

    def _shuffle(self, data, labels):
        """ shuffles datapoints and labels, returns the resulting dataset. """
        idx = onp.random.permutation(len(data))
        data, labels = data[idx], labels[idx]
        return data, labels
