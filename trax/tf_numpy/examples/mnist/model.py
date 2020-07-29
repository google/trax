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

"""Model for training on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.python.ops import numpy_ops as np


class Model(object):
    """A simple neural network with dense layers and sigmoid non-linearity.
    The network consists of `len(hidden_layers) + 1` dense layers. The sizes of
    the hidden layers are specified by the user in `hidden_layers` and the
    network takes care of adding layers to match the input and output size.

    Attributes:
        weights: A list of 2-d float32 arrays containing the
            layer weights and biases.
        grads: A list of 2-d float32 arrays containing the last gradients
            (useful for momentum).

    Methods:
        forward: Can be used to perform a forward pass on a batch of
            flattened images. Output is returned as a batch of one-hot
            vectors of the classes.
            train: method performs a forward and backward pass and updates the
            weights and biases.
        evaluate: method can be used to evaluate the network on a batch of
            examples.
    """

    def __init__(self, hidden_layers, input_size=784,
                num_classes=10, learning_rate=0.001):
        """Initializes the neural network model.

        Args:
            hidden_layers: List of ints specifying the sizes of hidden layers.
                Could be empty.
            input_size: Length of the input array. The network receives the
            input image as a flattened 1-d array. Defaults to 784(28*28),
                the default image size for MNIST.
            num_classes: The number of output classes. Defaults to 10.
        """
        hidden_layers = [input_size] + hidden_layers + [num_classes]
        self.layers = []
        self.Layer = collections.namedtuple('Layer', ['weights', 'biases'])
        for i in range(len(hidden_layers) - 1):
            layer = self.Layer(np.random.randn(hidden_layers[i],
                        hidden_layers[i + 1]).astype('float32'),
                        np.random.randn(hidden_layers[i + 1]).astype('float32'))
            self.layers.append(layer)

        # normalization for stable training
        #self.layers = [(param - np.mean(param))/np.var(param)
        #                for param in self.layers]

        # initialization of grads
        #self.grads = [np.zeros(weight.shape) for weight in self.layers[0]]

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x: 2-d array of size batch_size x image_size.

        Returns:
            A 2-d array of size batch_size x num_classes.
        """
        len_dim_0 = x.shape[0]
        x = np.reshape(x, (len_dim_0, 784))

        for w, b in self.layers:
            x = np.tanh(np.dot(x, w) + b)
        return x

    def mean_squared_error(self, y_out, y):
        y = tf.one_hot(y, 10, axis=1)
        return np.sum((y - y_out)**2)

    def train(self, x, y, momentum=0.2):
        """Runs a single training pass.

        Args:
            x: 2-d array of size batch_size x image_size.
            y: 2-d array of size batch_size x num_classes in one-hot notation.
            learning_rate: The learning rate.

        Returns:
            The loss of this training pass
        """
        x = np.asarray(x).astype('float32')
        with tf.GradientTape() as tape:
            tape.watch(self.layers)
            loss = self.mean_squared_error(self.forward(x), y)
            grads = tape.gradient(loss, self.layers)
            # Update the weights
            for idx, grad in enumerate(grads):
                new_weights = self.layers[idx].weights - (self.learning_rate * grad.weights)
                new_biases = self.layers[idx].biases - (self.learning_rate * grad.biases)
                self.layers[idx] = self.Layer(new_weights, new_biases)

        return loss

    def evaluate(self, x, y):
        """Returns the number of correct predictions.

        Args:
            x: 2-d array of size batch_size x image_size.
            y: 2-d array of size batch_size x num_classes.

        Returns:
            A scalar, the number of correct predictions.
        """
        y_actual = np.argmax(y, axis=1)
        y_predicted = np.argmax(self.forward(x), axis=1)
        correct = int(np.sum(np.array(y_actual == y_predicted)))
        return correct
