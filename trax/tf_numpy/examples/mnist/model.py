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

"""Model for training on MNIST data."""
from numpy import float32
from numpy import int32

import tensorflow.compat.v2 as tf

from trax.tf_numpy import numpy as np


class Model(object):
  """A simple neural network with dense layers and sigmoid non-linearity.

  The network consists of `len(hidden_layers) + 1` dense layers. The sizes of
  the hidden layers are specified by the user in `hidden_layers` and the
  network takes care of adding layers to match the input and output size.

  Attributes:
    weights: A list of 2-d float32 arrays containing the layer weights.
    biases: A list of 2-d float32 arrays containing the layer biases.

  Methods:
    forward: Can be used to perform a forward pass on a batch of
      flattened images. Output is returned as a batch of one-hot vectors of the
      classes.
    train: method performs a forward and backward pass and updates the
      weights and biases.
    evaluate: method can be used to evaluate the network on a batch of
      examples.
  """

  def __init__(self, hidden_layers, input_size=784, num_classes=10):
    """Initializes the neural network.

    Args:
      hidden_layers: List of ints specifying the sizes of hidden layers. Could
        be empty.
      input_size: Length of the input array. The network receives the input
        image as a flattened 1-d array. Defaults to 784(28*28), the default
        image size for MNIST.
      num_classes: The number of output classes. Defaults to 10.
    """
    hidden_layers = [input_size] + hidden_layers + [num_classes]
    self.weights = []
    self.biases = []
    for i in range(len(hidden_layers) - 1):
      # TODO(srbs): This is manually cast to float32 to avoid the cast in
      # np.dot since backprop fails for tf.cast op.
      self.weights.append(
          np.array(
              np.random.randn(hidden_layers[i + 1], hidden_layers[i]),
              copy=False,
              dtype=float32))
      self.biases.append(
          np.array(
              np.random.randn(hidden_layers[i + 1]), copy=False, dtype=float32))

  def forward(self, x):
    """Performs the forward pass.

    Args:
      x: 2-d array of size batch_size x image_size.

    Returns:
      A 2-d array of size batch_size x num_classes.
    """

    def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))

    for w, b in zip(self.weights, self.biases):
      x = sigmoid(np.dot(w, x.T).T + b)
    return x

  def train(self, x, y, learning_rate=0.01):
    """Runs a single training pass.

    Args:
      x: 2-d array of size batch_size x image_size.
      y: 2-d array of size batch_size x num_classes in one-hot notation.
      learning_rate: The learning rate.
    """
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    def mean_squared_error(x, y):
      diff = x - y
      return np.sum(diff * diff) / len(x)

    wb_tensors = self.weights + self.biases
    with tf.GradientTape() as g:
      g.watch(wb_tensors)
      loss = mean_squared_error(self.forward(x), y)
    gradients = g.gradient(loss, wb_tensors)
    gradients = [np.asarray(grad) for grad in gradients]

    new_weights_and_biases = []
    for v, dv in zip(self.weights + self.biases, gradients):
      new_weights_and_biases.append(v - learning_rate * dv)

    total_len = len(new_weights_and_biases)
    self.weights = new_weights_and_biases[:total_len // 2]
    self.biases = new_weights_and_biases[total_len // 2:]

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
    return int(
        np.sum(np.array(y_actual == y_predicted, copy=False, dtype=int32)))
