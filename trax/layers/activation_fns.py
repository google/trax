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
"""Layers that compute activation functions.

An activation layer computes element-wise a nonlinear function of the preceding
layer's output. Historically, an activation function was considered part of
each node in each layer of the neural network. Trax follows the common current
practice of separating the activation function as its own layer, which enables
easier experimentation across different activation functions.
"""

from trax import math
from trax.layers import base
from trax.layers.base import Fn
from trax.math import numpy as np


def Relu():
  return Fn('Relu', lambda x: np.maximum(x, np.zeros_like(x)))


def ParametricRelu(a=1.):
  return Fn('ParametricRelu', lambda x: np.maximum(a * x, np.zeros_like(x)))


def LeakyRelu(a=0.01):
  return Fn('LeakyRelu', lambda x: np.where(x >= 0, x, a * x))


def Elu(a=1.):
  return Fn('Elu', lambda x: np.where(x > 0, x, a * np.expm1(x)))


def Selu(alpha=1.6732632423543772848170429916717,
         lmbda=1.0507009873554804934193349852946):
  return Fn('Selu', lambda x: lmbda * np.where(x > 0, x, alpha * np.expm1(x)))


def Gelu():
  return Fn('Gelu', lambda x: x * 0.5 * (1.0 + math.erf(x / np.sqrt(2.0))))


def FastGelu():
  def f(x):  # pylint: disable=invalid-name
    return 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
  return Fn('FastGelu', f)


# pylint: disable=unnecessary-lambda
def Sigmoid():
  return Fn('Sigmoid', lambda x: math.expit(x))


def Tanh():
  return Fn('Tanh', lambda x: np.tanh(x))
# pylint: enable=unnecessary-lambda


def HardSigmoid():
  """Computes a linear approximation to sigmoid."""
  return Fn('HardSigmoid', lambda x: np.maximum(0, np.minimum(1, (1 + x))))


def HardTanh():
  """Computes a linear approximation to tanh."""
  return Fn('HardTanh', lambda x: np.maximum(-1, np.minimum(1, x)))


def Softplus():
  return Fn('Softplus', lambda x: np.logaddexp(x, 0.))


class ThresholdedLinearUnit(base.Layer):
  """Thresholded Linear Unit, c.f. https://arxiv.org/pdf/1911.09737.pdf ."""

  def new_weights(self, input_signature):
    del input_signature
    return (np.zeros((), dtype=np.float32),)

  def forward(self, inputs, weights):
    threshold = weights[0]
    return np.maximum(inputs, threshold)
