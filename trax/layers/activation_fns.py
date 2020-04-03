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
from trax.math import numpy as np


@base.layer()
def Relu(x, **unused_kwargs):
  return np.maximum(x, np.zeros_like(x))


@base.layer()
def ParametricRelu(x, a=1., **unused_kwargs):
  return np.maximum(a * x, np.zeros_like(x))


@base.layer()
def LeakyRelu(x, a=0.01, **unused_kwargs):
  return np.where(x >= 0, x, a * x)


@base.layer()
def Elu(x, a=1., **unused_kwargs):
  return np.where(x > 0, x, a * np.expm1(x))


@base.layer()
def Selu(x,
         alpha=1.6732632423543772848170429916717,
         lmbda=1.0507009873554804934193349852946):
  return lmbda * np.where(x > 0, x, alpha * np.expm1(x))


@base.layer()
def Gelu(x, **unused_kwargs):
  return x * 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


@base.layer()
def FastGelu(x, **unused_kwargs):
  return 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


@base.layer()
def Sigmoid(x, **unused_kwargs):
  return math.expit(x)


@base.layer()
def Tanh(x, **unused_kwargs):
  return np.tanh(x)


@base.layer()
def HardSigmoid(x, **unused_kwargs):
  """Computes a linear approximation to sigmoid."""
  return np.maximum(0, np.minimum(1, (1 + x)))


@base.layer()
def HardTanh(x, **unused_kwargs):
  """Computes a linear approximation to tanh."""
  return np.maximum(-1, np.minimum(1, x))


@base.layer()
def Softplus(x, **unused_kwargs):
  return np.logaddexp(x, 0.)


class ThresholdedLinearUnit(base.Layer):
  """Thresholded Linear Unit, c.f. https://arxiv.org/pdf/1911.09737.pdf ."""

  def new_weights(self, input_signature):
    del input_signature
    return (np.zeros((), dtype=np.float32),)

  def forward(self, inputs, weights):
    threshold = weights[0]
    return np.maximum(inputs, threshold)
