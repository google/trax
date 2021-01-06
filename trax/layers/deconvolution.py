# coding=utf-8
# Copyright 2021 The Trax Authors.
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
"""Trax Transpose Convolution Layers."""

import functools
import itertools
import operator

from jax import lax
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import initializers as init


class ConvTranspose(base.Layer):
  """Layer constructor function for a general Transpose Convolutional Layer."""

  def __init__(self,
               filters,
               kernel_size,
               strides=None,
               padding='VALID',
               rhs_dilation=None,
               dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
               kernel_initialzer=None,
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(ConvTranspose, self).__init__()
    self._filters = filters
    self._kernel_size = kernel_size
    self._padding = padding
    self._rhs_dilation = rhs_dilation
    self._dimension_numbers = dimension_numbers
    self._lhs_spec, self._rhs_spec, self._out_spec = dimension_numbers
    self._one = (1,) * len(kernel_size)
    self._strides = strides or self._one
    self._bias_initializer = bias_initializer
    rhs_spec = self._rhs_spec
    self._kernel_initializer = kernel_initialzer
    if kernel_initialzer is None:
      self._kernel_initializer = init.GlorotNormalInitializer(
          rhs_spec.index('O'), rhs_spec.index('I'))

  def _check_nhwc(self):
    msg = 'Deconvolutions on more than 4 dimensions only supported in NHWC.'
    assert self._lhs_spec == self._out_spec == 'NHWC', msg

  def forward(self, x):
    w, b = self.weights
    x_shape = list(x.shape)
    if len(x_shape) > 4:
      self._check_nhwc()
      new_batch_dim = functools.reduce(operator.mul, x.shape[:-3])
      x = jnp.reshape(x, [new_batch_dim] + list(x.shape[-3:]))
    res = lax.conv_transpose(x, w, self._strides, self._padding,
                             self._rhs_dilation, self._dimension_numbers) + b
    if len(x_shape) > 4:
      res = jnp.reshape(res, x_shape[:-3] + list(res.shape[-3:]))
    return res

  def _kernel_shape(self, input_shape):
    """Helper to calculate the kernel shape."""
    kernel_size_iter = iter(self._kernel_size)
    return [
        self._filters if c == 'O' else input_shape[self._lhs_spec.index('C')]
        if c == 'I' else next(kernel_size_iter) for c in self._rhs_spec
    ]

  def init_weights_and_state(self, input_signature):
    input_shape = input_signature.shape
    if len(input_shape) > 4:
      self._check_nhwc()
      new_batch_dim = functools.reduce(operator.mul, input_shape[:-3])
      input_shape = [new_batch_dim] + list(input_shape[-3:])
    kernel_shape = self._kernel_shape(input_shape)
    bias_shape = [self._filters if c == 'C' else 1 for c in self._out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(kernel_shape, rng1)
    b = self._bias_initializer(bias_shape, rng2)
    self.weights = (w, b)
