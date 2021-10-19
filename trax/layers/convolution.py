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
"""Trax convolution layers."""

import functools
import itertools
import operator

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import initializers as init


class Conv(base.Layer):
  """Layer constructor function for a general convolution layer."""

  def __init__(self, filters, kernel_size, strides=None, padding='VALID',
               dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
               kernel_initializer=None,
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True):
    super().__init__()
    self._filters = filters
    self._kernel_size = kernel_size
    self._padding = padding
    self._dimension_numbers = dimension_numbers
    self._lhs_spec, self._rhs_spec, self._out_spec = dimension_numbers
    self._one = (1,) * len(kernel_size)
    self._strides = strides or self._one
    self._bias_initializer = bias_initializer
    self._use_bias = use_bias
    rhs_spec = self._rhs_spec
    self._kernel_initializer = kernel_initializer
    if kernel_initializer is None:
      self._kernel_initializer = init.GlorotNormalInitializer(
          rhs_spec.index('O'), rhs_spec.index('I'))

  def _check_nhwc(self):
    msg = 'Convolutions on more than 4 dimensions only supported in NHWC.'
    assert self._lhs_spec == self._out_spec == 'NHWC', msg

  def forward(self, x):
    if self._use_bias:
      w, b = self.weights
    else:
      w = self.weights
    x_shape = list(x.shape)
    if len(x_shape) > 4:
      self._check_nhwc()
      new_batch_dim = functools.reduce(operator.mul, x_shape[:-3])
      x = jnp.reshape(x, [new_batch_dim] + x_shape[-3:])
    res = fastmath.conv(
        x, w, self._strides, self._padding, self._dimension_numbers,
        self._one)
    if self._use_bias:
      res = res + b
    if len(x_shape) > 4:
      res = jnp.reshape(res, x_shape[:-3] + list(res.shape[-3:]))
    return res

  def _kernel_shape(self, input_shape):
    """Helper to calculate the kernel shape."""
    kernel_size_iter = iter(self._kernel_size)
    return [self._filters if c == 'O' else
            input_shape[self._lhs_spec.index('C')] if c == 'I' else
            next(kernel_size_iter) for c in self._rhs_spec]

  def init_weights_and_state(self, input_signature):
    input_shape = input_signature.shape
    if len(input_shape) > 4:
      self._check_nhwc()
      new_batch_dim = functools.reduce(operator.mul, input_shape[:-3])
      input_shape = [new_batch_dim] + list(input_shape[-3:])
    kernel_shape = self._kernel_shape(input_shape)
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(kernel_shape, rng1)
    if self._use_bias:
      bias_shape = [self._filters if c == 'C' else 1 for c in self._out_spec]
      bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
      b = self._bias_initializer(bias_shape, rng2)
      self.weights = (w, b)
    else:
      self.weights = w


class CausalConv(Conv):
  """Causal (masked) convolution for [batch x time x depth] sequences.

  Maintains causality along time axis. Used in language modeling tasks.
  """

  def __init__(self,
               filters,
               kernel_width=3,
               kernel_initializer=None,
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True):
    super().__init__(
        filters=filters,
        kernel_size=(kernel_width,),
        strides=None,
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        use_bias=use_bias)

  def forward(self, x):
    assert self._padding == 'VALID'
    # Left pad with 0s. Applying an unmasked valid convolution on top of this
    # yields a causal convolution.
    # TODO(ddohan): Support strided and dilated convolutions.
    rate = 1
    effective_kernel_size = int((self._kernel_size[0] - 1) * rate + 1)
    pad = effective_kernel_size - 1
    x_leftpad = (
        jnp.pad(x, pad_width=[[0, 0], [pad, 0], [0, 0]], mode='constant'))
    return super().forward(x_leftpad)


def Conv1d(filters, kernel_size, stride=1, padding='VALID',
           kernel_initializer=None,
           bias_initializer=init.RandomNormalInitializer(1e-6),
           use_bias=True):
  return Conv(filters, (kernel_size,), strides=(stride,), padding=padding,
              dimension_numbers=('NWC', 'WIO', 'NWC'),
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              use_bias=use_bias)


def _zero_pad(x, pad, axis):  # pylint: disable = invalid-name
  """Helper for jnp.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return jnp.pad(x, pad_widths, mode='constant')


# @assert_shape('bld->bld')
class CausalDepthwiseConv(base.Layer):
  """A causal depthwise convolution layer."""

  def __init__(self,
               kernel_size=3,
               kernel_initializer=init.GlorotUniformInitializer(),
               use_bfloat16=False):
    """Returns a causal depthwise convolution layer."""
    super().__init__(n_in=1, n_out=1)
    self._kernel_size = kernel_size
    self._kernel_initializer = kernel_initializer
    self._use_bfloat16 = use_bfloat16

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    w = self.weights
    res = x * w[0, :][None, None, :]
    for i in range(1, self._kernel_size):
      x = _zero_pad(x, (1, 0), 1)
      x = x[:, :-1, :]
      res += x * w[i, :][None, None, :]
    return res

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    shape_w = (self._kernel_size, input_signature.shape[-1])
    rng_w, _ = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(shape_w, rng_w)
    if self._use_bfloat16:
      w = w.astype(jnp.bfloat16)
    self.weights = w
