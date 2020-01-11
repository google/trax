# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Implementations of common recurrent neural network cells (RNNs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax import math
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import convolution
from trax.layers import core
from trax.layers import initializers
from trax.math import numpy as np


class LSTMCell(base.Layer):
  """LSTM Cell.

  For a nice overview of the motivation and (i, o, f) gates, see this tutorial:
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/

  See this paper for a description and detailed study of all gate types:
  https://arxiv.org/pdf/1503.04069.pdf
  """

  def __init__(self,
               n_units,
               forget_bias=1.0,
               kernel_initializer=initializers.GlorotUniformInitializer(),
               bias_initializer=initializers.RandomNormalInitializer(1e-6)):
    super(LSTMCell, self).__init__(n_in=2, n_out=2)
    self._n_units = n_units
    self._forget_bias = forget_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, inputs, weights):
    x, lstm_state = inputs

    # LSTM state consists of c and h.
    c, h = np.split(lstm_state, 2, axis=-1)

    # Dense layer on the concatenation of x and h.
    w, b = weights
    y = np.dot(np.concatenate([x, h], axis=-1), w) + b

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = np.split(y, 4, axis=-1)

    new_c = c * math.sigmoid(f) + math.sigmoid(i) * np.tanh(j)
    new_h = np.tanh(new_c) * math.sigmoid(o)
    return new_h, np.concatenate([new_c, new_h], axis=-1)

  def new_weights(self, input_signature):
    # LSTM state last dimension must be twice n_units.
    assert input_signature[1].shape[-1] == 2 * self._n_units
    # The dense layer input is the input and half of the lstm state.
    input_shape = input_signature[0].shape[-1] + self._n_units
    rng1, rng2 = self.new_rngs(2)
    w = self._kernel_initializer((input_shape, 4 * self._n_units), rng1)
    b = self._bias_initializer((4 * self._n_units,), rng2) + self._forget_bias
    return (w, b)


@base.layer()
def MakeZeroState(x, depth_multiplier=1, **unused_kwargs):
  """Makes zeros of shape like x but removing the length (axis 1)."""
  assert len(x.shape) == 3, 'Expecting x of shape [batch, length, depth].'
  return np.zeros((x.shape[0], depth_multiplier * x.shape[-1]),
                  dtype=np.float32)


def LSTM(n_units):
  """LSTM running on axis 1."""
  zero_state = MakeZeroState(depth_multiplier=2)  # pylint: disable=no-value-for-parameter
  return cb.Serial(
      cb.Branch([], zero_state),
      cb.Scan(LSTMCell(n_units=n_units), axis=1),
      cb.Select([0], n_in=2)  # Drop RNN state.
  )


class GRUCell(base.Layer):
  """Builds a traditional GRU cell with dense internal transformations.

  Gated Recurrent Unit paper: https://arxiv.org/abs/1412.3555
  """

  def __init__(self,
               n_units,
               forget_bias=0.0,
               kernel_initializer=initializers.RandomUniformInitializer(0.01),
               bias_initializer=initializers.RandomNormalInitializer(1e-6)):
    super(GRUCell, self).__init__(n_in=2, n_out=2)
    self._n_units = n_units
    self._forget_bias = forget_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, inputs, weights):
    x, gru_state = inputs

    # Dense layer on the concatenation of x and h.
    w1, b1, w2, b2 = weights
    y = np.dot(np.concatenate([x, gru_state], axis=-1), w1) + b1

    # Update and reset gates.
    u, r = np.split(math.sigmoid(y), 2, axis=-1)

    # Candidate.
    c = np.dot(np.concatenate([x, r * gru_state], axis=-1), w2) + b2

    new_gru_state = u * gru_state + (1 - u) * np.tanh(c)
    return new_gru_state, new_gru_state

  def new_weights(self, input_signature):
    # State last dimension must be n_units.
    assert input_signature[1].shape[-1] == self._n_units
    # The dense layer input is the input and half of the GRU state.
    input_shape = input_signature[0].shape[-1] + self._n_units
    rng1, rng2, rng3, rng4 = self.new_rngs(4)
    w1 = self._kernel_initializer((input_shape, 2 * self._n_units), rng1)
    b1 = self._bias_initializer((2 * self._n_units,), rng2) + self._forget_bias
    w2 = self._kernel_initializer((input_shape, self._n_units), rng3)
    b2 = self._bias_initializer((self._n_units,), rng4)
    return (w1, b1, w2, b2)


def GRU(n_units):
  """GRU running on axis 1."""
  zero_state = MakeZeroState(depth_multiplier=1)  # pylint: disable=no-value-for-parameter
  return cb.Serial(
      cb.Branch([], zero_state),
      cb.Scan(LSTMCell(n_units=n_units), axis=1),
      cb.Select([0], n_in=2)  # Drop RNN state.
  )


def ConvGRUCell(n_units, kernel_size=(3, 3)):
  """Builds a convolutional GRU.

  Paper: https://arxiv.org/abs/1511.06432.

  Args:
    n_units: Number of hidden units
    kernel_size: Kernel size for convolution

  Returns:
    A Stax model representing a GRU cell with convolution transforms.
  """

  def BuildConv():
    return convolution.Conv(
        filters=n_units, kernel_size=kernel_size, padding='SAME')

  return GeneralGRUCell(
      candidate_transform=BuildConv,
      memory_transform_fn=None,
      gate_nonlinearity=core.Sigmoid,
      candidate_nonlinearity=core.Tanh)


def GeneralGRUCell(candidate_transform,
                   memory_transform_fn=None,
                   gate_nonlinearity=core.Sigmoid,
                   candidate_nonlinearity=core.Tanh,
                   dropout_rate_c=0.1,
                   sigmoid_bias=0.5):
  r"""Parametrized Gated Recurrent Unit (GRU) cell construction.

  GRU update equations:
  $$ Update gate: u_t = \sigmoid(U' * s_{t-1} + B') $$
  $$ Reset gate: r_t = \sigmoid(U'' * s_{t-1} + B'') $$
  $$ Candidate memory: c_t = \tanh(U * (r_t \odot s_{t-1}) + B) $$
  $$ New State: s_t = u_t \odot s_{t-1} + (1 - u_t) \odot c_t $$

  See combinators.Gate for details on the gating function.


  Args:
    candidate_transform: Transform to apply inside the Candidate branch. Applied
      before nonlinearities.
    memory_transform_fn: Optional transformation on the memory before gating.
    gate_nonlinearity: Function to use as gate activation. Allows trying
      alternatives to Sigmoid, such as HardSigmoid.
    candidate_nonlinearity: Nonlinearity to apply after candidate branch. Allows
      trying alternatives to traditional Tanh, such as HardTanh
    dropout_rate_c: Amount of dropout on the transform (c) gate. Dropout works
      best in a GRU when applied exclusively to this branch.
    sigmoid_bias: Constant to add before sigmoid gates. Generally want to start
      off with a positive bias.

  Returns:
    A model representing a GRU cell with specified transforms.
  """
  gate_block = [  # u_t
      candidate_transform(),
      base.Fn(lambda x: x + sigmoid_bias),
      gate_nonlinearity(),
  ]
  reset_block = [  # r_t
      candidate_transform(),
      base.Fn(lambda x: x + sigmoid_bias),  # Want bias to start positive.
      gate_nonlinearity(),
  ]
  candidate_block = [
      cb.Dup(),
      reset_block,
      cb.Multiply(),  # Gate S{t-1} with sigmoid(candidate_transform(S{t-1}))
      candidate_transform(),  # Final projection + tanh to get Ct
      candidate_nonlinearity(),  # Candidate gate

      # Only apply dropout on the C gate. Paper reports 0.1 as a good default.
      core.Dropout(rate=dropout_rate_c)
  ]
  memory_transform = memory_transform_fn() if memory_transform_fn else []
  return cb.Serial(
      cb.Branch(memory_transform, gate_block, candidate_block),
      cb.Gate(),
  )


@base.layer(n_in=3, n_out=2)
def InnerSRUCell(x, **unused_kwargs):
  """The inner (non-parallel) computation of an SRU."""
  cur_x_times_one_minus_f, cur_f, cur_state = x
  res = cur_f * cur_state + cur_x_times_one_minus_f
  return res, res


def SRU(n_units, activation=None):
  """SRU (Simple Recurrent Unit) layer as in https://arxiv.org/abs/1709.02755.

  As defined in the paper:
  (1) y_t = W x_t (+ B optionally, which we do)
  (2) f_t = sigmoid(Wf x_t + bf)
  (3) r_t = sigmoid(Wr x_t + br)
  (4) c_t = f_t * c_{t-1} + (1 - f_t) * y_t
  (5) h_t = r_t * activation(c_t) + (1 - r_t) * x_t

  We assume the input is of shape [batch, length, depth] and recurrence
  happens on the length dimension. This returns a single layer. It's best
  to use at least 2, they say in the paper, except inside a Transformer.

  Args:
    n_units: output depth of the SRU layer.
    activation: Optional activation function.

  Returns:
    The SRU layer.
  """
  # pylint: disable=no-value-for-parameter
  return cb.Serial(                                    # x
      cb.Branch(core.Dense(3 * n_units), []),          # r_f_y, x
      cb.Split(n_items=3),                             # r, f, y, x
      cb.Parallel(core.Sigmoid(), core.Sigmoid()),     # r, f, y, x
      base.Fn(lambda r, f, y: (y * (1.0 - f), f, r)),  # y * (1 - f), f, r, x
      cb.Parallel([], [], cb.Branch(MakeZeroState(), [])),
      cb.Scan(InnerSRUCell(), axis=1),
      cb.Select([0], n_in=2),                          # act(c), r, x
      activation or [],
      base.Fn(lambda c, r, x: c * r + x * (1 - r))
  )
  # pylint: enable=no-value-for-parameter
