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
"""Implementations of common recurrent neural network cells (RNNs)."""

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import activation_fns
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import convolution
from trax.layers import core
from trax.layers import initializers


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
    super().__init__(n_in=2, n_out=2)
    self._n_units = n_units
    self._forget_bias = forget_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, inputs):
    x, lstm_state = inputs

    # LSTM state consists of c and h.
    c, h = jnp.split(lstm_state, 2, axis=-1)

    # Dense layer on the concatenation of x and h.
    w, b = self.weights
    y = jnp.dot(jnp.concatenate([x, h], axis=-1), w) + b

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = jnp.split(y, 4, axis=-1)

    new_c = c * fastmath.sigmoid(f) + fastmath.sigmoid(i) * jnp.tanh(j)
    new_h = jnp.tanh(new_c) * fastmath.sigmoid(o)
    return new_h, jnp.concatenate([new_c, new_h], axis=-1)

  def init_weights_and_state(self, input_signature):
    # LSTM state last dimension must be twice n_units.
    if input_signature[1].shape[-1] != 2 * self._n_units:
      raise ValueError(
          f'Last dimension of state (shape: {str(input_signature[1].shape)}) '
          f'must be equal to 2*n_units ({2 * self._n_units})')
    # The dense layer input is the input and half of the lstm state.
    input_shape = input_signature[0].shape[-1] + self._n_units
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer((input_shape, 4 * self._n_units), rng1)
    b = self._bias_initializer((4 * self._n_units,), rng2) + self._forget_bias
    self.weights = (w, b)


def MakeZeroState(depth_multiplier=1):
  """Makes zeros of shape like x but removing the length (axis 1)."""
  def f(x):  # pylint: disable=invalid-name
    if len(x.shape) != 3:
      raise ValueError(f'Layer input should be a rank 3 tensor representing'
                       f' (batch_size, sequence_length, feature_depth); '
                       f'instead got shape {x.shape}.')
    return jnp.zeros((x.shape[0], depth_multiplier * x.shape[-1]),
                     dtype=jnp.float32)
  return base.Fn('MakeZeroState', f)


def LSTM(n_units, mode='train', return_state=False, initial_state=False):
  """LSTM running on axis 1.

  Args:
    n_units: `n_units` for the `LSTMCell`.
    mode: if 'predict' then we save the previous state for one-by-one inference.
    return_state: Boolean. Whether to return the latest status in addition to
      the output. Default: False.
    initial_state: Boolean. If the state RNN (c, h) is to be obtained from the
      stack. Default: False.

  Returns:
    A LSTM layer.
  """

  if not initial_state:
    zero_state = MakeZeroState(depth_multiplier=2)  # pylint: disable=no-value-for-parameter
    if return_state:
      return cb.Serial(
          cb.Branch([], zero_state),
          cb.Scan(LSTMCell(n_units=n_units), axis=1, mode=mode),
          name=f'LSTM_{n_units}',
          sublayers_to_print=[])
    else:
      return cb.Serial(
          cb.Branch([], zero_state),  # fill state RNN with zero.
          cb.Scan(LSTMCell(n_units=n_units), axis=1, mode=mode),
          cb.Select([0], n_in=2),  # Drop RNN state.
          # Set the name to LSTM and don't print sublayers.
          name=f'LSTM_{n_units}', sublayers_to_print=[])
  else:
    if return_state:
      return cb.Serial(
          cb.Scan(LSTMCell(n_units=n_units), axis=1, mode=mode),
          name=f'LSTM_{n_units}', sublayers_to_print=[])
    else:
      return cb.Serial(
          cb.Scan(LSTMCell(n_units=n_units), axis=1, mode=mode),
          cb.Select([0], n_in=2),  # Drop RNN state.
          name=f'LSTM_{n_units}', sublayers_to_print=[])


class GRUCell(base.Layer):
  """Builds a traditional GRU cell with dense internal transformations.

  Gated Recurrent Unit paper: https://arxiv.org/abs/1412.3555
  """

  def __init__(self,
               n_units,
               forget_bias=0.0,
               kernel_initializer=initializers.RandomUniformInitializer(0.01),
               bias_initializer=initializers.RandomNormalInitializer(1e-6)):
    super().__init__(n_in=2, n_out=2)
    self._n_units = n_units
    self._forget_bias = forget_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, inputs):
    x, gru_state = inputs

    # Dense layer on the concatenation of x and h.
    w1, b1, w2, b2 = self.weights
    y = jnp.dot(jnp.concatenate([x, gru_state], axis=-1), w1) + b1

    # Update and reset gates.
    u, r = jnp.split(fastmath.sigmoid(y), 2, axis=-1)

    # Candidate.
    c = jnp.dot(jnp.concatenate([x, r * gru_state], axis=-1), w2) + b2

    new_gru_state = u * gru_state + (1 - u) * jnp.tanh(c)
    return new_gru_state, new_gru_state

  def init_weights_and_state(self, input_signature):
    if input_signature[1].shape[-1] != self._n_units:
      raise ValueError(
          f'Second argument in input signature should have a final dimension of'
          f' {self._n_units}; instead got {input_signature[1].shape[-1]}.')

    # The dense layer input is the input and half of the GRU state.
    input_shape = input_signature[0].shape[-1] + self._n_units
    rng1, rng2, rng3, rng4 = fastmath.random.split(self.rng, 4)
    w1 = self._kernel_initializer((input_shape, 2 * self._n_units), rng1)
    b1 = self._bias_initializer((2 * self._n_units,), rng2) + self._forget_bias
    w2 = self._kernel_initializer((input_shape, self._n_units), rng3)
    b2 = self._bias_initializer((self._n_units,), rng4)
    self.weights = (w1, b1, w2, b2)


def GRU(n_units, mode='train'):
  """GRU running on axis 1."""
  zero_state = MakeZeroState(depth_multiplier=1)  # pylint: disable=no-value-for-parameter
  return cb.Serial(
      cb.Branch([], zero_state),
      cb.Scan(GRUCell(n_units=n_units), axis=1, mode=mode),
      cb.Select([0], n_in=2),  # Drop RNN state.
      # Set the name to GRU and don't print sublayers.
      name=f'GRU_{n_units}', sublayers_to_print=[]
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
      gate_nonlinearity=activation_fns.Sigmoid,
      candidate_nonlinearity=activation_fns.Tanh)


def GeneralGRUCell(candidate_transform,
                   memory_transform_fn=None,
                   gate_nonlinearity=activation_fns.Sigmoid,
                   candidate_nonlinearity=activation_fns.Tanh,
                   dropout_rate_c=0.1,
                   sigmoid_bias=0.5):
  r"""Parametrized Gated Recurrent Unit (GRU) cell construction.

  GRU update equations for update gate, reset gate, candidate memory, and new
  state:

  .. math::
    u_t &= \sigma(U' \times s_{t-1} + B') \\
    r_t &= \sigma(U'' \times s_{t-1} + B'') \\
    c_t &= \tanh(U \times (r_t \odot s_{t-1}) + B) \\
    s_t &= u_t \odot s_{t-1} + (1 - u_t) \odot c_t

  See `combinators.Gate` for details on the gating function.


  Args:
    candidate_transform: Transform to apply inside the Candidate branch. Applied
      before nonlinearities.
    memory_transform_fn: Optional transformation on the memory before gating.
    gate_nonlinearity: Function to use as gate activation; allows trying
      alternatives to `Sigmoid`, such as `HardSigmoid`.
    candidate_nonlinearity: Nonlinearity to apply after candidate branch; allows
      trying alternatives to traditional `Tanh`, such as `HardTanh`.
    dropout_rate_c: Amount of dropout on the transform (c) gate. Dropout works
      best in a GRU when applied exclusively to this branch.
    sigmoid_bias: Constant to add before sigmoid gates. Generally want to start
      off with a positive bias.

  Returns:
    A model representing a GRU cell with specified transforms.
  """
  gate_block = [  # u_t
      candidate_transform(),
      _AddSigmoidBias(sigmoid_bias),
      gate_nonlinearity(),
  ]
  reset_block = [  # r_t
      candidate_transform(),
      _AddSigmoidBias(sigmoid_bias),  # Want bias to start positive.
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


def InnerSRUCell():
  """The inner (non-parallel) computation of an SRU."""
  def f(cur_x_times_one_minus_f, cur_f, cur_state):  # pylint: disable=invalid-name
    res = cur_f * cur_state + cur_x_times_one_minus_f
    return res, res
  return base.Fn('InnerSRUCell', f, n_out=2)


def ScanSRUCell(mode, monkey_patched_mask=None):
  """The inner (non-parallel) computation of an SRU."""
  if monkey_patched_mask is None:
    return cb.Scan(InnerSRUCell(), axis=1, mode=mode)

  # This is necessary for Terraformer model. See comments there.
  # The mask will only be used in Terraformer in predict mode.
  assert mode == 'predict'

  def update_mask(mask, x_times_one_minus_f):  # pylint: disable=invalid-name
    initial = jnp.ones(x_times_one_minus_f.shape[:2], dtype=jnp.float32)
    if initial.shape[1] > 1:
      updated_mask = fastmath.dynamic_update_slice_in_dim(
          initial != 0, mask != 0, 1, axis=1)
    else:
      updated_mask = initial
    return updated_mask, x_times_one_minus_f

  def masked_inner_sru_cell(cur_mask, cur_x_times_one_minus_f, cur_f,  # pylint: disable=invalid-name
                            cur_state):
    res = ((cur_f * cur_state + cur_x_times_one_minus_f) * cur_mask
           + (1 - cur_mask) * cur_state)
    return res, res

  return cb.Serial(
      monkey_patched_mask.get_layer(),
      base.Fn('update_mask', update_mask, n_out=2),
      cb.Scan(base.Fn('MaskedInnerSRUCell', masked_inner_sru_cell, n_out=2),
              axis=1, mode=mode),
      )


def SRU(n_units, activation=None, mode='train'):
  r"""SRU (Simple Recurrent Unit) layer as in https://arxiv.org/abs/1709.02755.

  As defined in the paper:

  .. math::
    y_t &= W x_t + B \quad \hbox{(include $B$ optionally)} \\
    f_t &= \sigma(Wf x_t + bf) \\
    r_t &= \sigma(Wr x_t + br) \\
    c_t &= f_t \times c_{t-1} + (1 - f_t) \times y_t \\
    h_t &= r_t \times \hbox{activation}(c_t) + (1 - r_t) \times x_t

  We assume the input is of shape [batch, length, depth] and recurrence
  happens on the length dimension. This returns a single layer. It's best
  to use at least 2, they say in the paper, except inside a Transformer.

  Args:
    n_units: output depth of the SRU layer.
    activation: Optional activation function.
    mode: if 'predict' then we save the previous state for one-by-one inference

  Returns:
    The SRU layer.
  """
  sigmoid_activation = activation_fns.Sigmoid()
  return cb.Serial(                                         # x
      cb.Branch(core.Dense(3 * n_units), []),               # r_f_y, x
      cb.Split(n_items=3),                                  # r, f, y, x
      cb.Parallel(sigmoid_activation, sigmoid_activation),  # r, f, y, x
      base.Fn('',
              lambda r, f, y: (y * (1.0 - f), f, r),    # y * (1 - f), f, r, x
              n_out=3),
      cb.Parallel([], [], cb.Branch(MakeZeroState(), [])),
      ScanSRUCell(mode=mode),
      cb.Select([0], n_in=2),                               # act(c), r, x
      activation if activation is not None else [],
      base.Fn('FinalSRUGate', lambda c, r, x: c * r + x * (1 - r) * (3**0.5)),
      # Set the name to SRU and don't print sublayers.
      name=f'SRU_{n_units}', sublayers_to_print=[]
  )


def _AddSigmoidBias(sigmoid_bias):
  return base.Fn('AddSigmoidBias({sigmoid_bias})',
                 lambda x: x + sigmoid_bias)
