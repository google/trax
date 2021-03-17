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
"""Residual Shuffle-Exchange Networks.

https://arxiv.org/pdf/2004.04662.pdf
"""

import numpy as np
from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers.assert_shape import assert_shape


# pylint: disable=invalid-name
def _inverse_sigmoid(x):
  return np.log(x / (1 - x))


@assert_shape('...->...')
class _ClippedScaling(tl.Layer):
  """Pointwise multiplies by sigmoid(S) with a learnable vector S."""

  def __init__(self,
               residual_weight):
    super().__init__(n_in=1, n_out=1)
    self._residual_weight = residual_weight

  def forward(self, x):
    s = self.weights
    return jnp.multiply(x, fastmath.expit(s))

  def init_weights_and_state(self, input_signature):
    self.weights = _inverse_sigmoid(self._residual_weight) * np.ones(
        (input_signature.shape[-1])).astype('float32')


@assert_shape('bld->bld')
def ResidualSwitchUnit(
    d_model, dropout=0.1, mode='train', residual_weight=0.9):
  r"""RSU (Residual Switch Unit) layer as in https://arxiv.org/pdf/2004.04662.pdf.

  As defined in the paper:

  .. math::
    i &= [i_1, i_2] \\
    g &= GELU(LayerNorm(Z i)) \\
    c &= W g + B \\
    [o_1, o_2] &= \sigma(S) \bigodot i + h \bigodot c

  where Z, W, B, S are learnable parameters with sizes 2m × 4m, 4m × 2m, 2m, 2m.
  We assume that both i_1 and i_2 have size m. h is a scalar value.

  We assume the input is of shape [batch, length, depth].

  Args:
    d_model: output depth of the SRU layer
    dropout: dropout rate used in 'train' mode
    mode: mode for dropout layer
    residual_weight: value used in initializing vector S and constant h

  Returns:
    The RSU layer.
  """
  return tl.Serial(
      tl.Fn(
          'Reshape2Pairs',
          lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] // 2, -1)),
          n_out=1),
      tl.Residual(
          tl.Dense(4 * d_model, use_bias=False),
          tl.LayerNorm(),
          tl.Gelu(),
          tl.Dense(2 * d_model),
          tl.Fn('Scaling',
                lambda x: x * np.sqrt(1 - residual_weight**2) * 0.25,
                n_out=1),
          shortcut=_ClippedScaling(residual_weight)),
      tl.Fn(
          'UnPair',
          lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * 2, -1)),
          n_out=1),
      tl.Dropout(rate=dropout, mode=mode)
      )


def _ror(x, n, p=1):
  """Bitwise right rotation.

  Args:
    x: np.array
    n: Bit count to represent each value of x
    p: Bit positions to shift

  Returns:
    np.array: x with all values shifted by p positions in n bits
  """
  a = np.right_shift(x, p)
  b = np.left_shift(1, p) - 1
  c = np.bitwise_and(x, b)
  d = np.left_shift(c, n - p)

  return a + d


def _rol(x, n, p=1):
  """Bitwise left rotation.

  Args:
    x: np.array
    n: Bit count to represent each value of x
    p: Bit positions to shift

  Returns:
    np.array: x with all values shifted by p positions in n bits
  """
  a = np.left_shift(x, p)
  b = np.left_shift(1, n) - 1
  c = np.bitwise_and(a, b)
  d = np.right_shift(x, n - p)

  return np.bitwise_or(c, d)


def _shuffle_layer(inputs, shuffle_fn):
  """Shuffles the elements according to bitwise left or right rotation.

  Args:
    inputs: Tensor input from previous layer
    shuffle_fn: Shift function rol or ror

  Returns:
    tf.Tensor: Inputs shifted according to shuffle_fn
  """
  seq_length = inputs.shape[1]
  n_bits = np.int32(np.log(seq_length - 1) / np.log(2.0)) + 1

  indices = np.arange(0, seq_length).astype('int32')
  rev_indices = shuffle_fn(indices, n_bits)
  return jnp.take(inputs, rev_indices, axis=1, mode='clip')


@assert_shape('bld->bld')
def ShuffleLayer():
  return tl.Fn(
      'ShuffleLayer', lambda x: _shuffle_layer(x, _rol), n_out=1)


@assert_shape('bld->bld')
def ReverseShuffleLayer():
  return tl.Fn(
      'ReverseShuffleLayer', lambda x: _shuffle_layer(x, _ror), n_out=1)


@assert_shape('...,bld->...,bld')
def _ForwardStep(d_model, dropout, mode):
  """Takes (n_layer, state) and returns (n_layer, shuffle_layer(rsu(state)))."""
  return tl.Parallel([], tl.Serial(
      ResidualSwitchUnit(d_model, dropout, mode),
      ShuffleLayer(),
  ))


@assert_shape('...,bld->...,bld')
def _BackwardStep(d_model, dropout, mode):
  """Takes (n_layer, state) and returns (n_layer, reverse_shuffle_layer(rsu(state)))."""
  return tl.Parallel([], tl.Serial(
      ResidualSwitchUnit(d_model, dropout, mode),
      ReverseShuffleLayer(),
  ))


@assert_shape('bld->bld')
def BenesBlock(d_model, dropout, mode):
  def bit_sequence(inputs):
    seq_length = inputs.shape[1]
    n_bits = np.int32(np.log(seq_length - 1) / np.log(2.0)) + 1
    return jnp.arange(0, n_bits)
  return tl.Serial(
      tl.Dup(),
      tl.Fn('BitSeq', bit_sequence, n_out=1),
      tl.Scan(_ForwardStep(d_model, dropout, mode)),
      tl.Scan(_BackwardStep(d_model, dropout, mode)),
      tl.Select([1]),
  )


@assert_shape('bl->blv')
def ResidualShuffleExchange(vocab_size,
                            d_model,
                            input_dropout,
                            dropout,
                            mode='train',
                            n_blocks=2):
  """Returns a Residual Shuffle Exchange Network model."""
  benes_blocks = [BenesBlock(d_model, dropout, mode) for _ in range(n_blocks)]
  return tl.Serial(
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=input_dropout, mode=mode),
      # Apply Benes Block n_blocks times.
      *benes_blocks,
      ResidualSwitchUnit(d_model, dropout, mode),
      # Produce probabilities.
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )
