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

"""Deep Lookups for Transformer Positions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from trax import layers as tl
from trax.backend import numpy as np


# pylint: disable=g-complex-comprehension
# pylint: disable=no-value-for-parameter

POS_VECTOR_SIZE = 32
_ABSOLUTE_MAX_LEN = 10000
_POSITIONS = onp.random.uniform(size=[_ABSOLUTE_MAX_LEN, POS_VECTOR_SIZE])


@tl.layer()
def NewPositionalEncoding(x, positions=None, **kwargs):
  """Implements new positional encoding."""
  del kwargs
  x_length = np.shape(x)[1]
  pos = np.array(positions)[np.newaxis, :x_length, :]
  pos += np.zeros((np.shape(x)[0], 1, 1))  # Broadcast on batch.
  return pos


@tl.layer(n_in=1, n_out=2)
def CombineHeadsPos(x, n_heads=1, **unused_kwargs):
  """Mix x = (x0, p0, ..., xH, pH) into (x0, ...., xH), p_combined.

  The positions are averaged as vectors.

  Args:
    x: input vector, concatenated (x0, p0, ..., xH, pH).
    n_heads: number of heads.

  Returns:
    the vector with combined xs and one with combined positions.
  """
  seqlen = x.shape[1]
  d_head = x.shape[2]
  x = np.reshape(x, (-1, n_heads, seqlen, d_head))
  x = np.transpose(x, (0, 2, 1, 3))  # -> n_batch, seqlen, n_heads, d_head
  x = np.reshape(x, (-1, seqlen, n_heads * d_head))
  head_size = int(d_head) - POS_VECTOR_SIZE
  res, positions, idx = [], [], 0
  for _ in range(n_heads):
    res.append(x[:, :, idx:idx+head_size])
    idx += head_size
    positions.append(x[:, :, idx:idx+POS_VECTOR_SIZE])
    idx += POS_VECTOR_SIZE
  combined_position = sum(positions) / float(len(positions))
  return np.concatenate(res, axis=-1), combined_position


@tl.layer()
def QueryPositionKV(x, keys=None, values=None, binary=False, **unused_kwargs):
  """Query a table with a position vector."""
  if keys is None:
    return x
  k = np.array(keys)
  v = np.array(values)
  q = x
  if binary:
    q = np.concatenate([x, x], axis=-1)
  return tl.DotProductAttention(q, k, v, None, 0.0, None, None)


@tl.layer(n_in=10, n_out=6)
def Softmax5Branches(x_list, **unused_kwargs):
  """Softmax qs.

  The input xs is a list of weights and embedded queries of the form
  w_1 ... w_n q_1 ... q_n. The q_1 ... q_n will be kept, result appended.

  Args:
    x_list: the input weights and embeddings.

  Returns:
    q_1 .... q_n q' where q' is the weighted average of q_1 ... q_n according
    to softmax(w).
  """
  n_branches = 5
  softmax_activations = x_list[:n_branches]
  max_sa = softmax_activations[0]
  for x in softmax_activations:
    max_sa = np.maximum(max_sa, x)
  softmax_activations = [x - max_sa for x in softmax_activations]
  softmax_activations = [np.exp(x) for x in softmax_activations]
  sum_sa = sum(softmax_activations)
  softmax_activations = [x / sum_sa for x in softmax_activations]
  res = sum([x_list[i + n_branches] * softmax_activations[i]
             for i in range(n_branches)])
  return x_list[n_branches:] + (res,)


def PerformPositionOperations(positions):
  """Get a pair (vec, pos) and return (vec, pos, q1, ..., q5)."""
  succ_keys = positions[:-1, :]
  succ_values = positions[1:, :]
  subtract_1_keys = positions[1:, :]
  subtract_1_values = positions[:-1, :]
  l = int(positions.shape[0]) // 2
  add_keys = np.array([np.concatenate([positions[i, :], positions[j, :]])
                       for i in range(l) for j in range(l)])
  add_values = np.array([positions[i + j, :]
                         for i in range(l) for j in range(l)])
  # TODO(lukaszkaiser): try this below: "for j in range(i) for i in range(2*l)"
  sub_keys = np.array([np.concatenate([positions[i, :], positions[j, :]])
                       for j in range(l) for i in range(l)])
  sub_values = np.array([positions[max(i - j, 0), :]
                         for j in range(l) for i in range(l)])
  return tl.Serial([
      tl.Parallel([], [tl.Dup() for _ in range(5)]),
      tl.Parallel(
          [], [],
          QueryPositionKV(),
          QueryPositionKV(keys=succ_keys, values=succ_values),
          QueryPositionKV(keys=subtract_1_keys, values=subtract_1_values),
          QueryPositionKV(keys=add_keys, values=add_values, binary=True),
          QueryPositionKV(keys=sub_keys, values=sub_values, binary=True),
      )
  ])


def AppendLearnedPosOperation():
  """Get (vec, pos, q1, ...) and return (vec, pos, q1, ..., new_pos)."""
  # Create 5 scalar weights (length 1 vectors) from first component of input.
  make5scalars = tl.Serial([  # Take x and create x, s1, ..., s5.
      [tl.Dup(), tl.Parallel([], tl.Dense(1))]
      for _ in range(5)
  ])
  return tl.Serial([
      tl.Swap(),
      tl.Parallel([], make5scalars),  # pos, vec, w1, ..., w5, q1, ..., q5
      tl.Parallel([], [], Softmax5Branches()),  # pos, vec, q1, ..., q5, new_pos
      tl.Swap()
  ])


def LearnedPosOperations(positions, n_combinations):
  """Perform position operations and get different learned combinations of them.

  This
  (a) applies 5 different "queries" (intuitively, operations) to the input
    positions, and then,
  (b) produces n different SoftMax combinations of them using different
    learned weights (in each case the weight is a function of input 'vec')

  Note that n_combinations is independent of the fact that 5 operations will
  be applied, it's a different number.

  As for input-output spec, this layer gets a pair (vec, pos) and returns
  a n_combinations + 2 tuple (vec, pos, new-pos_1, ..., new-pos_n_combinations).

  Args:
    positions: random vectors representing positions.
    n_combinations: int, how many combinations to produce.

  Returns:
    the tuple (vec, pos, new-pos_1, ..., new-pos_n_combinations).
  """
  return tl.Serial([PerformPositionOperations(positions)] + [
      AppendLearnedPosOperation() for _ in range(n_combinations)
  ] + [  # Drop the 5 position operations created at the beginning.
      tl.Parallel([], [], tl.Drop(), tl.Drop(), tl.Drop(), tl.Drop(), tl.Drop())
  ])


class CopyPosToHeads(tl.Layer):
  """Copy position vectors to heads, possibly tiling if specified.

  Tiling meand that the same position part will be appended to each head,
  otherwise we expect a different tensor with positions for each head.
  """

  def __init__(self, n_heads=1, tile=True):
    n_pos = 1 if tile else n_heads
    super(CopyPosToHeads, self).__init__(n_in=n_pos + 1)
    self._n_heads = n_heads
    self._n_pos = n_pos

  def forward(self, inp, params=(), state=(), **kwargs):
    """Reshape input to have heads dimension and concatenate positions there."""
    del kwargs
    x = inp[0]
    n_batches, seqlen = x.shape[0], x.shape[1]
    d_head = x.shape[-1] // self._n_heads
    res = np.reshape(x, (n_batches, seqlen, self._n_heads, d_head))
    res = np.transpose(res, (0, 2, 1, 3))  # (batch, heads, len, depth)
    if self._n_pos == 1:  # Just one position given, tile into each head.
      pos_shape = list(res.shape)[:-1] + [inp[1].shape[-1]]
      pos = inp[1][:, None, :, :] + np.zeros(pos_shape)  # Add 0 to broadcast.
    else:  # As many positions as heads, concatenate them in.
      pos = [p[:, None, :, :] for p in inp[1:]]
      pos = np.concatenate(pos, axis=1)
    res = np.concatenate([res, pos], axis=-1)
    # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
    res = np.reshape(res, (-1, seqlen, d_head + POS_VECTOR_SIZE))
    return res, state


def AttentionPosition(positions, d_model, n_heads=8, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention."""
  return tl.Serial([  # Input: (activations, positions).
      LearnedPosOperations(positions, n_heads),  # act, pos, np*h
      tl.Dup(), tl.Parallel([], tl.Swap()), tl.Dup(),  # a, a, pos, a, np*h
      tl.Parallel(
          tl.ComputeAttentionHeads(n_heads=n_heads, d_head=d_model // n_heads),
          [tl.Dense(d_model), CopyPosToHeads(n_heads, tile=True)],
          [tl.Dense(d_model), CopyPosToHeads(n_heads, tile=False)]
      ),  # attn_vals, attn_keys, attn_queries
      tl.Swap(), tl.Parallel([], tl.Swap()), tl.Swap(),  # queries, keys, vals
      tl.DotProductCausalAttention(dropout=dropout, mode=mode),
      CombineHeadsPos(n_heads=n_heads),
      tl.Dense(d_model)
  ])


def ResidualFeedForward(d_model,
                        d_ff,
                        dropout,
                        mode):
  """Residual feed-forward layer with normalization at start."""
  stack = tl.Serial([
      tl.LayerNorm(),
      tl.Dense(d_ff),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(d_model),
      tl.Dropout(rate=dropout, mode=mode)
  ])
  return tl.Residual(stack)


def DecoderLayer(positions,
                 d_model,
                 d_ff,
                 n_heads,
                 dropout,
                 mode):
  """Transformer decoder layer.

  Args:
    positions: random vectors for positions
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  return tl.Serial([
      tl.Residual(  # Self-attention block.
          tl.LayerNorm(),
          AttentionPosition(positions, d_model, n_heads=n_heads,
                            dropout=dropout, mode=mode),
          tl.Dropout(rate=dropout, mode=mode)
      ),
      ResidualFeedForward(d_model, d_ff, dropout, mode=mode)
  ])


def PositionLookupTransformerLM(vocab_size=128,
                                d_model=256,
                                d_ff=512,
                                n_layers=3,
                                n_heads=4,
                                dropout=0.1,
                                max_len=100,
                                mode='train'):
  """Transformer language model (only uses the decoder part of Transformer).

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: maximal length
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  positions = _POSITIONS[:max_len, :]
  return tl.Serial(
      tl.ShiftRight(),
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dup(),
      tl.Parallel([], NewPositionalEncoding(positions=positions)),
      [DecoderLayer(positions, d_model, d_ff, n_heads, dropout, mode)
       for _ in range(n_layers)],
      tl.Parallel([], tl.Drop()),  # Drop positions.
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )
