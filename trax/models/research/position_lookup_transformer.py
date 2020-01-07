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

# python3
"""Deep Lookups for Transformer Positions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from trax import layers as tl
from trax.math import numpy as np


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


@tl.layer(n_in=10, n_out=1)
def Softmax5Branches(x_list, **unused_kwargs):
  """Softmax qs.

  The input xs is a list of weights and embedded queries of the form
  w_1 ... w_n q_1 ... q_n. The q_1 ... q_n will be kept, result appended.

  Args:
    x_list: the input weights and embeddings.

  Returns:
    the weighted average of q_1 ... q_n according to softmax(w).
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
  return res


@tl.symbolic
def PerformPositionOperations(pos, positions=None):
  """Gets pos and returns (q1, ..., q5)."""
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
  query_types = [
      QueryPositionKV(),
      QueryPositionKV(keys=succ_keys, values=succ_values),
      QueryPositionKV(keys=subtract_1_keys, values=subtract_1_values),
      QueryPositionKV(keys=add_keys, values=add_values, binary=True),
      QueryPositionKV(keys=sub_keys, values=sub_values, binary=True)]
  return [qt @ pos for qt in query_types]  # pylint: disable=syntax-error


# TODO(levskaya): consider allowing *qs when explicit n_in fed to @tl.symbolic
@tl.symbolic
def AppendLearnedPosOperation(vec, q1, q2, q3, q4, q5):
  """Get (vec, q1, ...) and return new_pos."""
  # Create 5 scalar weights (length 1 vectors) from first component of input.
  ws = [tl.Dense(1) @ vec for _ in range(5)]
  new_pos = Softmax5Branches() @ (ws + [q1, q2, q3, q4, q5])
  return new_pos


@tl.symbolic
def LearnedPosOperations(vec, pos, positions=None, n_combinations=None):
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
    vec: features
    pos: position features
    positions: random vectors representing positions.
    n_combinations: int, how many combinations to produce.

  Returns:
    the tuple (new-pos_1, ..., new-pos_n_combinations).
  """
  qs = list(PerformPositionOperations(positions=positions) @ pos)
  new_posns = [AppendLearnedPosOperation() @ ([vec,] + qs)
               for _ in range(n_combinations)]
  return new_posns


class CopyPosToHeads(tl.Layer):
  """Copy position vectors to heads, possibly tiling if specified.

  Tiling means that the same position part will be appended to each head,
  otherwise we expect a different tensor with positions for each head.
  """

  def __init__(self, n_heads=1, tile=True):
    n_pos = 1 if tile else n_heads
    super(CopyPosToHeads, self).__init__(n_in=n_pos + 1)
    self._n_heads = n_heads
    self._n_pos = n_pos

  def forward(self, inp, weights):
    """Reshape input to have heads dimension and concatenate positions there."""
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
    return res


@tl.symbolic
def AttentionPosition(vec, pos,
                      positions=None, d_model=None, n_heads=8,
                      dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention."""

  new_posns = list(LearnedPosOperations(positions=positions,
                                        n_combinations=n_heads) @ (vec, pos))

  hq = tl.Serial(tl.Dense(d_model),
                 CopyPosToHeads(n_heads, tile=False)) @ ([vec,] + new_posns)
  hk = tl.Serial(tl.Dense(d_model),
                 CopyPosToHeads(n_heads, tile=True)) @ (vec, pos)
  hv = tl.ComputeAttentionHeads(
      n_heads=n_heads, d_head=d_model // n_heads) @ vec

  x, pos = tl.Serial(
      tl.DotProductCausalAttention(dropout=dropout, mode=mode),
      CombineHeadsPos(n_heads=n_heads),
      tl.Dense(d_model)) @ (hq, hk, hv)

  return x, pos


def _DecoderBlock(positions,
                  d_model,
                  d_ff,
                  n_heads,
                  dropout,
                  mode):
  """Returns a layer sequence representing a Transformer decoder.

  (acts, pos) --> (acts', pos')

  Args:
    positions: random vectors for positions
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'
  """
  return tl.Serial(
      tl.Residual(  # Self-attention block.
          tl.LayerNorm(),
          AttentionPosition(positions=positions,
                            d_model=d_model,
                            n_heads=n_heads,
                            dropout=dropout,
                            mode=mode),
          tl.Dropout(rate=dropout, mode=mode)
      ),
      tl.Residual(
          tl.LayerNorm(),
          tl.Dense(d_ff),
          tl.Relu(),
          tl.Dropout(rate=dropout, mode=mode),
          tl.Dense(d_model),
          tl.Dropout(rate=dropout, mode=mode),
      )
  )


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

  decoder_blocks = [
      _DecoderBlock(positions, d_model, d_ff, n_heads, dropout, mode)
      for _ in range(n_layers)]

  return tl.Serial(
      tl.ShiftRight(),
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Branch([], NewPositionalEncoding(positions=positions)),
      decoder_blocks,
      tl.Select([0], n_in=2),  # Drop positions.
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )
