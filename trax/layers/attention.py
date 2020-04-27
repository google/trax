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
"""Attention Layers."""

import jax
import numpy as np

from trax import math
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.base import Fn
from trax.math import numpy as jnp


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def zero_pad(x, pad, axis):
  """Helper for jnp.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return jnp.pad(x, pad_widths, mode='constant',
                 constant_values=x.dtype.type(0))


def ShiftRight(n_shifts=1, mode='train'):
  """Layer to shift the tensor to the right by padding on axis 1."""
  def f(x):
    if mode == 'predict':
      # Do nothing in predict mode, as then the sequence length is 1.
      return x
    padded = zero_pad(x, (n_shifts, 0), 1)
    return padded[:, :-n_shifts]
  return Fn(f'ShiftRight({n_shifts})', f)


def PaddingMask(pad=0):
  def f(x):
    return jnp.reshape(x != pad, (x.shape[0], 1, 1, x.shape[-1]))
  return Fn(f'PaddingMask({pad})', f)


def EncoderDecoderMask():
  """Makes encoder-decoder mask from decoder input and a padding mask."""
  def f(decoder_input, padding_mask):
    padding_mask = jnp.reshape(
        padding_mask, (padding_mask.shape[0], 1, 1, padding_mask.shape[-1]))
    # Final mask shape is [batch, 1 for heads, decoder-len, encoder-len].
    return padding_mask + jnp.zeros((1, 1, decoder_input.shape[1], 1))
  return Fn('EncoderDecoderMask', f)


class PositionalEncoding(base.Layer):
  """Implements bare positional encoding."""

  def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
               mode='train'):
    super(PositionalEncoding, self).__init__()
    self._max_len = max_len
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self._dropout = dropout
    else:
      self._dropout = 0.0
    self._dropout_broadcast_dims = dropout_broadcast_dims
    self._mode = mode

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None):
    if self._mode != 'predict':
      x = inputs
      symbol_size = jnp.shape(x)[1]
      px = weights[:, :symbol_size, :]
      if self._dropout == 0:
        return (x + px, state)
      else:
        noise_shape = list(px.shape)
        for dim in self._dropout_broadcast_dims:
          noise_shape[dim] = 1
        keep_prob = 1.0 - self._dropout
        if math.backend_name() == 'jax':
          keep_prob = jax.lax.tie_in(x, jnp.full((), keep_prob, dtype=x.dtype))
        keep = math.random.bernoulli(rng, keep_prob, tuple(noise_shape))
        multiplier = keep.astype(x.dtype) / keep_prob
        return (x + px * multiplier, state)
    else:
      assert self._dropout == 0
      # State in this class is only used for fast inference. In that case,
      # the model is called with consecutive elements position-by-position.
      # This positional encoding layer needs to store the index of the current
      # position then and increment it on each call -- that's how state is used
      # and updated below.
      if inputs.shape[1] == 1:
        return (inputs + jnp.expand_dims(weights[0, state, :], 1), state + 1)
      else:
        emb = []
        for i in range(inputs.shape[0]):
          emb.append(jax.lax.dynamic_slice_in_dim(
              weights[0], state[i], inputs.shape[1], axis=0))
        return inputs + jnp.stack(emb, 0), state + inputs.shape[1]

  def new_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
    position = np.arange(0, self._max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, self._max_len, d_feature]
    weights = jnp.array(pe)  # Trainable parameters, initialized above.
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      state = jnp.zeros((batch_size,), dtype=jnp.int32)
    else:
      state = base.EMPTY_STATE
    return weights, state


def DotProductAttention(query, key, value, mask, dropout, mode, rng):
  """Core dot product self-attention.

  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
    dropout: float: dropout rate
    mode: 'eval' or 'train': whether to use dropout
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Self attention for q, k, v arrays.
  """
  depth = jnp.shape(query)[-1]
  dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth)
  if mask is not None:
    # TODO(kitaev): workaround for https://github.com/google/jax/issues/850
    # We must ensure that both mask and the -1e9 constant have a data dependency
    # on the input. Broadcasted copies of these use a lot of memory, so they
    # should be computed at runtime (rather than being global constants).
    if math.backend_name() == 'jax':
      mask = jax.lax.tie_in(dots, mask)
    # JAX's `full_like` already ties in -1e9 to dots.
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
  # Softmax.
  dots = jnp.exp(dots - math.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = math.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))
  out = jnp.matmul(dots, value)
  return out


class PureAttention(base.Layer):
  """Layer constructor function for a pure attention layer."""

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    super(PureAttention, self).__init__(n_in=4, n_out=2)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward_with_state(self, x, weights, state, rng):
    """Pure transformer-style multi-headed attention.

    Args:
      x: inputs (q, k, v, mask)
      weights: parameters (none)
      state: parameters (none)
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      Pure Multi-headed attention result, and the mask.
    """
    del weights
    n_heads, dropout, mode = self._n_heads, self._dropout, self._mode
    q, k, v, mask = x
    d_feature = q.shape[-1]
    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads
    nbatch = jnp.shape(q)[0]
    # nbatch, seqlen, d_feature --> nbatch, n_heads, seqlen, d_head
    def SplitHeads(x):
      return jnp.transpose(
          jnp.reshape(x, (nbatch, -1, n_heads, d_head)), (0, 2, 1, 3))
    # nbatch, n_heads, seqlen, d_head --> nbatch, seqlen, d_feature
    def JoinHeads(x):  # pylint: disable=invalid-name
      return jnp.reshape(
          jnp.transpose(x, (0, 2, 1, 3)), (nbatch, -1, n_heads * d_head))
    # Split heads, dot-product attention, rejoin heads.
    res = JoinHeads(
        DotProductAttention(
            SplitHeads(q), SplitHeads(k), SplitHeads(v), mask,
            dropout=dropout, mode=mode, rng=rng))
    return (res, mask), state  # Keep the mask.


def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form q, k, v, mask.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return cb.Serial(
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      PureAttention(  # pylint: disable=no-value-for-parameter
          n_heads=n_heads, dropout=dropout, mode=mode),
      core.Dense(d_feature),
  )


def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form (x, mask) and constructs (q, k, v) from x.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return cb.Serial(
      cb.Dup(), cb.Dup(),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  )


def _fast_inference_init_state(input_signature, buffer_length):
  """Returns an initial state for causal attention layer fast inference."""
  def zeros_for(batch_size, shape_dtype):
    shape, dtype = shape_dtype.as_tuple()
    depth = shape[-1]
    return jnp.zeros((batch_size, buffer_length, depth), dtype=dtype)

  batch_size = input_signature[0].shape[0]
  k = zeros_for(batch_size, input_signature[1])
  v = zeros_for(batch_size, input_signature[2])
  mask = jnp.zeros((batch_size, 1, buffer_length))
  seq_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
  return (k, v, mask, seq_indices)


def _fast_inference_update_state(inputs, state):
  """Updates state of a causal attention layer for fast inference."""
  assert math.backend_name() == 'jax', (
      'JAX backend is required to use the predict mode.')
  for x in inputs:
    assert x.shape[1] == 1, (
        'In predict mode the input sequence must be of length 1.')
  # Fast inference: run with only 1 query in each step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  (ks, vs, mask, seq_indices) = state
  batch_indices = jnp.arange(ks.shape[0])
  ks = jax.ops.index_update(
      ks, jax.ops.index[batch_indices, seq_indices, :], new_k[:, 0, :]
  )
  vs = jax.ops.index_update(
      vs, jax.ops.index[batch_indices, seq_indices, :], new_v[:, 0, :]
  )
  mask = jax.ops.index_update(
      mask, jax.ops.index[batch_indices, :, seq_indices], 1
  )
  return (ks, vs, mask, seq_indices + 1)


class DotProductCausalAttention(base.Layer):
  """A standard (non-memory-efficient) dot product attention implementation."""

  def __init__(self, dropout=0.0, mode='train'):
    super(DotProductCausalAttention, self).__init__(n_in=3, n_out=1)
    self._dropout = dropout
    self._mode = mode

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None):
    del weights
    q, k, v = inputs
    if self._mode != 'predict':
      mask_size = q.shape[-2]
      # Not all backends define jnp.tril. However, using np.tril is inefficient
      # in that it creates a large global constant. TODO(kitaev): try to find an
      # alternative that works across all backends.
      if math.backend_name() == 'jax':
        mask = jnp.tril(
            jnp.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
      else:
        mask = np.tril(
            np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
    else:
      assert self._mode == 'predict'
      state = _fast_inference_update_state(inputs, state)
      (k, v, mask, _) = state

    res = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res, state

  def new_weights_and_state(self, input_signature):
    if self._mode != 'predict':
      return base.EMPTY_WEIGHTS, base.EMPTY_STATE

    assert self._mode == 'predict'
    weights = base.EMPTY_WEIGHTS
    # Buffer length is hardcoded for now. TODO(pkozakowski): Pass it from the
    # model.
    max_len = 2048
    state = _fast_inference_init_state(input_signature, max_len)
    return weights, state


def CausalAttention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed causal attention.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: attention dropout
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result.
  """
  assert d_feature % n_heads == 0
  d_head = d_feature // n_heads

  def compute_attention_heads(x):
    batch_size = x.shape[0]
    seqlen = x.shape[1]
    # n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head
    x = jnp.reshape(x, (batch_size, seqlen, n_heads, d_head))
    # n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head
    x = jnp.transpose(x, (0, 2, 1, 3))
    # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
    return jnp.reshape(x, (-1, seqlen, d_head))

  ComputeAttentionHeads = Fn('ComputeAttentionHeads', compute_attention_heads)

  def compute_attention_output(x):
    seqlen = x.shape[1]
    x = jnp.reshape(x, (-1, n_heads, seqlen, d_head))
    x = jnp.transpose(x, (0, 2, 1, 3))  # -> n_batch, seqlen, n_heads, d_head
    return jnp.reshape(x, (-1, seqlen, n_heads * d_head))

  return cb.Serial(
      cb.Branch(
          [core.Dense(d_feature), ComputeAttentionHeads],
          [core.Dense(d_feature), ComputeAttentionHeads],
          [core.Dense(d_feature), ComputeAttentionHeads],
      ),
      DotProductCausalAttention(dropout=dropout, mode=mode),
      Fn('ComputeAttentionOutput', compute_attention_output),
      core.Dense(d_feature)
  )
