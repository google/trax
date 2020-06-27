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

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.base import Fn


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (activations, mask) to (new_activations, mask).

  This layer type represents one pass of multi-head self-attention, best
  known for its central role in Transformer models. Internally, it:

    - maps activations to `(queries, keys, values)` triples,
    - splits `queries`, `keys`, and `values` into multiple 'heads',
    - computes per-head attention weights from per-head `(queries, keys)`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights,
    - uses attention weights to combine per-head `values` vectors, and
    - fuses per-head results into activations matching original input shapes.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: Either 'train' or 'eval'.
  """
  return cb.Serial(
      cb.Dup(), cb.Dup(),  # TODO(jonni): replace with Select([0, 0, 0])
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  )


def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (q, k, v, mask) to (activations, mask).

  See `Attention` above for further context/details.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: Either 'train' or 'eval'.
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


class PureAttention(base.Layer):
  """Layer that maps from (queries, keys, values, mask) to (activations, mask).

  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:

    - splits `queries`, `keys`, and `values` into multiple 'heads',
    - computes per-head attention weights from per-head `(queries, keys)`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights,
    - uses attention weights to combine per-head `values` vectors, and
    - merges per-head results into activations matching original input shapes.
  """

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    super(PureAttention, self).__init__(n_in=4, n_out=2)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations, unmodified mask, and state.

    Args:
      inputs: Tuple of (queries, keys, values, mask).

    Returns:
      The pair (activations, mask).
    """
    q, k, v, mask = inputs

    batch_size = q.shape[0]
    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    d_head = d_feature // n_heads

    def _split_into_heads(x):
      """Reshapes tensors to prepare for multi-head computation."""
      # (b_size, seq_len, d_feature) --> (b_size, n_heads, seq_len, d_head)
      x = x.reshape((batch_size, -1, n_heads, d_head))
      x = x.transpose((0, 2, 1, 3))
      return x

    def _merge_heads(x):
      """Undoes splitting, after multi-head computation."""
      # (b_size, n_heads, seq_len, d_head) --> (b_size, seq_len, d_feature)
      x = x.transpose((0, 2, 1, 3))
      x = x.reshape((batch_size, -1, n_heads * d_head))
      return x

    per_head_results = DotProductAttention(_split_into_heads(q),
                                           _split_into_heads(k),
                                           _split_into_heads(v),
                                           mask,
                                           dropout=self._dropout,
                                           mode=self._mode,
                                           rng=self.rng)
    merged_results = _merge_heads(per_head_results)
    return (merged_results, mask)


def DotProductAttention(queries, keys, values, mask, dropout, mode, rng):
  """Computes new activations via masked attention-weighted sum of values.

  This function is the core of the attention mechanism. It:
    - computes per-head attention weights from per-head `(queries, keys)`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights, and
    - uses attention weights to combine per-head `values` vectors.

  Args:
    queries: Per-head activations representing attention queries.
    keys: Per-head activations representing attention keys.
    values: Per-head activations to be combined by computed attention weights.
    mask: Mask that distinguishes positions with real content vs. padding.
    dropout: Probababilistic rate for dropout applied to attention activations
        (based on query-key pairs) before dotting them with values.
    mode: Either 'train' or eval'. Dropout applies only in 'train' mode.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Per-head activations resulting from masked per-head attention-weighted
    sum of per-head values.
  """
  d_feature = queries.shape[-1]
  dots = jnp.matmul(queries, jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(d_feature)
  if mask is not None:
    # TODO(kitaev): workaround for https://github.com/google/jax/issues/850
    # We must ensure that both mask and the -1e9 constant have a data dependency
    # on the input. Broadcasted copies of these use a lot of memory, so they
    # should be computed at runtime (rather than being global constants).
    if fastmath.backend_name() == 'jax':
      mask = jax.lax.tie_in(dots, mask)
    # JAX's `full_like` already ties in -1e9 to dots.
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
  # Softmax.
  dots = jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = fastmath.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))
  out = jnp.matmul(dots, values)
  return out


def CausalAttention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `Attention`, this layer type represents one pass of multi-head
  self-attention, but with causal masking in place of padding-based masking.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: Either 'train' or 'eval'.
  """
  if d_feature % n_heads != 0:
    raise ValueError(
        f'Dimensionality of feature embedding ({d_feature}) is not a multiple '
        f'of the requested number of attention heads ({n_heads}).')

  d_head = d_feature // n_heads

  def _split_into_heads():
    """Layer that reshapes tensors to prepare for multi-headed computation."""
    def f(x):
      batch_size = x.shape[0]
      seq_len = x.shape[1]

      # (b_size, seq_len, d_feature) --> (b_size*n_heads, seq_len, d_head)
      x = x.reshape((batch_size, seq_len, n_heads, d_head))
      x = x.transpose((0, 2, 1, 3))
      x = x.reshape((-1, seq_len, d_head))
      return x
    return Fn('SplitIntoHeads', f)

  def _merge_heads():
    """Layer that undoes splitting, after multi-head computation."""
    def f(x):
      seq_len = x.shape[1]

      # (b_size*n_heads, seq_len, d_head) --> (b_size, seq_len, d_feature)
      x = x.reshape((-1, n_heads, seq_len, d_head))
      x = x.transpose((0, 2, 1, 3))
      x = x.reshape((-1, seq_len, n_heads * d_head))
      return x
    return Fn('MergeHeads', f)

  return cb.Serial(
      cb.Branch(
          [core.Dense(d_feature), _split_into_heads()],
          [core.Dense(d_feature), _split_into_heads()],
          [core.Dense(d_feature), _split_into_heads()],
      ),
      DotProductCausalAttention(dropout=dropout, mode=mode),
      _merge_heads(),
      core.Dense(d_feature),
  )


class DotProductCausalAttention(base.Layer):
  """Computes new activations via causally masked attention-weighted values."""

  def __init__(self, dropout=0.0, mode='train'):
    super(DotProductCausalAttention, self).__init__(n_in=3, n_out=1)
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    q, k, v = inputs

    if self._mode == 'predict':
      self.state = _fast_inference_update_state(inputs, self.state)
      (k, v, mask, _) = self.state
    else:
      mask_size = q.shape[-2]
      # Not all backends define jnp.tril. However, using np.tril is inefficient
      # in that it creates a large global constant. TODO(kitaev): try to find an
      # alternative that works across all backends.
      if fastmath.backend_name() == 'jax':
        mask = jnp.tril(
            jnp.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
      else:
        mask = np.tril(
            np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)

    res = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=self.rng)
    return res

  def init_weights_and_state(self, input_signature):
    if self._mode == 'predict':
      max_len = 2048  # Hardcoded.  TODO(pkozakowski): Pass it from the model.
      self.state = _fast_inference_init_state(input_signature, max_len)


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
  """Layer to distinguish positions with real content/tokens vs. padding."""
  def f(x):
    # TODO(jonni): Check/require that len(x.shape) == 2?
    batch_size = x.shape[0]
    d_feature = x.shape[-1]
    content_positions = (x != pad)
    return content_positions.reshape((batch_size, 1, 1, d_feature))
  return Fn(f'PaddingMask({pad})', f)


def EncoderDecoderMask():
  """Makes encoder-decoder mask from decoder input and a padding mask."""
  def f(decoder_input, mask):
    batch_size = mask.shape[0]
    d_feature = mask.shape[-1]
    mask = mask.reshape((batch_size, 1, 1, d_feature))
    # Final mask shape is [batch, 1 for heads, decoder-len, encoder-len].
    return mask + jnp.zeros((1, 1, decoder_input.shape[1], 1))
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

  def forward(self, inputs):
    if self._mode != 'predict':
      x = inputs
      symbol_size = jnp.shape(x)[1]
      px = self.weights[:, :symbol_size, :]
      if self._dropout == 0:
        return x + px
      else:
        noise_shape = list(px.shape)
        for dim in self._dropout_broadcast_dims:
          noise_shape[dim] = 1
        keep_prob = 1.0 - self._dropout
        if fastmath.backend_name() == 'jax':
          keep_prob = jax.lax.tie_in(x, jnp.full((), keep_prob, dtype=x.dtype))
        keep = fastmath.random.bernoulli(self.rng, keep_prob,
                                         tuple(noise_shape))
        multiplier = keep.astype(x.dtype) / keep_prob
        return x + px * multiplier
    else:
      if self._dropout != 0:
        raise ValueError(f'In predict mode, but dropout rate '
                         f'({self._dropout}) is not zero.')

      # State in this class is only used for fast inference. In that case,
      # the model is called with consecutive elements position-by-position.
      # This positional encoding layer needs to store the index of the current
      # position then and increment it on each call -- that's how state is used
      # and updated below.
      state = self.state
      if inputs.shape[1] == 1:
        self.state = state + 1
        return inputs + jnp.expand_dims(self.weights[0, state, :], 1)
      else:
        emb = []
        for i in range(inputs.shape[0]):
          emb.append(jax.lax.dynamic_slice_in_dim(
              self.weights[0], state[i], inputs.shape[1], axis=0))
        self.state = state + inputs.shape[1]
        return inputs + jnp.stack(emb, 0)

  def init_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
    position = np.arange(0, self._max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, self._max_len, d_feature]
    self.weights = jnp.array(pe)  # Trainable parameters, initialized above.
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      self.state = jnp.zeros((batch_size,), dtype=jnp.int32)


def _fast_inference_init_state(input_signature, buffer_length):
  """Returns an initial state for causal attention layer fast inference."""
  def zeros_for(batch_size, shape_dtype):
    shape, dtype = shape_dtype.as_tuple()
    d_feature = shape[-1]
    return jnp.zeros((batch_size, buffer_length, d_feature), dtype=dtype)

  batch_size = input_signature[0].shape[0]
  k = zeros_for(batch_size, input_signature[1])
  v = zeros_for(batch_size, input_signature[2])
  mask = jnp.zeros((batch_size, 1, buffer_length))
  seq_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
  return (k, v, mask, seq_indices)


def _fast_inference_update_state(inputs, state):
  """Updates state of a causal attention layer for fast inference."""
  if fastmath.backend_name() != 'jax':
    raise ValueError(f'JAX backend is required in predict mode, but found '
                     f'backend ({fastmath.backend_nameO()}).')
  for x in inputs:
    if x.shape[1] != 1:
      raise ValueError(f'In predict mode, input sequence must have length 1, '
                       f'instead has length {x.shape[1]}.')
  # Fast inference: run with only 1 query in each step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  (ks, vs, mask, seq_indices) = state
  batch_indices = jnp.arange(ks.shape[0])
  ks = jax.ops.index_update(
      ks, jax.ops.index[batch_indices, seq_indices, :], new_k[:, 0, :])
  vs = jax.ops.index_update(
      vs, jax.ops.index[batch_indices, seq_indices, :], new_v[:, 0, :])
  mask = jax.ops.index_update(
      mask, jax.ops.index[batch_indices, :, seq_indices], 1)
  return (ks, vs, mask, seq_indices + 1)
