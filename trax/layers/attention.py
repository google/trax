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
"""Attention-related layers.

Attention is a powerful extension of basic neural network ideas.
In a classic neural network:

    - node activations are floating point values (one float per node), and
    - inter-node connections are trainable weights (one float per connection).

Attention assembles networks of *vectors* and uses vector calculations to
derive connection strength; in other words:

    - node activations are floating point vectors, and
    - inter-node connections come from trainable vector computations.

Attention thus involves extra concepts/mechanisms -- queries, keys, values,
masks, attention heads -- that factor heavily into this module's API. See
specific classes and functions for details.

NOTE: Attention layers in this module include `mode`-dependent behavior.
The possible modes are:

    - `'train'`: in training -- dropouts and position shifts active
    - `'eval'`:  in evals -- dropouts inactive, position shifts active
    - `'predict'`: in prediction -- dropouts and position shifts inactive
"""

import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


# inputs are [batch, length, depth], [batch, 1, 1 length]
@assert_shape('bld,b11l->bld,b11l')
def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (activations, mask) to (new_activations, mask).

  This layer type represents one pass of multi-head self-attention, best
  known for its central role in Transformer models. Internally, it:

    - maps incoming sequence of activations to sequence of (query, key, value)
      triples,
    - splits queries, keys, and values into multiple 'heads',
    - computes per-head attention weights from per-head (queries, keys),
    - applies mask to screen out positions that come from padding tokens,
    - [in `'train'` mode] applies dropout to attention weights,
    - uses attention weights to combine per-head values vectors, and
    - fuses per-head results into outgoing activations matching original input
      activation shapes.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  return cb.Serial(
      cb.Select([0, 0, 0]),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  )


@assert_shape('bSq,blk,blv,b1xl->bSd,b1xl')
def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps (q, k, v, mask) to (activations, mask).

  See `Attention` above for further context/details.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
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


# 'k' is number of keys/values, while 'l' is number of queries. Typically they
# will be the same, but it is not necessary.
@assert_shape('blq,bkq,bkd,b1xk->bld,b1xk')
class PureAttention(base.Layer):
  """Returns a layer that maps (q, k, v, mask) to (activations, mask).

  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:

    - splits queries, keys, and values into multiple 'heads',
    - computes per-head attention weights from per-head (queries, keys),
    - applies mask to screen out positions that come from padding tokens,
    - [in `'train'` mode] applies dropout to attention weights,
    - uses attention weights to combine per-head values vectors, and
    - merges per-head results into outgoing activations matching original input
      activation vector shapes.
  """

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    """Returns a new PureAttention instance.

    Args:
      n_heads: Number of attention heads.
      dropout: Probababilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=4, n_out=2)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A (queries, keys, values, mask) tuple.
    """
    q, k, v, mask = inputs

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    per_head_results, dots = DotProductAttention(
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(q),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(k),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(v),
        mask,
        dropout=self._dropout,
        mode=self._mode,
        rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(n_heads, merged_batch_and_head=False).forward(
        per_head_results)
    return (merged_results, mask)


def DotProductAttention(queries, keys, values, mask, dropout, mode, rng):
  """Computes new activations via masked attention-weighted sum of values.

  This function is the core of the attention mechanism. It:
    - computes per-head attention weights from per-head `queries` and `keys`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights, and
    - uses attention weights to combine per-head `values` vectors.

  Args:
    queries: Per-head activations representing attention queries.
    keys: Per-head activations representing attention keys.
    values: Per-head activations to be combined by computed attention weights.
    mask: Mask that distinguishes positions with real content vs. padding.
    dropout: Probababilistic rate for dropout applied to attention strengths
        (based on query-key pairs) before applying them to values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Per-head activations resulting from masked per-head attention-weighted
    sum of per-head values.
  """
  d_feature = queries.shape[-1]
  dots = jnp.matmul(queries, jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(d_feature)
  if mask is not None:
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
  # Softmax.
  dots = jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = fastmath.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))
  out = jnp.matmul(dots, values)
  out = out.astype(jnp.float32)
  dots = dots.astype(jnp.float32)
  return out, dots


# (b_size, seq_len, d_feature) --> (b_size*n_heads, seq_len, d_head)
@assert_shape('bld->...lh')
def SplitIntoHeads(n_heads, merged_batch_and_head=True):
  """Returns a layer that reshapes tensors for multi-headed computation."""
  def f(x):
    batch_size, seq_len, d_feature = x.shape

    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a multiple'
          f' of the requested number of attention heads ({n_heads}).')

    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads

    # (b_size, seq_len, d_feature) --> (b_size*n_heads, seq_len, d_head)
    x = x.reshape((batch_size, seq_len, n_heads, d_head))
    x = x.transpose((0, 2, 1, 3))
    if merged_batch_and_head:
      x = x.reshape((batch_size * n_heads, seq_len, d_head))
    return x
  return Fn('SplitIntoHeads', f)


# (b_size*n_heads, seq_len, d_head) --> (b_size, seq_len, d_feature)
@assert_shape('...lh->bld')
def MergeHeads(n_heads, merged_batch_and_head=True):
  """Returns a layer that undoes splitting, after multi-head computation."""
  def f(x):
    if merged_batch_and_head:
      batchheads, seq_len, d_head = x.shape
      assert batchheads % n_heads == 0
      batch_size = batchheads // n_heads
      x = x.reshape((batch_size, n_heads, seq_len, d_head))
    else:
      batch_size, _, seq_len, d_head = x.shape

    # (b_size, n_heads, seq_len, d_head) --> (b_size, seq_len, d_feature)
    x = x.transpose((0, 2, 1, 3))
    x = x.reshape((batch_size, seq_len, n_heads * d_head))
    return x
  return Fn('MergeHeads', f)


@assert_shape('bld->bld')
def ConfigurableAttention(q_layer, k_layer, v_layer, final_layer,  # pylint: disable=invalid-name
                          qkv_attention_layer, n_heads=1):
  return cb.Serial(
      cb.Branch(
          [q_layer, SplitIntoHeads(n_heads)],
          [k_layer, SplitIntoHeads(n_heads)],
          [v_layer, SplitIntoHeads(n_heads)],
      ),
      qkv_attention_layer,
      MergeHeads(n_heads),
      final_layer
  )


@assert_shape('bld->bld')
def CausalAttention(d_feature, n_heads=1, dropout=0.0,
                    max_inference_length=2048, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `Attention`, this layer type represents one pass of multi-head
  self-attention, but with causal masking rather than padding-based masking.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  if d_feature % n_heads != 0:
    raise ValueError(
        f'Dimensionality of feature embedding ({d_feature}) is not a multiple '
        f'of the requested number of attention heads ({n_heads}).')

  return ConfigurableAttention(
      core.Dense(d_feature), core.Dense(d_feature), core.Dense(d_feature),
      core.Dense(d_feature), n_heads=n_heads,
      qkv_attention_layer=DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('bld,bld,bld->bld')
class DotProductCausalAttention(base.Layer):
  """Layer that computes attention strengths by masking out the "future".

  Causal attention uses masking to prevent a given sequence position from
  attending to positions greater than / following it. This is used, for
  example, when training autoregressive sequence models, or when decoding a
  sequence symbol by symbol.

  This layer performs the core per-head attention calculation. The layer
  assumes that any splitting into attention heads precedes it, and that any
  merging of attention heads will follow it.
  """

  def __init__(self, dropout=0.0, max_inference_length=2048, mode='train'):
    """Creates a DotProductCausalAttention instance.

    Args:
      dropout: Probababilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      max_inference_length: maximum length of sequences during inference.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=3, n_out=1)
    self._dropout = dropout
    self._mode = mode
    self._max_len = max_inference_length

  def forward(self, inputs):
    """Returns attention-computed activations.

    Args:
      inputs: A (queries, keys, values) tuple.
    """
    q, k, v = inputs

    if self._mode == 'predict':
      self.state = _fast_inference_update_state(inputs, self.state)
      (k, v, mask, _) = self.state
    else:
      mask_size = q.shape[-2]
      # Not all backends define jnp.tril. However, using np.tril is inefficient
      # in that it creates a large global constant. TODO(kitaev): try to find an
      # alternative that works across all backends.
      if fastmath.is_backend(fastmath.Backend.JAX):
        mask = jnp.tril(
            jnp.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
      else:
        mask = np.tril(
            np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)

    res, dots = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    return res

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in `'predict'` mode."""
    if self._mode == 'predict':
      self.state = _fast_inference_init_state(input_signature, self._max_len)


@assert_shape('...d->...d')
def ShiftRight(n_positions=1, mode='train'):
  """Returns a layer that can insert padding to shift the input sequence.

  Args:
    n_positions: Number of positions to shift the input sequence rightward;
        initial positions freed by the shift get padded with zeros.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  # TODO(jonni): Include pad arg, like PaddingMask, to allow non-default pads?
  def f(x):
    if mode == 'predict':
      return x
    padded = _zero_pad(x, (n_positions, 0), 1)
    return padded[:, :-n_positions]
  return Fn(f'ShiftRight({n_positions})', f)


@assert_shape('bs->b11l')
def PaddingMask(pad=0):
  """Returns a layer that maps integer sequences to padding masks.

  The layer expects as input a batch of integer sequences. The layer output is
  a tensor that marks for each sequence position whether the integer (e.g., a
  token ID) in that position represents padding -- value `pad` -- versus
  text/content -- all other values. The padding mask shape is
  (batch_size, 1, 1, encoder_sequence_length), such that axis 1 will broadcast
  to cover any number of attention heads and axis 2 will broadcast to cover
  decoder sequence positions.

  Args:
    pad: Integer that represents padding rather than a token/content ID.
  """
  def f(x):
    if len(x.shape) != 2:
      raise ValueError(
          f'Input to PaddingMask must be a rank 2 tensor with shape '
          f'(batch_size, sequence_length); instead got shape {x.shape}.')
    batch_size = x.shape[0]
    sequence_length = x.shape[1]
    content_positions = (x != pad)
    return content_positions.reshape((batch_size, 1, 1, sequence_length))
  return Fn(f'PaddingMask({pad})', f)


def EncoderDecoderMask():
  """Returns a layer that creates a mask for encoder-decoder cross attention.

  The layer expects two inputs:

      - decoder_input: batch of integer (e.g., token ID) sequences
      - mask: padding mask from the encoder

  The layer output is a mask that marks for each sequence position (for both
  encoder and decoder) whether that position can be attended to or not. The
  encoder-decoder mask shape is (batch_size, 1, decoder_sequence_length,
  encoder_sequence_length), such that axis 1 will automatically broadcast to
  cover any number of attention heads.
  """
  def f(decoder_input, mask):
    if len(decoder_input.shape) != 3:
      raise ValueError(
          f'Decoder input to EncoderDecoderMask must be a rank 3 tensor with '
          f'shape (batch_size, decoder_sequence_length, d_model); instead got '
          f'shape {decoder_input.shape}.')
    batch_size = mask.shape[0]
    encoder_sequence_length = mask.shape[-1]
    decoder_sequence_length = decoder_input.shape[1]
    mask = mask.reshape((batch_size, 1, 1, encoder_sequence_length))
    return mask + jnp.zeros((1, 1, decoder_sequence_length, 1))
  return Fn('EncoderDecoderMask', f)


@assert_shape('...d->...d')
class PositionalEncoding(base.Layer):
  """Implements bare positional encoding.

  Positional encoding includes a kind of dropout, if the layer is created in
  `'train'` mode with a nonzero `dropout` value. For such a layer, on each
  forward pass a subset of sequence positions selected at random will *not*
  receive positional marking.
  """

  def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
               use_bfloat16=False, mode='train'):
    """Creates a PositionalEncoding instance.

    Args:
      max_len: Maximum input sequence length.
      dropout: Probability of *not* adding positional encoding to a sequence
          position.
      dropout_broadcast_dims: Axes along which dropout mask values are
          broadcast rather than individually set at random.
      use_bfloat16: If `True`, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__()
    self._max_len = max_len
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self._dropout = dropout
    else:
      self._dropout = 0.0
    self._dropout_broadcast_dims = dropout_broadcast_dims
    self._use_bfloat16 = use_bfloat16
    self._mode = mode

  def forward(self, inputs):
    """Returns the input activations, with added positional information."""
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
          emb.append(fastmath.dynamic_slice_in_dim(
              self.weights[0], state[i], inputs.shape[1], axis=0))
        self.state = state + inputs.shape[1]
        return inputs + jnp.stack(emb, 0)

  def init_weights_and_state(self, input_signature):
    """Randomly initializes the positional encoding vectors.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this
          layer should compute on.
    """
    d_feature = input_signature.shape[-1]
    pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
    position = np.arange(0, self._max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, self._max_len, d_feature]
    if self._use_bfloat16:
      pe = pe.astype(jnp.bfloat16)
    self.weights = jnp.array(pe)  # Trainable parameters, initialized above.
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      self.state = jnp.zeros((batch_size,), dtype=jnp.int32)


def _zero_pad(x, pad, axis):
  """Helper for jnp.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return jnp.pad(x, pad_widths, mode='constant',
                 constant_values=x.dtype.type(0))


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
  return (k, v, mask, jnp.array(0))


def _fast_inference_update_state(inputs, state):
  """Updates state of a causal attention layer for fast inference.

  The layer state stores tensors with cached values of keys and values,
  as well as the mask and an index. To make shapes static, keys and values
  in the state are long, and the index indicates where the new keys and values
  from inputs need to be appended. Mask ensures that attention will only look
  at keys upto index.

  During update, we append new_keys and new_values to keys and values at
  position given by index. We also update mask (which starts as all-0s) to
  be 1 at the new keys positions. And we increment index by length of new keys.

  Args:
    inputs: a triple (new_queries, new_keys, new_values)
    state: layer state with (keys, values, mask, index)

  Returns:
    Updated state.
  """
  if not fastmath.is_backend(fastmath.Backend.JAX):
    raise ValueError(f'JAX backend is required in predict mode, but found '
                     f"backend ({fastmath.backend()['name']}).")

  # Fast inference: run step-by-step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  length = new_k.shape[1]
  (ks, vs, mask, idx) = state
  # TODO(lukaszkaiser): benchmark speed and decide if using a separate code path
  # with index_update when length == 1 is worth it.
  # Keys and values are of shape [batch_size, length, d_kv].
  ks = fastmath.dynamic_update_slice_in_dim(ks, new_k, idx, axis=1)
  vs = fastmath.dynamic_update_slice_in_dim(vs, new_v, idx, axis=1)
  # Mask is of shape [batch_size, 1 (for heads), length].
  new_mask = jnp.ones((mask.shape[0], mask.shape[1], length))
  mask = fastmath.dynamic_update_slice_in_dim(mask, new_mask, idx, axis=2)
  return (ks, vs, mask, idx + length)
