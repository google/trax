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
"""Relative attention related layers.

Implementation of Relative Attention mechanism first exposed in Transformer XL
paper: https://arxiv.org/pdf/1901.02860.pdf.
This particular implementation however focus on compatibility with
Funnel Transformer model from:
- Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
  Language Processing https://arxiv.org/abs/2006.03236
"""

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape
from trax.layers.attention import MergeHeads
from trax.layers.attention import SplitIntoHeads


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def RelativeAttentionWrapper(n_heads=1,
                             d_qk=64,
                             d_v=64,
                             causal=False,
                             masked=False,
                             output_dropout=0.0,
                             attention_dropout=0.0,
                             mode='train',
                             n_raw_tokens_generated=None,
                             max_inference_length=3072,
                             total_kv_pooling=1,
                             chunk_len=None,
                             chunk_offset=None):
  """Relative Attention wrapper."""
  del d_v, causal, masked, output_dropout
  return RelativeAttentionLMLayer(
      d_feature=d_qk * n_heads,
      total_kv_pooling=total_kv_pooling,
      n_heads=n_heads,
      dropout=attention_dropout,
      n_raw_tokens_generated=n_raw_tokens_generated,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      mode=mode)


@assert_shape('bld,...->bld,...')
def RelativeAttentionLayer(d_feature,
                           total_kv_pooling,
                           n_heads=1,
                           dropout=0.0,
                           n_raw_tokens_generated=1,
                           max_inference_length=3072,
                           chunk_len=None,
                           chunk_offset=None,
                           mode='train'):
  """Returns a layer that maps (q, k, v, masks) to (activations, masks).

  When number of keys is smaller than number of queries layer works in O(q^2*d).
  Otherwise it is O(q*k*d). That is because we need to shift relative distances
  by current_pooling. When we upsample this is current pooling is a fraction < 1
  Visual explanation:
  [01][23][45][67] -> [0][1][2][3][4][5][6][7]
  For token [0] we calculate relative distances as follows:
  * 0 2 4 6
  However for token [1] we need relative distances changed by 1, specifically:
  * -1 1 3 5
  So we not only need to calculate the distances that corresponds to spacing
  between the keys but also for the ones in between because there are more than
  one query tokens (on different positions which means different relative
  distances) for single key token.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer.
    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
      activations (based on query-key pairs) before dotting them with values.
    n_raw_tokens_generated: Number of tokens generated in a single pass through
      this layer. Used only in 'predict' non-training mode.
    max_inference_length: Maximum sequence length allowed in non-training
      modes.
    chunk_len (optional): Number of tokens per chunk. Setting this option will
      enable chunked attention.
    chunk_offset (optional): Offset for shifting chunks, for shifted chunked
      attention
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  pos_emb = PositionalEmbeddings(
      d_feature,
      total_kv_pooling,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      n_raw_tokens_generated=n_raw_tokens_generated,
      mode=mode)

  attention = RelativeAttention(  # pylint: disable=no-value-for-parameter
      total_kv_pooling=total_kv_pooling,
      n_heads=n_heads,
      dropout=dropout,
      n_raw_tokens_generated=n_raw_tokens_generated,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      mode=mode),

  assert d_feature % n_heads == 0
  d_head = d_feature // n_heads
  context_bias_layer = core.Weights(
      init.RandomNormalInitializer(1e-6), shape=(1, n_heads, 1, d_head))
  location_bias_layer = core.Weights(
      init.RandomNormalInitializer(1e-6), shape=(1, n_heads, 1, d_head))

  return cb.Serial(
      cb.Branch(
          cb.Serial(pos_emb, core.Dense(d_feature)),
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
          cb.Select([1])  # mask
      ),
      context_bias_layer,
      location_bias_layer,
      attention,
      core.Dense(d_feature),
  )


@assert_shape('bld->bld')
def RelativeAttentionLMLayer(d_feature,
                             total_kv_pooling,
                             n_heads=1,
                             dropout=0.0,
                             n_raw_tokens_generated=1,
                             max_inference_length=3072,
                             chunk_len=None,
                             chunk_offset=None,
                             mode='train'):
  """Returns a layer that maps (q, k, v) to (activations).

  Same as standard Relative attention layer but additionally based on sizes
  of queries and keys prepares a mask that masks out the future.
  Masking the future is the concept primarily used for Language Modelling.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer.
    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
      activations (based on query-key pairs) before dotting them with values.
    n_raw_tokens_generated: Number of tokens generated in a single pass through
      this layer. Used only in 'predict' non-training mode.
    max_inference_length: Maximum sequence length allowed in non-training
      modes.
    chunk_len (optional): Number of tokens per chunk. Setting this option will
      enable chunked attention.
    chunk_offset (optional): Offset for shifting chunks, for shifted chunked
      attention
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  attention = RelativeAttentionLayer(
      d_feature,
      total_kv_pooling,
      n_heads=n_heads,
      dropout=dropout,
      n_raw_tokens_generated=n_raw_tokens_generated,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      mode=mode)

  mask_layer = AttentionMaskLayer(
      total_kv_pooling=total_kv_pooling,
      max_inference_length=max_inference_length,
      chunk_len=chunk_len,
      chunk_offset=chunk_offset,
      n_raw_tokens_generated=n_raw_tokens_generated,
      mode=mode)

  return cb.Serial(
      cb.Branch(
          None,
          mask_layer,  # vecs, mask
      ),
      attention,  # vecs, mask
      cb.Select([0], n_in=2),  # vecs
  )


class RelativeAttention(base.Layer):
  """Relative attention.

  A layer that maps (location_bias, context_bias, pos_emb, q, k, v, mask)
  to (activations, mask).
  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:
    - splits queries, keys, and values into multiple 'heads',
    - splits positional embeddings into multiple 'heads',
    - computes per-head attention weights from per-head (queries, keys),
    - applies mask to screen out positions that come from padding tokens,
    - [in `'train'` mode] applies dropout to attention weights,
    - uses attention weights to combine per-head values vectors, and
    - merges per-head results into outgoing activations matching original input
      activation vector shapes.
  """

  def __init__(self,
               total_kv_pooling,
               n_heads=1,
               dropout=0.0,
               n_raw_tokens_generated=1,
               max_inference_length=3072,
               chunk_len=None,
               chunk_offset=None,
               mode='train'):
    """Returns a new PureAttention instance.

    Args:
      total_kv_pooling: Total shorten factor used in the model
      n_heads: Number of attention heads.
      dropout: Probabilistic rate for dropout applied to attention strengths
        (based on query-key pairs) before applying them to values.
      n_raw_tokens_generated: Number of tokens generated in a single pass
        through this layer. Used only in 'predict' non-training mode.
      max_inference_length: Maximum sequence length allowed in non-training
        modes.
      chunk_len (optional): Number of tokens per chunk. Setting this option will
        enable chunked attention.
      chunk_offset (optional): Offset for shifting chunks, for shifted chunked
        attention.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=7, n_out=2)
    self._total_kv_pooling = total_kv_pooling
    self._n_heads = n_heads
    self._dropout = dropout
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._max_len = max_inference_length
    self._chunk_len = chunk_len
    self._chunk_offset = chunk_offset
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A (location_bias, context_bias, pos_emb, q, k, v, mask) tuple.
    """
    location_bias, context_bias, pos_emb, q, k, v, mask = inputs

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    if self._mode == 'predict':
      self._fast_inference_update_state((k, v), self.state)
      (k, v, _) = self.state

    per_head_results, dots = DotProductAttention(
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(q),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(k),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(v),
        pos_emb.reshape((-1, n_heads, d_feature // n_heads)),
        context_bias,
        location_bias,
        mask,
        dropout=self._dropout,
        mode=self._mode,
        rng=self.rng,
        chunk_len=self._chunk_len,
        chunk_offset=self._chunk_offset)
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(
        n_heads, merged_batch_and_head=False).forward(per_head_results)
    return merged_results, mask

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      cache_signature = input_signature[4:6]
      self.state = self._fast_inference_init_state(cache_signature)

  def _fast_inference_init_state(self, input_signature):
    """Returns an initial state for causal attention layer fast inference."""

    def zeros_for_shape(bs, tokens_len, shape_dtype):
      shape, dtype = shape_dtype.as_tuple()
      d_feature = shape[-1]

      return jnp.zeros((bs, tokens_len, d_feature), dtype=dtype)

    batch_size = input_signature[0].shape[0]
    n_tokens = self._chunk_len if self._chunk_len is not None else self._max_len
    k = zeros_for_shape(batch_size, n_tokens, input_signature[0])
    v = zeros_for_shape(batch_size, n_tokens, input_signature[1])
    return k, v, jnp.array(0)

  def _fast_inference_update_state(self, inputs, state):
    """Updates state of a causal attention layer for fast inference.

    The layer state stores arrays with cached values of keys and values,
    as well as an index. To make shapes static, keys and values in the state are
    long, and the index indicates where the new keys and values from inputs need
    to be appended.

    During update, we append new_keys and new_values to keys and values at
    position given by index. And we increment index by length of new keys.
    We also create a mask to be 1 at appropriate positions (causal mask).

    Args:
      inputs: a double (new_keys, new_values)
      state: layer state with (keys, values, index)
    """
    # Fast inference: run step-by-step, storing the sequence
    # of keys and values calculated so far in state.
    new_k, new_v = inputs
    length = new_k.shape[1]
    (ks, vs, idx) = state

    # We cannot generate more than one token because it contradicts
    # all autoregressive properties
    assert length == 1

    new_index = idx // self._total_kv_pooling

    if self._chunk_len is not None:
      if self._chunk_offset != 0:
        new_index -= self._chunk_offset * (new_index >= self._chunk_offset)

      new_index = new_index % self._chunk_len

    # Keys and values are of shape [batch_size, length, d_kv].
    ks = fastmath.dynamic_update_slice_in_dim(ks, new_k, new_index, axis=1)
    vs = fastmath.dynamic_update_slice_in_dim(vs, new_v, new_index, axis=1)

    self.state = ks, vs, idx + self._n_raw_tokens_generated


def DotProductAttention(queries, keys, values, pos_emb, context_bias,
                        location_bias, mask, dropout, mode, rng, chunk_len,
                        chunk_offset):
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
    pos_emb: Per-head activations representing positional embeddings.
    context_bias: Global context bias from Transformer XL's attention.
    location_bias: Global location bias from Transformer XL's attention.
    mask: Mask that distinguishes positions with real content vs. padding.
    dropout: Probabilistic rate for dropout applied to attention strengths
      (based on query-key pairs) before applying them to values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
    rng: Single-use random number generator (JAX PRNG key).
    chunk_len (optional): Number of tokens per chunk. Setting this option will
      enable chunked attention.
    chunk_offset (optional): Offset for shifting chunks, for shifted chunked
      attention.

  Returns:
    Per-head activations resulting from masked per-head attention-weighted
    sum of per-head values.
  """
  batch_size, n_heads, original_l, d_feature = queries.shape

  def _calc_attn_scores(q, k):
    ac = jnp.einsum('bnid,bnjd->bnij', q + context_bias, k)
    bd = jnp.einsum('bnid,jnd->bnij', q + location_bias, pos_emb)

    if mode != 'predict':
      bd = _fast_matrix_shift(bd)

    dots = (ac + bd) / jnp.sqrt(d_feature)
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))

    # Softmax.
    dots = jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True))
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if dropout is not None and dropout > 0.0 and mode == 'train':
      keep = fastmath.random.bernoulli(rng, 1.0 - dropout, dots.shape)
      dots = jnp.where(keep, dots / (1.0 - dropout), jnp.zeros_like(dots))

    return dots

  if chunk_len is None or mode == 'predict':
    full_dots = _calc_attn_scores(queries, keys)
    out = jnp.matmul(full_dots, values)
  else:
    assert original_l % chunk_len == 0 and original_l >= chunk_len

    def chunk_split(v):
      total_len = v.shape[2]
      assert total_len % chunk_len == 0
      n_chunks = total_len // chunk_len

      chunked_shape = (batch_size, n_heads, n_chunks, chunk_len, d_feature)
      v = jnp.reshape(v, chunked_shape)
      v = v.swapaxes(1, 2)
      return jnp.reshape(v,
                         (batch_size * n_chunks, n_heads, chunk_len, d_feature))

    def chunk_join(v, total_len=original_l):
      assert total_len % chunk_len == 0
      n_chunks = total_len // chunk_len
      swapped_shape = (batch_size, n_chunks, n_heads, chunk_len, d_feature)
      v = jnp.reshape(v, swapped_shape)
      v = v.swapaxes(1, 2)
      return jnp.reshape(v, (batch_size, n_heads, total_len, d_feature))

    if chunk_offset == 0:
      queries, keys, values = map(chunk_split, [queries, keys, values])
      chunked_dots = _calc_attn_scores(queries, keys)
      chunked_result = jnp.matmul(chunked_dots, values)
      out = chunk_join(chunked_result)
    else:
      assert chunk_len > chunk_offset
      last_chunk_len = chunk_len - chunk_offset

      def split_along_l(v, mid_start, mid_end, end):
        pre = jnp.take(v, indices=range(mid_start), axis=2)
        mid = jnp.take(v, indices=range(mid_start, mid_end), axis=2)
        post = jnp.take(v, indices=range(mid_end, end), axis=2)
        return pre, mid, post

      def pad_to_chunk_len(v):
        width = [(0, 0)] * v.ndim
        width[2] = (0, chunk_len - v.shape[2])
        return jnp.pad(v, width, mode='constant', constant_values=0.0)

      def pad_borders(v):
        total_len = v.shape[2]
        pre, mid, post = split_along_l(v, chunk_offset,
                                       total_len - last_chunk_len, total_len)
        pre, post = map(pad_to_chunk_len, [pre, post])
        return jnp.concatenate([pre, mid, post], axis=2)

      def unpad_borders(v):
        padded_total_len = v.shape[2]
        assert padded_total_len == original_l + chunk_len
        pre_padded, mid, post_padded = split_along_l(
            v, chunk_len, padded_total_len - chunk_len, padded_total_len)
        pre = jnp.take(pre_padded, indices=range(chunk_offset), axis=2)
        post = jnp.take(post_padded, indices=range(last_chunk_len), axis=2)
        return jnp.concatenate([pre, mid, post], axis=2)

      queries, keys, values = map(lambda x: chunk_split(pad_borders(x)),
                                  [queries, keys, values])
      permuted_dots = _calc_attn_scores(queries, keys)
      permuted_out = chunk_join(
          jnp.matmul(permuted_dots, values), total_len=original_l + chunk_len)

      out = unpad_borders(permuted_out)

  out = out.astype(jnp.float32)
  return out, None  # We don't store full dots matrix


def calc_predict_next_token_index(state, total_kv_pooling, max_len, chunk_len,
                                  chunk_offset):
  """Arithmetic calculation for the current_token and sequence_length."""
  current_token = state // total_kv_pooling
  sequence_length = max_len

  if chunk_len is not None:
    if chunk_offset != 0:
      current_token -= chunk_offset * (current_token >= chunk_offset)
    current_token = current_token % chunk_len
    sequence_length = chunk_len
  return current_token, sequence_length


class PositionalEmbeddings(base.Layer):
  """Positional embedding for relative attention.

  Returns a layer that based on queries, keys and accumulated pool size of
  keys/values until this layer calculates sinusoidal positional embeddings
  for relative attention calculations.
  """

  def __init__(self,
               d_feature,
               total_kv_pooling,
               max_inference_length=3072,
               chunk_len=None,
               chunk_offset=None,
               n_raw_tokens_generated=1,
               mode='train'):
    """The init method of positional embeddings.

    Args:
      d_feature: Depth/dimensionality of feature embedding.
      total_kv_pooling: Accumulated pool size of keys/values until this layer.
      max_inference_length: Maximum sequence length allowed in non-training
        modes.
      chunk_len (optional): Number of tokens per chunk. Setting this option will
        enable chunked attention.
      chunk_offset (optional): Offset for shifting chunks, for shifted chunked
      attention.
      n_raw_tokens_generated: Number of tokens generated in a single pass
        through this layer. Used only in 'predict' non-training mode.
      mode: One of `'train'`, `'eval'`, or `'predict'`.

    Returns:
      Positional embedding.
    """
    super().__init__(n_in=1, n_out=1)
    self._d_feature = d_feature
    self._total_kv_pooling = total_kv_pooling
    self._max_len = max_inference_length
    self._chunk_len = chunk_len
    self._chunk_offset = chunk_offset
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._mode = mode

  def forward(self, inputs):
    positions = self.PositionsVectors(inputs.shape[1])
    pos_emb = Sinusoidal_Embeddings(positions, self._d_feature)
    return pos_emb

  def PositionsVectors(self, n_tokens):
    if self._mode == 'predict':
      current_token, sequence_length = calc_predict_next_token_index(
          self.state, self._total_kv_pooling, self._max_len, self._chunk_len,
          self._chunk_offset)
      positions = jnp.arange(0, sequence_length, 1.0) - current_token
      self.state = self.state + self._n_raw_tokens_generated
      return positions

    sequence_length = self._chunk_len if self._chunk_len is not None else n_tokens
    offset = sequence_length - 1  # offset to be compatible with predict mode
    positions = jnp.arange(sequence_length) - offset

    return positions

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      self.state = jnp.array(0)


def Sinusoidal_Embeddings(positions, d_feature):
  """Sinusoidal Embeddings.

  Computes out of 1-D integer absolute position vector the sinusoidal
  embeddings defined like in paper Attention is all you need (2017).
  Embeddings are shaped (positions, d_feature).

  Args:
    positions: a one-dimensional array of positions.
    d_feature: the number of sin-cos features.

  Returns:
    Positional embeddings.
  """
  inv_freq = 1 / (10000**(jnp.arange(0.0, d_feature, 2.0) / d_feature))
  sinusoid_freq = jnp.einsum('i,j->ij', positions, inv_freq)
  pos_emb = jnp.concatenate(
      [jnp.sin(sinusoid_freq), jnp.cos(sinusoid_freq)], axis=1)
  return pos_emb


def _fast_matrix_shift(x):
  # Implements necessary shift for relative positional attention calculations.
  shift = 1
  batch_size, n_head = x.shape[0], x.shape[1]
  queries_len, keys_len = x.shape[2], x.shape[3]
  zero_pad = jnp.zeros((batch_size, n_head, queries_len, shift))
  x = jnp.concatenate([zero_pad, x], axis=3)
  x = x.reshape(batch_size, n_head, keys_len + shift, queries_len)
  x = x[:, :, shift:, :]
  return x


class AttentionMaskLayer(base.Layer):
  """Creates attention mask layer.

  Returns:
    Returns a layer that based on queries, keys and accumulated pool size of
    keys/values until this layer calculates positional embeddings for
    causal relative attention calculations.

    Takes as input q, k, v and appends proper mask in the end.

    Causal attention uses masking to prevent a given sequence position from
    attending to positions greater than / following it. This is used, for
    example, when training autoregressive sequence models, or when decoding a
    sequence symbol by symbol.
  """

  def __init__(self,
               total_kv_pooling=1,
               max_inference_length=3072,
               chunk_len=None,
               chunk_offset=None,
               n_raw_tokens_generated=1,
               mode='train'):
    super().__init__(n_in=1, n_out=1)
    self._total_kv_pooling = total_kv_pooling
    self._max_len = max_inference_length
    self._chunk_len = chunk_len
    self._chunk_offset = chunk_offset
    self._n_raw_tokens_generated = n_raw_tokens_generated
    self._mode = mode

  def forward(self, inputs):
    inputs_len = inputs.shape[1]

    if self._mode == 'predict':
      # We cannot generate more than one token because it contradicts
      # all autoregressive properties
      assert inputs_len == 1

      current_token, sequence_length = calc_predict_next_token_index(
          self.state, self._total_kv_pooling, self._max_len, self._chunk_len,
          self._chunk_offset)

      mask = jnp.arange(sequence_length) <= current_token
      mask = jnp.reshape(mask, (1, sequence_length))
      self.state += self._n_raw_tokens_generated
      return mask

    if self._chunk_len is not None:
      return jnp.tril(
          jnp.ones((self._chunk_len, self._chunk_len), dtype=jnp.bool_))

    return jnp.tril(jnp.ones((inputs_len, inputs_len), dtype=jnp.bool_))

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      self.state = jnp.array(0)
