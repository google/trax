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


def RelativeAttentionWrapper(d_feature,
                             n_heads=1,
                             dropout=0.0,
                             max_inference_length=2048,
                             mode='train',
                             context_bias_layer=None,
                             location_bias_layer=None,
                             total_pooling=None):
  """Relative attention wrapper.

  Args:
    d_feature: Last/innermost dimension of activations in the input to and
      output from this layer.
    n_heads: Number of attention heads. Attention heads effectively split
      activation vectors into ``n_heads`` subvectors, of size ``d_feature /
      n_heads``.
    dropout: dropout rate.
    max_inference_length: max inference length.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
    context_bias_layer: context bias layer.
    location_bias_layer: location bias layer.
    total_pooling: total pooling.

  Returns:
    relative attention layer.

  Relative attention wrapper for compatibility with configurable attention,
  so that it can be called by `ApplyAttentionLayer`.
  """
  del max_inference_length

  attention = RelativeAttentionLMLayer(
      d_feature,
      context_bias_layer,
      location_bias_layer,
      total_pooling,
      n_heads=n_heads,
      dropout=dropout,
      mode=mode)

  return cb.Serial(cb.Select([0, 0, 0]), attention)


def get_rel_att_inputs(d_model, n_heads):
  """Global relative attentions bias initialization shared across layers."""
  assert d_model % n_heads == 0 and d_model % 2 == 0
  d_head = d_model // n_heads

  bias_initializer = init.RandomNormalInitializer(1e-6)
  context_bias_layer = core.Weights(
      bias_initializer, shape=(1, n_heads, 1, d_head))
  location_bias_layer = core.Weights(
      bias_initializer, shape=(1, n_heads, 1, d_head))
  return context_bias_layer, location_bias_layer


@assert_shape('bSq,blk,blv,b1xl->bSd,b1xl')
def RelativeAttentionLayer(d_feature,
                           context_bias_layer,
                           location_bias_layer,
                           total_kv_pooling,
                           separate_cls,
                           n_heads=1,
                           dropout=0.0,
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
    context_bias_layer: Global context bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers
    location_bias_layer: Global location bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer
    separate_cls: True/False if we separate_cls in calculations.

    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  return cb.Serial(
      cb.Branch(
          PositionalEmbeddings(d_feature, separate_cls, total_kv_pooling),
          cb.Select([0]), cb.Select([1])),
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      context_bias_layer,
      location_bias_layer,
      RelativeAttention(  # pylint: disable=no-value-for-parameter
          separate_cls=separate_cls,
          n_heads=n_heads,
          dropout=dropout,
          mode=mode),
      core.Dense(d_feature),
  )


@assert_shape('bSq,blk,blv->bSd')
def RelativeAttentionLMLayer(d_feature,
                             context_bias_layer,
                             location_bias_layer,
                             total_kv_pooling,
                             separate_cls=False,
                             n_heads=1,
                             dropout=0.0,
                             mode='train'):
  """Returns a layer that maps (q, k, v) to (activations).

  Same as standard Relative attention layer but additionally based on sizes
  of queries and keys prepares a mask that masks out the future.
  Masking the future is the concept primarily used for Language Modelling.
  Args:
    d_feature: Depth/dimensionality of feature embedding.
    context_bias_layer: Global context bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers
    location_bias_layer: Global location bias from Transformer XL's attention.
      There should be one such layer shared for all relative attention layers.
    total_kv_pooling: Accumulated pool size of keys/values used at this layer.
    separate_cls: True/False if we separate_cls in calculations.
    n_heads: Number of attention heads.
    dropout: Probabilistic rate for internal dropout applied to attention
      activations (based on query-key pairs) before dotting them with values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  attention = RelativeAttentionLayer(
      d_feature,
      context_bias_layer,
      location_bias_layer,
      total_kv_pooling,
      separate_cls,
      n_heads=n_heads,
      dropout=dropout,
      mode=mode)

  return cb.Serial(
      CreateAttentionMaskLayer(),  # q, k, v, mask
      attention,  # vecs, mask
      cb.Select([0], n_in=2),  # vecs
  )


class RelativeAttention(base.Layer):
  """Relative attention layer.

  Layer that maps (location_bias, context_bias, pos_emb, q, k, v, mask)
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

  def __init__(self, separate_cls, n_heads=1, dropout=0.0, mode='train'):
    """Returns a new PureAttention instance.

    Args:
      separate_cls: True/False if we separate_cls in calculations.
      n_heads: Number of attention heads.
      dropout: Probabilistic rate for dropout applied to attention strengths
          (based on query-key pairs) before applying them to values.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(n_in=7, n_out=2)
    self._separate_cls = separate_cls
    self._n_heads = n_heads
    self._dropout = dropout
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

    per_head_results, dots = DotProductAttention(
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(q),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(k),
        SplitIntoHeads(n_heads, merged_batch_and_head=False).forward(v),
        pos_emb.reshape((-1, n_heads, d_feature // n_heads)),
        context_bias,
        location_bias,
        mask,
        separate_cls=self._separate_cls,
        dropout=self._dropout,
        mode=self._mode,
        rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    merged_results = MergeHeads(
        n_heads, merged_batch_and_head=False).forward(per_head_results)
    return merged_results, mask


def DotProductAttention(queries, keys, values, pos_emb, context_bias,
                        location_bias, mask, separate_cls, dropout, mode, rng):
  """Computes new activations via masked attention-weighted sum of values.

  Args:
    queries: Per-head activations representing attention queries.
    keys: Per-head activations representing attention keys.
    values: Per-head activations to be combined by computed attention weights.
    pos_emb: Per-head activations representing positional embeddings.
    context_bias: Global context bias from Transformer XL's attention.
    location_bias: Global location bias from Transformer XL's attention.
    mask: Mask that distinguishes positions with real content vs. padding.
    separate_cls: True/False if we separate_cls in calculations.
    dropout: Probabilistic rate for dropout applied to attention strengths
      (based on query-key pairs) before applying them to values.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Per-head activations resulting from masked per-head attention-weighted
    sum of per-head values.

  This function is the core of the attention mechanism. It:
    - computes per-head attention weights from per-head `queries` and `keys`,
    - applies `mask` to screen out positions that come from padding tokens,
    - optionally applies dropout to attention weights, and
    - uses attention weights to combine per-head `values` vectors.
  """
  d_feature = queries.shape[-1]
  keys_len, queries_len = keys.shape[-2], queries.shape[-2]
  funnel_factor, is_upsampling = calc_funnel_ratio(keys_len, queries_len)

  ac = jnp.einsum('bnid,bnjd->bnij', queries + context_bias, keys)
  bd = jnp.einsum('bnid,jnd->bnij', queries + location_bias, pos_emb)
  bd = _fast_matrix_shift(bd, funnel_factor, is_upsampling)

  if separate_cls:
    # Masking out location part of attention for cls token
    bd = bd.at[:, :, :, 0].set(0)
    bd = bd.at[:, :, 0, :].set(0)

  dots = (ac + bd) / jnp.sqrt(d_feature)
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


def PositionalEmbeddings(d_feature, separate_cls, total_kv_pooling):
  """Positional embeddings.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    separate_cls: True/False if we separate_cls in calculations.
    total_kv_pooling: Accumulated pool size of keys/values until this layer.

  Returns:
    a layer that based on queries, keys and accumulated pool size of
    keys/values until this layer calculates sinusoidal positional embeddings
    for relative attention calculations.
  """

  def PositionsVectors(queries, keys):
    assert not separate_cls

    keys_len, queries_len = keys.shape[-2], queries.shape[-2]
    funnel_factor, is_upsampling = calc_funnel_ratio(keys_len, queries_len)

    if funnel_factor == 1:
      offset = keys_len - 1
      positions = (jnp.arange(keys_len) - offset) * total_kv_pooling
    else:
      if is_upsampling:
        positions = jnp.arange(-queries_len + 1, queries_len, 1.0)
      else:
        positions = jnp.arange(-keys_len + 1, keys_len, 1.0) * total_kv_pooling

    return positions

  def Sinusoidal_Embeddings(positions):
    inv_freq = 1 / (10000**(jnp.arange(0.0, d_feature, 2.0) / d_feature))
    sinusoid_freq = jnp.einsum('i,j->ij', positions, inv_freq)
    pos_emb = jnp.concatenate(
        [jnp.sin(sinusoid_freq), jnp.cos(sinusoid_freq)], axis=1)
    return pos_emb

  return cb.Serial(
      cb.Fn('Generate positions vectors', PositionsVectors, n_out=1),
      cb.Fn(
          'Transform to sinusoidal encodings', Sinusoidal_Embeddings, n_out=1))


def calc_funnel_ratio(keys_len, queries_len):
  """Calculate funnel ratio."""

  if queries_len > keys_len:  # Upsampling
    assert queries_len % keys_len == 0
    funnel_factor = queries_len // keys_len
    is_upsampling = True
  else:  # Downsampling
    assert keys_len % queries_len == 0
    funnel_factor = keys_len // queries_len
    is_upsampling = False

  return funnel_factor, is_upsampling


def _fast_matrix_shift(x, funnel_factor=1, is_upsampling=False):
  """Fast matrix shift."""

  if funnel_factor == 1 and not is_upsampling:
    shift = 1
    batch_size, n_head = x.shape[0], x.shape[1]
    queries_len, keys_len = x.shape[2], x.shape[3]
    zero_pad = jnp.zeros((batch_size, n_head, queries_len, shift))
    x = jnp.concatenate([zero_pad, x], axis=3)
    x = x.reshape(batch_size, n_head, keys_len + shift, queries_len)
    x = x[:, :, shift:, :]
    return x

  if is_upsampling:
    k = funnel_factor
    shift = 1
  else:
    k = 1
    shift = funnel_factor

  bsz, n_head = x.shape[0], x.shape[1]
  qlen, klen = x.shape[2], (x.shape[3] + 1) // 2

  zero_pad = jnp.zeros((bsz, n_head, qlen, shift))
  x = jnp.concatenate([zero_pad, x], axis=3)
  x = x.reshape(bsz, n_head, 2 * klen - 1 + shift, qlen)
  x = x[:, :, shift:, :]
  x = x.reshape(bsz, n_head, qlen, klen * 2 - 1)
  x = x[:, :, :, shift - 1:shift - 1 + klen:k]
  return x


@assert_shape('bqd,bkd,bvd->bqd,bkd,bvd,b1qk')
def CreateAttentionMaskLayer():
  """Creates attention mask layer.

  Returns a layer that based on queries, keys and accumulated pool size of
  keys/values until this layer calculates positional embeddings for
  causal relative attention calculations.

  Takes as input q, k, v and appends proper mask in the end.
  Causal attention uses masking to prevent a given sequence position from
  attending to positions greater than / following it. This is used, for
  example, when training autoregressive sequence models, or when decoding a
  sequence symbol by symbol.

  Returns:
    an attention mask layer.
  """

  def calculate_mask(queries, keys):
    batch_size = queries.shape[0]
    keys_len, queries_len = keys.shape[-2], queries.shape[-2]
    funnel_factor, is_upsampling = calc_funnel_ratio(keys_len, queries_len)

    return _funnel_mask(batch_size, keys_len, queries_len, funnel_factor,
                        is_upsampling)

  def _funnel_mask(batch_size, keys_len, queries_len, funnel_factor,
                   is_upsampling):
    """Funnel mask.

    Args:
      batch_size: batch size.
      keys_len: keys length.
      queries_len: queries length.
      funnel_factor: funnel factor.
      is_upsampling: True or False.

    Returns:
      funnel mask.

    This function based on keys/queries lengths creates a triangle mask
    that prevents tokens from attending to positions following it.

    If funnel_factor is not equal to 1 due to funnel upsampling or
    downsampling it adjusts created mask for funnel attention
    by repeating each element funnel_factor times.

    This is because after funnel layer one token attends to funnel_factor
    different tokens in downsampling. During upsampling on the other hand
    funnel_factor tokens are attending to single token before upsampling.
    """

    if funnel_factor != 1:
      if not is_upsampling:
        mask = jnp.tril(jnp.ones((queries_len, queries_len), dtype=jnp.bool_))
        mask = jnp.repeat(mask, funnel_factor, axis=-1)
      else:
        mask = jnp.tril(jnp.ones((keys_len, keys_len), dtype=jnp.bool_))
        mask = jnp.repeat(mask, funnel_factor, axis=-2)
    else:
      mask = jnp.tril(jnp.ones((queries_len, queries_len), dtype=jnp.bool_))

    return jnp.repeat(mask[None, None, :, :], batch_size, axis=0)

  return cb.Branch(
      cb.Select([0]), cb.Select([1]), cb.Select([2]),
      cb.Fn('create attention mask layer', calculate_mask, n_out=1))


@assert_shape('...d->...d')
def ShiftRightCls(cls_id):
  """Shifts right and insert cls.

  Args:
    cls_id: id of the cls token in embedding dictionary.  Returns a layer that
      shifts input tokens to the right by one and inserts an cls token to the
      beginning like in BERT paper.

  Returns:
    layer shifting to right and inserting cls.
  """

  def shift_right(x):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(cls_id))
    return padded[:, :-1]

  return cb.Fn('ShiftRightCls()', shift_right)
