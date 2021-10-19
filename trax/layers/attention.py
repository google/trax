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
r"""Attention-related layers, as used in Transformer(-like) models.

Attention is a trainable mechanism for mapping between collections of vectors:

.. math::
    \text{Attention}: \mathbf{X}^{n} \rightarrow \mathbf{X}^{n}\!,
    \ \text{for} \ \mathbf{X} \in \mathbb{R}^d

Whereas classic neural networks assemble nodes of *numbers* with weighted
connections:

    - node activations: floating point values (one float per node)
    - inter-node connections: trainable weights (one float per connection),

attention lets one assemble nodes of *vectors* and use further vectors to
calculate connection strengths:

    - node activations: floating point vectors, and
    - inter-node connections: computed using trainable vectors.

Computing connection strengths involves several concepts -- queries, keys,
values, masks, attention heads -- that factor heavily into the API below.

NOTE: Attention, positional encoding, and shift layers in this module include
``mode``-dependent behavior. The possible modes are:

    - ``'train'``: in training -- dropouts and position shifts active
    - ``'eval'``:  in evals -- dropouts inactive, position shifts active
    - ``'predict'``: in prediction -- dropouts and position shifts inactive
"""

import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import convolution
from trax.layers import core
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn
from trax.layers.research import sparsity


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


# inputs are [batch, length, depth], [batch, 1, 1 length]
@assert_shape('bld,b11l->bld,b11l')
def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Returns a layer that maps `(vectors, mask)` to `(new_vectors, mask)`.

  This layer type represents one pass of multi-head self-attention, from vector
  set to vector set, using masks to represent out-of-bound (e.g., padding)
  positions. It:

    - makes three copies of incoming activations and maps these to multi-head
      query (Q) vectors, key (K) vectors, and value (V) vectors, respectively;
    - for each head, computes the scaled dot product of each Q-K pair;
    - applies mask to screen out positions that come from padding tokens
      (indicated by 0 value);
    - [in ``'train'`` mode] applies dropout to Q-K dot products;
    - for each head, computes Q-K attention strengths using a per-query softmax
      of the Q-K dot products;
    - for each head, for each query position, combines V vectors according
      to the Q-K attention strengths; and
    - concatenates and fuses resulting per-head vectors into outgoing
      activations matching original input activation shapes.

  Args:
    d_feature: Last/innermost dimension of activations in the input to and
        output from this layer.
    n_heads: Number of attention heads. Attention heads effectively split
        activation vectors into ``n_heads`` subvectors, of size
        ``d_feature / n_heads``.
    dropout: Probababilistic rate for attention dropout, which overrides
        (sets to zero) some attention strengths derived from query-key
        matching. As a result, on a given forward pass, some value vectors
        don't contribute to the output, analogous to how regular dropout can
        cause some node activations to be ignored. Applies only if layer is
        created in ``'train'`` mode.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
  """
  return cb.Serial(
      cb.Select([0, 0, 0]),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  )


@assert_shape('bSq,blk,blv,b1xl->bSd,b1xl')
def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train',
                 cache_KV_in_predict=False, q_sparsity=None,
                 result_sparsity=None):
  """Returns a layer that maps `(AQ, AK, AV, mask)` to `(new-A, mask)`.

  Unlike :py:class:`Attention` above, :py:class:`AttentionQKV` allows the
  incoming activations (`AQ`, `AK`, and `AV`) to come from different sources.
  This is used, for instance, in encoder-decoder attention (Q-related
  activations `AQ` from the decoder, K- and V-related activations -- `AK` and
  `AV` -- from the encoder). Otherwise, see the :py:class:`Attention`
  description for further context/details.

  Args:
    d_feature: Last/innermost dimension of activations in the input to and
        output from this layer.
    n_heads: Number of attention heads. Attention heads effectively split
        activation vectors into ``n_heads`` subvectors, of size
        ``d_feature / n_heads``.
    dropout: Probababilistic rate for attention dropout, which overrides
        (sets to zero) some attention strengths derived from query-key
        matching. As a result, on a given forward pass, some value vectors
        don't contribute to the output, analogous to how regular dropout can
        cause some node activations to be ignored. Applies only if layer is
        created in ``'train'`` mode.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
    cache_KV_in_predict: Whether to cache K/V arrays in ``'predict'`` mode.
    q_sparsity: Sparsity with which to process queries. If ``None``,
        :py:class:`Dense` is used; if ``'noop'``, no processing is used.
    result_sparsity: Sparsity with which to process result of the attention.
        If ``None``, :py:class:`Dense` is used; if ``'noop'``, no processing is
        used.
  """
  def _SparsifiableDense(layer_sparsity):
    if layer_sparsity is None:
      return core.Dense(d_feature)
    elif layer_sparsity == 'noop':
      return cb.Serial()  # No-op layer.
    else:
      d_module = d_feature // layer_sparsity
      return cb.Serial(
          sparsity.FactoredDense(layer_sparsity, d_feature, d_feature),
          sparsity.LocallyConvDense(layer_sparsity, d_module, mode=mode,
                                    kernel_size=3, length_kernel_size=3)
      )

  def _CacheableDense():
    if cache_KV_in_predict and mode == 'predict':
      return cb.Cache(core.Dense(d_feature))
    else:
      return core.Dense(d_feature)

  def _PureAttention():
    return PureAttention(n_heads=n_heads, dropout=dropout, mode=mode)

  return cb.Serial(
      cb.Parallel(_SparsifiableDense(q_sparsity),
                  _CacheableDense(),
                  _CacheableDense()),
      _PureAttention(),
      _SparsifiableDense(result_sparsity),
  )


# 'k' is number of keys/values, while 'l' is number of queries. Typically they
# will be the same, but it is not necessary.
@assert_shape('blq,bkq,bkd,b1xk->bld,b1xk')
class PureAttention(base.Layer):
  """Returns a layer that maps `(Q, K, V, mask)` to `(activations, mask)`.

  This layer type performs the inner workings of one pass of multi-head
  self-attention. It:

    - subdivides incoming Q/K/V activations into multi-head versions;
    - for each head, computes the scaled dot product of each Q-K pair;
    - applies mask to screen out positions that come from padding tokens
      (indicated by 0 value);
    - [in ``'train'`` mode] applies dropout to Q-K dot products;
    - for each head, computes Q-K attention strengths using a per-query softmax
      of the Q-K dot products;
    - for each head, for each query position, combines V vectors according
      to the Q-K attention strengths; and
    - concatenates and fuses resulting per-head vectors into outgoing
      activations matching original input activation shapes.
  """

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    """Returns a new :py:class:`PureAttention` instance.

    Args:
      n_heads: Number of attention heads.
      dropout: Probababilistic rate for attention dropout, which overrides
          (sets to zero) some attention strengths derived from query-key
          matching. As a result, on a given forward pass, some value vectors
          don't contribute to the output, analogous to how regular dropout can
          cause some node activations to be ignored. Applies only if layer is
          created in ``'train'`` mode.
      mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
    """
    super().__init__(n_in=4, n_out=2)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed activations and unmodified mask.

    Args:
      inputs: A `(Q, K, V, mask)` tuple, whose query, key, and value
          activations have not yet been subdivided into heads.
    """
    q, k, v, mask = inputs

    d_feature = q.shape[-1]
    n_heads = self._n_heads
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Dimensionality of feature embedding ({d_feature}) is not a '
          f'multiple of the requested number of attention heads ({n_heads}).')

    per_head_results, dots = _per_head_attention(
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


def _per_head_attention(queries, keys, values, mask, dropout, mode, rng):
  """Computes new per-head activations via scaled dot-product attention.

  This function is the core of the attention mechanism. Given per-head
  ``queries`` (Q), ``keys`` (K), ``values`` (V), and ``mask``, it:

    - computes the scaled dot product of each Q-K pair;
    - applies ``mask`` to screen out positions that come from padding tokens
      (indicated by 0 value);
    - [in ``'train'`` mode] applies dropout to Q-K dot products;
    - computes Q-K attention strengths using a per-query softmax of the Q-K dot
      products; and
    - for each query position, combines V vectors according to the Q-K
      attention strengths.

  Args:
    queries: Per-head activations representing attention queries.
    keys: Per-head activations representing attention keys.
    values: Per-head activations to be combined by computed attention strengths.
    mask: Mask that distinguishes positions with real content vs. padding.
    dropout: Probababilistic rate for attention dropout, which overrides
        (sets to zero) some attention strengths derived from query-key
        matching. As a result, on a given forward pass, some value vectors
        don't contribute to the output, analogous to how regular dropout can
        cause some node activations to be ignored. Applies only in ``'train'``
        mode.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Tuple of (activations, attn_strengths), where activations are new per-head
    activation vectors and attn_strengths is a matrix of per-head attention
    strengths.
  """
  if dropout >= 1.0:
    raise ValueError(f'Dropout rate ({dropout}) must be lower than 1.')

  d_feature = queries.shape[-1]

  dots = jnp.matmul(queries, jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(d_feature)
  if mask is not None:
    dots = jnp.where(mask,
                     dots,
                     jnp.full_like(dots, -1e9))
  attn_strengths = (
      jnp.exp(dots - fastmath.logsumexp(dots, axis=-1, keepdims=True)))
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = fastmath.random.bernoulli(rng, 1.0 - dropout, attn_strengths.shape)
    attn_strengths = jnp.where(keep,
                               attn_strengths / (1.0 - dropout),
                               jnp.zeros_like(attn_strengths))
  activations = jnp.matmul(attn_strengths, values).astype(jnp.float32)
  attn_strengths = attn_strengths.astype(jnp.float32)
  return activations, attn_strengths


class DotProductAttention(base.Layer):
  """Returns a layer that computes per-head attention (via scaled dot-product).

  This layer computes the core of the attention mechanism. Given per-head
  queries (Q), keys (K), values (V), and mask, it:

    - computes the scaled dot product of each Q-K pair;
    - applies mask to screen out positions that come from padding tokens
      (indicated by 0 value);
    - [if created in ``'train'`` mode] applies dropout to Q-K dot products;
    - computes Q-K attention strengths using a per-query softmax of the Q-K dot
      products; and
    - for each query position, combines V vectors according to the Q-K
      attention strengths.
  """

  def __init__(self, dropout=0.0, mode='train'):
    """Creates a :py:class:`DotProductAttention` instance in a specific mode.

    Args:
      dropout: Probababilistic rate for attention dropout, which overrides
          (sets to zero) some attention strengths derived from query-key
          matching. As a result, on a given forward pass, some value vectors
          don't contribute to the output, analogous to how regular dropout can
          cause some node activations to be ignored. Applies only if layer is
          created in ``'train'`` mode.
      mode: One of ``'train'``, ``'eval'``, ``'predict'`` or ``'viz'``.
    """
    super().__init__(n_in=4, n_out=1)
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs):
    """Returns attention-computed per-head activations and unchanged mask.

    Args:
      inputs: A `(Q, K, V, mask)` tuple, whose query, key, and value
          activations have been subdivided into heads.
    """
    q, k, v, mask = inputs
    activations, attn_strengths = _per_head_attention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=self.rng)
    if self._mode == 'viz':
      self.state = attn_strengths
    return activations


# (b_size, seq_len, d_feature) --> (b_size*n_heads, seq_len, d_head)
@assert_shape('bld->...lh')
def SplitIntoHeads(n_heads, merged_batch_and_head=True):
  """Returns a layer that reshapes an array for multi-head computation."""
  def f(x):
    batch_size, seq_len, d_feature = x.shape
    if d_feature % n_heads != 0:
      raise ValueError(
          f'Feature embedding dimensionality ({d_feature}) is not a multiple'
          f' of the requested number of attention heads ({n_heads}).')

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
  """Returns a layer that rejoins heads, after multi-head computation."""
  def f(x):
    if merged_batch_and_head:
      dim_0, seq_len, d_head = x.shape
      if dim_0 % n_heads != 0:
        raise ValueError(
            f"Array's leading dimension ({dim_0}) is not a multiple of the"
            f" number of attention heads ({n_heads}).")

      batch_size = dim_0 // n_heads
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
  """Returns a configured multi-head self-attention layer.

  A :py:class:`ConfigurableAttention` layer acts similarly to
  :py:class:`Attention` layers, but with configurable components. It

    - makes three copies of incoming activations and uses ``q_layer``,
      ``k_layer``, and ``v_layer`` to map activations to multi-head query (Q)
      vectors, key (K) vectors, and value (V) vectors, respectively;
    - uses ``qkv_attention_layer`` to compute per-head attention, similar to
      :py:class:`DotProductAttention` or :py:class:`DotProductCausalAttention`;
    - concatenates and fuses resulting per-head vectors into activations
      matching original input activation shapes; and
    - applies a final layer, ``final_layer``, mapping activations to
      activations (with shape matching the original input activations).

  Args:
    q_layer: Layer that maps input activations to per-head query activations.
    k_layer: Layer that maps input activations to per-head key activations.
    v_layer: Layer that maps input activations to per-head value activations.
    final_layer: After main multi-head computation and rejoining of heads,
        layer that maps activations to activations (with shape matching the
        original input activations).
    qkv_attention_layer: Layer the does the core multi-head self-attention
        computation.
    n_heads: Number of attention heads. Attention heads effectively split
        activation vectors into ``n_heads`` subvectors, of size
        ``d_feature / n_heads``.
  """
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
def CausalAttention(d_feature,
                    n_heads=1,
                    dropout=0.0,
                    max_inference_length=2048,
                    use_dconv=False,
                    mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like :py:class:`Attention`, this layer type represents one pass of multi-head
  self-attention, but with causal masking rather than padding-based masking.

  Args:
    d_feature: Last/innermost dimension of activations in the input to and
        output from this layer.
    n_heads: Number of attention heads. Attention heads effectively split
        activation vectors into ``n_heads`` subvectors, of size
        ``d_feature / n_heads``.
    dropout: Probababilistic rate for attention dropout, which overrides
        (sets to zero) some attention strengths derived from query-key
        matching. As a result, on a given forward pass, some value vectors
        don't contribute to the output, analogous to how regular dropout can
        cause some node activations to be ignored. Applies only if layer is
        created in ``'train'`` mode.
    max_inference_length: Maximum sequence length allowed in non-training
        modes.
    use_dconv: if True, use depthwise convolutions on top of dense layers
      for Q, K and V.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
  """
  if d_feature % n_heads != 0:
    raise ValueError(
        f'Dimensionality of feature embedding ({d_feature}) is not a multiple '
        f'of the requested number of attention heads ({n_heads}).')

  def QKVLayer():
    """Function returning the Q, K and V layer."""
    if use_dconv:
      return cb.Serial(core.Dense(d_feature), convolution.CausalDepthwiseConv())
    else:
      return core.Dense(d_feature)

  return ConfigurableAttention(
      QKVLayer(),
      QKVLayer(),
      QKVLayer(),
      core.Dense(d_feature),
      n_heads=n_heads,
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
    """Creates a :py:class:`DotProductCausalAttention` instance.

    Args:
      dropout: Probababilistic rate for attention dropout, which overrides
          (sets to zero) some attention strengths derived from query-key
          matching. As a result, on a given forward pass, some value vectors
          don't contribute to the output, analogous to how regular dropout can
          cause some node activations to be ignored. Applies only if layer is
          created in ``'train'`` mode.
      max_inference_length: Maximum sequence length allowed in non-training
          modes.
      mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
    """
    super().__init__(n_in=3, n_out=1)
    self._dropout = dropout
    self._mode = mode
    self._max_len = max_inference_length
    self._portal_mask = self.monkey_patched_mask()  # pylint: disable=assignment-from-none

  def monkey_patched_mask(self):
    # This is necessary for Terraformer model. See comments there.
    # The mask will only be used in Terraformer in predict mode.
    return None

  def forward(self, inputs):
    """Returns attention-computed activations.

    Args:
      inputs: A (queries, keys, values) tuple.
    """
    q, k, v = inputs

    if self._portal_mask is not None:
      mask_for_predict = self._portal_mask.get_value()
    else:
      mask_for_predict = None

    if self._mode == 'predict':
      self.state, mask = _fast_inference_update_state(
          inputs, self.state,
          mask_for_predict=mask_for_predict)
      if self._portal_mask is not None:
        (_, k, v, _) = self.state
      else:
        (k, v, _) = self.state
    else:
      sequence_length = q.shape[-2]
      mask = _causal_mask(sequence_length)

    activations, attn_strengths = _per_head_attention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=self.rng)
    if self._mode == 'viz':
      self.state = attn_strengths
    return activations

  def init_weights_and_state(self, input_signature):
    """Initializes this layer for fast inference, if in ``'predict'`` mode."""
    if self._mode == 'predict':
      self.state = _fast_inference_init_state(
          input_signature, self._max_len,
          predict_mask=self._portal_mask)


def _causal_mask(length):
  # Not all backends define jnp.tril. However, using np.tril is inefficient
  # in that it creates a large global constant. TODO(kitaev): try to find an
  # alternative that works across all backends.
  if fastmath.is_backend(fastmath.Backend.JAX):
    return jnp.tril(jnp.ones((1, length, length), dtype=np.bool_), k=0)
  else:
    return np.tril(np.ones((1, length, length), dtype=np.bool_), k=0)


@assert_shape('...d->...d')
def ShiftRight(n_positions=1, mode='train'):
  """Returns a layer that can insert padding to shift the input sequence.

  Args:
    n_positions: Number of positions to shift the input sequence rightward;
        initial positions freed by the shift get padded with zeros. Applies
        only if layer is created in a non-``'eval'`` mode.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
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
  an N-D array that marks for each sequence position whether the integer (e.g.,
  a token ID) in that position represents padding -- value ``pad`` -- versus
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
          f'Input to PaddingMask must be a 2-D array with shape '
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
          f'Decoder input to EncoderDecoderMask must be a 3-D array with '
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
  ``'train'`` mode with a nonzero ``dropout`` value. For such a layer, on each
  forward pass a subset of sequence positions selected at random will *not*
  receive positional marking.
  """

  def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
               use_bfloat16=False, start_from_zero_prob=1.0,
               max_offset_to_add=0, d_feature=None, mode='train'):
    """Creates a :py:class:`PositionalEncoding` instance in a given mode.

    Args:
      max_len: Maximum input sequence length.
      dropout: Probability of *not* adding positional encoding to a sequence
          position. Applies only if layer is created in ``'train'`` mode.
      dropout_broadcast_dims: Axes along which dropout mask values are
          broadcast rather than individually set at random.
      use_bfloat16: If ``True``, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
      start_from_zero_prob: how often to start from 0 during training,
          (if 1.0, we always start from position 0, if less, we randomize).
      max_offset_to_add: maximum offset to add to the positions during training
        when randomizing; this offset plus input length must still be less than
        max_len for all training examples.
      d_feature: int or None; have this dimension for embeddings + shared FF if
        not None.
      mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
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
    self._start_from_zero_prob = start_from_zero_prob
    self._max_offset_to_add = max_offset_to_add
    self._mode = mode
    self._d_feature = d_feature

  def forward(self, inputs):
    """Returns the input activations, with added positional information."""
    weights = self.weights
    if self._d_feature is not None:
      weights, ff = weights
      weights = jnp.dot(weights[:inputs.shape[1], :], ff)
    if len(weights.shape) < 3:  # old checkpoints have 1 in first dim already
      weights = weights[None, :, :]  # [1, self._max_len, d_feature]
    if self._mode != 'predict':
      x = inputs
      symbol_size = jnp.shape(x)[1]
      if self._mode != 'train' or self._start_from_zero_prob >= 1.0:
        px = weights[:, :symbol_size, :]
      else:
        rng1, rng2 = fastmath.random.split(self.rng, 2)
        start = fastmath.random.randint(rng1, (), 0, self._max_offset_to_add)
        start_from_zero = fastmath.random.uniform(rng2, (), jnp.float32, 0, 1)
        start = jnp.where(start_from_zero < self._start_from_zero_prob,
                          jnp.zeros((), dtype=jnp.int32), start)
        px = fastmath.dynamic_slice_in_dim(weights, start, symbol_size,
                                           axis=1)
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
      # This positional encoding layer stores the index of the current
      # position and increments it on each call.
      emb = fastmath.dynamic_slice_in_dim(
          weights, self.state, inputs.shape[1], axis=1)
      self.state += inputs.shape[1]
      return inputs + emb

  def init_weights_and_state(self, input_signature):
    """Randomly initializes the positional encoding vectors.

    Args:
      input_signature: :py:class:`ShapeDtype` instance characterizing the input
          this layer should compute on.
    """
    d_feature = input_signature.shape[-1]
    if self._d_feature is not None:
      d_feature = self._d_feature
    pe = np.zeros((self._max_len, d_feature), dtype=np.float32)
    position = np.arange(0, self._max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)  # [self._max_len, d_feature]
    if self._use_bfloat16:
      pe = pe.astype(jnp.bfloat16)
    w = jnp.array(pe)  # Trainable parameters, initialized above.
    if self._d_feature is not None:
      ff = init.GlorotUniformInitializer()(
          (d_feature, input_signature.shape[-1]), self.rng)
      self.weights = w, ff
    else:
      self.weights = w
    if self._mode == 'predict':
      self.state = jnp.zeros((), dtype=jnp.int32)


def _zero_pad(x, pad, axis):
  """Helper for jnp.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return jnp.pad(x, pad_widths, mode='constant')


def _fast_inference_init_state(input_signature, buffer_length,
                               predict_mask=None):
  """Returns an initial state for causal attention layer fast inference."""
  def zeros_for(batch_size, shape_dtype):
    shape, dtype = shape_dtype.as_tuple()
    d_feature = shape[-1]
    return jnp.zeros((batch_size, buffer_length, d_feature), dtype=dtype)

  batch_size = input_signature[0].shape[0]
  k = zeros_for(batch_size, input_signature[1])
  v = zeros_for(batch_size, input_signature[2])
  if predict_mask is not None:
    mask_for_predict = jnp.zeros((buffer_length,)) != 0
    return (mask_for_predict, k, v, jnp.array(0))
  else:
    return (k, v, jnp.array(0))


def _fast_inference_update_state(inputs, state, mask_for_predict=None):
  """Updates state of a causal attention layer for fast inference.

  The layer state stores arrays with cached values of keys and values,
  as well as an index. To make shapes static, keys and values in the state are
  long, and the index indicates where the new keys and values from inputs need
  to be appended.

  During update, we append new_keys and new_values to keys and values at
  position given by index. And we increment index by length of new keys.
  We also create a mask to be 1 at appropriate positions (causal mask).

  Args:
    inputs: a triple (new_queries, new_keys, new_values)
    state: layer state with (keys, values, index)
    mask_for_predict: mask used for predict mode. This is used only in
      Terraformer.

  Returns:
    Updated state and mask to be used.
  """
  # Fast inference: run step-by-step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  if mask_for_predict is not None:
    (state_mask_for_predict, ks, vs, idx) = state
  else:
    (ks, vs, idx) = state
  length = new_k.shape[1]
  # TODO(lukaszkaiser): benchmark speed and decide if using a separate code path
  # with index_update when length == 1 is worth it.
  # Keys and values are of shape [batch_size, length, d_kv].
  ks = fastmath.dynamic_update_slice_in_dim(ks, new_k, idx, axis=1)
  vs = fastmath.dynamic_update_slice_in_dim(vs, new_v, idx, axis=1)
  k_length = ks.shape[1]

  # Mask is of shape [1, q_length, k_length].
  # Mask should be true for every pair of (query_token, key_token) such that
  # index of query_token is equal or larger to index of key_token.
  mask = (jnp.reshape(jnp.arange(k_length), (1, 1, k_length))
          <= jnp.reshape(jnp.arange(length) + idx, (1, length, 1)))
  if mask_for_predict is None:
    return (ks, vs, idx + length), mask
  else:
    state_mask_for_predict = fastmath.dynamic_update_slice_in_dim(
        state_mask_for_predict != 0, mask_for_predict.reshape((-1)) != 0, 0,
        axis=0)

    state_mask_for_predict = fastmath.dynamic_update_slice_in_dim(
        state_mask_for_predict != 0, jnp.ones((1,)) != 0,
        jnp.sum(mask_for_predict, dtype=jnp.int32), axis=0)

    state_mask_for_predict = fastmath.dynamic_update_slice_in_dim(
        state_mask_for_predict != 0, jnp.ones((1,)) != 0, idx, axis=0)
    placeholder = jnp.reshape(state_mask_for_predict != 0,
                              (1, 1, mask.shape[2],))
    mask = mask * placeholder

    return (state_mask_for_predict, ks, vs, idx + length), mask
