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
"""Funnel Transformer model.

Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
Language Processing https://arxiv.org/abs/2006.03236
"""
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.fastmath.ops import index_add
from trax.layers.assert_shape import assert_shape
from trax.models.transformer import _EncoderBlock
from trax.models.transformer import _FeedForwardBlock


@assert_shape('bld->bSd')
def PoolLayer(pool_layer=tl.AvgPool,
              pool_size=(2,),
              strides=(2,),
              separate_cls=True):
  """Returns a pool layer for Funnel Transformer.

  Args:
    pool_layer: Type of pooling layer used for downsampling;
        should be `tl.AvgPool` or `tl.MaxPool`.
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    separate_cls: If `True`, pooling in funnel blocks is not applied to
          embeddings of the first token (`cls` from BERT paper).
  """
  if separate_cls:
    cls_selection = tl.Fn('select_cls_token', lambda x: x[:, :1, :])
    tokens_after_cls = tl.Fn('rest_tokens', lambda x: x[:, 1:, :])

    return tl.Serial(
        tl.Branch(
            cls_selection,
            tl.Serial(
                tokens_after_cls,
                pool_layer(pool_size, strides)
            )
        ),
        tl.Concatenate(axis=1)
    )
  else:
    return pool_layer(pool_size, strides)


@assert_shape('b11l->b11S')
def MaskPool(pool_size=(2,), strides=(2,), separate_cls=True):
  return tl.Serial(
      tl.Fn('reshape', lambda x: x.swapaxes(1, -1).squeeze(axis=-1)),
      PoolLayer(tl.MaxPool, pool_size, strides, separate_cls),
      tl.Fn('reshape_back', lambda x: x[..., None].swapaxes(1, -1))
  )


@assert_shape('bld->bd')
def SelectFirst():
  return tl.Fn('select_first', lambda x: x[:, 0, :])


def _Upsampler(total_pool_size, separate_cls):
  """Returns an upsampling layer for Funnel Transformer.

  Args:
    total_pool_size: The combined pool size of previously used funnel blocks.
    separate_cls: If `True`, pooling in funnel blocks is not applied to
          embeddings of the first token (`cls` from BERT paper).
  """
  def _Upsample(short, long):
    if separate_cls:
      upsampled_short = jnp.concatenate(
          (short[:, :1, :],
           short[:, 1:, :].repeat(total_pool_size, axis=1)),
          axis=1)
      return index_add(
          long,
          (slice(None),
           slice(None, upsampled_short.shape[1]),
           slice(None)),
          upsampled_short)
    else:
      upsampled_short = short.repeat(total_pool_size, axis=1)
      return long + upsampled_short

  return tl.Fn('Upsampler', _Upsample)


def _FunnelBlock(d_model, d_ff, n_heads,
                 dropout, dropout_shared_axes, mode, ff_activation,
                 pool_layer, pool_size, strides, separate_cls):
  """Internal funnel block. Returns a list of layers implementing it.

  The input is an activation tensor.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.
    pool_layer: Type of pooling layer used for downsampling;
        should be `tl.AvgPool` or `tl.MaxPool`.
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    separate_cls: If `True`, pooling in funnel blocks is not applied to
          embeddings of the first token (`cls` from BERT paper).
  Returns:
      A list of layers that maps (activations, mask) to (activations', mask).
  """
  pooling = PoolLayer(pool_layer, pool_size, strides, separate_cls)
  mask_pooling = MaskPool(pool_size, strides, separate_cls)

  attention = tl.AttentionQKV(d_model, n_heads=n_heads, dropout=dropout,
                              mode=mode)
  hidden_dropout = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  return [                                          # h, mask
      tl.LayerNorm(),                               # h, mask
      tl.Branch(pooling, None),                     # h', h, mask
      tl.Residual(
          tl.Select([0, 1, 1, 2]),                  # h', h, h, mask
          attention,                                # attn, mask
          tl.Parallel(None, mask_pooling),          # attn, mask'
          hidden_dropout                            # attn, mask'
      ),                                            # funnel_activations, mask'
      tl.Residual(
          feed_forward
      )
  ]


def FunnelTransformerEncoder(vocab_size,
                             n_classes=10,
                             d_model=512,
                             d_ff=2048,
                             encoder_segment_lengths=(2, 2, 2),
                             n_heads=8,
                             max_len=2048,
                             dropout=0.1,
                             dropout_shared_axes=None,
                             mode='train',
                             ff_activation=tl.Relu,
                             pool_layer=tl.AvgPool,
                             pool_size=(2,),
                             strides=(2,),
                             separate_cls=True):
  """Returns a Funnel Encoder.

  This model performs text categorization:

    - input: rank 2 tensor representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). The tensor
      elements are integers in `range(vocab_size)`, and `0` values mark padding
      positions.

    - output: rank 2 tensor representing a batch of log-probability
      distributions over N categories; shape is (batch_size, `n_classes`).

  Args:
    vocab_size: Input vocabulary size -- each element of the input tensor
        should be an integer in `range(vocab_size)`. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    n_classes: Final dimension of the output tensors, representing N-way
        classification.
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
        block.
    encoder_segment_lengths: Tuple, where each element denotes the number of
        transformer encoder blocks preceding a funnel transformer block.
        There is no funnel block after the last sequence of encoder blocks,
        therefore the total number of blocks in the model is equal to
        `sum(encoder_segment_lengths) + len(encoder_segment_lengths) - 1`.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each encoder block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of `Layer`.
    pool_layer: Type of pooling layer used for downsampling in each of the
        funnel blocks; should be `tl.AvgPool` or `tl.MaxPool`.
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    separate_cls: If `True`, pooling in funnel blocks is not applied to
        embeddings of the first token (`cls` from BERT paper) and only final
        embedding of this token is used for categorization - the rest are
        discarded. If `False`, each token from the beginning is pooled and
        all embeddings are averaged and mapped to output categories like in
        original `TransformerEncoder` model.
  Returns:
    A Transformer model that maps strings (conveyed via token IDs) to
    probability-like activations over a range of output classes.
  """
  assert encoder_segment_lengths

  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  encoder_blocks = []
  n_encoder_segments = len(encoder_segment_lengths)

  for i in range(n_encoder_segments):
    # Building i'th segment
    for _ in range(encoder_segment_lengths[i]):
      # Create segment_size encoder blocks
      encoder_blocks.append(
          _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                        mode, ff_activation))

    # If not last segment, add funnel block
    if i != n_encoder_segments - 1:
      encoder_blocks.append(
          _FunnelBlock(d_model, d_ff, n_heads, dropout,
                       dropout_shared_axes, mode,
                       ff_activation, pool_layer, pool_size,
                       strides, separate_cls))

  cls_pooling = SelectFirst() if separate_cls else tl.Mean(axis=1)

  # Assemble and return the model.
  return tl.Serial(                               # toks
      # Encode.
      tl.Branch(
          positional_encoder, tl.PaddingMask()),  # vecs masks
      encoder_blocks,                             # vecs masks
      tl.Select([0], n_in=2),                     # vecs
      tl.LayerNorm(),                             # vecs

      # Map to output categories.
      cls_pooling,                                # cls
      tl.Dense(n_classes),                        # cls
  )


def FunnelTransformer(vocab_size,
                      d_model=512,
                      d_ff=2048,
                      encoder_segment_lengths=(2, 2, 2),
                      n_decoder_blocks=2,
                      n_heads=8,
                      max_len=2048,
                      dropout=0.1,
                      dropout_shared_axes=None,
                      mode='train',
                      ff_activation=tl.Relu,
                      pool_layer=tl.AvgPool,
                      pool_size=(2,),
                      separate_cls=True):
  """Returns a Full Funnel Transformer, that can be used for example for BERT.

  This model outputs token-level categorical distributions over all vocab:

    - input: rank 2 tensor representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). The tensor
      elements are integers in `range(vocab_size)`, and `0` values mark padding
      positions.

    - output: rank 3 tensor representing a batch of log-probability
      distributions over `vocab_size` categories for each token; shape is
      (batch_size, sequence_length, vocab_size).


  Args:
    vocab_size: Input vocabulary size -- each element of the input tensor
        should be an integer in `range(vocab_size)`. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
        block.
    encoder_segment_lengths: Tuple, where each element denotes the number of
        transformer encoder blocks preceding a funnel transformer block.
        There is no funnel block after the last sequence of encoder blocks,
        therefore the total number of blocks in the model is equal to
        `sum(encoder_segment_lengths) + len(encoder_segment_lengths) - 1`.
    n_decoder_blocks: Number of transformer blocks in the upsampling decoder.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each encoder block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of `Layer`.
    pool_layer: Type of pooling layer used for downsampling in each of the
        funnel blocks; should be `tl.AvgPool` or `tl.MaxPool`.
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    separate_cls: If `True`, pooling in funnel blocks is not applied to
        embeddings of the first token (`cls` from BERT paper) and only final
        embedding of this token is used for categorization - the rest are
        discarded. If `False`, each token from the beginning is pooled and
        all embeddings are averaged and mapped to output categories like in
        original `TransformerEncoder` model.
  """
  assert encoder_segment_lengths

  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  n_encoder_segments = len(encoder_segment_lengths)

  encoder_blocks_before_first_pooling = [
      _EncoderBlock(d_model, d_ff, n_heads, dropout,
                    dropout_shared_axes, mode, ff_activation)
      for _ in range(encoder_segment_lengths[0])]
  encoder_blocks_from_first_pooling = []

  for i in range(1, n_encoder_segments):
    # Building i'th segment

    # Add funnel block between segments
    encoder_blocks_from_first_pooling.append(
        _FunnelBlock(d_model, d_ff, n_heads, dropout,
                     dropout_shared_axes, mode,
                     ff_activation, pool_layer,
                     pool_size=pool_size, strides=pool_size,
                     separate_cls=separate_cls))

    for _ in range(encoder_segment_lengths[i]):
      # Create segment_size encoder blocks
      encoder_blocks_from_first_pooling.append(
          _EncoderBlock(d_model, d_ff, n_heads, dropout,
                        dropout_shared_axes, mode, ff_activation))

  decoder_blocks = [_EncoderBlock(d_model, d_ff, n_heads, dropout,
                                  dropout_shared_axes, mode, ff_activation)
                    for _ in range(n_decoder_blocks)]

  total_pool_size = pool_size[0] ** (len(encoder_segment_lengths) - 1)

  # Assemble and return the model.
  return tl.Serial(                               # toks
      tl.Branch(
          positional_encoder, tl.PaddingMask()),  # vecs masks
      encoder_blocks_before_first_pooling,        # vecs masks
      tl.Select([0, 1, 0, 1]),
      # vecs masks residual = vecs old_masks
      encoder_blocks_from_first_pooling,          # vecs masks residual masks
      tl.Select([0, 2, 3]),                       # vecs residual masks
      tl.Parallel(
          # residual from first segment is taken before
          # normalization, so apply it now
          None, tl.LayerNorm(), None),            # vecs norm(residual) masks
      _Upsampler(total_pool_size, separate_cls),  # vecs masks
      decoder_blocks,
      tl.Select([0], n_in=2),                     # vecs
      tl.LayerNorm(),
      tl.Dense(vocab_size),
  )
