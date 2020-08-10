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
"""Transformer models: encoder, decoder, language model, and encoder-decoder.

The "Transformer" name and network architecture were introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax

from trax import layers as tl
from trax.fastmath import numpy as jnp


def TransformerEncoder(vocab_size,
                       n_classes=10,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       dropout_shared_axes=None,
                       max_len=2048,
                       mode='train',
                       ff_activation=tl.Relu):
  """Returns a Transformer encoder merged with an N-way categorization head.

  This model performs text categorization:

    - input: rank 2 tensor representing a batch of text strings via token IDs;
      shape is (batch_size, text_length_in_tokens). The tensor elements are
      integers in `range(vocab_size)`. `0` values mark padding positions rather
      than actual token IDs.

    - output: rank 2 tensor representing a batch of log-probability
      distributions over N categories; shape is `(batch_size, n_classes)`.

  Args:
    vocab_size: "Vocabulary size" -- input integer IDs must be in
        `range(vocab_size)`. IDs typically come from preprocessing text data
        with a vocabulary-based tokenizer.
    n_classes: Size/depth of the output vectors, intended for an N-way
        classification task.
    d_model: The basic embedding size (vector depth) of the model. This is the
        vector size used by the initial embedding layer and at many intermediate
        points in the model.
    d_ff: Vector depth (typically greater than `d_model`) used in the
        feed-forward (`Dense`) layer of each encoder block.
    n_layers: Number of encoder blocks. Each encoder block includes attention,
        dropout, residual, feed-forward (`Dense`), and activation layers.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
    max_len: Maximum symbol length for positional encoding.
    mode: If `'train'`, each encoder block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: The activation function (layer) at the end of each encoder
        block.

  Returns:
    A Transformer model that maps strings (each represented as an array of
    token IDs) to probability-like activations over a range of output
    classes.
  """
  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  encoder_blocks = [
      _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                    mode, ff_activation)
      for i in range(n_layers)]

  # Assemble and return the model.
  return tl.Serial(                               # toks
      # Encode.
      tl.Branch(
          positional_encoder, tl.PaddingMask()),  # vecs masks
      encoder_blocks,                             # vecs masks
      tl.Select([0], n_in=2),                     # vecs
      tl.LayerNorm(),                             # vecs

      # Map to output categories.
      tl.Mean(axis=1),                            # vecs
      tl.Dense(n_classes),                        # vecs
      tl.LogSoftmax(),                            # vecs
  )


def TransformerDecoder(vocab_size=None,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       dropout_shared_axes=None,
                       max_len=2048,
                       mode='train',
                       ff_activation=tl.Relu):
  """Returns a Transformer decoder model.

  The input to the model is either continuous or discrete - controlled by
  vocab_size. Does not shift the input to the right, i.e. the output for
  timestep t is based on inputs up to timestep t inclusively.

  Args:
    vocab_size: int or None: vocab size if running on discrete input, None
      otherwise.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer decoder as a layer that maps from a continuous or discrete
    tensor to a continuous tensor.
  """
  positional_encoder = [
      (tl.Embedding(vocab_size, d_model) if vocab_size is not None
       else tl.Dense(d_model)),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  decoder_blocks = [
      # pylint: disable=g-complex-comprehension
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, dropout_shared_axes, mode, ff_activation)
      for i in range(n_layers)]

  # Assemble and return the model.
  return tl.Serial(        # toks
      positional_encoder,  # vecs
      decoder_blocks,      # vecs
      tl.LayerNorm(),      # vecs
  )


def TransformerLM(vocab_size,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  dropout_shared_axes=None,
                  max_len=2048,
                  mode='train',
                  ff_activation=tl.Relu):
  """Returns a Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode)]

  decoder_blocks = [
      # pylint: disable=g-complex-comprehension
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, dropout_shared_axes, mode, ff_activation)
      for i in range(n_layers)]

  # Assemble and return the model.
  return tl.Serial(              # tokens (or chunked tuple of tokens)
      tl.ShiftRight(mode=mode),  # toks
      positional_encoder,        # vecs
      decoder_blocks,            # vecs
      tl.LayerNorm(),            # vecs
      tl.Dense(vocab_size),      # vecs
      tl.LogSoftmax(),           # vecs
  )


def Transformer(input_vocab_size,
                output_vocab_size=None,
                d_model=512,
                d_ff=2048,
                n_encoder_layers=6,
                n_decoder_layers=6,
                n_heads=8,
                dropout=0.1,
                dropout_shared_axes=None,
                max_len=2048,
                mode='train',
                ff_activation=tl.Relu):
  """Returns a Transformer model.

  This model expects an input pair: source, target.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_encoder_layers: int: number of encoder layers
    n_decoder_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer model as a layer that maps from a source, target pair to
    activations over a vocab set.
  """
  def Embedder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
    ]

  in_embedder = Embedder(input_vocab_size)
  out_embedder = (in_embedder if output_vocab_size is None
                  else Embedder(output_vocab_size))

  # Positional encoding are not shared between encoder and decoder.
  # Since encoder doesn't run stepwise, we do not use predict mode there.
  encoder_mode = 'eval' if mode == 'predict' else mode
  in_encoder = in_embedder + [
      tl.PositionalEncoding(max_len=max_len, mode=encoder_mode)
  ]
  out_encoder = out_embedder + [
      tl.PositionalEncoding(max_len=max_len, mode=mode)
  ]

  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  encoder_blocks = [
      _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                    mode, ff_activation)
      for i in range(n_encoder_layers)]

  encoder = tl.Serial(
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  encoder_decoder_blocks = [
      _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                           mode, ff_activation)
      for i in range(n_decoder_layers)]

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 1, 1]),               # tok_e tok_d tok_d

      # Encode.
      tl.Branch([], tl.PaddingMask()),    # tok_e masks ..... .....
      encoder,                            # vec_e ..... ..... .....

      # Decode.
      tl.Select([2, 1, 0]),               # tok_d masks vec_e .....
      tl.ShiftRight(mode=mode),           # tok_d ..... ..... .....
      out_encoder,                        # vec_d ..... ..... .....
      tl.Branch(
          [], tl.EncoderDecoderMask()),   # vec_d masks ..... .....
      encoder_decoder_blocks,             # vec_d masks ..... .....
      tl.LayerNorm(),                     # vec_d ..... ..... .....

      # Map to output vocab.
      tl.Select([0], n_in=3),             # vec_d tok_d
      tl.Dense(output_vocab_size),        # vec_d .....
      tl.LogSoftmax(),                    # vec_d .....
  )


def _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                  mode, ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the layer is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  attention = tl.Attention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  dropout_ = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          attention,
          dropout_,
      ),
      tl.Residual(
          feed_forward
      ),
  ]


def _DecoderBlock(d_model, d_ff, n_heads,
                  dropout, dropout_shared_axes, mode, ff_activation):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode),

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  dropout_ = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          causal_attention,
          dropout_,
      ),
      tl.Residual(
          feed_forward
      ),
  ]


def _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation):
  """Returns a list of layers implementing a Transformer encoder-decoder block.

  The input is a triple (decoder_input, mask, encoder) where the mask is
  created from the original source to prevent attending to the padding part
  of the encoder.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers which maps triples (decoder_activations, mask,
    encoder_activations) to triples of the same sort.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  attention_qkv = tl.AttentionQKV(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  return [                             # vec_d masks vec_e
      tl.Residual(
          tl.LayerNorm(),              # vec_d ..... .....
          causal_attention,            # vec_d ..... .....
          _Dropout(),                  # vec_d ..... .....
      ),
      tl.Residual(
          tl.LayerNorm(),              # vec_d ..... .....
          tl.Select([0, 2, 2, 1, 2]),  # vec_d vec_e vec_e masks vec_e
          attention_qkv,               # vec_d masks vec_e
          _Dropout(),                  # vec_d masks vec_e
      ),
      tl.Residual(
          feed_forward                 # vec_d masks vec_e
      ),
  ]


def _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes,
                      mode, activation):
  """Returns a list of layers implementing a feed-forward block.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: list of integers, axes to share dropout mask
    mode: str: 'train' or 'eval'
    activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers which maps vectors to vectors.
  """
  dropout_middle = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  dropout_final = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      activation(),
      dropout_middle,
      tl.Dense(d_model),
      dropout_final,
  ]


def _ConcatWithPadding():
  """Concatenates two length padded (B, L, H) arrays (of different lenghts)."""

  # Arg shapes: (B, L1, H), (B, L2, H), (B, L1) & (B, L2)
  def __ConcatWithPadding(vec_e, vec_d, mask_e, mask_d):
    # pylint: disable=invalid-name
    B, L1, H = vec_e.shape
    L2 = vec_d.shape[1]
    # pylint: enable=invalid-name

    assert (B, L2, H) == vec_d.shape, f'{(B, L2, H)} != {vec_e.shape}'
    assert (B, L1) == mask_e.shape, f'{(B, L1)} != {mask_e.shape}'
    assert (B, L2) == mask_d.shape, f'{(B, L2)} != {mask_d.shape}'

    def _UpdateRow(x):
      # row_e - (L1, H), row_d - (L2, H), row_mask_e - (L1,)
      row_e, row_d, row_mask_e = x
      # final_row - (L1+L2, H)
      final_row = jnp.concatenate([row_e, jnp.zeros_like(row_d)], axis=0)
      # Find the last real token/vector of the encoder.
      e_idx = jnp.sum(row_mask_e, dtype=jnp.int32)
      # Starting after that index, update with the decoder row.
      return jax.lax.dynamic_update_slice(final_row, row_d, (e_idx, 0))

    return jax.lax.map(_UpdateRow, [vec_e, vec_d, mask_e])

  return tl.Fn('ConcatWithPadding', __ConcatWithPadding, n_out=1)


def _MaskOfRightShiftedArray(n_shifts=1, mode='train'):
  """Gives us the mask of a right shifted by n_shifts array."""

  def F(x):
    # TODO(afrozm): What to do in this case?
    if mode == 'predict':
      raise ValueError('MaskOfRightShiftedArray not implemented for predict.')

    mask = x != 0

    if n_shifts == 0:
      return mask

    # Need to set (B, n_shifts, ...) section to True.
    trues_shape = (x.shape[0], n_shifts) + mask.shape[2:]
    trues = jnp.full(trues_shape, True)
    return jnp.concatenate([trues, mask[:, n_shifts:, ...]], axis=1)
  return tl.Fn(f'MaskOfRightShiftedArray({n_shifts})', F)


def _StripFromConcatenateWithPadding():
  """Strip out the leading encoder tokens from the concatenated array."""

  def _StripEncToks(vec_ed, tok_e, tok_d):
    # pylint: disable=invalid-name
    B, L, H = vec_ed.shape
    L1 = tok_e.shape[1]
    L2 = tok_d.shape[1]
    # pylint: enable=invalid-name
    assert L == L1 + L2
    assert (B, L1) == tok_e.shape
    assert (B, L2) == tok_d.shape

    def _UpdateRow(x):
      # (L, H), (L1, H) & (L2, H)
      row_ed, row_e, _ = x
      mask_e = row_e != 0
      len_e = jnp.sum(mask_e, dtype=jnp.int32)
      # In `row_ed` start where encoder tokens/vecs end, i.e. are index `len_e`
      # and pick up (L2, H) tensor slice from there.
      return jax.lax.dynamic_slice(row_ed, (len_e, 0), (L2, H))

    return jax.lax.map(_UpdateRow, [vec_ed, tok_e, tok_d])

  return tl.Fn('StripFromConcatenateWithPadding', _StripEncToks, n_out=1)
