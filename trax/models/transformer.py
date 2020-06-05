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
"""Transformer Models."""

import jax

from trax import layers as tl
from trax import math
from trax.math import numpy as jnp


def TransformerEncoder(vocab_size,
                       n_classes=10,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       max_len=2048,
                       mode='train',
                       ff_activation=tl.Relu):
  """Returns a Transformer encoder model.

  The input to the model is a tensor of tokens.

  Args:
    vocab_size: int: vocab size
    n_classes: how many classes on output
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
  """
  positional_encoder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='emb_dropout', mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  encoder_blocks = [
      _EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
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
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer decoder as a layer that maps from a continuous or discrete
    tensor to a continuous tensor.
  """
  positional_encoder = [
      (tl.Embedding(d_model, vocab_size) if vocab_size is not None
       else tl.Dense(d_model)),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len)]

  decoder_blocks = [
      # pylint: disable=g-complex-comprehension
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, i, mode, ff_activation)
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
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  positional_encoder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='embedding', mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode)]

  decoder_blocks = [
      # pylint: disable=g-complex-comprehension
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, i, mode, ff_activation)
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
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer model as a layer that maps from a source, target pair to
    activations over a vocab set.
  """
  def PositionalEncoder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(d_model, vocab_size),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len),
    ]

  in_encoder = PositionalEncoder(input_vocab_size)
  out_encoder = (in_encoder if output_vocab_size is None
                 else PositionalEncoder(output_vocab_size))
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  encoder_blocks = [
      _EncoderBlock(
          d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
      for i in range(n_encoder_layers)]

  encoder = tl.Serial(
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  encoder_decoder_blocks = [
      _EncoderDecoderBlock(
          d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
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
      tl.ShiftRight(),                    # tok_d ..... ..... .....
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


def TransformerNoEncDecAttention(input_vocab_size,
                                 output_vocab_size=None,
                                 d_model=512,
                                 d_ff=2048,
                                 n_encoder_layers=6,
                                 n_decoder_layers=6,
                                 n_heads=8,
                                 dropout=0.1,
                                 max_len=2048,
                                 mode='train',
                                 ff_activation=tl.Relu):
  """Returns a Transformer model.

  This model expects an input pair: target, source.

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
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  def PositionalEncoder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(d_model, vocab_size),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len),
    ]

  in_encoder = PositionalEncoder(input_vocab_size)
  out_encoder = (in_encoder if output_vocab_size is None
                 else PositionalEncoder(output_vocab_size))
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  encoder_blocks = [
      _EncoderBlock(
          d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
      for i in range(n_encoder_layers)]

  encoder = tl.Serial(
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  decoder_blocks = [
      # pylint: disable=g-complex-comprehension
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, i, mode, ff_activation)
      for i in range(n_decoder_layers)]

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 0, 1, 1]),            # tok_e tok_e tok_d tok_d

      # Encode.
      tl.Branch([], tl.PaddingMask()),    # tok_e mask_e tok_e tok_d tok_d
      encoder,                            # vec_e mask_e tok_e tok_d tok_d

      # Simple encoder mask, doesn't contain extra dims.
      tl.Select([2, 0, 2], n_in=3),       # tok_e vec_e tok_e tok_d tok_d
      _MaskOfRightShiftedArray(
          n_shifts=0),                    # mask_e vec_e tok_e tok_d tok_d

      # Decode.
      tl.Select([3, 1, 0, 2]),          #  tok_d vec_e mask_e tok_e tok_d
      tl.ShiftRight(),                  # stok_d vec_e mask_e tok_e tok_d
      tl.Branch(
          [],
          _MaskOfRightShiftedArray()
      ),                                # stok_d mask_d vec_e mask_e tok_e tok_d
      out_encoder,                      # svec_d mask_d vec_e mask_e tok_e tok_d

      # Concat encoder and decoder.
      tl.Select([2, 0, 3, 1]),          # vec_e svec_d mask_e mask_d tok_e tok_d
      _ConcatWithPadding(),             # vec_ed tok_e tok_d

      # Decoder blocks with causal attention
      decoder_blocks,                     # vec_ed tok_e tok_d
      tl.LayerNorm(),                     # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector.
      tl.Select([0, 1, 2, 2]),               # vec_ed tok_e tok_d tok_d
      _StripFromConcatenateWithPadding(),    # vec_d tok_d

      # Map to output vocab.
      tl.Dense(output_vocab_size),        # vec_d tok_d
      tl.LogSoftmax(),                    # vec_d tok_d
  )


def _EncoderBlock(d_model, d_ff, n_heads, dropout, layer_idx, mode,
                  ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the layer is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  attention = tl.Attention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  dropout_ = tl.Dropout(
      rate=dropout, name='dropout_enc_attn', mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, layer_idx, mode, ff_activation)

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
                  dropout, layer_idx, mode, ff_activation):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode),

  dropout_ = tl.Dropout(
      rate=dropout, name='attention_%d' % layer_idx, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, layer_idx, mode, ff_activation)

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


def _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, layer_idx, mode,
                         ff_activation):
  """Returns a list of layers implementing a Transformer encoder-decoder block.

  The input is a triple (decoder_input, mask, encoder) where the mask is
  created from the original source to prevent attending to the padding part
  of the encoder.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers which maps triples (decoder_activations, mask,
    encoder_activations) to triples of the same sort.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, mode=mode)

  attention_qkv = tl.AttentionQKV(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, layer_idx, mode, ff_activation)

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


def _FeedForwardBlock(d_model, d_ff, dropout, layer_idx, mode, activation):
  """Returns a list of layers implementing a feed-forward block.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers which maps vectors to vectors.
  """
  dropout_middle = tl.Dropout(
      rate=dropout, name='ff_middle_%d' % layer_idx, mode=mode)
  dropout_final = tl.Dropout(
      rate=dropout, name='ff_final_%d' % layer_idx, mode=mode)

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

  # Implementational Note: An earlier version of `F` made `mask_concat` the
  # same shape as `enc_pad_dec_pad` and then sorted `enc_pad_dec_pad` with key
  # as `mask_concat` all with no stoppage of gradients. This version just sorts
  # `mask_concat` and stops gradients and then uses advanced indexing to select
  # out the stuff needed from `enc_pad_dec_pad`. Even with advanced indexing we
  # have to be careful to not select on arange(H) in the end, this blows up.

  # Arg shapes: (B, L1, H), (B, L2, H), (B, L1) & (B, L2)
  def F(vec_e, vec_d, mask_e, mask_d):
    # pylint: disable=invalid-name
    B, L1, H = vec_e.shape
    L2 = vec_d.shape[1]
    # pylint: enable=invalid-name

    assert (B, L2, H) == vec_d.shape, f'{(B, L2, H)} != {vec_e.shape}'
    assert (B, L1) == mask_e.shape, f'{(B, L1)} != {mask_e.shape}'
    assert (B, L2) == mask_d.shape, f'{(B, L2)} != {mask_d.shape}'

    # [-(L1+L2), -L2) but with padding 0-ed out - (B, L1).
    mask_e_key = jnp.arange(-(L1 + L2), -L2) * mask_e
    # [-L2,0) but with padding 0-ed out - (B, L2).
    mask_d_key = jnp.arange(-L2, 0) * mask_d

    # Shapes of `mask_concat` and `idxs` (B, L = L1+L2)
    mask_concat = jnp.concatenate([mask_e_key, mask_d_key], axis=1)
    _, idxs = math.sort_key_val(
        mask_concat,                     # (B, L)
        # jnp.arange(L1 + L2)[None, ...] -- (1, L) why does this not work?
        jnp.broadcast_to(jnp.arange(L1 + L2), mask_concat.shape),
        1)                               # sort on L
    idxs = math.stop_gradient(idxs)

    # Shape (B, L, H)
    enc_pad_dec_pad = jnp.concatenate([vec_e, vec_d], axis=1)

    # Taking along indices supplied by `idxs` moves padding to the end.
    enc_dec_padpad = enc_pad_dec_pad[jnp.arange(B)[:, None], idxs]
    return enc_dec_padpad

  return tl.Fn('ConcatWithPadding', F, n_out=1)


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

  # Shapes: (L1+L2, H), (L1,) and (L2,)
  def F(vec_ed, tok_e, tok_d):
    mask_e = tok_e != 0
    # Actual length of encoder tokens <= L1
    len_e = jnp.sum(mask_e)
    # Padded length of decoder tokens, this is L2.
    L2 = tok_d.shape[0]  # pylint: disable=invalid-name

    # vec_ed is of type [eeedd00000], we will roll it len_e=3 in reverse.
    # This gives us [dd00000eee] and now we take only the first L2 elements.
    return jnp.roll(vec_ed, -len_e, axis=0)[:L2]

  # TODO(afrozm): Try to do this with sort_key_val instead of roll to get rid of
  # the vmap.
  def _F(vec_ed, tok_e, tok_d):
    return jax.vmap(F)(vec_ed, tok_e, tok_d)

  # We could have written `tl.Fn(..., jax.vmap(F), ...)` here but Trax needs the
  # top-level function (here: jax.vmap) to not have variable or named arguments,
  # so we need a thin wrapper.
  return tl.Fn('StripFromConcatenateWithPadding', _F, n_out=1)


