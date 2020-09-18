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
"""Reformer Models."""

import jax

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.models.research import transformer_no_enc_dec_attention


StripFromConcatenateWithPadding = (
    transformer_no_enc_dec_attention.StripFromConcatenateWithPadding
)
ConcatWithPadding = transformer_no_enc_dec_attention.ConcatWithPadding


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def FeedForward(d_model, d_ff, dropout, activation, act_dropout, mode):
  """Feed-forward block with layer normalization at start."""
  if act_dropout is None:
    act_dropout = dropout
  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      tl.Dropout(rate=act_dropout, shared_axes=[-2], mode=mode),
      activation(),
      tl.Dense(d_model),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
  ]


def ChunkedFeedForward(d_model, d_ff, dropout, activation, act_dropout,
                       chunk_size, mode):
  """Chunked feed-forward block with layer normalization at start."""
  ff = FeedForward(d_model, d_ff, dropout, activation, act_dropout, mode)
  if chunk_size < 1:
    return ff
  def reshape_to_chunks(x):
    batch_times_length = x.shape[0] * x.shape[1]
    assert batch_times_length % chunk_size == 0
    n_chunks = batch_times_length // chunk_size
    return jnp.reshape(x, [n_chunks, 1, chunk_size] + list(x.shape[2:]))
  return [
      tl.Dup(),  # Just to have shape for later after scan.
      tl.Fn('ReshapeToChunks', reshape_to_chunks, n_out=1),
      tl.Scan(tl.Serial(ff), axis=0, n_carry=0, remat=True),
      tl.Fn('ReshapeXToY', lambda x, y: jnp.reshape(x, y.shape))
  ]


def FeedForwardWithOptions(d_model, d_ff, dropout, ff_activation, ff_dropout,
                           ff_chunk_size, ff_use_sru, ff_sparsity, mode):
  """Feed-Forward block with all the options."""
  if ff_use_sru:
    return [tl.SRU(d_model) for _ in range(ff_use_sru)]
  elif ff_sparsity:
    return [tl.LayerNorm(),
            tl.SparseFF(d_ff, n_elements_in_block=ff_sparsity,
                        d_lowrank=d_ff // ff_sparsity, mode=mode),
            tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode)]
  else:
    return [ChunkedFeedForward(d_model, d_ff, dropout, ff_activation,
                               ff_dropout, ff_chunk_size, mode)]


def DecoderBlock(d_model, d_ff, d_attention_key, d_attention_value,
                 n_heads, attention_type, dropout, ff_activation,
                 ff_dropout, ff_use_sru, ff_chunk_size, ff_sparsity, mode):
  """Reversible transformer decoder layer.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_heads: int: number of attention heads
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: the dropout rate in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  attention = attention_type(
      n_heads=n_heads, d_qk=d_attention_key, d_v=d_attention_value,
      causal=True, output_dropout=dropout, mode=mode)
  attention_half_residual = tl.ReversibleHalfResidual(
      tl.LayerNorm(),
      attention_layer=attention,
  )

  feed_forward = FeedForwardWithOptions(
      d_model, d_ff, dropout, ff_activation, ff_dropout,
      ff_chunk_size, ff_use_sru, ff_sparsity, mode)

  return [
      attention_half_residual,
      tl.ReversibleSwap(),
      tl.ReversibleHalfResidual(feed_forward),
      tl.ReversibleSwap(),
  ]


def PositionalEncoding(mode, dropout=None, max_len=None,
                       axial_pos_shape=None, d_axial_pos_embs=None):
  """Returns the positional encoding layer depending on the arguments."""
  if not axial_pos_shape:
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, mode=mode)
  elif axial_pos_shape == 'fixed-base':  # TODO(lukaszkaiser): remove this HACK
    positional_encoding = tl.FixedBasePositionalEncoding(mode=mode)
  elif axial_pos_shape == 'infinite':  # TODO(lukaszkaiser): remove this HACK
    positional_encoding = tl.InfinitePositionalEncoding(affine=False)
  elif axial_pos_shape == 'infinite-affine':
    # TODO(lukaszkaiser): remove this HACK
    positional_encoding = tl.InfinitePositionalEncoding()
  elif axial_pos_shape == 'time-bin':  # TODO(lukaszkaiser): remove this HACK
    positional_encoding = tl.TimeBinPositionalEncoding()
  else:
    assert d_axial_pos_embs is not None
    positional_encoding = tl.AxialPositionalEncoding(
        shape=axial_pos_shape, d_embs=d_axial_pos_embs,
        dropout_broadcast_dims=tuple(range(1, len(axial_pos_shape) + 1)),
        dropout=dropout, mode=mode)

  return positional_encoding


def ReformerLM(vocab_size,
               d_model=512,
               d_ff=2048,
               d_attention_key=64,
               d_attention_value=64,
               n_layers=6,
               n_heads=8,
               dropout=0.1,
               max_len=2048,
               attention_type=tl.SelfAttention,
               axial_pos_shape=(),
               d_axial_pos_embs=None,
               ff_activation=tl.FastGelu,
               ff_use_sru=0,
               ff_chunk_size=0,
               ff_sparsity=0,
               mode='train'):
  """Reversible transformer language model (only uses a decoder, no encoder).

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of *each half* of the two-part features
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    attention_type: class: attention class to use, such as SelfAttention.
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, and values must sum to d_model.
    ff_activation: the non-linearity in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    mode: str: 'train', 'eval', or 'predict'

  Returns:
    the layer.
  """
  positional_encoding = PositionalEncoding(
      mode, dropout, max_len, axial_pos_shape, d_axial_pos_embs)

  positional_embedder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
      positional_encoding,
  ]

  decoder_blocks = []

  if isinstance(attention_type, (tuple, list)):
    assert n_layers % len(attention_type) == 0
  else:
    attention_type = [attention_type]
  for layer_idx in range(n_layers):
    layer_attention_type = attention_type[layer_idx % len(attention_type)]
    decoder_block = DecoderBlock(
        d_model, d_ff, d_attention_key, d_attention_value, n_heads,
        attention_type=layer_attention_type,
        dropout=dropout,
        ff_activation=ff_activation,
        ff_dropout=dropout,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        ff_sparsity=ff_sparsity,
        mode=mode)
    decoder_blocks.append(decoder_block)

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      positional_embedder,
      tl.Dup(),
      tl.ReversibleSerial(decoder_blocks),
      tl.Concatenate(),
      # TODO(kitaev): Test whether dropout should go before or after the
      # LayerNorm, and whether dropout broadcasting is needed here.
      tl.LayerNorm(),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )


def ReformerShortenLM(vocab_size,
                      shorten_factor=1,
                      d_embedding=256,
                      d_model=512,
                      d_ff=2048,
                      d_attention_key=64,
                      d_attention_value=64,
                      n_layers=6,
                      n_heads=8,
                      dropout=0.1,
                      max_len=2048,
                      attention_type=tl.SelfAttention,
                      axial_pos_shape=(),
                      d_axial_pos_embs=None,
                      ff_activation=tl.FastGelu,
                      ff_use_sru=0,
                      ff_chunk_size=0,
                      ff_sparsity=0,
                      mode='train'):
  """Reversible transformer language model with shortening.

  When shorten_factor is F and processing an input of shape [batch, length],
  we embed the (shifted-right) input and then group each F elements (on length)
  into a single vector -- so that in the end we process a tensor of shape ::

      [batch, length // F, d_model]

  almost until the end -- at the end it's un-shortend and a SRU is applied.
  This reduces the length processed inside the main model body, effectively
  making the model faster but possibly slightly less accurate.

  Args:
    vocab_size: int: vocab size
    shorten_factor: by how much to shorten, see above
    d_embedding: the depth of the embedding layer and final logits
    d_model: int:  depth of *each half* of the two-part features
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    attention_type: class: attention class to use, such as SelfAttention.
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, values must sum to d_embedding.
    ff_activation: the non-linearity in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  assert mode != 'predict'  # TODO(lukaszkaiser,kitaev): fast inference

  if not axial_pos_shape:
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, mode=mode)
  else:
    assert d_axial_pos_embs is not None
    positional_encoding = tl.AxialPositionalEncoding(
        shape=axial_pos_shape, d_embs=d_axial_pos_embs,
        dropout_broadcast_dims=tuple(range(1, len(axial_pos_shape) + 1)),
        dropout=dropout, mode=mode)

  positional_embedder = [
      tl.Embedding(vocab_size, d_embedding),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
      positional_encoding,
  ]

  decoder_blocks = []

  if isinstance(attention_type, (tuple, list)):
    assert n_layers % len(attention_type) == 0
  else:
    attention_type = [attention_type]
  for layer_idx in range(n_layers):
    layer_attention_type = attention_type[layer_idx % len(attention_type)]
    decoder_block = DecoderBlock(
        d_model, d_ff, d_attention_key, d_attention_value, n_heads,
        attention_type=layer_attention_type,
        dropout=dropout,
        ff_activation=ff_activation,
        ff_dropout=dropout,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        ff_sparsity=ff_sparsity,
        mode=mode)
    decoder_blocks.append(decoder_block)

  # pylint: disable=g-long-lambda
  return tl.Serial(
      tl.ShiftRight(),
      positional_embedder,
      tl.Dup(),              # Stack has (x, x), the first will be shortened
      # Before shortening, we need to pad by shorten factor so as not to leak
      # information into the future. To understand why, imagine shorten factor
      # of 2 and sequence of length 4, so ABCD. If we shift just by 1, then we
      # would have 0ABC, which gets grouped to [0A][BC] on input, which is
      # predicting ABCD as targets. The problem is that [0A] has access to A
      # and [BC] has access to C -- it will learn to copy it, peek into
      # the future. Shifting twice to [00][AB] solves the problem as the first
      # "big" symbol becomes all-0 and the rest is shifted enough.
      tl.ShiftRight(n_positions=shorten_factor - 1),
      tl.Fn('Shorten', lambda x: jnp.reshape(  # Shorten -- move to depth.
          x, (x.shape[0], x.shape[1] // shorten_factor, -1)), n_out=1),
      tl.Dense(d_model),
      tl.Dup(),  # Stack has (short_x, short_x, x)
      tl.ReversibleSerial(decoder_blocks),
      tl.Select([0], n_in=2),
      tl.LayerNorm(),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
      tl.Dense(shorten_factor * d_embedding),
      tl.Fn('ProlongBack', lambda x: jnp.reshape(  # Prolong back.
          x, (x.shape[0], x.shape[1] * shorten_factor, -1)), n_out=1),
      tl.Concatenate(),  # Concatenate with just the embeddings.
      tl.CausalConv(d_embedding),
      tl.Relu(),
      tl.SRU(d_embedding),  # One RNN layer for conditional dependence.
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )
  # pylint: enable=g-long-lambda


def EncoderBlock(d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                 ff_dropout, ff_use_sru=0, ff_chunk_size=0, ff_sparsity=0,
                 mode='train'):
  """Returns a list of layers that implements a Reformer encoder block.

  The input to the layer is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: the dropout rate in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    mode: str: 'train' or 'eval'

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  if mode == 'predict':
    # Mode 'predict' means that the decoder should be run one token at a time.
    # The encoder only ever runs over full sequences, which is why it's switched
    # to 'eval' mode instead.
    mode = 'eval'

  attention = attention_type(
      n_heads=n_heads, d_qk=d_model//n_heads, d_v=d_model//n_heads,
      masked=True, causal=False,
      attention_dropout=dropout, output_dropout=dropout,
      mode=mode)
  attention_half_residual = tl.ReversibleHalfResidual(
      tl.LayerNorm(),
      attention_layer=attention,
  )

  feed_forward = FeedForwardWithOptions(
      d_model, d_ff, dropout, ff_activation, ff_dropout,
      ff_chunk_size, ff_use_sru, ff_sparsity, mode)

  return [
      attention_half_residual,
      tl.ReversibleSwap(),
      tl.ReversibleHalfResidual(feed_forward),
      tl.ReversibleSwap(),
  ]


def EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, ff_activation,
                        ff_dropout, mode):
  """Reversible transformer decoder layer.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: float: (optional) separate dropout rate for feed-forward layer
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  enc_dec_attention = tl.EncDecAttention(
      n_heads=n_heads, d_qk=d_model//n_heads, d_v=d_model//n_heads,
      attention_dropout=dropout, output_dropout=dropout,
      mode=mode)
  enc_dec_attention_half_residual = tl.ReversibleHalfResidual(
      tl.LayerNorm(),
      attention_layer=enc_dec_attention,
  )

  causal_attention = tl.SelfAttention(
      n_heads=n_heads, d_qk=d_model//n_heads, d_v=d_model//n_heads,
      causal=True,
      attention_dropout=dropout, output_dropout=dropout,
      mode=mode)
  causal_attention_half_residual = tl.ReversibleHalfResidual(
      tl.LayerNorm(),
      attention_layer=causal_attention,
  )

  feed_forward = FeedForward(
      d_model, d_ff, dropout, ff_activation, ff_dropout, mode)

  return [                             # vec_d1 vec_d2 vec_e masks
      causal_attention_half_residual,
      tl.ReversibleSwap(),
      enc_dec_attention_half_residual,
      tl.ReversibleSwap(),
      tl.ReversibleHalfResidual(feed_forward),
      tl.ReversibleSwap(),
  ]


def Reformer(input_vocab_size,
             output_vocab_size=None,
             d_model=512,
             d_ff=2048,
             n_encoder_layers=6,
             n_decoder_layers=6,
             n_heads=8,
             dropout=0.1,
             max_len=2048,
             ff_activation=tl.Relu,
             ff_dropout=None,
             mode='train'):
  """Reversible transformer encoder-decoder model.

  This model expects an input pair: target, source.

  At the moment, this model supports dot-product attention only. For the
  attention types in the Reformer paper, see ReformerLM.

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
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: float: (optional) separate dropout rate at feed-forward
      nonlinearity. This is called relu_dropout in T2T.
    mode: str: 'train' or 'eval'

  Returns:
    A Reformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  # The current API for custom gradients assumes that a layer must be
  # differentiable wrt all of its inputs, but the Transformer puts bool-dtype
  # masks on the stack. This causes jax to error, even though the so-called
  # "gradient" wrt the masks is never actually computed.
  # TODO(kitaev): remove this hack.
  if fastmath.is_backend(fastmath.Backend.JAX):
    jax.api._check_inexact_input_vjp = lambda x: None  # pylint: disable=protected-access

  def PositionalEncoder(vocab_size, mode):  # tokens --> vectors
    # TODO(kitaev): axial positional encoding is better for very long sequences.
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, mode=mode)
    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
        positional_encoding,
    ]

  # Mode 'predict' means that the decoder should be run one token at a time.
  # The encoder only ever runs over full sequences, which is why it's switched
  # to 'eval' mode instead.
  in_encoder = PositionalEncoder(
      input_vocab_size, mode='eval' if mode == 'predict' else mode)
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size
  out_encoder = PositionalEncoder(output_vocab_size, mode)

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      EncoderBlock(
          d_model, d_ff, n_heads, tl.SelfAttention, dropout, ff_activation,
          ff_dropout, mode=mode)
      for _ in range(n_encoder_layers)]
  # pylint: enable=g-complex-comprehension

  encoder = tl.Serial([
      in_encoder,
      tl.Dup(),
      tl.ReversibleSerial(encoder_blocks),
      tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
      tl.LayerNorm(),
  ])
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  encoder_decoder_blocks = [
      EncoderDecoderBlock(
          d_model, d_ff, n_heads, dropout, ff_activation, ff_dropout, mode)
      for _ in range(n_decoder_layers)]

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 1, 1]),                 # tok_e tok_d tok_d
      tl.Branch([], [tl.PaddingMask(),
                     tl.Fn('Squeeze',
                           lambda x: jnp.squeeze(x, (1, 2)), n_out=1)]),
      #                                     # tok_e mask  tok_d .....

      # Encode.
      encoder,                              # vec_e  mask tok_d .....

      # Decode.
      tl.Select([2, 0, 1]),                 # tok_d vec_e mask .....
      tl.ShiftRight(mode=mode),             # tok_d vec_e mask .....
      out_encoder,                          # vec_d vec_e mask .....
      tl.Dup(),                             # vec_d1 vec_d2 vec_e mask .....
      tl.ReversibleSerial(encoder_decoder_blocks),
      tl.Fn('XYAvg',
            lambda x, y: (x + y) / 2.0),    # vec_d vec_e mask .....
      tl.LayerNorm(),                       # vec_d vec_e mask .....

      # Map to output vocab.
      tl.Select([0], n_in=3),               # vec_d .....
      tl.Dense(output_vocab_size),          # vec_d .....
      tl.LogSoftmax(),                      # vec_d .....
  )


def Reformer2(input_vocab_size,
              output_vocab_size=None,
              d_model=512,
              d_ff=2048,
              d_attention_key=None,
              d_attention_value=None,
              n_encoder_layers=6,
              n_decoder_layers=6,
              n_heads=8,
              dropout=0.1,
              max_len=2048,
              encoder_attention_type=tl.SelfAttention,
              encoder_decoder_attention_type=tl.SelfAttention,
              axial_pos_shape='fixed-base',
              d_axial_pos_embs=None,
              ff_activation=tl.Relu,
              ff_use_sru=0,
              ff_chunk_size=0,
              ff_dropout=None,
              ff_sparsity=0,
              n_layers_forget=0,
              mode='train'):
  """Reversible transformer encoder-decoder model.

  This model expects an input pair: source, target.

  At the moment, this model supports dot-product attention only. For the
  attention types in the Reformer paper, see ReformerLM.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_encoder_layers: int: number of encoder layers
    n_decoder_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    encoder_attention_type: class: attention class to use, such as SelfAttention
    encoder_decoder_attention_type: class: attention class to use, such as
      SelfAttention
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, and values must sum to d_model.
    ff_activation: the non-linearity in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_dropout: float: (optional) separate dropout rate at feed-forward
      nonlinearity. This is called relu_dropout in T2T.
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    n_layers_forget: how often to have a forgetting block between layers
    mode: str: 'train' or 'eval'

  Returns:
    A Reformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  # The current API for custom gradients assumes that a layer must be
  # differentiable wrt all of its inputs, but the Transformer puts bool-dtype
  # masks on the stack. This causes jax to error, even though the so-called
  # "gradient" wrt the masks is never actually computed.
  # TODO(kitaev): remove this hack.
  if fastmath.is_backend(fastmath.Backend.JAX):
    jax.api._check_inexact_input_vjp = lambda x: None  # pylint: disable=protected-access

  # Set default dimensions for attention head key and value sizes.
  if d_attention_key is None:
    if d_model % n_heads != 0:
      raise ValueError(f'n_heads ({n_heads}) must divide d_model ({d_model})')
    d_attention_key = d_model // n_heads
  if d_attention_value is None:
    if d_model % n_heads != 0:
      raise ValueError(f'n_heads ({n_heads}) must divide d_model ({d_model})')
    d_attention_value = d_model // n_heads

  # Vector embeddings.
  def Embedder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
    ]

  in_embedder = Embedder(input_vocab_size)
  out_embedder = (in_embedder if output_vocab_size is None
                  else Embedder(output_vocab_size))

  def PositionalEnc(mode):
    return PositionalEncoding(
        mode, dropout, max_len, axial_pos_shape, d_axial_pos_embs)

  # Mode 'predict' means that the decoder should be run one token at a time.
  # The encoder only ever runs over full sequences, which is why it's switched
  # to 'eval' mode instead.
  encoder_mode = 'eval' if mode == 'predict' else mode
  in_encoder = in_embedder + [PositionalEnc(encoder_mode)]
  out_encoder = out_embedder + [PositionalEnc(mode)]
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      EncoderBlock(
          d_model, d_ff, n_heads, encoder_attention_type,
          dropout=dropout,
          ff_activation=ff_activation,
          ff_dropout=ff_dropout,
          ff_use_sru=ff_use_sru,
          ff_chunk_size=ff_chunk_size,
          ff_sparsity=ff_sparsity,
          mode=mode)
      for _ in range(n_encoder_layers)]
  # pylint: enable=g-complex-comprehension

  encoder = tl.Serial([                # vec_e mask_e tok_e tok_d tok_d
      tl.Dup(),                        # vec_e1 vec_e2 mask_e tok_e tok_d tok_d
      _ReversibleSerialForget(encoder_blocks, d_model, n_layers_forget),
      tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
      tl.Dense(d_model),
      tl.LayerNorm(),
  ])
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  decoder_blocks = []

  if isinstance(encoder_decoder_attention_type, (tuple, list)):
    assert n_decoder_layers % len(encoder_decoder_attention_type) == 0
  else:
    encoder_decoder_attention_type = [encoder_decoder_attention_type]
  for layer_idx in range(n_decoder_layers):
    layer_attention_type = encoder_decoder_attention_type[
        layer_idx % len(encoder_decoder_attention_type)]
    decoder_block = DecoderBlock(
        d_model, d_ff, d_attention_key, d_attention_value, n_heads,
        attention_type=layer_attention_type,
        dropout=dropout,
        ff_activation=ff_activation,
        ff_dropout=ff_dropout,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        ff_sparsity=ff_sparsity,
        mode=mode)
    decoder_blocks.append(decoder_block)

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 0, 0, 1, 1]),                # tok_e tok_e tok_e tok_d tok_d

      # Embed in and out tokens; done together as weights may be shared.
      tl.Parallel(in_encoder, [], [],            # vec_e tok_e tok_e vec_d tok_d
                  [tl.ShiftRight(mode=mode), out_encoder]),

      tl.Parallel([], [tl.PaddingMask(),
                       tl.Fn('Squeeze',
                             lambda x: jnp.squeeze(x, (1, 2)), n_out=1)]),
      #                                         # vec_e mask_e tok_e vec_d tok_d

      # Encode.
      encoder,                                  # vec_e mask_e tok_e vec_d tok_d

      # Decode.
      tl.Select([3, 0, 1, 2]),                 #  vec_d vec_e mask_e tok_e tok_d

      # Concat encoder and decoder, given encoder mask.
      tl.Select([1, 0]),                       # vec_e vec_d mask_e tok_e tok_d
      ConcatWithPadding(mode=mode),            # vec_ed tok_e tok_d

      # Run (encoder and) decoder blocks.
      tl.Dup(),                                    # vec_ed1 vec_ed2 tok_e tok_d
      _ReversibleSerialForget(decoder_blocks, d_model,
                              n_layers_forget),    # vec_ed1 vec_ed2 tok_e tok_d
      tl.Fn('XYAvg',
            lambda x, y: (x + y) / 2.0),           # vec_ed tok_e tok_d
      tl.LayerNorm(),                              # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector.
      tl.Select([0, 1, 2, 2]),                     # vec_ed tok_e tok_d tok_d
      StripFromConcatenateWithPadding(mode=mode),  # vec_d tok_d

      # Map to output vocab.
      tl.Dense(output_vocab_size),                 # vec_d tok_d
      tl.LogSoftmax(),                             # vec_d tok_d
  )


def _ReversibleSerialForget(layers, d_model, n_layers):
  """ReversibleSerial but with a forgetting block every n_layers."""
  if not n_layers or len(layers) <= n_layers + 1:
    return tl.ReversibleSerial(layers)
  layers1, layers2 = layers[:n_layers], layers[n_layers:]
  return tl.Serial(
      tl.ReversibleSerial(layers1),
      tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
      tl.Dense(d_model),
      tl.Dup(),
      _ReversibleSerialForget(layers2, d_model, n_layers)
  )
