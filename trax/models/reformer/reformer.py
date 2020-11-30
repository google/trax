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

from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.models.research import configurable_transformer as ct
from trax.models.research import transformer2 as t2


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def DecoderBlock(d_model, d_ff, d_attention_key, d_attention_value,
                 n_heads, attention_type, dropout, ff_activation,
                 ff_dropout, ff_use_sru, ff_chunk_size, ff_sparsity,
                 attention_chunk_size, n_attention_layers=1,
                 n_feedforward_layers=1, use_bfloat16=False, mode='train'):
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
    attention_chunk_size: int, if > 0 run attention chunked at this size
    n_attention_layers: how many residual causal attention layers should we
      have before the feed-forward block (default: 1, the standard block)
    n_feedforward_layers: how many FFNN layers should we have (default 1).
    use_bfloat16: whether to use bfloat16 for weights (default: False).
    mode: str: 'train' or 'eval'


  Returns:
    the layer.
  """
  # pylint: disable=g-complex-comprehension
  attention_half_residuals = [
      [tl.ReversibleHalfResidual(
          tl.LayerNorm(),
          attention_layer=ct.ApplyAttentionLayer(
              attention_type, d_model, n_heads, d_attention_key,
              d_attention_value, True, False, dropout, dropout,
              attention_chunk_size, mode)),
       tl.ReversibleSwap()
      ] for _ in range(n_attention_layers)]

  feed_forwards = [
      [tl.ReversibleHalfResidual(ct.FeedForwardWithOptions(
          d_model, d_ff, dropout, [-2], ff_activation, ff_dropout,
          ff_chunk_size, ff_use_sru, ff_sparsity, mode, use_bfloat16)),
       tl.ReversibleSwap()
      ] for _ in range(n_feedforward_layers)]
  # pylint: enable=g-complex-comprehension
  return attention_half_residuals + feed_forwards


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
               loss_sparsity_type='mult',
               loss_sparsity=0,
               loss_d_lowrank=0,
               loss_sparsity_prob=None,
               attention_chunk_size=0,
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
    loss_sparsity_type: str, type of sparsity to used in loss layer. See
      SparseDenseWithOptions for options. None if no sparsity should be used.
    loss_sparsity: int, the sparsity for loss layer (if used)
    loss_d_lowrank: int, the dimensions for intermediate layer (if used)
    loss_sparsity_prob: float, the probability for sparse version of loss to be
      used. If None, only sparse version is used.
    attention_chunk_size: int, if > 0 run attention chunked at this size
    mode: str: 'train', 'eval', or 'predict'

  Returns:
    the layer.
  """
  positional_encoding = ct.PositionalEncoder(
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
        attention_chunk_size=attention_chunk_size,
        mode=mode)
    decoder_blocks.append(decoder_block)

  dense_loss_layer = tl.SparseDenseWithOptions(
      vocab_size,
      d_input=d_model,
      sparsity_type=loss_sparsity_type,
      sparsity=loss_sparsity,
      d_lowrank=loss_d_lowrank,
      prob_sparse=loss_sparsity_prob,
      mode=mode)

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
      dense_loss_layer,
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
                      attention_chunk_size=0,
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
    attention_chunk_size: int, if > 0 run attention chunked at this size
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  assert mode != 'predict'  # TODO(lukaszkaiser,kitaev): fast inference

  positional_encoding = ct.PositionalEncoder(
      mode, dropout, max_len, axial_pos_shape, d_axial_pos_embs)

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
        attention_chunk_size=attention_chunk_size,
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
  )
  # pylint: enable=g-long-lambda


def EncoderBlock(d_model, d_ff, n_heads, attention_type, dropout, ff_activation,
                 ff_dropout, ff_use_sru=0, ff_chunk_size=0, ff_sparsity=0,
                 attention_chunk_size=0, use_bfloat16=False, mode='train'):
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
    attention_chunk_size: int, if > 0 run attention chunked at this size
    use_bfloat16: whether to use bfloat16 for weights (default: False)
    mode: str: 'train' or 'eval'

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  if mode == 'predict':
    # Mode 'predict' means that the decoder should be run one token at a time.
    # The encoder only ever runs over full sequences, which is why it's switched
    # to 'eval' mode instead.
    mode = 'eval'

  attention = ct.ApplyAttentionLayer(
      attention_type=attention_type, d_model=d_model, n_heads=n_heads,
      d_qk=d_model//n_heads, d_v=d_model//n_heads, masked=True, causal=False,
      attention_dropout=dropout, output_dropout=dropout,
      attention_chunk_size=attention_chunk_size, mode=mode)
  # TODO(lukaszkaiser): refactor efficient attention layers to unify the API
  # If we're using standard attention, we need to pass reshaped mask and not
  # return the mask to be compatible with the EfficientAttention API.
  if attention.n_out == 2:
    def reshape_mask(mask):
      return jnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
    attention = tl.Serial(
        tl.Fn('ReshapeMask', lambda x, y: (x, reshape_mask(y)), n_out=2),
        attention,
        tl.Select([0], n_in=2)
    )

  attention_half_residual = tl.ReversibleHalfResidual(
      tl.LayerNorm(),
      attention_layer=attention,
  )

  feed_forward = ct.FeedForwardWithOptions(
      d_model, d_ff, dropout, [-2], ff_activation, ff_dropout,
      ff_chunk_size, ff_use_sru, ff_sparsity, mode, use_bfloat16)

  return [
      attention_half_residual,
      tl.ReversibleSwap(),
      tl.ReversibleHalfResidual(feed_forward),
      tl.ReversibleSwap(),
  ]


def EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, ff_activation,
                        ff_dropout, mode, ff_use_sru=0, ff_chunk_size=0,
                        ff_sparsity=0):
  """Reversible transformer decoder layer.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: float: (optional) separate dropout rate for feed-forward layer
    mode: str: 'train' or 'eval'
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity

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

  feed_forward = ct.FeedForwardWithOptions(
      d_model, d_ff, dropout, [-2], ff_activation, ff_dropout,
      ff_chunk_size, ff_use_sru, ff_sparsity, mode)

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
             mode='train',
             axial_pos_shape=None,
             d_axial_pos_embs=None,
             ff_use_sru=0,
             ff_chunk_size=0,
             ff_sparsity=0):
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
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, and values must sum to d_model.
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity

  Returns:
    A Reformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  in_encoder, out_encoder, output_vocab_size = (
      ct.EmbeddingAndPositionalEncodings(
          input_vocab_size,
          d_model,
          mode,
          dropout,
          [-2],  # dropout_shared_axes
          max_len,
          output_vocab_size=output_vocab_size,
          axial_pos_shape=axial_pos_shape,
          d_axial_pos_embs=d_axial_pos_embs)
  )

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      EncoderBlock(
          d_model, d_ff, n_heads, tl.SelfAttention, dropout, ff_activation,
          ff_dropout, mode=mode, ff_use_sru=ff_use_sru,
          ff_chunk_size=ff_chunk_size, ff_sparsity=ff_sparsity)
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

  # pylint: disable=g-complex-comprehension
  encoder_decoder_blocks = [
      EncoderDecoderBlock(
          d_model, d_ff, n_heads, dropout, ff_activation, ff_dropout, mode,
          ff_use_sru=ff_use_sru, ff_chunk_size=ff_chunk_size,
          ff_sparsity=ff_sparsity)
      for _ in range(n_decoder_layers)]
  # pylint: enable=g-complex-comprehension

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
              loss_sparsity_type='mult',
              loss_sparsity=0,
              loss_d_lowrank=0,
              loss_sparsity_prob=None,
              attention_chunk_size=0,
              n_layers_forget=0,
              n_decoder_attention_layers=2,
              use_bfloat16=False,
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
    loss_sparsity_type: str, type of sparsity to used in loss layer. See
      SparseDenseWithOptions for options. None if no sparsity should be used.
    loss_sparsity: int, the sparsity for loss layer (if used)
    loss_d_lowrank: int, the dimensions for intermediate layer (if used)
    loss_sparsity_prob: float, the probability for sparse version of loss to be
      used. If None, only sparse version is used.
    attention_chunk_size: int, if > 0 run attention chunked at this size
    n_layers_forget: how often to have a forgetting block between layers
    n_decoder_attention_layers: how many attention layers in a decoder block
    use_bfloat16: whether to use bfloat16 for weights (default: False)
    mode: str: 'train' or 'eval'

  Returns:
    A Reformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
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
  in_encoder, out_encoder, output_vocab_size = (
      ct.EmbeddingAndPositionalEncodings(
          input_vocab_size,
          d_model,
          mode,
          dropout,
          [-2],  # dropout_shared_axes
          max_len,
          output_vocab_size=output_vocab_size,
          axial_pos_shape=axial_pos_shape,
          d_axial_pos_embs=d_axial_pos_embs,
          use_bfloat16=use_bfloat16)
  )

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
          attention_chunk_size=attention_chunk_size,
          use_bfloat16=use_bfloat16,
          mode=mode)
      for _ in range(n_encoder_layers)]
  # pylint: enable=g-complex-comprehension

  encoder = tl.Serial(     # vec_e mask_e tok_e tok_d tok_d
      tl.ReversibleSelect([0, 0]),     # vec_e1 vec_e2 mask_e tok_e tok_d tok_d
      _ReversibleSerialForget(encoder_blocks, d_model, n_layers_forget),
      tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
      tl.Dense(d_model, use_bfloat16=use_bfloat16),
      tl.LayerNorm(),
  )
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
        attention_chunk_size=attention_chunk_size,
        n_attention_layers=n_decoder_attention_layers,
        use_bfloat16=use_bfloat16,
        mode=mode)
    decoder_blocks.append(decoder_block)

  dense_loss_layer = tl.SparseDenseWithOptions(
      output_vocab_size,
      d_input=d_model,
      sparsity_type=loss_sparsity_type,
      sparsity=loss_sparsity,
      d_lowrank=loss_d_lowrank,
      prob_sparse=loss_sparsity_prob,
      use_bfloat16=use_bfloat16,
      mode=mode)

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

      # Concat encoder and decoder, given encoder mask.
      tl.ReversibleSelect([0, 3, 1, 2]),        # vec_e vec_d mask_e tok_e tok_d
      t2.ConcatWithPadding(mode=mode),          # vec_ed tok_e tok_d
      tl.ReversibleSelect([0, 0]),              # vec_ed vec_ed tok_e tok_d

      # Run decoder blocks.
      _ReversibleSerialForget(decoder_blocks, d_model,
                              n_layers_forget),    # vec_ed1 vec_ed2 tok_e tok_d
      tl.Fn('XYAvg',
            lambda x, y: (x + y) / 2.0),           # vec_ed tok_e tok_d
      tl.LayerNorm(),                              # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector.
      tl.Select([0, 1, 2, 2]),                        # vec_ed tok_e tok_d tok_d
      t2.StripFromConcatenateWithPadding(mode=mode),  # vec_d tok_d

      # Map to output vocab.
      dense_loss_layer,  # vec_d tok_d
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
