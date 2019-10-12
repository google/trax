# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Transformer Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from trax import layers as tl


def FeedForward(d_model, d_ff, dropout, layer_idx, mode):
  """Feed-forward block with layer normalization at start."""
  return tl.Serial(
      tl.LayerNorm(),
      tl.Dense(d_ff),
      tl.Relu(),
      tl.Dropout(rate=dropout, name='ff_middle_%d' % layer_idx, mode=mode),
      tl.Dense(d_model),
      tl.Dropout(rate=dropout, name='ff_final_%d' % layer_idx, mode=mode),
  )


def EncoderBlock(d_model, d_ff, n_heads, dropout, layer_idx, mode):
  """Returns a layer sequence that implements a Transformer encoder block.

  The input to the layer sequence is a pair, (activations, mask), where the
  mask was created from the original source tokens to prevent attending to the
  padding part of the input.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'

  Returns:
    A sequence of layers that maps an (activations, mask) pair to an
    (activations, mask) pair.
  """
  attention = [
      tl.LayerNorm(),
      tl.Attention(d_model, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Dropout(rate=dropout, name='enc_attn_dropout', mode=mode),
  ]
  feed_forward = [
      FeedForward(d_model, d_ff, dropout, layer_idx=layer_idx, mode=mode),
  ]
  return tl.Serial(
      tl.Residual(attention),
      tl.Residual(feed_forward),
  )


def TransformerEncoder(vocab_size,
                       n_classes=10,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       max_len=2048,
                       mode='train'):
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

  Returns:
    A Transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
  """
  embedder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='emb_dropout', mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Serial(                             #      tokens
      tl.Dup(),                                 # toks toks
      tl.Parallel(embedder, tl.PaddingMask()),  # vecs mask
      [EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode)
       for i in range(n_layers)],               # vecs mask
      tl.Parallel([], tl.Drop()),               # ____  0
      tl.LayerNorm(),                           # vecs
      tl.Mean(axis=1),  # Average on length.    # vecs
      tl.Dense(n_classes),                      # vecs
      tl.LogSoftmax(),                          # vecs
  )


def DecoderBlock(d_model, d_ff, n_heads, d_attention_key, d_attention_value,
                 attention_type, dropout, share_qk, layer_idx, mode):
  """Returns a layer sequence that implements a Transformer decoder block.

  The input to the layer sequence is an activation tensor.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'

  Returns:
    A sequence of layers that maps an activation tensor to an activation tensor.
  """
  self_attention = [
      tl.LayerNorm(),  # vec
      tl.CausalAttention(
          d_model, n_heads=n_heads, d_attention_key=d_attention_key,
          d_attention_value=d_attention_value, attention_type=attention_type,
          share_qk=share_qk, mode=mode),
      tl.Dropout(rate=dropout, name='attention_%d' % layer_idx, mode=mode),
  ]
  feed_forward = [
      FeedForward(d_model, d_ff, dropout, layer_idx=layer_idx, mode=mode),
  ]
  return tl.Serial(
      tl.Residual(self_attention),
      tl.Residual(feed_forward),
  )


def TransformerDecoder(vocab_size=None,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       d_attention_key=None,
                       d_attention_value=None,
                       attention_type=tl.DotProductCausalAttention,
                       dropout=0.1,
                       share_qk=False,
                       max_len=2048,
                       mode='train'):
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
    d_attention_key: int: depth of key vector for each attention head
        (default is d_model // n_heads)
    d_attention_value: int: depth of value vector for each attention head
        (default is d_model // n_heads)
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys in decoder attention
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    A Transformer decoder as a layer that maps from a continuous or discrete
    tensor to a continuous tensor.
  """
  if vocab_size is None:
    input_layer = tl.Dense
  else:
    input_layer = functools.partial(tl.Embedding, vocab_size=vocab_size)
  return tl.Serial(                  # vecs
      input_layer(d_model),         # vecs
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
      [DecoderBlock(  # pylint: disable=g-complex-comprehension
          d_model, d_ff, n_heads, d_attention_key, d_attention_value,
          attention_type, dropout, share_qk, i, mode)
       for i in range(n_layers)],   # vecs
      tl.LayerNorm(),               # vecs
  )


def TransformerLM(vocab_size,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  d_attention_key=None,
                  d_attention_value=None,
                  attention_type=tl.DotProductCausalAttention,
                  dropout=0.1,
                  share_qk=False,
                  max_len=2048,
                  n_chunks=0,
                  mode='train'):
  """Returns a Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head
        (default is d_model // n_heads)
    d_attention_value: int: depth of value vector for each attention head
        (default is d_model // n_heads)
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys in decoder attention
    max_len: int: maximum symbol length for positional encoding
    n_chunks: int: number of chunks (must match input pipeline)
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  if n_chunks == 0:
    concatenate_chunks = split_chunks = []
  else:
    concatenate_chunks = tl.Concatenate(n_items=n_chunks)
    split_chunks = tl.Split(n_sections=n_chunks, axis=-2)

  embedder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='embedding', mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  return tl.Serial(                  # tokens (or chunked tuple of tokens)
      concatenate_chunks,           # tokens
      tl.ShiftRight(mode=mode),     # toks
      embedder,                     # vecs
      [DecoderBlock(  # pylint: disable=g-complex-comprehension
          d_model, d_ff, n_heads, d_attention_key, d_attention_value,
          attention_type, dropout, share_qk, i, mode)
       for i in range(n_layers)],   # vecs
      tl.LayerNorm(),               # vecs
      tl.Dense(vocab_size),         # vecs
      tl.LogSoftmax(),              # vecs
      split_chunks,                 # vecs (or chunked tuple of vecs)
  )


def EncoderDecoder(d_model, d_ff, n_heads, dropout, layer_idx, mode):
  """Transformer encoder-decoder layer.

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

  Returns:
    the layer, returning a triple (decoder_activations, mask, encoder).
  """
  # tgt_vecs --> tgt_vecs'
  decoder_self_attention = tl.Serial(
      tl.LayerNorm(),
      tl.BasicCausalAttention(d_model,
                              n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Dropout(rate=dropout, mode=mode)
  )

  @tl.symbolic
  def decoder_to_encoder_attention(tgt_vecs, masks, src_vecs):
    attention = tl.AttentionQKV(d_model,
                                n_heads=n_heads, dropout=dropout, mode=mode)
    tgt_vecs = tl.LayerNorm() @ tgt_vecs
    tgt_vecs, masks = attention @ (tgt_vecs, src_vecs, src_vecs, masks)
    return tgt_vecs, masks, src_vecs

  # tgt_vecs --> tgt_vecs'
  feed_forward = FeedForward(d_model, d_ff, dropout,
                             layer_idx=layer_idx, mode=mode)

  # tgt_vecs, padmasks, src_vecs --> tgt_vecs', padmasks, src_vecs
  return tl.Serial(
      tl.Residual(decoder_self_attention),
      tl.Residual(decoder_to_encoder_attention),
      tl.Residual(feed_forward),
  )


def Transformer(input_vocab_size,
                output_vocab_size=None,
                d_model=512,
                d_ff=2048,
                n_layers=6,
                n_heads=8,
                dropout=0.1,
                max_len=2048,
                mode='train'):
  """Returns a Transformer model.

  This model expects an input pair: target, source.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    A Transformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  # create embeddings for source and target vocabs
  make_embedding_layer = lambda vocab_size: tl.Serial(
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  )
  src_embedding = make_embedding_layer(input_vocab_size)
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size
    tgt_embedding = src_embedding
  else:
    tgt_embedding = make_embedding_layer(output_vocab_size)

  @tl.symbolic
  def model(src_tokens, tgt_tokens):
    """Takes source-tokens, target-tokens; returns logits, target-tokens."""

    # embed source and target tokens
    src_vecs = src_embedding @ src_tokens
    tgt_vecs = tl.Serial(tl.ShiftRight(), tgt_embedding) @ tgt_tokens

    # create padding mask for encoder and decoder stacks
    enc_pad_masks = tl.PaddingMask() @ src_tokens
    endec_pad_masks = tl.EncoderDecoderMask() @ (tgt_vecs, enc_pad_masks)

    # encoder stack
    for i in range(n_layers):
      enc = EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode)
      src_vecs, _ = enc @ (src_vecs, enc_pad_masks)
    src_encoding = tl.LayerNorm() @ src_vecs

    # encoder-decoder stack
    for i in range(n_layers):
      encdec = EncoderDecoder(d_model, d_ff, n_heads, dropout, i, mode)
      tgt_vecs, _, _ = encdec @ (tgt_vecs, endec_pad_masks, src_encoding)
    tgt_decoding = tl.LayerNorm() @ tgt_vecs

    # map decoding to probabilities over output vocab
    logprobs = tl.Serial(tl.Dense(output_vocab_size),
                         tl.LogSoftmax()) @ tgt_decoding

    return logprobs, tgt_tokens

  return model
