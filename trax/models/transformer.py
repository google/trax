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

from trax import layers as tl


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
  def EncoderBlocks(n_blocks):  # vectors masks --> vectors masks
    return [
        _EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
        for i in range(n_blocks)]

  positional_embedder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='emb_dropout', mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]

  # Assemble and return the model.
  return tl.Serial(                                # toks
      # Encode.
      tl.Branch(
          positional_embedder, tl.PaddingMask()),  # vecs masks
      EncoderBlocks(n_layers),                     # vecs masks
      tl.LayerNorm(),                              # vecs .....

      # Map to output categories.
      tl.Select([0], n_in=2),                      # vecs
      tl.Mean(axis=1),                             # vecs
      tl.Dense(n_classes),                         # vecs
      tl.LogSoftmax(),                             # vecs
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
    d_attention_key: int: depth of key vector for each attention head (default
      is d_model // n_heads)
    d_attention_value: int: depth of value vector for each attention head
      (default is d_model // n_heads)
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys in decoder attention
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer decoder as a layer that maps from a continuous or discrete
    tensor to a continuous tensor.
  """
  def DecoderBlocks(n_blocks):  # vectors --> vectors
    return [  # pylint: disable=g-complex-comprehension
        _DecoderBlock(d_model, d_ff, n_heads,
                      d_attention_key, d_attention_value, attention_type,
                      dropout, share_qk, i, mode, ff_activation)
        for i in range(n_blocks)]

  embedding_or_dense = (
      tl.Embedding(d_model, vocab_size) if vocab_size is not None
      else tl.Dense(d_model))
  dropout_ = tl.Dropout(rate=dropout, mode=mode)
  positional_encoding = tl.PositionalEncoding(max_len=max_len)

  # Assemble and return the model.
  return tl.Serial(             # toks
      embedding_or_dense,       # vecs
      dropout_,                 # vecs
      positional_encoding,      # vecs
      DecoderBlocks(n_layers),  # vecs
      tl.LayerNorm(),           # vecs
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
    d_attention_key: int: depth of key vector for each attention head (default
      is d_model // n_heads)
    d_attention_value: int: depth of value vector for each attention head
      (default is d_model // n_heads)
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys in decoder attention
    max_len: int: maximum symbol length for positional encoding
    n_chunks: int: number of chunks (must match input pipeline)
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  def DecoderBlocks(n_blocks):  # vectors --> vectors
    return [  # pylint: disable=g-complex-comprehension
        _DecoderBlock(d_model, d_ff, n_heads,
                      d_attention_key, d_attention_value, attention_type,
                      dropout, share_qk, i, mode, ff_activation)
        for i in range(n_blocks)]

  if n_chunks == 0:
    concatenate_chunks = []
    split_chunks = []
  else:
    concatenate_chunks = tl.Concatenate(n_items=n_chunks)
    split_chunks = tl.Split(n_items=n_chunks, axis=-2)

  embedding = tl.Embedding(d_model, vocab_size)
  dropout_ = tl.Dropout(rate=dropout, name='embedding', mode=mode)
  positional_encoding = tl.PositionalEncoding(max_len=max_len, mode=mode)

  # Assemble and return the model.
  return tl.Serial(              # tokens (or chunked tuple of tokens)
      concatenate_chunks,        # toks
      tl.ShiftRight(mode=mode),  # toks
      embedding,                 # vecs
      dropout_,                  # vecs
      positional_encoding,       # vecs
      DecoderBlocks(n_layers),   # vecs
      tl.LayerNorm(),            # vecs
      tl.Dense(vocab_size),      # vecs
      tl.LogSoftmax(),           # vecs
      split_chunks,              # vecs (or chunked tuple of vecs)
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
  def PositionalEmbedder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(d_model, vocab_size),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len),
    ]

  def EncoderBlocks(n_blocks):  # vectors masks --> vectors masks
    return [
        _EncoderBlock(d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
        for i in range(n_blocks)]

  def EncoderDecoderBlocks(n_blocks):  # vectors masks --> vectors masks
    return [
        _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, i, mode,
                             ff_activation)
        for i in range(n_blocks)]

  in_embed = PositionalEmbedder(input_vocab_size)
  out_embed = (in_embed if output_vocab_size is None
               else PositionalEmbedder(output_vocab_size))
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 1, 1]),                    # tok_e tok_d tok_d

      # Encode.
      tl.Branch(
          in_embed, tl.PaddingMask()),         # vec_e masks ..... .....
      EncoderBlocks(n_encoder_layers),         # vec_d masks ..... .....
      tl.LayerNorm(),                          # vec_e ..... ..... .....

      # Decode.
      tl.Select([2, 1, 0]),                    # tok_d masks vec_e .....
      tl.ShiftRight(),                         # tok_d ..... ..... .....
      out_embed,                               # vec_d ..... ..... .....
      tl.Branch(
          [], tl.EncoderDecoderMask()),        # vec_d masks ..... .....
      EncoderDecoderBlocks(n_decoder_layers),  # vec_d masks ..... .....
      tl.LayerNorm(),                          # vec_d ..... ..... .....

      # Map to output vocab.
      tl.Parallel([], tl.Drop(), tl.Drop()),   # vec_d tok_d
      tl.Dense(output_vocab_size),             # vec_d .....
      tl.LogSoftmax(),                         # vec_d .....
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


def _DecoderBlock(d_model, d_ff, n_heads, d_attn_key, d_attn_value, attn_type,
                  dropout, share_qk, layer_idx, mode, ff_activation):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    d_attn_key: int: depth of key vector for each attention head
    d_attn_value: int: depth of value vector for each attention head
    attn_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, d_attention_key=d_attn_key,
      d_attention_value=d_attn_value, attention_type=attn_type,
      share_qk=share_qk, mode=mode),

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

  basic_causal_attention = tl.BasicCausalAttention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, layer_idx, mode, ff_activation)

  return [                             # vec_d masks vec_e
      tl.Residual(
          tl.LayerNorm(),              # vec_d ..... .....
          basic_causal_attention,      # vec_d masks .....
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
