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
"""ReZero Transformer Models.

ReZero transformer (https://arxiv.org/abs/2003.04887) is based on a simple
change to residual connections. Instead of adding a result of a skip connection
to the output of a layer, it has a learnable scalar alpha, initialized with
zero, which scales the output of the layer before adding it to the skip
connection.
"""

from trax import layers as tl
from trax.fastmath import numpy as jnp


def ResidualZero(*layers, shortcut=None):
  """Wraps a series of layers with a ReZero-style residual connection.

  Instead of computing `(shortcut) + (output of layers)`, like in classical
  Residual connection, ResidualZero computes
  `(shortcut) + alpha * (output of layers)`, where `alpha` is a learnable scalar
  initialized with zero.

  Args:
    *layers: One or more layers, to be applied in series.
    shortcut: If None (the usual case), the Residual layer computes the
        element-wise sum of the stack-top input with the output of the layer
        series. If specified, the `shortcut` layer applies to a copy of the
        inputs and (elementwise) adds its output to the output from the main
        layer series.

  Returns:
      A layer representing a residual connection paired with a layer series.
  """
  layers = _ensure_flat(layers)
  layer = layers[0] if len(layers) == 1 else tl.Serial(layers)
  # TODO(jaszczur): perhaps change inner Serial to Branch?
  return tl.Serial(
      tl.Branch(shortcut, tl.Serial(
          layer,
          tl.Weights(lambda shape, rng: jnp.zeros(shape, dtype=jnp.float32)),
          tl.Multiply()
          )),
      tl.Add(),  # pylint: disable=no-value-for-parameter
  )


def ReZeroTransformerEncoder(vocab_size,
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
  """Returns a ReZero transformer encoder model.

  The input to the model is a tensor of tokens.

  Args:
    vocab_size: int: vocab size
    n_classes: how many classes on output
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
    A ReZero transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
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
  )


def ReZeroTransformerDecoder(vocab_size=None,
                             d_model=512,
                             d_ff=2048,
                             n_layers=6,
                             n_heads=8,
                             dropout=0.1,
                             dropout_shared_axes=None,
                             max_len=2048,
                             mode='train',
                             ff_activation=tl.Relu):
  """Returns a ReZero transformer decoder model.

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
    A ReZero transformer decoder as a layer that maps from a continuous or
    discrete tensor to a continuous tensor.
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


def ReZeroTransformerLM(vocab_size,
                        d_model=512,
                        d_ff=2048,
                        n_layers=6,
                        n_heads=8,
                        dropout=0.1,
                        dropout_shared_axes=None,
                        max_len=2048,
                        mode='train',
                        ff_activation=tl.Relu):
  """Returns a ReZero transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall ReZero transformer.)

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
    A ReZero transformer language model as a layer that maps from a tensor of
    tokens to activations over a vocab set.
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
  )


def ReZeroTransformer(input_vocab_size,
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
  """Returns a ReZero transformer model.

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
    A ReZero transformer model as a layer that maps from a source, target pair
    to activations over a vocab set.
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
      ResidualZero(
          tl.LayerNorm(),
          attention,
          dropout_,
      ),
      ResidualZero(
          tl.LayerNorm(),
          feed_forward,
          dropout_,
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
      ResidualZero(
          tl.LayerNorm(),
          causal_attention,
          dropout_,
      ),
      ResidualZero(
          tl.LayerNorm(),
          feed_forward,
          dropout_,
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
      d_model, n_heads=n_heads, dropout=dropout, mode=mode,
      cache_KV_in_predict=True)

  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  return [                             # vec_d masks vec_e
      ResidualZero(
          tl.LayerNorm(),              # vec_d ..... .....
          causal_attention,            # vec_d ..... .....
          _Dropout(),                  # vec_d ..... .....
      ),
      ResidualZero(
          tl.LayerNorm(),              # vec_d ..... .....
          tl.Select([0, 2, 2, 1, 2]),  # vec_d vec_e vec_e masks vec_e
          attention_qkv,               # vec_d masks vec_e
          _Dropout(),                  # vec_d masks vec_e
      ),
      ResidualZero(
          tl.LayerNorm(),
          feed_forward,                # vec_d masks vec_e
          _Dropout(),
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
  dropout = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Dense(d_ff),
      activation(),
      dropout,
      tl.Dense(d_model),
  ]


# All module-private helper functions are below.
# pylint: disable=invalid-name


def _deep_flatten(items):
  """Returns a list of objects, flattening sublists/subtuples along the way.

  Example: _deep_flatten([1, (2, 3, (4, 5), [6, 7]), [[[8]]]]) would return
  the list [1, 2, 3, 4, 5, 6, 7, 8].

  Args:
    items: An iterable. If elements of this iterable are lists or tuples, they
        will be (recursively) flattened until non-list non-tuple objects are
        reached.

  Returns:
    A list of non-list, non-tuple objects.
  """
  def _flat_gen(xs):
    for x in xs:
      if isinstance(x, (list, tuple)):
        for y in _flat_gen(x):
          yield y
      else:
        yield x
  return list(_flat_gen(items))


def _ensure_flat(layers):
  """Ensures that layers is a single flat list of Layer instances."""
  if len(layers) == 1 and layers[0] is None:
    layers = ()
  else:
    layers = _deep_flatten(layers)
  for obj in layers:
    if not isinstance(obj, tl.Layer):
      raise ValueError(
          f'Found nonlayer object ({obj}) in layers: {layers}')
  return layers
