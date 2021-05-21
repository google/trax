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
"""Transformer models: encoder, decoder, language model, and encoder-decoder.

The "Transformer" name and network architecture were introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

from trax import layers as tl


# Defaults used across Transformer variants.
MODE = 'train'
D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
N_HEADS = 8
MAX_SEQUENCE_LENGTH = 2048
DROPOUT_RATE = .1
DROPOUT_SHARED_AXES = None
FF_ACTIVATION_TYPE = tl.Relu


def TransformerEncoder(vocab_size,
                       n_classes=10,
                       d_model=D_MODEL,
                       d_ff=D_FF,
                       n_layers=N_LAYERS,
                       n_heads=N_HEADS,
                       max_len=MAX_SEQUENCE_LENGTH,
                       dropout=DROPOUT_RATE,
                       dropout_shared_axes=DROPOUT_SHARED_AXES,
                       mode=MODE,
                       ff_activation=FF_ACTIVATION_TYPE):
  """Returns a Transformer encoder suitable for N-way classification.

  This model maps tokenized text to N-way (``n_classes``) activations:

    - input: Array representing a batch of text strings via token IDs plus
      padding markers; shape is (batch_size, sequence_length), where
      sequence_length <= ``max_len``. Array elements are integers in
      ``range(vocab_size)``, and 0 values mark padding positions.

    - output: Array representing a batch of raw (non-normalized) activations
      over ``n_classes`` categories; shape is (batch_size, ``n_classes``).

  Args:
    vocab_size: Input vocabulary size -- each element of the input array
        should be an integer in ``range(vocab_size)``. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    n_classes: Last/innermost dimension of output arrays, suitable for N-way
        classification.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of encoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder blocks. The same rate is also
        used for attention dropout in encoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each encoder block will include dropout; else, it
        will pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A Transformer model that maps strings (conveyed by token IDs) to
    raw (non-normalized) activations over a range of output classes.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _EncBlock():
    return _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)

  return tl.Serial(
      tl.Branch([], tl.PaddingMask()),  # Creates masks from copy of the tokens.
      tl.Embedding(vocab_size, d_model),
      _Dropout(),
      tl.PositionalEncoding(max_len=max_len),
      [_EncBlock() for _ in range(n_layers)],
      tl.Select([0], n_in=2),  # Drops the masks.
      tl.LayerNorm(),
      tl.Mean(axis=1),
      tl.Dense(n_classes),
  )


def TransformerDecoder(vocab_size=None,
                       d_model=D_MODEL,
                       d_ff=D_FF,
                       n_layers=N_LAYERS,
                       n_heads=N_HEADS,
                       max_len=MAX_SEQUENCE_LENGTH,
                       dropout=DROPOUT_RATE,
                       dropout_shared_axes=DROPOUT_SHARED_AXES,
                       mode=MODE,
                       ff_activation=FF_ACTIVATION_TYPE):
  """Returns a Transformer decoder.

  This model maps sequential inputs to sequential outputs:

    - input if ``vocab_size`` is specified: array representing a batch
      of text strings via token IDs plus padding markers; shape is
      (batch_size, sequence_length). The tensor elements are integers in
      ``range(vocab_size)``, and 0 values mark padding positions.

    - input if ``vocab_size`` is ``None``: 3-D array representing a batch of
      sequences of activation vectors; shape is (batch_size, sequence_length,
      ``d_model``).

    - output: 3-D array with shape (batch_size, sequence_length, ``d_model``).

  The model uses causal attention and does *not* shift the input to the right.
  Thus, the output for position `t` is based on inputs up to and including
  position `t`.

  Args:
    vocab_size: If specified, gives the input vocabulary size -- each element
        of the input tensor should be an integer in ``range(vocab_size)``.
        If ``None``, indicates that the model expects as input sequences of
        floating point vectors, each with ``d_model`` components.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of decoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within decoder blocks. The same rate is also
        used for attention dropout in decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each encoder block will include dropout; else, it
        will pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of :py:class:`Layer`.

  Returns:
    If ``vocab_size`` is defined: a Transformer model that maps strings
    (conveyed by token IDs) to sequences of activation vectors.

    If ``vocab_size`` is ``None``: a Transformer model that maps sequences of
    activation vectors to sequences of activation vectors.
  """
  def _EmbeddingOrDense():
    return (tl.Embedding(vocab_size, d_model) if vocab_size is not None
            else tl.Dense(d_model))

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _DecBlock():
    return _DecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)

  return tl.Serial(
      _EmbeddingOrDense(),
      _Dropout(),
      tl.PositionalEncoding(max_len=max_len),
      [_DecBlock() for _ in range(n_layers)],
      tl.LayerNorm(),
  )


def TransformerLM(vocab_size,
                  d_model=D_MODEL,
                  d_ff=D_FF,
                  n_layers=N_LAYERS,
                  n_heads=N_HEADS,
                  max_len=MAX_SEQUENCE_LENGTH,
                  dropout=DROPOUT_RATE,
                  dropout_shared_axes=DROPOUT_SHARED_AXES,
                  mode=MODE,
                  ff_activation=FF_ACTIVATION_TYPE):
  """Returns a Transformer language model.

  This model performs autoregressive language modeling:

    - input: Array representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). Array
      elements are integers in ``range(vocab_size)``, and 0 values mark padding
      positions.

    - output: 3-D array of raw activations with last/innermost dimension of
      ``vocab_size``, suitable for decoding into a batch of token strings;
      shape is (batch_size, sequence_length, ``vocab_size``).

  This model uses only the decoder part of the overall Transformer.

  Args:
    vocab_size: Input vocabulary size -- each element of the input array
        should be an integer in ``range(vocab_size)``. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of decoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within decoder blocks. The same rate is also
        used for attention dropout in decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'predict'``, use fast inference. If ``'train'``, each decoder
        block will include dropout; else, it will pass all values through
        unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A Transformer language model that maps strings (represented as token ID
    sequences) to sequences of raw (non-normalized) activation vectors; each
    vector in the sequence can be mapped (e.g., by `argmax`) to a token ID.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _DecBlock():
    return _DecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(vocab_size, d_model),
      _Dropout(),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
      [_DecBlock() for _ in range(n_layers)],
      tl.LayerNorm(),
      tl.Dense(vocab_size),
  )


def Transformer(input_vocab_size,
                output_vocab_size=None,
                d_model=D_MODEL,
                d_ff=D_FF,
                n_encoder_layers=N_LAYERS,
                n_decoder_layers=N_LAYERS,
                n_heads=N_HEADS,
                max_len=MAX_SEQUENCE_LENGTH,
                dropout=DROPOUT_RATE,
                dropout_shared_axes=DROPOUT_SHARED_AXES,
                mode=MODE,
                ff_activation=FF_ACTIVATION_TYPE):
  """Returns a full Transformer model.

  This model is an encoder-decoder that performs tokenized string-to-string
  ("source"-to-"target") transduction:

    - inputs (2):

        - source: Array representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length),
          where sequence_length <= ``max_len``. Array elements are integers in
          ``range(input_vocab_size)``, and 0 values mark padding positions.

        - target: Array representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length),
          where sequence_length <= ``max_len``. Array elements are integers in
          ``range(output_vocab_size)``, and 0 values mark padding positions.

    - output: 3-D array of raw activations with last/innermost dimension of
      ``output_vocab_size``, suitable for decoding into a batch of token
      strings; shape is (batch_size, sequence_length, ``vocab_size``).

  An example use would be to translate (tokenized) sentences from English to
  German.

  Args:
    input_vocab_size: Input vocabulary size -- each element of the input tensor
        should be an integer in ``range(vocab_size)``. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    output_vocab_size: If specified, gives the vocabulary size for the targets;
        if ``None``, then input and target integers (token IDs) are assumed to
        come from the same vocabulary.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_encoder_layers: Number of encoder blocks.
    n_decoder_layers: Number of decoder blocks.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder/decoder blocks. The same rate is
        also used for attention dropout in encoder/decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'predict'``, use fast inference. If ``'train'``, each
        encoder/decoder block will include dropout; else, it will pass all
        values through unaltered.
    ff_activation: Type of activation function at the end of each
        encoder/decoder block; must be an activation-type subclass of
        :py:class:`Layer`.

  Returns:
    A Transformer model as a layer that maps from a source-target tokenized
    text pair to activations over a vocab set.
  """
  # Avoid 'predict' mode in encoder, since encoder doesn't run stepwise.
  encoder_mode = 'eval' if mode == 'predict' else mode

  # Share embedding weights if no separate output vocab size.
  in_embedder = tl.Embedding(input_vocab_size, d_model)
  if output_vocab_size is None:
    out_embedder = in_embedder
    output_vocab_size = input_vocab_size
  else:
    out_embedder = tl.Embedding(output_vocab_size, d_model)

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _EncBlock():
    return _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)

  def _Encoder():
    encoder = tl.Serial(
        in_embedder,
        _Dropout(),
        tl.PositionalEncoding(max_len=max_len, mode=encoder_mode),
        [_EncBlock() for _ in range(n_encoder_layers)],
        tl.LayerNorm(),
    )
    return tl.Cache(encoder) if mode == 'predict' else encoder

  def _EncDecBlock():
    return _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout,
                                dropout_shared_axes, mode, ff_activation)

  # Input to model is encoder-side tokens and decoder-side tokens: tok_d, tok_e
  # Model output is decoder-side vectors and decoder-side tokens: vec_d  tok_d
  return tl.Serial(
      tl.Select([0, 1, 1]),  # Copies decoder tokens for use in loss.

      # Encode.
      tl.Branch([], tl.PaddingMask()),  # tok_e masks tok_d tok_d
      _Encoder(),

      # Decode.
      tl.Select([2, 1, 0]),  # Re-orders inputs: tok_d masks vec_e .....
      tl.ShiftRight(mode=mode),
      out_embedder,
      _Dropout(),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
      tl.Branch([], tl.EncoderDecoderMask()),  # vec_d masks ..... .....
      [_EncDecBlock() for _ in range(n_decoder_layers)],
      tl.LayerNorm(),
      tl.Select([0], n_in=3),  # Drops masks and encoding vectors.

      # Map vectors to match output vocab size.
      tl.Dense(output_vocab_size),
  )


def _EncoderBlock(d_model,
                  d_ff,
                  n_heads,
                  dropout,
                  dropout_shared_axes,
                  mode,
                  ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the block is a pair (activations, mask) where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input. The block's outputs are the same type/shape as its inputs,
  so that multiple blocks can be chained together.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder blocks. The same rate is also used
        for attention dropout in encoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A list of layers that act in series as a (repeatable) encoder block.
  """
  def _Attention():
    return tl.Attention(d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)

  return [
      tl.Residual(
          tl.LayerNorm(),
          _Attention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]


def _DecoderBlock(d_model,
                  d_ff,
                  n_heads,
                  dropout,
                  dropout_shared_axes,
                  mode,
                  ff_activation):
  """Returns a list of layers that implements a Transformer decoder block.

  The input to the block is a pair (activations, mask) where the mask encodes
  causal connections, preventing attention to future positions in the sequence.
  The block's outputs are the same type/shape as its inputs, so that multiple
  blocks can be chained together.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within decoder blocks. The same rate is also used
        for attention dropout in decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A list of layers that act in series as a (repeatable) decoder block.
  """
  def _CausalAttention():
    return tl.CausalAttention(d_model, n_heads=n_heads, dropout=dropout,
                              mode=mode),

  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          _CausalAttention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]


def _EncoderDecoderBlock(d_model,
                         d_ff,
                         n_heads,
                         dropout,
                         dropout_shared_axes,
                         mode,
                         ff_activation):
  """Returns a list of layers implementing a Transformer encoder-decoder block.

  The block input is a triple (decoder_activations, mask, encoder_activations)
  where the mask was created from the original input token IDs to prevent
  attending to padding positions for that input.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder/decoder blocks. The same rate is
        also used for attention dropout in encoder/decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A list of layers that act in series as a (repeatable) encoder-decoder
    block.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _AttentionQKV():
    return tl.AttentionQKV(d_model, n_heads=n_heads, dropout=dropout,
                           mode=mode, cache_KV_in_predict=True)

  def _CausalAttention():
    return tl.CausalAttention(d_model, n_heads=n_heads, mode=mode)

  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)

  return [                             # vec_d masks vec_e
      tl.Residual(
          tl.LayerNorm(),
          _CausalAttention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          tl.Select([0, 2, 2, 1, 2]),  # vec_d vec_e vec_e masks vec_e
          _AttentionQKV(),             # vec_d masks vec_e
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]


def _FeedForwardBlock(d_model,
                      d_ff,
                      dropout,
                      dropout_shared_axes,
                      mode,
                      activation):
  """Returns a list of layers that implements a feedforward block.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A list of layers that maps vectors to vectors.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Dense(d_ff),
      activation(),
      _Dropout(),
      tl.Dense(d_model),
  ]
