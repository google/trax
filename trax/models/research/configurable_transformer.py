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


def _FeedForward(d_model, d_ff, dropout, activation, act_dropout,
                 use_bfloat16, mode):
  """Feed-forward block with layer normalization at start."""
  if act_dropout is None:
    act_dropout = dropout
  return [
      tl.Dense(d_ff, use_bfloat16=use_bfloat16),
      tl.Dropout(rate=act_dropout, shared_axes=[-2], mode=mode),
      activation(),
      tl.Dense(d_model, use_bfloat16=use_bfloat16),
  ]


def FeedForwardWithOptions(d_model,
                           d_ff,
                           dropout,
                           dropout_shared_axes,
                           ff_activation,
                           ff_dropout,
                           ff_chunk_size,
                           ff_use_sru,
                           ff_sparsity,
                           center_layernorm,
                           mode,
                           use_bfloat16=False,
                           ff_sparsity_type='1inN'):
  """Feed-Forward block with all the options.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    ff_activation: Type of activation function at the end of each block; must be
      an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, tuple or string; if not 0, use sparse feed-forward block
      with this sparsity
    center_layernorm: whether to use centering in LayerNorm (default) or if
      to skip it, which is known as RMS normalization.
    mode: If `'train'`, each block will include dropout; else, it will pass all
      values through unaltered.
    use_bfloat16: whether to use bfloat16 for weights (default: False).
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
      use SwitchSparseFF if ff_sparsity_type=`'Switch'`

  Returns:
    A list of layers which maps vectors to vectors.
  """
  if ff_sparsity and ff_sparsity_type == '1inN':
    temperature, quant_prob = 0.1, 0.3
    if isinstance(ff_sparsity, str):
      # This is hacky but used to pass ff_sparsity in yaml sweep files.
      ff_sparsity = [(float(x) if '.' in x else int(x))
                     for x in ff_sparsity.split()]
    if isinstance(ff_sparsity, (list, tuple)):
      if len(ff_sparsity) == 2:
        n_elements_in_block, d_lowrank = ff_sparsity
      else:
        n_elements_in_block, d_lowrank, temperature, quant_prob = ff_sparsity
    else:
      assert isinstance(ff_sparsity, int)
      n_elements_in_block, d_lowrank = ff_sparsity, d_ff // ff_sparsity
    ff = tl.SparseFF(
        d_ff,
        n_elements_in_block=n_elements_in_block,
        d_lowrank=d_lowrank,
        temperature=temperature,
        quant_prob=quant_prob,
        use_bfloat16=use_bfloat16,
        mode=mode,
        dropout_rate=dropout,
        dropout_shared_axes=dropout_shared_axes,
        ff_chunk_size=ff_chunk_size)
  elif ff_sparsity and ff_sparsity_type == 'Block':
    ff = tl.BlockSparseFF(d_ff, n_experts=ff_sparsity, mode=mode)
  elif ff_sparsity and ff_sparsity_type == 'Switch':
    ff = tl.SwitchSparseFF(d_ff, n_experts=ff_sparsity, mode=mode)
  else:
    ff = _FeedForward(d_model, d_ff, dropout, ff_activation, ff_dropout,
                      use_bfloat16, mode)
  res = [tl.LayerNorm(center=center_layernorm), ff]
  if ff_sparsity_type != '1inN' or ff_sparsity == 0:
    # SparseFF has Dropout and BatchLeadingAxes built-in.
    res.append(tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes,
                          mode=mode))
    if ff_chunk_size > 0:
      res = tl.BatchLeadingAxes(tl.Chunk(tl.Serial(res), ff_chunk_size))
  if ff_use_sru:
    if isinstance(ff_use_sru, (list, tuple)):
      sru_n_layers, sru_n_units = ff_use_sru
    else:
      sru_n_layers, sru_n_units = ff_use_sru, 32
    sru = [tl.SRU(sru_n_units, mode=mode) for _ in range(sru_n_layers)]
    block = [tl.LayerNorm(center=center_layernorm), tl.Dense(sru_n_units)
             ] + sru + [tl.Dense(d_model)]
    res = tl.Residual(block, shortcut=res)
  return [res]


# TODO(lukaszkaiser): unify attention layers API and remove this branch
def ApplyAttentionLayer(attention_type, d_model, n_heads, d_qk, d_v, causal,
                        masked, attention_dropout, output_dropout,
                        attention_chunk_size, mode):
  """Runs the supplied attention layer."""
  try:
    attention = attention_type(
        n_heads=n_heads,
        d_qk=d_qk,
        d_v=d_v,
        causal=causal,
        masked=masked,
        output_dropout=output_dropout,
        attention_dropout=attention_dropout,
        mode=mode)
  except TypeError:  # No d_qk arguments in less advanced layers.
    attention = attention_type(
        d_model, n_heads=n_heads, dropout=attention_dropout, mode=mode)
  return tl.Chunk(attention, attention_chunk_size)


@tl.assert_shape('...d->...d')
def PositionalEncoder(mode,
                      dropout=None,
                      max_len=None,
                      pos_type=None,
                      pos_axial_shape=None,
                      pos_d_axial_embs=None,
                      pos_start_from_zero_prob=1.0,
                      pos_max_offset_to_add=0,
                      use_bfloat16=False):
  """Returns the positional encoding layer depending on the arguments.

  Args:
    mode: If `'predict'`, use fast inference. If `'train'`, each encoder/decoder
      block will include dropout; else, it will pass all values through
      unaltered.
    dropout: Stochastic rate (probability) for dropping an activation
      value when applying dropout after the embedding block.
    max_len: Maximum symbol length for positional encoding.
    pos_type: string, the type of positional embeddings to use.
    pos_axial_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match pos_axial_shape, and values must sum to d_model.
    pos_start_from_zero_prob: how often to start from 0 during training,
          (if 1.0, we always start from position 0, if less, we randomize).
    pos_max_offset_to_add: maximum offset to add to positions during training
        when randomizing; this offset plus input length must still be less than
        max_len for all training examples.
    use_bfloat16: If `True`, use bfloat16 weights instead of the default
      float32; this can save memory but may (rarely) lead to numerical issues.

  Returns:
    A layer that will do the positional encoding.
  """
  if not pos_type:
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, use_bfloat16=use_bfloat16,
        start_from_zero_prob=pos_start_from_zero_prob,
        max_offset_to_add=pos_max_offset_to_add, mode=mode)
  elif pos_type == 'sin-cos':
    positional_encoding = tl.SinCosPositionalEncoding(mode=mode)
  elif pos_type == 'fixed-base':
    positional_encoding = tl.FixedBasePositionalEncoding(mode=mode)
  elif pos_type == 'infinite':
    positional_encoding = tl.InfinitePositionalEncoding(affine=False)
  elif pos_type == 'infinite-affine':
    positional_encoding = tl.InfinitePositionalEncoding()
  elif pos_type == 'time-bin':
    positional_encoding = tl.TimeBinPositionalEncoding()
  elif pos_type == 'no':
    positional_encoding = tl.Serial()  # no positional encoding at all
  else:  # TODO(lukaszkaiser): name this type and check for the correct name
    assert pos_d_axial_embs is not None
    positional_encoding = tl.AxialPositionalEncoding(
        shape=pos_axial_shape, d_embs=pos_d_axial_embs,
        dropout_broadcast_dims=tuple(range(1, len(pos_axial_shape) + 1)),
        dropout=dropout, mode=mode)

  return positional_encoding


def EmbeddingAndPositionalEncodings(input_vocab_size,
                                    d_model,
                                    mode,
                                    embedding_dropout,
                                    dropout_shared_axes,
                                    max_len,
                                    output_vocab_size=None,
                                    pos_type=None,
                                    pos_axial_shape=None,
                                    pos_d_axial_embs=None,
                                    pos_start_from_zero_prob=1.0,
                                    pos_max_offset_to_add=0,
                                    use_bfloat16=False):
  """Returns the embedder and positional encoder.

  Args:
    input_vocab_size: Input vocabulary size -- each element of the input tensor
      should be an integer in `range(vocab_size)`. These integers typically
      represent token IDs from a vocabulary-based tokenizer.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    mode: If `'predict'`, use fast inference. If `'train'`, each encoder/decoder
      block will include dropout; else, it will pass all values through
      unaltered.
    embedding_dropout: Stochastic rate (probability) for dropping an activation
      value when applying dropout after the embedding block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    max_len: Maximum symbol length for positional encoding.
    output_vocab_size: If specified, gives the vocabulary size for the targets;
      if None, then input and target integers (token IDs) are assumed to come
      from the same vocabulary.
    pos_type: string, the type of positional embeddings to use.
    pos_axial_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match pos_axial_shape, and values must sum to d_model.
    pos_start_from_zero_prob: how often to start from 0 during training,
          (if 1.0, we always start from position 0, if less, we randomize).
    pos_max_offset_to_add: maximum offset to add to positions during training
        when randomizing; this offset plus input length must still be less than
        max_len for all training examples.
    use_bfloat16: If `True`, use bfloat16 weights instead of the default
      float32; this can save memory but may (rarely) lead to numerical issues.

  Returns:
    A tuple of (input encoder, output encoder, output vocab size used).
  """
  # tokens --> vectors
  def Embedder(vocab_size, embedding_mode):
    if vocab_size is not None:
      embedding = tl.Embedding(vocab_size, d_model, use_bfloat16=use_bfloat16)
    else:
      embedding = tl.Dense(d_model, use_bfloat16=use_bfloat16)
    return [
        embedding,
        tl.Dropout(rate=embedding_dropout,
                   shared_axes=dropout_shared_axes,
                   mode=embedding_mode),
    ]

  # NOTE: Positional encodings are not shared between encoder and decoder.

  # Since encoder doesn't run stepwise, we do not use predict mode there.
  encoder_mode = 'eval' if mode == 'predict' else mode
  in_embedder = Embedder(input_vocab_size, encoder_mode)
  in_encoder = in_embedder + [
      PositionalEncoder(encoder_mode,
                        dropout=embedding_dropout,
                        max_len=max_len,
                        pos_type=pos_type,
                        pos_axial_shape=pos_axial_shape,
                        pos_d_axial_embs=pos_d_axial_embs,
                        pos_start_from_zero_prob=pos_start_from_zero_prob,
                        pos_max_offset_to_add=pos_max_offset_to_add,
                        use_bfloat16=use_bfloat16)
  ]

  # If output_vocab_size is None, we reuse the same embedding matrix, otherwise
  # we initialize one.
  assert input_vocab_size or output_vocab_size
  if output_vocab_size is None:
    out_embedder = in_embedder
  else:
    out_embedder = Embedder(output_vocab_size, mode)

  out_encoder = out_embedder + [
      PositionalEncoder(mode,
                        dropout=embedding_dropout,
                        max_len=max_len,
                        pos_type=pos_type,
                        pos_axial_shape=pos_axial_shape,
                        pos_d_axial_embs=pos_d_axial_embs,
                        pos_start_from_zero_prob=pos_start_from_zero_prob,
                        pos_max_offset_to_add=pos_max_offset_to_add,
                        use_bfloat16=use_bfloat16)
  ]

  # Set this to the value actually used.
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  if input_vocab_size is None:
    in_encoder = tl.AssertFunction('...a->...b', in_encoder)
  else:
    in_encoder = tl.AssertFunction('...->...d', in_encoder)
  out_encoder = tl.AssertFunction('...->...d', out_encoder)

  return in_encoder, out_encoder, output_vocab_size


def ConfigurableTransformerEncoder(vocab_size,
                                   n_classes=10,
                                   d_model=512,
                                   d_ff=2048,
                                   n_layers=6,
                                   n_heads=8,
                                   max_len=2048,
                                   dropout=0.1,
                                   dropout_shared_axes=None,
                                   mode='train',
                                   ff_activation=tl.Relu,
                                   ff_dropout=0.1,
                                   ff_chunk_size=0,
                                   ff_use_sru=0,
                                   ff_sparsity=0,
                                   ff_sparsity_type='1inN',
                                   attention_chunk_size=0,
                                   attention_type=tl.Attention,
                                   pos_type=None,
                                   pos_axial_shape=None,
                                   pos_d_axial_embs=None):
  """Returns a Transformer encoder merged with an N-way categorization head.

  This model performs text categorization:

    - input: rank 2 tensor representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). The tensor
      elements are integers in `range(vocab_size)`, and `0` values mark padding
      positions.

    - output: rank 2 tensor representing a batch of log-probability
      distributions over N categories; shape is (batch_size, `n_classes`).

  Args:
    vocab_size: Input vocabulary size -- each element of the input tensor should
      be an integer in `range(vocab_size)`. These integers typically represent
      token IDs from a vocabulary-based tokenizer.
    n_classes: Final dimension of the output tensors, representing N-way
      classification.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
      block.
    n_layers: Number of encoder blocks. Each block includes attention, dropout,
      residual, feed-forward (`Dense`), and activation layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'train'`, each encoder block will include dropout; else, it will
      pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder block;
      must be an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    attention_chunk_size: int, if > 0 run attention chunked at this size
    attention_type: The attention layer to use for the encoder part.
    pos_type: string, the type of positional embeddings to use.
    pos_axial_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match pos_axial_shape, and values must sum to d_model.

  Returns:
    A Transformer model that maps strings (conveyed via token IDs) to
    probability-like activations over a range of output classes.
  """
  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      PositionalEncoder(
          mode, dropout, max_len, pos_type, pos_axial_shape, pos_d_axial_embs)
  ]

  positional_encoder = tl.AssertFunction('...->...d', positional_encoder)

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes, mode,
                   ff_activation, ff_dropout, ff_chunk_size, ff_use_sru,
                   ff_sparsity, ff_sparsity_type,
                   attention_chunk_size, attention_type)
      for i in range(n_layers)
  ]
  # pylint: enable=g-complex-comprehension

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


def ConfigurableTransformerLM(vocab_size,
                              d_model=512,
                              d_ff=2048,
                              n_layers=6,
                              n_heads=8,
                              max_len=2048,
                              dropout=0.1,
                              dropout_shared_axes=None,
                              mode='train',
                              ff_activation=tl.Relu,
                              ff_dropout=0.1,
                              ff_chunk_size=0,
                              ff_use_sru=0,
                              ff_sparsity=0,
                              ff_sparsity_type='1inN',
                              loss_sparsity_type='mult',
                              loss_sparsity=0,
                              loss_d_lowrank=0,
                              loss_sparsity_prob=None,
                              attention_chunk_size=0,
                              attention_type=tl.CausalAttention,
                              pos_type=None,
                              pos_axial_shape=None,
                              pos_d_axial_embs=None,
                              pos_start_from_zero_prob=1.0,
                              pos_max_offset_to_add=0):
  """Returns a Transformer language model.

  This model performs autoregressive language modeling:

    - input: rank 2 tensor representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). The tensor
      elements are integers in `range(vocab_size)`, and `0` values mark padding
      positions.

    - output: rank 3 tensor representing a batch of log-probability
      distributions for each sequence position over possible token IDs;
      shape is (batch_size, sequence_length, `vocab_size`).

  This model uses only the decoder part of the overall Transformer.

  Args:
    vocab_size: Input vocabulary size -- each element of the input tensor should
      be an integer in `range(vocab_size)`. These integers typically represent
      token IDs from a vocabulary-based tokenizer.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
      block.
    n_layers: Number of encoder blocks. Each block includes attention, dropout,
      residual, feed-forward (`Dense`), and activation layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'predict'`, use fast inference. If `'train'`, each encoder block
      will include dropout; else, it will pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder block;
      must be an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    loss_sparsity_type: string, type of sparsity to used in loss layer. See
      SparseDenseWithOptions for options. None if no sparsity should be used.
    loss_sparsity: int, the sparsity for loss layer (if used)
    loss_d_lowrank: int, the dimensions for intermediate layer (if used)
    loss_sparsity_prob: float, the probability for sparse version of loss to be
      used. If None, only sparse version is used.
    attention_chunk_size: int, if > 0 run attention chunked at this size
    attention_type: The attention layer to use for the decoder part.
    pos_type: string, the type of positional embeddings to use.
    pos_axial_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match pos_axial_shape, and values must sum to d_model.
    pos_start_from_zero_prob: how often to start from 0 during training,
      (if 1.0, we always start from position 0, if less, we randomize).
    pos_max_offset_to_add: maximum offset to add to positions during training
      when randomizing; this offset plus input length must still be less than
      max_len for all training examples.

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  positional_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
      PositionalEncoder(
          mode, dropout, max_len, pos_type, pos_axial_shape, pos_d_axial_embs,
          pos_start_from_zero_prob, pos_max_offset_to_add)
  ]

  # pylint: disable=g-complex-comprehension
  decoder_blocks = [
      DecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes, mode,
                   ff_activation, ff_dropout, ff_chunk_size, ff_use_sru,
                   ff_sparsity, ff_sparsity_type,
                   attention_chunk_size, attention_type)
      for i in range(n_layers)
  ]
  # pylint: enable=g-complex-comprehension

  # Assemble and return the model.
  return tl.Serial(               # tokens (or chunked tuple of tokens)
      tl.ShiftRight(mode=mode),   # toks
      positional_encoder,         # vecs
      decoder_blocks,             # vecs
      tl.LayerNorm(),             # vecs
      tl.SparseDenseWithOptions(  # vecs
          vocab_size, d_input=d_model, sparsity_type=loss_sparsity_type,
          sparsity=loss_sparsity, d_lowrank=loss_d_lowrank,
          prob_sparse=loss_sparsity_prob, mode=mode),
  )


def ConfigurableTransformer(input_vocab_size,
                            output_vocab_size=None,
                            d_model=512,
                            d_ff=2048,
                            n_encoder_layers=6,
                            n_decoder_layers=6,
                            n_heads=8,
                            max_len=2048,
                            dropout=0.1,
                            dropout_shared_axes=None,
                            mode='train',
                            ff_activation=tl.Relu,
                            ff_dropout=0.1,
                            ff_chunk_size=0,
                            ff_use_sru=0,
                            ff_sparsity=0,
                            ff_sparsity_type='1inN',
                            loss_sparsity_type='mult',
                            loss_sparsity=0,
                            loss_d_lowrank=0,
                            loss_sparsity_prob=None,
                            attention_chunk_size=0,
                            encoder_attention_type=tl.Attention,
                            encoder_decoder_attention_type=tl.CausalAttention,
                            pos_type=None,
                            pos_axial_shape=None,
                            pos_d_axial_embs=None,
                            enc_dec_attention_sparsity=0):
  """Returns a full Transformer model.

  This model is an encoder-decoder that performs tokenized string-to-string
  ("source"-to-"target") transduction:

    - inputs (2):

        - source: rank 2 tensor representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length). The
          tensor elements are integers in `range(input_vocab_size)`, and `0`
          values mark padding positions.

        - target: rank 2 tensor representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length). The
          tensor elements are integers in `range(output_vocab_size)`, and `0`
          values mark padding positions.

    - output: rank 3 tensor representing a batch of log-probability
      distributions for each sequence position over possible token IDs;
      shape is (batch_size, sequence_length, `vocab_size`).

  An example use would be to translate (tokenized) sentences from English to
  German.

  Args:
    input_vocab_size: Input vocabulary size -- each element of the input tensor
      should be an integer in `range(vocab_size)`. These integers typically
      represent token IDs from a vocabulary-based tokenizer.
    output_vocab_size: If specified, gives the vocabulary size for the targets;
      if None, then input and target integers (token IDs) are assumed to come
      from the same vocabulary.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
      and decoder block.
    n_encoder_layers: Number of encoder blocks.
    n_decoder_layers: Number of decoder blocks.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within an encoder/decoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'predict'`, use fast inference. If `'train'`, each encoder/decoder
      block will include dropout; else, it will pass all values through
      unaltered.
    ff_activation: Type of activation function at the end of each
      encoder/decoder block; must be an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    loss_sparsity_type: str, type of sparsity to used in loss layer. See
      SparseDenseWithOptions for options. None if no sparsity should be used.
    loss_sparsity: int, the sparsity for loss layer (if used)
    loss_d_lowrank: int, the dimensions for intermediate layer (if used)
    loss_sparsity_prob: float, the probability for sparse version of loss to be
      used. If None, only sparse version is used.
    attention_chunk_size: int, if > 0 run attention chunked at this size
    encoder_attention_type: The attention layer to use for the encoder part.
    encoder_decoder_attention_type: The attention layer to use for the
      encoder-decoder attention.
    pos_type: string, the type of positional embeddings to use.
    pos_axial_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match pos_axial_shape, and values must sum to d_model.
    enc_dec_attention_sparsity: int, if > 0 use this sparsity in attention.

  Returns:
    A Transformer model as a layer that maps from a source-target tokenized
    text pair to activations over a vocab set.
  """
  in_encoder, out_encoder, output_vocab_size = (
      EmbeddingAndPositionalEncodings(
          input_vocab_size,
          d_model,
          mode,
          dropout,
          dropout_shared_axes,
          max_len,
          output_vocab_size=output_vocab_size,
          pos_type=pos_type,
          pos_axial_shape=pos_axial_shape,
          pos_d_axial_embs=pos_d_axial_embs)
  )

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes, mode,
                   ff_activation, ff_dropout, ff_chunk_size, ff_use_sru,
                   ff_sparsity, ff_sparsity_type,
                   attention_chunk_size, encoder_attention_type)
      for i in range(n_encoder_layers)
  ]
  # pylint: enable=g-complex-comprehension

  encoder = tl.Serial(in_encoder, encoder_blocks, tl.LayerNorm())
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  # pylint: disable=g-complex-comprehension
  encoder_decoder_blocks = [
      EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                          mode, ff_activation, ff_dropout, ff_chunk_size,
                          ff_use_sru, ff_sparsity, ff_sparsity_type,
                          attention_chunk_size, encoder_decoder_attention_type,
                          enc_dec_attention_sparsity)
      for i in range(n_decoder_layers)
  ]
  # pylint: enable=g-complex-comprehension

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
      tl.SparseDenseWithOptions(          # vec_d .....
          output_vocab_size, d_input=d_model, sparsity_type=loss_sparsity_type,
          sparsity=loss_sparsity, d_lowrank=loss_d_lowrank,
          prob_sparse=loss_sparsity_prob, mode=mode),
  )


def EncoderBlock(d_model,
                 d_ff,
                 n_heads,
                 dropout,
                 dropout_shared_axes,
                 mode,
                 ff_activation,
                 ff_dropout,
                 ff_chunk_size,
                 ff_use_sru,
                 ff_sparsity,
                 ff_sparsity_type,
                 attention_chunk_size,
                 attention_type,
                 n_attention_layers=1,
                 n_feedforward_layers=1):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the block is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will pass all
      values through unaltered.
    ff_activation: Type of activation function at the end of each block; must be
      an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    attention_chunk_size: int, if > 0 run attention chunked at this size
    attention_type: The attention layer to use.
    n_attention_layers: how many residual causal attention layers should we
      have before the feed-forward block (default: 1, the standard block)
    n_feedforward_layers: how many FFNN layers should we have (default 1).

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  # `n_attention_layers` number of residuals of attention layer + dropout.
  # pylint: disable=g-complex-comprehension
  residual_attentions = [
      tl.Residual(tl.LayerNorm(),
                  ApplyAttentionLayer(attention_type,
                                      d_model,
                                      n_heads,
                                      d_model // n_heads,
                                      d_model // n_heads,
                                      causal=False,
                                      masked=True,
                                      attention_dropout=dropout,
                                      output_dropout=dropout,
                                      attention_chunk_size=attention_chunk_size,
                                      mode=mode),
                  tl.Dropout(rate=dropout,
                             shared_axes=dropout_shared_axes,
                             mode=mode)
                  )
      for _ in range(n_attention_layers)
  ]

  feed_forwards = [
      tl.Residual(
          FeedForwardWithOptions(d_model, d_ff, dropout,
                                 dropout_shared_axes, ff_activation,
                                 ff_dropout, ff_chunk_size, ff_use_sru,
                                 ff_sparsity, True, mode, False,
                                 ff_sparsity_type)
      )
      for _ in range(n_feedforward_layers)
  ]
  # pylint: enable=g-complex-comprehension

  return residual_attentions + feed_forwards


def DecoderBlock(d_model,
                 d_ff,
                 n_heads,
                 dropout,
                 dropout_shared_axes,
                 mode,
                 ff_activation,
                 ff_dropout,
                 ff_chunk_size,
                 ff_use_sru,
                 ff_sparsity,
                 ff_sparsity_type,
                 attention_chunk_size,
                 attention_type,
                 n_attention_layers=1,
                 n_feedforward_layers=1):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will pass all
      values through unaltered.
    ff_activation: Type of activation function at the end of each block; must be
      an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    attention_chunk_size: int, if > 0 run attention chunked at this size
    attention_type: The attention layer to use.
    n_attention_layers: how many residual causal attention layers should we
      have before the feed-forward block (default: 1, the standard block)
    n_feedforward_layers: how many FFNN layers should we have (default 1).

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  # pylint: disable=g-complex-comprehension
  causal_attentions = [ApplyAttentionLayer(
      attention_type,
      d_model,
      n_heads,
      d_model // n_heads,
      d_model // n_heads,
      causal=True,
      masked=False,
      attention_dropout=dropout,
      output_dropout=dropout,
      attention_chunk_size=attention_chunk_size,
      mode=mode) for _ in range(n_attention_layers)]

  residual_attentions = [
      tl.Residual(
          tl.LayerNorm(),
          causal_attentions[i],
          tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
      ) for i in range(n_attention_layers)]

  feed_forwards = [
      tl.Residual(
          FeedForwardWithOptions(d_model, d_ff, dropout,
                                 dropout_shared_axes, ff_activation,
                                 ff_dropout, ff_chunk_size, ff_use_sru,
                                 ff_sparsity, True, mode, False,
                                 ff_sparsity_type)
      )
      for _ in range(n_feedforward_layers)
  ]
  # pylint: enable=g-complex-comprehension

  return residual_attentions + feed_forwards


def EncoderDecoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                        mode, ff_activation, ff_dropout, ff_chunk_size,
                        ff_use_sru, ff_sparsity, ff_sparsity_type,
                        attention_chunk_size, attention_type,
                        enc_dec_attention_sparsity=0):
  """Returns a list of layers implementing a Transformer encoder-decoder block.

  The input is a triple (decoder_activations, mask, encoder_activiations) where
  the mask is created from the original input token IDs to prevent attending to
  the padding part of the encoder.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will pass all
      values through unaltered.
    ff_activation: Type of activation function at the end of each block; must be
      an activation-type subclass of `Layer`.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int or pair of ints; if > 0, we use this many SRU layers
      in addition to the feed-forward block (second int specifies sru size)
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
     ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    attention_chunk_size: int, if > 0 run attention chunked at this size
    attention_type: The attention layer to use.
    enc_dec_attention_sparsity: Sparsity to use in encoder-decoder attention.

  Returns:
    A list of layers which maps triples (decoder_activations, mask,
    encoder_activations) to triples of the same sort.
  """

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  # TODO(afrozm): This layer isn't configurable because: We currently don't have
  # any alternative for it (LSH cannot do it fundamentally, that's why we have
  # NoEncDec models, and local attention doesn't make sense in the general
  # setting where we don't know what in input is local to what in output;
  # some variants of FAVOR can do it, so maybe in the future,
  # but we don't have them yet).
  if isinstance(enc_dec_attention_sparsity, tuple):
    q_sparsity, result_sparsity = enc_dec_attention_sparsity
  elif enc_dec_attention_sparsity > 0:
    q_sparsity = enc_dec_attention_sparsity
    result_sparsity = 'noop'  # We simply skip Dense layer after attention.
  else:
    q_sparsity = None
    result_sparsity = None
  attention_qkv = tl.AttentionQKV(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode,
      cache_KV_in_predict=True,
      q_sparsity=q_sparsity, result_sparsity=result_sparsity)

  causal_attention = ApplyAttentionLayer(
      attention_type,
      d_model,
      n_heads,
      d_model // n_heads,
      d_model // n_heads,
      causal=True,
      masked=True,
      attention_dropout=dropout,
      output_dropout=dropout,
      attention_chunk_size=attention_chunk_size,
      mode=mode)

  feed_forward = FeedForwardWithOptions(d_model, d_ff, dropout,
                                        dropout_shared_axes, ff_activation,
                                        ff_dropout, ff_chunk_size, ff_use_sru,
                                        ff_sparsity, True, mode, False,
                                        ff_sparsity_type)

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
