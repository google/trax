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
"""The Terraformer model -- highly configurable version for research.

Implements the Terraformer model, introduced in the paper ...
"""

import functools
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.models.reformer import reformer
from trax.models.research import configurable_transformer as ct
from trax.models.research import transformer2 as t2

# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def ConfigurableTerraformer(input_vocab_size,
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
                            pos_type='fixed-base',
                            pos_axial_shape=(),
                            pos_d_axial_embs=None,
                            pos_start_from_zero_prob=1.0,
                            pos_max_offset_to_add=0,
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
                            forget_dense=True,
                            n_decoder_attention_layers=2,
                            use_bfloat16=False,
                            reversible_encoder=False,
                            use_two_swaps_per_encoder_block=True,
                            center_layernorm=True,
                            half_before_layer=None,
                            double_after_layer=None,
                            mode='train'):
  """Returns a highly configurable Terraformer encoder-decoder model.

  This model maps paired text sequences (source and target) to float-valued
  losses. If ``input_vocab_size`` is not ``None``, the layer takes
  two input sequences:

    - inputs (2):

        - source: 2-D int array representing a batch of text strings via token
          IDs plus padding markers; shape is `(batch_size, sequence_length)`,
          where sequence_length <= ``max_len``. Array elements are in
          ``range(input_vocab_size)``, and 0 values mark padding positions.

        - target: 2-D int array representing a batch of text strings via token
          IDs plus padding markers; shape is `(batch_size, sequence_length)`,
          where sequence_length <= ``max_len``. Array elements are in
          ``range(output_vocab_size)``, and 0 values mark padding positions.

    - output: 1-D float array of losses; shape is `(batch_size)`.

  If ``input_vocab_size`` is ``None``, the layer takes three input sequences:

    - inputs (3):

        - source: 3-D float array representing a batch of already-embedded text
          strings; shape is `(batch_size, sequence_length, d_model)`, where
          sequence_length <= ``max_len``.

        - mask: 2-D int array representing active versus masked positions; 0
          values mark masked (padding) positions.

        - target: 2-D int array representing a batch of text strings via token
          IDs plus padding markers; shape is `(batch_size, sequence_length)`,
          where sequence_length <= ``max_len``. Array elements are in
          ``range(output_vocab_size)``, and 0 values mark padding positions.

    - output: 1-D float array of losses; shape is `(batch_size)`.

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
    d_attention_key: Depth of key vectors in each attention head.
    d_attention_value: Depth of value vectors in each attention head.
    n_encoder_layers: Number of encoder blocks.
    n_decoder_layers: Number of decoder blocks.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder/decoder blocks. The same rate is
        also used for attention dropout in encoder/decoder blocks.
    max_len: Maximum symbol length for positional encoding.
    encoder_attention_type: Type of attention to use in the encoder; must be
        an attention-type subclass of :py:class:`trax.layers.Layer`.
    encoder_decoder_attention_type: Type of attention to use in the decoder;
        must be an attention-type subclass of :py:class:`trax.layers.Layer`.
    pos_type: String indicating the type of positional embeddings to use.
    pos_axial_shape: Shape (tuple of ints) to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    pos_d_axial_embs: Tuple of ints specifying the depth of position embedding
        for each axis. Tuple length must match ``pos_axial_shape``, and values
        must sum to ``d_model``.
    pos_start_from_zero_prob: Stochastic rate (probability) for starting
        positional encoding at position 0 during training. If 1.0, always start
        from position 0; if < 1.0, the non-zero starts will be uniformly
        distributed up to ``pos_max_offset_to_add``.
    pos_max_offset_to_add: Maximum offset to add to positions during training
        when randomizing. This offset plus input length must be less than
        ``max_len`` for all training examples.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`trax.layers.Layer`.
    ff_use_sru: If > 0, use this number of SRU layers in place of feedforward
        layers.
    ff_chunk_size: If > 0, chunk each feedforward layer into chunks of this
        size.
    ff_dropout: Stochastic rate (probability) for dropping an activation value
        at feedforward nonlinearities.
    ff_sparsity: If > 0, use sparse feedforward blocks with this level of
        sparsity.
    loss_sparsity_type: String indicating the type of sparsity to used in loss
        layer; see :py:class:`SparseDenseWithOptions` for options. If ``None``,
        use no sparsity.
    loss_sparsity: If > 0, use this level of sparsity in the loss layer.
    loss_d_lowrank: If > 0, use a (low-rank) intermediate layer, with this
        dimension, in the loss.
    loss_sparsity_prob: Stochastic rate (probability) for using the sparse
        version of the loss. If ``None``, use the sparse version exclusively.
    attention_chunk_size: If > 0, compute attention using chunks of this size.
    n_layers_forget: How often to have a forgetting block between layers.
    forget_dense: If True, use :py:class:`Dense` instances as forget layers;
        else use no-ops.
    n_decoder_attention_layers: Number of attention layers in a decoder block.
    use_bfloat16: If True, use bfloat16 for weights; else use float32.
    reversible_encoder: If True, make the encoder be reversible.
    use_two_swaps_per_encoder_block: If True, ensure that there is a an even
        number of swaps across the encoder.
    center_layernorm: If True, use centering in :py:class:`LayerNorm` (the
        default); else omit centering (which is known as RMS normalization).
    half_before_layer: If not None, specifies an n'th layer such that all
        layers before the n'th use half the normal values for ``d_model`` and
        ``d_ff``.
    double_after_layer: If not None, specifies an n'th layer such that all
        layers after the n'th use double the normal values for ``d_model`` and
        ``d_ff``.
    mode: If ``'train'``, include dropout in each encoder/decoder block; else
        dropout layers have no effect.

  Returns:
    A Terraformer encoder-decoder as a layer that maps from target and source
    text sequences to a scalar loss.
  """
  if mode == 'predict':
    portal_mask = _PortalInput()
  else:
    portal_mask = None

  # Set default dimensions for attention head key and value sizes.
  if (d_model / 2) % n_heads != 0:
    raise ValueError(f'n_heads ({n_heads}) must divide d_model/2 ({d_model/2})')
  if d_attention_key is None:
    d_attention_key = d_model // n_heads
  if d_attention_value is None:
    d_attention_value = d_model // n_heads

  # Set values of d_model, d_ff and d_qkv for the first stage.
  d_model1, d_ff1 = d_model, d_ff
  d_attention_key1, d_attention_value1 = d_attention_key, d_attention_value
  if half_before_layer:
    d_model1, d_ff1 = d_model / 2, d_ff / 2
    d_attention_key1 = d_attention_key / 2
    d_attention_value1 = d_attention_value / 2

  # Set values of d_model, d_ff and d_qkv for the final stage.
  d_model2, d_ff2 = d_model, d_ff
  d_attention_key2, d_attention_value2 = d_attention_key, d_attention_value
  if double_after_layer:
    d_model2, d_ff2 = d_model * 2, d_ff * 2
    d_attention_key2 = d_attention_key * 2
    d_attention_value2 = d_attention_value * 2

  # Vector embeddings.
  in_encoder, out_encoder, output_vocab_size = (
      ct.EmbeddingAndPositionalEncodings(
          input_vocab_size,
          d_model1,
          mode,
          dropout,
          [-2],  # dropout_shared_axes
          max_len,
          output_vocab_size=output_vocab_size,
          pos_type=pos_type,
          pos_axial_shape=pos_axial_shape,
          pos_d_axial_embs=pos_d_axial_embs,
          pos_start_from_zero_prob=pos_start_from_zero_prob,
          pos_max_offset_to_add=pos_max_offset_to_add,
          use_bfloat16=use_bfloat16)
  )

  def _EncoderBlock():
    return reformer.EncoderBlock(
        d_model1,
        d_ff1,
        n_heads,
        encoder_attention_type,
        dropout=dropout,
        ff_activation=ff_activation,
        ff_dropout=ff_dropout,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        ff_sparsity=ff_sparsity,
        attention_chunk_size=attention_chunk_size,
        center_layernorm=center_layernorm,
        use_bfloat16=use_bfloat16,
        use_two_swaps_per_block=use_two_swaps_per_encoder_block,
        mode=mode)

  def _Encoder():  # vec_e mask_e tok_e tok_d tok_d
    layers = [
        tl.ReversibleSelect([0, 0]),
        _ReversibleSerialForget(
            [_EncoderBlock() for _ in range(n_encoder_layers)],
            d_model1,
            n_layers_forget,
            forget_dense)
    ]
    if not reversible_encoder:
      layers += [
          _XYAvg(),
          tl.Dense(d_model1, use_bfloat16=use_bfloat16),
          tl.LayerNorm(),
      ]
    if mode == 'predict':
      return tl.Cache(tl.Serial(layers))
    else:
      return tl.Serial(layers)

  if mode == 'predict':
    # TODO(jaszczur): Remove temporary fix of Terraformer padding in predict.
    # In predict mode Terraformer needs masking for merged encoder-decoder
    # sequence. This monkey patches the layer with a mask to neccessary places.
    # This shouldn't be a permanent solution - mask should be passed through
    # the stack and all the layers.
    tl.attention.DotProductCausalAttention.monkey_patched_mask = (
        lambda x: portal_mask)
    tl.research.sparsity._RememberPad.monkey_patched_mask = (  # pylint: disable=protected-access
        lambda x: portal_mask)
    originalScanSRUCell = tl.rnn.ScanSRUCell
    tl.rnn.ScanSRUCell = functools.partial(tl.rnn.ScanSRUCell,
                                           monkey_patched_mask=portal_mask)

  decoder_blocks = []

  if isinstance(encoder_decoder_attention_type, (tuple, list)):
    assert n_decoder_layers % len(encoder_decoder_attention_type) == 0
  else:
    encoder_decoder_attention_type = [encoder_decoder_attention_type]
  for layer_idx in range(n_decoder_layers):
    layer_attention_type = encoder_decoder_attention_type[
        layer_idx % len(encoder_decoder_attention_type)]
    # Grow d_model, d_ff, and d_qkv if requested.
    d_m, d_f, d_k, d_v = d_model1, d_ff1, d_attention_key1, d_attention_value1
    if half_before_layer and layer_idx >= half_before_layer:
      d_m, d_f, d_k, d_v = d_model, d_ff, d_attention_key, d_attention_value
    if double_after_layer and layer_idx > double_after_layer:
      d_m, d_f, d_k, d_v = d_model2, d_ff2, d_attention_key2, d_attention_value2
    decoder_block = reformer.DecoderBlock(
        d_m, d_f, d_k, d_v, n_heads,
        attention_type=layer_attention_type,
        dropout=dropout,
        ff_activation=ff_activation,
        ff_dropout=ff_dropout,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        ff_sparsity=ff_sparsity,
        attention_chunk_size=attention_chunk_size,
        n_attention_layers=n_decoder_attention_layers,
        center_layernorm=center_layernorm,
        use_bfloat16=use_bfloat16,
        mode=mode)
    decoder_blocks.append(decoder_block)
    if half_before_layer and layer_idx == half_before_layer - 1:
      decoder_blocks.append(tl.ReversibleConcatenatePair())
    if double_after_layer and layer_idx == double_after_layer:
      decoder_blocks.append(tl.ReversibleConcatenatePair())

  if mode == 'predict':
    # After initializing the decoder we can revert to original state of
    # previously monkey-patched classes/functions.
    tl.attention.DotProductCausalAttention.monkey_patched_mask = (
        lambda x: None)
    tl.research.sparsity._RememberPad.monkey_patched_mask = (lambda x: None)  # pylint: disable=protected-access
    tl.rnn.ScanSRUCell = originalScanSRUCell

  def _Loss():
    return tl.SparseDenseWithOptions(
        output_vocab_size,
        d_input=d_model2,
        sparsity_type=loss_sparsity_type,
        sparsity=loss_sparsity,
        d_lowrank=loss_d_lowrank,
        prob_sparse=loss_sparsity_prob,
        use_bfloat16=use_bfloat16,
        mode=mode)

  def _enc_dec_concat():
    """Layers to merge encoder and decoder."""
    if reversible_encoder:
      return [
          tl.ReversibleSelect([0, 1, 4, 2, 3]),  # v_e v_d mask_e tok_e tok_d
          t2.ConcatWithPadding2(mode=mode),      # v_ed v_ed tok_e tok_d
      ]
    else:
      return [
          tl.ReversibleSelect([0, 3, 1, 2]),     # v_e v_d mask_e tok_e tok_d
          t2.ConcatWithPadding(mode=mode),       # v_ed tok_e tok_d
          tl.ReversibleSelect([0, 0]),           # v_ed v_ed tok_e tok_d
      ]

  def _inp_layers():
    if input_vocab_size is not None:
      return tl.AssertFunction(
          'bl,br->bld,bl,bl,br',  # b: batch, l/r: enc/dec length, d: vec depth
          tl.Serial(  # tok_e tok_d
              tl.Select([0, 0, 0, 1]),
              tl.Parallel(in_encoder, [tl.PaddingMask(),
                                       _RemoveAxes12()])
          ))  # vec_e mask_e tok_e tok_d
    else:
      # Input in this case is vec_e, mask_e, tok_d. Where all downstream
      # operations expect tok_e, we give it instead mask_e, expecting that
      # downstream ops only are looking for padding/not padding.
      return tl.AssertFunction(
          'blf,bl,br->bld,bl,bl,br',  # f: in-feature depth, d: out-vector depth
          tl.Serial(  # vec_e mask_e tok_d
              tl.Select([0, 1, 1, 2]),
              tl.Parallel(in_encoder, [], _AsTokenIDs())
          ))  # vec_e mask_e tok_e tok_d

  # Assemble and return the model.
  return tl.Serial(
      _inp_layers(),               # vec_e mask_e tok_e tok_d
      tl.Parallel([], portal_mask),

      tl.Select([0, 1, 2, 3, 3]),  # Copy decoder tokens for use in loss.

      # Embed in and out tokens; done together as weights may be shared.
      tl.Parallel([], [], [], [tl.ShiftRight(mode=mode),
                               out_encoder]),  # vec_e mask_e tok_e vec_d tok_d

      # Encode; then concat encoder and decoder, given encoder mask.
      _Encoder(),                             # vec_e mask_e tok_e vec_d tok_d
      _enc_dec_concat(),

      # Run decoder blocks.
      _ReversibleSerialForget(decoder_blocks, d_model2, n_layers_forget,
                              forget_dense),  # vec_ed1 vec_ed2 tok_e tok_d
      _XYAvg(),                               # vec_ed tok_e tok_d
      tl.LayerNorm(),                         # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector,
      # then compute loss.
      tl.Select([0, 1, 2, 2]),                        # vec_ed tok_e tok_d tok_d
      t2.StripFromConcatenateWithPadding(mode=mode),  # vec_d tok_d
      _Loss(),  # vec_d tok_d
  )


def _InsertAxes12():
  """Returns a layer that inserts two internal size-1 axes into an array."""
  return tl.Fn('InsertAxes12',
               lambda x: jnp.reshape(x, (x.shape[0], 1, 1, x.shape[1])))


def _RemoveAxes12():
  """Returns a layer that removes two internal size-1 axes from an array."""
  return tl.Fn('RemoveAxes12', lambda x: jnp.squeeze(x, (1, 2)))


def _AsTokenIDs():
  """Returns a layer that makes mask values look like token ID ints."""
  return tl.Fn('AsTokenIDs', lambda x: x.astype(jnp.int32))


def _XYAvg():
  """Returns a layer that computes the element-wise average of two arrays."""
  return tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0)


def _ReversibleSerialForget(layers, d_model, n_layers, forget_dense=True):
  """ReversibleSerial but with a forgetting block every n_layers."""
  if not n_layers or len(layers) <= n_layers + 1:
    return tl.ReversibleSerial(layers)
  layers1, layers2 = layers[:n_layers], layers[n_layers:]

  if forget_dense:
    forgetting_layer = tl.Serial(
        _XYAvg(),
        tl.Dense(d_model),
        tl.Dup(),
    )
  else:
    forgetting_layer = tl.Select([0, 1])

  return tl.Serial(
      tl.ReversibleSerial(layers1),
      forgetting_layer,
      _ReversibleSerialForget(layers2, d_model, n_layers, forget_dense)
  )


def _ConvertToNaNsOnAnyZero():
  def _convert_to_nans(x, y):
    # if all values in y are non-zeros, return x; otherwise return 0s
    return jnp.where(jnp.all(y, keepdims=False), x, x/0.), y
  return tl.Fn('ConvertToNaNsOnAnyZero', _convert_to_nans, n_out=2)


class _PortalInput(tl.Layer):
  """Portal input for monkey-patching of mask in predict mode."""

  def __init__(self):
    super().__init__(name='_PortalInput', n_out=1, n_in=1)
    self._portal_output = _PortalOutput(self)

  def forward(self, x):
    if isinstance(x, (list, tuple)):
      x = x[0]
    self.state = (x,)
    return x

  def init_weights_and_state(self, input_signature):
    """Initializes this layer's weights."""
    if isinstance(input_signature, (list, tuple)):
      input_signature = input_signature[0]
    self.state = (jnp.zeros(input_signature.shape),)

  def get_value(self):
    return self.state[0]

  def get_layer(self):
    return self._portal_output


class _PortalOutput(tl.Layer):
  """Portal input for monkey-patching of mask in predict mode."""

  def __init__(self, portal_input):
    super().__init__(name='_PortalOutput', n_out=1, n_in=0)
    self._portal_input = portal_input

  def forward(self, x):
    return self._portal_input.get_value()

  def get_value(self):
    return self._portal_input.get_value()
