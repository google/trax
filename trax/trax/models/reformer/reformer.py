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
from trax.layers.combinators import (  # pylint: disable=g-multiple-import
    _split_rngs, _inputs_from_stack, _outputs_onto_stack)

# TODO(afrozm): Move pvt fns here.
from trax.models.transformer import (  # pylint: disable=g-multiple-import
    _ConcatWithPadding, _MaskOfRightShiftedArray,
    _StripFromConcatenateWithPadding)

# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def FeedForward(d_model, d_ff, dropout, activation, act_dropout, mode):
  """Feed-forward block with layer normalization at start."""
  if act_dropout is None:
    act_dropout = dropout
  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      tl.Dropout(rate=act_dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
      activation(),
      tl.Dense(d_model),
      tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),  # pylint: disable=no-value-for-parameter
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


class ReversibleHalfResidualV2(tl.ReversibleLayer):
  """Half of a RevNet-style residual that optionally performs attention.

  When attention_layer is None, this layer has the signature ::

      [accumulator, *context] -> [accumulator + f(context), *context]

  The attention_layer must be an instance of EfficientAttentionBase or one of
  its subclasses (see efficient_attention.py), or None.

  Attention is special-cased for the following two reasons:

  - LSH attention needs to save bucket assignments from the forward pass to the
    backward pass, for training stability. This requires special-casing it.
  - We can call attention_layer.forward_and_or_backward to compute its output
    (needed for inverting a reversible residual layer) while simultaneously
    performing the backward pass. Sharing computation between these two
    operations improves training speed.
  """

  def __init__(self, *residual_layers, attention_layer=None):
    super(ReversibleHalfResidualV2, self).__init__()

    self.compute_residual = tl.Serial(*residual_layers)
    self.attention_layer = attention_layer

    if self.attention_layer is None:
      self._sublayers = (self.compute_residual,)
    else:
      assert hasattr(attention_layer, 'forward_and_or_backward')
      self._sublayers = (self.compute_residual, self.attention_layer)

    running_max = 0
    running_total = 0
    for layer in self._sublayers:
      running_total += layer.n_in
      running_max = max(running_max, running_total)
      running_total -= layer.n_out
    self._n_in = self._n_out = running_max + 1

  def forward(self, xs):
    rngs = _split_rngs(self.rng, len(self.sublayers))
    accumulator, *context = xs
    stack = context = tuple(context)
    new_state = []
    for layer, w, s, rng in zip(self.sublayers, self.weights, self.state, rngs):
      inputs = _inputs_from_stack(layer, stack)
      outputs, s = layer.pure_fn(inputs, w, s, rng)
      stack = _outputs_onto_stack(layer, outputs, stack)
      new_state.append(s)
    residual = stack[0] if isinstance(stack, (tuple, list)) else stack

    output = accumulator + residual
    stack = (output,) + context
    self.state = new_state
    return stack

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    raise NotImplementedError('Only reverse_and_grad is actually used.')

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       rng=None):
    rngs = _split_rngs(rng, len(self.sublayers))

    accumulator_output, *context = output
    context = tuple(context)
    accumulator_output_ct, *context_ct = ct
    context_ct = tuple(context_ct)

    # Forward pass through self.compute_residual. Outputs that will not receive
    # a gradient signal from subsequent layers are moved to aux.
    def call_compute_residual(x, weights):
      res, _ = self.compute_residual.pure_fn(
          x, weights=weights, state=state[0], rng=rngs[0])
      if not isinstance(res, (tuple, list)):
        return res, None
      else:
        n_differentiable = 1
        if self.attention_layer is not None:
          n_differentiable = min(len(res), self.attention_layer.n_in)
        return res[:n_differentiable], res[n_differentiable:]

    stack = context
    inputs = _inputs_from_stack(self.compute_residual, stack)
    outputs, compute_residual_vjpfun, outputs_aux = fastmath.vjp(
        call_compute_residual, inputs, weights[0], has_aux=True)
    if outputs_aux is not None:
      n_differentiable_outputs = len(outputs)
      outputs = outputs + outputs_aux
    stack = _outputs_onto_stack(self.compute_residual, outputs, stack)

    stack_ct = accumulator_output_ct
    if self.attention_layer is None:
      residual = stack[0] if isinstance(stack, (tuple, list)) else stack
    else:
      inputs = _inputs_from_stack(self.attention_layer, stack)
      (residual, _, attn_inputs_ct, attn_weights_ct
      ) = self.attention_layer.forward_and_or_backward(
          inputs, weights[1], new_state[1], rngs[1],
          output_grad=accumulator_output_ct,
          compute_output=True, update_state=False)
      stack_ct = _outputs_onto_stack(
          self.attention_layer, attn_inputs_ct, stack_ct,
          self.attention_layer.n_out, self.attention_layer.n_in)

    compute_residual_ct = _inputs_from_stack(
        self.compute_residual, stack_ct, self.compute_residual.n_out)
    if outputs_aux is not None:
      if not isinstance(compute_residual_ct, (tuple, list)):
        compute_residual_ct = (compute_residual_ct,)
      compute_residual_ct = compute_residual_ct[:n_differentiable_outputs]
      assert len(compute_residual_ct) == n_differentiable_outputs
    (compute_residual_inputs_ct, compute_residual_weights_ct
    ) = compute_residual_vjpfun(compute_residual_ct)
    stack_ct = _outputs_onto_stack(
        self.compute_residual, compute_residual_inputs_ct, stack_ct,
        self.compute_residual.n_out, self.compute_residual.n_in)
    if not isinstance(stack_ct, (tuple, list)):
      stack_ct = (stack_ct,)
    stack_ct = (accumulator_output_ct,) + fastmath.nested_map_multiarg(
        lambda x, y: x+y, context_ct[:len(stack_ct)], stack_ct
        ) + context_ct[len(stack_ct):]

    reconstructed_x = accumulator_output - residual
    stack = (reconstructed_x,) + context
    if self.attention_layer is None:
      weights_ct = (compute_residual_weights_ct,)
    else:
      weights_ct = (compute_residual_weights_ct, attn_weights_ct)
    return stack, (stack_ct, weights_ct)

  # pylint: disable=protected-access
  def init_weights_and_state(self, input_signature):
    stack = input_signature[1:]
    if len(stack) == 1:
      stack = stack[0]

    inputs = _inputs_from_stack(self.compute_residual, stack)
    weights, state = self.compute_residual.init(inputs)
    outputs, _ = self.compute_residual._forward_abstract(inputs)
    stack = _outputs_onto_stack(self.compute_residual, outputs, stack)

    if self.attention_layer is None:
      self.state = (state,)
      self.weights = (weights,)
    else:
      inputs = _inputs_from_stack(self.attention_layer, stack)
      attn_weights, attn_state = self.attention_layer.init(inputs)
      self.state = (state, attn_state)
      self.weights = (weights, attn_weights)
  # pylint: enable=protected-access


def DecoderBlock(d_model, d_ff, d_attention_key, d_attention_value,
                 n_heads, attention_type,
                 dropout, ff_activation, ff_use_sru, ff_chunk_size, mode):
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
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  attention = attention_type(
      n_heads=n_heads, d_qk=d_attention_key, d_v=d_attention_value,
      causal=True, output_dropout=dropout, mode=mode)
  attention_half_residual = ReversibleHalfResidualV2(
      tl.LayerNorm(),
      attention_layer=attention,
  )

  if ff_use_sru:
    feed_forward = [tl.SRU(d_model) for _ in range(ff_use_sru)]
  else:
    feed_forward = [ChunkedFeedForward(d_model, d_ff, dropout, ff_activation,
                                       dropout, ff_chunk_size, mode)]

  return [
      attention_half_residual,
      tl.ReversibleSwap(),
      ReversibleHalfResidualV2(feed_forward),
      tl.ReversibleSwap(),
  ]


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
    mode: str: 'train', 'eval', or 'predict'

  Returns:
    the layer.
  """
  d_emb = d_model
  if not axial_pos_shape:
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, mode=mode)
  elif axial_pos_shape == 'fixed-base':  # TODO(lukaszkaiser): remove this HACK
    positional_encoding = tl.FixedBasePositionalEncoding(mode=mode)
    d_emb //= 2
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

  positional_embedder = [
      tl.Embedding(vocab_size, d_emb),
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
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
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
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
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
      tl.ShiftRight(n_shifts=shorten_factor - 1),
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
                 ff_dropout, mode):
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
  attention_half_residual = ReversibleHalfResidualV2(
      tl.LayerNorm(),
      attention_layer=attention,
  )

  feed_forward = FeedForward(
      d_model, d_ff, dropout, ff_activation, ff_dropout, mode)

  return [
      attention_half_residual,
      tl.ReversibleSwap(),
      ReversibleHalfResidualV2(feed_forward),
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
  enc_dec_attention_half_residual = ReversibleHalfResidualV2(
      tl.LayerNorm(),
      attention_layer=enc_dec_attention,
  )

  causal_attention = tl.SelfAttention(
      n_heads=n_heads, d_qk=d_model//n_heads, d_v=d_model//n_heads,
      causal=True,
      attention_dropout=dropout, output_dropout=dropout,
      mode=mode)
  causal_attention_half_residual = ReversibleHalfResidualV2(
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
      ReversibleHalfResidualV2(feed_forward),
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
  if fastmath.backend_name() == 'jax':
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

  # TODO(kitaev): The regular trax Transformer shares vocab embeddings and
  # position embeddings between the encoder and decoder if output_vocab_size is
  # None. This isn't supported here because (a) Trax shares weights by sharing
  # layer instances, but we need two separate instances to have mode == 'eval'
  # for the encoder but mode == 'predict' for the decoder; and (b) tl.Cache does
  # not work if its sublayers participate in any weight sharing.

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
          ff_dropout, mode)
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


def ReformerNoEncDecAttention(input_vocab_size,
                              output_vocab_size=None,
                              d_model=512,
                              d_ff=2048,
                              d_attention_key=64,
                              d_attention_value=64,
                              n_encoder_layers=6,
                              n_decoder_layers=6,
                              n_heads=8,
                              dropout=0.1,
                              max_len=2048,
                              encoder_attention_type=tl.SelfAttention,
                              encoder_decoder_attention_type=tl.SelfAttention,
                              axial_pos_shape=(),
                              d_axial_pos_embs=None,
                              ff_activation=tl.Relu,
                              ff_use_sru=0,
                              ff_chunk_size=0,
                              ff_dropout=None,
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
  if fastmath.backend_name() == 'jax':
    jax.api._check_inexact_input_vjp = lambda x: None  # pylint: disable=protected-access

  def PositionalEncoder(vocab_size, mode):  # tokens --> vectors
    if not axial_pos_shape:
      positional_encoding = tl.PositionalEncoding(
          max_len=max_len, dropout=dropout, mode=mode)
    else:
      assert d_axial_pos_embs is not None
      positional_encoding = tl.AxialPositionalEncoding(
          shape=axial_pos_shape, d_embs=d_axial_pos_embs,
          dropout_broadcast_dims=tuple(range(1, len(axial_pos_shape) + 1)),
          dropout=dropout, mode=mode)

    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
        positional_encoding,
    ]

  # TODO(kitaev): The regular trax Transformer shares vocab embeddings and
  # position embeddings between the encoder and decoder if output_vocab_size is
  # None. This isn't supported here because (a) Trax shares weights by sharing
  # layer instances, but we need two separate instances to have mode == 'eval'
  # for the encoder but mode == 'predict' for the decoder; and (b) tl.Cache does
  # not work if its sublayers participate in any weight sharing.

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
          d_model, d_ff, n_heads, encoder_attention_type, dropout,
          ff_activation, ff_dropout, mode)
      for _ in range(n_encoder_layers)]
  # pylint: enable=g-complex-comprehension

  encoder = tl.Serial([                # tok_e mask_e tok_e tok_d tok_d
      in_encoder,                      # vec_e mask_e tok_e tok_d tok_d
      tl.Dup(),                        # vec_e1 vec_e2 mask_e tok_e tok_d tok_d
      tl.ReversibleSerial(encoder_blocks),
      tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
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
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        mode=mode)
    decoder_blocks.append(decoder_block)

  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 0, 1, 1]),                  # tok_e tok_e tok_d tok_d
      tl.Branch([], [tl.PaddingMask(),
                     tl.Fn('Squeeze',
                           lambda x: jnp.squeeze(x, (1, 2)), n_out=1)]),
      #                                         # tok_e mask_e tok_e tok_d tok_d

      # Encode.
      encoder,                                  # vec_e mask_e tok_e tok_d tok_d

      # Decode.
      tl.Select([3, 0, 1, 2]),                 #  tok_d vec_e mask_e tok_e tok_d
      tl.ShiftRight(mode=mode),                # stok_d vec_e mask_e tok_e tok_d
      tl.Branch(
          [],
          _MaskOfRightShiftedArray()
      ),                                # stok_d mask_d vec_e mask_e tok_e tok_d
      out_encoder,                      # svec_d mask_d vec_e mask_e tok_e tok_d

      # Concat encoder and decoder, given their masks.
      tl.Select([2, 0, 3, 1]),          # svec_d mask_d vec_e mask_e tok_e tok_d
      _ConcatWithPadding(),                        # vec_ed tok_e tok_d

      # Run (encoder and) decoder blocks.
      tl.Dup(),                                    # vec_ed1 vec_ed2 tok_e tok_d
      tl.ReversibleSerial(decoder_blocks),         # vec_ed1 vec_ed2 tok_e tok_d
      tl.Fn('XYAvg',
            lambda x, y: (x + y) / 2.0),           # vec_ed tok_e tok_d
      tl.LayerNorm(),                              # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector.
      tl.Select([0, 1, 2, 2]),                     # vec_ed tok_e tok_d tok_d
      _StripFromConcatenateWithPadding(),          # vec_d tok_d

      # Map to output vocab.
      tl.Dense(output_vocab_size),                 # vec_d tok_d
      tl.LogSoftmax(),                             # vec_d tok_d
  )
