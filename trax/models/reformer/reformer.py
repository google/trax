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

from trax import layers as tl
from trax.layers import base
from trax.layers.combinators import (  # pylint: disable=g-multiple-import
    _pop_rng_and_split, _inputs_from_stack, _outputs_onto_stack)
from trax.math import numpy as np
from trax.math import random


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


class Map(tl.Layer):
  """Combinator for applying a layer to a list or tuple."""

  def __init__(self, layer, n_sections=1, check_shapes=True):
    """Initialize the combinator.

    Args:
      layer: a layer to apply to each element.
      n_sections: how many sections to map to (default: 1).
      check_shapes: whether to check that shapes are identical (default: true).

    Returns:
      A new layer representing mapping layer to all elements of the input.
    """
    super(Map, self).__init__(n_in=n_sections, n_out=n_sections)
    if layer is None or isinstance(layer, (list, tuple)):
      layer = tl.Serial(layer)
    self._layer = layer
    # Generally a Map should be applied to lists where all elements have
    # the same shape -- because self._layer will only be initialized once
    # and it could have different parameters for different shapes. But there
    # are valid cases -- e.g., when self._layer has no parameters -- where we
    # can apply Map to different shapes -- set check_shapes=False in such cases.
    self._check_shapes = check_shapes
    self._n_sections = n_sections

  def forward_with_state(self, inputs, weights=(), state=(), **kwargs):
    if self._n_sections == 1:
      results = self._layer(inputs, weights=weights, state=state, **kwargs)
    else:
      rngs = _pop_rng_and_split(kwargs, len(inputs))
      results = [self._layer(x, weights=weights, state=state, rng=r, **kwargs)
                 for x, r in zip(inputs, rngs)]
      results = tuple(results)
    # TODO(kitaev): think about how to merge state across copies in the map.
    return results, self._layer.state

  def new_weights_and_state(self, input_signature):
    if self._n_sections == 1:
      return self._layer.init(input_signature)
    first_shape = input_signature[0].shape
    if self._check_shapes:
      for shape_dtype in input_signature:
        if shape_dtype.shape != first_shape:
          raise ValueError('Map layer can only be applied to list of elements '
                           'with the same shapes. This shape %s vs first shape '
                           '%s.' % (str(shape_dtype.shape), str(first_shape)))
    return self._layer.init(input_signature[0])

  @tl.Layer.weights.setter
  def weights(self, weights):
    self._weights = self._layer.weights = weights

  @tl.Layer.state.setter
  def state(self, state):
    self._state = self._layer.state = state

  def _set_input_signature_recursive(self, input_signature):
    self._input_signature = input_signature
    self._layer._set_input_signature_recursive(input_signature)  # pylint: disable=protected-access


class BroadcastedDropout(tl.Layer):
  """Layer constructor function for a broadcasted dropout layer."""

  def __init__(self, rate=0.0, mode='train', broadcast_dims=(-2,)):
    super(BroadcastedDropout, self).__init__()
    self._rate = rate
    if self._rate >= 1.0:
      raise ValueError('Dropout rate (%f) must be lower than 1.' % rate)
    self._broadcast_dims = broadcast_dims
    self._mode = mode

  def forward_with_state(self, x, weights, state, rng):
    """Dropout, with broadcasting to save memory."""
    del weights
    if rng is None:
      raise ValueError('BroadcastedDropout requires rng kwarg.')
    if self._mode == 'train' and self._rate > 0.0:
      noise_shape = list(x.shape)
      for dim in self._broadcast_dims:
        noise_shape[dim] = 1
      keep_prob = jax.lax.tie_in(rng, 1.0 - self._rate)
      keep = random.bernoulli(rng, keep_prob, tuple(noise_shape))
      multiplier = keep.astype(x.dtype) / jax.lax.tie_in(keep, keep_prob)
      return x * multiplier, state
    else:
      return x, state


def FeedForward(d_model, d_ff, dropout, activation, act_dropout, mode):
  """Feed-forward block with layer normalization at start."""
  if act_dropout is None:
    act_dropout = dropout
  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      BroadcastedDropout(rate=act_dropout, mode=mode),  # pylint: disable=no-value-for-parameter
      activation(),
      tl.Dense(d_model),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
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
    return np.reshape(x, [n_chunks, 1, chunk_size] + list(x.shape[2:]))
  return [
      tl.Dup(),  # Just to have shape for later after scan.
      tl.Fn(reshape_to_chunks, n_out=1),
      tl.Scan(tl.Serial(ff), axis=0, n_carry=0, remat=True),
      tl.Fn(lambda x, y: np.reshape(x, y.shape))
  ]


class SplitForOutput(tl.ReversibleLayer):
  """Splits activations into sections (for use right before the output layer).

  After the reversible portion of the network, there is a final output portion
  that's non-reversible (which at minimum includes normalization, output
  projection, and log-softmax). The output portion needs to operate on chunks
  of the sequence to avoid running out of memory for large vocabulary sizes.

  This layer concatenates the two subparts of the activations along the feature
  dimension, and then splits into chunks along the time dimension. We implement
  it is a subclass of tl.ReversibleLayer because we want to ensure that multiple
  copies of the activations don't exist simultaneously except in the middle of a
  memory copy operation.
  """

  def __init__(self, n_sections=2, axis=-2):
    super(SplitForOutput, self).__init__(n_in=2, n_out=n_sections)
    self._n_sections = n_sections
    self._axis = axis

  def forward(self, inputs, weights):
    del weights
    x1, x2 = inputs

    x1_split = np.split(x1, self._n_sections, self._axis)
    x2_split = np.split(x2, self._n_sections, self._axis)

    res = [np.concatenate(ys, -1) for ys in zip(x1_split, x2_split)]
    if len(res) == 1:
      return res[0]
    else:
      return tuple(res)

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    del weights, kwargs
    if not isinstance(output, (list, tuple)):
      output = [output]

    x1_split = []
    x2_split = []
    for y in output:
      y1, y2 = np.split(y, 2, -1)
      x1_split.append(y1)
      x2_split.append(y2)

    x1 = np.concatenate(x1_split, self._axis)
    x2 = np.concatenate(x2_split, self._axis)

    return (x1, x2)

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       **kwargs):
    del weights, kwargs
    return self.reverse(output), (self.reverse(ct), ())


@tl.layer()
def Chunk(x, weights, n_sections=2, **kwargs):
  del weights, kwargs
  assert x.shape[1] % n_sections == 0
  return np.reshape(x, (
      x.shape[0] * n_sections,
      x.shape[1] // n_sections,
      ) + x.shape[2:])


@tl.layer()
def Unchunk(x, weights, n_sections=2, **kwargs):
  del weights, kwargs
  assert x.shape[0] % n_sections == 0
  return np.reshape(x, (
      x.shape[0] // n_sections,
      x.shape[1] * n_sections,
      ) + x.shape[2:])


class ReversibleHalfResidual(tl.ReversibleLayer, tl.Serial):
  """Half of a RevNet-style residual (only updates part of the hidden state)."""

  def __init__(self, residual_layers):
    self.compute_residual = tl.Serial(         # x1_or_y1, x2,           ...
        tl.Select([1, 0, 1]),                  # x2, x1_or_y1, x2,       ...
        tl.Parallel([], [], residual_layers),  # x2, x1_or_y1, residual, ...
        tl.Select([2, 1, 0]),                  # residual, x1_or_y1, x2, ...
    )

    self.n_preserve = self.compute_residual.n_out - 2
    parallel_preserve = [[]] * self.n_preserve

    layers = [
        self.compute_residual,
        tl.Parallel(tl.Add(), *parallel_preserve)
    ]
    super(ReversibleHalfResidual, self).__init__(layers)

    self.subtract_top = tl.Parallel(tl.SubtractTop(), *parallel_preserve)
    self.reverse_layers = [self.compute_residual, self.subtract_top]

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    reconstructed_x = output
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = random.split(rng, self._n_layers)
    # Note that self.sublayers aligns exactly with self.reverse_layers in
    # terms of parameter and rng usage, so no re-ordering is required.
    for layer, p, s, ns, rng in zip(
        self.reverse_layers, weights, state, new_state, rngs):
      reconstructed_x = layer(reconstructed_x, weights=p,
                              state=s, new_state=ns, rng=rng, **kwargs)
    return reconstructed_x

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = random.split(rng, self._n_layers)

    def call_compute_residual(x, weights):
      res = self.compute_residual(x, weights=weights, state=state[0],
                                  rng=rngs[0], **kwargs)
      return res

    assert len(ct) == self.n_preserve + 1
    ct = (ct[0], ct[0]) + ct[1:]

    stack_with_residual, vjpfun = jax.vjp(
        call_compute_residual, output, weights[0])
    reconstructed_x = self.subtract_top(
        stack_with_residual, weights=weights[-1], state=state[-1], rng=rngs[-1],
        **kwargs)

    x_ct, residual_weights_ct = vjpfun(ct)
    assert not jax.tree_util.tree_leaves(weights[-1])
    add_top_weights_ct = weights[-1]
    return reconstructed_x, (x_ct, [residual_weights_ct, add_top_weights_ct])


class ApplyAttentionWrapper(tl.Parallel):
  """Like tl.Parallel(attention, [], []) but implements forward_and_backward."""

  def __init__(self, attention):
    assert hasattr(attention, 'forward_and_backward')
    super(ApplyAttentionWrapper, self).__init__(attention, [], [])
    self.attention = attention

  def forward_and_backward(self, inputs, ct, state, new_state, rng=None,
                           **kwargs):
    # Simultaneous forward pass and backprop through the attention mechanism.
    qkv = inputs[:3]
    passthrough = inputs[3:]
    out_ct = ct[0]
    passthrough_ct = ct[1:]
    if rng is not None:
      # Adjust RNG to match the forward pass.
      rng = random.split(rng, self._n_layers)[0]

    out, qkv_ct = self.attention.forward_and_backward(
        qkv, out_ct, rng=rng, state=state[0], new_state=new_state[0], **kwargs)
    return (out,) + passthrough, qkv_ct + passthrough_ct


class ReversibleAttentionHalfResidual(tl.ReversibleLayer, tl.Serial):
  """Half of a RevNet-style residual that performs attention.

  If inputs are (x1, x2), then outputs are (x1 + z, x2) where:
  z = post_attention(attention(pre_attention(x1)))

  Other than an efficiency optimization, this layer is equivalent to
  ReversibleHalfResidual([pre_attention, attention, post_attention]).

  The post_attention layers must be linear in their input (typically they will
  consists of reshaping and dense linear layers), which allows the following
  optimization. We can back-propagate the gradient signal from the output of
  ReversibleAttentionHalfResidual to the output of the "attention" portion based
  only on the network parameters. Then, attention.forward_and_backward can be
  used to recover the output of the "attention" portion while simultaneously
  performing the backward pass, which allows shared computation between the two
  directions.
  """

  def __init__(self, pre_attention, attention, post_attention):
    self.pre_attention = tl.Serial(
        # (x1_or_y1, x2) -> (x2, x1_or_y1, x2)
        tl.Parallel([], tl.Dup()),
        tl.Swap(),
        tl.Parallel(pre_attention, [], []),
    )
    assert hasattr(attention, 'forward_and_backward')
    self.attention = ApplyAttentionWrapper(attention)
    self.post_attention = tl.Parallel(post_attention, [], [])

    layers = [
        self.pre_attention,
        self.attention,
        self.post_attention,
        tl.Parallel(tl.Add(), []),
    ]
    super(ReversibleAttentionHalfResidual, self).__init__(layers)

    self.subtract_top = tl.Parallel(tl.SubtractTop(), [])
    self.reverse_layers = [
        self.pre_attention,
        self.attention,
        self.post_attention,
        self.subtract_top,
    ]

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = random.split(rng, self._n_layers)

    reconstructed_x = output
    # Note that self.sublayers aligns exactly with self.reverse_layers in
    # terms of parameter and rng usage, so no re-ordering is required.
    for layer, p, s, ns, rng in zip(self.reverse_layers, weights,
                                    state, new_state, rngs):
      reconstructed_x = layer.reverse(reconstructed_x, weights=p,
                                      state=s, new_state=ns, rng=rng, **kwargs)
    return reconstructed_x

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = random.split(rng, self._n_layers)

    # Forward pass through self.pre_attention, while preparing for
    # later backprop.
    def call_pre_attention(x, weights):
      res = self.pre_attention(x, weights=weights, state=state[0], rng=rngs[0],
                               **kwargs)
      return res
    stack, pre_attention_vjpfun = jax.vjp(call_pre_attention,
                                          output, weights[0])

    # Backprop through adding the residual
    assert len(ct) == 2
    ct = saved_ct = (ct[0], ct[0], ct[1])

    # Backprop through self.post_attention with respect to the inputs only
    def call_post_attention(x):
      res = self.post_attention(x, weights=weights[2], state=state[2],
                                rng=rngs[2], **kwargs)
      return res
    # Note: these are *not* the actual inputs to self.post_attention.
    # If self.post_attention is not linear, we will get incorrect gradients.
    dummy_inputs = (stack[-3], stack[-2], stack[-1])
    _, post_attention_vjpfun = jax.vjp(call_post_attention, dummy_inputs)
    (ct,) = post_attention_vjpfun(ct)

    # Simultaneous forward pass and backprop through the attention mechanism
    stack, ct = self.attention.forward_and_backward(
        stack, ct, rng=rngs[1], state=state[1], new_state=new_state[1],
        **kwargs)
    assert not jax.tree_util.tree_leaves(weights[1])
    attention_weights_ct = weights[1]  # This is valid when weights is empty.

    # Backprop through self.pre_attention
    x_ct, pre_attention_weights_ct = pre_attention_vjpfun(ct)

    # Forward pass for self.post_attention, and backprop with respect to the
    # parameters only
    def call_post_attention2(weights):
      res = self.post_attention(stack, weights=weights, state=state[2],
                                rng=rngs[2], **kwargs)
      return res
    stack, post_attention_vjpfun = jax.vjp(call_post_attention2, weights[2])
    (post_attention_weights_ct,) = post_attention_vjpfun(saved_ct)

    # Forward pass through subtracting the residual
    reconstructed_x = self.subtract_top(
        stack, weights=weights[-1], state=state[-1], rng=rngs[-1], **kwargs)

    assert not jax.tree_util.tree_leaves(weights[-1])
    add_top_weights_ct = weights[-1]
    weights_ct = [
        pre_attention_weights_ct,
        attention_weights_ct,
        post_attention_weights_ct,
        add_top_weights_ct,
    ]

    return reconstructed_x, (x_ct, weights_ct)


class ReversibleHalfResidualV2(tl.ReversibleLayer):
  """Half of a RevNet-style residual that optionally performs attention.

  This layer is designed to replace both ReversibleHalfResidual and
  ReversibleAttentionHalfResidual. When attention_layer is None, this layer has
  the signature [accumulator, *context] -> [accumulator + f(context), *context].
  The attention_layer must be an instance of EfficientAttentionBase or one of
  its subclasses (see efficient_attention_v2.py), or None.

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

  def forward_with_state(self, xs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, **kwargs):
    rngs = _pop_rng_and_split(kwargs, len(self.sublayers))

    accumulator, *context = xs
    stack = context = tuple(context)
    new_state = []
    for layer, w, s, rng in zip(self.sublayers, weights, state, rngs):
      inputs = _inputs_from_stack(layer, stack)
      outputs, s = layer.pure_fn(inputs, w, s, rng)
      stack = _outputs_onto_stack(layer, outputs, stack)
      new_state.append(s)
    residual = stack[0] if isinstance(stack, (tuple, list)) else stack

    output = accumulator + residual
    stack = (output,) + context
    return stack, new_state

  def reverse(self, *args, **kwargs):
    raise NotImplementedError('Only reverse_and_grad is actually used.')

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       **kwargs):
    rngs = _pop_rng_and_split(kwargs, len(self.sublayers))

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
    outputs, compute_residual_vjpfun, outputs_aux = jax.vjp(
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
    stack_ct = (accumulator_output_ct,) + jax.tree_multimap(
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
  def new_weights_and_state(self, input_signature):
    stack = input_signature[1:]
    if len(stack) == 1:
      stack = stack[0]

    inputs = _inputs_from_stack(self.compute_residual, stack)
    weights, state = self.compute_residual.init(inputs)
    outputs, _ = self.compute_residual._forward_abstract(inputs)
    stack = _outputs_onto_stack(self.compute_residual, outputs, stack)

    if self.attention_layer is None:
      return (weights,), (state,)
    else:
      inputs = _inputs_from_stack(self.attention_layer, stack)
      attn_weights, attn_state = self.attention_layer.init(inputs)
      return (weights, attn_weights), (state, attn_state)
  # pylint: enable=protected-access

  # pylint: disable=protected-access
  def _set_input_signature_recursive(self, input_signature):
    """Sets input signatures for this layer and sublayers, recursively.

    Args:
      input_signature: A `ShapeDtype` instance (if this layer takes one input)
          or a list/tuple of `ShapeDtype` instances.
    """
    self._input_signature = input_signature

    # Infer shapes and dtypes (signatures) through the successive sublayers.
    stack = input_signature[1:]
    for layer in self.sublayers:
      inputs = _inputs_from_stack(layer, stack)
      layer._set_input_signature_recursive(inputs)
      outputs, _ = layer._forward_abstract(inputs)
      stack = _outputs_onto_stack(layer, outputs, stack)
  # pylint: enable=protected-access


def DecoderBlock(d_model, d_ff, d_attention_key, d_attention_value,
                 n_heads, n_attention_chunks, attention_type,
                 dropout, share_qk, ff_activation, ff_use_sru, ff_chunk_size,
                 mode):
  """Reversible transformer decoder layer.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_heads: int: number of attention heads
    n_attention_chunks: int: number of chunks for attention
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: string, whether to share queries and keys
    ff_activation: the non-linearity in feed-forward layer
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  if not hasattr(attention_type, 'forward_unbatched'):
    if share_qk:
      pre_attention = [
          Chunk(n_sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
          tl.LayerNorm(),
          tl.Branch(
              tl.ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
              tl.ComputeAttentionHeads(
                  n_heads=n_heads, d_head=d_attention_value),
          ),
          tl.Dup(),
      ]
    else:
      pre_attention = [
          Chunk(n_sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
          tl.LayerNorm(),
          tl.Branch(
              tl.ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
              tl.ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
              tl.ComputeAttentionHeads(
                  n_heads=n_heads, d_head=d_attention_value),
          ),
      ]

    attention = attention_type(mode=mode)

    # ReversibleAttentionHalfResidual requires that post_attention be linear in
    # its input (so the backward pass can be computed without knowing the input)
    post_attention = [
        tl.ComputeAttentionOutput(n_heads=n_heads, d_model=d_model),
        Unchunk(n_sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
        BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
    ]

    attention_half_residual = ReversibleAttentionHalfResidual(
        pre_attention, attention, post_attention)
  else:
    attention = attention_type(
        n_heads=n_heads, d_qk=d_attention_key, d_v=d_attention_value,
        share_qk=share_qk, causal=True, output_dropout=dropout, mode=mode)
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
      ReversibleHalfResidual(feed_forward),
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
               n_chunks=0,
               n_attention_chunks=1,
               attention_type=tl.DotProductCausalAttention,
               share_qk=False,
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
    n_chunks: int: number of chunks (must match input pipeline)
    n_attention_chunks: int: number of chunks for attention
    attention_type: class: attention class to use, such as DotProductAttention.
    share_qk: bool, whether to share queries and keys.
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
  if n_chunks == 0:
    n_chunks = 1
    concatenate_input_chunks = []
  else:
    concatenate_input_chunks = tl.Concatenate(n_items=n_chunks)

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
      tl.Embedding(d_emb, vocab_size),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
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
        n_attention_chunks,
        attention_type=layer_attention_type,
        dropout=dropout,
        share_qk=(share_qk or issubclass(layer_attention_type,
                                         tl.LSHCausalAttention)),
        ff_activation=ff_activation,
        ff_use_sru=ff_use_sru,
        ff_chunk_size=ff_chunk_size,
        mode=mode)
    decoder_blocks.append(decoder_block)

  return tl.Serial(
      concatenate_input_chunks,
      tl.ShiftRight(mode=mode),
      positional_embedder,
      tl.Dup(),
      tl.ReversibleSerial(decoder_blocks + [
          SplitForOutput(n_sections=n_chunks, axis=-2),  # pylint: disable=no-value-for-parameter
      ]),
      Map([
          # TODO(kitaev): Test whether dropout should go before or after the
          # LayerNorm, and whether dropout broadcasting is needed here.
          tl.LayerNorm(),
          BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
          tl.Dense(vocab_size),
          tl.LogSoftmax(),
      ], n_sections=n_chunks),
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
                      n_attention_chunks=1,
                      attention_type=tl.DotProductCausalAttention,
                      share_qk=False,
                      axial_pos_shape=(),
                      d_axial_pos_embs=None,
                      ff_activation=tl.FastGelu,
                      ff_use_sru=0,
                      ff_chunk_size=0,
                      mode='train'):
  """Reversible transformer language model with shortening.

  When shorten_factor is F and processing an input of shape [batch, length],
  we embed the (shifted-right) input and then group each F elements (on length)
  into a single vector -- so that in the end we process a tensor of shape
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
    n_attention_chunks: int: number of chunks for attention
    attention_type: class: attention class to use, such as DotProductAttention.
    share_qk: bool, whether to share queries and keys.
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
      tl.Embedding(d_embedding, vocab_size),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
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
        n_attention_chunks,
        attention_type=layer_attention_type,
        dropout=dropout,
        share_qk=(share_qk or issubclass(layer_attention_type,
                                         tl.LSHCausalAttention)),
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
      tl.Fn(lambda x: np.reshape(  # Shorten -- move to depth.
          x, (x.shape[0], x.shape[1] // shorten_factor, -1)), n_out=1),
      tl.Dense(d_model),
      tl.Dup(),  # Stack has (short_x, short_x, x)
      tl.ReversibleSerial(decoder_blocks),
      tl.Select([0], n_in=2),
      tl.LayerNorm(),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
      tl.Dense(shorten_factor * d_embedding),
      tl.Fn(lambda x: np.reshape(  # Prolong back.
          x, (x.shape[0], x.shape[1] * shorten_factor, -1)), n_out=1),
      tl.Concatenate(),  # Concatenate with just the embeddings.
      tl.CausalConv(d_embedding),
      tl.Relu(),
      tl.SRU(d_embedding),  # One RNN layer for conditional dependence.
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )
  # pylint: enable=g-long-lambda


def EncoderBlock(d_model, d_ff, n_heads, dropout, ff_activation, ff_dropout,
                 mode):
  """Returns a list of layers that implements a Reformer encoder block.

  The input to the layer is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
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

  attention = tl.SelfAttention(
      n_heads=n_heads, d_qk=d_model//n_heads, d_v=d_model//n_heads,
      masked=True,
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
  jax.api._check_inexact_input_vjp = lambda x: None  # pylint: disable=protected-access

  def PositionalEncoder(vocab_size, mode):  # tokens --> vectors
    # TODO(kitaev): axial positional encoding is better for very long sequences.
    positional_encoding = tl.PositionalEncoding(
        max_len=max_len, dropout=dropout, mode=mode)
    return [
        tl.Embedding(d_model, vocab_size),
        BroadcastedDropout(rate=dropout, mode=mode),
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

  encoder_blocks = [
      EncoderBlock(
          d_model, d_ff, n_heads, dropout, ff_activation, ff_dropout, mode)
      for _ in range(n_encoder_layers)]

  encoder = tl.Serial([
      in_encoder,
      tl.Dup(),
      tl.ReversibleSerial(encoder_blocks),
      tl.Fn(lambda x, y: (x+y)/2.0),
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
      tl.Branch([], [tl.Dup(),              # tok_e tok_d tok_d
                     tl.PaddingMask(),
                     tl.Fn(lambda x: np.squeeze(x, (1, 2)), n_out=1)]),
      #                                     # tok_e mask  tok_d .....

      # Encode.
      encoder,                              # vec_e  mask tok_d .....

      # Decode.
      tl.Select([2, 0, 1]),                 # tok_d vec_e mask .....
      tl.ShiftRight(mode=mode),             # tok_d vec_e mask .....
      out_encoder,                          # vec_d vec_e mask .....
      tl.Dup(),                             # vec_d1 vec_d2 vec_e mask .....
      tl.ReversibleSerial(encoder_decoder_blocks),
      tl.Fn(lambda x, y: (x+y)/2.0),        # vec_d vec_e mask .....
      tl.LayerNorm(),                       # vec_d vec_e mask .....

      # Map to output vocab.
      tl.Select([0], n_in=3),               # vec_d .....
      tl.Dense(output_vocab_size),          # vec_d .....
      tl.LogSoftmax(),                      # vec_d .....
  )
