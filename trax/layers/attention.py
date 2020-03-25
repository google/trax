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
"""Attention Layers."""

import jax
import numpy as onp

from trax import math
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import initializers as init
from trax.math import numpy as np


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


def zero_pad(x, pad, axis):
  """Helper for np.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return np.pad(x, pad_widths, mode='constant',
                constant_values=x.dtype.type(0))


@base.layer()
def ShiftRight(x, n_shifts=1, mode='train', **unused_kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
  if mode == 'predict':
    # Do nothing in predict mode, as then the sequence length is 1.
    return x
  padded = zero_pad(x, (n_shifts, 0), 1)
  return padded[:, :-n_shifts]


@base.layer()
def PaddingMask(x, weights, pad=0, **kwargs):
  del weights, kwargs
  return np.reshape(x != pad, (x.shape[0], 1, 1, x.shape[-1]))


@base.layer(n_in=2)
def EncoderDecoderMask(x, **unused_kwargs):
  """Makes encoder-decoder mask from decoder input and a padding mask."""
  decoder_input, padding_mask = x
  padding_mask = np.reshape(
      padding_mask, (padding_mask.shape[0], 1, 1, padding_mask.shape[-1]))
  # Final mask shape is [batch, 1 for heads, decoder-len, encoder-len].
  return padding_mask + np.zeros((1, 1, decoder_input.shape[1], 1))


class PositionalEncoding(base.Layer):
  """Implements bare positional encoding."""

  def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
               mode='train'):
    super(PositionalEncoding, self).__init__()
    self._max_len = max_len
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self._dropout = dropout
    else:
      self._dropout = 0.0
    self._dropout_broadcast_dims = dropout_broadcast_dims
    self._mode = mode

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    if self._mode in ('train', 'eval'):
      x = inputs
      symbol_size = np.shape(x)[1]
      px = weights[:, :symbol_size, :]
      if self._dropout == 0:
        return (x + px, state)
      else:
        noise_shape = list(px.shape)
        for dim in self._dropout_broadcast_dims:
          noise_shape[dim] = 1
        keep_prob = 1.0 - self._dropout
        if math.backend_name() == 'jax':
          keep_prob = jax.lax.tie_in(x, np.full((), keep_prob, dtype=x.dtype))
        keep = math.random.bernoulli(rng, keep_prob, tuple(noise_shape))
        multiplier = keep.astype(x.dtype) / keep_prob
        return (x + px * multiplier, state)
    else:
      assert self._mode == 'predict'
      assert self._dropout == 0
      # State in this class is only used for fast inference. In that case,
      # the model is called with consecutive elements position-by-position.
      # This positional encoding layer needs to store the index of the current
      # position then and increment it on each call -- that's how state is used
      # and updated below.
      if inputs.shape[1] == 1:
        return (inputs + np.expand_dims(weights[0, state, :], 1), state + 1)
      else:
        emb = []
        for i in range(inputs.shape[0]):
          emb.append(jax.lax.dynamic_slice_in_dim(
              weights[0], state[i], inputs.shape[1], axis=0))
        return inputs + np.stack(emb, 0), state + inputs.shape[1]

  def new_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    pe = onp.zeros((self._max_len, d_feature), dtype=onp.float32)
    position = onp.arange(0, self._max_len)[:, onp.newaxis]
    div_term = onp.exp(
        onp.arange(0, d_feature, 2) * -(onp.log(10000.0) / d_feature))
    pe[:, 0::2] = onp.sin(position * div_term)
    pe[:, 1::2] = onp.cos(position * div_term)
    pe = pe[onp.newaxis, :, :]  # [1, self._max_len, d_feature]
    weights = np.array(pe)  # These are trainable parameters, initialized above.
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      state = np.zeros((batch_size,), dtype=np.int32)
    else:
      state = base.EMPTY_STATE
    return weights, state


class AxialPositionalEncoding(base.Layer):
  """Axial positional encoding."""
  # TODO(kitaev): support variable-length sequences.

  def __init__(self, shape=(64, 64, 3), d_embs=(384, 384, 256),
               kernel_initializer=init.RandomNormalInitializer(1.0),
               dropout=0.0, dropout_broadcast_dims=(), mode='train'):
    super(AxialPositionalEncoding, self).__init__()
    self._kernel_initializer = kernel_initializer
    assert len(shape) == len(d_embs)
    self._shape = shape
    self._d_embs = d_embs

    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self._dropout = dropout
    else:
      self._dropout = 0.0
    self._dropout_broadcast_dims = dropout_broadcast_dims
    self._mode = mode

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    embs = []
    for ax_emb in weights:
      ax_emb = np.broadcast_to(
          ax_emb, (inputs.shape[0],) + self._shape + (ax_emb.shape[-1],))
      embs.append(ax_emb)

    if self._mode == 'predict':
      assert self._dropout == 0.0
      emb = np.concatenate(embs, -1)
      emb = np.reshape(emb, (inputs.shape[0], -1, emb.shape[-1]))
      emb = jax.lax.dynamic_slice_in_dim(emb, state, inputs.shape[1], axis=1)
      return inputs + emb, state + inputs.shape[1]
    elif self._dropout == 0:
      # TODO(kitaev): concat-then-reshape (as is the case with dropout enabled)
      # leads to memory blow-up on TPU.
      # emb = np.concatenate(embs, -1)
      # return inputs + np.reshape(emb, inputs.shape), state
      return inputs + np.concatenate(
          [np.reshape(emb, inputs.shape[:-1] + (emb.shape[-1],))
           for emb in embs
          ], -1), state
    else:
      emb = np.concatenate(embs, -1)
      noise_shape = list(emb.shape)
      for dim in self._dropout_broadcast_dims:
        noise_shape[dim] = 1
      keep_prob = 1.0 - self._dropout
      if math.backend_name() == 'jax':
        keep_prob = jax.lax.tie_in(
            inputs, np.full((), keep_prob, dtype=inputs.dtype))
      keep = math.random.bernoulli(rng, keep_prob, tuple(noise_shape))
      multiplier = keep.astype(inputs.dtype) / keep_prob

      return inputs + np.reshape(emb * multiplier, inputs.shape), state

  def new_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    assert sum(self._d_embs) == d_feature

    rngs = self.new_rngs(len(self._d_embs))
    weights = []
    for ax, (ax_rng, d_emb) in enumerate(zip(rngs, self._d_embs)):
      ax_shape = [1] * len(self._shape)
      ax_shape[ax] = self._shape[ax]
      ax_shape = (1,) + tuple(ax_shape) + (d_emb,)
      ax_emb = self._kernel_initializer(ax_shape, ax_rng)
      weights.append(ax_emb)

    state = 0 if self._mode == 'predict' else base.EMPTY_STATE
    return tuple(weights), state


def DotProductAttention(query, key, value, mask, dropout, mode, rng):
  """Core dot product self-attention.

  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
    dropout: float: dropout rate
    mode: 'eval' or 'train': whether to use dropout
    rng: JAX PRNGKey: subkey for disposable use

  Returns:
    Self attention for q, k, v arrays.
  """
  depth = np.shape(query)[-1]
  dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
  if mask is not None:
    # TODO(kitaev): workaround for https://github.com/google/jax/issues/850
    # We must ensure that both mask and the -1e9 constant have a data dependency
    # on the input. Broadcasted copies of these use a lot of memory, so they
    # should be computed at runtime (rather than being global constants).
    if math.backend_name() == 'jax':
      mask = jax.lax.tie_in(dots, mask)
    # JAX's `full_like` already ties in -1e9 to dots.
    dots = np.where(mask, dots, np.full_like(dots, -1e9))
  # Softmax.
  dots = np.exp(dots - math.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = math.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = np.where(keep, dots / (1.0 - dropout), np.zeros_like(dots))
  out = np.matmul(dots, value)
  return out


class PureAttention(base.Layer):
  """Layer constructor function for a pure attention layer."""

  def __init__(self, n_heads=1, dropout=0.0, mode='train'):
    super(PureAttention, self).__init__(n_in=4, n_out=2)
    self._n_heads = n_heads
    self._dropout = dropout
    self._mode = mode

  def forward_with_state(self, x, weights, state, rng):
    """Pure transformer-style multi-headed attention.

    Args:
      x: inputs (q, k, v, mask)
      weights: parameters (none)
      state: parameters (none)
      rng: random number generator

    Returns:
      Pure Multi-headed attention result, and the mask.
    """
    del weights
    n_heads, dropout, mode = self._n_heads, self._dropout, self._mode
    q, k, v, mask = x
    d_feature = q.shape[-1]
    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads
    nbatch = np.shape(q)[0]
    # nbatch, seqlen, d_feature --> nbatch, n_heads, seqlen, d_head
    def SplitHeads(x):
      return np.transpose(
          np.reshape(x, (nbatch, -1, n_heads, d_head)), (0, 2, 1, 3))
    # nbatch, n_heads, seqlen, d_head --> nbatch, seqlen, d_feature
    def JoinHeads(x):  # pylint: disable=invalid-name
      return np.reshape(
          np.transpose(x, (0, 2, 1, 3)), (nbatch, -1, n_heads * d_head))
    # Split heads, dot-product attention, rejoin heads.
    res = JoinHeads(
        DotProductAttention(
            SplitHeads(q), SplitHeads(k), SplitHeads(v), mask,
            dropout=dropout, mode=mode, rng=rng))
    return (res, mask), state  # Keep the mask.


def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form q, k, v, mask.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return cb.Serial(
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      PureAttention(  # pylint: disable=no-value-for-parameter
          n_heads=n_heads, dropout=dropout, mode=mode),
      core.Dense(d_feature),
  )


def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form (x, mask) and constructs (q, k, v) from x.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return cb.Serial(
      cb.Dup(), cb.Dup(),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  )


class ShiftRightLearned(base.Layer):
  """Layer constructor function for shifting right by a learned vector."""

  def __init__(self, initializer=init.RandomNormalInitializer(0.01)):
    super(ShiftRightLearned, self).__init__()
    self._initializer = initializer

  def forward(self, x, weights):
    c = np.reshape(weights, [1, 1, -1])
    c += np.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
    return np.concatenate([c, x], axis=1)[:, :-1, :]

  def new_weights(self, input_signature):
    b = self._initializer((input_signature.shape[-1],), self.new_rng())
    return b


class ComputeAttentionHeads(base.Layer):
  """Computes queries/keys/values via linear projection.

  The output shape is (n_batch * n_heads, seqlen, d_head); the batch and head
  dimensions are fused to allow for more efficient memory layouts.
  """

  def __init__(self, n_heads=1, d_head=64,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(ComputeAttentionHeads, self).__init__()
    self._n_heads = n_heads
    self._d_head = d_head
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.
    # Note that AttentionQKV above is different in that it uses a bias term.

  def forward(self, x, weights):
    seqlen = x.shape[1]
    res = np.dot(x, weights)

    # n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head
    res = np.reshape(res, (x.shape[0], seqlen, self._n_heads, self._d_head))
    # n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head
    res = np.transpose(res, (0, 2, 1, 3))
    # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
    res = np.reshape(res, (-1, seqlen, self._d_head))

    return res

  def new_weights(self, input_signature):
    w = self._kernel_initializer(
        (input_signature.shape[-1], self._n_heads * self._d_head),
        self.new_rng())
    return w


class ComputeAttentionOutput(base.Layer):
  """Joins outputs from different heads via linear projection."""

  def __init__(self, n_heads=1, d_model=1024,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(ComputeAttentionOutput, self).__init__()
    self._n_heads = n_heads
    self._d_model = d_model
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.
    # Note that AttentionQKV above is different in that it uses a bias term.

  def forward(self, x, weights):
    seqlen = x.shape[1]
    d_head = x.shape[2]

    x = np.reshape(x, (-1, self._n_heads, seqlen, d_head))
    x = np.transpose(x, (0, 2, 1, 3))  # -> n_batch, seqlen, n_heads, d_head
    x = np.reshape(x, (-1, seqlen, self._n_heads * d_head))

    return np.dot(x, weights)

  def new_weights(self, input_signature):
    kernel_shape = (input_signature.shape[-1] * self._n_heads, self._d_model)
    w = self._kernel_initializer(kernel_shape, self.new_rng())
    return w


class BaseCausalAttention(base.Layer):
  """Base class for variants of causal self-attention."""

  def __init__(self, mode='train'):
    del mode
    super(BaseCausalAttention, self).__init__(n_in=3)

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    """Forward pass for the attention layer."""
    raise NotImplementedError()

  def forward_and_backward(self, inputs, grad, state, new_state, **kwargs):
    """Performs both forward and backward pass for the attention layer.

    This is used in reversible models: for the backward pass of a reversible
    model, we need to compute both the forward direction (to recover the
    previous layer's activations) and the backward direction simultaneously.
    Some computation can be shared between the forward and backward directions,
    which makes it more efficient to implement them jointly.

    This method assumes that the layer is stateless and has no parameters.

    Args:
      inputs: A tuple (q, k, v), where each element has shape
          n_batch*n_heads, seqlen, d_head
      grad: gradient signal for the layer output.
      state: start state
      new_state: updated state computed by the forward pass
      **kwargs: kwargs for the layer

    Returns:
      A nested-tuple structure (output, (q_grad, k_grad, v_grad)) that contains
      the output of the forward pass and the gradient signal for each input.
    """
    raise NotImplementedError()


def _fast_inference_init_state(input_signature, buffer_length):
  """Returns an initial state for causal attention layer fast inference."""
  def zeros_for(batch_size, shape_dtype):
    shape, dtype = shape_dtype.as_tuple()
    depth = shape[-1]
    return np.zeros((batch_size, buffer_length, depth), dtype=dtype)

  batch_size = input_signature[0].shape[0]
  k = zeros_for(batch_size, input_signature[1])
  v = zeros_for(batch_size, input_signature[2])
  mask = np.zeros((batch_size, 1, buffer_length))
  seq_indices = np.zeros((batch_size,), dtype=np.int32)
  return (k, v, mask, seq_indices)


def _fast_inference_update_state(inputs, state):
  """Updates state of a causal attention layer for fast inference."""
  assert math.backend_name() == 'jax', (
      'JAX backend is required to use the predict mode.')
  for x in inputs:
    assert x.shape[1] == 1, (
        'In predict mode the input sequence must be of length 1.')
  # Fast inference: run with only 1 query in each step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  (ks, vs, mask, seq_indices) = state
  batch_indices = np.arange(ks.shape[0])
  ks = jax.ops.index_update(
      ks, jax.ops.index[batch_indices, seq_indices, :], new_k[:, 0, :]
  )
  vs = jax.ops.index_update(
      vs, jax.ops.index[batch_indices, seq_indices, :], new_v[:, 0, :]
  )
  mask = jax.ops.index_update(
      mask, jax.ops.index[batch_indices, :, seq_indices], 1
  )
  return (ks, vs, mask, seq_indices + 1)


class DotProductCausalAttention(BaseCausalAttention):
  """A standard (non-memory-efficient) dot product attention implementation."""

  def __init__(self, dropout=0.0, mode='train'):
    super(DotProductCausalAttention, self).__init__()
    self._dropout = dropout
    self._mode = mode

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    del weights
    q, k, v = inputs
    if self._mode in ('train', 'eval'):
      mask_size = q.shape[-2]
      # Not all backends define np.tril. However, using onp.tril is inefficient
      # in that it creates a large global constant. TODO(kitaev): try to find an
      # alternative that works across all backends.
      if math.backend_name() == 'jax':
        mask = np.tril(
            np.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
      else:
        mask = onp.tril(
            onp.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    else:
      assert self._mode == 'predict'
      state = _fast_inference_update_state(inputs, state)
      (k, v, mask, _) = state

    res = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res, state

  def forward_and_backward(self, inputs, ct, state=base.EMPTY_STATE,
                           new_state=base.EMPTY_STATE, **kwargs):
    del new_state
    assert math.backend_name() == 'jax', (
        'JAX backend is required to use forward_and_backward.')
    # Simultaneous forward pass and backprop through the attention mechanism.
    def _do_forward(x):  # pylint: disable=invalid-name
      res, _ = self.forward_with_state(x, state=state, **kwargs)
      return res
    output, vjpfun = jax.vjp(_do_forward, inputs)
    return output, vjpfun(ct)[0]

  def new_weights_and_state(self, input_signature):
    if self._mode in ('train', 'eval'):
      return base.EMPTY_WEIGHTS, base.EMPTY_STATE

    assert self._mode == 'predict'
    weights = base.EMPTY_WEIGHTS
    # Buffer length is hardcoded for now. TODO(pkozakowski): Pass it from the
    # model.
    max_len = 2048
    state = _fast_inference_init_state(input_signature, max_len)
    return weights, state


def CausalAttention(d_feature, n_heads=1,
                    d_attention_key=None, d_attention_value=None,
                    attention_type=DotProductCausalAttention,
                    share_qk=False, mode='train'):
  """Transformer-style multi-headed causal attention.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head
        (default is d_feature // n_heads)
    d_attention_value: int: depth of value vector for each attention head
        (default is d_feature // n_heads)
    attention_type: subclass of BaseCausalAttention: attention class to use
    share_qk: bool, whether to share queries and keys
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result.
  """
  if d_attention_key is None:
    assert d_feature % n_heads == 0
    d_attention_key = d_feature // n_heads
  if d_attention_value is None:
    assert d_feature % n_heads == 0
    d_attention_value = d_feature // n_heads

  if share_qk:
    pre_attention = [
        cb.Dup(),
        cb.Parallel(
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_value),
        ),
        cb.Dup(),
    ]
  else:
    pre_attention = [
        cb.Dup(), cb.Dup(),
        cb.Parallel(
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_value),
        ),
    ]

  return cb.Serial(pre_attention + [
      attention_type(mode=mode),
      ComputeAttentionOutput(n_heads=n_heads, d_model=d_feature),
  ])
