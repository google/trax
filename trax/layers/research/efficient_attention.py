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

"""Attention Layers optimized for efficiency."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as python_math
import random

import jax

from trax import math
from trax.layers import attention
from trax.layers import base
from trax.math import numpy as np

# pylint: disable=protected-access
_fast_inference_init_state = attention._fast_inference_init_state
_fast_inference_update_state = attention._fast_inference_update_state
# pylint: enable=protected-access

# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


class MemoryEfficientCausalAttention(attention.BaseCausalAttention):
  """Memory-efficient dot product attention.

  This layer performs causal attention on long sequences without running out
  of memory. Instead of computing dot products for all query-key pairs at once,
  it uses a loop to compute attention for a small set of query positions at a
  time. The "loop_stride" parameter controls how many query positions are
  considered at each iteration of the loop.

  Note that this class does not slice along the batch/head dimension. Looping
  over batch elements and heads instead of query positions is also a viable
  option. We haven't implemented it, but it may perform well, too.
  """

  def __init__(self, loop_stride, dropout, mode, share_qk=False, hard_k=0):
    assert math.backend_name() == 'jax', (
        'JAX backend is required to use MemoryEfficientCausalAttention.')
    super(MemoryEfficientCausalAttention, self).__init__()
    self._loop_stride = loop_stride
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self.dropout = dropout
    else:
      self.dropout = None
    self._share_qk = share_qk
    self._hard_k = hard_k

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, **kwargs):
    del weights
    output, _ = self.forward_and_backward(inputs, None, **kwargs)
    return output, state

  def has_backward(self):
    return True

  def backward(self, inputs, output, ct, weights=base.EMPTY_WEIGHTS,
               state=base.EMPTY_STATE, **kwargs):
    del output, weights, state
    _, inputs_ct = self.forward_and_backward(inputs, ct, **kwargs)
    return inputs_ct, ()

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def forward_and_backward(self, inputs, ct, state=base.EMPTY_STATE,
                           new_state=base.EMPTY_STATE, rng=None, **kwargs):
    del state, new_state, kwargs
    query, key, value = inputs
    depth = np.shape(query)[-1]
    do_backprop = ct is not None
    # jax uses the term cotangent (ct) to refer to gradient signals, and
    # vector-Jacobian product (vjp) for back-propagation through a layer.

    def make_mask(N, M, k):  # pylint: disable=invalid-name
      """Constructs a slice of the causal attention mask.

      Args:
        N: number of query positions
        M: number of key positions
        k: position of the initial query element

      Returns:
        N x M mask, where 1.0 indicates that attention is not allowed.
      """
      x = jax.lax.tie_in(k, np.arange(N, dtype=np.int32))
      y = jax.lax.tie_in(k, np.arange(M, dtype=np.int32))
      mask = jax.lax.lt(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def make_self_mask(N, M, k):  # pylint: disable=invalid-name
      """Masks out elements attending to self.

      Args:
        N: number of query positions
        M: number of key positions
        k: position of the initial query element

      Returns:
        N x M mask, where 1.0 indicates that attention is not allowed.
      """
      x = jax.lax.tie_in(k, np.arange(N, dtype=np.int32))
      y = jax.lax.tie_in(k, np.arange(M, dtype=np.int32))
      mask = jax.lax.eq(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def forward_slice(query_slice, q_loop_idx, key, value):  # pylint: disable=invalid-name
      """Forward pass for a subset of the query vectors."""
      if self._share_qk:
        key = self.make_unit_length(key)

      dots = np.matmul(
          query_slice, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

      # Causal masking
      mask = make_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
      dots = dots - 1e9 * mask

      # Mask out attention to self except when no other targets are available.
      if self._share_qk:
        self_mask = make_self_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
        dots = dots - 1e5 * self_mask

      # Softmax.
      dots = np.exp(dots - math.logsumexp(dots, axis=-1, keepdims=True))

      if self.dropout is not None and self.dropout > 0.0:
        # Dropout is broadcast across the batch+head dimension
        dropout_shape = (1, dots.shape[-2], dots.shape[-1])
        slice_rng = jax.random.fold_in(rng, q_loop_idx)
        keep_prob = jax.lax.tie_in(dots, 1.0 - self.dropout)
        keep = math.random.bernoulli(slice_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
        dots = dots * multiplier

      if self._hard_k > 0:
        top_k = np.sort(dots)[..., -self._hard_k]  # Get the top-kth weight.
        top_k = jax.lax.stop_gradient(top_k)
        dots -= top_k[..., np.newaxis]  # Subtract (be 0 for lower ones).
        dots = np.maximum(dots, 0)
        dots_sum = np.sum(dots, axis=-1, keepdims=True)  # Re-normalize.
        dots /= dots_sum  # Re-normalize.

      out_slice = np.matmul(dots, value)
      return out_slice

    def forward_and_vjp_slice(query_slice, q_loop_idx, key, value, ct_slice):  # pylint: disable=invalid-name
      # Capture q_loop_idx to avoid calculated gradients wrt. it.
      def forward_slice_with_q_loop_idx(query_slice, key, value):  # pylint: disable=invalid-name
        return forward_slice(query_slice, q_loop_idx, key, value)

      output_slice, vjpfun = jax.vjp(
          forward_slice_with_q_loop_idx, query_slice, key, value)
      return output_slice, vjpfun(ct_slice)

    q_loop_idx = np.zeros((), dtype=np.int32)
    q_loop_max = query.shape[-2]
    q_loop_stride = self._loop_stride
    if q_loop_max == 1:  # For abstract runs with unknown shapes.
      q_loop_stride = 1
    assert q_loop_max % q_loop_stride == 0, (
        'Stride must evenly divide the number of query elements.')

    out_accum = np.zeros_like(query)
    if do_backprop:
      query_ct_accum = np.zeros_like(query)
      key_ct_accum = np.zeros_like(key)
      value_ct_accum = np.zeros_like(value)
      init_vals = (
          q_loop_idx, out_accum,
          query_ct_accum, key_ct_accum, value_ct_accum)
    else:
      init_vals = (q_loop_idx, out_accum)

    def cond_fun(vals):  # pylint: disable=invalid-name
      q_loop_idx = vals[0]
      return jax.lax.lt(q_loop_idx, q_loop_max)

    def body_fun(vals):  # pylint: disable=invalid-name
      """Compute a slice of the attention mechanism."""
      if do_backprop:
        (q_loop_idx, out_accum,
         query_ct_accum, key_ct_accum, value_ct_accum) = vals
      else:
        q_loop_idx, out_accum = vals

      query_slice = jax.lax.dynamic_slice_in_dim(
          query, q_loop_idx, q_loop_stride, axis=-2)

      if do_backprop:
        ct_slice = jax.lax.dynamic_slice_in_dim(
            ct, q_loop_idx, q_loop_stride, axis=-2)
        out_slice, partial_ct = forward_and_vjp_slice(
            query_slice, q_loop_idx, key, value, ct_slice)
        query_ct_accum = jax.lax.dynamic_update_slice_in_dim(
            query_ct_accum, partial_ct[0], q_loop_idx, axis=-2)
        key_ct_accum = key_ct_accum + partial_ct[1]
        value_ct_accum = value_ct_accum + partial_ct[2]
      else:
        out_slice = forward_slice(query_slice, q_loop_idx, key, value)

      out_accum = jax.lax.dynamic_update_slice_in_dim(
          out_accum, out_slice, q_loop_idx, axis=-2)
      q_loop_idx = q_loop_idx + q_loop_stride

      if do_backprop:
        return (q_loop_idx, out_accum,
                query_ct_accum, key_ct_accum, value_ct_accum)
      else:
        return (q_loop_idx, out_accum)

    final_vals = jax.lax.while_loop(cond_fun, body_fun, init_vals)

    if not do_backprop:
      return final_vals[1], None
    else:
      return final_vals[1], final_vals[2:]


class TimeBinCausalAttention(attention.BaseCausalAttention):
  """Causal attention where only nearby chunks of items attend to each other."""

  def __init__(self, mode, dropout=0.0, bin_length=None, n_bins=None,
               share_qk=False):
    super(TimeBinCausalAttention, self).__init__()
    if (bin_length is None) == (n_bins is None):
      raise ValueError('Exactly one of {bin_length, n_bins} must be set.')
    self.bin_length = bin_length
    self.n_bins = n_bins
    self._share_qk = share_qk
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self.dropout = dropout
    else:
      self.dropout = 0.0
    self._mode = mode

  def forward_and_backward(self, inputs, ct, state, new_state, **kwargs):
    assert math.backend_name() == 'jax', (
        'JAX backend is required to use forward_and_backward.')
    # Simultaneous forward pass and backprop through the attention mechanism.
    def _do_forward(x):  # pylint: disable=invalid-name
      res, _ = self.forward_with_state(x, state=state, **kwargs)
      return res
    output, vjpfun = jax.vjp(_do_forward, inputs)
    return output, vjpfun(ct)[0]

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def _pad_inputs(self, inputs):
    seq_len = inputs[0].shape[-2]
    n_bins = self.n_bins
    bin_length = self.bin_length
    if n_bins is None:
      n_bins = int(python_math.ceil(seq_len / bin_length))
    else:
      bin_length = int(python_math.ceil(seq_len / n_bins))
    pad_len = n_bins * bin_length - seq_len

    def pad_input(x):
      pad_widths = [(0, 0)] * len(x.shape)
      pad_widths[-2] = (0, pad_len)  # Padding on axis=-2
      return np.pad(x, pad_widths, mode='constant',
                    constant_values=x.dtype.type(0))

    padded_inputs = tuple(map(pad_input, inputs))
    return (padded_inputs, seq_len, n_bins)

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    del weights, kwargs
    if self._mode in ('train', 'eval'):
      output = self._forward_train_eval(inputs, rng)
      return (output, state)
    else:
      assert self._mode == 'predict'
      return self._forward_predict(inputs, state, rng)

  def _forward_train_eval(self, inputs, rng):
    (inputs, original_len, n_bins) = self._pad_inputs(inputs)
    q, k, v = inputs
    seqlen = q.shape[-2]
    # q/k/v are n_batch*n_heads, seqlen, d_head
    # Time indices for causal masking.
    t = jax.lax.tie_in(q, np.arange(seqlen))

    # Split off a "bin" axis for chunks of consecutive items.
    bq_t = np.reshape(t, (n_bins, -1))
    bq = np.reshape(q, (q.shape[0], n_bins, -1, q.shape[-1]))
    if self._share_qk:
      bk = self.make_unit_length(bq)
    else:
      bk = np.reshape(k, (k.shape[0], n_bins, -1, k.shape[-1]))
    bv = np.reshape(v, (v.shape[0], n_bins, -1, v.shape[-1]))

    # Allow each chunk to attend within itself, and also one chunk back.
    def look_one_back(x):
      # Output: pairs [ bin_i bin_{i-1} ] concatenated on the time axis.
      if len(x.shape) == 2:
        x_extra = np.concatenate([x[-1:, :], x[:-1, :]], axis=0)
        return np.concatenate([x, x_extra], axis=1)
      else:
        assert len(x.shape) == 4
        x_extra = np.concatenate([x[:, -1:, :, :], x[:, :-1, :, :]], axis=1)
        return np.concatenate([x, x_extra], axis=2)

    bkv_t = look_one_back(bq_t)
    bk = look_one_back(bk)
    bv = look_one_back(bv)

    # Dot-product attention.
    dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

    # Causal masking based on the time indices.
    mask = jax.lax.convert_element_type(
        jax.lax.lt(bq_t[None, :, :, None], bkv_t[None, :, None, :]),
        np.float32)
    dots = dots - 1e9 * mask

    # Mask out attention to self except when no other targets are available.
    if self._share_qk:
      self_mask = np.eye(dots.shape[2], dots.shape[3])
      self_mask = self_mask[np.newaxis, np.newaxis, :, :]
      self_mask = jax.lax.tie_in(dots, self_mask)
      dots = dots - 1e5 * self_mask

    # Softmax.
    dots = np.exp(dots - math.logsumexp(dots, axis=-1, keepdims=True))

    if self.dropout > 0.0:
      # Dropout is broadcast across the batch+head dimension
      dropout_shape = (1, dots.shape[-3], dots.shape[-2], dots.shape[-1])
      keep_prob = jax.lax.tie_in(dots, 1.0 - self.dropout)
      keep = math.random.bernoulli(rng, keep_prob, dropout_shape)
      multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
      dots = dots * multiplier

    bo = np.matmul(dots, bv)

    output = np.reshape(bo, (bo.shape[0], -1, bo.shape[-1]))
    assert output.shape == v.shape
    return output[..., :original_len, :]

  def _forward_predict(self, inputs, state, rng):
    if not self._share_qk:
      state = _fast_inference_update_state(inputs, state)
      (q, _, _) = inputs
      (ks, vs, mask, seq_indices) = state
    else:
      mask_excluding_attention_in_place = state[2]
      (q, _, v) = inputs
      k = self.make_unit_length(q)
      state = _fast_inference_update_state((q, k, v), state)
      (ks, vs, mask, seq_indices) = state
      # Only the initial position in a sequence may attend to itself.
      where = (seq_indices > 1)[None, None]
      mask = np.where(where, mask_excluding_attention_in_place, mask)

    output = attention.DotProductAttention(
        q, ks, vs, mask, dropout=self.dropout, mode=self._mode, rng=rng
    )

    def roll_single_seq(state):
      """Rolls the buffers backward to make space for new data.

      Works for just one sequence in a batch.

      Args:
        state: Tuple (keys, values, mask, index).

      Returns:
        New state for a single sequence.
      """
      (ks, vs, mask, index) = state
      # Move the second bin into the first one's place in both buffers.
      def roll_buffer(buf):
        return jax.ops.index_update(
            buf,
            jax.ops.index[:self.bin_length, :],
            buf[self.bin_length:, :],
        )
      (ks, vs) = map(roll_buffer, (ks, vs))
      # Zero out the second bin in the mask.
      mask = jax.ops.index_update(
          mask, jax.ops.index[:, self.bin_length:], 0
      )
      # Update the index to match the rolled buffers.
      index -= self.bin_length
      return (ks, vs, mask, index)

    @jax.vmap
    def maybe_roll_state(state):
      """Rolls the buffers if they reach the end.

      Vectorized to handle batches of sequences.

      Args:
        state: Tuple (keys, values, mask, index).

      Returns:
        New state for a batch of sequences.
      """
      (_, _, _, index) = state
      # Once we get to the end of the buffer, move the second bin back to make
      # space for new data: [ bin_i bin_{i+1} | ] -> [ bin_{i+1} | bin_{i+1} ],
      # where | is where index points at in the buffer.
      return jax.lax.cond(
          pred=(index == 2 * self.bin_length),
          true_operand=state,
          true_fun=roll_single_seq,
          false_operand=state,
          false_fun=(lambda x: x),
      )
    state = maybe_roll_state(state)
    return (output, state)

  def new_weights_and_state(self, input_signature):
    if self._mode in ('train', 'eval'):
      return base.EMPTY_WEIGHTS, base.EMPTY_STATE

    assert self._mode == 'predict'
    assert self.bin_length is not None, (
        'For fast inference, TimeBinCausalAttention must be parameterized by '
        'bin_length.'
    )
    weights = base.EMPTY_WEIGHTS
    state = _fast_inference_init_state(
        input_signature, 2 * self.bin_length
    )
    return weights, state


class LSHCausalAttention(attention.BaseCausalAttention):
  """Causal attention based on locality-sensitive hashing."""

  def __init__(self,
               dropout,
               mode,
               n_bins=64,
               n_hashes=1,
               n_buckets=64,
               one_rng=False,
               allow_duplicate_attention=False,
               attend_across_buckets=False,
               hard_k=0,
               factorize_hash=False,
               rehash_each_round=True,
               drop_for_hash_rate=0.0,
               data_rotation=False,
               data_rotation_farthest=False,
               data_rotation_farthest_num=8,
               max_len_for_inference=16384,
               bucket_capacity_for_inference=256):
    # TODO(kitaev): use shared bucket length config that's shared for train+eval
    super(LSHCausalAttention, self).__init__(mode=mode)
    self._mode = mode
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self._dropout = dropout
    else:
      self._dropout = 0.0

    assert n_buckets >= n_bins, 'This setting is not recommended: too few bins.'
    assert rehash_each_round or allow_duplicate_attention, (
        'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
        ' is not implemented.')
    self.n_bins = n_bins
    self.n_hashes = n_hashes
    self.n_buckets = n_buckets
    self._drop_for_hash_rate = drop_for_hash_rate
    self._one_rng = one_rng
    self._factorize_hash = factorize_hash
    self._prng = None
    if one_rng:
      seed = random.randint(0, 2**31 - 1)
      self._prng = math.random.get_prng(seed)

    self._allow_duplicate_attention = allow_duplicate_attention
    self._attend_across_buckets = attend_across_buckets
    self._hard_k = hard_k
    self._rehash_each_round = rehash_each_round
    # If True, the rotation matrices for hashing are derived from data, instead
    # of being completely random.
    self._data_rotation = data_rotation
    self._data_rotation_farthest = data_rotation_farthest
    self._data_rotation_farthest_num = data_rotation_farthest_num

    self._max_len_for_inference = max_len_for_inference
    self._bucket_capacity_for_inference = bucket_capacity_for_inference

  def forward_with_state(self, inputs, weights=base.EMPTY_WEIGHTS,
                         state=base.EMPTY_STATE, rng=None, **kwargs):
    del weights, kwargs
    if self._mode == 'predict':
      output, state = self.batch_predict(inputs[0], inputs[2], state, rng=rng)
    else:
      output, state, _ = self.batch_call_and_or_grad(
          inputs[0], inputs[2], new_state=None, return_state=True, rng=rng)
    return output, state

  def forward_and_backward(self, inputs, ct, state=base.EMPTY_STATE,
                           new_state=base.EMPTY_STATE, rng=None, **kwargs):
    del kwargs
    assert self._mode != 'predict'
    output, _, (qk_ct, v_ct) = self.batch_call_and_or_grad(
        inputs[0], inputs[2], ct=ct, new_state=new_state, rng=rng)
    return output, (qk_ct, np.zeros_like(inputs[1]), v_ct)

  @property
  def has_backward(self):
    return True

  def backward(self, inputs, output, ct, weights=base.EMPTY_WEIGHTS,
               state=base.EMPTY_STATE, new_state=base.EMPTY_STATE, rng=None,
               **kwargs):
    del output, weights, state
    assert self._mode != 'predict'
    _, _, (qk_ct, v_ct) = self.batch_call_and_or_grad(
        inputs[0], inputs[2], return_output=False,
        ct=ct, new_state=new_state, rng=rng)
    inputs_ct = (qk_ct, np.zeros_like(inputs[1]), v_ct)
    return inputs_ct, ()

  def new_weights_and_state(self, input_signature):
    qk = input_signature[0]
    if self._mode != 'predict':
      state = np.zeros(
          (qk.shape[0], self.n_hashes * qk.shape[1]), dtype=np.int32)
    else:
      # Having separate key/value caches for each hashing round would use a
      # lot of memory, so instead each hashing round stores indices into a
      # single key/value cache. Even with this approach, the maximum sequence
      # length that fits in memory is shorter for fast inference than training.
      # Fast inference is "fast" in that it avoids recomputation, which is the
      # exact opposite tradeoff of memory-saving tricks like reversibility.
      # There is still some potential room for memory savings by caching
      # activations rather than key-value pairs (currently not implemented
      # because it would require changes outside of this class).
      batch_size = input_signature[0].shape[0]
      max_len = self._max_len_for_inference
      bucket_capacity = self._bucket_capacity_for_inference
      d_qk = input_signature[0].shape[-1]
      d_v = input_signature[2].shape[-1]
      dtype = input_signature[0].dtype

      ks = np.zeros((batch_size, max_len, d_qk), dtype=dtype)
      vs = np.zeros((batch_size, max_len, d_v), dtype=dtype)
      mask = np.full((batch_size, max_len), -1e9, dtype=dtype)
      index = 0
      bucket_assignments = np.full(
          (batch_size * self.n_hashes * self.n_buckets, bucket_capacity),
          max_len, dtype=np.int32)
      assignment_locs = np.zeros(
          batch_size * self.n_hashes * self.n_buckets, dtype=np.int32)
      hash_rng = self.new_rng()

      state = (
          ks, vs, mask, index, bucket_assignments, assignment_locs, hash_rng)

    return self.new_weights(input_signature), state

  def batch_call_and_or_grad(self, qk, v, ct=None, return_output=True,
                             new_state=None, return_state=False,
                             rng=None):
    assert return_output or ct is not None, 'No work to perform!'
    if new_state is not None and new_state is not base.EMPTY_STATE:
      buckets = new_state
    else:
      buckets = None

    # The approach here is to perform attention for one batch element and head
    # at a time. Note that there is absolutely no interaction across examples or
    # heads: this layer has no parameters, and hashing patterns are also
    # different across examples/heads. As a result, batching doesn't give any
    # performance gains except in the case of accelerator under-utilization. We
    # assume that hash-based attention will be applied primarily to long
    # sequences, where unbatched attention for a single head has sufficient
    # computation to fill up the accelerator.

    batch_loop_idx = np.zeros((), dtype=np.int32)
    batch_loop_max = qk.shape[0]

    init_vals = (batch_loop_idx,)
    if return_output:
      out_accum = np.zeros_like(qk)
      init_vals = init_vals + (out_accum,)
    if return_state:
      buckets_accum = np.zeros(
          [qk.shape[0], self.n_hashes * qk.shape[1]], dtype=np.int32)
      init_vals = init_vals + (buckets_accum,)
    if ct is not None:
      qk_ct_accum = np.zeros_like(qk)
      v_ct_accum = np.zeros_like(v)
      init_vals = init_vals + (qk_ct_accum, v_ct_accum)

    def cond_fun(vals):
      batch_loop_idx = vals[0]
      return jax.lax.lt(batch_loop_idx, batch_loop_max)

    def body_fun(vals):
      """Performs attention for a single batch element and head."""
      batch_loop_idx = vals[0]
      if self._prng is None:
        hash_slice_rng = jax.random.fold_in(rng, batch_loop_idx)
        hash_rng, slice_rng = math.random.split(hash_slice_rng)
      else:
        # TODO(kitaev): Maybe use the same RNG across examples (but not heads)?
        hash_rng = jax.random.fold_in(self._prng, batch_loop_idx)
        slice_rng = jax.random.fold_in(rng, batch_loop_idx)
      qk_slice = jax.lax.dynamic_index_in_dim(
          qk, batch_loop_idx, axis=0, keepdims=False)
      v_slice = jax.lax.dynamic_index_in_dim(
          v, batch_loop_idx, axis=0, keepdims=False)

      if buckets is None:
        buckets_slice = self.hash_vectors(qk_slice, rng=hash_rng)
      else:
        buckets_slice = jax.lax.dynamic_index_in_dim(
            buckets, batch_loop_idx, axis=0, keepdims=False)

      if ct is None:
        out_slice = self.single_call(
            qk_slice, v_slice, buckets_slice, rng=slice_rng)
      else:
        def _do_single_call(qk_slice, v_slice):
          return self.single_call(
              qk_slice, v_slice, buckets_slice, rng=slice_rng)
        ct_slice = jax.lax.dynamic_index_in_dim(
            ct, batch_loop_idx, axis=0, keepdims=False)
        out_slice, vjpfun = jax.vjp(_do_single_call, qk_slice, v_slice)
        qk_ct_slice, v_ct_slice = vjpfun(ct_slice)

      new_vals = (batch_loop_idx + 1,)
      if return_output:
        out_accum = vals[1]
        out_accum = jax.lax.dynamic_update_index_in_dim(
            out_accum, out_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (out_accum,)
      if return_state:
        buckets_accum = vals[2]
        buckets_accum = jax.lax.dynamic_update_index_in_dim(
            buckets_accum, buckets_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (buckets_accum,)
      if ct is not None:
        qk_ct_accum, v_ct_accum = vals[-2:]
        qk_ct_accum = jax.lax.dynamic_update_index_in_dim(
            qk_ct_accum, qk_ct_slice, batch_loop_idx, axis=0)
        v_ct_accum = jax.lax.dynamic_update_index_in_dim(
            v_ct_accum, v_ct_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (qk_ct_accum, v_ct_accum)

      return new_vals

    final_vals = jax.lax.while_loop(cond_fun, body_fun, init_vals)

    if return_output:
      out = final_vals[1]
    else:
      out = None

    if return_state:
      state = final_vals[2]
    else:
      state = None

    if ct is not None:
      input_ct = final_vals[-2:]
    else:
      input_ct = None

    return out, state, input_ct

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def drop_for_hash(self, x, rng):
    rate = self._drop_for_hash_rate
    if self._mode == 'train' and rate > 0.0:
      keep = math.random.bernoulli(rng, 1.0 - rate, x.shape)
      return np.where(keep, x / (1.0 - rate), np.zeros_like(x))
    return x

  def _sample_rotation(self, shape, vecs, rng):
    """Samples a rotation matrix, either randomly or based on `vecs`."""

    if not self._data_rotation:
      return jax.random.normal(rng, shape).astype('float32')

    assert len(shape) == 3
    unused_n_dim, n_hashes, r_div_2 = shape

    assert len(vecs.shape) == 2
    n_vecs = vecs.shape[0]

    rng1, rng2 = math.random.split(rng, num=2)

    # We need to sample 2 * n_hashes * r_div_2 vectors from `vecs` at random.
    num_needed = 2 * n_hashes * r_div_2
    if n_vecs < num_needed:
      # shape = (n_hashes, r_div_2)
      random_idxs_1 = jax.random.randint(
          rng1, (n_hashes, r_div_2), 0, n_vecs)
      random_idxs_2 = jax.random.randint(
          rng2, (n_hashes, r_div_2), 0, n_vecs)
    else:
      # Sample without replacement.
      shuffled_indices = jax.random.shuffle(rng1, np.arange(n_vecs))
      random_idxs = np.reshape(shuffled_indices[:num_needed],
                               (2, n_hashes, r_div_2))
      random_idxs_1 = random_idxs[0]
      random_idxs_2 = random_idxs[1]

    if self._data_rotation_farthest:
      # shape = (n_hashes * r_div_2, )
      random_idxs_1 = np.reshape(random_idxs_1, (-1,))
      random_vecs_1 = vecs[random_idxs_1]

      # Sample candidates for vec2s.
      rng, subrng = math.random.split(rng)
      # shape = (self._data_rotation_farthest_num, n_hashes * r_div_2)
      candidate_idxs_2 = jax.random.randint(
          subrng, (self._data_rotation_farthest_num, n_hashes * r_div_2), 0,
          n_vecs)
      candidate_vecs_2 = vecs[candidate_idxs_2]
      # shape = candidate_idxs_2.shape
      distances = -np.abs(
          np.einsum('hd,chd->ch', random_vecs_1, candidate_vecs_2))
      # shape = (n_hashes * r_div_2,)
      farthest_idxs = np.argmax(distances, axis=0)
      # candidate_vecs_2.shape
      random_vecs_2 = candidate_vecs_2[farthest_idxs,
                                       np.arange(n_hashes * r_div_2)]

      # reshape to (n_hashes, r_div_2, n_dim)
      random_vecs_1 = np.reshape(random_vecs_1, (n_hashes, r_div_2, -1))
      random_vecs_2 = np.reshape(random_vecs_2, (n_hashes, r_div_2, -1))
    else:
      # shape = (n_hashes, r_div_2, n_dim)
      random_vecs_1 = vecs[random_idxs_1]
      random_vecs_2 = vecs[random_idxs_2]

    # shape = (n_dim, n_hashes, r_div_2)
    return np.transpose(random_vecs_2 - random_vecs_1, axes=[2, 0, 1])

  def hash_vectors(self, vecs, rng):
    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    assert self.n_buckets % 2 == 0

    # If we factorize the hash, find a factor dividing n_buckets nicely.
    rot_size, factor_list = self.n_buckets, [self.n_buckets]
    if self._factorize_hash:
      # If we are given a list of factors, verify it and use later.
      if isinstance(self._factorize_hash, list):
        rot_size, product = 0, 1
        factor_list = self._factorize_hash
        for factor in factor_list:
          assert factor % 2 == 0
          product *= factor
          rot_size += factor
        assert product == self.n_buckets
      else:  # Find one factor if just set to True.
        # We want to represent self.n_buckets = factor * rest so that
        # (1) both factor and rest are even, and (2) factor + rest is minimal.
        # To compute this we start from factor = sqrt(n_buckets) and go down
        # with it until we find one that satisfies the constraints above.
        factor = int(python_math.sqrt(self.n_buckets))
        while factor > 0 and not (
            self.n_buckets % factor == 0 and
            factor % 2 == 0 and
            (self.n_buckets // factor) % 2 == 0):
          factor -= 1
        if factor > 2:  # Factor of 2 does not warrant the effort.
          rot_size = factor + (self.n_buckets // factor)
          factor_list = [factor, self.n_buckets // factor]

    rotations_shape = (
        vecs.shape[-1],
        self.n_hashes if self._rehash_each_round else 1,
        rot_size // 2)

    rng = jax.lax.tie_in(vecs, rng)
    rng, subrng = math.random.split(rng)
    random_rotations = self._sample_rotation(rotations_shape, vecs, rng)

    # TODO(lukaszkaiser): the dropout mask will be used for all rounds of
    # hashing, so it's shared between them. Check if that's what we want.
    dropped_vecs = self.drop_for_hash(vecs, subrng)
    rotated_vecs = np.einsum('tf,fhb->htb', dropped_vecs, random_rotations)

    if self._rehash_each_round:
      if self._factorize_hash and len(factor_list) > 1:
        # We factorized self.n_buckets as the product of factor_list.
        # Get the buckets for them and combine.
        buckets, cur_sum, cur_product = None, 0, 1
        for factor in factor_list:
          rv = rotated_vecs[..., cur_sum:cur_sum + (factor // 2)]
          cur_sum += factor // 2
          rv = np.concatenate([rv, -rv], axis=-1)
          if buckets is None:
            buckets = np.argmax(rv, axis=-1)
          else:
            buckets += cur_product * np.argmax(rv, axis=-1)
          cur_product *= factor
      else:
        rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
        buckets = np.argmax(rotated_vecs, axis=-1)
      # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
      # bucket numbers from different hashing rounds don't overlap.
      offsets = jax.lax.tie_in(buckets, np.arange(self.n_hashes))
      offsets = np.reshape(offsets * self.n_buckets, (-1, 1))
      buckets = np.reshape(buckets + offsets, (-1,))
    else:
      assert not self._factorize_hash
      rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
      # In this configuration, we map each item to the top self.n_hashes buckets
      rotated_vecs = np.squeeze(rotated_vecs, 0)
      bucket_range = jax.lax.tie_in(vecs, np.arange(rotated_vecs.shape[-1]))
      bucket_range = np.reshape(bucket_range, (1, -1))
      bucket_range = np.broadcast_to(bucket_range, rotated_vecs.shape)

      _, buckets = jax.lax.sort_key_val(
          rotated_vecs, bucket_range, dimension=-1)
      buckets = buckets[:, -self.n_hashes:]
      buckets = np.reshape(np.moveaxis(buckets, 0, -1), (-1,))

    return buckets

  def single_call(self, qk, v, buckets, rng=None):
    # We use the same vector as both a query and a key.
    seqlen = qk.shape[-2]
    assert int(buckets.shape[0]) == self.n_hashes * seqlen

    ticker = jax.lax.tie_in(qk, np.arange(self.n_hashes * seqlen))
    buckets_and_t = seqlen * buckets + (ticker % seqlen)
    buckets_and_t = jax.lax.stop_gradient(buckets_and_t)

    # Hash-based sort ("s" at the start of variable names means "sorted")
    sbuckets_and_t, sticker = jax.lax.sort_key_val(
        buckets_and_t, ticker, dimension=-1)
    _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
    sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
    sticker = jax.lax.stop_gradient(sticker)
    undo_sort = jax.lax.stop_gradient(undo_sort)

    st = (sticker % seqlen)
    sqk = np.take(qk, st, axis=0)
    sv = np.take(v, st, axis=0)

    # Split off a "bin" axis so that attention only occurs within chunks.
    bq_t = bkv_t = np.reshape(st, (self.n_hashes * self.n_bins, -1))
    bqk = np.reshape(sqk, (self.n_hashes * self.n_bins, -1, sqk.shape[-1]))
    bv = np.reshape(sv, (self.n_hashes * self.n_bins, -1, sv.shape[-1]))
    bq_buckets = bkv_buckets = np.reshape(
        sbuckets_and_t // seqlen, (self.n_hashes * self.n_bins, -1))

    # Hashing operates on unit-length vectors. Unnormalized query vectors are
    # fine because they effectively provide a learnable temperature for the
    # attention softmax, but normalizing keys is needed so that similarity for
    # the purposes of attention correctly corresponds to hash locality.
    bq = bqk
    bk = self.make_unit_length(bqk)

    # Allow each chunk to attend within itself, and also one chunk back. Chunk
    # boundaries might occur in the middle of a sequence of items from the
    # same bucket, so this increases the chances of attending to relevant items.
    # TODO(kitaev): benchmark whether XLA pad operation is noticeably faster.
    def look_one_back(x):
      if len(x.shape) == 2:
        x_extra = np.concatenate([x[-1:, :], x[:-1, :]], axis=0)
      else:
        x_extra = np.concatenate([x[-1:, :, :], x[:-1, :, :]], axis=0)
      return np.concatenate([x, x_extra], axis=1)

    bk = look_one_back(bk)
    bv = look_one_back(bv)
    bkv_t = look_one_back(bkv_t)
    bkv_buckets = look_one_back(bkv_buckets)

    # Dot-product attention.
    dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

    # Causal masking
    mask = jax.lax.convert_element_type(
        jax.lax.lt(bq_t[:, :, None], bkv_t[:, None, :]),
        np.float32)
    dots = dots - 1e9 * mask

    # Mask out attention to self except when no other targets are available.
    self_mask = jax.lax.convert_element_type(
        jax.lax.eq(bq_t[:, :, None], bkv_t[:, None, :]),
        np.float32)
    dots = dots - 1e5 * self_mask

    # Mask out attention to other hash buckets.
    if not self._attend_across_buckets:
      bucket_mask = jax.lax.convert_element_type(
          jax.lax.ne(bq_buckets[:, :, None], bkv_buckets[:, None, :]),
          np.float32)
      dots = dots - 1e7 * bucket_mask

    # Don't double-count query-key pairs across multiple rounds of hashing.
    # There are two possible strategies here. (1) The default is to count how
    # many times a query-key pair is repeated, and to lower its log-prob
    # correspondingly at each repetition. (2) When hard_k is set, the code
    # instead masks all but the first occurrence of each query-key pair.
    # TODO(kitaev): is one strategy faster or more numerically stable?
    if not self._allow_duplicate_attention:
      locs1 = undo_sort // bq_t.shape[-1]
      locs2 = (locs1 + 1) % (self.n_hashes * self.n_bins)
      if not self._attend_across_buckets:
        locs1 = buckets * (self.n_hashes * self.n_bins) + locs1
        locs2 = buckets * (self.n_hashes * self.n_bins) + locs2
      locs = np.moveaxis(np.concatenate([
          np.reshape(locs1, (self.n_hashes, seqlen)),
          np.reshape(locs2, (self.n_hashes, seqlen)),
      ], 0), 0, -1)  # produces shape (seqlen, 2 * self.n_hashes)
      slocs = np.take(locs, st, axis=0)
      b_locs = np.reshape(
          slocs, (self.n_hashes * self.n_bins, -1, 2 * self.n_hashes))
      # Queries always use the primary location (based on locs1).
      b_locs1 = b_locs[:, :, None, :self.n_hashes]
      if self._hard_k > 0:
        range_n_hashes = jax.lax.tie_in(b_locs, np.arange(self.n_hashes))
        nouse_locs = (range_n_hashes[:, None] > range_n_hashes[None, :])
        nouse_locs = 2 * nouse_locs - 1  # 1 = use, -1 = don't use
        nouse_locs = np.reshape(
            np.broadcast_to(nouse_locs[:, None, :],
                            (self.n_hashes, self.n_bins, self.n_hashes)),
            (self.n_hashes * self.n_bins, 1, 1, self.n_hashes))
        b_locs1 = b_locs1 * nouse_locs
      bq_locs = np.broadcast_to(
          b_locs1,
          b_locs.shape[:2] + (2, self.n_hashes))
      bq_locs = np.reshape(bq_locs, b_locs.shape)
      bkv_locs = look_one_back(b_locs)

      dup_counts = np.sum(
          jax.lax.convert_element_type(
              jax.lax.eq(bq_locs[:, :, None, :], bkv_locs[:, None, :, :]),
              np.float32),
          axis=-1)
      assert dup_counts.shape == dots.shape
      if self._hard_k > 0:
        dots = dots - 1e7 * jax.lax.stop_gradient(dup_counts)
      else:
        dots = dots - jax.lax.stop_gradient(np.log(dup_counts + 1e-9))

    # Each query only attends to the top k most relevant keys.
    if self._hard_k > 0:
      b_top_dots = np.sort(dots)[..., -self._hard_k:]  # Get the top k dots.
      b_top_dots = jax.lax.stop_gradient(b_top_dots)
      s_top_dots = np.reshape(b_top_dots, (-1, self._hard_k))
      top_dots = np.take(s_top_dots, undo_sort, axis=0)

      merged_top_dots = np.moveaxis(
          np.reshape(top_dots, (self.n_hashes, seqlen, self._hard_k)), 0, -1)
      merged_top_dots = np.reshape(merged_top_dots, (seqlen, -1))

      dots_thresh = np.sort(merged_top_dots)[:, -self._hard_k]
      # It's possible to compute the partition function at this point, but right
      # now this codepath isn't set up for backprop, and there might also be
      # issues computing it this way if two dot-products are exactly equal.

      sdots_thresh = dots_thresh[st]
      bdots_thresh = np.reshape(sdots_thresh, (self.n_hashes * self.n_bins, -1))
      bdots_thresh = jax.lax.stop_gradient(bdots_thresh)

      top_k_mask = jax.lax.convert_element_type(
          dots < bdots_thresh[..., None], np.float32)
      dots = dots - 1e7 * jax.lax.stop_gradient(top_k_mask)

    # Softmax.
    dots_logsumexp = math.logsumexp(dots, axis=-1, keepdims=True)
    dots = np.exp(dots - dots_logsumexp)

    if self._dropout > 0.0:
      # Dropout is broadcast across the bin dimension
      dropout_shape = (1, dots.shape[-2], dots.shape[-1])
      keep_prob = jax.lax.tie_in(dots, 1.0 - self._dropout)
      keep = math.random.bernoulli(rng, keep_prob, dropout_shape)
      multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
      dots = dots * multiplier

    bo = np.matmul(dots, bv)
    so = np.reshape(bo, (-1, bo.shape[-1]))
    slogits = np.reshape(dots_logsumexp, (-1,))

    def unsort_for_output_impl(so, slogits):
      o = np.take(so, undo_sort, axis=0)
      # Sorting is considerably faster than gather, but first we need to get the
      # XLA compiler to abandon the idea of fusing this sort with the input sort
      # (which introduces a computation cycle and leads to a crash).
      # TODO(kitaev): remove "sticker_" variable if XLA is fixed.
      sticker_ = sticker + jax.lax.convert_element_type(
          slogits[0] > 0, sticker.dtype)
      _, logits = jax.lax.sort_key_val(sticker_, slogits, dimension=-1)
      return o, logits

    def unsort_for_output_vjp(so, slogits):
      """Custom gradient for unsort_for_output."""
      so = jax.lax.stop_gradient(so)
      slogits = jax.lax.stop_gradient(slogits)
      o, logits = unsort_for_output_impl(so, slogits)
      def vjpfun(o_logits_grads):
        so_grad = np.take(o_logits_grads[0], sticker, axis=0)
        # TODO(kitaev): this exists to match the forward pass, but I'm not sure
        # if it's actually required.
        buckets_and_t_ = buckets_and_t + jax.lax.convert_element_type(
            o_logits_grads[1][0] > 0, buckets_and_t.dtype)
        _, slogits_grad = jax.lax.sort_key_val(
            buckets_and_t_, o_logits_grads[1], dimension=-1)
        return (so_grad, slogits_grad)
      return (o, logits), vjpfun

    unsort_for_output = jax.custom_transforms(unsort_for_output_impl)
    jax.defvjp_all(unsort_for_output, unsort_for_output_vjp)
    o, logits = unsort_for_output_impl(so, slogits)

    if self.n_hashes == 1:
      out = o
    else:
      o = np.reshape(o, (self.n_hashes, seqlen, o.shape[-1]))
      logits = np.reshape(logits, (self.n_hashes, seqlen, 1))
      probs = np.exp(logits - math.logsumexp(logits, axis=0, keepdims=True))
      out = np.sum(o * probs, axis=0)

    assert out.shape == v.shape
    return out

  def batch_predict(self, qk, v, state, rng=None):
    assert not self._data_rotation, (
        'Fast inference with data-dependent rotation is unsupported.')
    assert self._hard_k == 0, 'Fast inference with hard_k is not implemented.'
    assert self._dropout == 0.0, (
        'Fast inference with dropout is not implemented.')

    (ks, vs, mask, index, bucket_assignments, assignment_locs, hash_rng) = state

    # TODO(kitaev): separate random projection for each attention head
    assert qk.shape[1] == 1
    batch_size = qk.shape[0]

    buckets = np.reshape(
        self.hash_vectors(np.squeeze(qk, 1), hash_rng),
        (self.n_hashes, batch_size))
    buckets = np.swapaxes(buckets, 0, 1)

    k = self.make_unit_length(qk)
    ks = jax.ops.index_update(ks, jax.ops.index[:, index, :], k[:, 0, :])
    vs = jax.ops.index_update(vs, jax.ops.index[:, index, :], v[:, 0, :])
    # Mask out attention to self except when no other targets are available.
    # Invalid elements are masked at -1e9 strength, rather than -1e5.
    cur_mask = jax.ops.index_update(mask, jax.ops.index[:, index], -1e5)
    mask = jax.ops.index_update(mask, jax.ops.index[:, index], 0.0)

    # Update bucket_assignments and assignment_locs.
    batch_idxs = np.broadcast_to(
        np.reshape(np.arange(batch_size), (-1, 1)), buckets.shape)
    batch_offsets = batch_idxs * self.n_hashes * self.n_buckets
    batch_bucket_idxs = np.reshape(batch_offsets + buckets, (-1,))
    update_locs = assignment_locs[batch_bucket_idxs]
    bucket_assignments = jax.ops.index_update(
        bucket_assignments,
        jax.ops.index[batch_bucket_idxs, update_locs],
        np.broadcast_to(index, batch_bucket_idxs.shape))
    assignment_locs = jax.ops.index_update(
        assignment_locs,
        jax.ops.index[batch_bucket_idxs],
        (update_locs + 1) % self._bucket_capacity_for_inference)

    # kv_refs: batch_size, n_kv
    kv_refs = bucket_assignments[batch_bucket_idxs]
    kv_refs = np.reshape(kv_refs, (batch_size, -1))
    kv_refs = np.sort(kv_refs, -1)

    # cur_mask: batch_size, 1, n_kv
    # cur_k: batch_size, n_kv, d_qk
    # cur_v: batch_size, n_kv, d_v
    cur_mask = np.take_along_axis(cur_mask, kv_refs, 1)[:, None, :]
    # The low-level implementation of np.take_along_axis runs out of memory on
    # TPU, so we perform the equivalent with manual index arithmetic.
    # cur_k = np.take_along_axis(ks, kv_refs[:, :, None], 1)
    # cur_v = np.take_along_axis(vs, kv_refs[:, :, None], 1)
    ref_offsets = jax.lax.tie_in(kv_refs, np.arange(batch_size))
    kv_refs_flat = kv_refs + np.reshape(
        ref_offsets, (-1, 1)) * ks.shape[1]
    kv_refs_flat = np.reshape(kv_refs_flat, (-1,))
    cur_k = np.reshape(ks, (-1, ks.shape[-1]))[kv_refs_flat]
    cur_v = np.reshape(vs, (-1, vs.shape[-1]))[kv_refs_flat]
    cur_k = np.reshape(cur_k, (batch_size, -1, ks.shape[-1]))
    cur_v = np.reshape(cur_v, (batch_size, -1, vs.shape[-1]))

    dots = np.matmul(qk, np.swapaxes(cur_k, -1, -2)) / np.sqrt(qk.shape[-1])
    dots = dots + cur_mask

    if not self._allow_duplicate_attention:
      no_repeat_mask = jax.lax.convert_element_type(
          kv_refs == np.pad(
              kv_refs, [[0, 0], [1, 0]], constant_values=-99)[:, :-1],
          np.float32)
      dots = dots - 1e7 * no_repeat_mask[:, None, :]

    # Softmax.
    dots_logsumexp = math.logsumexp(dots, axis=-1, keepdims=True)
    dots = np.exp(dots - dots_logsumexp)

    out = np.matmul(dots, cur_v)  # batch_size, 1, d_v
    assert out.shape == v.shape

    new_state = (
        ks, vs, mask, index + 1, bucket_assignments, assignment_locs, hash_rng)
    return out, new_state
