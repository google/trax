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
"""Attention Layers optimized for efficiency (second-pass implementation).

The approach taken in the first round of efficient attention implementations
revealed several limitations, which this code attempts to address:

1. Simultaneously instantiating queries, keys, and values for all heads can
   exceed the memory budget. Transformers are typically tuned such that
   n_heads * d_attention_key == d_model. Since attention involves queries, keys,
   AND values, the memory to store them can be ~3x the memory needed to store
   the input activations. Once the O(n^2) dot-product bottleneck is removed
   -- as is the case in all of our efficient attention implementations -- this
   becomes the next critical bottleneck for scaling up Transformer models.

2. Attention masking is implemented by associating an integer (typically, the
   sequence position) with each query and key vector, and defining a function
   to compute attention masks from this information. The standard attention API
   (attention.py) is unscalable because it instantiates O(n^2)-size attention
   masks, and the previous efficient implementations (efficient_attention.py)
   only supported causal masking.
"""
import functools
import math
import jax

from trax import fastmath
from trax.fastmath import numpy as np
from trax.layers import base
from trax.layers import initializers as init


####################################################### Functions


def length_normalized(x, epsilon=1e-6):
  variance = np.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / np.sqrt(variance + epsilon)
  return norm_inputs


def hash_vecs(vecs, n_buckets_in, n_hashes, rng):
  """Hash vectors into buckets.

  Args:
    vecs: vectors to hash, a tensor of shape [batch_size, depth]
    n_buckets_in: an int or a list of ints, number of hash buckets;
      if it is a list, we do hierarchical hashing as specified by the list
    n_hashes: number of hashes
    rng: random generator to use for hashing

  Returns:
    A pair (buckets, n_buckets) where buckets is a tensor of shape
    [n_hashes, batch_size] of integers -- the hash bucket IDs, and
    n_buckets is an int, the total number of hash buckets, equal to
    the product of all items in n_buckets_in.
  """
  # See https://arxiv.org/pdf/1509.02897.pdf
  # We sample a different random rotation for each round of hashing to
  # decrease the probability of hash misses.
  if isinstance(n_buckets_in, int):
    assert n_buckets_in % 2 == 0
    rot_size = n_buckets_in
    n_buckets = n_buckets_in
  else:
    # Factorize the hash if n_buckets_in is a list or tuple
    rot_size, n_buckets = 0, 1
    for factor in n_buckets_in:
      assert factor % 2 == 0
      rot_size += factor
      n_buckets *= factor

  rotations_shape = (vecs.shape[-1], n_hashes, rot_size // 2)
  random_rotations = fastmath.random.normal(rng, rotations_shape).astype(
      np.float32)
  if fastmath.is_backend(fastmath.Backend.JAX):
    rotated_vecs = np.einsum('tf,fhb->htb', vecs, random_rotations)
  else:
    random_rotations = np.reshape(random_rotations,
                                  [-1, n_hashes * (rot_size // 2)])
    rotated_vecs = np.dot(vecs, random_rotations)
    rotated_vecs = np.reshape(rotated_vecs, [-1, n_hashes, rot_size//2])
    rotated_vecs = np.transpose(rotated_vecs, (1, 0, 2))

  if isinstance(n_buckets_in, int) or len(n_buckets_in) == 1:
    rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
    buckets = np.argmax(rotated_vecs, axis=-1).astype(np.int32)
  else:
    # Get the buckets for them and combine.
    buckets, cur_sum, cur_product = None, 0, 1
    for factor in n_buckets_in:
      rv = rotated_vecs[..., cur_sum:cur_sum + (factor // 2)]
      cur_sum += factor // 2
      rv = np.concatenate([rv, -rv], axis=-1)
      if buckets is None:
        buckets = np.argmax(rv, axis=-1).astype(np.int32)
      else:
        buckets += cur_product * np.argmax(rv, axis=-1).astype(np.int32)
      cur_product *= factor

  return buckets, n_buckets  # buckets is now (n_hashes, batch_size)


def look_adjacent(x, n_chunks_before, n_chunks_after):
  """Used to implement attention between consecutive chunks.

  Args:
    x: array of shape [n_chunks, chunk_len, ...]
    n_chunks_before: Number of previous chunks to attend to.
    n_chunks_after: Number of subsequent chunks to attend to.
  Returns:
    array of shape [n_chunks, N * chunk_len, ...], where
    N = (1 + n_chunks_before + n_chunks_after).
  """
  if n_chunks_before == 0 and n_chunks_after == 0:
    return x

  slices = []
  for i in range(-n_chunks_before, n_chunks_after + 1):
    if i == 0:
      slices.append(x)
    else:
      slices.append(np.concatenate([x[i:, ...], x[:i, ...]], axis=0))
  return np.concatenate(slices, axis=1)


def mask_self_attention(
    dots, q_info, kv_info, causal=True, exclude_self=True, masked=False):
  """Performs masking for self-attention."""
  q_info = q_info.astype(np.float32)
  kv_info = kv_info.astype(np.float32)
  if causal:
    mask = fastmath.lt(q_info, kv_info)
    dots = dots - 1e9 * mask
  if exclude_self:
    mask = np.equal(q_info, kv_info)
    dots = dots - 1e5 * mask
  if masked:
    zeros_like_kv_info = np.zeros_like(kv_info)
    mask = fastmath.lt(kv_info, zeros_like_kv_info).astype(np.float32)
    dots = dots - 1e9 * mask
  return dots


def attend(
    q, k=None, v=None,
    q_chunk_len=None, kv_chunk_len=None,
    n_chunks_before=0, n_chunks_after=0,
    mask_fn=None, q_info=None, kv_info=None,
    dropout=0.0, rng=None,
    ):
  """Dot-product attention, with optional chunking and/or masking.

  Args:
    q: Query vectors, shape [q_len, d_qk]
    k: Key vectors, shape [kv_len, d_qk]; or None
    v: Value vectors, shape [kv_len, d_v]
    q_chunk_len: Set to non-zero to enable chunking for query vectors
    kv_chunk_len: Set to non-zero to enable chunking for key/value vectors
    n_chunks_before: Number of adjacent previous chunks to attend to
    n_chunks_after: Number of adjacent subsequent chunks to attend to
    mask_fn: TODO(kitaev) doc
    q_info: Query-associated metadata for masking
    kv_info: Key-associated metadata for masking
    dropout: Dropout rate
    rng: RNG for dropout

  Returns:
    A tuple (output, dots_logsumexp). The output has shape [q_len, d_v], and
    dots_logsumexp has shape [q_len]. The logsumexp of the attention
    probabilities is useful for combining multiple rounds of attention (as in
    LSH attention).
  """
  assert v is not None
  share_qk = (k is None)

  # `q_info` and `kv_info` if supplied are 0 indexed, we want them to be 1
  # indexed instead so that we can mask position 0 as well - see Github #820

  if q_info is None:
    q_info = np.arange(1, q.shape[-2] + 1, dtype=np.int32)
  else:
    q_info += 1

  if kv_info is None and not share_qk:
    kv_info = np.arange(1, v.shape[-2] + 1, dtype=np.int32)
  elif kv_info is not None:
    kv_info += 1

  # Split q/k/v into chunks along the time axis, if desired.
  if q_chunk_len is not None:
    q = np.reshape(q, (-1, q_chunk_len, q.shape[-1]))
    q_info = np.reshape(q_info, (-1, q_chunk_len))

  if share_qk:
    assert kv_chunk_len is None or kv_chunk_len == q_chunk_len
    k = q
    kv_chunk_len = q_chunk_len
    if kv_info is None:
      kv_info = q_info
    elif kv_chunk_len is not None:
      # kv_info is not None, but reshape as required.
      kv_info = np.reshape(kv_info, (-1, kv_chunk_len))
  elif kv_chunk_len is not None:
    k = np.reshape(k, (-1, kv_chunk_len, k.shape[-1]))
    kv_info = np.reshape(kv_info, (-1, kv_chunk_len))

  if kv_chunk_len is not None:
    v = np.reshape(v, (-1, kv_chunk_len, v.shape[-1]))

  if share_qk:
    k = length_normalized(k)
  k = k / np.sqrt(k.shape[-1])

  # Optionally include adjacent chunks.
  if q_chunk_len is not None or kv_chunk_len is not None:
    assert q_chunk_len is not None and kv_chunk_len is not None
  else:
    assert n_chunks_before == 0 and n_chunks_after == 0

  k = look_adjacent(k, n_chunks_before, n_chunks_after)
  v = look_adjacent(v, n_chunks_before, n_chunks_after)
  kv_info = look_adjacent(kv_info, n_chunks_before, n_chunks_after)

  # Dot-product attention.
  dots = np.matmul(q, np.swapaxes(k, -1, -2))

  # Masking
  if mask_fn is not None:
    dots = mask_fn(dots, q_info[..., :, None], kv_info[..., None, :])

  # Softmax.
  dots_logsumexp = fastmath.logsumexp(dots, axis=-1, keepdims=True)
  dots = np.exp(dots - dots_logsumexp)

  if dropout > 0.0:
    assert rng is not None
    # Dropout is broadcast across the bin dimension
    dropout_shape = (dots.shape[-2], dots.shape[-1])
    # TODO(kitaev): verify that tie-in is safe to remove (in light of jax fix)
    keep_prob = 1.0 - dropout
    keep = fastmath.random.bernoulli(rng, keep_prob, dropout_shape)
    multiplier = keep.astype(dots.dtype) / keep_prob
    dots = dots * multiplier

  # The softmax normalizer (dots_logsumexp) is used by multi-round LSH attn.
  out = np.matmul(dots, v)
  out = np.reshape(out, (-1, out.shape[-1]))
  dots_logsumexp = np.reshape(dots_logsumexp, (-1,))
  return out, dots_logsumexp


def apply_broadcasted_dropout(vecs, dropout_rate, rng):
  """Apply dropout, broadcasted across all but the last dimension of `vecs`."""
  if dropout_rate > 0.0:
    assert rng is not None
    keep_prob = 1.0 - dropout_rate
    keep = fastmath.random.bernoulli(rng, keep_prob, (vecs.shape[-1],))
    multiplier = keep.astype(vecs.dtype) / keep_prob
    return vecs * multiplier
  else:
    return vecs


# The new implementations below don't use custom_transforms in JAX but
# do cause Tracer errors, so we don't use them for now.


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  def permute_impl(p, unused_ip, val):
    return np.take(val, p, axis=axis)
  def permute_fwd(p, ip, val):
    return np.take(val, p, axis=axis), ip
  def permute_bwd(ip, permuted_grad):
    # JAX autodiff would synthesize a scatter operation because it doesn't
    # know that the indices are a permutation. However on TPU, gathers are
    # faster than scatters (at least in the regime the LSH attention uses).
    return (None, None, np.take(permuted_grad, ip, axis=axis))
  permute = fastmath.custom_vjp(permute_impl, permute_fwd, permute_bwd)
  return permute(permutation, inverse_permutation, val)


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  def permute_impl(k, unused_ik, val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = fastmath.sort_key_val(k, val, dimension=axis)
    return permuted
  def permute_fwd(k, ik, val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = fastmath.sort_key_val(k, val, dimension=axis)
    return permuted, ik
  def permute_bwd(ik, permuted_grad):
    _, val_grad = fastmath.sort_key_val(
        ik, permuted_grad, dimension=axis)
    return (None, None, val_grad)
  permute = fastmath.custom_vjp(permute_impl, permute_fwd, permute_bwd)
  return permute(keys, inverse_keys, val)


####################################################### Classes


class EfficientAttentionBase(base.Layer):
  """Base class for efficient attention.

  This is a base class that implements memory-efficient batching for both the
  forward and backward passes. Subclasses should override
  `create_weights_unbatched`, `create_state_unbatched`, `forward_unbatched`, and
  optionally `incremental_forward_unbatched` to define the actual attention
  mechanism.
  """

  def __init__(self, n_heads, n_in=1, n_parallel_heads=None,
               incremental=False, predict_mem_len=None, predict_drop_len=None,
               use_python_loop=False, use_reference_code=False):
    """Constructs an EfficientAttentionBase instance.

    Args:
      n_heads: Number of attention heads.
      n_in: Number of inputs to the layer (default 1).
      n_parallel_heads: Number of attention heads to compute in parallel.

          - If `n_parallel_heads` is None (default), the entire layer is
            computed with maximum parallelism. This mode is the fastest, but
            also uses the most memory. Start with this mode, but switch to one
            of the others if memory runs out.
          - If `n_parallel_heads` is 1, attention is computed one head at a
            time, and one example at a time. This mode uses the least memory
            but is not as fast as batched attention. Use this mode when working
            with very long sequences, such that any amount of parallelism won't
            fit in memory.
          - If `n_parallel_heads` is a multiple of `n_heads`, attention is
            computed for sub-batches of (`n_parallel_heads // n_heads`)
            examples at a time.
          - If `1 < n_parallel_heads < n_heads`, attention is computed for
            several heads at a time, but only within a single example. It must
            be the case that `n_heads` is a multiple of `n_parallel_heads`. Use
            this mode for long sequences, to strike a balance between
            parallelism and memory usage.
      incremental: If `True`, enable fast inference for self-attention types.
          Note that this flag should *not* be set when doing encoder-decoder
          attention, but only when doing self-attention.
      predict_mem_len: Number of input positions to remember in a cache
          when doing fast inference. Whenever the cache fills up, some input
          elements will be forgotten.
      predict_drop_len: Number of input elements to drop once the fast
          inference input cache fills up.
      use_python_loop: Set to True to use a Python loop when iterating over
          sub-batches of examples/heads (as opposed to a JAX/XLA loop).
          This option will increase compilation time and jitted code size,
          potentially drastically. Using it is not recommended except for
          testing/debugging. In particular, note that enabling this option on
          TPU can decrease the maximum model size that will fit in memory.
      use_reference_code: Set to True to fall back to the reference
          implementation of batched attention. This option will increase
          compilation time and jitted code size, potentially drastically. Using
          it is not recommended except for testing/debugging.
    """
    super().__init__(n_in=n_in, n_out=1)
    self.n_heads = n_heads
    self.incremental = incremental
    if self.incremental:
      if predict_mem_len is None or predict_drop_len is None:
        raise ValueError('This configuration does not support fast inference.')
      if not 0 < predict_drop_len <= predict_mem_len:
        raise ValueError(
            'Bad parameter values: (predict_mem_len, predict_drop_len) = ',
            predict_mem_len, predict_drop_len)
      self.predict_mem_len = predict_mem_len
      self.predict_drop_len = predict_drop_len

    if n_parallel_heads:
      if ((n_parallel_heads > n_heads and n_parallel_heads % n_heads != 0)
          or (n_parallel_heads < n_heads and n_heads % n_parallel_heads != 0)):
        raise ValueError(
            'n_parallel_heads must be a multiple or fraction of n_heads')
      self.n_parallel_heads = n_parallel_heads
    else:
      self.n_parallel_heads = None
    self.use_python_loop = use_python_loop
    self.use_reference_code = use_reference_code

  def init_weights_and_state(self, input_signature):
    if not isinstance(input_signature, (tuple, list)):
      input_signature = (input_signature,)
    input_signature_unbatched = fastmath.nested_map(
        lambda x: type(x)(shape=x.shape[1:], dtype=x.dtype),
        input_signature)
    batch_size = int(input_signature[0].shape[0])

    weights = []
    weight_rngs = fastmath.random.split(self.rng, self.n_heads)
    for i in range(self.n_heads):
      weights.append(self.create_weights_unbatched(input_signature_unbatched,
                                                   weight_rngs[i]))
    state = []
    state_rngs = fastmath.random.split(self.rng, self.n_heads * batch_size)
    for i in range(self.n_heads * batch_size):
      state.append(self.create_state_unbatched(input_signature_unbatched,
                                               state_rngs[i]))

    stack_along_axis_0 = lambda *x: np.stack(x, axis=0)
    weights = fastmath.nested_map_multiarg(stack_along_axis_0, *weights)
    state = fastmath.nested_map_multiarg(stack_along_axis_0, *state)

    if self.incremental:
      mem = fastmath.nested_map(
          lambda x: np.zeros(  # pylint: disable=g-long-lambda
              x.shape[:1] + (self.predict_mem_len,) + x.shape[2:],
              dtype=x.dtype),
          input_signature)
      mem_end = np.zeros((), dtype=np.int32)
      state = (mem_end, mem, state)

    self.state = tuple(state)
    self.weights = tuple(weights)

  def create_weights_unbatched(self, input_signature, rng):
    raise NotImplementedError(
        'Subclasses should override create_weights_unbatched')

  def create_state_unbatched(self, input_signature, rng):
    return ()

  def forward_unbatched(self, *inputs, weights, state):
    """Perform attention for a single batch element and head.

    Subclasses should override this method.

    Args:
      *inputs: Inputs for a single example (subclasses may use different inputs)
      weights: Weights for a single attention head
      state: State for a single example & attention head pair.

    Returns:
      A tuple (output, new_state) -- output and new state for a single example
      and attention head.
    """
    raise NotImplementedError('Subclasses should override forward_unbatched')

  def incremental_forward_unbatched(self, *inputs, q_start, q_len,
                                    weights, state):
    """Perform fast inference for a single batch element and head.

    Subclasses should override this method.

    Args:
      *inputs: Inputs for a single example (subclasses may use different inputs)
      q_start: Index along the sequence-length dimension that points to the
        first input element that should be used as a query (and not just a key).
      q_len: Number of new query elements in this call to the attention
        mechanism. This is typically 1 for autoregressive decoding, but may be
        longer if initializing a language model with a prefix.
      weights: Weights for a single attention head
      state: State for a single example & attention head pair.

    Returns:
      A tuple (output, new_state) -- output and new state for a single example
      and attention head.
    """
    raise NotImplementedError(
        'Fast inference is not implemented for this attention type.')

  def forward(self, inputs):
    """Computes this layer's output as part of a forward pass through the model.

    Args:
      inputs: Layer inputs (subclasses may use different inputs)

    Returns:
      A tuple (output, new_state).
    """
    weights, state, rng = self.weights, self.state, self.rng
    if not self.use_reference_code:
      # By default, an efficient, batched implementation is used.
      output, new_state, _, _ = self.forward_and_or_backward(
          inputs, weights, state, rng, compute_output=True, update_state=True)
      self.state = new_state
      return output

    # The reference implementation below provides a more readable overview of
    # what this class does. It's not optimized, however, and should only be used
    # when testing this class for correctness.
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    batch_size = int(inputs[0].shape[0])
    seqlen = inputs[0].shape[-2]
    d_model = inputs[0].shape[-1]

    if self.incremental:
      inputs, state, q_start, new_mem, new_mem_end = self.use_predict_mem(
          inputs, state)

    output_accum = [np.zeros((seqlen, d_model)) for _ in range(batch_size)]
    new_state = []
    for example_idx in range(batch_size):
      for head_idx in range(self.n_heads):
        # pylint: disable=cell-var-from-loop
        single_inputs = fastmath.nested_map(lambda x: x[example_idx], inputs)
        single_weights = fastmath.nested_map(lambda w: w[head_idx], weights)
        single_state = fastmath.nested_map(
            lambda s: s[example_idx * self.n_heads + head_idx], state)
        # pylint: enable=cell-var-from-loop
        if self.incremental:
          single_out, single_new_state = self.incremental_forward_unbatched(
              *single_inputs, q_start=q_start, q_len=seqlen,
              weights=single_weights, rng=rng,
              state=single_state, update_state=True)
        else:
          single_out, single_new_state = self.forward_unbatched(
              *single_inputs, weights=single_weights, rng=rng,
              state=single_state, update_state=True)
        new_state.append(single_new_state)
        output_accum[example_idx] = output_accum[example_idx] + single_out

    output = np.stack(output_accum, 0)
    if new_state and fastmath.tree_leaves(new_state[0]):
      new_state = fastmath.nested_map_multiarg(
          lambda *s: np.stack(s, 0), *new_state)
    else:
      new_state = state
    if self.incremental:
      new_state = (new_mem_end, new_mem, new_state)
    self.state = tuple(new_state)
    return output

  def use_predict_mem(self, inputs, state):
    """Update input cache for fast inference."""
    mem_end, mem, state = state
    seqlen = inputs[0].shape[-2]

    if seqlen <= self.predict_drop_len and seqlen < self.predict_mem_len:
      # This branch is called when only a small number of tokens are appended to
      # the sequence, e.g. when generating one token at a time. A fixed number
      # of tokens (self.predict_drop_tokens) will be dropped from memory if
      # needed, and then new values will be inserted into the memory.
      def roll_mem(buf):
        return np.concatenate(
            [buf[:, self.predict_drop_len:],
             np.zeros_like(buf[:, :self.predict_drop_len])], axis=1)

      do_roll_mem = (mem_end + seqlen > self.predict_mem_len)
      mem = fastmath.cond(
          pred=do_roll_mem,
          true_operand=mem,
          true_fun=lambda x: fastmath.nested_map(roll_mem, x),
          false_operand=mem,
          false_fun=lambda x: x,
      )
      mem_end = np.where(do_roll_mem, mem_end - self.predict_drop_len, mem_end)
      def update_mem(mem_element, new_vals):
        assert new_vals.shape[1] == seqlen
        if seqlen == 1:
          return fastmath.index_update(
              mem_element, jax.ops.index[:, mem_end], new_vals[:, 0, ...])
        else:
          return fastmath.dynamic_update_slice_in_dim(
              mem_element, new_vals, mem_end, axis=1)
      inputs = fastmath.nested_map_multiarg(update_mem, mem, inputs)
      return inputs, state, mem_end, inputs, mem_end + seqlen
    else:
      assert seqlen > self.predict_drop_len or seqlen == self.predict_mem_len
      # This branch handles the case where a large number of tokens are being
      # introduced all at once. The code here assumes that we are at the start
      # of the sequence, which matches the typical use case of decoding from a
      # language model given a long prefix. Note that if we're not at the start
      # of the sequence, the code here won't work.
      new_flat_mem = []
      for inp in fastmath.tree_leaves(inputs):
        assert inp.shape[1] == seqlen
        if seqlen == self.predict_mem_len:
          new_mem_val = inp
        elif seqlen > self.predict_mem_len:
          new_mem_val = inp[:, -self.predict_mem_len:]  # pylint: disable=invalid-unary-operand-type
        else:
          new_mem_val = np.concatenate([
              inp,
              np.zeros(inp.shape[:1]
                       + (self.predict_mem_len - inp.shape[1],)
                       + inp.shape[2:],
                       dtype=inp.dtype)
          ], axis=1)
        new_flat_mem.append(new_mem_val)
      mem, _ = fastmath.tree_unflatten(new_flat_mem, mem)

      # This code only works at the start of the sequence. There's no "assert"
      # primitive we can use to signal an error, so we instead signal the error
      # by introducing NaNs into the computation.
      def replace_with_nan_if_not_seq_start(x):
        if x.dtype != np.float32:
          return x
        return fastmath.cond(
            pred=np.equal(mem_end, np.array(0, dtype=mem_end.dtype)),
            true_operand=x, true_fun=lambda x: x,
            false_operand=x, false_fun=lambda x: x * np.nan)
      inputs = fastmath.nested_map(replace_with_nan_if_not_seq_start, inputs)
      return inputs, state, 0, mem, np.minimum(seqlen, self.predict_mem_len)

  @property
  def has_backward(self):
    # Use an efficient backward pass, unless we're running the reference code.
    return not self.use_reference_code

  def backward(self, inputs, output, grad, weights, state, new_state, rng=None,
               **kwargs):
    """Custom backward pass, for efficiency (see forward_and_or_backward)."""
    assert not self.use_reference_code
    del output, state, kwargs
    _, _, inputs_grad, weights_grad = self.forward_and_or_backward(
        inputs, weights, new_state, rng, output_grad=grad,
        compute_output=False, update_state=False)
    return inputs_grad, weights_grad

  def forward_and_or_backward(
      self, inputs, weights, state, rng, output_grad=None,
      compute_output=True, update_state=True):
    """Performs batched forward and/or backward passes.

    See `forward` for a reference implementation of what this layer does. The
    reference implementation is not very efficient, however, and this method
    provides a more performant version.

    Args:
      inputs: inputs to the attention layer
      weights: weights for the attention layer
      state: state of the attention layer
      rng: PRNG key for the layer (shared across all examples and heads)
      output_grad: gradient of the loss wrt the output of the layer, or None.
          This function performs the backward pass iff `output_grad` is not
          None.
      compute_output: bool: whether to return the output of the forward pass
          (for example, a pure backwards pass does not need to return the
          output).
      update_state: bool: whether to return an updated layer state.

    Returns:
      A tuple (output, new_state, inputs_grad, weights_grad).

      - output is not None iff compute_output is True
      - new_state is not None iff update_state is True
      - inputs_grad & weights_grad are not None iff output_grad is not None
    """
    # TODO(kitaev): profile ~4% speed drop compared to previous implementation
    #     in some conditions. Other conditions (e.g. the enwik8 model) appear
    #     to have the same overall training speed.
    # TODO(b/148460708): reduce memory usage further
    # TODO(kitaev): there should be a higher-level API (like vmap) that does
    #     batching, instead of needing 3 separate manual implementations here.

    # Notes regarding the implementation:
    # (a) Multiple heads or examples are batched together. There are three
    #     different regimes possible: one head at a time (for long sequences and
    #     expensive attention types), several attention heads at a time (for
    #     long sequences but less-expensive attention types), and several
    #     examples at a time (for large batches of shorter sequences). For the
    #     time being, each of these regimes has its own code.
    # (b) Python loops produce large computation graphs when jitted, so the
    #     default is to use a JAX loop instead.
    # (c) No intermediate quantities are cached for the backward pass. Instead,
    #     the forward pass is re-computed when doing backprop. This approach is
    #     often called "checkpointing" or "rematerialization". When not all
    #     examples or heads fit in memory simultaneously, the implementation
    #     should be [FW-BW-1] and NOT [FW-BW-2], because the latter has worse
    #     memory locality. I don't think JAX autodiff can synthesize [FW-BW-1]
    #     automatically, so the looping for the backward pass is done manually.
    #
    #     [FW-BW-1] for example, head in zip(examples, heads):
    #                 forward(example, head)
    #                 backward(example, head)  # uses intermediates from forward
    #
    #     [FW-BW-2] for example, head in zip(examples, heads):
    #                 forward(example, head)
    #               for example, head in zip(examples, heads):
    #                 backward(example, head)

    have_single_input = not isinstance(inputs, (tuple, list))
    if have_single_input:
      inputs = (inputs,)
    batch_size = int(inputs[0].shape[0])
    seqlen = inputs[0].shape[-2]
    d_model = inputs[0].shape[-1]

    compute_grad = (output_grad is not None)
    assert compute_output or compute_grad, 'No work to perform!'

    if not self.incremental:
      forward_unbatched = functools.partial(
          self.forward_unbatched, rng=rng, update_state=update_state)
    else:
      if update_state:
        inputs, state, q_start, new_mem, new_mem_end = self.use_predict_mem(
            inputs, state)
      else:
        # This assumes that the memory stores all of the inputs, which would not
        # be valid if doing backprop in mode 'predict' with long lengths.
        new_mem_end, inputs, state = state
        q_start = new_mem_end - seqlen

      forward_unbatched = functools.partial(
          self.incremental_forward_unbatched,
          q_start=fastmath.stop_gradient(q_start),
          q_len=fastmath.stop_gradient(seqlen),
          rng=rng, update_state=update_state)

    # Adjust degree of parallelism based on the batch size.
    n_parallel_heads = batch_size * self.n_heads
    if self.n_parallel_heads and self.n_parallel_heads < n_parallel_heads:
      n_parallel_heads = self.n_parallel_heads

    def tree_update(tree, indices, new_values):
      return fastmath.nested_map_multiarg(
          lambda x, y: fastmath.index_update(x, jax.ops.index[indices], y),
          tree, new_values)

    def tree_add(tree, indices, new_values):
      return fastmath.nested_map_multiarg(
          lambda x, y: fastmath.index_add(x, jax.ops.index[indices], y),
          tree, new_values)

    if compute_grad:
      inputs_is_differentiable = fastmath.nested_map(
          lambda x: np.issubdtype(x.dtype, np.inexact), inputs)
      def split_differentiable(xs):
        differentiable_xs = fastmath.nested_map_multiarg(
            lambda x, is_differentiable: x if is_differentiable else None,
            xs, inputs_is_differentiable)
        non_differentiable_xs = fastmath.nested_map_multiarg(
            lambda x, is_differentiable: None if is_differentiable else x,
            xs, inputs_is_differentiable)
        return differentiable_xs, non_differentiable_xs
      def join_differentiable(differentiable_xs, non_differentiable_xs):
        """Reconstitute inputs pytree from differentiable/non-d. partitions."""
        differentiable_leaves = fastmath.tree_leaves(differentiable_xs)
        non_differentiable_leaves = fastmath.tree_leaves(non_differentiable_xs)
        leaves = []
        for is_differentiable in fastmath.tree_leaves(inputs_is_differentiable):
          if is_differentiable:
            leaves.append(differentiable_leaves.pop(0))
          else:
            leaves.append(non_differentiable_leaves.pop(0))
        assert not differentiable_leaves
        assert not non_differentiable_leaves
        tree, _ = fastmath.tree_unflatten(leaves, inputs)
        return tree

      def vjp(fn, inp, *args, has_aux=False):
        d_inp, nd_inp = split_differentiable(inp)
        def fn_closed_over_nd_inp(d_inp, *args):
          inp = join_differentiable(d_inp, nd_inp)
          return fn(inp, *args)
        return fastmath.vjp(fn_closed_over_nd_inp, d_inp, *args,
                            has_aux=has_aux)

    if n_parallel_heads == 1:
      def run_inner(idx, loop_val):
        """Runs one slice of attention (for a single head)."""
        o_all, s_all, i_ct_all, w_ct_all = loop_val
        example_idx = idx // self.n_heads
        head_idx = idx % self.n_heads

        i_h = fastmath.nested_map(lambda x: x[example_idx], inputs)
        w_h = fastmath.nested_map(lambda w: w[head_idx], weights)
        s_h = fastmath.nested_map(lambda s: s[idx], state)

        def forward_fn(i_h, w_h):
          return forward_unbatched(
              *i_h, weights=w_h, state=fastmath.stop_gradient(s_h))

        if compute_grad:
          o_h, backward_fn, s_h = vjp(forward_fn, i_h, w_h, has_aux=True)
          ct_h = output_grad[example_idx]
          assert o_h.shape == ct_h.shape
          i_ct_h, w_ct_h = backward_fn(ct_h)
        else:
          o_h, s_h = forward_fn(i_h, w_h)

        if compute_output:
          o_all = fastmath.index_add(o_all, example_idx, o_h)
        if update_state:
          s_all = tree_update(s_all, idx, s_h)
        if compute_grad:
          i_ct_all = tree_add(i_ct_all, example_idx, i_ct_h)
          w_ct_all = tree_add(w_ct_all, head_idx, w_ct_h)
        return (o_all, s_all, i_ct_all, w_ct_all)
    elif n_parallel_heads < self.n_heads:
      assert self.n_heads % n_parallel_heads == 0
      def run_inner(idx, loop_val):
        """Runs one slice of attention (multiple heads, but one example)."""
        o_all, s_all, i_ct_all, w_ct_all = loop_val
        idx = idx * self.n_parallel_heads
        example_idx = idx // self.n_heads
        head_idx_lo = idx % self.n_heads
        head_range = head_idx_lo + np.arange(n_parallel_heads, dtype=np.int32)
        state_range = idx + np.arange(n_parallel_heads, dtype=np.int32)

        i_mh = fastmath.nested_map(lambda x: x[example_idx], inputs)
        w_mh = fastmath.nested_map(lambda w: w[head_range], weights)
        s_mh = fastmath.nested_map(lambda s: s[state_range], state)
        def forward_unbatched_h(i_h, w_h, s_h):
          return forward_unbatched(*i_h, weights=w_h, state=s_h)
        def forward_fn(i_mh, w_mh):
          o_mh, new_s_mh = fastmath.vmap(
              forward_unbatched_h, in_axes=(None, 0, 0), out_axes=0)(
                  i_mh, w_mh, s_mh)
          o_mh = np.sum(o_mh, axis=0)
          return o_mh, new_s_mh

        if compute_grad:
          o_mh, backward_fn, s_mh = vjp(forward_fn, i_mh, w_mh, has_aux=True)
          ct_mh = output_grad[example_idx]
          assert o_mh.shape == ct_mh.shape
          i_ct_mh, w_ct_mh = backward_fn(ct_mh)
        else:
          o_mh, s_mh = forward_fn(i_mh, w_mh)

        if compute_output:
          o_all = fastmath.index_add(o_all, example_idx, o_mh)
        if update_state:
          s_all = tree_update(s_all, state_range, s_mh)
        if compute_grad:
          i_ct_all = tree_add(i_ct_all, example_idx, i_ct_mh)
          w_ct_all = tree_add(w_ct_all, head_range, w_ct_mh)
        return (o_all, s_all, i_ct_all, w_ct_all)
    else:
      assert n_parallel_heads % self.n_heads == 0
      def forward_single_example(i_x, w_all, s_x):
        def forward_unbatched_h(i_h, w_h, s_h):
          return forward_unbatched(*i_h, weights=w_h, state=s_h)
        o_x, s_x = fastmath.vmap(
            forward_unbatched_h, in_axes=(None, 0, 0), out_axes=(0, 0))(
                i_x, w_all, s_x)
        o_x = np.sum(o_x, axis=0)
        return o_x, s_x
      def run_inner(idx, loop_val):
        """Runs one slice of attention (all heads for one or more examples)."""
        o_all, s_all, i_ct_all, w_ct_all = loop_val
        idx = idx * n_parallel_heads
        example_idx_lo = idx // self.n_heads
        example_range = example_idx_lo + np.arange(
            n_parallel_heads // self.n_heads, dtype=np.int32)
        state_range = idx + np.arange(n_parallel_heads, dtype=np.int32)

        i_mex = fastmath.nested_map(lambda x: x[example_range], inputs)
        s_mex = fastmath.nested_map(
            lambda s: np.reshape(s[state_range],  # pylint: disable=g-long-lambda
                                 (-1, self.n_heads) + s.shape[1:]),
            state)
        def forward_fn(i_mex, w_all):
          o_mex, new_s_mex = fastmath.vmap(
              forward_single_example, in_axes=(0, None, 0), out_axes=(0, 0))(
                  i_mex, w_all, s_mex)
          new_s_mex = fastmath.nested_map(
              lambda s: np.reshape(s, (n_parallel_heads,) + s.shape[2:]),
              new_s_mex)
          return o_mex.astype(i_mex[0].dtype), new_s_mex

        if compute_grad:
          o_mex, backward_fn, s_mex = vjp(forward_fn, i_mex, weights,
                                          has_aux=True)
          ct_mex = output_grad[example_range]
          assert o_mex.shape == ct_mex.shape, str(ct_mex.shape)
          assert o_mex.dtype == ct_mex.dtype, str(ct_mex.dtype)
          i_ct_mex, w_ct_mex = backward_fn(ct_mex)
        else:
          o_mex, s_mex = forward_fn(i_mex, weights)

        if compute_output:
          o_all = fastmath.index_add(o_all, jax.ops.index[example_range], o_mex)
        if update_state:
          s_all = tree_update(s_all, state_range, s_mex)
        if compute_grad:
          i_ct_all = tree_update(i_ct_all, example_range, i_ct_mex)
          w_ct_all = fastmath.nested_map_multiarg(
              lambda old_all, delta_all: old_all + delta_all,
              w_ct_all, w_ct_mex)
        return (o_all, s_all, i_ct_all, w_ct_all)

    o_all = s_all = i_ct_all = w_ct_all = None
    if compute_output:
      o_all = np.zeros(
          (batch_size, seqlen, d_model), dtype=inputs[0].dtype)
    if update_state:
      s_all = state
    if compute_grad:
      i_ct_all = fastmath.nested_map(np.zeros_like, inputs)
      i_ct_all, i_nondifferentiable_dummy_ct = split_differentiable(i_ct_all)
      w_ct_all = fastmath.nested_map(np.zeros_like, weights)

    loop_val = (o_all, s_all, i_ct_all, w_ct_all)

    assert (batch_size * self.n_heads) % n_parallel_heads == 0
    loop_hi = (batch_size * self.n_heads) // n_parallel_heads
    if self.use_python_loop or loop_hi == 1:
      for idx in range(loop_hi):
        loop_val = run_inner(idx, loop_val)
    else:
      loop_val = fastmath.fori_loop(
          0, loop_hi, run_inner, loop_val)

    (o_all, s_all, i_ct_all, w_ct_all) = loop_val

    if compute_grad:
      i_ct_all = join_differentiable(i_ct_all, i_nondifferentiable_dummy_ct)

    if self.incremental and update_state:
      s_all = (new_mem_end, new_mem, s_all)

    if have_single_input and compute_grad:
      assert isinstance(i_ct_all, tuple) and len(i_ct_all) == 1
      return (o_all, s_all, i_ct_all[0], w_ct_all)
    else:
      return (o_all, s_all, i_ct_all, w_ct_all)


class SelfAttention(EfficientAttentionBase):
  """Memory-efficient self-attention (second attempt)."""

  def __init__(self,
               n_heads=2, d_qk=64, d_v=64, share_qk=False,
               causal=False, masked=False,
               chunk_len=None, n_chunks_before=0, n_chunks_after=0,
               bias=False,
               mode='train',
               predict_mem_len=None, predict_drop_len=None,
               attention_dropout=0.0,
               output_dropout=0.0,
               n_parallel_heads=None,
               use_python_loop=False,
               use_reference_code=False,
              ):
    """Construct a self-attention layer.

    Args:
      n_heads: int: Number of attention heads
      d_qk: int: Depth of query ond key vectors
      d_v: int: Depth of value vectors
      share_qk: bool: Set to True to share query and key projection weights
      causal: bool: Set to True to mask out attention to future items
      masked: bool: Set to True to accept an additional mask argument, that
        allows masking out attention to padding tokens.
      chunk_len (optional): Number of tokens per chunk. Setting this option will
        enable chunked attention.
      n_chunks_before: Number of previous chunks to attend to, when using
        chunked attention.
      n_chunks_after: Number of subsequent chunks to attend to, when using
        chunked attention. Don't use this option for causal attention, because
        attention to future tokens will be masked out anyway. However, note that
        cross-chunk attention "wraps around" in both directions, so this option
        is never a strict no-op.
      bias: bool: Set to True to add bias vectors when computing query/key/value
      mode: 'train', 'eval', or 'predict'
      predict_mem_len: int: Number of input positions to remember in a cache
        when doing fast inference. Whenever the cache fills up, some input
        elements will be forgotten. When chunking is enabled, the default is to
        store chunk_len * (1 + n_chunks_before) elements.
      predict_drop_len: int: Number of input elements to drop once the fast
        inference input cache fills up. When chunking is enabled, the default is
        to drop exactly chunk_len elements.
      attention_dropout: Dropout probability for attention mask.
      output_dropout: Dropout probability for the layer output.
      n_parallel_heads: see EfficientAttentionBase. This option controls the
        trade-off between parallelism and memory usage.
      use_python_loop: For testing/debugging (see EfficientAttentionBase)
      use_reference_code: For testing/debugging (see EfficientAttentionBase)
    """
    if mode == 'predict':
      assert causal, 'Only causal attention supports fast inference'
      assert chunk_len is not None or (predict_mem_len and predict_drop_len)
      predict_mem_len = predict_mem_len or (chunk_len * (1 + n_chunks_before))
      predict_drop_len = predict_drop_len or chunk_len
    super().__init__(
        n_heads=n_heads,
        n_in=(2 if masked else 1),
        n_parallel_heads=n_parallel_heads,
        incremental=(mode == 'predict'),
        predict_mem_len=predict_mem_len,
        predict_drop_len=predict_drop_len,
        use_python_loop=use_python_loop,
        use_reference_code=use_reference_code,
        )
    self.d_qk = d_qk
    self.d_v = d_v
    self.share_qk = share_qk
    self.causal = causal
    self.masked = masked
    self.chunk_len = chunk_len
    self.n_chunks_before = n_chunks_before
    self.n_chunks_after = n_chunks_after
    self.bias = bias
    self.mode = mode
    if mode == 'train':
      self.attention_dropout = attention_dropout
      self.output_dropout = output_dropout
    else:
      self.attention_dropout = 0.0
      self.output_dropout = 0.0

  def _kernel_initializer(self, shape, rng):
    # Attention uses Glorot uniform initalization with respect to the *total*
    # dimension of queries/key/values across all heads. We initialize one head
    # at a time in this class, so init.GlorotUniformInitializer won't work.
    # This initialization type is for parity with previous Trax & tensor2tensor
    # Transformers; it's not clear if it's strictly needed for model accuracy.
    lim = np.sqrt(6.0 / (shape[0] + shape[1] * self.n_heads))
    return fastmath.random.uniform(rng, shape, np.float32, -lim, lim)

  def create_weights_unbatched(self, input_signature, rng):
    if isinstance(input_signature, (tuple, list)):
      input_signature = input_signature[0]
    d_model = input_signature.shape[-1]
    rng_q, rng_k, rng_v, rng_o = fastmath.random.split(rng, 4)
    w_q = self._kernel_initializer((d_model, self.d_qk), rng_q)
    if not self.share_qk:
      w_k = self._kernel_initializer((d_model, self.d_qk), rng_k)
    w_v = self._kernel_initializer((d_model, self.d_v), rng_v)
    w_o = np.transpose(self._kernel_initializer((d_model, self.d_v), rng_o))

    if self.bias:
      b_q = np.zeros(self.d_qk)
      b_v = np.zeros(self.d_v)
      if self.share_qk:
        return (w_q, w_v, w_o, b_q, b_v)
      else:
        b_k = np.zeros(self.d_qk)
        return (w_q, w_k, w_v, w_o, b_q, b_k, b_v)

    if self.share_qk:
      return (w_q, w_v, w_o)
    else:
      return (w_q, w_k, w_v, w_o)

  def forward_unbatched(self, x, mask=None, *,
                        weights, state, rng, update_state):
    del update_state
    attend_rng, output_rng = fastmath.random.split(rng)
    if self.bias:
      if self.share_qk:
        w_q, w_v, w_o, b_q, b_v = weights
      else:
        w_q, w_k, w_v, w_o, b_q, b_k, b_v = weights
    else:
      if self.share_qk:
        w_q, w_v, w_o = weights
      else:
        w_q, w_k, w_v, w_o = weights

    q = np.matmul(x, w_q)
    k = None
    if not self.share_qk:
      k = np.matmul(x, w_k)
    v = np.matmul(x, w_v)

    if self.bias:
      q = q + b_q
      if not self.share_qk:
        k = k + b_k
      v = v + b_v

    mask_fn = functools.partial(
        mask_self_attention,
        causal=self.causal, exclude_self=self.share_qk, masked=self.masked)
    q_info = kv_info = np.arange(q.shape[-2], dtype=np.int32)

    assert (mask is not None) == self.masked
    if self.masked:
      # mask is a boolean array (True means "is valid token")
      ones_like_mask = np.ones_like(mask, dtype=np.int32)
      kv_info = kv_info * np.where(mask, ones_like_mask, -ones_like_mask)

    o, _ = attend(
        q, k, v,
        q_chunk_len=self.chunk_len,
        kv_chunk_len=self.chunk_len,
        n_chunks_before=self.n_chunks_before,
        n_chunks_after=self.n_chunks_after,
        mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
        dropout=self.attention_dropout, rng=attend_rng,
        )

    out = np.matmul(o, w_o)
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    return out, state

  def incremental_forward_unbatched(self, x, mask=None, *,
                                    q_start, q_len,
                                    weights, state, rng, update_state):
    del update_state
    attend_rng, output_rng = fastmath.random.split(rng)
    if self.share_qk:
      w_q, w_v, w_o = weights
    else:
      w_q, w_k, w_v, w_o = weights

    q_range = q_start + np.arange(q_len, dtype=np.int32)
    if q_len == 1:
      # On TPU, np.matmul(a[:1], b) and np.matmul(a, b)[:1] are not
      # floating-point equivalent, at least in non-jitted code. We correct the
      # discrepancy by duplicating the slice. Floating-point noise may not be
      # an issue when using models, but it makes it harder to write tests that
      # compare fast and slow inference code for equivalence.
      q = np.matmul(np.concatenate([x[q_range]] * 2, 0), w_q)
    else:
      q = np.matmul(x[q_range], w_q)
    if self.share_qk:
      k = length_normalized(np.matmul(x, w_q))
    else:
      k = np.matmul(x, w_k)
    v = np.matmul(x, w_v)

    mask_fn = functools.partial(
        mask_self_attention,
        causal=self.causal, exclude_self=self.share_qk, masked=self.masked)
    q_info = q_range
    kv_info = np.arange(k.shape[-2], dtype=np.int32)

    if self.chunk_len is not None and q_len > self.chunk_len:
      assert q_start == 0
      assert q_len % self.chunk_len == 0
      o, _ = attend(
          q, k, v,
          q_chunk_len=self.chunk_len,
          kv_chunk_len=self.chunk_len,
          n_chunks_before=self.n_chunks_before,
          n_chunks_after=self.n_chunks_after,
          mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
          dropout=self.attention_dropout, rng=attend_rng,
          )
    else:
      o, _ = attend(
          q, k, v,
          mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
          dropout=self.attention_dropout, rng=attend_rng,
          )

    out = np.matmul(o, w_o)
    if q_len == 1:
      out = out[:1]
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    return out, state


class LSHSelfAttention(SelfAttention):
  """LSH self-attention (second implementation)."""

  def __init__(self,
               n_heads=2, d_qk=64, d_v=64, share_qk='unused',
               causal=False,
               masked=False,
               chunk_len=128, n_chunks_before=1, n_chunks_after=0,
               n_hashes=1,
               n_buckets=None,
               mode='train',
               predict_mem_len=2048, predict_drop_len=256,
               attention_dropout=0.0,
               output_dropout=0.0,
               n_parallel_heads=1,
               use_python_loop=False,
               use_reference_code=False,
               max_length_for_buckets=None,
              ):
    """Construct an LSH self-attention layer."""
    super().__init__(
        n_heads=n_heads, d_qk=d_qk, d_v=d_v, share_qk=True,
        causal=causal,
        masked=masked,
        chunk_len=chunk_len,
        n_chunks_before=n_chunks_before, n_chunks_after=n_chunks_after,
        mode=mode,
        predict_mem_len=predict_mem_len, predict_drop_len=predict_drop_len,
        attention_dropout=attention_dropout,
        output_dropout=output_dropout,
        n_parallel_heads=n_parallel_heads,
        use_python_loop=use_python_loop,
        use_reference_code=use_reference_code,
        )
    self.n_hashes = n_hashes
    self.n_buckets = n_buckets
    self._max_length_for_buckets = max_length_for_buckets

  def create_state_unbatched(self, input_signature, rng):
    if isinstance(input_signature, (tuple, list)):
      input_signature = input_signature[0]
    # The `rng` argument passed to forward_unbatched is shared across all
    # examples and heads. This facilitates using broadcasted dropout, which
    # saves memory and hasn't been shown to hurt model quality. Even though the
    # same sharing is likely to be safe when selecting random hash functions
    # for LSH, we haven't run experiments to demonstrate this. To be on the safe
    # side we include a per-head RNG in the state for the purpose of doing LSH.
    if not self.incremental:
      length = self._max_length_for_buckets or input_signature.shape[0]
      buckets = np.zeros(self.n_hashes * length, dtype=np.int32)
      return (buckets, rng)
    else:
      buckets = np.zeros(
          self.n_hashes * self.predict_mem_len, dtype=np.int32)
      buckets_idx = np.zeros((), dtype=np.int32)
      return (buckets, buckets_idx, rng)

  def hash_vectors(self, vecs, rng, mask=None):
    n_buckets_list = self.n_buckets

    # Determine the number of buckets needed from input length if not set.
    if n_buckets_list is None:
      length = vecs.shape[0]
      n_buckets = 2 * max(1, length // self.chunk_len)
      if n_buckets <= 128:
        n_buckets_list = n_buckets
      else:  # Factorize n_buckets.
        n_buckets_div = 2**math.ceil(math.log2(math.sqrt(n_buckets)))
        # Both factors must be even.
        n_buckets_rest = 2 * (n_buckets // (2 * n_buckets_div))
        n_buckets_list = [n_buckets_div, n_buckets_rest]

    # Hash vectors.
    buckets, n_buckets = hash_vecs(vecs, n_buckets_list, self.n_hashes, rng)

    if mask is not None:
      n_buckets += 1  # Create an extra bucket for padding tokens only
      buckets = np.where(mask[None, :], buckets, n_buckets - 1)

    # buckets is now (n_hashes, seqlen). Next we add offsets so that
    # bucket numbers from different hashing rounds don't overlap.
    offsets = np.arange(self.n_hashes, dtype=np.int32)
    offsets = np.reshape(offsets * n_buckets, (-1, 1))
    buckets = np.reshape(buckets + offsets, (-1,))
    return buckets

  def forward_unbatched(self, x, mask=None, *, weights, state, rng,
                        update_state):
    attend_rng, output_rng = fastmath.random.split(rng)
    w_q, w_v, w_o = weights

    q = np.matmul(x, w_q)
    v = np.matmul(x, w_v)

    if update_state:
      _, old_hash_rng = state
      hash_rng, hash_subrng = fastmath.random.split(old_hash_rng)
      buckets = self.hash_vectors(q, hash_subrng, mask)
      s_buckets = buckets
      if self._max_length_for_buckets:
        length = self.n_hashes * self._max_length_for_buckets
        if buckets.shape[0] < length:
          s_buckets = np.concatenate(
              [buckets, np.zeros(length - buckets.shape[0], dtype=np.int32)],
              axis=0)
      state = (s_buckets, hash_rng)
    else:
      buckets, _ = state
      if self._max_length_for_buckets:
        buckets = buckets[:self.n_hashes * x.shape[0]]

    seqlen = x.shape[0]
    assert int(buckets.shape[0]) == self.n_hashes * seqlen

    ticker = np.arange(self.n_hashes * seqlen, dtype=np.int32)
    buckets_and_t = seqlen * buckets + (ticker % seqlen)
    buckets_and_t = fastmath.stop_gradient(buckets_and_t)

    # Hash-based sort ("s" at the start of variable names means "sorted")
    sbuckets_and_t, sticker = fastmath.sort_key_val(
        buckets_and_t, ticker, dimension=-1)
    _, undo_sort = fastmath.sort_key_val(sticker, ticker, dimension=-1)
    sbuckets_and_t = fastmath.stop_gradient(sbuckets_and_t)
    sticker = fastmath.stop_gradient(sticker)
    undo_sort = fastmath.stop_gradient(undo_sort)

    st = (sticker % seqlen)
    sq = np.take(q, st, axis=0)
    sv = np.take(v, st, axis=0)

    mask_fn = functools.partial(mask_self_attention, causal=self.causal,
                                exclude_self=True, masked=self.masked)
    q_info = st

    assert (mask is not None) == self.masked
    kv_info = None
    if self.masked:
      # mask is a boolean array (True means "is valid token")
      smask = np.take(mask, st, axis=0)
      ones_like_mask = np.ones_like(smask, dtype=np.int32)
      kv_info = q_info * np.where(smask, ones_like_mask, -ones_like_mask)

    so, slogits = attend(
        sq, k=None, v=sv,
        q_chunk_len=self.chunk_len,
        n_chunks_before=self.n_chunks_before,
        n_chunks_after=self.n_chunks_after,
        mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
        dropout=self.attention_dropout, rng=attend_rng,
        )

    # np.take(so, undo_sort, axis=0); np.take(slogits, undo_sort, axis=0) would
    # also work, but these helpers include performance optimizations for TPU.
    o = permute_via_gather(so, undo_sort, sticker, axis=0)
    logits = permute_via_sort(slogits, sticker, buckets_and_t, axis=-1)

    if self.n_hashes > 1:
      o = np.reshape(o, (self.n_hashes, seqlen, o.shape[-1]))
      logits = np.reshape(logits, (self.n_hashes, seqlen, 1))
      probs = np.exp(logits - fastmath.logsumexp(logits, axis=0, keepdims=True))
      o = np.sum(o * probs, axis=0)

    assert o.shape == (seqlen, w_v.shape[-1])
    out = np.matmul(o, w_o)
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    return out, state

  def incremental_forward_unbatched(self, x, *,
                                    q_start, q_len,
                                    weights, state, rng, update_state):
    assert update_state, (
        'This setting not supported (e.g. no backprop for fast inference)')
    if q_len > 1:
      if isinstance(q_start, int):
        assert q_start == 0, 'Chunks larger than 1 only work at start for now.'
      if x.shape[0] % self.chunk_len == 0:
        x_padded = x
      else:
        pad_amount = self.chunk_len - (x.shape[0] % self.chunk_len)
        x_padded = np.pad(x, ((0, pad_amount), (0, 0)), mode='constant')
      buckets, buckets_idx, hash_rng = state
      q = np.matmul(x_padded, weights[0])
      buckets_update = self.hash_vectors(q, hash_rng)

      out, _ = self.forward_unbatched(
          x_padded, weights=weights, state=(buckets_update, hash_rng),
          rng=rng, update_state=False)

      out = out[:q_len]
      buckets = np.reshape(buckets, (self.n_hashes, -1))
      buckets_update = np.reshape(
          buckets_update, (self.n_hashes, -1))[:, :q_len]
      if q_len > self.predict_mem_len:
        buckets_update = buckets_update[:, -self.predict_mem_len:]  # pylint: disable=invalid-unary-operand-type
      buckets = fastmath.dynamic_update_slice_in_dim(
          buckets, buckets_update, q_start, axis=1)
      buckets = np.reshape(buckets, (-1,))

      return out, (buckets, buckets_idx + q_len, hash_rng)

    # This codepath is for handling one token at a time.
    assert q_len == 1
    buckets, buckets_idx, hash_rng = state

    def roll_buckets(buckets):
      buckets = np.reshape(buckets, (self.n_hashes, -1))
      new_buckets = np.concatenate(
          [buckets, np.zeros((self.n_hashes, self.predict_drop_len),
                             dtype=buckets.dtype)
          ], axis=1)
      new_buckets = fastmath.dynamic_slice_in_dim(
          new_buckets, buckets_idx - q_start, buckets.shape[-1], axis=1)
      new_buckets = np.reshape(new_buckets, (-1,))
      return new_buckets

    buckets = fastmath.cond(
        pred=buckets_idx > q_start,
        true_operand=buckets,
        true_fun=roll_buckets,
        false_operand=buckets,
        false_fun=lambda x: x,
    )

    attend_rng, output_rng = fastmath.random.split(rng)
    w_q, w_v, w_o = weights

    q_range = q_start + np.arange(q_len, dtype=np.int32)
    # On TPU, np.matmul(a[:1], b) and np.matmul(a, b)[:1] are not
    # floating-point equivalent, at least in non-jitted code. We correct the
    # discrepancy by duplicating the slice. Floating-point noise may not be
    # an issue when using models, but it makes it harder to write tests that
    # compare fast and slow inference code for equivalence.
    q = np.matmul(np.concatenate([x[q_range]] * 2, 0), w_q)

    q_buckets = self.hash_vectors(q, hash_rng)
    q_buckets = np.reshape(q_buckets, (self.n_hashes, 2))[:, :q_len]

    unflattened_buckets = fastmath.dynamic_update_slice_in_dim(
        np.reshape(buckets, (self.n_hashes, -1)),
        q_buckets, q_start, axis=1)
    buckets = np.reshape(unflattened_buckets, (-1,))
    is_valid_target = np.any(unflattened_buckets == q_buckets, axis=0)

    assert q_buckets.shape[-1] == 1  # Is true when q_len == 1
    seqlen = x.shape[0]
    arange_seqlen = np.arange(seqlen, dtype=np.int32)
    kv_priorities = np.where(
        arange_seqlen > (q_start + q_len),
        -(seqlen + arange_seqlen), arange_seqlen)
    kv_priorities = kv_priorities + seqlen * is_valid_target.astype(np.int32)
    _, kv_indices = fastmath.sort_key_val(kv_priorities, arange_seqlen)
    kv_indices = kv_indices[
        -self.n_hashes * self.chunk_len * (1 + self.n_chunks_before):]
    assert self.n_chunks_after == 0

    x_attend_to = x[kv_indices]
    k = length_normalized(np.matmul(x_attend_to, w_q))
    v = np.matmul(x_attend_to, w_v)

    mask_fn = functools.partial(
        mask_self_attention, causal=True, masked=True, exclude_self=True)
    q_info = q_start + np.arange(q_len, dtype=np.int32)
    kv_info = kv_indices.astype(np.int32)
    q_info = q_info.astype(np.int32)
    # TODO(kitaev): is it better to mask out attention across buckets?
    # kv_info = np.where(is_valid_target[kv_indices], kv_indices, -kv_indices)
    o, _ = attend(
        q, k, v,
        mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
        dropout=self.attention_dropout, rng=attend_rng,
        )

    out = np.matmul(o, w_o)
    if q_len == 1:
      out = out[:1]
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    buckets_idx = np.array(q_start + q_len, dtype=buckets_idx.dtype)
    return out, (buckets, buckets_idx, hash_rng)


class EncDecAttention(EfficientAttentionBase):
  """Memory-efficient encoder-decoder attention."""

  def __init__(self,
               n_heads=2, d_qk=64, d_v=64,
               masked=True,
               mode='train',
               attention_dropout=0.0,
               output_dropout=0.0,
               n_parallel_heads=None,
               use_python_loop=False,
               use_reference_code=False,
              ):
    super().__init__(
        n_heads=n_heads,
        n_in=(3 if masked else 2),
        n_parallel_heads=n_parallel_heads,
        use_python_loop=use_python_loop,
        use_reference_code=use_reference_code,
        )
    self.d_qk = d_qk
    self.d_v = d_v
    self.masked = masked
    self.mode = mode
    if mode == 'train':
      self.attention_dropout = attention_dropout
      self.output_dropout = output_dropout
    else:
      self.attention_dropout = 0.0
      self.output_dropout = 0.0

  def _kernel_initializer(self, shape, rng):
    # Attention uses Glorot uniform initalization with respect to the *total*
    # dimension of queries/key/values across all heads. We initialize one head
    # at a time in this class, so init.GlorotUniformInitializer won't work.
    # This initialization type is for parity with previous Trax & tensor2tensor
    # Transformers; it's not clear if it's strictly needed for model accuracy.
    lim = np.sqrt(6.0 / (shape[0] + shape[1] * self.n_heads))
    return fastmath.random.uniform(rng, shape, np.float32, -lim, lim)

  def create_weights_unbatched(self, input_signature, rng):
    d_model = input_signature[0].shape[-1]
    d_kv_antecedent = input_signature[1].shape[-1]
    rng_q, rng_k, rng_v, rng_o = fastmath.random.split(rng, 4)
    w_q = self._kernel_initializer((d_model, self.d_qk), rng_q)
    w_k = self._kernel_initializer((d_kv_antecedent, self.d_qk), rng_k)
    w_v = self._kernel_initializer((d_kv_antecedent, self.d_v), rng_v)
    w_o = np.transpose(self._kernel_initializer((d_model, self.d_v), rng_o))
    return (w_q, w_k, w_v, w_o)

  def forward_unbatched(self, q_antecedent, kv_antecedent, mask=None, *,
                        weights, state, rng, update_state):
    del update_state
    attend_rng, output_rng = fastmath.random.split(rng)
    w_q, w_k, w_v, w_o = weights

    q = np.matmul(q_antecedent, w_q)
    k = np.matmul(kv_antecedent, w_k)
    v = np.matmul(kv_antecedent, w_v)

    if not self.masked:
      assert mask is None
      q_info = kv_info = mask_fn = None
    else:
      # mask is a boolean array (True means "is valid token")
      assert mask is not None
      q_info = None
      kv_info = (~mask).astype(np.int32)  # pylint: disable=invalid-unary-operand-type
      def mask_fn(dots, q_info, kv_info):
        del q_info
        mask = kv_info.astype(np.float32)
        dots = dots - 1e9 * mask
        return dots

    o, _ = attend(
        q, k, v,
        mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
        dropout=self.attention_dropout, rng=attend_rng,
        )

    out = np.matmul(o, w_o)
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    return out, state


class LSHFF(base.Layer):
  """Feed-forward block with LSH.

  The original (non-LSH) feed-forward block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  The original block can be slow in decoding due to the need to fetch a lot of
  weights from memory. The LSH block aims to exploit this sparsity. So in the
  first Dense(d_ff) layer, instead of making a full matrix multiplication,
  this block only multiplies by the parts of the weights matrix that have
  the highest chance to give non-0 after Relu. This is determined by taking
  a number of locality-sensitive hashes and masking to only include weights
  that have one hash identical to the multiplied element.
  """

  def __init__(self, d_ff, n_buckets, n_hashes=4, mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a LSH feed-forward block."""
    super().__init__(name=f'LSHFF_{d_ff}')
    self._mode = mode
    self._d_ff = d_ff
    self._n_buckets = n_buckets
    self._n_hashes = n_hashes
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    w1, w2, b2 = self.weights
    x_shape = x.shape
    x = np.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Hash x into hash buckets; x_buckets is [n_hashes, joint_batch].
    x_buckets, _ = hash_vecs(x, self._n_buckets, self._n_hashes, self.rng)

    # Hash w1 into hash buckets; w1_buckets is [n_hashes, d_ff].
    # Note that we use the same self.rng - so the same hash vectors as for x.
    w1_buckets, _ = hash_vecs(w1, self._n_buckets, self._n_hashes, self.rng)

    # Create a mask to determine which x's have the same hash as which w1's.
    # First: just subtract the hashes and make them non-negative.
    hash_mask = (x_buckets[:, :, None] - w1_buckets[:, None, :])**2
    hash_mask = fastmath.stop_gradient(hash_mask)  # make sure no gradients here
    # hash_mask is [n_hashes, joint_batch, d_ff], 0 iff hashes were equal
    hash_mask = 1 - np.minimum(hash_mask, 1)  # now 1 if equal, 0 otherwise
    # we now sum over n_hashes and use min, it's 1 iff any of n_hashes was equal
    hash_mask = np.minimum(np.sum(hash_mask, axis=0), 1)
    hash_mask = hash_mask.astype(np.float32)  # convert to float to use mask

    # First dense layer of the block, with hash masking.
    mid = np.dot(x, w1.T) * hash_mask  # [joint_batch, d_ff]

    # Relu and the second dense layer, as in a standard feed-forward block.
    # Note: we merge the second block into this layer because of future plans,
    # not anything implemented yet. The potential gain would be as follows:
    # in predict mode, we would pre-hash (once) both w1 and w2 and only do
    # matmuls (and memory copies) for the parts that correspond to the hash
    # of the input. The hash of w1 determines which parts of Relu are 0, so
    # it also determines which parts of w2 can be skipped.
    relu = np.where(mid <= 0, np.zeros_like(mid), mid)
    res = np.dot(relu, w2) + b2
    return np.reshape(res, x_shape)  # un-flatten if needed

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature.shape[-1]
    shape_w1 = (self._d_ff, d_model)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_w1, rng_w2, rng_b2 = fastmath.random.split(self.rng, 3)
    w1 = self._kernel_initializer(shape_w1, rng_w1)
    w2 = self._kernel_initializer(shape_w2, rng_w2)
    b2 = self._bias_initializer(shape_b2, rng_b2)
    self.weights = (w1, w2, b2)
