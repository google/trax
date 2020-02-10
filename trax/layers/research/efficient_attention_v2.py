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
(efficient_attention.py) revealed several limitations, which this code attempts
to address:
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
import jax
from jax.scipy.special import logsumexp

from trax.layers import base
from trax.math import numpy as np


####################################################### Functions


def length_normalized(x, epsilon=1e-6):
  variance = np.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / np.sqrt(variance + epsilon)
  return norm_inputs


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
  if causal:
    mask = jax.lax.convert_element_type(jax.lax.lt(q_info, kv_info), np.float32)
    dots = dots - 1e9 * mask
  if exclude_self:
    mask = jax.lax.convert_element_type(jax.lax.eq(q_info, kv_info), np.float32)
    dots = dots - 1e5 * mask
  if masked:
    zeros_like_kv_info = jax.lax.tie_in(kv_info, np.zeros_like(kv_info))
    mask = jax.lax.convert_element_type(
        jax.lax.lt(kv_info, zeros_like_kv_info), np.float32)
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

  if q_info is None:
    q_info = np.arange(q.shape[-2])

  if kv_info is None and not share_qk:
    kv_info = np.arange(v.shape[-2])

  # Split q/k/v into chunks along the time axis, if desired.
  if q_chunk_len is not None:
    q = np.reshape(q, (-1, q_chunk_len, q.shape[-1]))
    q_info = np.reshape(q_info, (-1, q_chunk_len))

  if share_qk:
    assert kv_chunk_len is None or kv_chunk_len == q_chunk_len
    k = q
    kv_chunk_len = q_chunk_len
    kv_info = q_info
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
  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  dots = np.exp(dots - dots_logsumexp)

  if dropout > 0.0:
    assert rng is not None
    # Dropout is broadcast across the bin dimension
    dropout_shape = (dots.shape[-2], dots.shape[-1])
    # TODO(kitaev): verify that tie-in is safe to remove (in light of jax fix)
    keep_prob = jax.lax.tie_in(dots, 1.0 - dropout)
    keep = jax.random.bernoulli(rng, keep_prob, dropout_shape)
    multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
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
    keep_prob = jax.lax.tie_in(vecs, 1.0 - dropout_rate)
    keep = jax.random.bernoulli(rng, keep_prob, (vecs.shape[-1],))
    multiplier = keep.astype(vecs.dtype) / jax.lax.tie_in(keep, keep_prob)
    return vecs * multiplier
  else:
    return vecs


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  def permute_impl(val):
    return np.take(val, permutation, axis=axis)
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      return (np.take(permuted_grad, inverse_permutation, axis=axis),)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  def permute_impl(val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = jax.lax.sort_key_val(keys, val, dimension=axis)
    return permuted
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      _, val_grad = jax.lax.sort_key_val(
          inverse_keys, permuted_grad, dimension=axis)
      return (val_grad,)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


####################################################### Classes


class EfficientAttentionBase(base.Layer):
  """Base class for efficient attention.

  This is a base class that implements memory-efficient batching for both the
  forward and backward passes. Subclasses should override
  `create_weights_unbatched`, `create_state_unbatched`, and `forward_unbatched`
  to define the actual attention mechanism.
  """

  def __init__(self, n_heads, n_in=1, n_parallel_heads=None,
               use_python_loop=False, use_reference_code=False):
    """Construct an EfficientAttentionBase instance.

    Args:
      n_heads: int: Number of attention heads
      n_in: int: Number of inputs to the layer (default 1)
      n_parallel_heads: int: Number of attention heads to compute in parallel.
        if n_parallel_heads is None (default): The entire layer is computed with
          maximum parallelism. This mode is the fastest, but also uses the most
          memory. Start with this mode, but switch to one of the others if
          memory runs out.
        if n_parallel_heads is 1: Attention is computed one head at a time, and
          one example at a time. This mode uses the least memory but is not as
          fast as batched attention. Use this mode when working with very long
          sequences, such that any amount of parallelism won't fit in memory.
        if n_parallel_heads is a multiple of n_heads: Attention is computed for
          sub-batches of (n_parallel_heads // n_heads) examples at a time.
        if 1 < n_parallel_heads < n_heads: Attention is computed for several
          heads at a time, but only within a single example. It must be the case
          that n_heads is a multiple of n_parallel_heads. Use this mode for long
          sequences, to strike a balance between parallelism and memory usage.
      use_python_loop: bool: Set to True to use a Python loop when iterating
        over sub-batches of examples/heads (as opposed to a JAX/XLA loop). This
        option will increase compilation time and jitted code size, potentially
        drastically. Using it is not recommended except for testing/debugging.
        In particular, note that enabling this option on TPU can decrease the
        maximum model size that will fit in memory.
      use_reference_code: bool: Set to True to fall back to the reference
        implementation of batched attention. This option will increase
        compilation time and jitted code size, potentially drastically. Using it
        is not recommended except for testing/debugging.
    """
    super().__init__(n_in=n_in, n_out=1)
    self.n_heads = n_heads
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

  def new_weights_and_state(self, input_signature):
    input_signature_unbatched = jax.tree_map(
        lambda x: type(x)(shape=x.shape[1:], dtype=x.dtype),
        input_signature)
    if isinstance(input_signature, (tuple, list)):
      batch_size = int(input_signature[0].shape[0])
    else:
      batch_size = int(input_signature.shape[0])

    weights = []
    weight_rngs = self.new_rngs(self.n_heads)
    for i in range(self.n_heads):
      weights.append(self.create_weights_unbatched(input_signature_unbatched,
                                                   weight_rngs[i]))
    state = []
    state_rngs = self.new_rngs(self.n_heads * batch_size)
    for i in range(self.n_heads * batch_size):
      state.append(self.create_state_unbatched(input_signature_unbatched,
                                               state_rngs[i]))

    stack_along_axis_0 = lambda *x: np.stack(x, axis=0)
    weights = jax.tree_multimap(stack_along_axis_0, *weights)
    state = jax.tree_multimap(stack_along_axis_0, *state)
    return weights, state

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

  def forward_with_state(self, inputs, weights, state, rng=None):
    """Computes this layer's output as part of a forward pass through the model.

    Args:
      inputs: Layer inputs (subclasses may use different inputs)
      weights: Layer weights
      state: Complete state of the layer
      rng: PRNG key. Note that the RNG is shared across all examples and heads.
        This sharing is useful to reduce memory usage for dropout (all dropout
        instances are automatically broadcasted across the batch and head
        dimensions). Attention types that need separate random numbers for each
        example and head may store their own RNG in the model state.

    Returns:
      A tuple (output, new_state).
    """
    if not self.use_reference_code:
      # By default, an efficient, batched implementation is used.
      output, new_state, _, _ = self.forward_and_or_backward(
          inputs, weights, state, rng, compute_output=True, update_state=True)
      return output, new_state

    # The reference implementation below provides a more readable overview of
    # what this class does. It's not optimized, however, and should only be used
    # when testing this class for correctness.
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    batch_size = int(inputs[0].shape[0])
    seqlen = inputs[0].shape[-2]
    d_model = inputs[0].shape[-1]
    output_accum = [np.zeros((seqlen, d_model)) for _ in range(batch_size)]
    new_state = []
    for example_idx in range(batch_size):
      for head_idx in range(self.n_heads):
        # pylint: disable=cell-var-from-loop
        single_inputs = jax.tree_map(lambda x: x[example_idx], inputs)
        single_weights = jax.tree_map(lambda w: w[head_idx], weights)
        single_state = jax.tree_map(
            lambda s: s[example_idx * self.n_heads + head_idx], state)
        # pylint: enable=cell-var-from-loop
        single_out, single_new_state = self.forward_unbatched(
            *single_inputs, weights=single_weights, rng=rng, state=single_state,
            update_state=True)
        new_state.append(single_new_state)
        output_accum[example_idx] = output_accum[example_idx] + single_out

    output = np.stack(output_accum, 0)
    if new_state and jax.tree_leaves(new_state[0]):
      new_state = jax.tree_multimap(lambda *s: np.stack(s, 0), *new_state)
    else:
      new_state = state
    return output, new_state

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

    See `forward_with_state` for a reference implementation of what this layer
    does. The reference implementation is not very efficient, however, and this
    method provides a more performant version.

    Args:
      inputs: inputs to the attention layer
      weights: weights for the attention layer
      state: state of the attention layer
      rng: PRNG key for the layer (shared across all examples and heads)
      output_grad: gradient of the loss wrt the output of the layer, or None.
        This function performs the backward pass iff `output_grad` is not None.
      compute_output: bool: whether to return the output of the forward pass
        (for example, a pure backwards pass does not need to return the output).
      update_state: bool: whether to return an updated layer state.
    Returns:
      A tuple (output, new_state, inputs_grad, weights_grad).
      - output is not None iff compute_output is True
      - new_state is not None iff update_state is True
      - inputs_grad & weights_grad are not None iff output_grad is not None

    Notes regarding the implementation:
    (a) Multiple heads or examples are batched together. There are three
        different regimes possible: one head at a time (for long sequences and
        expensive attention types), several attention heads at a time (for
        long sequences but less-expensive attention types), and several
        examples at a time (for large batches of shorter sequences). For the
        time being, each of these regimes has its own code.
    (b) Python loops produce large computation graphs when jitted, so the
        default is to use a JAX loop instead.
    (c) No intermediate quantities are cached for the backward pass. Instead,
        the forward pass is re-computed when doing backprop. This approach is
        often called "checkpointing" or "rematerialization". When not all
        examples or heads fit in memory simultaneously, the implementation
        should be [FW-BW-1] and NOT [FW-BW-2], because the latter has worse
        memory locality. I don't think JAX autodiff can synthesize [FW-BW-1]
        automatically, so the looping for the backward pass is done manually.

        [FW-BW-1] for example, head in zip(examples, heads):
                    forward(example, head)
                    backward(example, head)  # uses intermediates from forward

        [FW-BW-2] for example, head in zip(examples, heads):
                    forward(example, head)
                  for example, head in zip(examples, heads):
                    backward(example, head)
    """
    # TODO(kitaev): profile ~4% speed drop compared to previous implementation
    #     in some conditions. Other conditions (e.g. the enwik8 model) appear
    #     to have the same overall training speed.
    # TODO(b/148460708): reduce memory usage further
    # TODO(kitaev): there should be a higher-level API (like vmap) that does
    #     batching, instead of needing 3 separate manual implementations here.

    have_single_input = not isinstance(inputs, (tuple, list))
    if have_single_input:
      inputs = (inputs,)
    batch_size = int(inputs[0].shape[0])
    seqlen = inputs[0].shape[-2]
    d_model = inputs[0].shape[-1]

    compute_grad = (output_grad is not None)
    assert compute_output or compute_grad, 'No work to perform!'

    # Adjust degree of parallelism based on the batch size.
    n_parallel_heads = batch_size * self.n_heads
    if self.n_parallel_heads and self.n_parallel_heads < n_parallel_heads:
      n_parallel_heads = self.n_parallel_heads

    def tree_update(tree, indices, new_values):
      return jax.tree_multimap(
          lambda x, y: jax.ops.index_update(x, jax.ops.index[indices], y),
          tree, new_values)

    def tree_add(tree, indices, new_values):
      return jax.tree_multimap(
          lambda x, y: jax.ops.index_add(x, jax.ops.index[indices], y),
          tree, new_values)

    if compute_grad:
      inputs_is_differentiable = jax.tree_map(
          lambda x: np.issubdtype(x.dtype, np.inexact), inputs)
      def split_differentiable(xs):
        differentiable_xs = jax.tree_multimap(
            lambda x, is_differentiable: x if is_differentiable else None,
            xs, inputs_is_differentiable)
        non_differentiable_xs = jax.tree_multimap(
            lambda x, is_differentiable: None if is_differentiable else x,
            xs, inputs_is_differentiable)
        return differentiable_xs, non_differentiable_xs
      def join_differentiable(differentiable_xs, non_differentiable_xs):
        """Reconstitute inputs pytree from differentiable/non-d. partitions."""
        differentiable_leaves = list(jax.tree_leaves(differentiable_xs))
        non_differentiable_leaves = list(jax.tree_leaves(non_differentiable_xs))
        leaves = []
        for is_differentiable in jax.tree_leaves(inputs_is_differentiable):
          if is_differentiable:
            leaves.append(differentiable_leaves.pop(0))
          else:
            leaves.append(non_differentiable_leaves.pop(0))
        assert not differentiable_leaves
        assert not non_differentiable_leaves
        return jax.tree_unflatten(jax.tree_structure(inputs), leaves)

      def vjp(fn, inp, *args, has_aux=False):
        d_inp, nd_inp = split_differentiable(inp)
        def fn_closed_over_nd_inp(d_inp, *args):
          inp = join_differentiable(d_inp, nd_inp)
          return fn(inp, *args)
        return jax.vjp(fn_closed_over_nd_inp, d_inp, *args, has_aux=has_aux)

    if n_parallel_heads == 1:
      def run_inner(idx, loop_val):
        """Runs one slice of attention (for a single head)."""
        o_all, s_all, i_ct_all, w_ct_all = loop_val
        example_idx = idx // self.n_heads
        head_idx = idx % self.n_heads

        i_h = jax.tree_map(lambda x: x[example_idx], inputs)
        w_h = jax.tree_map(lambda w: w[head_idx], weights)
        s_h = jax.tree_map(lambda s: s[idx], state)

        def forward_fn(i_h, w_h):
          return self.forward_unbatched(
              *i_h, weights=w_h, state=jax.lax.stop_gradient(s_h), rng=rng,
              update_state=update_state)

        if compute_grad:
          o_h, backward_fn, s_h = vjp(forward_fn, i_h, w_h, has_aux=True)
          ct_h = output_grad[example_idx]
          assert o_h.shape == ct_h.shape
          i_ct_h, w_ct_h = backward_fn(ct_h)
        else:
          o_h, s_h = forward_fn(i_h, w_h)

        if compute_output:
          o_all = jax.ops.index_add(o_all, example_idx, o_h)
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
        # Use iota here instead of np.arange, because np.arange will fail to
        # infer that the slice size is a compile-time constant.
        head_range = head_idx_lo + jax.lax.iota(np.int32, n_parallel_heads)
        state_range = idx + jax.lax.iota(np.int32, n_parallel_heads)

        i_mh = jax.tree_map(lambda x: x[example_idx], inputs)
        w_mh = jax.tree_map(lambda w: w[head_range], weights)
        s_mh = jax.tree_map(lambda s: s[state_range], state)
        def forward_unbatched(i_h, w_h, s_h):
          return self.forward_unbatched(
              *i_h, weights=w_h, state=s_h, rng=rng, update_state=update_state)
        def forward_fn(i_mh, w_mh):
          o_mh, new_s_mh = jax.vmap(
              forward_unbatched, in_axes=(None, 0, 0), out_axes=0)(
                  i_mh, w_mh, s_mh)
          o_mh = o_mh.sum(0)
          return o_mh, new_s_mh

        if compute_grad:
          o_mh, backward_fn, s_mh = vjp(forward_fn, i_mh, w_mh, has_aux=True)
          ct_mh = output_grad[example_idx]
          assert o_mh.shape == ct_mh.shape
          i_ct_mh, w_ct_mh = backward_fn(ct_mh)
        else:
          o_mh, s_mh = forward_fn(i_mh, w_mh)

        if compute_output:
          o_all = jax.ops.index_add(o_all, example_idx, o_mh)
        if update_state:
          s_all = tree_update(s_all, state_range, s_mh)
        if compute_grad:
          i_ct_all = tree_add(i_ct_all, example_idx, i_ct_mh)
          w_ct_all = tree_add(w_ct_all, head_range, w_ct_mh)
        return (o_all, s_all, i_ct_all, w_ct_all)
    else:
      assert n_parallel_heads % self.n_heads == 0
      def forward_single_example(i_x, w_all, s_x):
        def forward_unbatched(i_h, w_h, s_h):
          return self.forward_unbatched(
              *i_h, weights=w_h, state=s_h, rng=rng, update_state=update_state)
        o_x, s_x = jax.vmap(
            forward_unbatched, in_axes=(None, 0, 0), out_axes=(0, 0))(
                i_x, w_all, s_x)
        o_x = o_x.sum(0)
        return o_x, s_x
      def run_inner(idx, loop_val):
        """Runs one slice of attention (all heads for one or more examples)."""
        o_all, s_all, i_ct_all, w_ct_all = loop_val
        idx = idx * n_parallel_heads
        example_idx_lo = idx // self.n_heads
        # Use iota here instead of np.arange, because np.arange will fail to
        # infer that the slice size is a compile-time constant.
        example_range = example_idx_lo + jax.lax.iota(
            np.int32, n_parallel_heads // self.n_heads)
        state_range = idx + jax.lax.iota(np.int32, n_parallel_heads)

        i_mex = jax.tree_map(lambda x: x[example_range], inputs)
        s_mex = jax.tree_map(
            lambda s: np.reshape(s[state_range],  # pylint: disable=g-long-lambda
                                 (-1, self.n_heads) + s.shape[1:]),
            state)
        def forward_fn(i_mex, w_all):
          o_mex, new_s_mex = jax.vmap(
              forward_single_example, in_axes=(0, None, 0), out_axes=(0, 0))(
                  i_mex, w_all, s_mex)
          new_s_mex = jax.tree_map(
              lambda s: np.reshape(s, (n_parallel_heads,) + s.shape[2:]),
              new_s_mex)
          return o_mex, new_s_mex

        if compute_grad:
          o_mex, backward_fn, s_mex = vjp(forward_fn, i_mex, weights,
                                          has_aux=True)
          ct_mex = output_grad[example_range]
          assert o_mex.shape == ct_mex.shape
          i_ct_mex, w_ct_mex = backward_fn(ct_mex)
        else:
          o_mex, s_mex = forward_fn(i_mex, weights)

        if compute_output:
          o_all = jax.ops.index_add(o_all, jax.ops.index[example_range], o_mex)
        if update_state:
          s_all = tree_update(s_all, state_range, s_mex)
        if compute_grad:
          i_ct_all = tree_update(i_ct_all, example_range, i_ct_mex)
          w_ct_all = jax.tree_multimap(
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
      i_ct_all = jax.tree_map(np.zeros_like, inputs)
      i_ct_all, i_nondifferentiable_dummy_ct = split_differentiable(i_ct_all)
      w_ct_all = jax.tree_map(np.zeros_like, weights)

    loop_val = (o_all, s_all, i_ct_all, w_ct_all)

    assert (batch_size * self.n_heads) % n_parallel_heads == 0
    loop_hi = (batch_size * self.n_heads) // n_parallel_heads
    if self.use_python_loop or loop_hi == 1:
      for idx in range(loop_hi):
        loop_val = run_inner(idx, loop_val)
    else:
      loop_val = jax.lax.fori_loop(
          0, loop_hi, run_inner, loop_val)

    (o_all, s_all, i_ct_all, w_ct_all) = loop_val

    if compute_grad:
      i_ct_all = join_differentiable(i_ct_all, i_nondifferentiable_dummy_ct)

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
               mode='train',
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
      mode: 'train' or 'eval'
      attention_dropout: Dropout probability for attention mask.
      output_dropout: Dropout probability for the layer output.
      n_parallel_heads: see EfficientAttentionBase. This option controls the
        trade-off between parallelism and memory usage.
      use_python_loop: For testing/debugging (see EfficientAttentionBase)
      use_reference_code: For testing/debugging (see EfficientAttentionBase)
    """
    super().__init__(
        n_heads=n_heads,
        n_in=(2 if masked else 1),
        n_parallel_heads=n_parallel_heads,
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
    return jax.random.uniform(rng, shape, np.float32, -lim, lim)

  def create_weights_unbatched(self, input_signature, rng):
    if isinstance(input_signature, (tuple, list)):
      input_signature = input_signature[0]
    d_model = input_signature.shape[-1]
    rng_q, rng_k, rng_v, rng_o = jax.random.split(rng, 4)
    w_q = self._kernel_initializer((d_model, self.d_qk), rng_q)
    if not self.share_qk:
      w_k = self._kernel_initializer((d_model, self.d_qk), rng_k)
    w_v = self._kernel_initializer((d_model, self.d_v), rng_v)
    w_o = np.transpose(self._kernel_initializer((d_model, self.d_v), rng_o))
    if self.share_qk:
      return (w_q, w_v, w_o)
    else:
      return (w_q, w_k, w_v, w_o)

  def forward_unbatched(self, x, mask=None, *,
                        weights, state, rng, update_state):
    del update_state
    attend_rng, output_rng = jax.random.split(rng)
    if self.share_qk:
      w_q, w_v, w_o = weights
    else:
      w_q, w_k, w_v, w_o = weights

    q = np.matmul(x, w_q)
    k = None
    if not self.share_qk:
      k = np.matmul(x, w_k)
    v = np.matmul(x, w_v)

    mask_fn = functools.partial(
        mask_self_attention,
        causal=self.causal, exclude_self=self.share_qk, masked=self.masked)
    q_info = kv_info = jax.lax.tie_in(x, np.arange(q.shape[-2]))

    assert (mask is not None) == self.masked
    if self.masked:
      # mask is a boolean array (True means "is valid token")
      ones_like_mask = jax.lax.tie_in(x, np.ones_like(mask, dtype=np.int32))
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


class LSHSelfAttention(SelfAttention):
  """LSH self-attention (second implementation)."""

  def __init__(self,
               n_heads=2, d_qk=64, d_v=64, share_qk='unused',
               causal=False,
               chunk_len=None, n_chunks_before=1, n_chunks_after=0,
               n_hashes=1,
               n_buckets=256,
               mode='train',
               attention_dropout=0.0,
               output_dropout=0.0,
               n_parallel_heads=1,
               use_python_loop=False,
               use_reference_code=False,
              ):
    """Construct an LSH self-attention layer."""
    super().__init__(
        n_heads=n_heads, d_qk=d_qk, d_v=d_v, share_qk=True,
        causal=causal,
        chunk_len=chunk_len,
        n_chunks_before=n_chunks_before, n_chunks_after=n_chunks_after,
        mode=mode,
        attention_dropout=attention_dropout,
        output_dropout=output_dropout,
        n_parallel_heads=n_parallel_heads,
        use_python_loop=use_python_loop,
        use_reference_code=use_reference_code,
        )
    self.n_hashes = n_hashes
    self.n_buckets = n_buckets

  def create_state_unbatched(self, input_signature, rng):
    if isinstance(input_signature, (tuple, list)):
      input_signature = input_signature[0]
    buckets = np.zeros(self.n_hashes * input_signature.shape[0], dtype=np.int32)
    # The `rng` argument passed to forward_unbatched is shared across all
    # examples and heads. This facilitates using broadcasted dropout, which
    # saves memory and hasn't been shown to hurt model quality. Even though the
    # same sharing is likely to be safe when selecting random hash functions
    # for LSH, we haven't run experiments to demonstrate this. To be on the safe
    # side we include a per-head RNG in the state for the purpose of doing LSH.
    return (buckets, rng)

  def hash_vectors(self, vecs, rng):
    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    if isinstance(self.n_buckets, int):
      assert self.n_buckets % 2 == 0
      rot_size = self.n_buckets
      n_buckets = self.n_buckets
    else:
      # Factorize the hash if self.n_buckets is a list or tuple
      rot_size, n_buckets = 0, 1
      for factor in self.n_buckets:
        assert factor % 2 == 0
        rot_size += factor
        n_buckets *= factor

    rotations_shape = (vecs.shape[-1], self.n_hashes, rot_size // 2)

    rng = jax.lax.stop_gradient(jax.lax.tie_in(vecs, rng))
    random_rotations = jax.random.normal(rng, rotations_shape).astype('float32')
    rotated_vecs = np.einsum('tf,fhb->htb', vecs, random_rotations)

    if isinstance(self.n_buckets, int) or len(self.n_buckets) == 1:
      rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
      buckets = np.argmax(rotated_vecs, axis=-1)
    else:
      # Get the buckets for them and combine.
      buckets, cur_sum, cur_product = None, 0, 1
      for factor in self.n_buckets:
        rv = rotated_vecs[..., cur_sum:cur_sum + (factor // 2)]
        cur_sum += factor // 2
        rv = np.concatenate([rv, -rv], axis=-1)
        if buckets is None:
          buckets = np.argmax(rv, axis=-1)
        else:
          buckets += cur_product * np.argmax(rv, axis=-1)
        cur_product *= factor

    # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
    # bucket numbers from different hashing rounds don't overlap.
    offsets = jax.lax.tie_in(buckets, np.arange(self.n_hashes))
    offsets = np.reshape(offsets * n_buckets, (-1, 1))
    buckets = np.reshape(buckets + offsets, (-1,))

    return buckets

  def forward_unbatched(self, x, *, weights, state, rng, update_state):
    attend_rng, output_rng = jax.random.split(rng)
    w_q, w_v, w_o = weights

    q = np.matmul(x, w_q)
    v = np.matmul(x, w_v)

    if update_state:
      _, old_hash_rng = state
      hash_rng, hash_subrng = jax.random.split(old_hash_rng)
      buckets = self.hash_vectors(q, hash_subrng)
      state = (buckets, hash_rng)
    else:
      buckets, _ = state

    seqlen = x.shape[0]
    assert int(buckets.shape[0]) == self.n_hashes * seqlen

    ticker = jax.lax.tie_in(x, np.arange(self.n_hashes * seqlen))
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
    sq = np.take(q, st, axis=0)
    sv = np.take(v, st, axis=0)

    mask_fn = functools.partial(
        mask_self_attention, causal=self.causal, exclude_self=True)
    q_info = st
    so, slogits = attend(
        sq, k=None, v=sv,
        q_chunk_len=self.chunk_len,
        n_chunks_before=self.n_chunks_before,
        n_chunks_after=self.n_chunks_after,
        mask_fn=mask_fn, q_info=q_info,
        dropout=self.attention_dropout, rng=attend_rng,
        )

    # np.take(so, undo_sort, axis=0); np.take(slogits, undo_sort, axis=0) would
    # also work, but these helpers include performance optimizations for TPU.
    o = permute_via_gather(so, undo_sort, sticker, axis=0)
    logits = permute_via_sort(slogits, sticker, buckets_and_t, axis=-1)

    if self.n_hashes > 1:
      o = np.reshape(o, (self.n_hashes, seqlen, o.shape[-1]))
      logits = np.reshape(logits, (self.n_hashes, seqlen, 1))
      probs = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
      o = np.sum(o * probs, axis=0)

    assert o.shape == (seqlen, w_v.shape[-1])
    out = np.matmul(o, w_o)
    out = apply_broadcasted_dropout(out, self.output_dropout, output_rng)
    return out, state


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
    return jax.random.uniform(rng, shape, np.float32, -lim, lim)

  def create_weights_unbatched(self, input_signature, rng):
    d_model = input_signature[0].shape[-1]
    d_kv_antecedent = input_signature[1].shape[-1]
    rng_q, rng_k, rng_v, rng_o = jax.random.split(rng, 4)
    w_q = self._kernel_initializer((d_model, self.d_qk), rng_q)
    w_k = self._kernel_initializer((d_kv_antecedent, self.d_qk), rng_k)
    w_v = self._kernel_initializer((d_kv_antecedent, self.d_v), rng_v)
    w_o = np.transpose(self._kernel_initializer((d_model, self.d_v), rng_o))
    return (w_q, w_k, w_v, w_o)

  def forward_unbatched(self, q_antecedent, kv_antecedent, mask=None, *,
                        weights, state, rng, update_state):
    del update_state
    attend_rng, output_rng = jax.random.split(rng)
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
        mask = jax.lax.convert_element_type(kv_info, np.float32)
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
