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
"""Attention Layers optimized for efficiency.

This file continues a journey of optimized attention implementations that
started in the trax framework; see
https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py

Implementation notes:
1. Many attention implementations compute O(n^2) query-key dot products all in
   parallel, which can easily use up all available memory. However, there is no
   requirement to compute all dot products in parallel, and instead attention
   can be run for a subset of queries at a time. The attention implementations
   here are designed to have configurable chunking. Further optimizatons such
   as local attention and LSH attention are primarily aimed at reducing training
   time, and not memory usage.
2. Once chunking is in place, the next potential way to run out of memory is to
   simultaneously instantiate queries, keys, and values for all heads at the
   same time. Transformers are typically tuned such that
   num_heads * d_attention_key == d_model. Since attention involves queries,
   keys, and values, the memory to store them can be ~3x the memory needed to
   store the input activations. Therefore, each chunk of the computation is
   responsible for its own query/key/value/output projections.
3. Attention masking is implemented by associating an integer (typically, the
   sequence position) with each query and key vector, and defining a function
   to compute attention masks from this information. The flax attention
   built-ins pass around O(n^2)-size attention mask tensors instead, which is
   not scalable for long sequences. Many Transformer implementations opt to
   compute this large mask tensor once and then re-use it across all layers of
   the model. This can save on compute, but it incurs a memory cost that also
   impacts the maximum memory available to other layers (e.g. feed-forward and
   output softmax layers). Computing full masks on-demand may be a bit slower,
   but we deem this tradeoff worth it because of the memory savings it brings.
4. It is our observation that for long sequences, the speed of an attention
   mechanism is limited not by the number of floating point operations (such as
   dot products), but rather by memory access speeds.
"""

import functools

from flax import nn
from flax.examples.google.efficient_transformers import multihead
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


NEG_INFINITY = -1e9


class MultiHeadWrapper(nn.Module):
  """Wrapper for batching attention across examples and heads."""

  def apply(self, *args, wrapped_module,
            num_heads=1, num_parallel_heads=None, use_python_loop=False,
            **kwargs):
    # Re-use the same rng key across all examples and heads. This will result in
    # broadcasted dropout, which saves memory.
    # TODO(kitaev): options to swap broadcasted RNG on/off
    rng = nn.make_rng() if nn.is_stochastic() else None

    def init_single_head(init_rng, args, kwargs):
      if rng is None:
        _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      return head_params

    def init_wrapped_module(rng, unused_shape):
      single_example_args = jax.tree_map(lambda x: x[:1], args)
      return multihead.chunked_multihead_map(
          init_single_head,
          in_has_batch_dim=(False, True, False),
          in_has_head_dim=(True, False, False),
          out_has_batch_dim=False,
          out_has_head_dim=True,
          use_python_loop=True,
          )(jax.random.split(rng, num_heads), single_example_args, kwargs)
    # TODO(kitaev): The original intent was to have this be a transparent module
    # but for some reason naming this parameter '0' and inheriting from
    # nn.base.TransparentModule is not enough to stop this parameter name from
    # explicitly showing up in the parameter tree.
    params = self.param('attn', None, init_wrapped_module)

    def run_single_example_and_head(params, args, kwargs):
      if rng is None:
        return wrapped_module.call(params, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          return wrapped_module.call(params, *args, **kwargs)

    return multihead.chunked_multihead_map(
        run_single_example_and_head,
        in_has_batch_dim=(False, True, False),
        in_has_head_dim=(True, False, False),
        out_has_batch_dim=True,
        out_has_head_dim=False,
        num_parallel_heads=num_parallel_heads,
        use_python_loop=use_python_loop,
    )(params, args, kwargs)


def make_multihead(module_type):
  return MultiHeadWrapper.partial(wrapped_module=module_type)


class ManuallyBatchedAttentionWrapper(nn.Module):
  """Wrapper for manually batched attention."""

  def apply(self, *args, wrapped_module, **kwargs):
    # An extra 'attn' scope is needed to match param structure with attention
    # types that use make_multihead.
    return wrapped_module(*args, name='attn', **kwargs)


def not_multihead(module_type):
  return ManuallyBatchedAttentionWrapper.partial(wrapped_module=module_type)


@make_multihead
class BertSelfAttention(nn.Module):
  """Masked dot-product self-attention."""

  def apply(self,
            hidden_states, mask=None, *,
            d_qkv=64,
            attention_dropout_rate=0.0,
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True):
    """Applies attention for a single batch element and head."""
    d_model = hidden_states.shape[-1]
    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(d_qkv,),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)
    query, key, value = (dense(hidden_states, name='query'),
                         dense(hidden_states, name='key'),
                         dense(hidden_states, name='value'))
    attention_scores = jnp.einsum('TN,FN->FT', key, query)
    attention_scores = attention_scores / jnp.sqrt(d_qkv)
    if mask is not None:
      padding_mask = (1.0 - mask[None, :]) * NEG_INFINITY
      attention_scores = attention_scores + padding_mask
    attention_scores = nn.softmax(attention_scores)
    attention_probs = nn.dropout(
        attention_scores, rate=attention_dropout_rate,
        deterministic=deterministic)
    hidden_states = jnp.einsum('FT,TH->FH', attention_probs, value)
    hidden_states = nn.linear.DenseGeneral(
        hidden_states,
        features=d_model,
        axis=(-1,),
        kernel_init=output_kernel_init,
        name='output')
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states


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
      slices.append(jnp.concatenate([x[i:, ...], x[:i, ...]], axis=0))
  return jnp.concatenate(slices, axis=1)


def length_normalized(x, epsilon=1e-6):
  variance = jnp.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / jnp.sqrt(variance + epsilon)
  return norm_inputs


def mask_self_attention(
    dots, q_info, kv_info, causal=True, exclude_self=True, masked=False):
  """Performs masking for self-attention."""
  if causal:
    mask = jax.lax.convert_element_type(
        jax.lax.lt(q_info, kv_info), jnp.float32)
    dots = dots - 1e9 * mask
  if exclude_self:
    mask = jax.lax.convert_element_type(
        jax.lax.eq(q_info, kv_info), jnp.float32)
    dots = dots - 1e5 * mask
  if masked:
    zeros_like_kv_info = jax.lax.tie_in(kv_info, jnp.zeros_like(kv_info))
    mask = jax.lax.convert_element_type(
        jax.lax.lt(kv_info, zeros_like_kv_info), jnp.float32)
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
    mask_fn: TODO(kitaev): doc
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
    q_info = jnp.arange(q.shape[-2])

  if kv_info is None and not share_qk:
    kv_info = jnp.arange(v.shape[-2])

  # Split q/k/v into chunks along the time axis, if desired.
  if q_chunk_len is not None:
    q = jnp.reshape(q, (-1, q_chunk_len, q.shape[-1]))
    q_info = jnp.reshape(q_info, (-1, q_chunk_len))

  if share_qk:
    assert kv_chunk_len is None or kv_chunk_len == q_chunk_len
    k = q
    kv_chunk_len = q_chunk_len
    if kv_info is None:
      kv_info = q_info
    elif kv_chunk_len is not None:
      # kv_info is not None, but reshape as required.
      kv_info = jnp.reshape(kv_info, (-1, kv_chunk_len))
  elif kv_chunk_len is not None:
    k = jnp.reshape(k, (-1, kv_chunk_len, k.shape[-1]))
    kv_info = jnp.reshape(kv_info, (-1, kv_chunk_len))

  if kv_chunk_len is not None:
    v = jnp.reshape(v, (-1, kv_chunk_len, v.shape[-1]))

  if share_qk:
    k = length_normalized(k)
  k = k / jnp.sqrt(k.shape[-1])

  # Optionally include adjacent chunks.
  if q_chunk_len is not None or kv_chunk_len is not None:
    assert q_chunk_len is not None and kv_chunk_len is not None
  else:
    assert n_chunks_before == 0 and n_chunks_after == 0

  k = look_adjacent(k, n_chunks_before, n_chunks_after)
  v = look_adjacent(v, n_chunks_before, n_chunks_after)
  kv_info = look_adjacent(kv_info, n_chunks_before, n_chunks_after)

  # Dot-product attention.
  dots = jnp.matmul(q, jnp.swapaxes(k, -1, -2))

  # Masking
  if mask_fn is not None:
    dots = mask_fn(dots, q_info[..., :, None], kv_info[..., None, :])

  # Softmax.
  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  dots = jnp.exp(dots - dots_logsumexp)

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
  out = jnp.matmul(dots, v)
  out = jnp.reshape(out, (-1, out.shape[-1]))
  dots_logsumexp = jnp.reshape(dots_logsumexp, (-1,))
  return out, dots_logsumexp


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here. The most likely cause is that
  # it can't close over values: https://github.com/google/jax/issues/2676
  # The error only occurs in some configurations (e.g. use_python_loop = True,
  # num_parallel_heads = 1) but not others.
  permutation = jax.lax.stop_gradient(permutation)
  inverse_permutation = jax.lax.stop_gradient(inverse_permutation)
  def permute_impl(val):
    return jnp.take(val, permutation, axis=axis)
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      return (jnp.take(permuted_grad, inverse_permutation, axis=axis),)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here (see permute_via_gather).
  keys = jax.lax.stop_gradient(keys)
  inverse_keys = jax.lax.stop_gradient(inverse_keys)
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


@make_multihead
class SlowSmyrfSelfAttention(nn.Module):
  """Research in progress -- jax batching for this is inefficient."""

  def apply(self,
            hidden_states, mask=None, *,
            d_qkv=64,
            attention_dropout_rate='ignored',
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True,
            cluster_size=16,
            n_hashes=2,
            n_hashes_for_eval=None,
            ):
    """Applies attention for a single batch element and head."""
    if deterministic and n_hashes_for_eval:
      n_hashes = n_hashes_for_eval

    d_model = hidden_states.shape[-1]
    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(d_qkv,),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)
    query, key, value = (dense(hidden_states, name='query'),
                         dense(hidden_states, name='key'),
                         dense(hidden_states, name='value'))

    seqlen = query.shape[0]

    q_norms = jnp.sqrt(jnp.sum(query ** 2, axis=-1, keepdims=True))
    k_norms = jnp.sqrt(jnp.sum(key ** 2, axis=-1, keepdims=True))
    q_max = jnp.max(q_norms, axis=-2, keepdims=True)
    k_max = jnp.max(k_norms, axis=-2, keepdims=True)

    q_ext = jnp.sqrt(q_max ** 2 + k_max ** 2 - q_norms ** 2)
    q_zero = jnp.zeros_like(q_ext)
    q_mod = jnp.concatenate([query, q_zero, q_ext], axis=-1)

    k_ext = jnp.sqrt(q_max ** 2 + k_max ** 2 - k_norms ** 2)
    k_zero = jnp.zeros_like(k_ext)
    k_mod = jnp.concatenate([key, k_ext, k_zero], axis=-1)

    alpha = jax.random.normal(nn.make_rng(), (q_mod.shape[-1], n_hashes))
    q_proj = jnp.matmul(q_mod, alpha)
    k_proj = jnp.matmul(k_mod, alpha)

    if mask is not None:
      # Relocate padding elements
      q_proj = q_proj + (1.0 - mask[:, None]) * 1e8
      k_proj = k_proj + (1.0 - mask[:, None]) * 1e8

    q_proj = jnp.swapaxes(q_proj, 0, 1)
    k_proj = jnp.swapaxes(k_proj, 0, 1)

    q_sort = jnp.argsort(q_proj, axis=1)
    k_sort = jnp.argsort(k_proj, axis=1)

    ticker = jax.lax.tie_in(query, jnp.arange(n_hashes * seqlen))
    sticker = jnp.reshape(
        q_sort + seqlen * jnp.arange(n_hashes)[:, None], (-1,))

    sq_t = jnp.reshape(q_sort, (-1,))
    skv_t = jnp.reshape(k_sort, (-1,))
    _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)

    # Hash-based sort ("s" at the start of variable names means "sorted")
    sq = jnp.take(query, sq_t, axis=0)
    sk = jnp.take(key, skv_t, axis=0)
    sv = jnp.take(value, skv_t, axis=0)
    kv_info = jnp.take(mask, skv_t, axis=0)
    kv_info = jnp.where(kv_info, kv_info, kv_info - 1)

    mask_fn = functools.partial(
        mask_self_attention, causal=False, exclude_self=False, masked=True)
    so, slogits = attend(
        sq, sk, sv,
        q_chunk_len=cluster_size,
        kv_chunk_len=cluster_size,
        n_chunks_before=0,
        n_chunks_after=0,
        mask_fn=mask_fn, kv_info=kv_info,
        )

    # jnp.take(so, undo_sort, axis=0); jnp.take(slogits, undo_sort, axis=0)
    # would also work, but these helpers include performance optimizations for
    # TPU.
    o = permute_via_gather(so, undo_sort, sticker, axis=0)
    logits = permute_via_sort(slogits, sticker, undo_sort, axis=0)

    if n_hashes > 1:
      o = jnp.reshape(o, (n_hashes, seqlen, o.shape[-1]))
      logits = jnp.reshape(logits, (n_hashes, seqlen, 1))
      probs = jnp.exp(logits - logsumexp(logits, axis=0, keepdims=True))
      o = jnp.sum(o * probs, axis=0)

    hidden_states = o
    hidden_states = nn.linear.DenseGeneral(
        hidden_states,
        features=d_model,
        axis=(-1,),
        kernel_init=output_kernel_init,
        name='output')
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states


class MyDense(nn.Module):
  """Manually batched dense projection to produce query/key/value vectors."""

  def apply(self, hidden_states, axis, features, kernel_init, bias_init, bias):
    d_model = hidden_states.shape[-1]
    assert axis == -1
    num_heads, d_qkv = features
    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      del shape, dtype
      kernel = kernel_init(rng, (d_model, num_heads * d_qkv))
      kernel = jnp.reshape(kernel, (d_model, num_heads, d_qkv))
      return jnp.swapaxes(kernel, 0, 1)

    kernel = self.param('kernel', (num_heads, d_model, d_qkv), kernel_init_wrap)
    bias = self.param('bias', (num_heads, d_qkv), bias_init)
    return jnp.einsum('BFM,NMQ->BFNQ', hidden_states, kernel) + bias


class MyDenseOut(nn.Module):
  """Manually batched dense projection to produce attention output vectors."""

  def apply(self, o, num_heads, d_qkv, d_model, kernel_init, bias_init, bias):
    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      del shape, dtype
      return jnp.reshape(
          kernel_init(rng, (num_heads * d_qkv, d_model)),
          (num_heads, d_qkv, d_model))
    kernel = self.param('kernel', (num_heads, d_qkv, d_model), kernel_init_wrap)
    bias = self.param('bias', (num_heads, d_model), bias_init)
    hidden_states = jnp.einsum('BNFV,NVM->BFM', o, kernel) + jnp.sum(bias, 0)
    return hidden_states


def clustered_take(val, indices, inverse_indices):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here. The most likely cause is that
  # it can't close over values: https://github.com/google/jax/issues/2676
  # The error only occurs in some configurations (e.g. use_python_loop = True,
  # num_parallel_heads = 1) but not others.
  indices = jax.lax.stop_gradient(indices)
  inverse_indices = jax.lax.stop_gradient(inverse_indices)
  def permute_impl(val):
    flat_val = jnp.reshape(val, (-1, val.shape[-1]))
    return jnp.take(flat_val, indices, axis=0)
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      permuted_grad = jnp.reshape(
          permuted_grad, (-1, permuted_grad.shape[-1]))
      inp_grad = jnp.take(permuted_grad, inverse_indices, axis=0)
      inp_grad = jnp.reshape(inp_grad, (
          val.shape[0], val.shape[2], -1, val.shape[1], val.shape[-1]))
      inp_grad = jnp.sum(inp_grad, 2)
      inp_grad = jnp.swapaxes(inp_grad, 1, 2)
      return (inp_grad,)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


@not_multihead
class SmyrfSelfAttention(nn.Module):
  """Research in progress."""

  def apply(self,
            hidden_states, mask=None, *,
            num_heads=2,
            num_parallel_heads='ignored',
            d_qkv=64,
            attention_dropout_rate='ignored',
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True,
            cluster_size=16,
            n_hashes=2,
            n_hashes_for_eval=None,
            optimize_for_tpu=True,
            ):
    """Applies attention for all examples and heads."""
    if deterministic and n_hashes_for_eval:
      n_hashes = n_hashes_for_eval

    d_model = hidden_states.shape[-1]

    dense = MyDense.partial(
        axis=-1,
        features=(num_heads, d_qkv),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)

    # project inputs to multi-headed q/k/v
    query, key, value = (dense(hidden_states, name='query'),
                         dense(hidden_states, name='key'),
                         dense(hidden_states, name='value'))

    # batch x time x num_heads x d_qkv
    batch_size = query.shape[0]
    seqlen = query.shape[1]

    q_norms = jnp.sqrt(jnp.sum(query ** 2, axis=-1, keepdims=True))
    k_norms = jnp.sqrt(jnp.sum(key ** 2, axis=-1, keepdims=True))
    q_max = jnp.max(q_norms, axis=1, keepdims=True)
    k_max = jnp.max(k_norms, axis=1, keepdims=True)

    q_ext = jnp.sqrt(q_max ** 2 + k_max ** 2 - q_norms ** 2)
    q_zero = jnp.zeros_like(q_ext)
    q_mod = jnp.concatenate([query, q_zero, q_ext], axis=-1)

    k_ext = jnp.sqrt(q_max ** 2 + k_max ** 2 - k_norms ** 2)
    k_zero = jnp.zeros_like(k_ext)
    k_mod = jnp.concatenate([key, k_ext, k_zero], axis=-1)

    alpha = jax.random.normal(nn.make_rng(), (q_mod.shape[-1], n_hashes))
    q_proj = jnp.matmul(q_mod, alpha)
    k_proj = jnp.matmul(k_mod, alpha)

    if mask is not None:
      # Relocate padding elements
      q_proj = q_proj + (1.0 - mask[:, :, None, None]) * 1e8
      k_proj = k_proj + (1.0 - mask[:, :, None, None]) * 1e8

    # batch, time, head, hash -> batch, head, hash, time
    q_proj = jnp.moveaxis(q_proj, 1, -1)
    k_proj = jnp.moveaxis(k_proj, 1, -1)
    sq_t = jnp.argsort(q_proj, axis=-1)
    skv_t = jnp.argsort(k_proj, axis=-1)
    # sort: batch, time, head, d_qkv -> batch, head, hash, time, d_qkv
    q_sort = jnp.reshape(
        jnp.arange(batch_size)[:, None, None, None] * seqlen * num_heads
        + sq_t * num_heads
        + jnp.arange(num_heads)[None, :, None, None],
        (-1,))
    kv_sort = jnp.reshape(
        jnp.arange(batch_size)[:, None, None, None] * seqlen * num_heads
        + skv_t * num_heads
        + jnp.arange(num_heads)[None, :, None, None],
        (-1,))

    # unsort: batch, head, hash, time, d_qkv -> batch, head, hash, time, d_qkv
    q_unsort = jnp.argsort(sq_t, axis=-1)
    q_unsort = jnp.reshape(
        jnp.arange(batch_size)[:, None, None, None] * (num_heads
                                                       * n_hashes * seqlen)
        + jnp.arange(num_heads)[None, :, None, None] * n_hashes * seqlen
        + jnp.arange(n_hashes)[None, None, :, None] * seqlen
        + q_unsort,
        (-1,))
    q_resort = jnp.argsort(q_unsort, -1)
    kv_unsort = jnp.argsort(skv_t, axis=-1)
    kv_unsort = jnp.reshape(
        jnp.arange(batch_size)[:, None, None, None] * (num_heads
                                                       * n_hashes * seqlen)
        + jnp.arange(num_heads)[None, :, None, None] * n_hashes * seqlen
        + jnp.arange(n_hashes)[None, None, :, None] * seqlen
        + kv_unsort,
        (-1,))

    # Hash-based sort ("s" at the start of variable names means "sorted")
    if optimize_for_tpu:
      q_sort = jnp.reshape(q_sort, (-1, cluster_size))
      kv_sort = jnp.reshape(kv_sort, (-1, cluster_size))
      kv = jnp.concatenate([key, value], -1)
      sq = clustered_take(query, q_sort, q_unsort)
      skv = clustered_take(kv, kv_sort, kv_unsort)
      sk = skv[..., :d_qkv]
      sv = skv[..., -d_qkv:]
      flat_mask = jnp.reshape(mask, (-1,))
      kv_info = jnp.take(flat_mask, kv_sort // num_heads, axis=0)
      kv_info = jnp.where(kv_info, kv_info, kv_info - 1)
    else:
      sq = jnp.take(jnp.reshape(query, (-1, d_qkv)), q_sort, axis=0)
      sk = jnp.take(jnp.reshape(key, (-1, d_qkv)), kv_sort, axis=0)
      sv = jnp.take(jnp.reshape(value, (-1, d_qkv)), kv_sort, axis=0)
      flat_mask = jnp.reshape(mask, (-1,))
      kv_info = jnp.take(flat_mask, kv_sort // num_heads, axis=0)
      kv_info = jnp.where(kv_info, kv_info, kv_info - 1)
      sq = jnp.reshape(sq, (-1, cluster_size, d_qkv))
      sk = jnp.reshape(sk, (-1, cluster_size, d_qkv))
      sv = jnp.reshape(sv, (-1, cluster_size, d_qkv))
      kv_info = jnp.reshape(kv_info, (-1, cluster_size))

    mask_fn = functools.partial(
        mask_self_attention, causal=False, exclude_self=False, masked=True)
    so, slogits = attend(
        sq, sk, sv,
        mask_fn=mask_fn, kv_info=kv_info,
        )
    so = jnp.reshape(so, (-1, so.shape[-1]))
    slogits = jnp.reshape(slogits, (-1,))

    # jnp.take(so, q_unsort, axis=0); jnp.take(slogits, q_unsort, axis=0) would
    # also work, but these helpers include performance optimizations for TPU.
    o = permute_via_gather(so, q_unsort, q_resort, axis=0)
    logits = permute_via_sort(slogits, q_resort, q_unsort, axis=0)

    if n_hashes > 1:
      o = jnp.reshape(o, (batch_size, num_heads, n_hashes, seqlen, o.shape[-1]))
      logits = jnp.reshape(logits, (batch_size, num_heads, n_hashes, seqlen, 1))
      probs = jnp.exp(logits - logsumexp(logits, axis=2, keepdims=True))
      o = jnp.sum(o * probs, axis=2)
    else:
      o = jnp.reshape(o, (batch_size, num_heads, seqlen, o.shape[-1]))

    hidden_states = MyDenseOut(
        o, num_heads, d_qkv, d_model, output_kernel_init, bias_init, bias,
        name='output')
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states


@make_multihead
class LSHSelfAttention(nn.Module):
  """LSH self-attention, like the Reformer."""

  def apply(self,
            hidden_states, mask=None, *,
            d_qkv=64,
            attention_dropout_rate='ignored',
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True,
            causal=True,
            chunk_len=64, n_chunks_before=1, n_chunks_after=0,
            n_hashes=1,
            n_buckets=8,
            ):
    """Applies attention for a single batch element and head."""
    d_model = hidden_states.shape[-1]
    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(d_qkv,),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)
    q, v = (dense(hidden_states, name='query'),
            dense(hidden_states, name='value'))

    seqlen = q.shape[0]

    buckets = self.hash_vectors(
        q, nn.make_rng(), mask, num_buckets=n_buckets, num_hashes=n_hashes)

    ticker = jax.lax.tie_in(hidden_states, jnp.arange(n_hashes * seqlen))
    buckets_and_t = seqlen * buckets + (ticker % seqlen)
    buckets_and_t = jax.lax.stop_gradient(buckets_and_t)

    sbuckets_and_t, sticker = jax.lax.sort_key_val(
        buckets_and_t, ticker, dimension=-1)
    _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
    sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
    sticker = jax.lax.stop_gradient(sticker)
    undo_sort = jax.lax.stop_gradient(undo_sort)

    st = (sticker % seqlen)
    sq = jnp.take(q, st, axis=0)
    sv = jnp.take(v, st, axis=0)
    q_info = st

    if mask is None:
      kv_info = None
    else:
      smask = jnp.take(mask, st, axis=0)
      kv_info = jnp.where(smask, st, smask - 1)

    mask_fn = functools.partial(
        mask_self_attention, causal=causal, exclude_self=True,
        masked=(mask is not None))
    so, slogits = attend(
        sq, k=None, v=sv,
        q_chunk_len=chunk_len,
        n_chunks_before=n_chunks_before,
        n_chunks_after=n_chunks_after,
        mask_fn=mask_fn, q_info=q_info, kv_info=kv_info,
        )

    # jnp.take(so, undo_sort, axis=0); jnp.take(slogits, undo_sort, axis=0)
    # would also work, but these helpers include performance optimizations for
    # TPU.
    o = permute_via_gather(so, undo_sort, sticker, axis=0)
    logits = permute_via_sort(slogits, sticker, undo_sort, axis=0)

    if n_hashes > 1:
      o = jnp.reshape(o, (n_hashes, seqlen, o.shape[-1]))
      logits = jnp.reshape(logits, (n_hashes, seqlen, 1))
      probs = jnp.exp(logits - logsumexp(logits, axis=0, keepdims=True))
      o = jnp.sum(o * probs, axis=0)

    hidden_states = o
    hidden_states = nn.linear.DenseGeneral(
        hidden_states,
        features=d_model,
        axis=(-1,),
        kernel_init=output_kernel_init,
        name='output')
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states

  def hash_vectors(self, vecs, rng, mask=None, *, num_buckets, num_hashes):
    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    if isinstance(num_buckets, int):
      assert num_buckets % 2 == 0
      rot_size = num_buckets
      n_buckets = num_buckets
    else:
      # Factorize the hash if num_buckets is a list or tuple
      rot_size, n_buckets = 0, 1
      for factor in num_buckets:
        assert factor % 2 == 0
        rot_size += factor
        n_buckets *= factor

    rotations_shape = (vecs.shape[-1], num_hashes, rot_size // 2)
    rng = jax.lax.stop_gradient(jax.lax.tie_in(vecs, rng))
    random_rotations = jax.random.normal(rng, rotations_shape).astype(
        jnp.float32)
    rotated_vecs = jnp.einsum('tf,fhb->htb', vecs, random_rotations)

    if isinstance(num_buckets, int) or len(num_buckets) == 1:
      rotated_vecs = jnp.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
      buckets = jnp.argmax(rotated_vecs, axis=-1)
    else:
      # Get the buckets for them and combine.
      buckets, cur_sum, cur_product = None, 0, 1
      for factor in num_buckets:
        rv = rotated_vecs[..., cur_sum:cur_sum + (factor // 2)]
        cur_sum += factor // 2
        rv = jnp.concatenate([rv, -rv], axis=-1)
        if buckets is None:
          buckets = jnp.argmax(rv, axis=-1)
        else:
          buckets += cur_product * jnp.argmax(rv, axis=-1)
        cur_product *= factor

    if mask is not None:
      n_buckets += 1  # Create an extra bucket for padding tokens only
      buckets = jnp.where(mask[None, :], buckets, n_buckets - 1)

    # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
    # bucket numbers from different hashing rounds don't overlap.
    offsets = jax.lax.tie_in(buckets, jnp.arange(num_hashes))
    offsets = jnp.reshape(offsets * n_buckets, (-1, 1))
    buckets = jnp.reshape(buckets + offsets, (-1,))

    return buckets
