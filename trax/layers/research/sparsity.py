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
"""Layers used for experiments with sparsity."""

import math
import random as pyrandom
import jax

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as np
from trax.layers import base
from trax.layers import core
from trax.layers import initializers as init
from trax.layers import metrics
from trax.layers import reversible
from trax.layers.assert_shape import assert_shape


@assert_shape('...->...')
class ReversibleReshapePermute(reversible.ReversibleLayer):
  """Simple and fast, reversible, random-looking permutation layer.

  This layer permutates the last dimension (usually the embedding dimension)
  with simple reshapes. It uses the same permutation for every embedding, and
  permutation never changes.
  The layer works only when the last dimension is a power of 2. The
  permutation is not truly random, as it just uses reshapes to get a fast
  random-looking permutation. It has, however, a permutation cycle length
  of just log2(dimension_size).
  """

  def forward(self, x):
    shape = x.shape
    x = x.reshape(shape[:-1]+(-1, self._get_multiplier(x)))
    t_x = np.einsum('...ab->...ba', x)  # transpose
    return t_x.reshape(shape)

  def reverse(self, x, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng
    shape = x.shape
    x = x.reshape(shape[:-1]+(self._get_multiplier(x), -1))
    t_x = np.einsum('...ab->...ba', x)  # transpose
    return t_x.reshape(shape)

  def _get_multiplier(self, x):
    """Return a size of the new dimension for reshaping.

    We want to split the last dimension into two using approximately equal
    dimensions, we could split a dimension of size 512 into 16 * 32.
    However, not all numbers will work equally well, because we have a different
    cycle length for permutations for different numbers. For example, for
    dimension size 1024 and multiplier 32 we would get the same permutation
    already after applying permutation twice (cycle length is 2), but with
    multiplier 8 we would get the same permutation after appling permutation 10
    times (cycle length is 10).

    For powers of two the cycle length is limited by log2(dimension_size).
    This function returns the biggest multiplier smaller than
    sqrt(dimension_size) that keeps the longest possible cycle lenght of the
    permutation.

    Args:
        x: The input tensor.

    Returns:
        An appropriate multiplier for the permutation reshape.
    """
    last_dim = x.shape[-1]

    def big_relatively_prime(n):
      # The longest possible cycle is achieved iff log2(multiplier) and
      # log2(dimension_size) are relatively prime. We choose the biggest such
      # number smaller than sqrt(dimension_size).
      for i in range(n//2, 0, -1):
        if n%i != 0:
          return  i
      return 1

    max_cycle_len = int(math.log(last_dim, 2))
    assert 2 ** max_cycle_len == last_dim

    return 2 ** big_relatively_prime(max_cycle_len)


@assert_shape('...->...')
class ReversibleRandomPermute(reversible.ReversibleLayer):
  """Reversible, random permutation layer.

  This layer permutates the last dimension (usually the embedding dimension)
  by indexing and slicing. It uses the same random permutation for every
  embedding, and this permutation never changes.
  """

  def forward(self, x):
    permutation, _ = self._get_permutation_and_reverse_permutation(x)
    return x[..., permutation]

  def reverse(self, x, weights=(), state=(), new_state=(), rng=None):
    _, rev_permutation = self._get_permutation_and_reverse_permutation(x)
    return x[..., rev_permutation]

  def _get_permutation_and_reverse_permutation(self, x):
    # TODO(jaszczur): random seed should be stored in state.
    # Currently there is no way of doing it reliably.
    last_dim = x.shape[-1]
    permutation = list(range(last_dim))
    rand = pyrandom.Random(42)
    rand.shuffle(permutation)
    rev_permutation = [permutation.index(i) for i in range(last_dim)]
    return permutation, rev_permutation


@assert_shape('...a->...bc')
def SplitLastAxis(num_splits):  # pylint: disable=invalid-name
  return tl.Fn(f'SplitLastAxis_{num_splits}',
               lambda x: np.reshape(x, x.shape[:-1] + (num_splits, -1)))


@assert_shape('...ab->...c')
def MergeLastTwoAxes():  # pylint: disable=invalid-name
  return tl.Fn('SplitLastAxis',
               lambda x: np.reshape(x, x.shape[:-2] + (-1,)))


@assert_shape('...a->...b')
def LocallyConnectedDense(n_modules, n_units, kernel_size=1,  # pylint: disable=invalid-name
                          kernel_initializer=init.GlorotUniformInitializer(),
                          bias_initializer=init.RandomNormalInitializer(1e-6),
                          use_bias=True):
  """Layer using LocallyConnected1d for approximation of Dense layer.

  The layer splits the last axis of a tensor into `n_modules`, then runs
  LocallyConnected1d (grouped convolution) on all those modules, and
  concatenates their results. It is essentially a locally-sensitive
  approximation of Dense layer, with number of parameters smaller by the factor
  of `n_modules / kernel_size`.

  Args:
    n_modules: Indicates how many modules (pixels) should be input and output
        split into for processing.
    n_units: how many outputs (filters) should each module generate.
    kernel_size: The size of the kernel to be used.
    kernel_initializer: Function that creates a matrix of (random) initial
        connection weights `W` for the layer.
    bias_initializer: Function that creates a vector of (random) initial
        bias weights `b` for the layer.
    use_bias: If `True`, compute an affine map `y = Wx + b`; else compute
        a linear map `y = Wx`.

  Returns:
      LocallyConnectedDense base.Layer.
  """
  if n_modules == 1:
    return tl.Dense(n_units, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer, use_bias=use_bias)
  return tl.Serial(
      tl.SplitLastAxis(n_modules),
      tl.LocallyConnected1d(
          n_units, kernel_size, kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer, use_bias=use_bias, padding='WRAP'),
      tl.MergeLastTwoAxes())


@assert_shape('bld->bld')
def ModularCausalAttention(d_feature, n_heads=1, dropout=0.0,  # pylint: disable=invalid-name
                           max_inference_length=2048, n_modules=1,
                           kernel_size=1, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  it uses LocallyConnectedDense instead of Dense layer for computing Q/K/V.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    n_modules: Number of modules used in LocallyConnectedDense.
    kernel_size: Kernel size used in LocallyConnectedDense.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  @assert_shape('...a->...b')
  def ProcessingLayer():  # pylint: disable=invalid-name
    if n_modules == 1:
      return tl.Dense(d_feature)
    else:
      assert d_feature % n_modules == 0
      return LocallyConnectedDense(n_modules, d_feature // n_modules,
                                   kernel_size=kernel_size)

  return tl.ConfigurableAttention(
      ProcessingLayer(), ProcessingLayer(), ProcessingLayer(),
      ProcessingLayer(), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('bld->bld')
def LowRankCausalAttention(d_feature, n_heads=1, dropout=0.0,  # pylint: disable=invalid-name
                           max_inference_length=2048, lowrank=64,
                           mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  it uses low-rank approximation of kernel in Dense layer for computing Q/K/V.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    lowrank: The rank of low-rank approximation.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  @assert_shape('...a->...a')
  def ProcessingLayer():  # pylint: disable=invalid-name
    return tl.Serial(
        tl.Dense(lowrank),
        tl.Dense(d_feature)
        )

  return tl.ConfigurableAttention(
      ProcessingLayer(), ProcessingLayer(), ProcessingLayer(),
      ProcessingLayer(), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('bld->bld')
def MultiplicativeCausalAttention(d_feature, sparsity=1, n_heads=1, dropout=0.0,  # pylint: disable=invalid-name
                                  max_inference_length=2048, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  for computing Q/K/V instead of a Dense layer it multiplies each embedding
  dimension by a scalar specific to each dimension and each head; then it
  produces Q/K/V by applying the same dense layer to each head. In comparison
  to standard dense layer for computing Q/K/V, this layer uses less parameters
  while still being able to express many functions, like a permutation.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    sparsity: The sparsity of the layer; usually it should be equal to n_heads.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  assert d_feature % sparsity == 0
  d_module = d_feature // sparsity

  @assert_shape('...a->...a')
  def ProcessingLayer():  # pylint: disable=invalid-name
    return tl.Serial(
        # Weight below is used for per-head preprocessing of an embedding.
        tl.Weights(init.RandomNormalInitializer(stddev=0.5),
                   shape=[sparsity, d_feature]),
        # Weight below is dense kernel, shared across heads.
        tl.Weights(init.GlorotUniformInitializer(), [d_feature, d_module]),
        # To save memory the per-head preprocessing and multiplying by the
        # kernel is done in the same einsum.
        tl.Fn('AttentionEinsum',
              (lambda kernel, multiplier, embeds:  # pylint: disable=g-long-lambda
               np.einsum('dx,hd,bld->blhx', kernel, multiplier, embeds))),
        MergeLastTwoAxes(),
        # Weight below is bias after dense, per-head.
        tl.Weights(init.RandomNormalInitializer(1e-6), [d_feature]),
        tl.Add(),
        )

  return tl.ConfigurableAttention(
      ProcessingLayer(), ProcessingLayer(), ProcessingLayer(),
      ProcessingLayer(), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


def CausalFavor(d_feature, n_heads=1, dropout=0.0,  # pylint: disable=invalid-name
                numerical_stabilizer=0.001, precision=None, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  causal attention, but using FAVOR fast attention as in the following paper:
  https://arxiv.org/abs/2006.03555

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    numerical_stabilizer: float, small number used for numerical stability.
    precision: passed to np.einsum to define arithmetic precision.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  del dropout, mode  # not implemented yet but needed in the API

  def favor_numerator_fwd(init_prefix_sum_value, precision,
                          query_prime, key_prime, value):
    def body(p, qkv):
      (q, k, v) = qkv
      p += np.einsum('...m,...d->...md', k, v, precision=precision)
      x_slice = np.einsum('...m,...md->...d', q, p, precision=precision)
      return p, x_slice
    p, w = fastmath.scan(body, init_prefix_sum_value,
                         (query_prime, key_prime, value))
    return w, (precision, p, query_prime, key_prime, value)

  def favor_numerator_bwd(pqkv, w_ct):
    precision, p, qs, ks, vs = pqkv

    def body(carry, qkv_xct):
      p, p_ct = carry
      q, k, v, x_ct = qkv_xct
      q_ct = np.einsum('...d,...md->...m', x_ct, p, precision=precision)
      p_ct += np.einsum('...d,...m->...md', x_ct, q, precision=precision)
      k_ct = np.einsum('...md,...d->...m', p_ct, v, precision=precision)
      v_ct = np.einsum('...md,...m->...d', p_ct, k, precision=precision)
      p -= np.einsum('...m,...d->...md', k, v, precision=precision)
      return (p, p_ct), (q_ct, k_ct, v_ct)

    _, (qs_ct, ks_ct, vs_ct) = fastmath.scan(
        body, (p, np.zeros_like(p)), (qs, ks, vs, w_ct), reverse=True)
    return (None, None, qs_ct, ks_ct, vs_ct)

  def favor_numerator(init_prefix_sum_value, precision, query_prime,
                      key_prime, value):
    w, _ = favor_numerator_fwd(init_prefix_sum_value, precision,
                               query_prime, key_prime, value)
    return w

  favor_numerator = fastmath.custom_vjp(
      favor_numerator, favor_numerator_fwd, favor_numerator_bwd)

  def favor_denominator_fwd(init_prefix_sum_value, precision,
                            query_prime, key_prime):
    def body(p, qk):
      q, k = qk
      p += k
      x = np.einsum('...m,...m->...', q, p, precision=precision)
      return p, x

    p, r = fastmath.scan(body, init_prefix_sum_value, (query_prime, key_prime))
    return r, (precision, query_prime, key_prime, p)

  def favor_denominator_bwd(qkp, r_ct):
    precision, qs, ks, p = qkp

    def body(carry, qkx):
      p, p_ct = carry
      q, k, x_ct = qkx
      q_ct = np.einsum('...,...m->...m', x_ct, p, precision=precision)
      p_ct += np.einsum('...,...m->...m', x_ct, q, precision=precision)
      k_ct = p_ct
      p -= k
      return (p, p_ct), (q_ct, k_ct)

    _, (qs_ct, ks_ct) = fastmath.scan(
        body, (p, np.zeros_like(p)), (qs, ks, r_ct), reverse=True)
    return (None, None, qs_ct, ks_ct)

  def favor_denominator(init_prefix_sum_value, precision, query_prime,
                        key_prime):
    r, _ = favor_denominator_fwd(init_prefix_sum_value, precision,
                                 query_prime, key_prime)
    return r

  favor_denominator = fastmath.custom_vjp(
      favor_denominator, favor_denominator_fwd, favor_denominator_bwd)

  favor_denominator.defvjp(favor_denominator_fwd, favor_denominator_bwd)

  def relu(x):
    return np.where(x <= 0, np.zeros_like(x), x)

  def favor(query, key, value):
    query_prime = relu(query) + numerical_stabilizer
    key_prime = relu(key) + numerical_stabilizer
    prefix_sum_tensor_shape = (key.shape[0], key.shape[-1], value.shape[-1])
    t_slice_shape = (key.shape[0], key.shape[-1])
    init_prefix_sum_value_numerator = np.zeros(prefix_sum_tensor_shape)
    init_prefix_sum_value_denominator = np.zeros(t_slice_shape)

    w = favor_numerator(init_prefix_sum_value_numerator, precision,
                        np.moveaxis(query_prime, 1, 0),
                        np.moveaxis(key_prime, 1, 0),
                        np.moveaxis(value, 1, 0))
    r = favor_denominator(init_prefix_sum_value_denominator,
                          precision,
                          np.moveaxis(query_prime, 1, 0),
                          np.moveaxis(key_prime, 1, 0))
    w = np.moveaxis(w, 0, 1)
    r = np.moveaxis(r, 0, 1)
    # r = r + 2 * numerical_stabilizer * (np.abs(r) <= numerical_stabilizer)
    r = np.reciprocal(r)
    r = np.expand_dims(r, len(r.shape))
    renormalized_attention = w * r
    return renormalized_attention

  return tl.ConfigurableAttention(
      core.Dense(d_feature), core.Dense(d_feature), core.Dense(d_feature),
      core.Dense(d_feature), n_heads=n_heads,
      qkv_attention_layer=base.Fn('FAVOR', favor))


class SparseFF(base.Layer):
  """Feed-forward block with sparsity.

  The original (non-sparse) FF block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  The original block can be slow in decoding due to the need to fetch a lot of
  weights from memory. This sparse block only allows one non-zero element
  in a block of a specified size. This is trained with straight-through Gumbel
  softmax trick.
  """

  def __init__(self, d_ff, n_elements_in_block=32, d_lowrank=64,
               temperature=0.7, mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a sparse feed-forward block."""
    super().__init__(name=f'SparseFF_{d_ff}')
    self._mode = mode
    self._d_ff = d_ff
    self._d_lowrank = d_lowrank
    # Q: what temperature is actually most useful in training?
    self._temperature = temperature if mode == 'train' else 0.0
    self._n_elements_in_block = n_elements_in_block
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    # Helper numbers as d_ff will be divided by n_elements_in_block.
    assert self._d_ff % self._n_elements_in_block == 0
    self._d1 = self._d_ff // self._n_elements_in_block
    self._d2 = self._n_elements_in_block

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    m1, m2, mb, w1, w2, b2 = self.weights
    if self._mode != 'predict':
      w1 = np.reshape(w1.T, (-1, self._d_ff))
      w2 = np.reshape(w2, (self._d_ff, -1))
    x_shape = x.shape
    x = np.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Q: should we add bias and/or put relu after the low-rank m1 dot?
    mask_logits = np.dot(np.dot(x, m1), m2) + mb
    mask_logits = np.reshape(mask_logits, [-1, self._d1, self._d2])
    # Softmax.
    mask_logsumexp = fastmath.logsumexp(mask_logits, axis=-1, keepdims=True)
    log_mask = mask_logits - mask_logsumexp
    mask = np.exp(log_mask)
    # Gumbel-softmax with straight-through discretization.
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    u = fastmath.random.uniform(rng1, mask.shape, np.float32, 1e-6, 1.0 - 1e-6)
    g = -np.log(-np.log(u))
    quant_mask = np.argmax(log_mask + g * self._temperature, axis=-1)
    if self._mode == 'train':
      # Tricks from Section 2.1 in https://arxiv.org/abs/1801.09797
      quant_mask = metrics.one_hot(quant_mask, self._n_elements_in_block)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (50% of the batches) use the soft-mask instead of
      # the quantized mask to improve training stability (see the paper above).
      # Q: is selecting 50% of batches the best? Other %? Mixed in-batch?
      select = fastmath.random.uniform(rng2, (), np.float32, -1.0, 1.0)
      quant_mask = np.where(select > 0.0, quant_mask, mask)
      quant_mask = np.reshape(quant_mask, [-1, self._d_ff])

    if self._mode == 'train':
      # In training, run full matmul to get benefits from the above tricks.
      mid = np.dot(x, w1) * quant_mask  # [joint_batch, d_ff]
      relu = np.where(mid <= 0, np.zeros_like(mid), mid)
      res = np.dot(relu, w2) + b2
    elif self._mode == 'predict':
      # w1 = np.reshape(w1.T, (self._d1, self._d2, -1))
      # w2 = np.reshape(w2, (self._d1, self._d2, -1))
      # This implementation mimicks inference. It's not efficient for large
      # size of joint_batch, but at inference that will be 1 most of the time.
      # Shapes:
      # quant_mask is [joint_batch, self._d1]
      # w1 is [d_model, self._d1, self._d2]
      # we'll index w1 with advanced numpy indexing, first range over
      # self._d1 times the batch size, second range being quant_mask
      batch_size = quant_mask.shape[0]
      idx1 = np.array([np.arange(self._d1)] * batch_size)
      # flatten indices and select from w1
      idx1 = np.reshape(idx1, [-1])
      idx2 = np.reshape(quant_mask, [-1])
      w = w1[idx1, idx2, :]  # now we have per-element weights with batch dim
      w = np.reshape(w, [batch_size, self._d1, -1])
      mid = np.einsum('ai,aji->aj', x, w)
      relu = np.where(mid <= 0, np.zeros_like(mid), mid)
      # w2 is [self._d1, self._d2, d_model]
      v = w2[idx1, idx2, :]
      v = np.reshape(v, [batch_size, self._d1, -1])
      res = np.einsum('ai,aij->aj', relu, v) + b2
    else:
      quant_mask = metrics.one_hot(quant_mask, self._n_elements_in_block)
      quant_mask = np.reshape(quant_mask, [-1, self._d_ff])
      mid = np.dot(x, w1) * quant_mask  # [joint_batch, d_ff]
      relu = np.where(mid <= 0, np.zeros_like(mid), mid)
      res = np.dot(relu, w2) + b2

    return np.reshape(res, x_shape)  # un-flatten if needed

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature.shape[-1]
    shape_m1 = (d_model, self._d_lowrank)
    shape_m2 = (self._d_lowrank, self._d_ff)
    shape_mb = (self._d_ff,)
    shape_w1 = (d_model, self._d_ff)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_m1, rng_m2, rng_mb, rng_w1, rng_w2, rng_b2 = fastmath.random.split(
        self.rng, 6)
    m1 = self._kernel_initializer(shape_m1, rng_m1)
    m2 = self._kernel_initializer(shape_m2, rng_m2)
    mb = self._bias_initializer(shape_mb, rng_mb)
    w1 = self._kernel_initializer(shape_w1, rng_w1)
    w2 = self._kernel_initializer(shape_w2, rng_w2)
    b2 = self._bias_initializer(shape_b2, rng_b2)

    w1 = np.reshape(w1.T, (self._d1, self._d2, -1))
    w2 = np.reshape(w2, (self._d1, self._d2, -1))
    self.weights = (m1, m2, mb, w1, w2, b2)


class BlockSparseFF(base.Layer):
  """Feed-forward block with block sparsity.

  The original (non-sparse) FF block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  This block sparse layer mimics mixture of experts architecture.
  It divides the dimension of d_ff in each weight matrix to # of blocks equal to
  num_experts and activates only one non-zero block from the weights matrix.
  This is trained with straight-through Gumbel softmax trick.
  """

  def __init__(self,
               d_ff,
               num_experts=64,
               temperature=0.7,
               mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a block sparse feed-forward block."""
    super().__init__(name=f'BlockSparseFF_{d_ff}')
    self._mode = mode
    self._d_ff = d_ff
    self._num_experts = num_experts
    self._temperature = temperature if mode == 'train' else 0.0
    self._n_elements_in_block = d_ff // num_experts
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    assert self._d_ff % self._num_experts == 0

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
        initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    m1, w1, w2, b2 = self.weights
    x_shape = x.shape
    x = np.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Q: check if we need bias and/or put relu after the m1 dot?
    mask_logits = np.dot(x, m1)
    # Softmax.
    mask_logsumexp = fastmath.logsumexp(mask_logits, axis=-1, keepdims=True)
    log_mask = mask_logits - mask_logsumexp
    mask = np.exp(log_mask)
    # Gumbel-softmax with straight-through discretization.
    # TODO(lukaszkaiser, chowdhery): Extract this block and share
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    u = fastmath.random.uniform(rng1, mask.shape, np.float32, 1e-6, 1.0 - 1e-6)
    g = -np.log(-np.log(u))
    selected_experts = np.argmax(log_mask + g * self._temperature, axis=-1)
    if self._mode == 'train':
      # Tricks from Section 2.1 in https://arxiv.org/abs/1801.09797
      quant_mask = metrics.one_hot(selected_experts, self._num_experts)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (50% of the batches) use the soft-mask instead of
      # the quantized mask to improve training stability (see the paper above).
      # Q: is selecting 50% of batches the best? Other %? Mixed in-batch?
      select = fastmath.random.uniform(rng2, (), np.float32, -1.0, 1.0)
      quant_mask = np.where(select > 0.0, quant_mask, mask)
    else:
      quant_mask = metrics.one_hot(selected_experts, self._num_experts)
    quant_mask = np.reshape(quant_mask, [-1, self._num_experts, 1])
    quant_mask_shape = quant_mask.shape
    batch_size = quant_mask.shape[0]

    if self._mode == 'predict' and batch_size == 1:
      # This implementation mimicks inference for batch_size 1.
      start_idx = selected_experts[0] * self._n_elements_in_block
      # w1 is [d_model, d_ff], w is [d_model, n_elements_in_block]
      w = jax.lax.dynamic_slice(w1, [0, start_idx],
                                [w1.shape[0], self._n_elements_in_block])
      mid = np.dot(x, w)
      relu = np.where(mid <= 0, np.zeros_like(mid), mid)
      # w2 is [d_ff, d_model], v is [n_elements_in_block, d_model]
      v = jax.lax.dynamic_slice(w2, [start_idx, 0],
                                [self._n_elements_in_block, w2.shape[-1]])
      v = np.reshape(v, [self._n_elements_in_block, -1])
      res = np.dot(relu, v) + b2
    else:
      expanded_mask = np.broadcast_to(
          quant_mask,
          (quant_mask_shape[0], quant_mask.shape[1], self._n_elements_in_block))
      expanded_mask = np.reshape(expanded_mask, (-1, self._d_ff))
      mid = np.dot(x, w1) * expanded_mask  # [joint_batch, d_ff]
      relu = np.where(mid <= 0, np.zeros_like(mid), mid)
      res = np.dot(relu, w2) + b2

    return np.reshape(res, x_shape)  # un-flatten if needed

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature.shape[-1]
    shape_m1 = (d_model, self._num_experts)
    shape_w1 = (d_model, self._d_ff)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_m1, rng_w1, rng_w2, rng_b2 = fastmath.random.split(self.rng, 4)
    m1 = self._kernel_initializer(shape_m1, rng_m1)
    w1 = self._kernel_initializer(shape_w1, rng_w1)
    w2 = self._kernel_initializer(shape_w2, rng_w2)
    b2 = self._bias_initializer(shape_b2, rng_b2)

    self.weights = (m1, w1, w2, b2)
