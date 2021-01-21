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
"""Layers used for experiments with sparsity."""

import functools
import math
import random as pyrandom

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import core
from trax.layers import initializers as init
from trax.layers import reversible
from trax.layers.assert_shape import assert_shape


# We use mixed CamelCase and snake_case names in this file.
# pylint: disable=invalid-name


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
    t_x = jnp.einsum('...ab->...ba', x)  # transpose
    return t_x.reshape(shape)

  def reverse(self, x, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng
    shape = x.shape
    x = x.reshape(shape[:-1]+(self._get_multiplier(x), -1))
    t_x = jnp.einsum('...ab->...ba', x)  # transpose
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
def SplitLastAxis(num_splits):
  return tl.Fn(f'SplitLastAxis_{num_splits}',
               lambda x: jnp.reshape(x, x.shape[:-1] + (num_splits, -1)))


@assert_shape('...ab->...c')
def MergeLastTwoAxes():
  return tl.Fn('SplitLastAxis',
               lambda x: jnp.reshape(x, x.shape[:-2] + (-1,)))


@assert_shape('...a->...b')
def LocallyConnectedDense(n_modules, n_units, kernel_size=1,
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
def ModularCausalAttention(d_feature, n_heads=1, sparsity=None, dropout=0.0,
                           max_inference_length=2048,
                           kernel_size=1, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  it uses LocallyConnectedDense instead of Dense layer for computing Q/K/V.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    sparsity: Number of modules used in LocallyConnectedDense.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    kernel_size: Kernel size used in LocallyConnectedDense.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  n_modules = n_heads if sparsity is None else sparsity
  @assert_shape('...a->...b')
  def ProcessingLayer():
    assert d_feature % n_modules == 0
    return LocallyConnectedDense(n_modules, d_feature // n_modules,
                                 kernel_size=kernel_size)

  return tl.ConfigurableAttention(
      ProcessingLayer(), ProcessingLayer(), ProcessingLayer(),
      ProcessingLayer(), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('...a->...b')
def LocallyConvDense(n_modules, n_units, kernel_size=1, length_kernel_size=1):
  """Layer using local convolutions for approximation of Dense layer.

  The layer splits the last axis of a tensor into `n_modules`, then runs
  a convolution on all those modules, and concatenates their results.
  It is similar to LocallyConnectedDense above, but shares weights.

  Args:
    n_modules: Indicates how many modules (pixels) should be input and output
        split into for processing.
    n_units: how many outputs (filters) should each module generate.
    kernel_size: The size of the kernel to be used.
    length_kernel_size: If > 1, also do causal convolution on the previous axis,
      which is often the sentence length in sequence models.

  Returns:
      LocallyConvDense base.Layer.
  """
  if n_modules == 1:
    return tl.Dense(n_units)
  if kernel_size % 2 != 1:
    raise ValueError('Currently we only handle odd kernel sizes.')
  half = (kernel_size - 1) // 2
  pad_widths = [[0, 0], [length_kernel_size - 1, 0], [half, half], [0, 0]]
  return tl.Serial(
      tl.SplitLastAxis(n_modules),
      tl.Fn('Pad', lambda x: jnp.pad(x, pad_width=pad_widths)),
      tl.Conv(n_units, kernel_size=(length_kernel_size, kernel_size)),
      tl.MergeLastTwoAxes()
  )


@assert_shape('bld->bld')
def ConvCausalAttention(d_feature, n_heads=1, sparsity=None, dropout=0.0,
                        max_inference_length=2048,
                        kernel_size=1, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  it uses LocallyConvDense instead of Dense layer for computing Q/K/V.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    sparsity: Number of modules used in LocallyConvDense.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    kernel_size: Kernel size used in LocallyConnectedDense.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  n_modules = n_heads if sparsity is None else sparsity
  @assert_shape('...a->...b')
  def ProcessingLayer():
    assert d_feature % n_modules == 0
    return LocallyConvDense(n_modules, d_feature // n_modules,
                            kernel_size=kernel_size)

  return tl.ConfigurableAttention(
      ProcessingLayer(), ProcessingLayer(), ProcessingLayer(),
      ProcessingLayer(), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('...a->...b')
def LowRankDense(n_units, d_lowrank):
  return tl.Serial(
      tl.Dense(d_lowrank),
      tl.Dense(n_units)
      )


@assert_shape('...a->...b')
def EinsumDense(d_input, d_output, use_bias):
  """Returns a reimplementation of Dense layer, using einsum.

  While this is an equivalent of a Dense layer, it seems to be faster when used
  in decoding if used with bias (see decoding_timing_test.py ).
  This layer can be removed when we understand better the reason for the
  difference in decoding speed.

  Args:
    d_input: Dimensionality of the input tensor.
    d_output: Dimensionality of the output tensor.
    use_bias: Whether to use bias.
  """
  layers = [
      tl.Weights(init.GlorotUniformInitializer(), [d_output, d_input]),
      tl.Fn('EinsumDense',
            (lambda kernel, embeds:  # pylint: disable=g-long-lambda
             jnp.einsum('xd,...d->...x', kernel, embeds)))
  ]
  if use_bias:
    layers.extend([
        tl.Weights(init.RandomNormalInitializer(1e-6), [d_output]),
        tl.Add()
    ])
  return tl.Serial(layers)


def RandomLayer(layer_a, layer_b, prob_a):
  """Runs `layer_a` with probability `prob_a`, otherwise runs `layer_b`."""
  condition = tl.Serial(
      tl.RandomUniform(),
      tl.Fn('SmallerThan', lambda x: x < prob_a)
      )
  return tl.Cond(condition, layer_a, layer_b)


@assert_shape('...a->...b')
def SparseDenseWithOptions(n_units, d_input=None, sparsity_type=None,
                           sparsity=0, d_lowrank=None, prob_sparse=None,
                           mode=None, use_bias=True, use_bfloat16=False):
  """Configurable sparse version of Dense layer."""
  if prob_sparse is not None:
    if mode is not None and mode != 'train':
      # For non-training modes, we want to use a sparse variant.
      # This is different than simply prob_sparse being None, as the weights of
      # the model are different.
      prob_sparse = 1.0
    return RandomLayer(
        SparseDenseWithOptions(n_units, d_input, sparsity_type, sparsity,
                               d_lowrank, use_bias=use_bias,
                               use_bfloat16=use_bfloat16),
        tl.Dense(n_units, use_bias=use_bias, use_bfloat16=use_bfloat16),
        prob_sparse)

  if sparsity_type is None or sparsity_type == 'None' or sparsity == 0:
    return tl.Dense(n_units, use_bias=use_bias, use_bfloat16=use_bfloat16)
  if sparsity_type == 'mult':
    return MultiplicativeSparseDense(
        sparsity, d_input, n_units, use_bias=use_bias,
        use_bfloat16=use_bfloat16)

  assert not use_bfloat16  # use_bfloat16 is unsupported for other variants
  if sparsity_type == 'lowrank':
    assert use_bias  # use_bias=False is unsupported
    return LowRankDense(n_units, d_lowrank)
  if sparsity_type == 'einsum':
    return EinsumDense(d_input, n_units, use_bias=use_bias)
  if sparsity_type == 'local':
    assert use_bias  # use_bias = False is unsupported
    assert n_units % sparsity == 0
    return LocallyConnectedDense(sparsity, n_units/sparsity)
  if sparsity_type == 'local3':
    assert use_bias  # use_bias = False is unsupported
    assert n_units % sparsity == 0
    return LocallyConnectedDense(sparsity, n_units/sparsity, kernel_size=3)

  raise ValueError('Unknown sparsity type: {}'.format(sparsity_type))


@assert_shape('bld->bld')
def LowRankCausalAttention(d_feature, n_heads=1, dropout=0.0,
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

  return tl.ConfigurableAttention(
      LowRankDense(d_feature, lowrank), LowRankDense(d_feature, lowrank),
      LowRankDense(d_feature, lowrank), LowRankDense(d_feature, lowrank),
      n_heads=n_heads, qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('...a->...b')
def MultiplicativeSparseDense(sparsity, d_input, d_output=None,
                              use_bias=True, use_bfloat16=False):
  """Returns a replacement of Dense layer which uses less parameters.

  The layer uses number of modules equal to `sparsity`. It multiplies each
  dimension of the input tensor by a scalar specific to each dimension and each
  module separately; then it applies Dense(d_output/sparsity) to each module.
  Compared to standard dense layer, MultiplicativeSparseDense uses less
  parameters while still being able to express many interesting functions (for
  example a permutation).

  Args:
    sparsity: The sparsity of the layer; the output vector is divided into this
        number of modules.
    d_input: Dimensionality of input tensor.
    d_output: Dimensionality of output tensor; by default equal to d_input.
    use_bias: Whether to use bias.
    use_bfloat16: Whether to use bfloat16 for weights.
  """

  assert d_output % sparsity == 0
  d_module = d_output // sparsity

  layers = [
      # Weight below is used for per-head preprocessing of an embedding.
      tl.Weights(init.RandomNormalInitializer(stddev=0.5),
                 shape=[sparsity, d_input], use_bfloat16=use_bfloat16),
      # Weight below is dense kernel, shared across heads.
      tl.Weights(init.GlorotUniformInitializer(), [d_input, d_module],
                 use_bfloat16=use_bfloat16),
      # To save memory the per-head preprocessing and multiplying by the
      # kernel is done in the same einsum.
      tl.Fn('AttentionEinsum',
            (lambda kernel, multiplier, embeds:  # pylint: disable=g-long-lambda
             jnp.einsum('dx,hd,...d->...hx', kernel, multiplier, embeds))),
      MergeLastTwoAxes(),
  ]
  if use_bias:
    layers.extend([
        # Weight below is bias after dense, per-head.
        tl.Weights(init.RandomNormalInitializer(1e-6), [d_output],
                   use_bfloat16=use_bfloat16),
        tl.Add(),
    ])
  return tl.Serial(layers)


@assert_shape('...a->...a')
def MultiplicativeModularSparseDense(sparsity, d_feature):
  """Returns a replacement of Dense layer which uses less parameters.

  The layer uses number of modules equal to `sparsity`. It is a combination of
  multiplicative dense and locally connected dense layers.

  Args:
    sparsity: The sparsity of the layer; the output vector is divided into this
        number of modules.
    d_feature: Dimensionality of input and output tensor.
  """

  assert d_feature % sparsity == 0
  d_module = d_feature // sparsity

  return tl.Serial(
      # Weight below is used for per-head preprocessing of an embedding.
      tl.Weights(init.RandomNormalInitializer(stddev=0.5),
                 shape=[sparsity, d_feature]),
      # Weight below is a kernel of multiplicative dense, shared across heads.
      tl.Weights(init.GlorotUniformInitializer(), [d_feature, d_module]),
      # Weight below is a kernel of modular dense.
      tl.Weights(functools.partial(init.GlorotUniformInitializer(),
                                   nonreceptive_dims=[0]),
                 [sparsity, d_module, d_module]),
      # To save memory the per-head preprocessing and multiplying by
      # kernels is done in a single einsum.
      tl.Fn('SparseDenseEinsum',
            (lambda kmod, kmult, multiplier, embeds:  # pylint: disable=g-long-lambda
             jnp.einsum('hxo,dx,hd,...d->...ho', kmod, kmult, multiplier, embeds
                        ))),
      MergeLastTwoAxes(),
      # Weight below is bias after dense, per-head.
      tl.Weights(init.RandomNormalInitializer(1e-6), [d_feature]),
      tl.Add(),
      )


@assert_shape('bld->bld')
def MultiplicativeCausalAttention(d_feature, n_heads=1, sparsity=None,
                                  dropout=0.0, max_inference_length=2048,
                                  mode='train'):
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
    n_heads: Number of attention heads.
    sparsity: The sparsity of the layer; usually it should be equal to n_heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  sparsity = n_heads if sparsity is None else sparsity
  return tl.ConfigurableAttention(
      MultiplicativeSparseDense(sparsity, d_feature, d_feature),
      MultiplicativeSparseDense(sparsity, d_feature, d_feature),
      MultiplicativeSparseDense(sparsity, d_feature, d_feature),
      MultiplicativeSparseDense(sparsity, d_feature, d_feature),
      n_heads=n_heads, qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('bld->bld')
def MultiplicativeModularCausalAttention(
    d_feature, n_heads=1, sparsity=None, dropout=0.0, max_inference_length=2048,
    mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  for computing Q/K/V instead of a Dense layer it combines
  MultiplicativeSparseDense layer with LocallyConnectedLayer.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    sparsity: The sparsity of the layer; usually it should be equal to n_heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  sparsity = n_heads if sparsity is None else sparsity
  return tl.ConfigurableAttention(
      MultiplicativeModularSparseDense(sparsity, d_feature),
      MultiplicativeModularSparseDense(sparsity, d_feature),
      MultiplicativeModularSparseDense(sparsity, d_feature),
      MultiplicativeModularSparseDense(sparsity, d_feature), n_heads=n_heads,
      qkv_attention_layer=tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))


@assert_shape('bld->bld')
def MultiplicativeConvCausalAttention(
    d_feature, n_heads=1, sparsity=None, length_kernel_size=3,
    dropout=0.0, max_inference_length=2048, mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  for computing Q/K/V instead of a Dense layer it combines
  MultiplicativeSparseDense layer with LocallyConvLayer.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    sparsity: The sparsity of the layer; usually it should be equal to n_heads.
    length_kernel_size: Size of convolution kernel on the length dimension.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    max_inference_length: maximum length for inference.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  sparsity = n_heads if sparsity is None else sparsity
  d_module = d_feature // sparsity
  return tl.Serial(
      tl.Select([0, 0]),  # duplicate activations
      MultiplicativeSparseDense(sparsity, d_feature, d_feature),  # shared q, k
      tl.Select([0, 0, 0]),  # use for q, k, v
      tl.Parallel(
          [LocallyConvDense(sparsity, d_module, kernel_size=3,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
          [LocallyConvDense(sparsity, d_module, kernel_size=3,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
          [tl.Concatenate(),  # use permuted and original for v
           LocallyConvDense(sparsity, d_module, kernel_size=1,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
      ),
      tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode),
      tl.MergeHeads(n_heads),
  )


def Favor(d_feature, n_heads=1, dropout=0.0,
          numerical_stabilizer=0.001, mode='train'):
  """Returns a layer that maps (activations, mask) to (new_activations, mask).

  See the FAVOR paper for details: https://arxiv.org/abs/2006.03555

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    numerical_stabilizer: float, small number used for numerical stability.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  del dropout, mode  # not implemented yet but needed in the API

  def bidirectional_numerator(query_prime, key_prime, value):
    kvs = jnp.einsum('lbm,lbd->bmd', key_prime, value)
    return jnp.einsum('lbm,bmd->lbd', query_prime, kvs)

  def bidirectional_denominator(query_prime, key_prime):
    all_ones = jnp.ones([query_prime.shape[0]])
    ks_sum = jnp.einsum('lbm,l->bm', key_prime, all_ones)
    return jnp.einsum('lbm,bm->lb', query_prime, ks_sum)

  def relu(x):
    return jnp.where(x <= 0, jnp.zeros_like(x), x)

  def favor(query, key, value, mask):
    query_prime = relu(query) + numerical_stabilizer
    key_prime = relu(key) + numerical_stabilizer
    mask_batch_1_length = jnp.reshape(
        mask, [key.shape[0] // n_heads, 1, key.shape[1]]).astype(jnp.float32)
    mask_heads = mask_batch_1_length + jnp.zeros((1, n_heads, 1))
    key_prime *= jnp.reshape(mask_heads, [key.shape[0], key.shape[1], 1])

    w = bidirectional_numerator(jnp.moveaxis(query_prime, 1, 0),
                                jnp.moveaxis(key_prime, 1, 0),
                                jnp.moveaxis(value, 1, 0))
    r = bidirectional_denominator(jnp.moveaxis(query_prime, 1, 0),
                                  jnp.moveaxis(key_prime, 1, 0))
    w = jnp.moveaxis(w, 0, 1)
    r = jnp.moveaxis(r, 0, 1)
    r = jnp.reciprocal(r)
    r = jnp.expand_dims(r, len(r.shape))
    renormalized_attention = w * r
    return renormalized_attention, mask

  return  tl.ConfigurableAttention(
      tl.Dense(d_feature), tl.Dense(d_feature), tl.Dense(d_feature),
      tl.Dense(d_feature),
      tl.Fn('FAVOR', favor, n_out=2), n_heads=n_heads)


def CausalFavor(d_feature, n_heads=1, dropout=0.0,
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
    precision: passed to jnp.einsum to define arithmetic precision.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  del dropout, mode  # not implemented yet but needed in the API

  def favor_numerator_fwd(init_prefix_sum_value, precision,
                          query_prime, key_prime, value):
    def body(p, qkv):
      (q, k, v) = qkv
      p += jnp.einsum('...m,...d->...md', k, v, precision=precision)
      x_slice = jnp.einsum('...m,...md->...d', q, p, precision=precision)
      return p, x_slice
    p, w = fastmath.scan(body, init_prefix_sum_value,
                         (query_prime, key_prime, value))
    return w, (precision, p, query_prime, key_prime, value)

  def favor_numerator_bwd(pqkv, w_ct):
    precision, p, qs, ks, vs = pqkv

    def body(carry, qkv_xct):
      p, p_ct = carry
      q, k, v, x_ct = qkv_xct
      q_ct = jnp.einsum('...d,...md->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...d,...m->...md', x_ct, q, precision=precision)
      k_ct = jnp.einsum('...md,...d->...m', p_ct, v, precision=precision)
      v_ct = jnp.einsum('...md,...m->...d', p_ct, k, precision=precision)
      p -= jnp.einsum('...m,...d->...md', k, v, precision=precision)
      return (p, p_ct), (q_ct, k_ct, v_ct)

    _, (qs_ct, ks_ct, vs_ct) = fastmath.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, vs, w_ct), reverse=True)
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
      x = jnp.einsum('...m,...m->...', q, p, precision=precision)
      return p, x

    p, r = fastmath.scan(body, init_prefix_sum_value, (query_prime, key_prime))
    return r, (precision, query_prime, key_prime, p)

  def favor_denominator_bwd(qkp, r_ct):
    precision, qs, ks, p = qkp

    def body(carry, qkx):
      p, p_ct = carry
      q, k, x_ct = qkx
      q_ct = jnp.einsum('...,...m->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...,...m->...m', x_ct, q, precision=precision)
      k_ct = p_ct
      p -= k
      return (p, p_ct), (q_ct, k_ct)

    _, (qs_ct, ks_ct) = fastmath.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, r_ct), reverse=True)
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
    return jnp.where(x <= 0, jnp.zeros_like(x), x)

  def favor(query, key, value):
    query_prime = relu(query) + numerical_stabilizer
    key_prime = relu(key) + numerical_stabilizer
    prefix_sum_tensor_shape = (key.shape[0], key.shape[-1], value.shape[-1])
    t_slice_shape = (key.shape[0], key.shape[-1])
    init_prefix_sum_value_numerator = jnp.zeros(prefix_sum_tensor_shape)
    init_prefix_sum_value_denominator = jnp.zeros(t_slice_shape)

    w = favor_numerator(init_prefix_sum_value_numerator, precision,
                        jnp.moveaxis(query_prime, 1, 0),
                        jnp.moveaxis(key_prime, 1, 0),
                        jnp.moveaxis(value, 1, 0))
    r = favor_denominator(init_prefix_sum_value_denominator,
                          precision,
                          jnp.moveaxis(query_prime, 1, 0),
                          jnp.moveaxis(key_prime, 1, 0))
    w = jnp.moveaxis(w, 0, 1)
    r = jnp.moveaxis(r, 0, 1)
    r = jnp.reciprocal(r)
    r = jnp.expand_dims(r, len(r.shape))
    renormalized_attention = w * r
    return renormalized_attention

  return tl.ConfigurableAttention(
      core.Dense(d_feature), core.Dense(d_feature), core.Dense(d_feature),
      core.Dense(d_feature), n_heads=n_heads,
      qkv_attention_layer=base.Fn('CausalFAVOR', favor))


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
               temperature=0.1, quant_prob=0.3, use_bfloat16=False,
               big_weights_in_bfloat16=True, mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a sparse feed-forward block."""
    super().__init__(name=f'SparseFF_{d_ff}')
    self._mode = mode
    self._use_bfloat16 = use_bfloat16
    self._big_weights_in_bfloat16 = big_weights_in_bfloat16
    self._d_ff = d_ff
    self._d_lowrank = d_lowrank
    # Q: what temperature is actually most useful in training?
    self._temperature = temperature if mode == 'train' else 0.0
    self._quant_prob = quant_prob
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
      w1 = jnp.reshape(w1.T, (-1, self._d_ff))
      w2 = jnp.reshape(w2, (self._d_ff, -1))
    else:
      # This is a work-around of a bug in the previous if statement, which makes
      # w1 array shuffled. Fixing it properly would invalidate previous
      # checkpoints, so this is a temporary work-around.
      w1 = jnp.transpose(w1, (1, 0, 2))
      w1 = jnp.reshape(w1, (self._d1, self._d2, -1))

    x_shape = x.shape
    x = jnp.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Q: should we add bias and/or put relu after the low-rank m1 dot?
    mask_logits = jnp.dot(jnp.dot(x, m1), m2) + mb
    mask_logits = jnp.reshape(mask_logits, [-1, self._d1, self._d2])
    # Softmax.
    mask_logsumexp = fastmath.logsumexp(mask_logits, axis=-1, keepdims=True)
    log_mask = mask_logits - mask_logsumexp
    mask = jnp.exp(log_mask)
    # Gumbel-softmax with straight-through discretization.
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    u = fastmath.random.uniform(rng1, mask.shape, jnp.float32, 1e-6, 1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    quant_mask = jnp.argmax(log_mask + g * self._temperature, axis=-1)
    if self._mode == 'train':
      # Tricks from Section 2.1 in https://arxiv.org/abs/1801.09797
      quant_mask = tl.one_hot(quant_mask, self._n_elements_in_block)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (quant_prob of the batches) use the soft-mask instead
      # of the quantized mask to improve training stability (see paper above).
      select = fastmath.random.uniform(rng2, (), jnp.float32, 0.0, 1.0)
      quant_mask = jnp.where(select < self._quant_prob, quant_mask, mask)
      quant_mask = jnp.reshape(quant_mask, [-1, self._d_ff])

    if self._mode == 'train':
      # In training, run full matmul to get benefits from the above tricks.
      mid = jnp.dot(x, w1) * quant_mask  # [joint_batch, d_ff]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      res = jnp.dot(relu, w2) + b2
    elif self._mode == 'predict':
      # w1 = jnp.reshape(w1.T, (self._d1, self._d2, -1))
      # w2 = jnp.reshape(w2, (self._d1, self._d2, -1))
      # This implementation mimicks inference. It's not efficient for large
      # size of joint_batch, but at inference that will be 1 most of the time.
      # Shapes:
      # quant_mask is [joint_batch, self._d1]
      # w1 is [self._d1, self._d2, d_model]
      # we'll index w1 with advanced numpy indexing, first range over
      # self._d1 times the batch size, second range being quant_mask
      batch_size = quant_mask.shape[0]
      idx1 = jnp.array([jnp.arange(self._d1)] * batch_size)
      # flatten indices and select from w1
      idx1 = jnp.reshape(idx1, [-1])
      idx2 = jnp.reshape(quant_mask, [-1])
      w = w1[idx1, idx2, :]  # now we have per-element weights with batch dim
      w = jnp.reshape(w, [batch_size, self._d1, -1])
      mid = jnp.einsum('ai,aji->aj', x, w)
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      # w2 is [self._d1, self._d2, d_model]
      v = w2[idx1, idx2, :]
      v = jnp.reshape(v, [batch_size, self._d1, -1])
      res = jnp.einsum('ai,aij->aj', relu, v) + b2
    else:
      quant_mask = tl.one_hot(quant_mask, self._n_elements_in_block)
      quant_mask = jnp.reshape(quant_mask, [-1, self._d_ff])
      mid = jnp.dot(x, w1) * quant_mask  # [joint_batch, d_ff]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      res = jnp.dot(relu, w2) + b2

    return jnp.reshape(res, x_shape)  # un-flatten if needed

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
    if self._use_bfloat16:
      m1 = m1.astype(jnp.bfloat16)
      m2 = m2.astype(jnp.bfloat16)
      mb = mb.astype(jnp.bfloat16)
      b2 = b2.astype(jnp.bfloat16)
    if self._use_bfloat16 or self._big_weights_in_bfloat16:
      w1 = w1.astype(jnp.bfloat16)
      w2 = w2.astype(jnp.bfloat16)

    w1 = jnp.reshape(w1.T, (self._d1, self._d2, -1))
    w2 = jnp.reshape(w2, (self._d1, self._d2, -1))
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
    x = jnp.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Q: check if we need bias and/or put relu after the m1 dot?
    mask_logits = jnp.dot(x, m1)
    # Softmax.
    mask_logsumexp = fastmath.logsumexp(mask_logits, axis=-1, keepdims=True)
    log_mask = mask_logits - mask_logsumexp
    mask = jnp.exp(log_mask)
    # Gumbel-softmax with straight-through discretization.
    # TODO(lukaszkaiser, chowdhery): Extract this block and share
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    u = fastmath.random.uniform(rng1, mask.shape, jnp.float32, 1e-6, 1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    selected_experts = jnp.argmax(log_mask + g * self._temperature, axis=-1)
    if self._mode == 'train':
      # Tricks from Section 2.1 in https://arxiv.org/abs/1801.09797
      quant_mask = tl.one_hot(selected_experts, self._num_experts)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (50% of the batches) use the soft-mask instead of
      # the quantized mask to improve training stability (see the paper above).
      # Q: is selecting 50% of batches the best? Other %? Mixed in-batch?
      select = fastmath.random.uniform(rng2, (), jnp.float32, -1.0, 1.0)
      quant_mask = jnp.where(select > 0.0, quant_mask, mask)
    else:
      quant_mask = tl.one_hot(selected_experts, self._num_experts)
    quant_mask = jnp.reshape(quant_mask, [-1, self._num_experts, 1])
    quant_mask_shape = quant_mask.shape
    batch_size = quant_mask.shape[0]

    if self._mode == 'predict' and batch_size == 1:
      # This implementation mimicks inference for batch_size 1.
      start_idx = selected_experts[0] * self._n_elements_in_block
      # w1 is [d_model, d_ff], w is [d_model, n_elements_in_block]
      w = fastmath.dynamic_slice(w1, [0, start_idx],
                                 [w1.shape[0], self._n_elements_in_block])
      mid = jnp.dot(x, w)
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      # w2 is [d_ff, d_model], v is [n_elements_in_block, d_model]
      v = fastmath.dynamic_slice(w2, [start_idx, 0],
                                 [self._n_elements_in_block, w2.shape[-1]])
      v = jnp.reshape(v, [self._n_elements_in_block, -1])
      res = jnp.dot(relu, v) + b2
    else:
      expanded_mask = jnp.broadcast_to(
          quant_mask,
          (quant_mask_shape[0], quant_mask.shape[1], self._n_elements_in_block))
      expanded_mask = jnp.reshape(expanded_mask, (-1, self._d_ff))
      mid = jnp.dot(x, w1) * expanded_mask  # [joint_batch, d_ff]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      res = jnp.dot(relu, w2) + b2

    return jnp.reshape(res, x_shape)  # un-flatten if needed

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
