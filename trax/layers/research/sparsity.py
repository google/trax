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
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.fastmath import random
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
               lambda x: jnp.reshape(x, tuple(x.shape)[:-1] + (num_splits, -1)))


@assert_shape('...ab->...c')
def MergeLastTwoAxes():
  return tl.Fn('MergeLastTwoAxes',
               lambda x: jnp.reshape(x, tuple(x.shape)[:-2] + (-1,)))


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


class _RememberPad(base.Layer):
  """Layer which remembers last N elements in predict mode."""

  def __init__(self, n_items_to_remember, mode):
    """Returns a layer which remembers last N elements in predict mode.

    For predict mode, the layer remembers last N elements and pads with them.
    For other modes, it pads with zeros. The layer pads/remembers elements from
    the second axis.

    Args:
      n_items_to_remember: Number of items to remember/pad with.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
    """
    super().__init__(name='_RememberPad')
    self._n_items_to_remember = n_items_to_remember
    self._mode = mode
    self._portal_mask = self.monkey_patched_mask()  # pylint: disable=assignment-from-none

  def monkey_patched_mask(self):
    # This is necessary for Terraformer model. See comments there.
    # The mask will only be used in Terraformer in predict mode.
    return None

  def forward(self, x):
    if self._n_items_to_remember == 0:
      return x
    if self._mode == 'predict':
      x = jnp.concatenate([self.state[0], x], axis=1)
      if self._portal_mask is not None and 'init' in self.state[1]:
        # TODO(jaszczur): In predict mode with monkey-patched mask, we
        # currently assume that batch size is 1.
        assert x.shape[0] == 1
        mask = self._portal_mask.get_value()
        count_padding = jnp.sum(mask == 0, dtype=jnp.int32)
        self.state = (fastmath.dynamic_slice_in_dim(
            x, x.shape[1] - (self._n_items_to_remember + count_padding),
            self._n_items_to_remember, axis=1), {'forward': ()})
      else:
        self.state = (x[:, -self._n_items_to_remember:, ...], {'forward': ()})
    else:
      pad_widths = [[0, 0] for _ in range(len(x.shape))]
      pad_widths[1][0] = self._n_items_to_remember
      x = jnp.pad(x, pad_width=pad_widths, mode='constant')
    return x

  def init_weights_and_state(self, input_signature):
    """Initializes this layer's weights."""
    if isinstance(input_signature, (list, tuple)):
      input_signature = input_signature[0]
    self.weights = ()
    if self._mode == 'predict':
      shape = list(input_signature.shape)
      shape[1] = self._n_items_to_remember
      self.state = (jnp.zeros(shape, dtype=jnp.float32), {'init': ()})
    else:
      self.state = ()


@assert_shape('...a->...b')
def LocallyConvDense(n_modules, n_units, mode, kernel_size=1,
                     length_kernel_size=1):
  """Layer using local convolutions for approximation of Dense layer.

  The layer splits the last axis of a tensor into `n_modules`, then runs
  a convolution on all those modules, and concatenates their results.
  It is similar to LocallyConnectedDense above, but shares weights.

  Args:
    n_modules: Indicates how many modules (pixels) should be input and output
        split into for processing.
    n_units: how many outputs (filters) should each module generate.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
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
  pad_widths = [[0, 0], [0, 0], [half, half], [0, 0]]
  return tl.Serial(
      tl.SplitLastAxis(n_modules),
      tl.Fn('Pad', lambda x: jnp.pad(x, pad_width=pad_widths, mode='constant')),
      _RememberPad(length_kernel_size-1, mode=mode),
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
    return LocallyConvDense(n_modules, d_feature // n_modules, mode=mode,
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
    return FactoredDense(sparsity, d_input, n_units, use_bias=use_bias,
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
def FactoredDense(n_modules, d_in, d_out, use_bias=True, use_bfloat16=False):
  r"""Returns a Dense-like layer, internally factored to use fewer parameters.

  This layer treats an activation vector as if divided into :math:`M`
  subvectors (``n_modules`` 'modules'). It uses this factored view to compute
  a :py:class:`Dense`-like mapping with high mixing/connectivity, but using
  approximately :math:`1/M` the number of weights of a similarly dimensioned
  :py:class:`Dense` layer.

  More specifically, each activation vector of dimensionality ``n_in`` is
  multiplied element-wise (a generalized form of gating) with ``n_modules``
  vectors also of dimensionality ``n_in``. The resulting vectors are projected
  to the subvector/module dimensionality ``d_out / n_modules`` via a matrix
  multiply, and finally reshaped back to a single vector of dimensionality
  ``d_out``. Optionally, a bias vector of dimensionality ``d_out`` is added at
  the end. All the above-mentioned non-input objects -- gating vectors,
  projection matrix, and optional bias -- are trainable weights.

  Args:
    n_modules: Number by which an activation vector is divided into subvectors
        (modules) for the factored computation.
    d_in: Last/innermost dimension of input array.
    d_out: Last/innermost dimension of output array.
    use_bias: If True, add bias vectors at the end of the layer; else end the
        layer with the matrix multiply.
    use_bfloat16: If True, use bfloat16 weights; else use float32 weights.
  """
  if d_out % n_modules != 0:
    raise ValueError(f'Value d_out ({d_out}) must be a multiple of arg '
                     f'n_modules ({n_modules}).')
  d_module = d_out // n_modules

  def GatingVectors():
    return tl.Weights(init.RandomNormalInitializer(stddev=0.5),
                      shape=[n_modules, d_in],
                      use_bfloat16=use_bfloat16)

  def ProjectionMatrix():
    return tl.Weights(init.GlorotUniformInitializer(),
                      shape=[d_in, d_module],
                      use_bfloat16=use_bfloat16),

  def Bias():
    return tl.Weights(init.RandomNormalInitializer(1e-6),
                      shape=[d_out],
                      use_bfloat16=use_bfloat16),

  layers = [
      GatingVectors(),
      ProjectionMatrix(),
      _GateAndProject(),
      MergeLastTwoAxes(),
  ]
  if use_bias:
    layers += [Bias(), tl.Add()]

  return tl.Serial(layers)


def _GateAndProject():
  """Returns a combined gating+projection layer that saves on memory."""

  def f(projection, gating, x):
    # Args arrive in reverse order because of how they were put on the stack.
    # Einsum indices: d (d_in), n (n_modules), m (d_module = d_out/n_modules)
    return jnp.einsum('...d,nd,dm->...nm', x, gating, projection)

  return tl.Fn('_GateAndProject', f)


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
      FactoredDense(sparsity, d_feature, d_feature),
      FactoredDense(sparsity, d_feature, d_feature),
      FactoredDense(sparsity, d_feature, d_feature),
      FactoredDense(sparsity, d_feature, d_feature),
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
  FactoredDense layer with LocallyConnectedLayer.

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
    d_feature, n_heads=1, sparsity=None, length_kernel_size=3, dropout=0.0,
    force_no_dropout=False, max_inference_length=2048, share_qk=False,
    output_layer_type='none', v_concat_type='none', mode='train'):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  self-attention with causal masking rather than padding-based masking. However,
  for computing Q/K/V instead of a Dense layer it combines
  FactoredDense layer with LocallyConvLayer.

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    sparsity: The sparsity of the layer; usually it should be equal to n_heads.
    length_kernel_size: Size of convolution kernel on the length dimension.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    force_no_dropout: If True, force dropout to be 0.0 independent of the above
        value; used to override some configurations.
    max_inference_length: maximum length for inference.
    share_qk: if True, average Q and K embeddings and share for both Q and K.
    output_layer_type: Which sparse layers to use for processing output from the
        attention mechanism. One of `'none'`, `'mult'`, `'conv'`,
        or `'multconv'`.
    v_concat_type: What kind of concatenation to use when computing V tensor.
        One of `'original'`, `'fixed'`, or `'none'`. `'none'` means using just
        output from mutliplicative layer shared by Q, K, V. `'fixed'` means
        using output from multiplicative layer concatenated, for each module,
        with the layer input. `'original'` means using concatenation without
        properly taking modules into account; this method was used in
        experiments previously, so it is included for backwards-compatibility.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  assert output_layer_type in ['none', 'mult', 'conv', 'multconv']
  assert v_concat_type in ['original', 'fixed', 'none']

  dropout = 0.0 if force_no_dropout else dropout
  sparsity = n_heads if sparsity is None else sparsity
  d_module = d_feature // sparsity

  output_layers = []
  if 'mult' in output_layer_type:
    output_layers.append(FactoredDense(
        sparsity, d_feature, d_feature))
  if 'conv' in output_layer_type:
    output_layers.append(LocallyConvDense(
        sparsity, d_module, mode=mode, kernel_size=3,
        length_kernel_size=length_kernel_size))

  if v_concat_type == 'original':
    # 'original'` uses concatenation without properly taking modules into
    # account; this method was used in experiments previously, so it is included
    # for backwards-compatibility.
    concat_layers = [tl.Concatenate()]  # use permuted and original for v
  elif v_concat_type == 'fixed':
    # `'fixed'` uses the output from multiplicative layer concatenated, for each
    # module, with the layer input. This means that every module in Conv layer
    # has access both to parts of embeddings which were used to compute Q/K of
    # this particular module, and it ha access to parts of the embedding which
    # will be modified by this module.
    concat_layers = [
        tl.Parallel(
            tl.Fn('Reshape1', lambda x: jnp.reshape(  # pylint: disable=g-long-lambda
                x, (x.shape[0], x.shape[1], sparsity, d_module))),
            tl.Fn('Reshape2', lambda x: jnp.reshape(  # pylint: disable=g-long-lambda
                x, (x.shape[0], x.shape[1], sparsity, d_module)))),
        tl.Concatenate(),
        tl.Fn('Reshape3',
              lambda x: jnp.reshape(x, (x.shape[0], x.shape[1], 2*d_feature))),
    ]
  elif v_concat_type == 'none':
    # `'none'` doesn't use concatenation: we throw away the original layer
    # input and pass to Conv only output of shared Multiplicative layer.
    concat_layers = [tl.Select([0], n_in=2)]

  if share_qk:
    return tl.Serial(
        tl.Select([0, 0]),  # pre-qkv, pre-v-for-concat
        FactoredDense(sparsity, d_feature, d_feature),  # shared q k
        tl.Select([0, 0]),  # pre-qk, pre-v, pre-v-for-concat
        LocallyConvDense(sparsity, d_module, mode=mode, kernel_size=3,
                         length_kernel_size=length_kernel_size),
        tl.SplitIntoHeads(n_heads),
        tl.Select([0, 0]),  # use for q and k
        tl.Parallel(
            [],
            [],
            [concat_layers,
             LocallyConvDense(sparsity, d_module, mode=mode, kernel_size=1,
                              length_kernel_size=length_kernel_size),
             tl.SplitIntoHeads(n_heads)],
        ),
        tl.DotProductCausalAttention(
            dropout=dropout, max_inference_length=max_inference_length,
            mode=mode),
        tl.MergeHeads(n_heads),
        output_layers,
    )
  return tl.Serial(
      tl.Select([0, 0]),  # duplicate activations
      FactoredDense(sparsity, d_feature, d_feature),  # shared q, k
      tl.Select([0, 0, 0]),  # use for q, k, v
      tl.Parallel(
          [LocallyConvDense(sparsity, d_module, mode=mode, kernel_size=3,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
          [LocallyConvDense(sparsity, d_module, mode=mode, kernel_size=3,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
          [concat_layers,
           LocallyConvDense(sparsity, d_module, mode=mode, kernel_size=1,
                            length_kernel_size=length_kernel_size),
           tl.SplitIntoHeads(n_heads)],
      ),
      tl.DotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode),
      tl.MergeHeads(n_heads),
      output_layers,
  )


class FavorAttention(base.Layer):
  """Implements FAVOR+ attention.

  Original paper: https://arxiv.org/abs/2006.03555
  The layer expects 4 inputs: (Q, K, V, MASK), and returns two outputs:
  (RENORMALIZED_ATTENTION, MASK).

  Attributes:

    d_feature: Dimensionality of feature embedding.
    n_heads: Number of attention heads.
    n_random_features: Free dimension size for the orthogonal random matrix.
    numerical_stabilizer: float, small number used for numerical stability.
    use_approximate_softmax: Bool, if True uses approximate softmax, otherwise
                             Relu.
    scale_by_norm: Boolean; whether to scale orthogonal random matrix.
    normalize_data: predicate indicating whether data should be normalized.
    epsilon: numerical stabilizer.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  def __init__(self, d_feature=4, n_heads=1, n_random_features=256,
               numerical_stabilizer=0.001,
               use_approximate_softmax=False, scale_by_norm=True,
               normalize_data=False,
               epsilon=0.0001, mode='train'):
    super().__init__(n_in=4, n_out=2)
    self._d_feature = d_feature
    self._n_heads = n_heads
    self._n_random_features = n_random_features
    self._numerical_stabilizer = numerical_stabilizer
    self._mode = mode
    self._use_approximate_softmax = use_approximate_softmax
    self._normalize_data = normalize_data
    self._epsilon = epsilon
    if self._use_approximate_softmax:
      rng = random.get_prng(0)
      self._projection_matrix = self.get_2d_array(
          rng=rng, n_rows=self._n_random_features,
          n_columns=(self._d_feature // self._n_heads),
          scale_by_norm=scale_by_norm,
          normalize_data=normalize_data, epsilon=epsilon)
    else:
      self._projection_matrix = None

  def nonnegative_softmax_kernel_feature_creator(self, x, is_query):
    """Constructs nonnegative kernel features for fast softmax attention.

    Args:
      x: input for which features are computed.
      is_query: predicate indicating whether input data corresponds to
                queries or keys.

    Returns:
      Random features for fast softmax attention.
    """
    if self._normalize_data:
      # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
      # w_norm = w * data_normalizer for w in {q,k}.
      data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(x.shape[-1])))
    else:
      data_normalizer = 1.0
    ratio = 1.0 / jnp.sqrt(self._projection_matrix.shape[0])
    # TODO(wgaj): Double-check... Should there be only one batch dimension...?
    data_mod_shape = x.shape[0:1] + self._projection_matrix.shape
    data_thick_random_matrix = (jnp.zeros(data_mod_shape) +
                                self._projection_matrix)

    data_dash = jnp.einsum('Bij, Bkj -> Bik',
                           data_normalizer * x,
                           data_thick_random_matrix)
    diag_data = jnp.square(x)
    diag_data = jnp.sum(diag_data, axis=x.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = jnp.expand_dims(diag_data, axis=x.ndim - 1)

    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (1,)
    if is_query:
      data_dash = ratio * (
          jnp.exp(data_dash - diag_data -
                  jnp.max(data_dash, axis=last_dims_t, keepdims=True)) +
          self._epsilon)
    else:
      data_dash = ratio * (
          jnp.exp(data_dash - diag_data - jnp.max(
              data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
          self._epsilon)

    return data_dash

  @staticmethod
  def get_2d_array(rng, n_rows=256, n_columns=0, scale_by_norm=True,
                   normalize_data=False, epsilon=0.0001):
    """Generator for approximate softmax orthogonal kernel feature matrix.

    Args:
      rng: Random number generator.
      n_rows: Number of rows.
      n_columns: Number of columns.
      scale_by_norm: Boolean; whether to scale orthogonal random matrix.
      normalize_data: predicate indicating whether data should be normalized.
      epsilon: numerical stabilizer.

    Returns:
      Orthogonal kernel feature matrix.
    """
    n_full_blocks = int(n_rows / n_columns)
    block_list = []
    rng_key = rng
    for _ in range(n_full_blocks):
      rng, rng_input = random.split(rng)
      unstructured_block = random.normal(rng_input, (n_columns, n_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = n_rows - n_full_blocks * n_columns
    if remaining_rows > 0:
      rng, rng_input = random.split(rng)
      unstructured_block = random.normal(rng_input, (n_columns, n_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    if scale_by_norm:
      multiplier = jnp.linalg.norm(
          random.normal(rng_key, (n_rows, n_columns)), axis=1)
    else:
      multiplier = jnp.sqrt(float(n_columns)) * jnp.ones((n_rows))

    return jnp.matmul(jnp.diag(multiplier), final_matrix)

  @staticmethod
  def bidirectional_numerator(query_prime, key_prime, value):
    kvs = jnp.einsum('lbm,lbd->bmd', key_prime, value)
    return jnp.einsum('lbm,bmd->lbd', query_prime, kvs)

  @staticmethod
  def bidirectional_denominator(query_prime, key_prime):
    all_ones = jnp.ones([query_prime.shape[0]])
    ks_sum = jnp.einsum('lbm,l->bm', key_prime, all_ones)
    return jnp.einsum('lbm,bm->lb', query_prime, ks_sum)

  @staticmethod
  def relu(x):
    return jnp.where(x <= 0, jnp.zeros_like(x), x)

  def forward(self, inputs):
    query, key, value, mask = inputs
    if self._use_approximate_softmax:
      query_prime = self.nonnegative_softmax_kernel_feature_creator(query, True)
      key_prime = self.nonnegative_softmax_kernel_feature_creator(key, False)
    else:
      query_prime = self.relu(query) + self._numerical_stabilizer
      key_prime = self.relu(key) + self._numerical_stabilizer
    mask_batch_1_length = jnp.reshape(
        mask, [key.shape[0] // self._n_heads, 1, key.shape[1]]).astype(
            jnp.float32)
    mask_heads = mask_batch_1_length + jnp.zeros((1, self._n_heads, 1))
    key_prime *= jnp.reshape(mask_heads, [key.shape[0], key.shape[1], 1])

    w = self.bidirectional_numerator(jnp.moveaxis(query_prime, 1, 0),
                                     jnp.moveaxis(key_prime, 1, 0),
                                     jnp.moveaxis(value, 1, 0))
    r = self.bidirectional_denominator(jnp.moveaxis(query_prime, 1, 0),
                                       jnp.moveaxis(key_prime, 1, 0))
    w = jnp.moveaxis(w, 0, 1)
    r = jnp.moveaxis(r, 0, 1)
    r = jnp.reciprocal(r)
    r = jnp.expand_dims(r, len(r.shape))
    renormalized_attention = w * r
    return renormalized_attention, mask


def Favor(d_feature, n_heads=1, n_random_features=256, dropout=0.0,
          numerical_stabilizer=0.001, use_approximate_softmax=False,
          scale_by_norm=0, normalize_data=False, epsilon=0.0001, mode='train'):
  """Returns a layer that maps (activations, mask) to (new_activations, mask).

  See the FAVOR paper for details: https://arxiv.org/abs/2006.03555

  Args:
    d_feature: Depth/dimensionality of feature embedding.
    n_heads: Number of attention heads.
    n_random_features: Free dimension size for the orthogonal random matrix.
    dropout: Probababilistic rate for internal dropout applied to attention
        activations (based on query-key pairs) before dotting them with values.
    numerical_stabilizer: float, small number used for numerical stability.
    use_approximate_softmax: Bool, if True uses approximate softmax, otherwise
                             Relu.
    scale_by_norm: Boolean; whether to scale orthogonal random matrix.
    normalize_data: predicate indicating whether data should be normalized.
    epsilon: numerical stabilizer.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  del dropout  # not implemented yet but needed in the API

  return  tl.ConfigurableAttention(
      tl.Dense(d_feature), tl.Dense(d_feature), tl.Dense(d_feature),
      tl.Dense(d_feature),
      tl.FavorAttention(d_feature, n_heads, n_random_features,
                        numerical_stabilizer, use_approximate_softmax,
                        scale_by_norm, normalize_data, epsilon, mode),
      n_heads=n_heads)


class CausalFavorAttention(base.Layer):
  """Returns a layer that maps activations to activations, with causal masking.

  Like `CausalAttention`, this layer type represents one pass of multi-head
  causal attention, but using FAVOR fast attention as in the following paper:
  https://arxiv.org/abs/2006.03555

  Layer expects three inputs (Q, K, V), and returns one output
   RENORMALIZED_ATTENTION.

  Attributes:
    numerical_stabilizer: float, small number used for numerical stability.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """

  def __init__(self, numerical_stabilizer=0.001, mode='train'):
    super().__init__(n_in=3, n_out=1)
    self._numerical_stabilizer = numerical_stabilizer
    self._mode = mode

  def forward(self, inputs):
    def favor_numerator_fwd(init_prefix_sum_value,
                            query_prime, key_prime, value):
      def body(p, qkv):
        (q, k, v) = qkv
        p += jnp.einsum('...m,...d->...md', k, v)
        x_slice = jnp.einsum('...m,...md->...d', q, p)
        return p, x_slice
      p, w = fastmath.scan(body, init_prefix_sum_value,
                           (query_prime, key_prime, value))
      return w, (p, query_prime, key_prime, value)

    def favor_numerator_bwd(pqkv, w_ct):
      p, qs, ks, vs = pqkv

      def body(carry, qkv_xct):
        p, p_ct = carry
        q, k, v, x_ct = qkv_xct
        q_ct = jnp.einsum('...d,...md->...m', x_ct, p)
        p_ct += jnp.einsum('...d,...m->...md', x_ct, q)
        k_ct = jnp.einsum('...md,...d->...m', p_ct, v)
        v_ct = jnp.einsum('...md,...m->...d', p_ct, k)
        p -= jnp.einsum('...m,...d->...md', k, v)
        return (p, p_ct), (q_ct, k_ct, v_ct)

      _, (qs_ct, ks_ct, vs_ct) = fastmath.scan(
          body, (p, jnp.zeros_like(p)), (qs, ks, vs, w_ct), reverse=True)
      return (None, qs_ct, ks_ct, vs_ct)

    def favor_numerator(init_prefix_sum_value, query_prime,
                        key_prime, value):
      w, _ = favor_numerator_fwd(init_prefix_sum_value,
                                 query_prime, key_prime, value)
      return w

    favor_numerator = fastmath.custom_vjp(
        favor_numerator, favor_numerator_fwd, favor_numerator_bwd)

    def favor_denominator_fwd(init_prefix_sum_value,
                              query_prime, key_prime):
      def body(p, qk):
        q, k = qk
        p += k
        x = jnp.einsum('...m,...m->...', q, p)
        return p, x

      p, r = fastmath.scan(body, init_prefix_sum_value, (query_prime,
                                                         key_prime))
      return r, (query_prime, key_prime, p)

    def favor_denominator_bwd(qkp, r_ct):
      qs, ks, p = qkp

      def body(carry, qkx):
        p, p_ct = carry
        q, k, x_ct = qkx
        q_ct = jnp.einsum('...,...m->...m', x_ct, p)
        p_ct += jnp.einsum('...,...m->...m', x_ct, q)
        k_ct = p_ct
        p -= k
        return (p, p_ct), (q_ct, k_ct)

      _, (qs_ct, ks_ct) = fastmath.scan(
          body, (p, jnp.zeros_like(p)), (qs, ks, r_ct), reverse=True)
      return (None, qs_ct, ks_ct)

    def favor_denominator(init_prefix_sum_value, query_prime,
                          key_prime):
      r, _ = favor_denominator_fwd(init_prefix_sum_value,
                                   query_prime, key_prime)
      return r

    favor_denominator = fastmath.custom_vjp(
        favor_denominator, favor_denominator_fwd, favor_denominator_bwd)

    favor_denominator.defvjp(favor_denominator_fwd, favor_denominator_bwd)

    def relu(x):
      return jnp.where(x <= 0, jnp.zeros_like(x), x)

    query, key, value = inputs
    query_prime = relu(query) + self._numerical_stabilizer
    key_prime = relu(key) + self._numerical_stabilizer
    prefix_sum_tensor_shape = (key.shape[0], key.shape[-1], value.shape[-1])
    t_slice_shape = (key.shape[0], key.shape[-1])
    init_prefix_sum_value_numerator = jnp.zeros(prefix_sum_tensor_shape)
    init_prefix_sum_value_denominator = jnp.zeros(t_slice_shape)

    w = favor_numerator(init_prefix_sum_value_numerator,
                        jnp.moveaxis(query_prime, 1, 0),
                        jnp.moveaxis(key_prime, 1, 0),
                        jnp.moveaxis(value, 1, 0))
    r = favor_denominator(init_prefix_sum_value_denominator,
                          jnp.moveaxis(query_prime, 1, 0),
                          jnp.moveaxis(key_prime, 1, 0))
    w = jnp.moveaxis(w, 0, 1)
    r = jnp.moveaxis(r, 0, 1)
    r = jnp.reciprocal(r)
    r = jnp.expand_dims(r, len(r.shape))
    renormalized_attention = w * r
    return renormalized_attention


def CausalFavor(d_feature, n_heads=1, dropout=0.0,
                numerical_stabilizer=0.001, mode='train'):
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
    mode: One of `'train'`, `'eval'`, or `'predict'`.
  """
  del dropout
  return tl.ConfigurableAttention(
      core.Dense(d_feature), core.Dense(d_feature), core.Dense(d_feature),
      core.Dense(d_feature), n_heads=n_heads,
      qkv_attention_layer=tl.CausalFavorAttention(numerical_stabilizer,
                                                  mode))


class _RememberInReverse(base.Layer):
  """Layer remembering the input in forward pass. For reversible models."""

  def __init__(self, output=True):
    """Layer remembering the input in forward pass. For reversible models.

    During the first pass through the model this layer saves the input as
    state, and returns the input unmodified. During the second pass through the
    model the layer outputs the input from the first pass. This is used to
    combat numerical stability problems in Terraformer. It doesn't do anything
    in non-reversible models.

    Args:
      output: Whether to pass the input or not.
    """
    n_out = 1 if output else 0
    self._output = output
    super().__init__(name='_RememberInReverse', n_out=n_out)

  def forward(self, x):
    if 'running_second_time_yes' in self.state[1]:
      result = self.state[0]
    else:
      result = x
    self.state = (x, {'running_second_time': ()})

    if self._output:
      return result
    else:
      return tuple()

  def init_weights_and_state(self, input_signature):
    """Initializes this layer's weights."""
    if isinstance(input_signature, (list, tuple)):
      input_signature = input_signature[0]
    self.weights = ()
    self.state = (jnp.zeros(input_signature.shape, dtype=jnp.int32),
                  {'running_second_time': ()})


class _RecallQuantMaskInReverse(base.Layer):
  """Layer recalling quant mask from specific _RememberInReverse.

  This layer is needed for memory-efficient training of reversible model with
  ff chunking. During forward pass it simply returns minus ones, which are
  ignored in the controller. During reverse_and_grad it returns a quant_mask
  which was memorized (saved to state) by a RememberInReverse layer.

  This enable us to save quant_mask right after chunking, and load it again
  (when reversing) right before chunking.
  """

  def __init__(self, remember_layer, elements):
    self._remember_layer = remember_layer
    self._elements = elements
    super().__init__(name='_RecallQuantMaskInReverse', n_in=1, n_out=2)

  def forward(self, x):
    if (self._remember_layer.state and
        'running_second_time_yes' in self._remember_layer.state[1]):
      # It's reverse_and_grad, so we pull the quant_mask from remembering layer.
      result = self._remember_layer.state[0]
    else:
      result = -jnp.ones((x.shape[0], self._elements), dtype=jnp.int32)
    return (x, result)


class _SparseFFController(base.Layer):
  """The controller part of Sparse Feed-Forward layer."""

  def __init__(self, d_ff, n_elements_in_block, d_lowrank, temperature,
               use_bfloat16, mode, kernel_initializer, bias_initializer,
               also_return_nondiscrete_output):
    """Returns a sparse feed-forward block."""
    n_out = 2 if also_return_nondiscrete_output else 1
    super().__init__(name=f'_SparseFFController_{d_ff}', n_in=2, n_out=n_out)
    self._use_bfloat16 = use_bfloat16
    self._d_ff = d_ff
    self._d_lowrank = d_lowrank
    # Q: what temperature is actually most useful in training?
    self._temperature = temperature if mode == 'train' else 0.0
    self._mode = mode
    self._n_elements_in_block = n_elements_in_block
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    # Helper numbers as d_ff will be divided by n_elements_in_block.
    assert self._d_ff % self._n_elements_in_block == 0
    self._d1 = self._d_ff // self._n_elements_in_block
    self._d2 = self._n_elements_in_block
    self._also_return_nondiscrete_output = also_return_nondiscrete_output

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    x, recalled_quant_mask = x
    m1, m2, mb = self.weights

    x_shape = x.shape
    x = jnp.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    # Q: should we add bias and/or put relu after the low-rank m1 dot?
    # Replacing multiplication and reshape by this einsum brings training speed
    # improvement (see also reshape in initialization).
    mask_logits = jnp.einsum('bd,dl,lxy->bxy', x, m1, m2) + mb

    if self._also_return_nondiscrete_output:
      # Softmax.
      mask_logsumexp = fastmath.logsumexp(mask_logits, axis=-1, keepdims=True)
      log_mask = mask_logits - mask_logsumexp
      mask = jnp.exp(log_mask)
      # Gumbel-softmax with straight-through discretization.
      if self._temperature == 0.0:
        quant_mask = jnp.argmax(log_mask, axis=-1)
      else:
        u = fastmath.random.uniform(self.rng, mask.shape, jnp.float32, 1e-6,
                                    1.0 - 1e-6)
        g = -jnp.log(-jnp.log(u))
        quant_mask = jnp.argmax(log_mask + g * self._temperature, axis=-1)
    else:
      quant_mask = jnp.argmax(mask_logits, axis=-1)

    if self._mode == 'train':
      # We use recalled_quant_mask if it's different than -1; otherwise
      # we use a quant_mask which we have just computed.
      quant_mask = jnp.where(recalled_quant_mask == -1,
                             quant_mask, recalled_quant_mask)

    if self._also_return_nondiscrete_output:
      return quant_mask, mask
    else:
      return quant_mask

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    x_input_signature = input_signature[0]
    d_model = x_input_signature.shape[-1]
    shape_m1 = (d_model, self._d_lowrank)
    shape_m2 = (self._d_lowrank, self._d_ff)
    shape_mb = (self._d_ff,)

    rng_m1, rng_m2, rng_mb = fastmath.random.split(self.rng, 3)
    m1 = self._kernel_initializer(shape_m1, rng_m1)
    m2 = self._kernel_initializer(shape_m2, rng_m2)
    mb = self._bias_initializer(shape_mb, rng_mb)
    if self._use_bfloat16:
      m1 = m1.astype(jnp.bfloat16)
      m2 = m2.astype(jnp.bfloat16)
      mb = mb.astype(jnp.bfloat16)

    # Reshapes below, with einsum in feedforward, improve the training speed.
    m2 = jnp.reshape(m2, [self._d_lowrank, self._d1, self._d2])
    mb = jnp.reshape(mb, [self._d1, self._d2])

    self.weights = (m1, m2, mb)


class _SparseFFMain(base.Layer):
  """The main (non-controller) part of Sparse Feed-Forward layer."""

  def __init__(self, d_ff, n_elements_in_block, d_lowrank, quant_prob,
               use_bfloat16, big_weights_in_bfloat16, mode, kernel_initializer,
               bias_initializer, multiply_by_controller_output, kernel_scaling):
    """Returns a sparse feed-forward block."""
    n_in = 3 if mode == 'train' or multiply_by_controller_output else 2
    super().__init__(name=f'_SparseFFMain_{d_ff}', n_in=n_in, n_out=2)
    self._mode = mode
    self._use_bfloat16 = use_bfloat16
    self._big_weights_in_bfloat16 = big_weights_in_bfloat16
    self._d_ff = d_ff
    self._d_lowrank = d_lowrank
    self._quant_prob = quant_prob
    self._n_elements_in_block = n_elements_in_block
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    # Helper numbers as d_ff will be divided by n_elements_in_block.
    assert self._d_ff % self._n_elements_in_block == 0
    self._d1 = self._d_ff // self._n_elements_in_block
    self._d2 = self._n_elements_in_block
    self._multiply_by_controller_output = multiply_by_controller_output
    self._kernel_scaling = kernel_scaling

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input.
    """
    if self._mode == 'train' or self._multiply_by_controller_output:
      quant_mask, mask, x = x
    else:
      quant_mask, x = x
    original_quant_mask = quant_mask

    w1, w2, b2 = self.weights

    if self._mode == 'predict':
      w1 = jnp.transpose(w1, (1, 2, 0))  # dm, d1, d2 -> d1, d2, dm
      w2 = jnp.transpose(w2, (1, 0, 2))  # d2, d1, dm -> d1, d2, dm
    x_shape = x.shape
    x = jnp.reshape(x, [-1, x_shape[-1]])  # Easier to operate on flattened x.

    if self._mode == 'train':
      # Tricks from Section 2.1 in https://arxiv.org/abs/1801.09797
      quant_mask = tl.one_hot(quant_mask, self._n_elements_in_block)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (quant_prob of the batches) use the soft-mask instead
      # of the quantized mask to improve training stability (see paper above).
      select = fastmath.random.uniform(self.rng, (), jnp.float32, 0.0, 1.0)
      quant_mask = jnp.where(select < self._quant_prob, quant_mask, mask)

      # In training, run full matmul to get benefits from the above tricks.
      mid = jnp.einsum('bd,dxy->bxy', x, w1) * quant_mask
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      if self._multiply_by_controller_output:
        # We multiply only for quantized decisions, since for non-quantized
        # decisions we've already multiplied the output.
        mask_mult = jnp.where(select < self._quant_prob,
                              mask, jnp.ones_like(mask))
        # Stop-gradient is here, because we already have a pass-through gradient
        # (for quantized decisions).
        mask_mult = fastmath.stop_gradient(mask_mult)
        relu = relu * mask_mult
      res = jnp.einsum('bxy,yxd->bd', relu, w2) + b2
    elif self._mode == 'predict':
      # This implementation mimicks inference. It's not efficient for large
      # size of joint_batch, but at inference that will be 1 most of the time.
      # Shapes:
      # quant_mask is [joint_batch, self._d1]
      # w1 is [d_model, self._d1, self._d2]
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
      if self._multiply_by_controller_output:
        mask_mult = jnp.take_along_axis(mask, quant_mask[..., None], -1)[..., 0]
        relu = relu * mask_mult
      # w2 is [self._d1, self._d2, d_model]
      v = w2[idx1, idx2, :]
      v = jnp.reshape(v, [batch_size, self._d1, -1])
      res = jnp.einsum('ai,aij->aj', relu, v) + b2
    else:
      quant_mask = tl.one_hot(quant_mask, self._n_elements_in_block)
      mid = jnp.einsum('bd,dxy->bxy', x, w1) * quant_mask
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      if self._multiply_by_controller_output:
        relu = relu * mask
      res = jnp.einsum('bxy,yxd->bd', relu, w2) + b2

    return original_quant_mask, jnp.reshape(res, x_shape)

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature[-1].shape[-1]
    shape_w1 = (d_model, self._d_ff)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_w1, rng_w2, rng_b2 = fastmath.random.split(self.rng, 3)
    if base.N_WEIGHTS_SHARDS > 1:
      # In sharded-weights mode, put the weights on CPU on init
      # as they will be sharded later.
      w1 = tl.on_cpu(self._kernel_initializer(shape_w1, rng_w1))
      w2 = tl.on_cpu(self._kernel_initializer(shape_w2, rng_w2))
    else:
      w1 = self._kernel_initializer(shape_w1, rng_w1)
      w2 = self._kernel_initializer(shape_w2, rng_w2)

    b2 = self._bias_initializer(shape_b2, rng_b2)
    if self._use_bfloat16:
      b2 = b2.astype(jnp.bfloat16)
    if self._use_bfloat16 or self._big_weights_in_bfloat16:
      w1 = w1.astype(jnp.bfloat16)
      w2 = w2.astype(jnp.bfloat16)

    w1 = jnp.reshape(w1, (-1, self._d1, self._d2))
    w2 = jnp.reshape(w2, (self._d2, self._d1, -1))

    if self._kernel_scaling:
      # This keeps expected variance of the output regardless of N.
      w2 = w2 * (self._n_elements_in_block ** 0.5)

    self.weights = (w1, w2, b2)


def SparseFF(
    d_ff, n_elements_in_block=32, d_lowrank=64, temperature=0.1, quant_prob=0.3,
    use_bfloat16=False, big_weights_in_bfloat16=False, mode='train',
    kernel_initializer=init.GlorotUniformInitializer(),
    bias_initializer=init.RandomNormalInitializer(1e-6),
    dropout_rate=0.0, dropout_shared_axes=None, ff_chunk_size=0,
    multiply_by_controller_output=False, kernel_scaling=False):
  """Returns Feed-forward block with sparsity.

  The original (non-sparse) FF block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  The original block can be slow in decoding due to the need to fetch a lot of
  weights from memory. This sparse block only allows one non-zero element
  in a block of a specified size. This is trained with straight-through Gumbel
  softmax trick.

  Args:
    d_ff: Depth/dimensionality of FeedForward layer.
    n_elements_in_block: The sparsity level. The layer is divided into blocks of
      this size, and each block has only a single element active.
    d_lowrank: The dimensionality of low-rank controller.
    temperature: The temperature of the controller during training.
    quant_prob: During training this proportion of blocks will have quantized
      mask (i.e. a single element active). The rest will use a soft mask.
    use_bfloat16: Whether to use bfloat16 for weights.
    big_weights_in_bfloat16: : Whether to use bfloat16 for main weights of the
      FeedForward layer.
    mode: One of `'train'`, `'eval'`, or `'predict'`.
    kernel_initializer: Function that creates a matrix of (random) initial
        connection weights `W` for the layer.
    bias_initializer: Function that creates a vector of (random) initial
        bias weights `b` for the layer.
    dropout_rate: Probability for dropping an activation value.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks.
    multiply_by_controller_output: whether to multiply the middle activation
      layer of FF by controller output (i.e. softmax).
    kernel_scaling: Whether to scale the kernel matrix (during init) to keep the
      variance of the layer output regardless of n_elements_in_block.
  """

  if mode == 'train' or multiply_by_controller_output:
    also_return_nondiscrete_output = True
  else:
    also_return_nondiscrete_output = False
  controller = _SparseFFController(
      d_ff=d_ff, n_elements_in_block=n_elements_in_block,
      d_lowrank=d_lowrank, temperature=temperature,
      use_bfloat16=use_bfloat16, mode=mode,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      also_return_nondiscrete_output=also_return_nondiscrete_output)

  main = [
      _SparseFFMain(
          d_ff=d_ff, n_elements_in_block=n_elements_in_block,
          d_lowrank=d_lowrank, quant_prob=quant_prob, use_bfloat16=use_bfloat16,
          big_weights_in_bfloat16=big_weights_in_bfloat16, mode=mode,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          multiply_by_controller_output=multiply_by_controller_output,
          kernel_scaling=kernel_scaling),
      # quant_mask, emb
      tl.Select([1, 0]),
      # emb, quant_mask
      tl.Dropout(rate=dropout_rate, shared_axes=dropout_shared_axes, mode=mode),
      tl.Select([1, 0]),
      # quant_mask, emb
  ]

  # We will "remember" quant_mask _after_ chunking, and "recall" this same
  # quant_mask during reverse_and_grad _before_ chunking.
  remembering = _RememberInReverse(output=False)
  recalling = _RecallQuantMaskInReverse(
      remember_layer=remembering, elements=d_ff//n_elements_in_block)

  return tl.BatchLeadingAxes(tl.Serial(
      recalling,  # emb, quant_mask
      tl.Chunk(chunk_size=ff_chunk_size, layer=tl.Serial(
          # emb, quant_mask
          tl.Select((0, 1, 0)),  # emb, quant_mask, emb
          controller,  # quant_mask, mask, emb
          main,  # quant_mask, emb/output
          )),
      remembering,  # emb/output
      ))


class BlockSparseFF(base.Layer):
  """Feed-forward block with block sparsity.

  The original (non-sparse) FF block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  This block sparse layer mimics mixture of experts architecture.
  It divides the dimension of d_ff in each weight matrix to # of blocks equal to
  n_experts and activates only one non-zero block from the weights matrix.
  This is trained with straight-through Gumbel softmax trick.
  """

  def __init__(self,
               d_ff,
               n_experts=64,
               temperature=0.7,
               mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a block sparse feed-forward block."""
    super().__init__(name=f'BlockSparseFF_{d_ff}')
    self._mode = mode
    self._d_ff = d_ff
    self._n_experts = n_experts
    self._temperature = temperature if mode == 'train' else 0.0
    self._n_elements_in_block = d_ff // n_experts
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    assert self._d_ff % self._n_experts == 0

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
      quant_mask = tl.one_hot(selected_experts, self._n_experts)
      quant_mask = fastmath.stop_gradient(quant_mask)
      quant_mask += mask - fastmath.stop_gradient(mask)  # straight-through
      # We will sometimes (50% of the batches) use the soft-mask instead of
      # the quantized mask to improve training stability (see the paper above).
      # Q: is selecting 50% of batches the best? Other %? Mixed in-batch?
      select = fastmath.random.uniform(rng2, (), jnp.float32, -1.0, 1.0)
      quant_mask = jnp.where(select > 0.0, quant_mask, mask)
    else:
      quant_mask = tl.one_hot(selected_experts, self._n_experts)
    quant_mask = jnp.reshape(quant_mask, [-1, self._n_experts, 1])
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
          (quant_mask.shape[0], quant_mask.shape[1], self._n_elements_in_block))
      expanded_mask = jnp.reshape(expanded_mask, (-1, self._d_ff))
      mid = jnp.dot(x, w1) * expanded_mask  # [joint_batch, d_ff]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      res = jnp.dot(relu, w2) + b2

    return jnp.reshape(res, x_shape)  # un-flatten if needed

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature.shape[-1]
    shape_m1 = (d_model, self._n_experts)
    shape_w1 = (d_model, self._d_ff)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_m1, rng_w1, rng_w2, rng_b2 = fastmath.random.split(self.rng, 4)
    m1 = self._kernel_initializer(shape_m1, rng_m1)
    w1 = self._kernel_initializer(shape_w1, rng_w1)
    w2 = self._kernel_initializer(shape_w2, rng_w2)
    b2 = self._bias_initializer(shape_b2, rng_b2)

    self.weights = (m1, w1, w2, b2)


class SwitchSparseFF(base.Layer):
  """Feed-forward block with switch-style block sparsity.

  The original (non-sparse) FF block is a triple Dense(d_ff)-Relu-Dense
  that takes an input, makes it of size d_ff (usually larger than it was) and
  then brings it back to the original size after Relu. It is commonly used in
  Transformer models where it often accounts for most of the trainable weights.

  This block sparse layer mimics mixture of experts architecture.
  It divides the dimension of d_ff in each weight matrix to # of blocks equal to
  n_experts and activates only one non-zero block from the weights matrix.
  This is trained with methods following the Switch Transformer.
  """

  def __init__(self,
               d_ff,
               n_experts=64,
               temperature=0.1,
               mode='train',
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    """Returns a switch-style training block sparse feed-forward block."""
    super().__init__(name=f'SwitchSparseFF_{d_ff}')
    self._mode = mode
    self._d_ff = d_ff
    self._n_experts = n_experts
    self._temperature = temperature if mode == 'train' else 0.0
    self._n_elements_in_block = d_ff // n_experts
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    assert self._d_ff % self._n_experts == 0

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
    # Gumbel noise to allow sampling from the softmax.
    rng1, _ = fastmath.random.split(self.rng, 2)
    u = fastmath.random.uniform(rng1, mask.shape, jnp.float32, 1e-6, 1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    selected_experts = jnp.argmax(log_mask + g * self._temperature, axis=-1)
    quant_mask = tl.one_hot(selected_experts, self._n_experts)
    quant_mask = fastmath.stop_gradient(quant_mask)
    quant_mask *= mask  # go to just the selected expert
    quant_mask = jnp.reshape(quant_mask, [-1, self._n_experts, 1])
    batch_size = quant_mask.shape[0]

    if self._mode == 'predict' and batch_size == 1:
      mask_flat = jnp.reshape(mask, [-1, self._n_experts])
      selected_flat = jnp.reshape(selected_experts, [-1])
      selected_mask_flat = mask_flat[np.arange(selected_flat.size),
                                     selected_flat]
      # This implementation mimicks inference for batch_size 1.
      start_idx = selected_experts[0] * self._n_elements_in_block
      # w1 is [d_model, d_ff], w is [d_model, n_elements_in_block]
      w = fastmath.dynamic_slice(w1, [0, start_idx],
                                 [w1.shape[0], self._n_elements_in_block])
      mid = jnp.dot(x, w)
      mid *= jnp.reshape(selected_mask_flat, mid.shape[:-1])[..., None]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      # w2 is [d_ff, d_model], v is [n_elements_in_block, d_model]
      v = fastmath.dynamic_slice(w2, [start_idx, 0],
                                 [self._n_elements_in_block, w2.shape[-1]])
      v = jnp.reshape(v, [self._n_elements_in_block, -1])
      res = jnp.dot(relu, v) + b2
    else:
      expanded_mask = jnp.broadcast_to(
          quant_mask,
          (quant_mask.shape[0], quant_mask.shape[1], self._n_elements_in_block))
      expanded_mask = jnp.reshape(expanded_mask, (-1, self._d_ff))
      mid = jnp.dot(x, w1) * expanded_mask  # [joint_batch, d_ff]
      relu = jnp.where(mid <= 0, jnp.zeros_like(mid), mid)
      res = jnp.dot(relu, w2) + b2

    return jnp.reshape(res, x_shape)  # un-flatten if needed

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights."""
    d_model = input_signature.shape[-1]
    shape_m1 = (d_model, self._n_experts)
    shape_w1 = (d_model, self._d_ff)
    shape_w2 = (self._d_ff, d_model)
    shape_b2 = (d_model,)

    rng_m1, rng_w1, rng_w2, rng_b2 = fastmath.random.split(self.rng, 4)
    m1 = self._kernel_initializer(shape_m1, rng_m1)
    w1 = self._kernel_initializer(shape_w1, rng_w1)
    w2 = self._kernel_initializer(shape_w2, rng_w2)
    b2 = self._bias_initializer(shape_b2, rng_b2)

    self.weights = (m1, w1, w2, b2)
