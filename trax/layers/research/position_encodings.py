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
"""Experimenting with position encodings."""

import logging
import jax
import numpy as np
import trax
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base as layer_base
from trax.layers import initializers as init


class AxialPositionalEncoding(layer_base.Layer):
  """Axial positional encoding."""
  # TODO(kitaev): support variable-length sequences.

  def __init__(self, shape=(64, 64, 3), d_embs=(384, 384, 256),
               kernel_initializer=init.RandomNormalInitializer(1.0),
               dropout=0.0, dropout_broadcast_dims=(), mode='train'):
    super().__init__()
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

  def forward(self, inputs):
    rng, state = self.rng, self.state
    embs = []
    for ax_emb in self.weights:
      ax_emb = jnp.broadcast_to(
          ax_emb, (inputs.shape[0],) + self._shape + (ax_emb.shape[-1],))
      embs.append(ax_emb)

    if self._mode == 'predict':
      assert self._dropout == 0.0
      emb = jnp.concatenate(embs, -1)
      emb = jnp.reshape(emb, (inputs.shape[0], -1, emb.shape[-1]))
      emb = fastmath.dynamic_slice_in_dim(emb, state, inputs.shape[1], axis=1)
      self.state = state + inputs.shape[1]
      return inputs + emb
    elif self._dropout == 0:
      # TODO(kitaev): concat-then-reshape (as is the case with dropout enabled)
      # leads to memory blow-up on TPU.
      # emb = jnp.concatenate(embs, -1)
      # return inputs + jnp.reshape(emb, inputs.shape), state
      return inputs + jnp.concatenate(
          [jnp.reshape(emb, inputs.shape[:-1] + (emb.shape[-1],))
           for emb in embs
          ], -1)
    else:
      emb = jnp.concatenate(embs, -1)
      noise_shape = list(emb.shape)
      for dim in self._dropout_broadcast_dims:
        noise_shape[dim] = 1
      keep_prob = 1.0 - self._dropout
      keep = fastmath.random.bernoulli(rng, keep_prob, tuple(noise_shape))
      multiplier = keep.astype(inputs.dtype) / keep_prob
      return inputs + jnp.reshape(emb * multiplier, inputs.shape)

  def init_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    if sum(self._d_embs) != d_feature:
      raise ValueError(
          f'sum(self._d_embs) != d_feature: '
          f'sum({self._d_embs}) vs d_feature: {d_feature}')

    rngs = fastmath.random.split(self.rng, len(self._d_embs))
    weights = []
    for ax, (ax_rng, d_emb) in enumerate(zip(rngs, self._d_embs)):
      ax_shape = [1] * len(self._shape)
      ax_shape[ax] = self._shape[ax]
      ax_shape = (1,) + tuple(ax_shape) + (d_emb,)
      ax_emb = self._kernel_initializer(ax_shape, ax_rng)
      weights.append(ax_emb)

    # State is EMPTY_STATE by default, stays so except for predict mode.
    if self._mode == 'predict':
      self.state = np.array(0, dtype=np.int32)
    self.weights = tuple(weights)


class FixedBasePositionalEncoding(layer_base.Layer):
  """Implements fixed-base positional encoding."""

  def __init__(self, bases=[11, 13, 14, 15], n_digits=8,  #  pylint: disable=dangerous-default-value
               start_from_zero_one_in=100, base_dropout_one_in=100,
               mode='train', initializer=init.RandomUniformInitializer(1e-4)):
    super().__init__()
    self._bases = bases
    self._n_digits = n_digits
    self._mode = mode
    self._initializer = initializer
    self._start_from_zero_one_in = start_from_zero_one_in
    self._base_dropout_one_in = base_dropout_one_in

  def forward(self, x):
    rng = self.rng
    base_weights, start_vec = self.weights
    batch_size, length = x.shape[0], x.shape[1]
    max_pos = min(self._bases)**self._n_digits
    rng1, rng2, rng3 = fastmath.random.split(rng, 3)
    assert length < max_pos, 'length (%d) >= max_pos (%d)' % (length, max_pos)
    positions = jnp.arange(0, length)[None, :]
    # In training we'll randomize starts for better generalization.
    # We use the trainable start_vec to compensate and give model a way
    # to learn what is the starting position in a sequence.
    if self._mode == 'train':
      # In 1% of training cases still start from 0 to be exactly as in eval.
      start_from_nonzero = fastmath.random.randint(
          rng1, (batch_size,), 0, self._start_from_zero_one_in)
      start_from_nonzero = jnp.minimum(1, start_from_nonzero)
      random_start = fastmath.random.randint(
          rng2, (batch_size,), 0, max_pos-length)
      random_start *= start_from_nonzero
      positions += random_start[:, None]
    if self._mode == 'predict':
      positions += self.state
    res = []
    for bn, base in enumerate(self._bases):
      pos_embeddings = []
      cur_positions = positions
      for i in range(self._n_digits):
        cur_indices = jnp.mod(cur_positions, base)
        cur_positions = cur_positions // base
        s = base_weights[bn][i]
        pos_embeddings.append(cur_indices.astype(jnp.float32)[:, :, None] * s)
      embeddings = jnp.concatenate(pos_embeddings, axis=-1)
      if self._mode == 'train':
        base_dropout = fastmath.random.randint(
            rng3, (batch_size,), 0, self._base_dropout_one_in)
        base_dropout = jnp.minimum(1, base_dropout).astype(jnp.float32)
        embeddings *= base_dropout[:, None, None]
      res.append(embeddings)
    res = sum(res)  # Sum embeddings from all bases.
    # Add start_vec to the first position only to mark it as starting.
    res0 = res[:, 0, :][:, None, :]
    start_pos = res0 + start_vec
    if self._mode == 'predict':
      start_pos = jnp.where(jnp.equal(self.state, 0), start_pos, res0)
      self.state += length  # Add input length to state.
    res = jnp.concatenate([start_pos, res[:, 1:, :]], axis=1)
    return x + res

  def init_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    if d_feature % self._n_digits != 0:
      raise ValueError(
          f'd_feature({d_feature}) % self._n_digits({self._n_digits}) != 0')
    d_weight = d_feature // self._n_digits
    rng1, rng2 = fastmath.random.split(self.rng, 2)
    base_weights = [[self._initializer((1, d_weight), rng)
                     for rng in fastmath.random.split(rng1, self._n_digits)]
                    for _ in self._bases]
    # Special vector to mark the starting position.
    start_vec = self._initializer((1, 1, d_feature), rng2)
    self.weights = (base_weights, start_vec)
    if self._mode == 'predict':
      self.state = jnp.zeros((), dtype=jnp.int32)


def threefry_2x32_prf(key, x: jnp.ndarray) -> jnp.ndarray:
  """Apply the threefry PRF to an array of inputs.

  This function is vectorized over x.
  For threefry_2x32: K = X = uint32[2]

  Args:
    key: uint32[2] the key of the PRF
    x: uint32[..., 2] the inputs

  Returns:
    y: uint32[..., 2] the outputs
  """
  if not (key.shape == (2,) and key.dtype == jnp.uint32):
    raise TypeError('key must be uint32[2]', key)
  if not (x.shape[-1:] == (2,) and x.dtype == jnp.uint32):
    raise TypeError('x must be uint32[..., 2]', x)
  # Threefry-2x32 expects this weird format:
  x_3f = jnp.moveaxis(x, source=-1, destination=0).flatten()
  y_3f = jax.random.threefry_2x32(key, x_3f)
  y = jnp.moveaxis(
      jnp.reshape(y_3f, (2,) + x.shape[:-1]), source=0, destination=-1)
  return y


def threefry_2x32_prange(key, lo: int = 0, hi: int = 2):
  """Splits a key into a stream of random keys.

  This uses the little-endian counter mode.

  Args:
    key: uint32[2] the key to split
    lo: the range to start extracting from
    hi: the range to stop extracting from

  Returns:
    keys: uint32[hi - lo, 2] the split keys
  """
  if not (key.shape == (2,) and key.dtype == jnp.uint32):
    raise ValueError('key must be uint32[2]')
  if not hi < 2**32:
    # You shouldn't really be using more than half the key size anyways.
    raise NotImplementedError('only 32-bit sizes are supported')
  # Create a 64-bit counter:
  i_lo = jnp.arange(lo, hi, dtype=jnp.uint32)
  i_hi = jnp.zeros_like(i_lo)
  i = jnp.stack([i_lo, i_hi], axis=-1)
  return threefry_2x32_prf(key, i)


class InfinitePositionalEncoding(layer_base.Layer):
  """Infinite positional encoding."""

  def __init__(
      self, drift=.03, affine=True, transform='any',
      time_bin_length=None,
      mode='train'):
    """Initializes the encoding.

    The encoding tries to roughly evenly traverse the latent space.
    The recurrence time is dependent on how many bits per dimension you use.

    There are two parameters to control randomization:
    - randomizing the origin every 1/drift steps by letting it drift
    - randomizing the origin per call

    Args:
      drift: variance in position difference per unit of difference
      affine: whether to randomize the origin every call
      transform: learnable transform after encoding (any/diag/none)
      time_bin_length: Add features AxialPositionalEncoding learns if
        TimeBinCausalAttention is the first layer.
        bin_length should match TBCA.bin_length
        If you set transform='diag', this flag increases your model capacity to
        close to transform='any', though it will still train slower.
      mode: if 'predict', allow evaluating one token at a time
    """
    super().__init__()
    if transform not in ('any', 'diag', 'none'):
      raise ValueError(transform)
    self._noise_rng = jax.random.split(jax.random.PRNGKey(234234535))[0]
    assert self._noise_rng is not None
    self._noise = None
    self._drift = drift
    self._affine = affine
    self._transform = transform
    self._time_bin_length = time_bin_length
    self._mode = mode

  def _get_noise(self, lo: int, hi: int, depth: int):
    """Return pseudorandom noise with shape float[length, depth].

    Args:
      lo: where to start sampling
      hi: where to stop sampling
      depth: noise depth

    Returns:
      noise[lo:hi, :]: the noise, where noise.diff(axis=0) is i.i.d. U(-1,1)
    """
    if self._noise is None or self._noise.shape[0] < hi:
      # Resize the noise:
      new_length = 1
      while new_length < hi:
        new_length *= 2
      noise = threefry_2x32_prange(self._noise_rng, 0, new_length * depth)
      noise = noise.reshape((new_length, depth, 2))[:, :, 0]
      # Normalize to [-sqrt(3), sqrt(3)]:
      noise = noise.astype(jnp.float32) / 2**31 - 1
      noise = noise * 3**.5
      # TODO(tying): use multiscale noise for memory-efficient sampling
      noise = noise.cumsum(axis=0)
      self._noise = noise
    assert self._noise.shape[0] >= hi
    assert self._noise.shape[1] == depth
    return self._noise[lo:hi, :]

  def _get_embeddings(self, lo: int, hi: int, depth, rng=None):
    """Get embeddings float[length, depth].

    Args:
      lo: where to start sampling
      hi: where to stop sampling
      depth: embedding depth
      rng: rng for random phase

    Returns:
      embeddings: float[length, depth]
    """
    noise = self._get_noise(lo, hi, (depth + 1) // 2)
    # Make the stddev around 1 after 1/drift.
    noise = noise * self._drift**.5

    t, c = np.mgrid[lo:hi, :depth]
    # Make even channels cos, odd channels sin:
    c_div_2, c_mod_2 = divmod(c, 2)
    # Off-by-one correction for odd depth:
    drift = self._drift
    if depth > 2:
      drift = drift**(((depth+1)//2)/(depth//2))
    # Spend roughly half the frequencies on noise:
    freq = jnp.geomspace(.5, .5 * drift**2, num=(depth + 1) // 2)[c_div_2]
    cycles = c_mod_2 / 4 + freq * t + noise[:, c_div_2[0, :]] / 4
    assert cycles.shape == (hi - lo, depth), cycles.shape

    # Get random phases:
    if self._affine:
      assert rng is not None
      cycles = cycles + trax.fastmath.random.uniform(
          rng, (1, depth,), minval=0, maxval=1)

    # Convert from cycles to radians:
    embeddings = jnp.cos(jnp.pi * 2 * cycles)

    # Set the last channels to the time bin features:
    if self._time_bin_length is not None:
      inter_bin_idx, intra_bin_idx = divmod(t[:, -1:], self._time_bin_length)
      bin_parity = inter_bin_idx % 2
      bin_fraction = intra_bin_idx / self._time_bin_length
      embeddings = jnp.concatenate(
          [
              embeddings[:, :-3],
              1 / (1 + inter_bin_idx),
              bin_fraction,
              bin_parity.astype(jnp.float32),
          ], -1)

    assert embeddings.shape == (hi - lo, depth), embeddings.shape
    return embeddings

  def forward(self, inputs):
    rng, state = self.rng, self.state
    d_feature = inputs.shape[-1]
    input_len = inputs.shape[-2]

    if self._mode == 'predict':
      # Assume all the positions are pretty close to each other.
      index, predict_rng = state
      lo = index.min()
      hi = index.max() + 1
      emb = self._get_embeddings(lo=lo, hi=hi, depth=d_feature, rng=predict_rng)
      emb = emb[index - lo, jnp.newaxis, :]
      index = index + 1
      state = index, predict_rng
    else:
      emb = self._get_embeddings(lo=0, hi=input_len, depth=d_feature, rng=rng)
      emb = emb[jnp.newaxis, :input_len, :]
    # TODO(tying): check that XLA swaps matmul(slice(x)) -> slice(matmul(x)),
    # or inline this code into get_embeddings/get_noise
    if self._transform == 'diag':
      emb = emb * jax.nn.softplus(self.weights)
    elif self._transform == 'any':
      emb = emb @ self.weights
    self.state = state
    return inputs + emb

  def init_weights_and_state(self, input_signature):
    d_feature = input_signature.shape[-1]
    if self._transform == 'diag':
      # Initialize it to a small value because JAX has a bug in softplus.
      scale_isoftplus = jnp.zeros((d_feature,), dtype=jnp.float32) + 1e-4
      weights = scale_isoftplus
    elif self._transform == 'any':
      ortho = trax.layers.initializers.OrthogonalInitializer()
      weights = ortho((d_feature, d_feature), self.rng)
    else:
      weights = layer_base.EMPTY_WEIGHTS
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      self.state = jnp.zeros((batch_size,), dtype=jnp.int32), self.rng
    self.weights = weights


class TimeBinPositionalEncoding(layer_base.Layer):
  """Just the engineered features from InfinitePositionalEncoding."""
  num_features = 3

  def __init__(self, time_bin_length, mode='train'):
    """Initializes the encoding.

    Args:
      time_bin_length: TimeBinCausalAttention.bin_length of the first layer.
      mode: if 'predict', allow evaluating one token at a time
    """
    super().__init__()
    self._time_bin_length = time_bin_length
    self._mode = mode

  def _get_embeddings(self, t):
    """Get embeddings float[..., num_features].

    Args:
      t: int[...] position (i.e. jnp.arange(..., jnp.int32))

    Returns:
      embeddings: float[..., num_features]
    """
    inter_bin_idx, intra_bin_idx = divmod(t, self._time_bin_length)
    bin_parity = inter_bin_idx % 2
    bin_fraction = intra_bin_idx / self._time_bin_length
    embeddings = jnp.stack([
        1 / (1 + inter_bin_idx),
        bin_fraction,
        bin_parity.astype(jnp.float32),
    ], -1)

    assert embeddings.shape == t.shape + (self.num_features,), embeddings.shape
    return embeddings

  def forward(self, inputs):
    state = self.state
    depth = inputs.shape[-1]

    if self._mode == 'predict':
      emb = self._get_embeddings(t=state)
      emb = emb[:, jnp.newaxis, :]
      state = state + 1
    else:
      input_len = inputs.shape[-2]
      emb = self._get_embeddings(t=jnp.arange(input_len, dtype=jnp.int32))
      # Leave batch axis as 1 for broadcasting:
      emb = emb[jnp.newaxis, :, :]
      emb = jnp.broadcast_to(emb, inputs.shape[:-1] + (3,))

    # Replace the last num_features channels of input.
    inputs = jnp.concatenate([inputs[..., :-self.num_features], emb], -1)
    if inputs.shape[-1] > depth:
      logging.warning(
          'dropping feature(s): %d down to %d', inputs.shape[-1], depth)
      inputs = inputs[..., -depth:]

    assert inputs.shape[-1] == depth, inputs.shape
    self.state = state
    return inputs

  def init_weights_and_state(self, input_signature):
    if self._mode == 'predict':
      batch_size = input_signature.shape[0]
      self.state = jnp.zeros((batch_size,), dtype=jnp.int32)
