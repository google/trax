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
"""Trax initializers."""

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
from trax.fastmath import numpy as jnp
from trax.fastmath import random


def _GetFans(shape, out_dim=-1, in_dim=-2, nonreceptive_dims=None):
  """Get the fan-in and fan-out sizes for the given shape and dims."""
  # Temporary fix until numpy.delete supports negative indices.
  if out_dim < 0:
    out_dim += len(shape)
  if in_dim < 0:
    in_dim += len(shape)

  if nonreceptive_dims is None:
    nonreceptive_dims = []
  if not isinstance(nonreceptive_dims, (list, tuple)):
    nonreceptive_dims = [nonreceptive_dims]

  receptive_field = jnp.prod(np.delete(shape, [in_dim, out_dim,
                                               *nonreceptive_dims]))
  if len(shape) >= 2:
    fan_in, fan_out = shape[in_dim], shape[out_dim]
  elif len(shape) == 1:
    fan_in = shape[0]
    fan_out = shape[0]
  else:
    fan_in = 1.
    fan_out = 1.
    fan_in *= receptive_field
    fan_out *= receptive_field
  return fan_in, fan_out


def InitializerFromFile(path):
  """Loads parameters from .npy file."""

  def Initializer(shape, rng):
    del rng
    logging.info('Loading pretrained embeddings from %s', path)
    with tf.io.gfile.GFile(path, 'rb') as f:
      parameters = jnp.load(f)
    assert jnp.shape(parameters) == shape, (
        'Expected shape %s, got %s' % (shape, jnp.shape(parameters)))
    return parameters

  return Initializer


def _PureShape(shape):
  """Make sure shape does not contain int tensors by calling int()."""
  return [int(x) for x in shape]


def RandomNormalInitializer(stddev=1e-2):
  """Returns an initializer for random normal coefficients."""
  return lambda shape, rng: (stddev * random.normal(  # pylint: disable=g-long-lambda
      rng, _PureShape(shape)).astype('float32'))


def RandomUniformInitializer(lim=1.0):
  """Returns an initializer for random uniform coefficients."""
  # Make sure shape does not contain int tensors by calling int() below.
  return lambda shape, rng: random.uniform(  # pylint: disable=g-long-lambda
      rng, _PureShape(shape), jnp.float32, -lim, lim)


def ScaledInitializer(out_dim, in_dim, scale, mode, distribution):
  """Returns an initializer that adjusts its scale based on weight shapes."""
  if scale <= 0.:
    raise ValueError('scale must be positive float, {} given'.format(scale))
  if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
    raise ValueError(
        'Invalid mode argument:, {}, must be either fan_in, fan_out or fan_avg'
        .format(mode))

  def Init(shape, rng, nonreceptive_dims=None):
    """Returns random values for initializing weights of the given `shape`."""
    shape = _PureShape(shape)
    fan_in, fan_out = _GetFans(shape, out_dim, in_dim, nonreceptive_dims)
    gain = scale
    if mode == 'fan_in':
      gain /= fan_in
    elif mode == 'fan_out':
      gain /= fan_out
    elif mode == 'fan_avg':
      gain /= (fan_in + fan_out) / 2
    if distribution == 'truncated_normal':
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = jnp.sqrt(gain) / .87962566103423978
      new_weights = random.truncated_normal(rng, -2, 2, shape) * stddev
      return new_weights.astype('float32')
    elif distribution == 'normal':
      new_weights = random.normal(rng, shape) * jnp.sqrt(gain)
      return new_weights.astype('float32')
    elif distribution == 'uniform':
      lim = jnp.sqrt(3. * gain)
      return random.uniform(rng, shape, jnp.float32, -lim, lim)
    else:
      raise ValueError('invalid distribution for ScaleInitializer')

  return Init


def GlorotNormalInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """Returns an initializer for random Glorot-scaled coefficients."""
  return ScaledInitializer(out_dim, in_dim, scale, 'fan_avg', 'normal')


def GlorotUniformInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """Returns an initializer for random uniform Glorot-scaled coefficients."""
  return ScaledInitializer(out_dim, in_dim, scale, 'fan_avg', 'uniform')


def LeCunNormalInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """Returns an initializer for random LeCun-scaled coefficients."""
  return ScaledInitializer(out_dim, in_dim, scale, 'fan_in', 'normal')


def LeCunUniformInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """Returns an initializer for random uniform LeCun-scaled coefficients."""
  return ScaledInitializer(out_dim, in_dim, scale, 'fan_in', 'uniform')


def KaimingNormalInitializer(out_dim=-1, in_dim=-2, param=0.):
  """Returns an initializer for random Kaiming-scaled coefficients."""
  return ScaledInitializer(
      out_dim, in_dim, 2.0 / jnp.sqrt(1 + param**2), 'fan_in', 'normal')


def KaimingUniformInitializer(out_dim=-1, in_dim=-2, param=0.):
  """Returns an initializer for random uniform Kaiming-scaled coefficients."""
  return ScaledInitializer(
      out_dim, in_dim, 2.0 / jnp.sqrt(1 + param**2), 'fan_in', 'uniform')


def OrthogonalInitializer(stddev=1.0):
  """Returns an orthogonal initializer."""
  def Init(shape, rng):
    """Returns orthogonalized random normal values with the given `shape`."""
    # Have at least 2 elements in shape.
    cur_shape = list(shape)
    while len(cur_shape) < 2:
      cur_shape = [1] + cur_shape

    # Flatten the input shape with the last dimension remaining.
    n_rows = 1
    for dim in cur_shape[:-1]:
      n_rows *= dim
    n_cols = cur_shape[-1]
    flat_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)

    # Generate a random matrix
    a = random.normal(rng, flat_shape, dtype=jnp.float32)

    # Compute the qr factorization
    q, r = jnp.linalg.qr(a)

    # Make Q uniform
    d = jnp.diag(r)
    q *= jnp.sign(d)

    # Transpose and reshape back q if needed.
    if n_rows < n_cols:
      q = jnp.transpose(q)
    q = jnp.reshape(q, shape)

    # Return scaled as requested.
    return stddev * q

  return Init


def AtariConvInit(kernel_shape, rng, dtype=jnp.float32):
  """The standard init for Conv laters and Atari."""
  filter_height, filter_width, fan_in, _ = kernel_shape
  std = 1 / jnp.sqrt(fan_in * filter_height * filter_width)
  return random.uniform(rng, kernel_shape, dtype, minval=-std, maxval=std)
