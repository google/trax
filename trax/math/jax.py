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

"""Trax math: JAX backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import lax
from jax import random as jax_random
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as onp
import tensorflow_datasets as tfds

from trax.shapes import signature


def jax_conv(inp, fltr, window_strides, padding, dimension_numbers,
             filter_dilation=None):
  """A wrapper around `lax.conv_general_dilated`.

  It requires `dimension_numbers` and disallows `inp_dilation`.

  Args:
    inp: an (N+2)-D array. The input of the convolution.
    fltr: an (N+2)-D array. The filter (i.e. kernel) of the convolution.
    window_strides: the strides for moving the convolution window.
    padding: a string, either 'VALID' or 'SAME'. The padding algorithm.
    dimension_numbers: a tuple of three strings encoding the data format of
      input, filter and output. 'I' means input; 'O' means output; 'C' means
      channel; other characters such as 'W', 'H' and 'D' means spatial
      dimensions.
    filter_dilation: the dilation rates for the filter. Dilating the filter
      means adding "holes" to the filter.

  Returns:
    An (N+2)-D array. The convolution result.
  """
  return lax.conv_general_dilated(inp, fltr, window_strides, padding,
                                  lhs_dilation=None,
                                  rhs_dilation=filter_dilation,
                                  dimension_numbers=dimension_numbers)


def _pooling_general(inputs, reducer, init_val, rescaler=None,
                     pool_size=(2, 2), strides=None, padding='VALID'):
  """Helper: general pooling computation used in pooling layers later."""
  spatial_strides = strides or (1,) * len(pool_size)
  rescale = rescaler(pool_size, spatial_strides, padding) if rescaler else None
  dims = (1,) + pool_size + (1,)  # NHWC
  strides = (1,) + spatial_strides + (1,)
  out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
  return rescale(out, inputs) if rescale else out


def jax_max_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.max, -jnp.inf, pool_size=pool_size,
                          strides=strides, padding=padding)


def jax_sum_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.add, 0., pool_size=pool_size,
                          strides=strides, padding=padding)


def _normalize_by_window_size(dims, spatial_strides, padding):  # pylint: disable=invalid-name
  def rescale(outputs, inputs):
    one = jnp.ones(inputs.shape[1:-1], dtype=inputs.dtype)
    window_sizes = lax.reduce_window(
        one, 0., lax.add, dims, spatial_strides, padding)
    return outputs / window_sizes[..., jnp.newaxis]
  return rescale


def jax_avg_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.add, 0., _normalize_by_window_size,
                          pool_size, strides=strides, padding=padding)


def _jax_scan(f, xs, init_value, axis=0, remat=False):
  """Scans the f over the given axis of xs.

  In pseudo-python, the scan function would look as follows:

  def scan(f, xs, init_value, axis):
    xs  = [xs[..., i, ...] for i in range(xs.shape[axis])]
    cur_value = init_value
    ys = []
    for x in xs:
      y, cur_value = f(x, cur_value)
      ys.append(y)
    return np.stack(ys, axis), cur_value

  Args:
    f: function (x, carry) -> (y, new_carry)
    xs: tensor, x will be xs slices on axis
    init_value: tensor, initial value of the carry-over
    axis: int, the axis on which to slice xs
    remat: whether to re-materialize f

  Returns:
    A pair (ys, last_value) as described above.
  """
  def swapaxes(x):
    transposed_axes = list(range(len(x.shape)))
    transposed_axes[axis] = 0
    transposed_axes[0] = axis
    return jnp.transpose(x, axes=transposed_axes)
  if axis != 0:
    xs = nested_map(swapaxes, xs)
  def transposed_f(c, x):
    y, d = f(x, c)
    return d, y
  if remat:
    last_value, ys = lax.scan(jax.remat(transposed_f), init_value, xs)
  else:
    last_value, ys = lax.scan(transposed_f, init_value, xs)
  if axis != 0:
    ys = nested_map(swapaxes, ys)
  return ys, last_value


def nested_map(f, obj):
  """Maps `f` recursively inside any dicts/lists/tuples in `obj`.

  Args:
    f: A function taking a single object as input. f's input must NOT be a
        dict, list, or tuple, or any subclass of those.
    obj: Either an input object to f or some nested structure of collections
        of (collections of ...) input objects to f.

  Returns:
    An object with the same nested structure as `obj`, but with each input
    object `x` replaced by `f(x)`.
  """
  if isinstance(obj, list):
    return [nested_map(f, y) for y in obj]
  if isinstance(obj, tuple):
    return tuple([nested_map(f, y) for y in obj])
  if isinstance(obj, dict):
    return {k: nested_map(f, v) for (k, v) in obj.items()}
  return f(obj)


def jax_abstract_eval(f):
  """Returns a function that evaluates `f` given input shapes and dtypes.

  It transforms function `f` to a function that performs the same computation as
  `f` but only on shapes and dtypes (a.k.a. shape inference).

  Args:
    f: the function to be transformed.

  Returns:
    A function whose input arguments can be either the same as `f`'s or only
    their shapes/dtypes represented by `ShapeDtype`, and whose return values are
    `ShapeDtype`s with the same nested structure as `f`'s return values.
  """
  def shape_fun(*args, **kwargs):
    jax_shapes = jax.eval_shape(f, *args, **kwargs)
    return nested_map(signature, jax_shapes)
  return shape_fun


# The default value of dtype is different from jax_random.randint
def jax_randint(key, shape, minval, maxval, dtype=onp.int32):
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: int or array of ints broadcast-compatible with ``shape``, a minimum
      (inclusive) value for the range.
    maxval: int or array of ints broadcast-compatible with  ``shape``, a maximum
      (exclusive) value for the range.
    dtype: optional, an int dtype for the returned values (default int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  return jax_random.randint(key, shape, minval=minval, maxval=maxval,
                            dtype=dtype)


JAX_BACKEND = {
    'name': 'jax',
    'np': jnp,
    'logsumexp': jax_special.logsumexp,
    'expit': jax_special.expit,
    'erf': jax_special.erf,
    'conv': jax_conv,
    'avg_pool': jax_avg_pool,
    'max_pool': jax_max_pool,
    'sum_pool': jax_sum_pool,
    'scan': _jax_scan,
    'cond': lax.cond,
    'lt': lax.lt,
    'stop_gradient': lax.stop_gradient,
    'jit': jax.jit,
    'grad': jax.grad,
    'pmap': jax.pmap,
    'psum': lax.psum,
    'abstract_eval': jax_abstract_eval,
    'random_uniform': jax_random.uniform,
    'random_randint': jax_randint,
    'random_normal': jax_random.normal,
    'random_bernoulli': jax_random.bernoulli,
    'random_get_prng': jax.jit(jax_random.PRNGKey),
    'random_split': jax_random.split,
    'dataset_as_numpy': tfds.as_numpy,
    'device_count': jax.local_device_count,
}
