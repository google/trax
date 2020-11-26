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

"""Trax fast math: JAX backend."""

import functools
import jax
from jax import lax
from jax import random as jax_random
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from trax.fastmath import numpy as tnp
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
  return rescale(out, inputs) if rescale else out  # pylint: disable=not-callable


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
    return tnp.nested_map(signature, jax_shapes)
  return shape_fun


# The default value of dtype is different from jax_random.randint
def jax_randint(key, shape, minval, maxval, dtype=np.int32):
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


def _to_numpy(x):
  """Converts non-NumPy tensors to NumPy arrays."""
  return x if isinstance(x, np.ndarray) else x.numpy()


def _dataset_as_numpy(ds, batch_size=None):
  """Speed up tfds.as_numpy by batching and then iterating over the batches."""
  batch_size = batch_size or 1
  try:  # Check that dense_to_ragged_batch exists.
    if batch_size < 2:  # Fall back to default if no batching requested.
      raise AttributeError
    ds_batch = ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    for example in tfds.as_numpy(ds_batch):
      flat_example = tnp.tree_flatten(example)
      np_flat_example = [_to_numpy(x) for x in flat_example]
      for single_example_flat in zip(*np_flat_example):
        single_example, _ = tnp.tree_unflatten(single_example_flat, example)
        yield single_example
  except AttributeError:
    # In TF 1.X there is not dense_to_ragged_batch: fallback.
    for example in tfds.as_numpy(ds):
      yield example


def _custom_grad(f_vjp, f_original):
  f_ = jax.custom_transforms(f_original)
  jax.defvjp_all(f_, f_vjp)
  return f_


def _custom_vjp(f, f_fwd, f_bwd, nondiff_argnums=()):
  @functools.partial(jax.custom_vjp, nondiff_argnums=nondiff_argnums)
  def _f(*args, **kwargs):
    return f(*args, **kwargs)
  _f.defvjp(f_fwd, f_bwd)
  return _f


JAX_BACKEND = {
    'name': 'jax',
    'np': jnp,
    'abstract_eval': jax_abstract_eval,
    'avg_pool': jax_avg_pool,
    'cond': lax.cond,
    'conv': jax_conv,
    'custom_vjp': _custom_vjp,
    'custom_grad': _custom_grad,
    'dataset_as_numpy': _dataset_as_numpy,
    'device_count': jax.local_device_count,
    'dynamic_slice': jax.lax.dynamic_slice,
    'dynamic_slice_in_dim': jax.lax.dynamic_slice_in_dim,
    'dynamic_update_slice': jax.lax.dynamic_update_slice,
    'dynamic_update_slice_in_dim': jax.lax.dynamic_update_slice_in_dim,
    'erf': jax_special.erf,
    'expit': jax_special.expit,
    'fori_loop': lax.fori_loop,
    'grad': jax.grad,
    'value_and_grad': jax.value_and_grad,
    'index_add': jax.ops.index_add,
    'index_max': jax.ops.index_max,
    'index_min': jax.ops.index_min,
    'index_update': jax.ops.index_update,
    'jit': jax.jit,
    'logsumexp': jax_special.logsumexp,
    'lt': lax.lt,
    'max_pool': jax_max_pool,
    'pmap': jax.pmap,
    'psum': lax.psum,
    'random_bernoulli': jax_random.bernoulli,
    'random_get_prng': jax.jit(jax_random.PRNGKey),
    'random_normal': jax_random.normal,
    'random_randint': jax_randint,
    'random_split': jax_random.split,
    'random_fold_in': jax_random.fold_in,
    'random_uniform': jax_random.uniform,
    'remat': jax.remat,
    'scan': lax.scan,
    'sort_key_val': jax.lax.sort_key_val,
    'stop_gradient': lax.stop_gradient,
    'sum_pool': jax_sum_pool,
    'vjp': jax.vjp,
    'vmap': jax.vmap,
}
