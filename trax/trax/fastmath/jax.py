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

import jax
from jax import lax
from jax import random as jax_random
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as np
import tensorflow as tf
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


def _is_namedtuple_instance(x):
  """Checks if `x` is an instance of a `namedtuple` type."""
  if not isinstance(x, tuple):
    return False
  return hasattr(x, '_fields')


def _is_at_level(obj, level):
  """Checks if `obj` is an at level `level`."""
  is_leaf = not isinstance(obj, (list, tuple, dict))
  if level == 0 or is_leaf:
    return (level == 0) == is_leaf

  if isinstance(obj, dict):
    elems = obj.values()
  else:
    elems = obj
  return elems and all(_is_at_level(x, level - 1) for x in elems)


def _is_made_of_nones(obj):
  """Checks if `obj` is a nested structure of `None`s."""
  elems = tree_flatten(obj)
  # Returning False for an empty list, because it doesn't have any Nones inside.
  return elems and all(x is None for x in elems)


def nested_map(f, obj, level=0, ignore_nones=True):
  """Maps `f` recursively inside any dicts/lists/tuples in `obj`.

  Args:
    f: A function taking a single object as input. f's input must NOT be a
        dict, list, or tuple, or any subclass of those.
    obj: Either an input object to f or some nested structure of collections
        of (collections of ...) input objects to f.
    level: Level in the nested structure to stop at, counted from the leaves -
        so level 0 is the leaf, level 1 is such that all of its children are at
        level 0 etc.
    ignore_nones: Whether to ignore Nones in the structure, i.e. return None
        without calling `f`.

  Returns:
    An object with the same nested structure as `obj`, but with each input
    object `x` replaced by `f(x)`.
  """
  if _is_at_level(obj, level):
    if ignore_nones and _is_made_of_nones(obj):
      return None
    else:
      return f(obj)

  if _is_namedtuple_instance(obj):
    return type(obj)(*nested_map(f, list(obj), level=level))
  if isinstance(obj, list):
    return [nested_map(f, y, level=level) for y in obj]
  if isinstance(obj, tuple):
    return tuple([nested_map(f, y, level=level) for y in obj])
  if isinstance(obj, dict):
    return {k: nested_map(f, v, level=level) for (k, v) in obj.items()}

  raise ValueError('Non-exhaustive pattern match for {}.'.format(obj))


def nested_map_multiarg(f, *objs, ignore_nones=True):
  """Maps multi-arg `f` recursively inside any dicts/lists/tuples in `objs`.

  Args:
    f: A function taking len(objs) inputs. f's input must NOT be a
        dict, list, or tuple, or any subclass of those.
    *objs: Either input objects to f or some nested structure of collections
        of (collections of ...) input objects to f.
    ignore_nones: Whether to ignore Nones in the structure, i.e. return None
        without calling `f`.

  Returns:
    An object with the same nested structure as `objs[0]`, but with each input
    object `x` replaced by `f(*xs)`.
  """
  if isinstance(objs[0], list):
    return [nested_map_multiarg(f, *[o[i] for o in objs])
            for i in range(len(objs[0]))]
  if isinstance(objs[0], tuple):
    return tuple([nested_map_multiarg(f, *[o[i] for o in objs])
                  for i in range(len(objs[0]))])
  if isinstance(objs[0], dict):
    return {k: nested_map_multiarg(f, *[o[k] for o in objs])
            for k in objs[0].keys()}
  if ignore_nones and _is_made_of_nones(objs):
    return None
  return f(*objs)


def nested_zip(objs):
  """Zips the leaves of each nested structure in `objs`.

  Args:
    objs: List of nested structures to zip.

  Returns:
    An object with the same nested structure as each element of `objs`, with
    leaves zipped together into tuples.
  """
  assert isinstance(objs, (list, tuple))
  assert objs, 'Cannot zip an empty sequence.'

  if _is_at_level(objs, 1):
    return tuple(objs)

  if _is_namedtuple_instance(objs[0]):
    return type(objs[0])(*nested_zip(list(map(list, objs))))
  if isinstance(objs[0], list):
    return [nested_zip([obj[i] for obj in objs]) for i in range(len(objs[0]))]
  if isinstance(objs[0], tuple):
    return nested_zip(list(map(list, objs)))
  if isinstance(objs[0], dict):
    return {k: nested_zip([obj[k] for obj in objs]) for k in objs[0].keys()}

  raise ValueError('Non-exhaustive pattern match for {}.'.format(objs[0]))


def nested_stack(objs, axis=0, np_module=np):
  """Stacks the numpy arrays inside any dicts/lists/tuples in `objs`.

  Args:
    objs: List of nested structures to stack.
    axis: Axis to stack along.
    np_module: numpy module to use - typically numpy or jax.numpy.

  Returns:
    An object with the same nested structure as each element of `objs`, with
    leaves stacked together into numpy arrays. Nones are propagated, i.e. if
    each element of the stacked sequence is None, the output will be None.
  """
  # nested_map the stacking operation, but stopping at level 1 so at tuples of
  # numpy arrays.
  return nested_map(
      lambda x: np_module.stack(x, axis=axis),
      nested_zip(objs),
      level=1,
  )


def tree_flatten(tree):
  """Flatten a tree into a list."""
  if isinstance(tree, (list, tuple)):
    # In python, sum of lists starting from [] is the concatenation.
    return sum([tree_flatten(t) for t in tree], [])
  if isinstance(tree, dict):
    # Only use the values in case of a dictionary node.
    return sum([tree_flatten(v) for v in tree.values()], [])
  return [tree]


def tree_leaves(tree, ignore_nones=True):
  """Gets the leaves of a tree."""

  # Right now this is just `tree_flatten`, but we keep this separate since
  # JAX's tree_flatten returns the structure of the tree as well.
  flattened = tree_flatten(tree)
  return [flat for flat in flattened if (not ignore_nones) or flat is not None]


def tree_unflatten(flat, tree, copy_from_tree=None):
  """Unflatten a list into a tree given the tree shape as second argument.

  Args:
    flat: a flat list of elements to be assembled into a tree.
    tree: a tree with the structure we want to have in the new tree.
    copy_from_tree: optional list of elements that we just copy from tree.
      This argument is used when the flat version does not contain all elements
      of the expected tree but just a subset, while the rest are filled from
      the tree itself. It allows to omit "unnecessary" elements. For example,
      consider trees (A, (B, X), X) and (X, (A, X), B) where X is some element
      we do not care about. Flattening the first tree and removing X will yield
      a flat list [A, B] and the second tree can then be reconstructed from this
      list and the tree (X, (E, X), E) with copy_from_tree=[X]. One example
      where this is used is the weights-tree of a model, where layers with no
      weights have () in the tree and we use copy_from_tree=[()] to restore
      a model from a file that only has a list of trainable weights.

  Returns:
    A pair (new_tree, rest_of_flat) where the new tree that has the structure
    of tree but with leaves from flat, and the remaining elements of flat if
    more were provided than the number of leaves of tree (useful for recursion).
  """
  if copy_from_tree is not None and tree in copy_from_tree:
    return tree, flat
  if isinstance(tree, (list, tuple)):
    new_tree, rest = [], flat
    for t in tree:
      new_t, rest = tree_unflatten(rest, t, copy_from_tree=copy_from_tree)
      new_tree.append(new_t)
    new_tree = tuple(new_tree) if isinstance(tree, tuple) else new_tree
    return new_tree, rest
  if isinstance(tree, dict):
    new_tree, rest = {}, flat
    for k in tree:
      new_v, rest = tree_unflatten(rest, tree[k], copy_from_tree=copy_from_tree)
      new_tree[k] = new_v
    return new_tree, rest
  return flat[0], flat[1:]


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
      flat_example = tree_flatten(example)
      np_flat_example = [_to_numpy(x) for x in flat_example]
      for single_example_flat in zip(*np_flat_example):
        single_example, _ = tree_unflatten(single_example_flat, example)
        yield single_example
  except AttributeError:
    # In TF 1.X there is not dense_to_ragged_batch: fallback.
    for example in tfds.as_numpy(ds):
      yield example


def _custom_grad(f_vjp, f_original):
  f_ = jax.custom_transforms(f_original)
  jax.defvjp_all(f_, f_vjp)
  return f_


JAX_BACKEND = {
    'name': 'jax',
    'np': jnp,
    'abstract_eval': jax_abstract_eval,
    'avg_pool': jax_avg_pool,
    'cond': lax.cond,
    'conv': jax_conv,
    'custom_grad': _custom_grad,
    'dataset_as_numpy': _dataset_as_numpy,
    'device_count': jax.local_device_count,
    'erf': jax_special.erf,
    'expit': jax_special.expit,
    'grad': jax.grad,
    'value_and_grad': jax.value_and_grad,
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
    'random_uniform': jax_random.uniform,
    'scan': _jax_scan,
    'sort_key_val': jax.lax.sort_key_val,
    'stop_gradient': lax.stop_gradient,
    'sum_pool': jax_sum_pool,
    'vjp': jax.vjp,
}
