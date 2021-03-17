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

"""Trax fast math: pure numpy backend."""

import numpy as np
from scipy.special import logsumexp
from trax.shapes import signature


def get_prng(seed):
  """JAX-compatible way of getting PRNG seeds."""
  if np.shape(seed):
    raise TypeError('PRNGKey seed must be a scalar.')
  convert = lambda k: np.reshape(np.asarray(k, np.uint32), [1])
  k1 = convert(np.bitwise_and(np.right_shift(seed, 32), 0xFFFFFFFF))
  k2 = convert(np.bitwise_and(seed, 0xFFFFFFFF))
  return np.concatenate([k1, k2], 0)


def random_uniform(rng, shape=(), dtype=np.float64, minval=0.0, maxval=1.0):
  del rng
  return np.random.uniform(minval, maxval, size=shape).astype(dtype)


def random_normal(rng, shape=(), dtype=np.float64):
  del rng
  return np.random.normal(size=shape).astype(dtype)


def random_randint(rng, shape, minval, maxval, dtype=np.int64):
  del rng
  return np.random.randint(minval, maxval, size=shape).astype(dtype)


def random_bernoulli(rng, p=0.5, shape=()):
  del rng
  return np.random.binomial(1, p, size=shape)


def np_abstract_eval(f):
  """Abstract evaluation in numpy by running the real function on 0s."""
  def abstract_f(*args, **kwargs):
    real_args = [nested_map(lambda x: np.zeros(x.shape, x.dtype), a)
                 for a in args]
    real_res = f(*real_args, **kwargs)
    return signature(real_res)
  return abstract_f


NUMPY_BACKEND = {
    'abstract_eval': np_abstract_eval,
    'local_device_count': lambda: 1,
    'global_device_count': lambda: 1,
    'jit': lambda f: f,
    'logsumexp': logsumexp,
    'name': 'numpy',
    'np': np,
    'random_bernoulli': random_bernoulli,
    'random_get_prng': get_prng,
    'random_normal': random_normal,
    'random_randint': random_randint,
    'random_split': lambda prng, num=2: (None,) * num,
    'random_uniform': random_uniform,
    'expit': lambda x: 1. / (1. + np.exp(-x)),
}


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
            for k in objs[0]}
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
    return {k: nested_zip([obj[k] for obj in objs]) for k in objs[0]}

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
