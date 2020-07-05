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

"""Trax fast math: TF backend."""

import numpy as np
import tensorflow.compat.v2 as tf

from trax.shapes import ShapeDtype
from trax.tf_numpy import extensions as tf_np_extensions
from trax.tf_numpy import numpy as tf_np


def tf_abstract_eval(f):
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
  f_shape = tf_np_extensions.eval_on_shapes(f)
  def from_shape_type(x):
    if isinstance(x, ShapeDtype):
      return tf.TensorSpec(x.shape, x.dtype)
    else:
      return x
  def to_shape_type(x):  # pylint: disable=missing-docstring
    # TODO(wangpeng): handle partial output shapes using `tf.shape`.
    def to_numpy_shape(s):
      if s.is_fully_defined():
        return tuple(s.as_list())
      else:
        raise ValueError("The output shapes (%s) of the dry-run'ed function are"
                         ' not fully defined.' % s)
    def to_numpy_dtype(t):
      return np.dtype(t.as_numpy_dtype)
    if isinstance(x, tf.TensorSpec):
      return ShapeDtype(to_numpy_shape(x.shape), to_numpy_dtype(x.dtype))
    else:
      return x
  def f_return(*args):
    args = tf.nest.map_structure(from_shape_type, args)
    res = f_shape(*args)
    return tf.nest.map_structure(to_shape_type, res)
  return f_return


# The arguments order is different from tf_np_extensions.uniform
def tf_randint(key, shape, minval, maxval, dtype=np.int32):
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
  return tf_np_extensions.uniform(key, shape, minval=minval, maxval=maxval,
                                  dtype=dtype)


_tf_xla_forced_compile_enabled = False


def tf_xla_forced_compile_enabled():
  return _tf_xla_forced_compile_enabled


def set_tf_xla_forced_compile(b):
  global _tf_xla_forced_compile_enabled
  _tf_xla_forced_compile_enabled = b


def _tf_jit(*args, **kwargs):
  kwargs['xla_forced_compile'] = tf_xla_forced_compile_enabled()
  kwargs.pop('donate_argnums', None)  # donate_argnums not used in TF
  return tf_np_extensions.jit(*args, **kwargs)


def _tf_pmap(*args, **kwargs):
  kwargs.pop('donate_argnums', None)  # donate_argnums not used in TF
  return tf_np_extensions.pmap(*args, **kwargs)


TF_BACKEND = {
    'name': 'tf',
    'np': tf_np,
    'jit': _tf_jit,
    'stop_gradient': tf_np_extensions.stop_gradient,
    'grad': tf_np_extensions.grad,
    'vjp': tf_np_extensions.vjp,
    'custom_grad': tf_np_extensions.custom_grad,
    'abstract_eval': tf_abstract_eval,
    'expit': tf_np_extensions.expit,
    'erf': tf_np_extensions.erf,
    'logsumexp': tf_np_extensions.logsumexp,
    'conv': tf_np_extensions.conv,
    'lt': lambda x, y: x < y,
    'avg_pool': tf_np_extensions.avg_pool,
    'max_pool': tf_np_extensions.max_pool,
    'sort_key_val': tf_np_extensions.sort_key_val,
    'random_uniform': tf_np_extensions.uniform,
    'random_randint': tf_randint,
    'random_normal': tf_np_extensions.normal,
    'random_bernoulli': tf_np_extensions.bernoulli,
    'random_get_prng': tf_np_extensions.prng,
    'random_split': tf_np_extensions.split,
    'dataset_as_numpy': tf_np_extensions.dataset_as_numpy,
    'device_count': lambda: max(len(tf_np_extensions.accelerators()), 1),
    'pmap': _tf_pmap,
    'psum': tf_np_extensions.psum,
}
