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

"""Utility functions for internal use."""

# TODO(wangpeng): Use tf_inspect once we move into TF.
import funcsigs
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import arrays
from trax.tf_numpy.numpy_impl import dtypes


tensor_to_ndarray = arrays.tensor_to_ndarray


def _canonicalize_axis(axis, rank):
  return _canonicalize_axes([axis], rank)[0]


def _canonicalize_axes(axes, rank):
  rank = _maybe_static(rank)

  if isinstance(rank, tf.Tensor):
    canonicalizer = (
        lambda axis: cond(axis < 0, lambda: axis + rank, lambda: axis))
  else:
    canonicalizer = lambda axis: axis+rank if axis < 0 else axis

  return [canonicalizer(axis) for axis in axes]


def _to_tf_type(dtype):
  """Converts a native python or numpy type to TF DType.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A tensorflow `DType`.
  """
  return tf.as_dtype(dtype)


def _to_numpy_type(dtype):
  """Converts a native python or TF DType to numpy type.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A NumPy `dtype`.
  """
  if isinstance(dtype, tf.DType):
    return dtype.as_numpy_dtype
  return np.dtype(dtype)


def finfo(dtype):
  """Returns properties of floating point types.

  Note that currently it just forwards to the numpy namesake, while tensorflow
  and numpy dtypes may have different properties.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A class describing properties of `dtype`, as described by
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html
  """
  return np.finfo(_to_numpy_type(dtype))


def isscalar(val):
  """Returns whether `val` is a scalar value or scalar Tensor."""
  if isinstance(val, (np.ndarray, arrays.ndarray, tf.Tensor)):
    return len(val.shape) == 0  # pylint: disable=g-explicit-length-test
  return np.isscalar(val)


# Can't use np_doc because np.result_type is a builtin function.
def result_type(*arrays_and_dtypes):
  """Returns the type resulting from applying NumPy type promotion to arguments.

  Args:
    *arrays_and_dtypes: A list of array_like objects or dtypes.

  Returns:
    A numpy dtype.
  """
  def maybe_get_dtype(x):
    # Don't put np.ndarray in this list, because np.result_type looks at the
    # value (not just dtype) of np.ndarray to decide the result type.
    if isinstance(x, (arrays.ndarray, arrays.ShardedNdArray,
                      tf.Tensor, tf.IndexedSlices)):
      return _to_numpy_type(x.dtype)
    elif isinstance(x, tf.DType):
      return _to_numpy_type(x)
    return x
  arrays_and_dtypes = [maybe_get_dtype(x) for x in
                       tf.nest.flatten(arrays_and_dtypes)]
  if not arrays_and_dtypes:
    # If arrays_and_dtypes is an empty list, let numpy decide what the dtype is.
    arrays_and_dtypes = [np.asarray([])]
  return dtypes._result_type(*arrays_and_dtypes)


def promote_types(type1, type2):
  """Returns the type resulting from applying NumPy type promotion.

  Args:
    type1: A numpy type.
    type2: A numpy type.

  Returns:
    A numpy type.
  """
  type1 = _to_numpy_type(type1)
  type2 = _to_numpy_type(type2)
  return dtypes.canonicalize_dtype(np.promote_types(type1, type2))


def _has_docstring(f):
  return hasattr(f, '__doc__') and isinstance(f.__doc__, str) and f.__doc__


def _add_blank_line(s):
  if s.endswith('\n'):
    return s + '\n'
  else:
    return s + '\n\n'


def _np_signature(f):
  """An enhanced funcsigs.signature that can handle numpy.ufunc."""
  if not isinstance(f, np.ufunc):
    return funcsigs.signature(f)
  def names_from_num(prefix, n):
    if n <= 0:
      return []
    elif n == 1:
      return [prefix]
    else:
      return [prefix + str(i + 1) for i in range(n)]
  input_names = names_from_num('x', f.nin)
  output_names = names_from_num('out', f.nout)
  keyword_only_params = [
      ('where', True),
      ('casting', 'same_kind'),
      ('order', 'K'),
      ('dtype', None),
      ('subok', True),
      ('signature', None),
      ('extobj', None)]
  params = []
  params += [funcsigs.Parameter(name, funcsigs.Parameter.POSITIONAL_ONLY)
             for name in input_names]
  if f.nout > 1:
    params += [funcsigs.Parameter(name, funcsigs.Parameter.POSITIONAL_ONLY,
                                  default=None)
               for name in output_names]
  params += [funcsigs.Parameter(
      'out', funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
      default=None if f.nout == 1 else (None,) * f.nout)]
  params += [funcsigs.Parameter(name, funcsigs.Parameter.KEYWORD_ONLY,
                                default=default)
             for name, default in keyword_only_params]
  return funcsigs.Signature(params)


# Python 2 doesn't allow keyword-only argument. Python prior to 3.8 doesn't
# allow positional-only argument. So we conflate positional-only, keyword-only
# and positional-or-keyword arguments here.
def _is_compatible_param_kind(a, b):
  def relax(k):
    if k in (funcsigs.Parameter.POSITIONAL_ONLY,
             funcsigs.Parameter.KEYWORD_ONLY):
      return funcsigs.Parameter.POSITIONAL_OR_KEYWORD
    return k
  return relax(a) == relax(b)


def np_doc(np_fun):
  """Attachs numpy docstring to a function.

  Args:
    np_fun: the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_sig = _np_signature(np_fun)
  def decorator(f):
    """The decorator."""
    sig = funcsigs.signature(f)
    unsupported_params = []
    for name in np_sig.parameters:
      if name not in sig.parameters:
        unsupported_params.append(name)
    f.__doc__ = _np_doc_helper(f, np_fun, unsupported_params)
    return f
  return decorator


def _np_doc_helper(f, np_f, unsupported_params=None):
  """Helper to get docs."""
  if not unsupported_params and not _has_docstring(f) and _has_docstring(np_f):
    return np_f.__doc__
  doc = 'TensorFlow variant of `numpy.%s`.\n\n' % np_f.__name__
  if unsupported_params:
    doc += 'Unsupported arguments: ' + ', '.join(
        '`' + name + '`' for name in unsupported_params) + '.\n\n'
  if _has_docstring(f):
    doc += f.__doc__
    doc = _add_blank_line(doc)
  if _has_docstring(np_f):
    doc += 'Documentation for `numpy.%s`:\n\n' % np_f.__name__
    doc += np_f.__doc__
  return doc


def np_doc_only(np_f):
  """Attachs numpy docstring to a function.

  This differs from np_doc in that it doesn't check for a match in signature.

  Args:
    np_f: the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_f` to the
    decorated function.
  """

  def decorator(f):
    f.__doc__ = _np_doc_helper(f, np_f)
    return f

  return decorator


def tf_broadcast(*args):
  """Broadcast tensors.

  Args:
    *args: a list of tensors whose shapes are broadcastable against each other.

  Returns:
    Tensors broadcasted to the common shape.
  """
  if len(args) <= 1:
    return args
  sh = tf.shape(args[0])
  for arg in args[1:]:
    sh = tf.broadcast_dynamic_shape(sh, tf.shape(arg))
  return [tf.broadcast_to(arg, sh) for arg in args]


# TODO(wangpeng): Move the following functions to a separate file and check for
#   float dtypes in each of them.


def get_static_value(x):
  """A version of tf.get_static_value that returns None on float dtypes.

  It returns None on float dtypes in order to avoid breaking gradients.

  Args:
    x: a tensor.

  Returns:
    Same as `tf.get_static_value`, except that it returns None when `x` has a
    float dtype.
  """
  if isinstance(x, tf.Tensor) and (x.dtype.is_floating or x.dtype.is_complex):
    return None
  return tf.get_static_value(x)


def _maybe_static(x):
  value = get_static_value(x)
  if value is None:
    return x
  else:
    return value


# All the following functions exist because get_static_value can't handle
# their TF counterparts.


def cond(pred, true_fn, false_fn):
  """A version of tf.cond that tries to evaluate the condition."""
  v = get_static_value(pred)
  if v is None:
    return tf.cond(pred, true_fn, false_fn)
  if v:
    return true_fn()
  else:
    return false_fn()


def add(a, b):
  """A version of tf.add that eagerly evaluates if possible."""
  return _maybe_static(a) + _maybe_static(b)


def subtract(a, b):
  """A version of tf.subtract that eagerly evaluates if possible."""
  return _maybe_static(a) - _maybe_static(b)


def greater(a, b):
  """A version of tf.greater that eagerly evaluates if possible."""
  return _maybe_static(a) > _maybe_static(b)


def greater_equal(a, b):
  """A version of tf.greater_equal that eagerly evaluates if possible."""
  return _maybe_static(a) >= _maybe_static(b)


def less_equal(a, b):
  """A version of tf.less_equal that eagerly evaluates if possible."""
  return _maybe_static(a) <= _maybe_static(b)


def logical_and(a, b):
  """A version of tf.logical_and that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return _maybe_static(b)
      else:
        return a_value
    else:
      return a_value & _maybe_static(b)
  else:
    return a & _maybe_static(b)


def logical_or(a, b):
  """A version of tf.logical_or that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return a_value
      else:
        return _maybe_static(b)
    else:
      return a_value | _maybe_static(b)
  else:
    return a | _maybe_static(b)


def getitem(a, slice_spec):
  """A version of __getitem__ that eagerly evaluates if possible."""
  return _maybe_static(a)[slice_spec]


def reduce_all(input_tensor, axis=None, keepdims=False):
  """A version of tf.reduce_all that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return tf.reduce_all(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.all(axis=axis, keepdims=keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False):
  """A version of tf.reduce_any that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return tf.reduce_any(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.any(axis=axis, keepdims=keepdims)
