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

"""Utility functions for internal use."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(wangpeng): Use tf_inspect once we move into TF.
import funcsigs
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import dtypes


tensor_to_ndarray = arrays.tensor_to_ndarray


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
    if isinstance(x, (np.ndarray, arrays.ndarray, arrays.ShardedNdArray,
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
    for name, param in sig.parameters.items():
      np_param = np_sig.parameters.get(name)
      if np_param is None:
        raise TypeError('Cannot find parameter "%s" in the numpy function\'s '
                        'signature' % name)
      if not _is_compatible_param_kind(param.kind, np_param.kind):
        raise TypeError('Parameter "%s" is of kind %s while in numpy it is of '
                        'kind %s' % (name, param.kind, np_param.kind))
      has_default = (param.default != funcsigs.Parameter.empty)
      np_has_default = (np_param.default != funcsigs.Parameter.empty)
      if has_default != np_has_default:
        raise TypeError('Parameter "%s" should%s have a default value' %
                        (name, '' if np_has_default else ' not'))
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


def cond(pred, true_fn, false_fn):
  """A version of tf.cond that tries to evaluate the condition."""
  v = tf.get_static_value(pred)
  if v is None:
    return tf.cond(pred, true_fn, false_fn)
  if v:
    return true_fn()
  else:
    return false_fn()


def _maybe_static(x):
  value = tf.get_static_value(x)
  if value is None:
    return x
  else:
    return value


def add(a, b):
  """A version of tf.add that eagerly evaluates if possible."""
  # It's needed becaues tf.get_static_value doesn't handle tf.add
  return _maybe_static(a) + _maybe_static(b)


def subtract(a, b):
  """A version of tf.subtract that eagerly evaluates if possible."""
  return _maybe_static(a) - _maybe_static(b)


def greater(a, b):
  """A version of tf.greater that eagerly evaluates if possible."""
  return _maybe_static(a) > _maybe_static(b)


def logical_or(a, b):
  """A version of tf.logical_or that eagerly evaluates if possible."""
  # Because TF overloads `|` as logical_or, we need to use `|` here. It's OK if
  # both `a` and `b` are evaluated, since `a | b` == `a or b` when a and b are
  # bools.
  return _maybe_static(a) | _maybe_static(b)


def getitem(a, slice_spec):
  """A version of __getitem__ that eagerly evaluates if possible."""
  return _maybe_static(a)[slice_spec]
