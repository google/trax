# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Mathematical operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import dtypes
from trax.tf_numpy.numpy import utils
from trax.tf_numpy.numpy.array_creation import promote_args_types


def dot(a, b):
  """The dot product of two arrays. See numpy.dot for more details.

  This relies on `tf.tensordot` which does not support types int64 and float64.
  So arrays of those types are "unsafely" cast to int32 and float32.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    b: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray.
  """
  a, b = promote_args_types(a, b)
  if utils.isscalar(a) or utils.isscalar(b):
    a = utils.tensor_to_ndarray(tf.expand_dims(a.data, -1))
    b = utils.tensor_to_ndarray(tf.expand_dims(b.data, -1))
    a_axis = b_axis = -1
  else:
    a_axis = -1
    # TODO(agarwal): handle ndim being None when in graph mode.
    b_axis = -2 if b.ndim > 1 else -1
  # TODO(srbs): When the shape of the output is a scalar e.g. when performing
  # a dot-product of two vectors, numpy returns a scalar object and not an
  # instance of ndarray.

  # tensordot/MatMul does not support int64 and float64 so we manually cast to
  # the compatible types. The conversion may be unsafe.
  # TODO(srbs): Figure out why MatMul does not support larger types.
  output_type = None
  if a.dtype == np.int64:
    logging.warning('Unsafe cast to int32.')
    a = utils.tensor_to_ndarray(tf.cast(a.data, tf.int32))
    b = utils.tensor_to_ndarray(tf.cast(b.data, tf.int32))
    output_type = tf.int64
  elif a.dtype == np.float64:
    logging.warning('Unsafe cast to float32.')
    a = utils.tensor_to_ndarray(tf.cast(a.data, tf.float32))
    b = utils.tensor_to_ndarray(tf.cast(b.data, tf.float32))
    output_type = tf.float64

  result_t = tf.tensordot(a.data, b.data, [[a_axis], [b_axis]])
  if output_type:
    result_t = tf.cast(result_t, output_type)
  return utils.tensor_to_ndarray(result_t)


# TODO(wangpeng): Make bitwise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = promote_args_types(a, b)
  else:
    a = array_creation.asarray(a)
    b = array_creation.asarray(b)
  return utils.tensor_to_ndarray(tf_fun(a.data, b.data))


@utils.np_doc(np.add)
def add(x1, x2):
  def add_or_or(x1, x2):
    if x1.dtype == tf.bool:
      return tf.logical_or(x1, x2)
    return tf.add(x1, x2)
  return _bin_op(add_or_or, x1, x2)


@utils.np_doc(np.subtract)
def subtract(x1, x2):
  return _bin_op(tf.subtract, x1, x2)


@utils.np_doc(np.multiply)
def multiply(x1, x2):
  def mul_or_and(x1, x2):
    if x1.dtype == tf.bool:
      return tf.logical_and(x1, x2)
    return tf.multiply(x1, x2)
  return _bin_op(mul_or_and, x1, x2)


@utils.np_doc(np.maximum)
def maximum(x1, x2):
  def max_or_or(x1, x2):
    if x1.dtype == tf.bool:
      return tf.logical_or(x1, x2)
    return tf.math.maximum(x1, x2)
  return _bin_op(max_or_or, x1, x2)


@utils.np_doc(np.minimum)
def minimum(x1, x2):
  def min_or_and(x1, x2):
    if x1.dtype == tf.bool:
      return tf.logical_and(x1, x2)
    return tf.math.minimum(x1, x2)
  return _bin_op(min_or_and, x1, x2)


# TODO(wangpeng): support broadcasting and 1-D case
@utils.np_doc(np.matmul)
def matmul(x1, x2):
  return _bin_op(tf.matmul, x1, x2)


@utils.np_doc(np.power)
def power(x1, x2):
  return _bin_op(tf.math.pow, x1, x2)


@utils.np_doc(np.float_power)
def float_power(x1, x2):
  return power(x1, x2)


@utils.np_doc(np.arctan2)
def arctan2(x1, x2):
  return _bin_op(tf.math.atan2, x1, x2)


def _scalar(tf_fn, x, promote_to_float=False):
  """Computes the tf_fn(x) for each element in `x`.

  Args:
    tf_fn: function that takes a single Tensor argument.
    x: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    promote_to_float: whether to cast the argument to a float dtype
      (`dtypes.default_float_type`) if it is not already.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `dtypes.default_float_type`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
  x = array_creation.asarray(x)
  if promote_to_float and x.dtype not in (np.float16, np.float32, np.float64):
    x = x.astype(dtypes.default_float_type())
  return utils.tensor_to_ndarray(tf_fn(x.data))


@utils.np_doc(np.log)
def log(x):
  return _scalar(tf.math.log, x, True)


@utils.np_doc(np.exp)
def exp(x):
  return _scalar(tf.exp, x, True)


@utils.np_doc(np.sqrt)
def sqrt(x):
  return _scalar(tf.sqrt, x, True)


@utils.np_doc(np.abs)
def abs(x):
  return _scalar(tf.math.abs, x)


@utils.np_doc(np.absolute)
def absolute(x):
  return abs(x)


@utils.np_doc(np.fabs)
def fabs(x):
  return abs(x)


@utils.np_doc(np.ceil)
def ceil(x):
  return _scalar(tf.math.ceil, x, True)


@utils.np_doc(np.floor)
def floor(x):
  return _scalar(tf.math.floor, x, True)


@utils.np_doc(np.conj)
def conj(x):
  return _scalar(tf.math.conj, x)


@utils.np_doc(np.negative)
def negative(x):
  return _scalar(tf.math.negative, x)


@utils.np_doc(np.reciprocal)
def reciprocal(x):
  return _scalar(tf.math.reciprocal, x)


@utils.np_doc(np.signbit)
def signbit(x):
  def f(x):
    if x.dtype == tf.bool:
      return tf.fill(x.shape, False)
    return x < 0
  return _scalar(f, x)


@utils.np_doc(np.sin)
def sin(x):
  return _scalar(tf.math.sin, x, True)


@utils.np_doc(np.cos)
def cos(x):
  return _scalar(tf.math.cos, x, True)


@utils.np_doc(np.tan)
def tan(x):
  return _scalar(tf.math.tan, x, True)


@utils.np_doc(np.sinh)
def sinh(x):
  return _scalar(tf.math.sinh, x, True)


@utils.np_doc(np.cosh)
def cosh(x):
  return _scalar(tf.math.cosh, x, True)


@utils.np_doc(np.tanh)
def tanh(x):
  return _scalar(tf.math.tanh, x, True)


@utils.np_doc(np.arcsin)
def arcsin(x):
  return _scalar(tf.math.asin, x, True)


@utils.np_doc(np.arccos)
def arccos(x):
  return _scalar(tf.math.acos, x, True)


@utils.np_doc(np.arctan)
def arctan(x):
  return _scalar(tf.math.atan, x, True)


@utils.np_doc(np.arcsinh)
def arcsinh(x):
  return _scalar(tf.math.asinh, x, True)


@utils.np_doc(np.arccosh)
def arccosh(x):
  return _scalar(tf.math.acosh, x, True)


@utils.np_doc(np.arctanh)
def arctanh(x):
  return _scalar(tf.math.atanh, x, True)


@utils.np_doc(np.sum)
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Computes sum of all array elements or along specified axes.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute sum.
      If None, returns sum of all elements in array.
    dtype: Optional. The type of the output array. If None, defaults to the
      dtype of `a` unless `a` is an integer type with precision less than `int`
      in which case the output type is `int.`
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray.
  """
  # TODO(wangpeng): check that we fully match numpy behavior.
  a = array_creation.asarray(a, dtype=dtype)
  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np.promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = array_creation.asarray(a, dtype=output_type)

  return utils.tensor_to_ndarray(tf.reduce_sum(input_tensor=a.data, axis=axis,
                                               keepdims=keepdims))


@utils.np_doc(np.max)
def max(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Computes the max of all array elements or along specified axes.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute the max.
      If None, returns the max of all elements in array.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray with the same dtype as `a`.
  """
  # TODO(wangpeng): check that we fully match numpy behavior.
  a = array_creation.asarray(a)

  return utils.tensor_to_ndarray(tf.reduce_max(input_tensor=a.data, axis=axis,
                                               keepdims=keepdims))
