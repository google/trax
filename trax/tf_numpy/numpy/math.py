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

"""Mathematical operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import array_methods
from trax.tf_numpy.numpy import dtypes
from trax.tf_numpy.numpy import utils


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
  a, b = array_creation.promote_args_types(a, b)
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
    logging.warning("Unsafe cast to int32.")
    a = utils.tensor_to_ndarray(tf.cast(a.data, tf.int32))
    b = utils.tensor_to_ndarray(tf.cast(b.data, tf.int32))
    output_type = tf.int64
  elif a.dtype == np.float64:
    logging.warning("Unsafe cast to float32.")
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
    a, b = array_creation.promote_args_types(a, b)
  else:
    a = array_creation.asarray(a)
    b = array_creation.asarray(b)
  return utils.tensor_to_ndarray(tf_fun(a.data, b.data))


@utils.np_doc(np.add)
def add(x1, x2):
  def add_or_or(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
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
      assert x2.dtype == tf.bool
      return tf.logical_and(x1, x2)
    return tf.multiply(x1, x2)
  return _bin_op(mul_or_and, x1, x2)


@utils.np_doc(np.true_divide)
def true_divide(x1, x2):
  def f(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
      float_ = dtypes.default_float_type()
      x1 = tf.cast(x1, float_)
      x2 = tf.cast(x2, float_)
    return tf.math.truediv(x1, x2)
  return _bin_op(f, x1, x2)


divide = true_divide


@utils.np_doc(np.floor_divide)
def floor_divide(x1, x2):
  def f(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
      x1 = tf.cast(x1, tf.int8)
      x2 = tf.cast(x2, tf.int8)
    return tf.math.floordiv(x1, x2)
  return _bin_op(f, x1, x2)


@utils.np_doc(np.mod)
def mod(x1, x2):
  def f(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
      x1 = tf.cast(x1, tf.int8)
      x2 = tf.cast(x2, tf.int8)
    return tf.math.mod(x1, x2)
  return _bin_op(f, x1, x2)


remainder = mod


@utils.np_doc(np.divmod)
def divmod(x1, x2):
  return floor_divide(x1, x2), mod(x1, x2)


@utils.np_doc(np.maximum)
def maximum(x1, x2):
  def max_or_or(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
      return tf.logical_or(x1, x2)
    return tf.math.maximum(x1, x2)
  return _bin_op(max_or_or, x1, x2)


@utils.np_doc(np.minimum)
def minimum(x1, x2):
  def min_or_and(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
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


@utils.np_doc(np.nextafter)
def nextafter(x1, x2):
  return _bin_op(tf.math.nextafter, x1, x2)


@utils.np_doc(np.heaviside)
def heaviside(x1, x2):
  def f(x1, x2):
    return tf.where(x1 < 0, tf.constant(0, dtype=x2.dtype),
                    tf.where(x1 > 0, tf.constant(1, dtype=x2.dtype), x2))
  return _bin_op(f, x1, x2)


@utils.np_doc(np.hypot)
def hypot(x1, x2):
  return sqrt(square(x1) + square(x2))


def _pad_left_to(n, old_shape):
  return (1,) * (n - len(old_shape)) + old_shape


@utils.np_doc(np.kron)
def kron(a, b):
  a, b = array_creation.promote_args_types(a, b)
  ndim = __builtins__["max"](a.ndim, b.ndim)
  if a.ndim < ndim:
    a = array_methods.reshape(a, _pad_left_to(ndim, a.shape))
  if b.ndim < ndim:
    b = array_methods.reshape(b, _pad_left_to(ndim, b.shape))
  a_reshaped = array_methods.reshape(a, [i for d in a.shape for i in (d, 1)])
  b_reshaped = array_methods.reshape(b, [i for d in b.shape for i in (1, d)])
  out_shape = tuple(np.multiply(a.shape, b.shape))
  return array_methods.reshape(a_reshaped * b_reshaped, out_shape)


@utils.np_doc(np.outer)
def outer(a, b):
  def f(a, b):
    return tf.reshape(a, [-1, 1]) * tf.reshape(b, [-1])
  return _bin_op(f, a, b)


# This can also be implemented via tf.reduce_logsumexp
@utils.np_doc(np.logaddexp)
def logaddexp(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return array_methods.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))))


@utils.np_doc(np.logaddexp2)
def logaddexp2(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return array_methods.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp2(-abs(delta))) / np.log(2))


@utils.np_doc(np.polyval)
def polyval(p, x):
  def f(p, x):
    if p.shape.rank == 0:
      p = tf.reshape(p, [1])
    p = tf.unstack(p)
    # TODO(wangpeng): Make tf version take a tensor for p instead of a list.
    y = tf.math.polyval(p, x)
    # If the polynomial is 0-order, numpy requires the result to be broadcast to
    # `x`'s shape.
    if len(p) == 1:
      y = tf.broadcast_to(y, x.shape)
    return y
  return _bin_op(f, p, x)


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
  if promote_to_float and not np.issubdtype(x.dtype, np.floating):
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


@utils.np_doc(np.deg2rad)
def deg2rad(x):
  def f(x):
    return x * (np.pi / 180.0)
  return _scalar(f, x, True)


@utils.np_doc(np.rad2deg)
def rad2deg(x):
  return x * (180.0 / np.pi)


_tf_float_types = [tf.bfloat16, tf.float16, tf.float32, tf.float64]


@utils.np_doc(np.angle)
def angle(z, deg=False):
  def f(x):
    if x.dtype in _tf_float_types:
      # Workaround for b/147515503
      return tf.where(x < 0, np.pi, 0)
    else:
      return tf.math.angle(x)
  y = _scalar(f, z, True)
  if deg:
    y = rad2deg(y)
  return y


@utils.np_doc(np.cbrt)
def cbrt(x):
  def f(x):
    # __pow__ can't handle negative base, so we use `abs` here.
    rt = tf.math.abs(x) ** (1.0 / 3)
    return tf.where(x < 0, -rt, rt)
  return _scalar(f, x, True)


@utils.np_doc(np.conjugate)
def conjugate(x):
  return _scalar(tf.math.conj, x, True)


@utils.np_doc(np.exp2)
def exp2(x):
  def f(x):
    return 2 ** x
  return _scalar(f, x, True)


@utils.np_doc(np.expm1)
def expm1(x):
  return _scalar(tf.math.expm1, x, True)


@utils.np_doc(np.fix)
def fix(x):
  def f(x):
    return tf.where(x < 0, tf.math.ceil(x), tf.math.floor(x))
  return _scalar(f, x, True)


@utils.np_doc(np.iscomplex)
def iscomplex(x):
  return array_methods.imag(x) != 0


@utils.np_doc(np.isreal)
def isreal(x):
  return array_methods.imag(x) == 0


@utils.np_doc(np.iscomplexobj)
def iscomplexobj(x):
  x = array_creation.asarray(x)
  return np.issubdtype(x.dtype, np.complexfloating)


@utils.np_doc(np.isrealobj)
def isrealobj(x):
  return not iscomplexobj(x)


@utils.np_doc(np.isnan)
def isnan(x):
  return _scalar(tf.math.is_nan, x, True)


@utils.np_doc(np.isfinite)
def isfinite(x):
  return _scalar(tf.math.is_finite, x, True)


@utils.np_doc(np.isinf)
def isinf(x):
  return _scalar(tf.math.is_inf, x, True)


@utils.np_doc(np.isneginf)
def isneginf(x):
  return x == array_creation.full_like(x, -np.inf)


@utils.np_doc(np.isposinf)
def isposinf(x):
  return x == array_creation.full_like(x, np.inf)


@utils.np_doc(np.log2)
def log2(x):
  return log(x) / np.log(2)


@utils.np_doc(np.log10)
def log10(x):
  return log(x) / np.log(10)


@utils.np_doc(np.log1p)
def log1p(x):
  return _scalar(tf.math.log1p, x, True)


@utils.np_doc(np.positive)
def positive(x):
  return _scalar(lambda x: x, x, True)


@utils.np_doc(np.sinc)
def sinc(x):
  def f(x):
    pi_x = x * np.pi
    return tf.where(x == 0, tf.ones_like(x), tf.math.sin(pi_x) / pi_x)
  return _scalar(f, x, True)


@utils.np_doc(np.square)
def square(x):
  return _scalar(tf.math.square, x)


@utils.np_doc(np.diff)
def diff(a, n=1, axis=-1):
  def f(a):
    nd = a.shape.rank
    if (axis + nd if axis < 0 else axis) >= nd:
      raise ValueError("axis %s is out of bounds for array of dimension %s" %
                       (axis, nd))
    if n < 0:
      raise ValueError("order must be non-negative but got %s" % n)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = tf.not_equal if a.dtype == tf.bool else tf.subtract
    for _ in range(n):
      a = op(a[slice1], a[slice2])
    return a
  return _scalar(f, a)


def _atleast_nd(n, new_shape, *arys):
  """Reshape arrays to be at least `n`-dimensional.

  Args:
    n: The minimal rank.
    new_shape: a function that takes `n` and the old shape and returns the
      desired new shape.
    arys: ndarray(s) to be reshaped.

  Returns:
    The reshaped array(s).
  """
  def f(x):
    x = array_creation.asarray(x)
    if x.ndim < n:
      x = array_methods.reshape(x, new_shape(n, x.shape))
    return x
  arys = list(map(f, arys))
  if len(arys) == 1:
    return arys[0]
  else:
    return arys


@utils.np_doc(np.atleast_1d)
def atleast_1d(*arys):
  return _atleast_nd(1, _pad_left_to, *arys)


@utils.np_doc(np.atleast_2d)
def atleast_2d(*arys):
  return _atleast_nd(2, _pad_left_to, *arys)


@utils.np_doc(np.atleast_3d)
def atleast_3d(*arys):
  def new_shape(_, old_shape):
    ndim = len(old_shape)
    if ndim == 0:
      return (1, 1, 1)
    elif ndim == 1:
      return (1,) + old_shape + (1,)
    return old_shape + (1,)
  return _atleast_nd(3, new_shape, *arys)


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
