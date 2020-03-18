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
from trax.tf_numpy.numpy import arrays
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
  a, b = array_creation._promote_dtype(a, b)
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


# TODO(wangpeng): Make element-wise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = array_creation._promote_dtype(a, b)
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
  def _avoid_float64(x1, x2):
    if x1.dtype == x2.dtype and x1.dtype in (tf.int32, tf.int64):
      x1 = tf.cast(x1, dtype=tf.float32)
      x2 = tf.cast(x2, dtype=tf.float32)
    return x1, x2

  def f(x1, x2):
    if x1.dtype == tf.bool:
      assert x2.dtype == tf.bool
      float_ = dtypes.default_float_type()
      x1 = tf.cast(x1, float_)
      x2 = tf.cast(x2, float_)
    if not dtypes.is_allow_float64():
      # tf.math.truediv in Python3 produces float64 when both inputs are int32
      # or int64. We want to avoid that when is_allow_float64() is False.
      x1, x2 = _avoid_float64(x1, x2)
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


@utils.np_doc(np.matmul)
def matmul(x1, x2):  # pylint: disable=missing-docstring
  def f(x1, x2):
    try:
      return utils.cond(tf.rank(x2) == 1,
                        lambda: tf.tensordot(x1, x2, axes=1),
                        lambda: utils.cond(tf.rank(x1) == 1,  # pylint: disable=g-long-lambda
                                           lambda: tf.tensordot(  # pylint: disable=g-long-lambda
                                               x1, x2, axes=[[0], [-2]]),
                                           lambda: tf.matmul(x1, x2)))
    except tf.errors.InvalidArgumentError as err:
      raise ValueError(str(err))
  return _bin_op(f, x1, x2)


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
  a, b = array_creation._promote_dtype(a, b)
  ndim = max(a.ndim, b.ndim)
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


@utils.np_doc(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08):
  def f(a, b):
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = tf.convert_to_tensor(rtol, dtype)
      atol_ = tf.convert_to_tensor(atol, dtype)
      return tf.math.abs(a - b) <= atol_ + rtol_ * tf.math.abs(b)
    else:
      return a == b
  return _bin_op(f, a, b)


def _tf_gcd(x1, x2):
  def _gcd_cond_fn(x1, x2):
    return tf.reduce_any(x2 != 0)
  def _gcd_body_fn(x1, x2):
    # tf.math.mod will raise an error when any element of x2 is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    x2_safe = tf.where(x2 != 0, x2, tf.constant(1, x2.dtype))
    x1, x2 = (tf.where(x2 != 0, x2, x1),
              tf.where(x2 != 0, tf.math.mod(x1, x2_safe),
                       tf.constant(0, x2.dtype)))
    return (tf.where(x1 < x2, x2, x1), tf.where(x1 < x2, x1, x2))
  if (not np.issubdtype(x1.dtype.as_numpy_dtype, np.integer) or
      not np.issubdtype(x2.dtype.as_numpy_dtype, np.integer)):
    raise ValueError("Arguments to gcd must be integers.")
  shape = tf.broadcast_static_shape(x1.shape, x2.shape)
  x1 = tf.broadcast_to(x1, shape)
  x2 = tf.broadcast_to(x2, shape)
  gcd, _ = tf.while_loop(_gcd_cond_fn, _gcd_body_fn,
                         (tf.math.abs(x1), tf.math.abs(x2)))
  return gcd


@utils.np_doc(np.gcd)
def gcd(x1, x2):
  return _bin_op(_tf_gcd, x1, x2)


@utils.np_doc(np.lcm)
def lcm(x1, x2):
  def f(x1, x2):
    d = _tf_gcd(x1, x2)
    # Same as the `x2_safe` trick above
    d_safe = tf.where(d == 0, tf.constant(1, d.dtype), d)
    return tf.where(d == 0, tf.constant(0, d.dtype),
                    tf.math.abs(x1 * x2) // d_safe)
  return _bin_op(f, x1, x2)


def _bitwise_binary_op(tf_fn, x1, x2):
  def f(x1, x2):
    is_bool = (x1.dtype == tf.bool)
    if is_bool:
      assert x2.dtype == tf.bool
      x1 = tf.cast(x1, tf.int8)
      x2 = tf.cast(x2, tf.int8)
    r = tf_fn(x1, x2)
    if is_bool:
      r = tf.cast(r, tf.bool)
    return r
  return _bin_op(f, x1, x2)


@utils.np_doc(np.bitwise_and)
def bitwise_and(x1, x2):
  return _bitwise_binary_op(tf.bitwise.bitwise_and, x1, x2)


@utils.np_doc(np.bitwise_or)
def bitwise_or(x1, x2):
  return _bitwise_binary_op(tf.bitwise.bitwise_or, x1, x2)


@utils.np_doc(np.bitwise_xor)
def bitwise_xor(x1, x2):
  return _bitwise_binary_op(tf.bitwise.bitwise_xor, x1, x2)


@utils.np_doc(np.bitwise_not)
def bitwise_not(x):
  def f(x):
    if x.dtype == tf.bool:
      return tf.logical_not(x)
    return tf.bitwise.invert(x)
  return _scalar(f, x)


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


def _make_nan_reduction(onp_reduction, reduction, init_val):
  @utils.np_doc(onp_reduction)
  def nan_reduction(a, axis=None, dtype=None, keepdims=False):
    a = array_creation.asarray(a)
    v = array_creation.asarray(init_val, dtype=a.dtype)
    return reduction(array_methods.where(isnan(a), v, a),
                     axis=axis, dtype=dtype, keepdims=keepdims)
  return nan_reduction


nansum = _make_nan_reduction(np.nansum, array_methods.sum, 0)
nanprod = _make_nan_reduction(np.nanprod, array_methods.prod, 1)


@utils.np_doc(np.nanmean)
def nanmean(a, axis=None, dtype=None, keepdims=None):
  a = array_creation.asarray(a)
  if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
    return array_methods.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  normalizer = array_methods.sum(nan_mask, axis=axis, dtype=np.int64,
                                 keepdims=keepdims)
  return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer


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


def flip(f):
  def _f(a, b):
    return f(b, a)
  return _f


setattr(arrays.ndarray, '__abs__', absolute)
setattr(arrays.ndarray, '__floordiv__', floor_divide)
setattr(arrays.ndarray, '__rfloordiv__', flip(floor_divide))
setattr(arrays.ndarray, '__mod__', mod)
setattr(arrays.ndarray, '__rmod__', flip(mod))
setattr(arrays.ndarray, '__add__', add)
setattr(arrays.ndarray, '__radd__', flip(add))
setattr(arrays.ndarray, '__sub__', subtract)
setattr(arrays.ndarray, '__rsub__', flip(subtract))
setattr(arrays.ndarray, '__mul__', multiply)
setattr(arrays.ndarray, '__rmul__', flip(multiply))
setattr(arrays.ndarray, '__pow__', power)
setattr(arrays.ndarray, '__rpow__', flip(power))
setattr(arrays.ndarray, '__truediv__', true_divide)
setattr(arrays.ndarray, '__rtruediv__', flip(true_divide))


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  dtype = utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = array_creation.asarray(x1, dtype=dtype)
  x2 = array_creation.asarray(x2, dtype=dtype)
  x1 = x1.data
  x2 = x2.data
  if cast_bool_to_int and x1.dtype == tf.bool:
    x1 = tf.cast(x1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
  return utils.tensor_to_ndarray(tf_fun(x1, x2))


@utils.np_doc(np.equal)
def equal(x1, x2):
  return _comparison(tf.equal, x1, x2)


@utils.np_doc(np.not_equal)
def not_equal(x1, x2):
  return _comparison(tf.not_equal, x1, x2)


@utils.np_doc(np.greater)
def greater(x1, x2):
  return _comparison(tf.greater, x1, x2, True)


@utils.np_doc(np.greater_equal)
def greater_equal(x1, x2):
  return _comparison(tf.greater_equal, x1, x2, True)


@utils.np_doc(np.less)
def less(x1, x2):
  return _comparison(tf.less, x1, x2, True)


@utils.np_doc(np.less_equal)
def less_equal(x1, x2):
  return _comparison(tf.less_equal, x1, x2, True)


@utils.np_doc(np.array_equal)
def array_equal(a1, a2):
  def f(a1, a2):
    if a1.shape != a2.shape:
      return tf.constant(False)
    return tf.reduce_all(tf.equal(a1, a2))
  return _comparison(f, a1, a2)


def _logical_binary_op(tf_fun, x1, x2):
  x1 = array_creation.asarray(x1, dtype=np.bool_)
  x2 = array_creation.asarray(x2, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf_fun(x1.data, x2.data))


@utils.np_doc(np.logical_and)
def logical_and(x1, x2):
  return _logical_binary_op(tf.logical_and, x1, x2)


@utils.np_doc(np.logical_or)
def logical_or(x1, x2):
  return _logical_binary_op(tf.logical_or, x1, x2)


@utils.np_doc(np.logical_xor)
def logical_xor(x1, x2):
  return _logical_binary_op(tf.math.logical_xor, x1, x2)


@utils.np_doc(np.logical_not)
def logical_not(x):
  x = array_creation.asarray(x, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf.logical_not(x.data))

setattr(arrays.ndarray, '__invert__', logical_not)
setattr(arrays.ndarray, '__lt__', less)
setattr(arrays.ndarray, '__le__', less_equal)
setattr(arrays.ndarray, '__gt__', greater)
setattr(arrays.ndarray, '__ge__', greater_equal)
setattr(arrays.ndarray, '__eq__', equal)
setattr(arrays.ndarray, '__ne__', not_equal)


@utils.np_doc(np.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=float):
  if dtype:
    dtype = utils.result_type(dtype)
  start = array_creation.asarray(start, dtype=dtype)
  stop = array_creation.asarray(stop, dtype=dtype)
  if num == 0:
    return empty(dtype)
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  step = np.nan
  if endpoint:
    result = tf.linspace(start.data, stop.data, num)
    if num > 1:
      step = (stop - start) / (num - 1)
  else:
    # tf.linspace does not support endpoint=False so we manually handle it
    # here.
    if num > 1:
      step = (stop - start) / num
      result = tf.linspace(start.data, (stop - step).data, num)
    else:
      result = tf.linspace(start.data, stop.data, num)
  if dtype:
    result = tf.cast(result, dtype)
  if retstep:
    return arrays.tensor_to_ndarray(result), step
  else:
    return arrays.tensor_to_ndarray(result)


@utils.np_doc(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
  if dtype:
    dtype = utils.result_type(dtype)
  result = linspace(start, stop, num=num, endpoint=endpoint)
  result = tf.pow(base, result.data)
  if dtype:
    result = tf.cast(result, dtype)
  return arrays.tensor_to_ndarray(result)


@utils.np_doc(np.ptp)
def ptp(a, axis=None, keepdims=None):
  return (array_methods.amax(a, axis=axis, keepdims=keepdims) -
          array_methods.amin(a, axis=axis, keepdims=keepdims))


@utils.np_doc_only(np.concatenate)
def concatenate(arys, axis=0):
  if not arys:
    raise ValueError('Need at least one array to concatenate.')
  dtype = utils.result_type(*arys)
  arys = [array_creation.asarray(array, dtype=dtype).data for array in arys]
  return arrays.tensor_to_ndarray(tf.concat(arys, axis))


@utils.np_doc_only(np.tile)
def tile(a, reps):
  a = array_creation.asarray(a).data
  reps = array_creation.asarray(reps, dtype=tf.int32).reshape([-1]).data

  a_rank = tf.rank(a)
  reps_size = tf.size(reps)
  reps = tf.pad(
      reps, [[tf.math.maximum(a_rank - reps_size, 0), 0]],
      constant_values=1)
  a_shape = tf.pad(
      tf.shape(a), [[tf.math.maximum(reps_size - a_rank, 0), 0]],
      constant_values=1)
  a = tf.reshape(a, a_shape)

  return arrays.tensor_to_ndarray(tf.tile(a, reps))


@utils.np_doc(np.count_nonzero)
def count_nonzero(a, axis=None):
  return arrays.tensor_to_ndarray(
      tf.math.count_nonzero(array_creation.asarray(a).data, axis))
