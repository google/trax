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

import sys

import numpy as np
import six

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import array_ops
from trax.tf_numpy.numpy_impl import arrays
from trax.tf_numpy.numpy_impl import dtypes
from trax.tf_numpy.numpy_impl import utils


@utils.np_doc_only(np.dot)
def dot(a, b):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    return utils.cond(
        utils.logical_or(tf.rank(a) == 0, tf.rank(b) == 0),
        lambda: a * b,
        lambda: utils.cond(  # pylint: disable=g-long-lambda
            tf.rank(b) == 1,
            lambda: tf.tensordot(a, b, axes=[[-1], [-1]]),
            lambda: tf.tensordot(a, b, axes=[[-1], [-2]])))
  return _bin_op(f, a, b)


# TODO(wangpeng): Make element-wise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = array_ops._promote_dtype(a, b)  # pylint: disable=protected-access
  else:
    a = array_ops.array(a)
    b = array_ops.array(b)
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


@utils.np_doc(np.clip)
def clip(a, a_min, a_max):  # pylint: disable=missing-docstring
  if a_min is None and a_max is None:
    raise ValueError('Not more than one of `a_min` and `a_max` may be `None`.')
  if a_min is None:
    return minimum(a, a_max)
  elif a_max is None:
    return maximum(a, a_min)
  else:
    a, a_min, a_max = array_ops._promote_dtype(a, a_min, a_max)  # pylint: disable=protected-access
    return utils.tensor_to_ndarray(
        tf.clip_by_value(*utils.tf_broadcast(a.data, a_min.data, a_max.data)))


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
      six.reraise(ValueError, ValueError(str(err)), sys.exc_info()[2])
  return _bin_op(f, x1, x2)


@utils.np_doc(np.tensordot)
def tensordot(a, b, axes=2):
  return _bin_op(lambda a, b: tf.tensordot(a, b, axes=axes), a, b)


@utils.np_doc_only(np.inner)
def inner(a, b):
  def f(a, b):
    return utils.cond(utils.logical_or(tf.rank(a) == 0, tf.rank(b) == 0),
                      lambda: a * b,
                      lambda: tf.tensordot(a, b, axes=[[-1], [-1]]))
  return _bin_op(f, a, b)


@utils.np_doc(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    # We can't assign to captured variable `axisa`, so make a new variable
    axis_a = axisa
    axis_b = axisb
    axis_c = axisc
    if axis is not None:
      axis_a = axis
      axis_b = axis
      axis_c = axis
    if axis_a < 0:
      axis_a = utils.add(axis_a, tf.rank(a))
    if axis_b < 0:
      axis_b = utils.add(axis_b, tf.rank(b))
    def maybe_move_axis_to_last(a, axis):
      def move_axis_to_last(a, axis):
        return tf.transpose(
            a, tf.concat(
                [tf.range(axis), tf.range(axis + 1, tf.rank(a)), [axis]],
                axis=0))
      return utils.cond(
          axis == utils.subtract(tf.rank(a), 1),
          lambda: a,
          lambda: move_axis_to_last(a, axis))
    a = maybe_move_axis_to_last(a, axis_a)
    b = maybe_move_axis_to_last(b, axis_b)
    a_dim = utils.getitem(tf.shape(a), -1)
    b_dim = utils.getitem(tf.shape(b), -1)
    def maybe_pad_0(a, size_of_last_dim):
      def pad_0(a):
        return tf.pad(a, tf.concat([tf.zeros([tf.rank(a) - 1, 2], tf.int32),
                                    tf.constant([[0, 1]], tf.int32)], axis=0))
      return utils.cond(size_of_last_dim == 2,
                        lambda: pad_0(a),
                        lambda: a)
    a = maybe_pad_0(a, a_dim)
    b = maybe_pad_0(b, b_dim)
    c = tf.linalg.cross(*utils.tf_broadcast(a, b))
    if axis_c < 0:
      axis_c = utils.add(axis_c, tf.rank(c))
    def move_last_to_axis(a, axis):
      r = tf.rank(a)
      return tf.transpose(
          a, tf.concat(
              [tf.range(axis), [r - 1], tf.range(axis, r - 1)], axis=0))
    c = utils.cond(
        (a_dim == 2) & (b_dim == 2),
        lambda: c[..., 2],
        lambda: utils.cond(  # pylint: disable=g-long-lambda
            axis_c == utils.subtract(tf.rank(c), 1),
            lambda: c,
            lambda: move_last_to_axis(c, axis_c)))
    return c
  return _bin_op(f, a, b)


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
  y = _bin_op(f, x1, x2)
  if not np.issubdtype(y.dtype, np.inexact):
    y = y.astype(dtypes.default_float_type())
  return y


@utils.np_doc(np.hypot)
def hypot(x1, x2):
  return sqrt(square(x1) + square(x2))


@utils.np_doc(np.kron)
def kron(a, b):
  # pylint: disable=protected-access,g-complex-comprehension
  a, b = array_ops._promote_dtype(a, b)
  ndim = max(a.ndim, b.ndim)
  if a.ndim < ndim:
    a = array_ops.reshape(a, array_ops._pad_left_to(ndim, a.shape))
  if b.ndim < ndim:
    b = array_ops.reshape(b, array_ops._pad_left_to(ndim, b.shape))
  a_reshaped = array_ops.reshape(a, [i for d in a.shape for i in (d, 1)])
  b_reshaped = array_ops.reshape(b, [i for d in b.shape for i in (1, d)])
  out_shape = tuple(np.multiply(a.shape, b.shape))
  return array_ops.reshape(a_reshaped * b_reshaped, out_shape)


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
  return array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))))


@utils.np_doc(np.logaddexp2)
def logaddexp2(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return array_ops.where(
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
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = tf.convert_to_tensor(rtol, dtype.real_dtype)
      atol_ = tf.convert_to_tensor(atol, dtype.real_dtype)
      result = (tf.math.abs(a - b) <= atol_ + rtol_ * tf.math.abs(b))
      if equal_nan:
        result = result | (tf.math.is_nan(a) & tf.math.is_nan(b))
      return result
    else:
      return a == b
  return _bin_op(f, a, b)


@utils.np_doc(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  return array_ops.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


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
  x = array_ops.asarray(x)
  if promote_to_float and not np.issubdtype(x.dtype, np.inexact):
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
  return _scalar(tf.math.conj, x)


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
  return array_ops.imag(x) != 0


@utils.np_doc(np.isreal)
def isreal(x):
  return array_ops.imag(x) == 0


@utils.np_doc(np.iscomplexobj)
def iscomplexobj(x):
  x = array_ops.array(x)
  return np.issubdtype(x.dtype, np.complexfloating)


@utils.np_doc(np.isrealobj)
def isrealobj(x):
  return not iscomplexobj(x)


@utils.np_doc(np.isnan)
def isnan(x):
  return _scalar(tf.math.is_nan, x, True)


def _make_nan_reduction(onp_reduction, reduction, init_val):
  """Helper to generate nan* functions."""
  @utils.np_doc(onp_reduction)
  def nan_reduction(a, axis=None, dtype=None, keepdims=False):
    a = array_ops.array(a)
    v = array_ops.array(init_val, dtype=a.dtype)
    return reduction(
        array_ops.where(isnan(a), v, a),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims)
  return nan_reduction


nansum = _make_nan_reduction(np.nansum, array_ops.sum, 0)
nanprod = _make_nan_reduction(np.nanprod, array_ops.prod, 1)


@utils.np_doc(np.nanmean)
def nanmean(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=missing-docstring
  a = array_ops.array(a)
  if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
    return array_ops.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  if dtype is None:
    dtype = a.dtype
  normalizer = array_ops.sum(
      nan_mask, axis=axis, dtype=dtype, keepdims=keepdims)
  return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer


@utils.np_doc(np.isfinite)
def isfinite(x):
  return _scalar(tf.math.is_finite, x, True)


@utils.np_doc(np.isinf)
def isinf(x):
  return _scalar(tf.math.is_inf, x, True)


@utils.np_doc(np.isneginf)
def isneginf(x):
  return x == array_ops.full_like(x, -np.inf)


@utils.np_doc(np.isposinf)
def isposinf(x):
  return x == array_ops.full_like(x, np.inf)


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
  return _scalar(lambda x: x, x)


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


def _flip_args(f):
  def _f(a, b):
    return f(b, a)
  return _f


setattr(arrays.ndarray, '__abs__', absolute)
setattr(arrays.ndarray, '__floordiv__', floor_divide)
setattr(arrays.ndarray, '__rfloordiv__', _flip_args(floor_divide))
setattr(arrays.ndarray, '__mod__', mod)
setattr(arrays.ndarray, '__rmod__', _flip_args(mod))
setattr(arrays.ndarray, '__add__', add)
setattr(arrays.ndarray, '__radd__', _flip_args(add))
setattr(arrays.ndarray, '__sub__', subtract)
setattr(arrays.ndarray, '__rsub__', _flip_args(subtract))
setattr(arrays.ndarray, '__mul__', multiply)
setattr(arrays.ndarray, '__rmul__', _flip_args(multiply))
setattr(arrays.ndarray, '__pow__', power)
setattr(arrays.ndarray, '__rpow__', _flip_args(power))
setattr(arrays.ndarray, '__truediv__', true_divide)
setattr(arrays.ndarray, '__rtruediv__', _flip_args(true_divide))


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  dtype = utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = array_ops.array(x1, dtype=dtype)
  x2 = array_ops.array(x2, dtype=dtype)
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
  x1 = array_ops.array(x1, dtype=np.bool_)
  x2 = array_ops.array(x2, dtype=np.bool_)
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
  x = array_ops.array(x, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf.logical_not(x.data))

setattr(arrays.ndarray, '__invert__', logical_not)
setattr(arrays.ndarray, '__lt__', less)
setattr(arrays.ndarray, '__le__', less_equal)
setattr(arrays.ndarray, '__gt__', greater)
setattr(arrays.ndarray, '__ge__', greater_equal)
setattr(arrays.ndarray, '__eq__', equal)
setattr(arrays.ndarray, '__ne__', not_equal)


@utils.np_doc(np.linspace)
def linspace(  # pylint: disable=missing-docstring
    start, stop, num=50, endpoint=True, retstep=False, dtype=float, axis=0):
  if dtype:
    dtype = utils.result_type(dtype)
  start = array_ops.array(start, dtype=dtype).data
  stop = array_ops.array(stop, dtype=dtype).data
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  step = tf.convert_to_tensor(np.nan)
  if endpoint:
    result = tf.linspace(start, stop, num, axis=axis)
    if num > 1:
      step = (stop - start) / (num - 1)
  else:
    # tf.linspace does not support endpoint=False so we manually handle it
    # here.
    if num > 1:
      step = ((stop - start) / num)
      new_stop = tf.cast(stop, step.dtype) - step
      start = tf.cast(start, new_stop.dtype)
      result = tf.linspace(start, new_stop, num, axis=axis)
    else:
      result = tf.linspace(start, stop, num, axis=axis)
  if dtype:
    result = tf.cast(result, dtype)
  if retstep:
    return arrays.tensor_to_ndarray(result), arrays.tensor_to_ndarray(step)
  else:
    return arrays.tensor_to_ndarray(result)


@utils.np_doc(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
  dtype = utils.result_type(start, stop, dtype)
  result = linspace(
      start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis).data
  result = tf.pow(tf.cast(base, result.dtype), result)
  if dtype:
    result = tf.cast(result, dtype)
  return arrays.tensor_to_ndarray(result)


@utils.np_doc(np.ptp)
def ptp(a, axis=None, keepdims=None):
  return (array_ops.amax(a, axis=axis, keepdims=keepdims) -
          array_ops.amin(a, axis=axis, keepdims=keepdims))


@utils.np_doc_only(np.concatenate)
def concatenate(arys, axis=0):
  if not isinstance(arys, (list, tuple)):
    arys = [arys]
  if not arys:
    raise ValueError('Need at least one array to concatenate.')
  dtype = utils.result_type(*arys)
  arys = [array_ops.array(array, dtype=dtype).data for array in arys]
  return arrays.tensor_to_ndarray(tf.concat(arys, axis))


@utils.np_doc_only(np.tile)
def tile(a, reps):
  a = array_ops.array(a).data
  reps = array_ops.array(reps, dtype=tf.int32).reshape([-1]).data

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
      tf.math.count_nonzero(array_ops.array(a).data, axis))


@utils.np_doc(np.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # TODO(nareshmodi): make string tensors also work.
  if kind not in ('quicksort', 'stable'):
    raise ValueError("Only 'quicksort' and 'stable' arguments are supported.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")
  stable = (kind == 'stable')

  a = array_ops.array(a).data

  def _argsort(a, axis, stable):
    if axis is None:
      a = tf.reshape(a, [-1])
      axis = 0

    return tf.argsort(a, axis, stable=stable)

  tf_ans = tf.cond(
      tf.rank(a) == 0, lambda: tf.constant([0]),
      lambda: _argsort(a, axis, stable))

  return array_ops.array(tf_ans, dtype=np.intp)


@utils.np_doc(np.sort)
def sort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  if kind != 'quicksort':
    raise ValueError("Only 'quicksort' is supported.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")

  a = array_ops.array(a)

  if axis is None:
    result_t = tf.sort(tf.reshape(a.data, [-1]), 0)
    return utils.tensor_to_ndarray(result_t)
  else:
    return utils.tensor_to_ndarray(tf.sort(a.data, axis))


def _argminmax(fn, a, axis=None):
  a = array_ops.array(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = tf.reshape(a.data, [-1])
  else:
    a_t = array_ops.atleast_1d(a).data
  return utils.tensor_to_ndarray(fn(input=a_t, axis=axis))


@utils.np_doc(np.argmax)
def argmax(a, axis=None):
  return _argminmax(tf.argmax, a, axis)


@utils.np_doc(np.argmin)
def argmin(a, axis=None):
  return _argminmax(tf.argmin, a, axis)


@utils.np_doc(np.append)
def append(arr, values, axis=None):
  if axis is None:
    return concatenate([array_ops.ravel(arr), array_ops.ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


@utils.np_doc(np.average)
def average(a, axis=None, weights=None, returned=False):  # pylint: disable=missing-docstring
  if axis is not None and not isinstance(axis, six.integer_types):
    # TODO(wangpeng): Support tuple of ints as `axis`
    raise ValueError('`axis` must be an integer. Tuple of ints is not '
                     'supported yet. Got type: %s' % type(axis))
  a = array_ops.array(a)
  if weights is None:  # Treat all weights as 1
    if not np.issubdtype(a.dtype, np.inexact):
      a = a.astype(utils.result_type(a.dtype, dtypes.default_float_type()))
    avg = tf.reduce_mean(a.data, axis=axis)
    if returned:
      if axis is None:
        weights_sum = tf.size(a.data)
      else:
        weights_sum = tf.shape(a.data)[axis]
      weights_sum = tf.cast(weights_sum, a.data.dtype)
  else:
    if np.issubdtype(a.dtype, np.inexact):
      out_dtype = utils.result_type(a.dtype, weights)
    else:
      out_dtype = utils.result_type(a.dtype, weights,
                                    dtypes.default_float_type())
    a = array_ops.array(a, out_dtype).data
    weights = array_ops.array(weights, out_dtype).data

    def rank_equal_case():
      tf.debugging.Assert(tf.reduce_all(tf.shape(a) == tf.shape(weights)),
                          [tf.shape(a), tf.shape(weights)])
      weights_sum = tf.reduce_sum(weights, axis=axis)
      avg = tf.reduce_sum(a * weights, axis=axis) / weights_sum
      return avg, weights_sum
    if axis is None:
      avg, weights_sum = rank_equal_case()
    else:
      def rank_not_equal_case():
        tf.debugging.Assert(tf.rank(weights) == 1, [tf.rank(weights)])
        weights_sum = tf.reduce_sum(weights)
        axes = tf.convert_to_tensor([[axis], [0]])
        avg = tf.tensordot(a, weights, axes) / weights_sum
        return avg, weights_sum
      # We condition on rank rather than shape equality, because if we do the
      # latter, when the shapes are partially unknown but the ranks are known
      # and different, utils.cond will run shape checking on the true branch,
      # which will raise a shape-checking error.
      avg, weights_sum = utils.cond(tf.rank(a) == tf.rank(weights),
                                    rank_equal_case, rank_not_equal_case)

  avg = array_ops.array(avg)
  if returned:
    weights_sum = array_ops.broadcast_to(weights_sum, tf.shape(avg.data))
    return avg, weights_sum
  return avg


@utils.np_doc(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None):  # pylint: disable=missing-docstring
  if dtype:
    dtype = utils.result_type(dtype)
  a = array_ops.asarray(a, dtype).data

  if offset == 0:
    a_shape = a.shape
    if a_shape.rank is not None:
      rank = len(a_shape)
      if (axis1 == -2 or axis1 == rank - 2) and (axis2 == -1 or
                                                 axis2 == rank - 1):
        return utils.tensor_to_ndarray(tf.linalg.trace(a))

  a = array_ops.diagonal(a, offset, axis1, axis2)
  return array_ops.sum(a, -1, dtype)


@utils.np_doc(np.meshgrid)
def meshgrid(*xi, **kwargs):
  """This currently requires copy=True and sparse=False."""
  sparse = kwargs.get('sparse', False)
  if sparse:
    raise ValueError('tf.numpy doesnt support returning sparse arrays yet')

  copy = kwargs.get('copy', True)
  if not copy:
    raise ValueError('tf.numpy only supports copy=True')

  indexing = kwargs.get('indexing', 'xy')

  xi = [array_ops.asarray(arg).data for arg in xi]
  kwargs = {'indexing': indexing}

  outputs = tf.meshgrid(*xi, **kwargs)
  outputs = [utils.tensor_to_ndarray(output) for output in outputs]

  return outputs
