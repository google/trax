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

from trax.tf_numpy.numpy import arrays as arrays_lib
from trax.tf_numpy.numpy import dtypes
from trax.tf_numpy.numpy import utils


def empty(shape, dtype=float):
  """Returns an empty array with the specified shape and dtype.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  return zeros(shape, dtype)


def empty_like(a, dtype=None):
  """Returns an empty array with the shape and possibly type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  return zeros_like(a, dtype)


def zeros(shape, dtype=float):
  """Returns an ndarray with the given shape and type filled with zeros.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(shape, arrays_lib.ndarray):
    shape = shape.data
  return arrays_lib.tensor_to_ndarray(tf.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None):
  """Returns an array of zeros with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, arrays_lib.ndarray):
    a = a.data
  if dtype is None:
    # We need to let utils.result_type decide the dtype, not tf.zeros_like
    dtype = utils.result_type(a)
  else:
    # TF and numpy has different interpretations of Python types such as
    # `float`, so we let `utils.result_type` decide.
    dtype = utils.result_type(dtype)
  dtype = tf.as_dtype(dtype)  # Work around b/149877262
  return arrays_lib.tensor_to_ndarray(tf.zeros_like(a, dtype))


def ones(shape, dtype=float):
  """Returns an ndarray with the given shape and type filled with ones.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(shape, arrays_lib.ndarray):
    shape = shape.data
  return arrays_lib.tensor_to_ndarray(tf.ones(shape, dtype=dtype))


def ones_like(a, dtype=None):
  """Returns an array of ones with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, arrays_lib.ndarray):
    a = a.data
  if dtype is None:
    dtype = utils.result_type(a)
  else:
    dtype = utils.result_type(dtype)
  return arrays_lib.tensor_to_ndarray(tf.ones_like(a, dtype))


@utils.np_doc(np.eye)
def eye(N, M=None, k=0, dtype=float):  # pylint: disable=invalid-name
  if dtype:
    dtype = utils.result_type(dtype)
  if not M:
    M = N
  # Making sure N, M and k are `int`
  N = int(N)
  M = int(M)
  k = int(k)
  if k >= M or -k >= N:
    # tf.linalg.diag will raise an error in this case
    return zeros([N, M], dtype=dtype)
  if k == 0:
    return arrays_lib.tensor_to_ndarray(tf.eye(N, M, dtype=dtype))
  # We need the precise length, otherwise tf.linalg.diag will raise an error
  diag_len = min(N, M)
  if k > 0:
    if N >= M:
      diag_len -= k
    elif N + k > M:
      diag_len = M - k
  elif k <= 0:
    if M >= N:
      diag_len += k
    elif M - k > N:
      diag_len = N + k
  diagonal = tf.ones([diag_len], dtype=dtype)
  return arrays_lib.tensor_to_ndarray(
      tf.linalg.diag(diagonal=diagonal, num_rows=N, num_cols=M, k=k))


def identity(n, dtype=float):
  """Returns a square array with ones on the main diagonal and zeros elsewhere.

  Args:
    n: number of rows/cols.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray of shape (n, n) and requested type.
  """
  return eye(N=n, M=n, dtype=dtype)


def full(shape, fill_value, dtype=None):
  """Returns an array with given shape and dtype filled with `fill_value`.

  Args:
    shape: A valid shape object. Could be a native python object or an object
      of type ndarray, numpy.ndarray or tf.TensorShape.
    fill_value: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `fill_value`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `fill_value` can not be broadcast to shape `shape`.
  """
  fill_value = asarray(fill_value, dtype=dtype)
  if utils.isscalar(shape):
    shape = tf.reshape(shape, [1])
  return arrays_lib.tensor_to_ndarray(tf.broadcast_to(fill_value.data, shape))


def full_like(a, fill_value, dtype=None):
  """Returns an array with same shape and dtype as `a` filled with `fill_value`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    fill_value: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `a`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `fill_value` can not be broadcast to shape `shape`.
  """
  a = asarray(a)
  dtype = dtype or utils.result_type(a)
  return full(a.shape, fill_value, dtype)


# TODO(wangpeng): investigate whether we can make `copy` default to False.
# TODO(wangpeng): utils.np_doc can't handle np.array because np.array is a
#   builtin function. Make utils.np_doc support builtin functions.
def array(val, dtype=None, copy=True, ndmin=0):
  """Creates an ndarray with the contents of val.

  Args:
    val: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `val`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.
    copy: Determines whether to create a copy of the backing buffer. Since
      Tensors are immutable, a copy is made only if val is placed on a different
      device than the current one. Even if `copy` is False, a new Tensor may
      need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
      is an ndarray or a Tensor.
    ndmin: The minimum rank of the returned array.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(val, arrays_lib.ndarray):
    result_t = val.data
  else:
    result_t = val

  if copy and isinstance(result_t, tf.Tensor):
    # Note: In eager mode, a copy of `result_t` is made only if it is not on
    # the context device.
    result_t = tf.identity(result_t)

  if not isinstance(result_t, tf.Tensor):
    if not dtype:
      dtype = utils.result_type(result_t)
    # We can't call `convert_to_tensor(result_t, dtype=dtype)` here because
    # convert_to_tensor doesn't allow incompatible arguments such as (5.5, int)
    # while np.array allows them. We need to convert-then-cast.
    result_t = arrays_lib.convert_to_tensor(result_t)
    result_t = tf.cast(result_t, dtype=dtype)
  elif dtype:
    result_t = tf.cast(result_t, dtype)
  ndims = tf.rank(result_t)
  def true_fn():
    old_shape = tf.shape(result_t)
    new_shape = tf.concat([tf.ones(ndmin - ndims, tf.int32), old_shape], axis=0)
    return tf.reshape(result_t, new_shape)
  result_t = utils.cond(utils.greater(ndmin, ndims), true_fn, lambda: result_t)
  return arrays_lib.tensor_to_ndarray(result_t)


@utils.np_doc(np.asarray)
def asarray(a, dtype=None):
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(a, arrays_lib.ndarray) and (
      not dtype or dtype == a.dtype):
    return a
  return array(a, dtype, copy=False)


@utils.np_doc(np.asanyarray)
def asanyarray(a, dtype=None):
  return asarray(a, dtype)


@utils.np_doc(np.ascontiguousarray)
def ascontiguousarray(a, dtype=None):
  return array(a, dtype, ndmin=1)


# Numerical ranges.
def arange(start, stop=None, step=1, dtype=None):
  """Returns `step`-separated values in the range [start, stop).

  Args:
    start: Start of the interval. Included in the range.
    stop: End of the interval. If not specified, `start` is treated as 0 and
      `start` value is used as `stop`. If specified, it is not included in the
      range if `step` is integer. When `step` is floating point, it may or may
      not be included.
    step: The difference between 2 consecutive values in the output range.
      It is recommended to use `linspace` instead of using non-integer values
      for `step`.
    dtype: Optional. Type of the resulting ndarray. Could be a python type, a
      NumPy type or a TensorFlow `DType`. If not provided, the largest type of
      `start`, `stop`, `step` is used.

  Raises:
    ValueError: If step is zero.
  """
  if not step:
    raise ValueError('step must be non-zero.')
  if dtype:
    dtype = utils.result_type(dtype)
  else:
    if stop is None:
      dtype = utils.result_type(start, step)
    else:
      dtype = utils.result_type(start, step, stop)
  if step > 0 and ((stop is not None and start > stop) or
                   (stop is None and start < 0)):
    return array([], dtype=dtype)
  if step < 0 and ((stop is not None and start < stop) or
                   (stop is None and start > 0)):
    return array([], dtype=dtype)
  # TODO(srbs): There are some bugs when start or stop is float type and dtype
  # is integer type.
  return arrays_lib.tensor_to_ndarray(
      tf.cast(tf.range(start, limit=stop, delta=step), dtype=dtype))


@utils.np_doc(np.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=float):
  if dtype:
    dtype = utils.result_type(dtype)
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  if not num:
    return empty([0])
  step = 1.
  if endpoint:
    if num > 1:
      step = tf.pow((stop / start), 1 / (num - 1))
  else:
    step = tf.pow((stop / start), 1 / num)
  result = tf.cast(tf.range(num), step.dtype)
  result = tf.pow(step, result)
  result = tf.multiply(result, start)
  if dtype:
    result = tf.cast(result, dtype=dtype)
  return arrays_lib.tensor_to_ndarray(result)


# Building matrices.
def diag(v, k=0):
  """Returns the array diagonal or constructs a diagonal array.

  If `v` is a 1-d array, returns a 2-d array with v as the diagonal shifted
  to the right/left if `k` is positive/negative.
  If `v` is a 2-d array, returns the 1-d array diagonal shifted to the
  right/left if `k` is positive/negative.

  Args:
    v: 1-d or 2-d array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    k: Position of the diagonal. Defaults to 0, the main diagonal. Positive
      values refer to diagonals shifted right, negative values refer to
      diagonals shifted left.

  Returns:
    1-d or 2-d ndarray.

  Raises:
    ValueError: If v is not 1-d or 2-d.
  """
  v = asarray(v)
  if v.ndim == 0 or v.ndim > 2:
    raise ValueError('Input to diag must be 1-d or 2-d only.')
  if v.ndim == 1:
    if v.shape[0] == 0:
      size = abs(k)
      return zeros((size, size), dtype=v.dtype)
    result = tf.linalg.tensor_diag(v.data)
    if k:
      if k < 0:
        padding = [[-k, 0], [0, -k]]
      else:
        padding = [[0, k], [k, 0]]
      result = tf.pad(tensor=result, paddings=padding)
  else:
    n, m = v.shape
    if not n or not m:
      return empty(0, dtype=v.dtype)
    result = v.data
    if k:
      if k < 0:
        k = -k  # For sanity.
        if k >= n:
          return empty(0, dtype=v.dtype)
        else:
          # We intentionally cut a square matrix since diag_part only
          # supports square matrices.
          size = min(n - k, m)
          result = tf.slice(result, [k, 0], [size, size])
      else:
        if k >= m:
          return empty(0, dtype=v.dtype)
        else:
          # We intentionally cut a square matrix since diag_part only
          # supports square matrices.
          size = min(m - k, n)
          result = tf.slice(result, [0, k], [size, size])
    elif m != n:
      # We intentionally cut a square matrix since diag_part only
      # supports square matrices.
      min_n_m = min(n, m)
      result = tf.slice(result, [0, 0], [min_n_m, min_n_m])
    result = tf.linalg.tensor_diag_part(result)
  return arrays_lib.tensor_to_ndarray(result)


def diagflat(v, k=0):
  """Returns a 2-d array with flattened `v` as diagonal.

  Args:
    v: array_like of any rank. Gets flattened when setting as diagonal.
      Could be an ndarray, a Tensor or any object that can be converted to a
      Tensor using `tf.convert_to_tensor`.
    k: Position of the diagonal. Defaults to 0, the main diagonal. Positive
      values refer to diagonals shifted right, negative values refer to
      diagonals shifted left.

  Returns:
    2-d ndarray.
  """
  v = asarray(v)
  return diag(tf.reshape(v.data, [-1]), k)


def _promote_dtype(*arrays):
  dtype = utils.result_type(*arrays)
  return [asarray(a, dtype=dtype) for a in arrays]


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
    a, b = _promote_dtype(a, b)
  else:
    a = asarray(a)
    b = asarray(b)
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
    def tf_cross(a, b):
      # A version of tf.cross that supports broadcasting
      sh = tf.broadcast_dynamic_shape(tf.shape(a), tf.shape(b))
      a = tf.broadcast_to(a, sh)
      b = tf.broadcast_to(b, sh)
      return tf.linalg.cross(a, b)
    c = tf_cross(a, b)
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
  return _bin_op(f, x1, x2)


@utils.np_doc(np.hypot)
def hypot(x1, x2):
  return sqrt(square(x1) + square(x2))


def _pad_left_to(n, old_shape):
  old_shape = asarray(old_shape, dtype=np.int32).data
  new_shape = tf.pad(
      old_shape, [[tf.math.maximum(n - tf.size(old_shape), 0), 0]],
      constant_values=1)
  return asarray(new_shape)


@utils.np_doc(np.kron)
def kron(a, b):
  a, b = _promote_dtype(a, b)
  ndim = max(a.ndim, b.ndim)
  if a.ndim < ndim:
    a = reshape(a, _pad_left_to(ndim, a.shape))
  if b.ndim < ndim:
    b = reshape(b, _pad_left_to(ndim, b.shape))
  a_reshaped = reshape(a, [i for d in a.shape for i in (d, 1)])
  b_reshaped = reshape(b, [i for d in b.shape for i in (1, d)])
  out_shape = tuple(np.multiply(a.shape, b.shape))
  return reshape(a_reshaped * b_reshaped, out_shape)


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
  return where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))))


@utils.np_doc(np.logaddexp2)
def logaddexp2(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return where(
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
  x = asarray(x)
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
  return imag(x) != 0


@utils.np_doc(np.isreal)
def isreal(x):
  return imag(x) == 0


@utils.np_doc(np.iscomplexobj)
def iscomplexobj(x):
  x = asarray(x)
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
    a = asarray(a)
    v = asarray(init_val, dtype=a.dtype)
    return reduction(where(isnan(a), v, a),
                     axis=axis, dtype=dtype, keepdims=keepdims)
  return nan_reduction


@utils.np_doc(np.sum)
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  return _reduce(tf.reduce_sum, a, axis=axis, dtype=dtype, keepdims=keepdims,
                 tf_bool_fn=tf.reduce_any)


@utils.np_doc(np.prod)
def prod(a, axis=None, dtype=None, keepdims=None):
  return _reduce(tf.reduce_prod, a, axis=axis, dtype=dtype, keepdims=keepdims,
                 tf_bool_fn=tf.reduce_all)


nansum = _make_nan_reduction(np.nansum, sum, 0)
nanprod = _make_nan_reduction(np.nanprod, prod, 1)


@utils.np_doc(np.nanmean)
def nanmean(a, axis=None, dtype=None, keepdims=None):
  a = asarray(a)
  if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
    return mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  normalizer = sum(nan_mask, axis=axis, dtype=np.int64,
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
  return x == full_like(x, -np.inf)


@utils.np_doc(np.isposinf)
def isposinf(x):
  return x == full_like(x, np.inf)


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
    # pylint: disable=g-long-lambda
    x = asarray(x)
    return asarray(
        utils.cond(
            utils.greater(n, tf.rank(x)), lambda: reshape(
                x, new_shape(n, tf.shape(x.data))).data, lambda: x.data))

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
def atleast_3d(*arys):  # pylint: disable=missing-docstring

  def new_shape(_, old_shape):
    # pylint: disable=g-long-lambda
    ndim = tf.size(old_shape)
    return utils.cond(
        ndim == 0, lambda: tf.constant([1, 1, 1], dtype=tf.int32),
        lambda: utils.cond(
            ndim == 1, lambda: tf.pad(old_shape, [[1, 1]], constant_values=1),
            lambda: tf.pad(old_shape, [[0, 1]], constant_values=1)))

  return _atleast_nd(3, new_shape, *arys)


def flip(f):
  def _f(a, b):
    return f(b, a)
  return _f


setattr(arrays_lib.ndarray, '__abs__', absolute)
setattr(arrays_lib.ndarray, '__floordiv__', floor_divide)
setattr(arrays_lib.ndarray, '__rfloordiv__', flip(floor_divide))
setattr(arrays_lib.ndarray, '__mod__', mod)
setattr(arrays_lib.ndarray, '__rmod__', flip(mod))
setattr(arrays_lib.ndarray, '__add__', add)
setattr(arrays_lib.ndarray, '__radd__', flip(add))
setattr(arrays_lib.ndarray, '__sub__', subtract)
setattr(arrays_lib.ndarray, '__rsub__', flip(subtract))
setattr(arrays_lib.ndarray, '__mul__', multiply)
setattr(arrays_lib.ndarray, '__rmul__', flip(multiply))
setattr(arrays_lib.ndarray, '__pow__', power)
setattr(arrays_lib.ndarray, '__rpow__', flip(power))
setattr(arrays_lib.ndarray, '__truediv__', true_divide)
setattr(arrays_lib.ndarray, '__rtruediv__', flip(true_divide))


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  dtype = utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = asarray(x1, dtype=dtype)
  x2 = asarray(x2, dtype=dtype)
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
  x1 = asarray(x1, dtype=np.bool_)
  x2 = asarray(x2, dtype=np.bool_)
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
  x = asarray(x, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf.logical_not(x.data))

setattr(arrays_lib.ndarray, '__invert__', logical_not)
setattr(arrays_lib.ndarray, '__lt__', less)
setattr(arrays_lib.ndarray, '__le__', less_equal)
setattr(arrays_lib.ndarray, '__gt__', greater)
setattr(arrays_lib.ndarray, '__ge__', greater_equal)
setattr(arrays_lib.ndarray, '__eq__', equal)
setattr(arrays_lib.ndarray, '__ne__', not_equal)


@utils.np_doc(np.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=float):
  if dtype:
    dtype = utils.result_type(dtype)
  start = asarray(start, dtype=dtype)
  stop = asarray(stop, dtype=dtype)
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
    return arrays_lib.tensor_to_ndarray(result), step
  else:
    return arrays_lib.tensor_to_ndarray(result)


@utils.np_doc(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
  if dtype:
    dtype = utils.result_type(dtype)
  result = linspace(start, stop, num=num, endpoint=endpoint)
  result = tf.pow(base, result.data)
  if dtype:
    result = tf.cast(result, dtype)
  return arrays_lib.tensor_to_ndarray(result)


@utils.np_doc(np.ptp)
def ptp(a, axis=None, keepdims=None):
  return (amax(a, axis=axis, keepdims=keepdims) -
          amin(a, axis=axis, keepdims=keepdims))


@utils.np_doc_only(np.concatenate)
def concatenate(arys, axis=0):
  if not arys:
    raise ValueError('Need at least one array to concatenate.')
  dtype = utils.result_type(*arys)
  arys = [asarray(array, dtype=dtype).data for array in arys]
  return arrays_lib.tensor_to_ndarray(tf.concat(arys, axis))


@utils.np_doc_only(np.tile)
def tile(a, reps):
  a = asarray(a).data
  reps = asarray(reps, dtype=tf.int32).reshape([-1]).data

  a_rank = tf.rank(a)
  reps_size = tf.size(reps)
  reps = tf.pad(
      reps, [[tf.math.maximum(a_rank - reps_size, 0), 0]],
      constant_values=1)
  a_shape = tf.pad(
      tf.shape(a), [[tf.math.maximum(reps_size - a_rank, 0), 0]],
      constant_values=1)
  a = tf.reshape(a, a_shape)

  return arrays_lib.tensor_to_ndarray(tf.tile(a, reps))


@utils.np_doc(np.count_nonzero)
def count_nonzero(a, axis=None):
  return arrays_lib.tensor_to_ndarray(
      tf.math.count_nonzero(asarray(a).data, axis))


@utils.np_doc(np.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # TODO(nareshmodi): make string tensors also work.
  if kind not in ('quicksort', 'stable'):
    raise ValueError("Only 'quicksort' and 'stable' arguments are supported.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")
  stable = (kind == 'stable')

  a = asarray(a).data

  def _argsort(a, axis, stable):
    if axis is None:
      a = tf.reshape(a, [-1])
      axis = 0

    return tf.argsort(a, axis, stable=stable)

  tf_ans = tf.cond(
      tf.rank(a) == 0, lambda: tf.constant([0]),
      lambda: _argsort(a, axis, stable))

  return asarray(tf_ans, dtype=np.intp)


@utils.np_doc(np.argmax)
def argmax(a, axis=None):
  a = asarray(a)
  a = atleast_1d(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = tf.reshape(a.data, [-1])
  else:
    a_t = a.data
  return utils.tensor_to_ndarray(tf.argmax(input=a_t, axis=axis))


@utils.np_doc(np.argmin)
def argmin(a, axis=None):
  a = asarray(a)
  a = atleast_1d(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = tf.reshape(a.data, [-1])
  else:
    a_t = a.data
  return utils.tensor_to_ndarray(tf.argmin(input=a_t, axis=axis))


def all(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Whether all array elements or those along an axis evaluate to true.

  Casts the array to bool type if it is not already and uses `tf.reduce_all` to
  compute the result.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Could be an int or a tuple of integers. If not specified,
      the reduction is performed over all array indices.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray. Note that unlike NumPy this does not return a scalar bool if
    `axis` is None.
  """
  a = asarray(a, dtype=bool)
  return utils.tensor_to_ndarray(
      tf.reduce_all(input_tensor=a.data, axis=axis, keepdims=keepdims))


def any(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Whether any element in the entire array or in an axis evaluates to true.

  Casts the array to bool type if it is not already and uses `tf.reduce_any` to
  compute the result.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Could be an int or a tuple of integers. If not specified,
      the reduction is performed over all array indices.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray. Note that unlike NumPy this does not return a scalar bool if
    `axis` is None.
  """
  a = asarray(a, dtype=bool)
  return utils.tensor_to_ndarray(
      tf.reduce_any(input_tensor=a.data, axis=axis, keepdims=keepdims))


def clip(a, a_min=None, a_max=None):
  """Clips array values to lie within a given range.

  Uses `tf.clip_by_value`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    a_min: array_like. Must be a scalar or a shape that can be broadcast to
      `a.shape`. At least one of `a_min` or `a_max` should be non-None.
    a_max: array_like. Must be a scalar or a shape that can be broadcast to
      `a.shape`. At least one of `a_min` or `a_max` should be non-None.

  Returns:
    An ndarray with trimmed values with the same shape and dtype as `a`.

  Raises:
    ValueError: if both a_min and a_max are None.
  """
  if a_min is None and a_max is None:
    raise ValueError('Both a_min and a_max cannot be None.')
  a = asarray(a)
  # Unlike np.clip, tf.clip_by_value requires both min and max values to be
  # specified so we set them to the smallest/largest values of the array dtype.
  if a_min is None:
    a_min = np.iinfo(a.dtype).min
  if a_max is None:
    a_max = np.iinfo(a.dtype).max
  a_min = asarray(a_min, dtype=a.dtype)
  a_max = asarray(a_max, dtype=a.dtype)
  return utils.tensor_to_ndarray(
      tf.clip_by_value(a.data, a_min.data, a_max.data))


def compress(condition, a, axis=None):
  """Compresses `a` by selecting values along `axis` with `condition` true.

  Uses `tf.boolean_mask`.

  Args:
    condition: 1-d array of bools. If `condition` is shorter than the array
      axis (or the flattened array if axis is None), it is padded with False.
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Axis along which to select elements. If None, `condition` is
      applied on flattened array.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `condition` is not of rank 1.
  """
  condition = asarray(condition, dtype=bool)
  a = asarray(a)

  if condition.ndim != 1:
    raise ValueError('condition must be a 1-d array.')

  # `np.compress` treats scalars as 1-d arrays.
  if a.ndim == 0:
    a = ravel(a)

  if axis is None:
    a = ravel(a)
    axis = 0

  if axis < 0:
    axis += a.ndim

  assert axis >= 0 and axis < a.ndim

  # `tf.boolean_mask` requires the first dimensions of array and condition to
  # match. `np.compress` pads condition with False when it is shorter.
  condition_t = condition.data
  a_t = a.data
  if condition.shape[0] < a.shape[axis]:
    padding = tf.fill([a.shape[axis] - condition.shape[0]], False)
    condition_t = tf.concat([condition_t, padding], axis=0)
  return utils.tensor_to_ndarray(tf.boolean_mask(tensor=a_t, mask=condition_t,
                                                 axis=axis))


def copy(a):
  """Returns a copy of the array."""
  return array(a, copy=True)


def cumprod(a, axis=None, dtype=None):
  """Returns cumulative product of `a` along an axis or the flattened array.

  Uses `tf.cumprod`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Axis along which to compute products. If None, operation is
      performed on the flattened array.
    dtype: Optional. The type of the output array. If None, defaults to the
      dtype of `a` unless `a` is an integer type with precision less than `int`
      in which case the output type is `int.`

  Returns:
    An ndarray with the same number of elements as `a`. If `axis` is None, the
    output is a 1-d array, else it has the same shape as `a`.
  """
  a = asarray(a, dtype=dtype)

  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np.promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = asarray(a, dtype=output_type)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  if axis < 0:
    axis += a.ndim
  assert axis >= 0 and axis < a.ndim
  return utils.tensor_to_ndarray(tf.math.cumprod(a.data, axis))


def cumsum(a, axis=None, dtype=None):
  """Returns cumulative sum of `a` along an axis or the flattened array.

  Uses `tf.cumsum`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Axis along which to compute sums. If None, operation is
      performed on the flattened array.
    dtype: Optional. The type of the output array. If None, defaults to the
      dtype of `a` unless `a` is an integer type with precision less than `int`
      in which case the output type is `int.`

  Returns:
    An ndarray with the same number of elements as `a`. If `axis` is None, the
    output is a 1-d array, else it has the same shape as `a`.
  """
  a = asarray(a, dtype=dtype)

  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np.promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = asarray(a, dtype=output_type)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  if axis < 0:
    axis += a.ndim
  assert axis >= 0 and axis < a.ndim
  return utils.tensor_to_ndarray(tf.cumsum(a.data, axis))


def imag(a):
  """Returns imaginary parts of all elements in `a`.

  Uses `tf.imag`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray with the same shape as `a`.
  """
  a = asarray(a)
  # TODO(srbs): np.imag returns a scalar if a is a scalar, whereas we always
  # return an ndarray.
  return utils.tensor_to_ndarray(tf.math.imag(a.data))


_TO_INT64 = 0
_TO_FLOAT = 1


def _reduce(tf_fn, a, axis=None, dtype=None, keepdims=None,
            promote_int=_TO_INT64, tf_bool_fn=None, preserve_bool=False):
  """A general reduction function.

  Args:
    tf_fn: the TF reduction function.
    a: the array to be reduced.
    axis: (optional) the axis along which to do the reduction. If None, all
      dimensions are reduced.
    dtype: (optional) the dtype of the result.
    keepdims: (optional) whether to keep the reduced dimension(s).
    promote_int: how to promote integer and bool inputs. There are three
      choices: (1) _TO_INT64: always promote them to int64 or uint64; (2)
      _TO_FLOAT: always promote them to a float type (determined by
      dtypes.default_float_type); (3) None: don't promote.
    tf_bool_fn: (optional) the TF reduction function for bool inputs. It
      will only be used if `dtype` is explicitly set to `np.bool_` or if `a`'s
      dtype is `np.bool_` and `preserve_bool` is True.
    preserve_bool: a flag to control whether to use `tf_bool_fn` if `a`'s dtype
      is `np.bool_` (some reductions such as np.sum convert bools to
      integers, while others such as np.max preserve bools.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if keepdims is None:
    keepdims = False
  a = asarray(a, dtype=dtype)
  if ((dtype == np.bool_ or preserve_bool and a.dtype == np.bool_)
      and tf_bool_fn is not None):
    return utils.tensor_to_ndarray(
        tf_bool_fn(input_tensor=a.data, axis=axis, keepdims=keepdims))
  if dtype is None:
    dtype = a.dtype
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
      if promote_int == _TO_INT64:
        # If a is an integer/bool type and whose bit width is less than 64,
        # numpy up-casts it to 64-bit.
        if dtype == np.bool_:
          is_signed = True
          width = 8  # We can use any number here that is less than 64
        else:
          is_signed = np.issubdtype(dtype, np.signedinteger)
          width = np.iinfo(dtype).bits
        if width < 64:
          if is_signed:
            dtype = np.int64
          else:
            dtype = np.uint64
          a = a.astype(dtype)
      elif promote_int == _TO_FLOAT:
        a = a.astype(dtypes.default_float_type())

  return utils.tensor_to_ndarray(
      tf_fn(input_tensor=a.data, axis=axis, keepdims=keepdims))


@utils.np_doc(np.mean)
def mean(a, axis=None, dtype=None, keepdims=None):
  return _reduce(tf.math.reduce_mean, a, axis=axis, dtype=dtype,
                 keepdims=keepdims, promote_int=_TO_FLOAT)


@utils.np_doc(np.amax)
def amax(a, axis=None, keepdims=None):
  return _reduce(tf.reduce_max, a, axis=axis, dtype=None, keepdims=keepdims,
                 promote_int=None, tf_bool_fn=tf.reduce_any, preserve_bool=True)


@utils.np_doc(np.amin)
def amin(a, axis=None, keepdims=None):
  return _reduce(tf.reduce_min, a, axis=axis, dtype=None, keepdims=keepdims,
                 promote_int=None, tf_bool_fn=tf.reduce_all, preserve_bool=True)


@utils.np_doc(np.var)
def var(a, axis=None, keepdims=None):
  return _reduce(tf.math.reduce_variance, a, axis=axis, dtype=None,
                 keepdims=keepdims, promote_int=_TO_FLOAT)


@utils.np_doc(np.std)
def std(a, axis=None, keepdims=None):
  return _reduce(tf.math.reduce_std, a, axis=axis, dtype=None,
                 keepdims=keepdims, promote_int=_TO_FLOAT)


def ravel(a):
  """Flattens `a` into a 1-d array.

  If `a` is already a 1-d ndarray it is returned as is.

  Uses `tf.reshape`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    A 1-d ndarray.
  """
  a = asarray(a)
  if a.ndim == 1:
    return a
  return utils.tensor_to_ndarray(tf.reshape(a.data, [-1]))


def real(val):
  """Returns real parts of all elements in `a`.

  Uses `tf.real`.

  Args:
    val: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray with the same shape as `a`.
  """
  val = asarray(val)
  # TODO(srbs): np.real returns a scalar if val is a scalar, whereas we always
  # return an ndarray.
  return utils.tensor_to_ndarray(tf.math.real(val.data))


@utils.np_doc(np.repeat)
def repeat(a, repeats, axis=None):
  a = asarray(a).data
  repeats = asarray(repeats).data
  return utils.tensor_to_ndarray(tf.repeat(a, repeats, axis))


def around(a, decimals=0):
  """Rounds each array element to the specified number of decimals.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    decimals: Optional, defaults to 0. The number of decimal places to round to.
      Could be negative.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  factor = math.pow(10, decimals)
  a_t = tf.multiply(a.data, factor)
  a_t = tf.round(a_t)
  a_t = tf.math.divide(a_t, factor)
  return utils.tensor_to_ndarray(a_t)


def reshape(a, newshape):
  """Reshapes an array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    newshape: 0-d or 1-d array_like.

  Returns:
    An ndarray with the contents and dtype of `a` and shape `newshape`.
  """
  a = asarray(a)
  if isinstance(newshape, arrays_lib.ndarray):
    newshape = newshape.data
  return utils.tensor_to_ndarray(tf.reshape(a.data, newshape))


def expand_dims(a, axis):
  """Expand the shape of an array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: int. axis on which to expand the shape.

  Returns:
    An ndarray with the contents and dtype of `a` and shape expanded on axis.
  """
  a = asarray(a)
  return utils.tensor_to_ndarray(tf.expand_dims(a.data, axis=axis))


def squeeze(a, axis=None):
  """Removes single-element axes from the array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: scalar or list/tuple of ints.

  TODO(srbs): tf.squeeze throws error when axis is a Tensor eager execution
  is enabled. So we cannot allow axis to be array_like here. Fix.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  return utils.tensor_to_ndarray(tf.squeeze(a, axis))


def transpose(a, axes=None):
  """Permutes dimensions of the array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axes: array_like. A list of ints with length rank(a) or None specifying the
      order of permutation. The i'th dimension of the output array corresponds
      to axes[i]'th dimension of the `a`. If None, the axes are reversed.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  if axes is not None:
    axes = asarray(axes)
  return utils.tensor_to_ndarray(tf.transpose(a=a.data, perm=axes))


def swapaxes(a, axis1, axis2):
  """Interchange two axes of an array.

  Args:
    a: array_like. Input array.
    axis1: int. First axis.
    axis2: int. Second axis.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  # TODO(wangpeng): handling partial shapes with unknown ranks
  n = len(a.shape)
  if not (-n <= axis1 and axis1 < n):
    raise ValueError('axis1 must be in range [-%s, %s); got %s' % (n, n, axis1))
  if not (-n <= axis2 and axis2 < n):
    raise ValueError('axis2 must be in range [-%s, %s); got %s' % (n, n, axis2))
  if axis1 < 0:
    axis1 += n
  if axis2 < 0:
    axis2 += n
  perm = list(range(n))
  perm[axis1] = axis2
  perm[axis2] = axis1
  return transpose(a, perm)


def _setitem(arr, index, value):
  """Sets the `value` at `index` in the array `arr`.

  This works by replacing the slice at `index` in the tensor with `value`.
  Since tensors are immutable, this builds a new tensor using the `tf.concat`
  op. Currently, only 0-d and 1-d indices are supported.

  Note that this may break gradients e.g.

  a = tf_np.array([1, 2, 3])
  old_a_t = a.data

  with tf.GradientTape(persistent=True) as g:
    g.watch(a.data)
    b = a * 2
    a[0] = 5
  g.gradient(b.data, [a.data])  # [None]
  g.gradient(b.data, [old_a_t])  # [[2., 2., 2.]]

  Here `d_b / d_a` is `[None]` since a.data no longer points to the same
  tensor.

  Args:
    arr: array_like.
    index: scalar or 1-d integer array.
    value: value to set at index.

  Returns:
    ndarray

  Raises:
    ValueError: if `index` is not a scalar or 1-d array.
  """
  # TODO(srbs): Figure out a solution to the gradient problem.
  arr = asarray(arr)
  index = asarray(index)
  if index.ndim == 0:
    index = ravel(index)
  elif index.ndim > 1:
    raise ValueError('index must be a scalar or a 1-d array.')
  value = asarray(value, dtype=arr.dtype)
  if arr.shape[len(index):] != value.shape:
    value = full(arr.shape[len(index):], value)
  prefix_t = arr.data[:index.data[0]]
  postfix_t = arr.data[index.data[0] + 1:]
  if len(index) == 1:
    arr._data = tf.concat(  # pylint: disable=protected-access
        [prefix_t, tf.expand_dims(value.data, 0), postfix_t], 0)
  else:
    subarray = arr[index.data[0]]
    _setitem(subarray, index[1:], value)
    arr._data = tf.concat(  # pylint: disable=protected-access
        [prefix_t, tf.expand_dims(subarray.data, 0), postfix_t], 0)


setattr(arrays_lib.ndarray, 'transpose', transpose)
setattr(arrays_lib.ndarray, 'reshape', reshape)
setattr(arrays_lib.ndarray, '__setitem__', _setitem)


def pad(array, pad_width, mode, constant_values=0):
  """Pads an array.

  Args:
    array: array_like of rank N. Input array.
    pad_width: {sequence, array_like, int}.
      Number of values padded to the edges of each axis.
      ((before_1, after_1), ... (before_N, after_N)) unique pad widths
      for each axis.
      ((before, after),) yields same before and after pad for each axis.
      (pad,) or int is a shortcut for before = after = pad width for all
      axes.
    mode: string. One of the following string values:
      'constant'
          Pads with a constant value.
      'reflect'
          Pads with the reflection of the vector mirrored on
          the first and last values of the vector along each
          axis.
      'symmetric'
          Pads with the reflection of the vector mirrored
          along the edge of the array.
      **NOTE**: The supported list of `mode` does not match that of numpy's.
    constant_values: scalar with same dtype as `array`.
      Used in 'constant' mode as the pad value.  Default is 0.


  Returns:
    An ndarray padded array of rank equal to `array` with shape increased
    according to `pad_width`.

  Raises:
    ValueError if `mode` is not supported.
  """
  if not (mode == 'constant' or mode == 'reflect' or mode == 'symmetric'):
    raise ValueError('Unsupported padding mode: ' + mode)
  mode = mode.upper()
  array = asarray(array)
  pad_width = asarray(pad_width, dtype=tf.int32)
  return utils.tensor_to_ndarray(tf.pad(
      tensor=array.data, paddings=pad_width.data, mode=mode,
      constant_values=constant_values))


def take(a, indices, axis=None):
  """Take elements from an array along an axis.

  See https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html for
  description.

  Args:
    a: array_like. The source array.
    indices: array_like. The indices of the values to extract.
    axis: int, optional. The axis over which to select values. By default, the
      flattened input array is used.

  Returns:
    A ndarray. The returned array has the same type as `a`.
  """
  a = asarray(a)
  indices = asarray(indices)
  a = a.data
  if axis is None:
    a = tf.reshape(a, [-1])
    axis = 0
  return utils.tensor_to_ndarray(tf.gather(a, indices.data, axis=axis))


def where(condition, x, y):
  """Return an array with elements from `x` or `y`, depending on condition.

  Args:
    condition: array_like, bool. Where True, yield `x`, otherwise yield `y`.
    x: see below.
    y: array_like, optional. Values from which to choose. `x`, `y` and
      `condition` need to be broadcastable to some shape.

  Returns:
    An array.
  """
  condition = asarray(condition, dtype=np.bool_)
  x, y = _promote_dtype(x, y)
  return utils.tensor_to_ndarray(tf.where(condition.data, x.data, y.data))


def shape(a):
  """Return the shape of an array.

  Args:
    a: array_like. Input array.

  Returns:
    Tuple of ints.
  """
  a = asarray(a)
  return a.shape


def ndim(a):
  a = asarray(a)
  return a.ndim


def isscalar(a):
  return ndim(a) == 0


def _boundaries_to_sizes(a, boundaries, axis):
  """Converting boundaries of splits to sizes of splits.

  Args:
    a: the array to be split.
    boundaries: the boundaries, as in np.split.
    axis: the axis along which to split.

  Returns:
    A list of sizes of the splits, as in tf.split.
  """
  if axis >= len(a.shape):
    raise ValueError('axis %s is out of bound for shape %s' % (axis, a.shape))
  total_size = a.shape[axis]
  sizes = []
  sizes_sum = 0
  prev = 0
  for i, b in enumerate(boundaries):
    size = b - prev
    if size < 0:
      raise ValueError('The %s-th boundary %s is smaller than the previous '
                       'boundary %s' % (i, b, prev))
    size = min(size, max(0, total_size - sizes_sum))
    sizes.append(size)
    sizes_sum += size
    prev = b
  sizes.append(max(0, total_size - sizes_sum))
  return sizes


def split(a, indices_or_sections, axis=0):
  """Split an array into multiple sub-arrays.

  See https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html for
  reference.

  Args:
    a: the array to be splitted.
    indices_or_sections: int or 1-D array, representing the number of even
      splits or the boundaries between splits.
    axis: the axis along which to split.

  Returns:
    A list of sub-arrays.
  """
  a = asarray(a)
  if not isinstance(indices_or_sections, six.integer_types):
    indices_or_sections = _boundaries_to_sizes(a, indices_or_sections, axis)
  result = tf.split(a.data, indices_or_sections, axis=axis)
  return [utils.tensor_to_ndarray(a) for a in result]


def broadcast_to(a, shape):
  """Broadcasts an array to the desired shape if possible.

  Args:
    a: array_like
    shape: a scalar or 1-d tuple/list.

  Returns:
    An ndarray.
  """
  return full(shape, a)


@utils.np_doc(np.stack)
def stack(arrays, axis=0):
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return asarray(tf.stack(unwrapped_arrays, axis))


@utils.np_doc(np.hstack)
def hstack(tup):
  arrays = [atleast_1d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  rank = tf.rank(unwrapped_arrays[0])
  return utils.cond(rank == 1, lambda: tf.concat(unwrapped_arrays, axis=0),
                    lambda: tf.concat(unwrapped_arrays, axis=1))


@utils.np_doc(np.vstack)
def vstack(tup):
  arrays = [atleast_2d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return tf.concat(unwrapped_arrays, axis=0)


@utils.np_doc(np.dstack)
def dstack(tup):
  arrays = [atleast_3d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return tf.concat(unwrapped_arrays, axis=2)
