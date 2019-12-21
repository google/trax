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


def maximum(a, b):
  return _bin_op(tf.math.maximum, a, b)


def minimum(a, b):
  return _bin_op(tf.math.minimum, a, b)


# TODO(wangpeng): support broadcasting and 1-D case
def matmul(a, b):
  return _bin_op(tf.matmul, a, b)


def _scalar(x, tf_fn):
  """Computes the tf_fn(x) for each element in `x`.

  Args:
    x: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    tf_fn: function that takes a single Tensor argument.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `dtypes.default_float_type`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
  x = array_creation.asarray(x)
  if x.dtype not in (np.float16, np.float32, np.float64):
    x = x.astype(dtypes.default_float_type())
  return utils.tensor_to_ndarray(tf_fn(x.data))


# TODO(agarwal): programmatically add documentation to functions below..
def log(x):
  return _scalar(x, tf.math.log)


def exp(x):
  return _scalar(x, tf.exp)


def tanh(x):
  return _scalar(x, tf.tanh)


@utils.np_doc(np.sqrt)
def sqrt(x):
  return _scalar(x, tf.sqrt)


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
