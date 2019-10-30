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

"""Utility functions for internal use."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import dtypes


def maybe_cast(a, dtype):
  if not dtype:
    return a
  dtype = to_tf_type(dtype)
  if array_dtype(a) != dtype:
    return tf.cast(a, dtype)
  return a


tensor_to_ndarray = arrays.tensor_to_ndarray


def to_tf_type(dtype):
  """Converts a native python or numpy type to TF DType.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A tensorflow `DType`.
  """
  if isinstance(dtype, tf.DType):
    return dtype
  return tf.as_dtype(dtypes.canonicalize_dtype(np.dtype(dtype)))


def to_numpy_type(dtype):
  """Converts a native python or TF DType to numpy type.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A NumPy `dtype`.
  """
  if isinstance(dtype, tf.DType):
    return dtype.as_numpy_dtype
  return dtypes.canonicalize_dtype(np.dtype(dtype))


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
  return np.finfo(to_numpy_type(dtype))


def array_dtype(a):
  """Returns the tensorflow `DType` of the array.

  Note: This is similar to doing tf.convert_to_tensor(a).dtype but not
  exactly the same. When `a` is a python scalar or a python array_like
  object, Tensorflow attempts to treat it as an int32 or float32 whereas
  numpy defaults to int64 and float64 respectively.

  Args:
    a: Could be a numpy ndarray, a Tensor or a python object that can be
      converted to numpy ndarray.

  Returns:
    A `DType`.
  """
  if isinstance(a, tf.Tensor):
    return a.dtype
  elif isinstance(a, tf.IndexedSlices):
    return a.dtype
  elif isinstance(a, np.ndarray) or isinstance(a, arrays.ndarray):
    return tf.as_dtype(a.dtype)
  elif isinstance(a, arrays.ShardedNdArray):
    return a.tensors[0].dtype
  else:
    # If this is a python object, defer to numpy to decide the dtype.
    np_dtype = np.array(a, copy=False).dtype
    np_dtype = dtypes.canonicalize_dtype(np_dtype)
    return tf.as_dtype(np_dtype)


def get_shape_from_ndarray(shape):
  """Converts an ndarray into a valid shape object.

  - If the shape has dtype int32 or int64, the backing tensor of the ndarray
  is returned.
  - Otherwise a TensorShape object is returned if the shape can be losslessly
  cast to an integral type.

  Args:
    shape: ndarray.

  Returns:
    A Tensor or a TensorShape object.

  Raises:
    ValueError: If the shape is invalid.
  """
  if shape.ndim > 1:
    raise ValueError('shape must have rank <= 1.')
  if shape.dtype.type in (np.int32, np.int64):
    shape = shape.data
  else:
    # Ensure this is a valid shape by converting to TensorShape
    # which raises an exception if there is data loss during casting.
    # TODO(srbs): Raise error when graph building is enabled.
    shape = tf.TensorShape(shape.numpy())
  return shape


def isscalar(val):
  """Returns whether `val` is a scalar value or scalar Tensor."""
  if hasattr(val, 'shape'):  # Handles xy.ndarray and Tensor.
    return len(val.shape) == 0  # pylint: disable=g-explicit-length-test
  return np.isscalar(val)


def scalar_to_vector(val):
  """Converts a scalar value to a vector."""
  if isinstance(val, arrays.ndarray):
    return tensor_to_ndarray(tf.reshape(val.data, [-1]))
  elif isinstance(val, np.ndarray):
    return np.ravel(val)
  else:
    return [val]


def result_type(*arrays_and_dtypes):
  """Returns the type resulting from applying NumPy type promotion to `arrays`.

  Args:
    *arrays_and_dtypes: A list of array_like objects or dtypes.

  Returns:
    A TF Dtype.

  Raises:
    ValueError: If not input arrays are provided.
  """
  def get_dtype(x):
    """Get the dtype from an array or a dtype."""
    try:
      return np.dtype(x)
    except TypeError:
      return array_dtype(x).as_numpy_dtype
  np_dtypes = [get_dtype(x) for x in arrays_and_dtypes]
  return dtypes.get_result_type(*np_dtypes)
