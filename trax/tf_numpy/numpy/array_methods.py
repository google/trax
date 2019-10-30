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

"""Common array methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from numpy import iinfo as np_iinfo
from numpy import promote_types as np_promote_types
import six
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import array_manipulation
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import utils


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
  a = array_creation.asarray(a, dtype=bool)
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
  a = array_creation.asarray(a, dtype=bool)
  return utils.tensor_to_ndarray(
      tf.reduce_any(input_tensor=a.data, axis=axis, keepdims=keepdims))


def argmax(a, axis=None):
  """Returns the indices of the maximum values along an array axis.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. The axis along which to compute argmax. If None, index of
      the max element in the flattened array is returned.
  Returns:
    An ndarray with the same shape as `a` with `axis` removed if not None.
    If `axis` is None, a scalar array is returned.
  """
  a = array_creation.asarray(a)
  if axis is None or utils.isscalar(a):
    # When axis is None or the array is a scalar, numpy flattens the array.
    a_t = tf.reshape(a.data, [-1])
  else:
    a_t = a.data
  return utils.tensor_to_ndarray(tf.argmax(input=a_t, axis=axis))


def argmin(a, axis=None):
  """Returns the indices of the minimum values along an array axis.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. The axis along which to compute argmin. If None, index of
      the min element in the flattened array is returned.

  Returns:
    An ndarray with the same shape as `a` with `axis` removed if not None.
    If `axis` is None, a scalar array is returned.
  """
  a = array_creation.asarray(a)
  if axis is None or utils.isscalar(a):
    # When axis is None or the array is a scalar, numpy flattens the array.
    a_t = tf.reshape(a.data, [-1])
  else:
    a_t = a.data
  return utils.tensor_to_ndarray(tf.argmin(input=a_t, axis=axis))


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
  a = array_creation.asarray(a)
  # Unlike np.clip, tf.clip_by_value requires both min and max values to be
  # specified so we set them to the smallest/largest values of the array dtype.
  if a_min is None:
    a_min = np_iinfo(a.dtype).min
  if a_max is None:
    a_max = np_iinfo(a.dtype).max
  a_min = array_creation.asarray(a_min, dtype=a.dtype)
  a_max = array_creation.asarray(a_max, dtype=a.dtype)
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
  condition = array_creation.asarray(condition, dtype=bool)
  a = array_creation.asarray(a)

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
  return array_creation.array(a, copy=True)


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
  a = array_creation.asarray(a, dtype=dtype)

  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np_promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = array_creation.asarray(a, dtype=output_type)

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
  a = array_creation.asarray(a, dtype=dtype)

  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np_promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = array_creation.asarray(a, dtype=output_type)

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
  a = array_creation.asarray(a)
  # TODO(srbs): np.imag returns a scalar if a is a scalar, whereas we always
  # return an ndarray.
  return utils.tensor_to_ndarray(tf.math.imag(a.data))


def amax(a, axis=None, keepdims=None):
  """Returns the maximum value along the axes or in the entire array.

  Uses `tf.reduce_max`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute the max.
      If None, operation is performed on flattened array.
    keepdims: If true, retains reduced dimensions with length 1.
  """
  a = array_creation.asarray(a)
  return utils.tensor_to_ndarray(tf.reduce_max(input_tensor=a.data, axis=axis,
                                               keepdims=keepdims))


def mean(a, axis=None, dtype=None, keepdims=None):
  """Computes the mean of elements across dimensions of a tensor.

  Uses `tf.reduce_mean`.

  Note that the output dtype for this is different from tf.reduce_mean.
  For integer arrays, the output type is float64 whereas for float arrays
  it is the same as the array type. The output type for tf.reduce_mean is
  always the same as the input array.

  ```python
  tf.reduce_mean([1,2,3]) # 2
  np.mean([1,2,3]) # 2.
  ```

  Args:
    a: Instance of ndarray or numpy array_like.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute mean.
      If None, operation is performed on flattened array.
    dtype: Optional. Type of the output array. If None, defaults to the dtype
      of `a`, unless `a` is an integer type in which case this defaults to
      `float64`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray.
  """
  a = array_creation.asarray(a)
  if dtype:
    dtype = utils.to_tf_type(dtype)
  else:
    tf_dtype = tf.as_dtype(a.dtype)
    if tf_dtype.is_integer or tf_dtype.is_bool:
      dtype = tf.float64
  a_t = utils.maybe_cast(a.data, dtype)
  return utils.tensor_to_ndarray(tf.reduce_mean(input_tensor=a_t, axis=axis,
                                                keepdims=keepdims))


def amin(a, axis=None, keepdims=None):
  """Returns the minimum value along the axes or in the entire array.

  Uses `tf.reduce_min`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute min.
      If None, operation is performed on flattened array.
    keepdims: If true, retains reduced dimensions with length 1.
  """
  a = array_creation.asarray(a)
  return utils.tensor_to_ndarray(tf.reduce_min(input_tensor=a.data, axis=axis,
                                               keepdims=keepdims))


def prod(a, axis=None, dtype=None, keepdims=None):
  """Computes the product of elements across dimensions of a tensor.

  Uses `tf.reduce_prod`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute products.
      If None, returns product of all elements in array.
    dtype: Optional. The type of the output array. If None, defaults to the
      dtype of `a` unless `a` is an integer type with precision less than `int`
      in which case the output type is `int.`
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray.
  """
  a = array_creation.asarray(a, dtype=dtype)
  if dtype is None and tf.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np_promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = array_creation.asarray(a, dtype=output_type)

  return utils.tensor_to_ndarray(tf.reduce_prod(input_tensor=a.data, axis=axis,
                                                keepdims=keepdims))


def ptp(a, axis=None):
  """Returns difference between max and min values along an axis.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional 0-d or 1-d array_like. Axes along which to compute mean.
      If None, returns difference between max and min values of the entire
      array.

  Returns:
    An ndarray with same shape as `a` with `axis` dimensions reduced or a scalar
    ndarray if `axis` is None.
  """
  return amax(a, axis=axis) - amin(a, axis=axis)


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
  a = array_creation.asarray(a)
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
  val = array_creation.asarray(val)
  # TODO(srbs): np.real returns a scalar if val is a scalar, whereas we always
  # return an ndarray.
  return utils.tensor_to_ndarray(tf.math.real(val.data))


def repeat(a, repeats, axis=None):
  """Repeat elements of the array along specified axes.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    repeats: 0-d or 1-d array_like. The number of times each element along
      `axis` will be repeated. If this has size 1, each element along the axis
      is repeated the same number of times.
    axis: Optional. The axis along which to repeat. If None, the input array
      is flattened.

  Returns:
    An ndarray with same type as `a`.

  Raises:
    ValueError: If `repeats` has rank > 1 or an incompatible shape.
  """
  a = array_creation.asarray(a)
  repeats = array_creation.asarray(repeats)
  if repeats.ndim > 1:
    raise ValueError('repeats must be a scalar or 1-d array.')
  repeats = ravel(repeats)  # Convert to 1-d array.
  # As per documentation, if axis is None, the input is flattened
  # and a flattened output is returned.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += a.ndim

  # Broadcast repeats to match shape of axis.
  if len(repeats) == 1:
    repeats = utils.tensor_to_ndarray(tf.tile(repeats.data, [a.shape[axis]]))

  if a.shape[axis] != len(repeats):
    raise ValueError('Shape mismatch. `repeats` expected to have shape ({},)'
                     ' but has ({},)'.format(a.shape[axis], len(repeats)))

  # Example:
  #
  # a: [[1, 2, 3],
  #     [4, 5, 6]]
  # axis: 1
  # repeats: [3, 1, 2]
  # Output: [[1, 1, 1, 2, 3, 3],
  #          [4, 4, 4, 5, 6, 6]]
  #
  # Algorithm:
  # 1. Calculate cumulative sum of repeats.
  repeats_cumsum = cumsum(repeats)  # [3, 4, 6]
  # 2. Use `scatter_nd` to generate an indices list for use in `tf.gather`.
  scatter_indices_t = repeats_cumsum[:-1].data  # [3, 4]
  scatter_indices_t = tf.expand_dims(scatter_indices_t, 1)  # [[3], [4]]
  scatter_updates_t = tf.ones([len(repeats) - 1], dtype=tf.int32)  # [1, 1]
  scatter_shape_t = ravel(repeats_cumsum[-1]).data  # [6]
  #    `tf.scatter_nd([[3], [4]], [1, 1], [6])` -> `[0, 0, 0, 1, 1, 0]`
  indices_t = tf.scatter_nd(scatter_indices_t, scatter_updates_t,
                            scatter_shape_t)
  indices_t = tf.cumsum(indices_t)  # [0, 0, 0, 1, 2, 2]
  # 3. Use `tf.gather` to gather indices along `axis`.
  result_t = tf.gather(a, indices_t, axis=axis)

  return utils.tensor_to_ndarray(result_t)


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
  a = array_creation.asarray(a)
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
  a = array_creation.asarray(a)
  newshape = array_creation.asarray(newshape)
  return utils.tensor_to_ndarray(
      tf.reshape(a.data, utils.get_shape_from_ndarray(newshape)))


def expand_dims(a, axis):
  """Expand the shape of an array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    axis: int. axis on which to expand the shape.

  Returns:
    An ndarray with the contents and dtype of `a` and shape expanded on axis.
  """
  a = array_creation.asarray(a)
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
  a = array_creation.asarray(a)
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
  a = array_creation.asarray(a)
  if axes is not None:
    axes = array_creation.asarray(axes)
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
  a = array_creation.asarray(a)
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
  arr = array_creation.asarray(arr)
  index = array_creation.asarray(index)
  if index.ndim == 0:
    index = ravel(index)
  elif index.ndim > 1:
    raise ValueError('index must be a scalar or a 1-d array.')
  value = array_creation.asarray(value, dtype=arr.dtype)
  if arr.shape[len(index):] != value.shape:
    value = array_manipulation.broadcast_to(value, arr.shape[len(index):])
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


setattr(arrays.ndarray, 'transpose', transpose)
setattr(arrays.ndarray, 'reshape', reshape)
setattr(arrays.ndarray, '__setitem__', _setitem)


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
  array = array_creation.asarray(array)
  pad_width = array_creation.asarray(pad_width, dtype=tf.int32)
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
  a = array_creation.asarray(a)
  indices = array_creation.asarray(indices)
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
  condition = array_creation.asarray(condition, dtype=np.bool_)
  x, y = array_creation.promote_args_types(x, y)
  return utils.tensor_to_ndarray(tf.where(condition.data, x.data, y.data))


def shape(a):
  """Return the shape of an array.

  Args:
    a: array_like. Input array.

  Returns:
    Tuple of ints.
  """
  a = array_creation.asarray(a)
  return a.shape


def ndim(a):
  a = array_creation.asarray(a)
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
  a = array_creation.asarray(a)
  if not isinstance(indices_or_sections, six.integer_types):
    indices_or_sections = _boundaries_to_sizes(a, indices_or_sections, axis)
  result = tf.split(a.data, indices_or_sections, axis=axis)
  return [utils.tensor_to_ndarray(a) for a in result]
