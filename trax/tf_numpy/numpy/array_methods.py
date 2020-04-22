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

"""Common array methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import six
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import dtypes
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
    output_type = np.promote_types(a.dtype, int)
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
    output_type = np.promote_types(a.dtype, int)
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
  a = array_creation.asarray(a, dtype=dtype)
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


@utils.np_doc(np.sum)
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  return _reduce(tf.reduce_sum, a, axis=axis, dtype=dtype, keepdims=keepdims,
                 tf_bool_fn=tf.reduce_any)


@utils.np_doc(np.prod)
def prod(a, axis=None, dtype=None, keepdims=None):
  return _reduce(tf.reduce_prod, a, axis=axis, dtype=dtype, keepdims=keepdims,
                 tf_bool_fn=tf.reduce_all)


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


@utils.np_doc(np.repeat)
def repeat(a, repeats, axis=None):
  a = array_creation.asarray(a).data
  repeats = array_creation.asarray(repeats).data
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
  if isinstance(newshape, arrays.ndarray):
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
    value = array_creation.full(arr.shape[len(index):], value)
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
  x, y = array_creation._promote_dtype(x, y)
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
