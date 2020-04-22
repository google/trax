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

"""Tests for tf numpy array methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import array_methods
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import math


class ArrayMethodsTest(tf.test.TestCase):

  def setUp(self):
    super(ArrayMethodsTest, self).setUp()
    self.array_transforms = [
        lambda x: x,
        tf.convert_to_tensor,
        np.array,
        array_creation.array,
    ]

  def testAllAny(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arr = fn(arr)
        self.match(
            array_methods.all(arr, *args, **kwargs), np.all(
                arr, *args, **kwargs))
        self.match(
            array_methods.any(arr, *args, **kwargs), np.any(
                arr, *args, **kwargs))

    run_test(0)
    run_test(1)
    run_test([])
    run_test([[True, False], [True, True]])
    run_test([[True, False], [True, True]], axis=0)
    run_test([[True, False], [True, True]], axis=0, keepdims=True)
    run_test([[True, False], [True, True]], axis=1)
    run_test([[True, False], [True, True]], axis=1, keepdims=True)
    run_test([[True, False], [True, True]], axis=(0, 1))
    run_test([[True, False], [True, True]], axis=(0, 1), keepdims=True)
    run_test([5.2, 3.5], axis=0)
    run_test([1, 0], axis=0)

  def testClip(self):

    def run_test(arr, *args, **kwargs):
      check_dtype = kwargs.pop('check_dtype', True)
      for fn in self.array_transforms:
        arr = fn(arr)
        self.match(
            math.clip(arr, *args, **kwargs),
            np.clip(arr, *args, **kwargs),
            check_dtype=check_dtype)

    # NumPy exhibits weird typing behavior when a/a_min/a_max are scalars v/s
    # lists, e.g.,
    #
    # np.clip(np.array(0, dtype=np.int32), -5, 5).dtype == np.int64
    # np.clip(np.array([0], dtype=np.int32), -5, 5).dtype == np.int32
    # np.clip(np.array([0], dtype=np.int32), [-5], [5]).dtype == np.int64
    #
    # So we skip matching type. In tf-numpy the type of the output array is
    # always the same as the input array.
    run_test(0, -1, 5, check_dtype=False)
    run_test(-1, -1, 5, check_dtype=False)
    run_test(5, -1, 5, check_dtype=False)
    run_test(-10, -1, 5, check_dtype=False)
    run_test(10, -1, 5, check_dtype=False)
    run_test(10, None, 5, check_dtype=False)
    run_test(10, -1, None, check_dtype=False)
    run_test([0, 20, -5, 4], -1, 5, check_dtype=False)
    run_test([0, 20, -5, 4], None, 5, check_dtype=False)
    run_test([0, 20, -5, 4], -1, None, check_dtype=False)
    run_test([0.5, 20.2, -5.7, 4.4], -1.5, 5.1, check_dtype=False)

    run_test([0, 20, -5, 4], [-5, 0, -5, 0], [0, 5, 0, 5], check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], [2, 0, 2], 5, check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], 0, [5, 3, 1], check_dtype=False)

  def testCompress(self):

    def run_test(condition, arr, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arg1 = fn1(condition)
          arg2 = fn2(arr)
          self.match(
              array_methods.compress(arg1, arg2, *args, **kwargs),
              np.compress(
                  np.asarray(arg1).astype(np.bool), arg2, *args, **kwargs))

    run_test([True], 5)
    run_test([False], 5)
    run_test([], 5)
    run_test([True, False, True], [1, 2, 3])
    run_test([True, False], [1, 2, 3])
    run_test([False, True], [[1, 2], [3, 4]])
    run_test([1, 0, 1], [1, 2, 3])
    run_test([1, 0], [1, 2, 3])
    run_test([0, 1], [[1, 2], [3, 4]])
    run_test([True], [[1, 2], [3, 4]])
    run_test([False, True], [[1, 2], [3, 4]], axis=1)
    run_test([False, True], [[1, 2], [3, 4]], axis=0)
    run_test([False, True], [[1, 2], [3, 4]], axis=-1)
    run_test([False, True], [[1, 2], [3, 4]], axis=-2)

  def testCopy(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.copy(arg, *args, **kwargs),
            np.copy(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([True])
    run_test(np.arange(9).reshape((3, 3)).tolist())

  def testCumProdAndSum(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.cumprod(arg, *args, **kwargs),
            np.cumprod(arg, *args, **kwargs))
        self.match(
            array_methods.cumsum(arg, *args, **kwargs),
            np.cumsum(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1, 2, 3], dtype=float)
    run_test([1, 2, 3], dtype=np.float32)
    run_test([1, 2, 3], dtype=np.float64)
    run_test([1., 2., 3.])
    run_test([1., 2., 3.], dtype=int)
    run_test([1., 2., 3.], dtype=np.int32)
    run_test([1., 2., 3.], dtype=np.int64)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  def testImag(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.imag(arg, *args, **kwargs),
            # np.imag may return a scalar so we convert to a np.ndarray.
            np.array(np.imag(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testAMaxAMin(self):

    def run_test(arr, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              array_methods.amax(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amax(arr_arg, axis=axis, *args, **kwargs))
          self.match(
              array_methods.amin(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amin(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def testMean(self):

    def run_test(arr, *args, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              array_methods.mean(arr_arg, axis=axis_arg, *args, **kwargs),
              np.mean(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 1])
    run_test([1., 2., 1.])
    run_test([1., 2., 1.], dtype=int)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def testProd(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.prod(arg, *args, **kwargs),
            np.prod(arg, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test(np.array([1, 2, 3], dtype=np.int16))
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def testPtp(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            math.ptp(arg, *args, **kwargs), np.ptp(
                arg, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  def testRavel(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.ravel(arg, *args, **kwargs),
            np.ravel(arg, *args, **kwargs))

    run_test(5)
    run_test(5.)
    run_test([])
    run_test([[]])
    run_test([[], []])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]])
    run_test(np.arange(8).reshape((2, 2, 2)).tolist())

  def testReal(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.real(arg, *args, **kwargs),
            np.array(np.real(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testRepeat(self):

    def run_test(arr, repeats, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          repeats_arg = fn2(repeats)
          self.match(
              array_methods.repeat(arr_arg, repeats_arg, *args, **kwargs),
              np.repeat(arr_arg, repeats_arg, *args, **kwargs))

    run_test(1, 2)
    run_test([1, 2], 2)
    run_test([1, 2], [2])
    run_test([1, 2], [1, 2])
    run_test([[1, 2], [3, 4]], 3, axis=0)
    run_test([[1, 2], [3, 4]], 3, axis=1)
    run_test([[1, 2], [3, 4]], [3], axis=0)
    run_test([[1, 2], [3, 4]], [3], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=0)
    run_test([[1, 2], [3, 4]], [3, 2], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-2)

  def testAround(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            array_methods.around(arg, *args, **kwargs),
            np.around(arg, *args, **kwargs))

    run_test(5.5)
    run_test(5.567, decimals=2)
    run_test([])
    run_test([1.27, 2.49, 2.75], decimals=1)
    run_test([23.6, 45.1], decimals=-1)

  def testReshape(self):

    def run_test(arr, newshape, *args, **kwargs):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          newshape_arg = fn2(newshape)
          self.match(
              array_methods.reshape(arr_arg, newshape_arg, *args, **kwargs),
              np.reshape(arr_arg, newshape, *args, **kwargs))

    run_test(5, [-1])
    run_test([], [-1])
    run_test([1, 2, 3], [1, 3])
    run_test([1, 2, 3], [3, 1])
    run_test([1, 2, 3, 4], [2, 2])
    run_test([1, 2, 3, 4], [2, 1, 2])

  def testExpandDims(self):

    def run_test(arr, axis):
      self.match(
          array_methods.expand_dims(arr, axis),
          np.expand_dims(arr, axis))

    run_test([1, 2, 3], 0)
    run_test([1, 2, 3], 1)

  def testSqueeze(self):

    def run_test(arr, *args, **kwargs):
      for fn in self.array_transforms:
        arg = fn(arr)
        # Note: np.squeeze ignores the axis arg for non-ndarray objects.
        # This looks like a bug: https://github.com/numpy/numpy/issues/8201
        # So we convert the arg to np.ndarray before passing to np.squeeze.
        self.match(
            array_methods.squeeze(arg, *args, **kwargs),
            np.squeeze(np.array(arg), *args, **kwargs))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([[1, 2, 3]])
    run_test([[[1], [2], [3]]])
    run_test([[[1], [2], [3]]], axis=0)
    run_test([[[1], [2], [3]]], axis=2)
    run_test([[[1], [2], [3]]], axis=(0, 2))
    run_test([[[1], [2], [3]]], axis=-1)
    run_test([[[1], [2], [3]]], axis=-3)

  def testTranspose(self):

    def run_test(arr, axes=None):
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axes_arg = fn2(axes) if axes is not None else None
          self.match(
              array_methods.transpose(arr_arg, axes_arg),
              np.transpose(arr_arg, axes))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([5, 6, 7])
    run_test(np.arange(30).reshape(2, 3, 5).tolist())
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 1, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 2, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 0, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 2, 0])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 0, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 1, 0])

  def testSetItem(self):

    def run_test(arr, index, value):
      for fn in self.array_transforms:
        value_arg = fn(value)
        tf_array = array_creation.array(arr)
        np_array = np.array(arr)
        tf_array[index] = value_arg
        # TODO(srbs): "setting an array element with a sequence" is thrown
        # if we do not wrap value_arg in a numpy array. Investigate how this can
        # be avoided.
        np_array[index] = np.array(value_arg)
        self.match(tf_array, np_array)

    run_test([1, 2, 3], 1, 5)
    run_test([[1, 2], [3, 4]], 0, [6, 7])
    run_test([[1, 2], [3, 4]], 1, [6, 7])
    run_test([[1, 2], [3, 4]], (0, 1), 6)
    run_test([[1, 2], [3, 4]], 0, 6)  # Value needs to broadcast.

  def match_shape(self, actual, expected, msg=None):
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)
    if msg:
      msg = 'Shape: {} is not a tuple for {}'.format(actual.shape, msg)
    self.assertIsInstance(actual.shape, tuple, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None, check_dtype=True):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, arrays.ndarray)
    if check_dtype:
      self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape:
      self.assertAllClose(actual.tolist(), expected.tolist())
    else:
      self.assertAllClose(actual.tolist(), expected.tolist())

  def testPad(self):
    t = [[1, 2, 3], [4, 5, 6]]
    paddings = [[1, 1,], [2, 2]]
    self.assertAllEqual(array_methods.pad(t, paddings, 'constant'),
                        [[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 2, 3, 0, 0],
                         [0, 0, 4, 5, 6, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]])

    self.assertAllEqual(array_methods.pad(t, paddings, 'reflect'),
                        [[6, 5, 4, 5, 6, 5, 4],
                         [3, 2, 1, 2, 3, 2, 1],
                         [6, 5, 4, 5, 6, 5, 4],
                         [3, 2, 1, 2, 3, 2, 1]])

    self.assertAllEqual(array_methods.pad(t, paddings, 'symmetric'),
                        [[2, 1, 1, 2, 3, 3, 2],
                         [2, 1, 1, 2, 3, 3, 2],
                         [5, 4, 4, 5, 6, 6, 5],
                         [5, 4, 4, 5, 6, 6, 5]])

  def testTake(self):
    a = [4, 3, 5, 7, 6, 8]
    indices = [0, 1, 4]
    self.assertAllEqual([4, 3, 6], array_methods.take(a, indices))
    indices = [[0, 1], [2, 3]]
    self.assertAllEqual([[4, 3], [5, 7]], array_methods.take(a, indices))
    a = [[4, 3, 5], [7, 6, 8]]
    self.assertAllEqual([[4, 3], [5, 7]], array_methods.take(a, indices))
    a = np.random.rand(2, 16, 3)
    axis = 1
    self.assertAllEqual(np.take(a, indices, axis=axis),
                        array_methods.take(a, indices, axis=axis))

  def testWhere(self):
    self.assertAllEqual([[1.0, 1.0], [1.0, 1.0]],
                        array_methods.where([True], [1.0, 1.0],
                                            [[0, 0], [0, 0]]))

  def testShape(self):
    self.assertAllEqual((1, 2), array_methods.shape([[0, 0]]))

  def testSwapaxes(self):
    x = [[1, 2, 3]]
    self.assertAllEqual([[1], [2], [3]], array_methods.swapaxes(x, 0, 1))
    self.assertAllEqual([[1], [2], [3]], array_methods.swapaxes(x, -2, -1))
    x = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        array_methods.swapaxes(x, 0, 2))
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        array_methods.swapaxes(x, -3, -1))

  def testNdim(self):
    self.assertAllEqual(0, array_methods.ndim(0.5))
    self.assertAllEqual(1, array_methods.ndim([1, 2]))

  def testIsscalar(self):
    self.assertTrue(array_methods.isscalar(0.5))
    self.assertTrue(array_methods.isscalar(5))
    self.assertTrue(array_methods.isscalar(False))
    self.assertFalse(array_methods.isscalar([1, 2]))

  def assertListEqual(self, a, b):
    self.assertAllEqual(len(a), len(b))
    for x, y in zip(a, b):
      self.assertAllEqual(x, y)

  def testSplit(self):
    x = array_creation.arange(9)
    y = array_methods.split(x, 3)
    self.assertListEqual([([0, 1, 2]),
                          ([3, 4, 5]),
                          ([6, 7, 8])], y)

    x = array_creation.arange(8)
    y = array_methods.split(x, [3, 5, 6, 10])
    self.assertListEqual([([0, 1, 2]),
                          ([3, 4]),
                          ([5]),
                          ([6, 7]),
                          ([])], y)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
