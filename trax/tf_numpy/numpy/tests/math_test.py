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

"""Tests for tf numpy mathematical methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import array_methods
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import math


class MathTest(tf.test.TestCase):

  def setUp(self):
    super(MathTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        tf.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        array_creation.array,
        lambda x: array_creation.array(x, dtype=np.float32),
        lambda x: array_creation.array(x, dtype=np.float64),
    ]
    self.types = [np.int32, np.int64, np.float32, np.float64]

  def _testBinaryOp(self, math_fun, np_fun, name, operands=None,
                    extra_operands=None,
                    check_promotion=True,
                    check_promotion_result_type=True):

    def run_test(a, b):
      for fn in self.array_transforms:
        arg1 = fn(a)
        arg2 = fn(b)
        self.match(
            math_fun(arg1, arg2),
            np_fun(arg1, arg2),
            msg='{}({}, {})'.format(name, arg1, arg2))
      # Tests type promotion
      for type_a in self.types:
        for type_b in self.types:
          if not check_promotion and type_a != type_b:
            continue
          arg1 = array_creation.array(a, dtype=type_a)
          arg2 = array_creation.array(b, dtype=type_b)
          self.match(
              math_fun(arg1, arg2),
              np_fun(arg1, arg2),
              msg='{}({}, {})'.format(name, arg1, arg2),
              check_type=check_promotion_result_type)

    if operands is None:
      operands = [(5, 2),
                  (5, [2, 3]),
                  (5, [[2, 3], [6, 7]]),
                  ([1, 2, 3], 7),
                  ([1, 2, 3], [5, 6, 7])]
    for operand1, operand2 in operands:
      run_test(operand1, operand2)
    if extra_operands is not None:
      for operand1, operand2 in extra_operands:
        run_test(operand1, operand2)

  def testDot(self):
    extra_operands = [
        ([1, 2], [[5, 6, 7], [8, 9, 10]]),
        (np.arange(2 * 3 * 5).reshape([2, 3, 5]).tolist(),
         np.arange(5 * 7 * 11).reshape([7, 5, 11]).tolist())]
    return self._testBinaryOp(math.dot, np.dot, 'dot',
                              extra_operands=extra_operands)

  def testMinimum(self):
    # The numpy version has strange result type when promotion happens,
    # so set check_promotion_result_type to False.
    return self._testBinaryOp(math.minimum, np.minimum, 'minimum',
                              check_promotion_result_type=False)

  def testMaximum(self):
    # The numpy version has strange result type when promotion happens,
    # so set check_promotion_result_type to False.
    return self._testBinaryOp(math.maximum, np.maximum, 'maximum',
                              check_promotion_result_type=False)

  def testMatmul(self):
    operands = [([[1, 2]], [[3, 4, 5], [6, 7, 8]])]
    return self._testBinaryOp(math.matmul, np.matmul, 'matmul',
                              operands=operands)

  def _testUnaryOp(self, math_fun, np_fun, name):

    def run_test(a):
      for fn in self.array_transforms:
        arg1 = fn(a)
        self.match(math_fun(arg1), np_fun(arg1),
                   msg='{}({})'.format(name, arg1))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])

  def testLog(self):
    self._testUnaryOp(math.log, np.log, 'log')

  def testExp(self):
    self._testUnaryOp(math.exp, np.exp, 'exp')

  def testTanh(self):
    self._testUnaryOp(math.tanh, np.tanh, 'tanh')

  def testSqrt(self):
    self._testUnaryOp(math.sqrt, np.sqrt, 'sqrt')

  def _testReduce(self, math_fun, np_fun, name):
    axis_transforms = [
        lambda x: x,  # Identity,
        tf.convert_to_tensor,
        np.array,
        array_creation.array,
        lambda x: array_creation.array(x, dtype=np.float32),
        lambda x: array_creation.array(x, dtype=np.float64),
    ]

    def run_test(a, **kwargs):
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in axis_transforms:
          arg1 = fn1(a)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              math_fun(arg1, axis=axis_arg, **kwargs),
              np_fun(arg1, axis=axis, **kwargs),
              msg='{}({}, axis={}, keepdims={})'.format(
                  name, arg1, axis, kwargs.get('keepdims')))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])
    run_test([[2, -3], [-6, 7]], axis=0)
    run_test([[2, -3], [-6, 7]], axis=0, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=1)
    run_test([[2, -3], [-6, 7]], axis=1, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=(0, 1))
    run_test([[2, -3], [-6, 7]], axis=(1, 0))

  def match(self, actual, expected, msg='', check_type=True):
    self.assertIsInstance(actual, arrays.ndarray)
    if check_type:
      self.assertEqual(
          actual.dtype, expected.dtype,
          'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(
              actual.dtype, expected.dtype, msg))
    self.assertEqual(
        actual.shape, expected.shape,
        'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(
            actual.shape, expected.shape, msg))
    np.testing.assert_almost_equal(actual.tolist(), expected.tolist())

  def testSum(self):
    self._testReduce(array_methods.sum, np.sum, 'sum')

  def testAmax(self):
    self._testReduce(array_methods.amax, np.amax, 'amax')


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
