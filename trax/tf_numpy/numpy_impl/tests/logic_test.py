# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for tf numpy random number methods."""
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import array_ops
from trax.tf_numpy.numpy_impl import arrays
from trax.tf_numpy.numpy_impl import math_ops


class LogicTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        tf.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.int32),
        lambda x: np.array(x, dtype=np.int64),
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        array_ops.array,
        lambda x: array_ops.array(x, dtype=tf.int32),
        lambda x: array_ops.array(x, dtype=tf.int64),
        lambda x: array_ops.array(x, dtype=tf.float32),
        lambda x: array_ops.array(x, dtype=tf.float64),
    ]

  def testEqual(self):

    def run_test(x1, x2=None):
      if x2 is None:
        x2 = x1
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arg1 = fn1(x1)
          arg2 = fn2(x2)
          self.match(
              math_ops.equal(arg1, arg2),
              np.equal(
                  make_numpy_compatible(arg1), make_numpy_compatible(arg2)))

    run_test(1)
    run_test(1, 2)
    run_test([1, 2])
    run_test([1, 2, 3], [2])
    run_test([[1, 2], [3, 4]], [1, 2])
    run_test([[1, 2], [1, 4]], [1, 2])
    run_test([1, 2], [[1, 2], [1, 4]])
    run_test([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    run_test([[1, 2], [3, 4]], [[1, 3], [3, 4]])

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

  def match(self, actual, expected, msg=None):
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape:
      self.assertEqual(actual.tolist(), expected.tolist())
    else:
      self.assertSequenceEqual(actual.tolist(), expected.tolist())


def make_numpy_compatible(s):
  return s if not isinstance(s, arrays.ndarray) else s.data.numpy()


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
