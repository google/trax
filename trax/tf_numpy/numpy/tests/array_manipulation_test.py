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

"""Tests for tf numpy array manipulation methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_ops
from trax.tf_numpy.numpy import arrays


class ArrayManipulationTest(tf.test.TestCase):

  def setUp(self):
    super(ArrayManipulationTest, self).setUp()
    self.array_transforms = [
        lambda x: x,
        tf.convert_to_tensor,
        np.array,
        array_ops.array,
    ]

  def testBroadcastTo(self):

    def run_test(arr, shape):
      for fn in self.array_transforms:
        arg1 = fn(arr)
        self.match(
            array_ops.broadcast_to(arg1, shape), np.broadcast_to(arg1, shape))

    run_test(1, 2)
    run_test(1, (2, 2))
    run_test([1, 2], (2, 2))
    run_test([[1], [2]], (2, 2))
    run_test([[1, 2]], (3, 2))
    run_test([[[1, 2]], [[3, 4]], [[5, 6]]], (3, 4, 2))

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


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
