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

"""Tests for backpropgration on tf-numpy functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
# Needed for ndarray.__setitem__
from trax.tf_numpy.numpy import array_methods
from trax.tf_numpy.numpy import math


class BackpropTest(tf.test.TestCase):

  def test_setitem(self):
    # Single integer index.
    a = array_creation.array([1., 2., 3.])
    b = array_creation.array(5.)
    c = array_creation.array(10.)

    tensors = [arr.data for arr in [a, b, c]]
    with tf.GradientTape() as g:
      g.watch(tensors)
      a[1] = b + c
      loss = array_methods.sum(a)

    gradients = g.gradient(loss.data, tensors)
    self.assertSequenceEqual(
        array_creation.array(gradients[0]).tolist(), [1., 0., 1.])
    self.assertEqual(array_creation.array(gradients[1]).tolist(), 1.)
    self.assertEqual(array_creation.array(gradients[2]).tolist(), 1.)

    # Tuple index.
    a = array_creation.array([[[1., 2.], [3., 4.]],
                              [[5., 6.], [7., 8.]]])  # 2x2x2 array.
    b = array_creation.array([10., 11.])

    tensors = [arr.data for arr in [a, b]]
    with tf.GradientTape() as g:
      g.watch(tensors)
      a[(1, 0)] = b
      loss = array_methods.sum(a)

    gradients = g.gradient(loss.data, tensors)
    self.assertSequenceEqual(
        array_creation.array(gradients[0]).tolist(),
        [[[1., 1.], [1., 1.]], [[0., 0.], [1., 1.]]])
    self.assertEqual(array_creation.array(gradients[1]).tolist(), [1., 1.])


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
