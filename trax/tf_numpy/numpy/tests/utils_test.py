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

"""Tests for utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import utils


class UtilsTest(tf.test.TestCase):

  # pylint: disable=unused-argument
  def testNpDoc(self):
    def np_fun(x):
      """np_fun docstring."""
      return
    @utils.np_doc(np_fun)
    def f():
      """f docstring."""
      return
    expected = """TensorFlow variant of `numpy.np_fun`.

Unsupported arguments: `x`.

f docstring.

Documentation for `numpy.np_fun`:

np_fun docstring."""
    self.assertEqual(f.__doc__, expected)

  def testNpDocErrors(self):
    def np_fun(x, y=1, **kwargs):
      return
    # pylint: disable=unused-variable
    with self.assertRaisesRegexp(TypeError, 'Cannot find parameter'):
      @utils.np_doc(np_fun)
      def f1(a):
        return
    with self.assertRaisesRegexp(TypeError, 'is of kind'):
      @utils.np_doc(np_fun)
      def f2(x, kwargs):
        return
    with self.assertRaisesRegexp(
        TypeError, 'Parameter "y" should have a default value'):
      @utils.np_doc(np_fun)
      def f3(x, y):
        return


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
