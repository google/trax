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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy import extensions
import trax.tf_numpy.numpy as tf_np

from tensorflow.python.ops.numpy_ops import np_math_ops  # pylint: disable=g-direct-tensorflow-import


class VmapTest(tf.test.TestCase, parameterized.TestCase):

  def test_vmap_in_axes_list(self):
    # https://github.com/google/jax/issues/2367
    dictionary = {'a': 5., 'b': tf_np.ones(2)}
    x = tf_np.zeros(3)
    y = tf_np.arange(3.)

    def f(dct, x, y):
      return dct['a'] + dct['b'] + x + y

    out1 = extensions.vmap(f, (None, 0, 0))(dictionary, x, y)
    out2 = extensions.vmap(f, [None, 0, 0])(dictionary, x, y)
    self.assertAllClose(out1, out2)

  def test_vmap_in_axes_tree_prefix_error(self):
    # https://github.com/google/jax/issues/795
    self.assertRaisesRegex(
        ValueError,
        'vmap in_axes specification must be a tree prefix of the corresponding '
        r'value, got specification \(0, 0\) for value tree ',
        lambda: extensions.vmap(lambda x: x, in_axes=(0, 0))(tf_np.ones(3)))

  def test_vmap_in_axes_leaf_types(self):
    with self.assertRaisesRegex(TypeError,
                                r'vmap in_axes must be an int, None, or .*'):
      extensions.vmap(
          lambda x: x, in_axes=(tf_np.array([1., 2.]),))(
              tf_np.array([1., 2.]))

  def test_vmap_out_axes_leaf_types(self):
    with self.assertRaisesRegex(TypeError,
                                r'vmap out_axes must be an int, None, or .*'):
      extensions.vmap(
          lambda x: x, out_axes=(tf_np.array([1., 2.]),))(
              tf_np.array([1., 2.]))

  def test_vmap_unbatched_object_passthrough_issue_183(self):
    # https://github.com/google/jax/issues/183
    fun = lambda f, x: f(x)
    vfun = extensions.vmap(fun, (None, 0))
    ans = vfun(lambda x: x + 1, tf_np.arange(3))
    self.assertAllClose(ans, np.arange(1, 4))

  def test_vmap_mismatched_axis_sizes_error_message_issue_705(self):
    # https://github.com/google/jax/issues/705
    with self.assertRaisesRegex(
        ValueError, 'vmap must have at least one non-None value in in_axes'):
      # If the output is mapped, there must be a non-None in_axes
      extensions.vmap(lambda x: x, in_axes=None)(tf_np.array([1., 2.]))

    # Error is: TypeError: only integer scalar arrays can be converted to a
    # scalar index
    with self.assertRaisesRegex(
        ValueError, 'vmap out_axes specification must be a tree prefix of the '
        'corresponding value.*'):
      extensions.vmap(
          lambda x: x, in_axes=0, out_axes=(2, 3))(
              tf_np.array([1., 2.]))

  def test_vmap_structured_in_axes(self):
    a, b, c, d = 2, 3, 4, 5
    k = 6  # batch size
    x = np.ones((k, a, b))  # batch axis in different locations
    y = np.ones((b, k, c))
    z = np.ones((c, d, k))

    def foo(tree_arg):
      x, (y, z) = tree_arg
      return tf_np.dot(x, tf_np.dot(y, z))

    tree = (x, (y, z))
    vfoo = extensions.vmap(foo, in_axes=((0, (1, 2)),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    Point = collections.namedtuple('Point', ['x', 'y'])
    tree = (x, Point(y, z))
    vfoo = extensions.vmap(foo, in_axes=((0, Point(1, 2)),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    def foo2(tree_arg):
      x, dct = tree_arg
      y, z = dct['a'], dct['b']
      return tf_np.dot(x, tf_np.dot(y, z))

    tree = (x, {'a': y, 'b': z})
    vfoo = extensions.vmap(foo2, in_axes=((0, {'a': 1, 'b': 2}),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

    tree = (x, collections.OrderedDict([('a', y), ('b', z)]))
    vfoo = extensions.vmap(
        foo2, in_axes=((0, collections.OrderedDict([('a', 1), ('b', 2)])),))
    self.assertEqual(vfoo(tree).shape, (6, 2, 5))

  def test_vmap_out_axes(self):
    f = extensions.vmap(lambda x: x, out_axes=0)
    inp = tf_np.arange(6).reshape([2, 3])
    self.assertAllClose(inp, f(inp))
    self.assertAllClose([inp, inp], f((inp, inp)))

    f = extensions.vmap(lambda x: x, out_axes=-1)
    self.assertAllClose(inp.T, f(inp))

    f = extensions.vmap(lambda x: x, out_axes=None)
    self.assertAllClose(inp[0], f(inp))

    f = extensions.vmap(lambda x: x, out_axes=([0], (-1, None), {'a': 1}))
    a, b, c = f(([inp], (inp, inp), {'a': inp}))
    self.assertAllClose([inp], a)
    self.assertAllClose((inp.T, inp[0]), b)
    self.assertAllClose(inp.T, c['a'])

  def test_negative_axes(self):
    x = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    self.assertAllClose(
        extensions.vmap(tf_np.sum, in_axes=-3)(x), tf_np.sum(x, axis=(1, 2)))
    self.assertAllClose(
        extensions.vmap(tf_np.sum, in_axes=-2)(x), tf_np.sum(x, axis=(0, 2)))
    self.assertAllClose(
        extensions.vmap(tf_np.sum, in_axes=-1)(x), tf_np.sum(x, axis=(0, 1)))

    identity = lambda y: y
    self.assertAllClose(x, extensions.vmap(identity, in_axes=0, out_axes=-3)(x))
    self.assertAllClose(
        x.transpose(1, 0, 2),
        extensions.vmap(identity, in_axes=0, out_axes=-2)(x))
    self.assertAllClose(
        x.transpose(1, 2, 0),
        extensions.vmap(identity, in_axes=0, out_axes=-1)(x))

    self.assertAllClose(
        np.full((5,), 7),
        extensions.vmap(lambda *xs: xs, in_axes=(0, None),
                        out_axes=(0, -1))(np.arange(5), 7)[1])


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  np_math_ops.enable_numpy_methods_on_tensor()
  tf.test.main()
