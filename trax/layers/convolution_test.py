# coding=utf-8
# Copyright 2021 The Trax Authors.
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

# Lint as: python3
"""Tests for convolution layers."""

from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl


class ConvolutionTest(absltest.TestCase):

  def test_call(self):
    layer = tl.Conv(30, (3, 3))
    x = np.ones((9, 5, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (9, 3, 3, 30))

  def test_use_bias_true(self):
    layer = tl.Conv(30, (3, 3), use_bias=True)
    x = np.ones((9, 5, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (9, 3, 3, 30))

    self.assertIsInstance(layer.weights, tuple)
    self.assertLen(layer.weights, 2)
    self.assertEqual(layer.weights[0].shape, (3, 3, 20, 30))
    self.assertEqual(layer.weights[1].shape, (30,))

  def test_use_bias_false(self):
    layer = tl.Conv(30, (3, 3), use_bias=False)
    x = np.ones((9, 5, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (9, 3, 3, 30))
    # With use_bias=False, layer.weights is just 'w' and there is no 'b'.
    self.assertEqual(layer.weights.shape, (3, 3, 20, 30))

  def test_call_rebatch(self):
    layer = tl.Conv(30, (3, 3))
    x = np.ones((2, 9, 5, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (2, 9, 3, 3, 30))


class CausalConvolutionTest(absltest.TestCase):

  def test_causal_conv(self):
    layer = tl.CausalConv(filters=30, kernel_width=3)
    x = np.ones((9, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (9, 5, 30))

    # TODO(ddohan): How to test for causality? Gradient check between positions?

  def test_causal_conv_use_bias_false(self):
    layer = tl.CausalConv(filters=30, kernel_width=3, use_bias=False)
    x = np.ones((9, 5, 20))
    layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (9, 5, 30))

    self.assertEqual(layer.weights.shape, (3, 20, 30))


if __name__ == '__main__':
  absltest.main()
