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

"""Tests for base layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp
from trax import backend
from trax.layers import base
from trax.shapes import ShapeDtype


class BaseLayerTest(absltest.TestCase):

  def test_layer_decorator_and_shape_agreement(self):
    @base.layer()
    def add_one(x, **unused_kwargs):
      return x + 1

    output_shape = base.check_shape_agreement(
        add_one(), ShapeDtype((12, 17)))  # pylint: disable=no-value-for-parameter
    self.assertEqual(output_shape, (12, 17))

  def test_layer_decorator_no_weights(self):
    # Layer with no defined new_weights_fn should give False for has_weights.
    @base.layer()
    def add_one(x, **unused_kwargs):
      return x + 1

    layer = add_one()  # pylint: disable=no-value-for-parameter
    self.assertFalse(layer.has_weights)

  def test_layer_decorator_has_weights(self):
    # Layer with defined new_weights_fn should give True for has_weights.
    @base.layer(new_weights_fn=onp.ones_like)
    def add_one(x, **unused_kwargs):
      return x + 1

    layer = add_one()  # pylint: disable=no-value-for-parameter
    self.assertTrue(layer.has_weights)

  def test_custom_zero_grad(self):

    class IdWithZeroGrad(base.Layer):

      def forward(self, x, weights):
        return x

      @property
      def has_backward(self):
        return True

      def backward(self, inputs, output, ct, weights, state, **kwargs):
        return (backend.numpy.zeros_like(ct), ())

    layer = IdWithZeroGrad()
    rng = backend.random.get_prng(0)
    input_signature = ShapeDtype((9, 17))
    random_input = backend.random.uniform(rng, input_signature.shape,
                                          minval=-1.0, maxval=1.0)
    layer.initialize_once(input_signature, rng)
    f = lambda x: backend.numpy.mean(layer(x))
    grad = backend.grad(f)(random_input)
    self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
    self.assertEqual(sum(sum(grad * grad)), 0.0)  # Each one is 0.

  def test_custom_id_grad(self):

    class IdWithIdGrad(base.Layer):

      def forward(self, x, weights):
        return x

      @property
      def has_backward(self):
        return True

      def backward(self, inputs, output, ct, weights, state, **kwargs):
        return (inputs, ())

    layer = IdWithIdGrad()
    rng = backend.random.get_prng(0)
    input_signature = ShapeDtype((9, 17))
    random_input = backend.random.uniform(rng, input_signature.shape,
                                          minval=-1.0, maxval=1.0)
    layer.initialize_once(input_signature, rng)
    f = lambda x: backend.numpy.mean(layer(x))
    grad = backend.grad(f)(random_input)
    self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
    self.assertEqual(sum(sum(grad)), sum(sum(random_input)))  # Same as input.

if __name__ == '__main__':
  absltest.main()
