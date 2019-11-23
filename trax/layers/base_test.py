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
from trax import backend
from trax.layers import base
from trax.shapes import ShapeDtype


class BaseLayerTest(absltest.TestCase):

  def test_new_rng_deterministic(self):
    input_signature = ShapeDtype((2, 3, 5))
    layer1 = base.Layer()
    layer2 = base.Layer(n_in=2, n_out=2)
    _, _ = layer1.init(input_signature)
    _, _ = layer2.init(input_signature)
    rng1 = layer1.new_rng()
    rng2 = layer2.new_rng()
    self.assertEqual(rng1.tolist(), rng2.tolist())

  def test_new_rng_new_value_each_call(self):
    input_signature = ShapeDtype((2, 3, 5))
    layer = base.Layer()
    _, _ = layer.init(input_signature)
    rng1 = layer.new_rng()
    rng2 = layer.new_rng()
    rng3 = layer.new_rng()
    self.assertNotEqual(rng1.tolist(), rng2.tolist())
    self.assertNotEqual(rng2.tolist(), rng3.tolist())

  def test_new_rngs_deterministic(self):
    inputs1 = ShapeDtype((2, 3, 5))
    inputs2 = (ShapeDtype((2, 3, 5)), ShapeDtype((2, 3, 5)))
    layer1 = base.Layer()
    layer2 = base.Layer(n_in=2, n_out=2)
    _, _ = layer1.init(inputs1)
    _, _ = layer2.init(inputs2)
    rng1, rng2 = layer1.new_rngs(2)
    rng3, rng4 = layer2.new_rngs(2)
    self.assertEqual(rng1.tolist(), rng3.tolist())
    self.assertEqual(rng2.tolist(), rng4.tolist())

  def test_new_rngs_new_values_each_call(self):
    input_signature = ShapeDtype((2, 3, 5))
    layer = base.Layer()
    _, _ = layer.init(input_signature)
    rng1, rng2 = layer.new_rngs(2)
    rng3, rng4 = layer.new_rngs(2)
    self.assertNotEqual(rng1.tolist(), rng2.tolist())
    self.assertNotEqual(rng3.tolist(), rng4.tolist())
    self.assertNotEqual(rng1.tolist(), rng3.tolist())
    self.assertNotEqual(rng2.tolist(), rng4.tolist())

  def test_layer_decorator_and_shape_agreement(self):
    @base.layer()
    def add_one(x, **unused_kwargs):
      return x + 1

    output_shape = base.check_shape_agreement(
        add_one(), ShapeDtype((12, 17)))  # pylint: disable=no-value-for-parameter
    self.assertEqual(output_shape, (12, 17))

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
    layer.init(input_signature)
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
    layer.init(input_signature)
    f = lambda x: backend.numpy.mean(layer(x))
    grad = backend.grad(f)(random_input)
    self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
    self.assertEqual(sum(sum(grad)), sum(sum(random_input)))  # Same as input.

if __name__ == '__main__':
  absltest.main()
