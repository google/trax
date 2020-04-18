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

# Lint as: python3
"""Tests for base layer."""

from absl.testing import absltest
from trax import math
from trax.layers import base
from trax.math import numpy as np
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

  def test_output_signature(self):
    input_signature = (ShapeDtype((2, 3, 5)), ShapeDtype((2, 3, 5)))
    layer = base.Fn(lambda x, y: x + y)  # n_in = 2, n_out = 1
    output_signature = layer.output_signature(input_signature)
    self.assertEqual(output_signature, ShapeDtype((2, 3, 5)))

    input_signature = ShapeDtype((5, 7))
    layer = base.Fn(lambda x: (x, 2 * x, 3 * x))  # n_in = 1, n_out = 3
    output_signature = layer.output_signature(input_signature)
    self.assertEqual(output_signature, (ShapeDtype((5, 7)),) * 3)
    self.assertNotEqual(output_signature, (ShapeDtype((4, 7)),) * 3)
    self.assertNotEqual(output_signature, (ShapeDtype((5, 7)),) * 2)

  def test_fn_layer_example(self):
    layer = base.Fn(lambda x, y: (x + y, np.concatenate([x, y], axis=0)))
    input_signature = (ShapeDtype((2, 7)), ShapeDtype((2, 7)))
    expected_shape = ((2, 7), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)
    inp = (np.array([2]), np.array([3]))
    x, xs = layer(inp)
    self.assertEqual(int(x), 5)
    self.assertEqual([int(y) for y in xs], [2, 3])

  def test_fn_layer_fails_wrong_f(self):
    with self.assertRaisesRegex(ValueError, 'default arg'):
      base.Fn(lambda x, sth=None: x)
    with self.assertRaisesRegex(ValueError, 'keyword arg'):
      base.Fn(lambda x, **kwargs: x)

  def test_fn_layer_varargs_n_in(self):
    with self.assertRaisesRegex(ValueError, 'variable arg'):
      base.Fn(lambda *args: args[0])
    # Check that varargs work when n_in is set.
    id_layer = base.Fn(lambda *args: args[0], n_in=1)
    input_signature = ShapeDtype((2, 7))
    expected_shape = (2, 7)
    output_shape = base.check_shape_agreement(id_layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_fn_layer_difficult_n_out(self):
    with self.assertRaisesRegex(ValueError, 'n_out'):
      # Determining the output of this layer is hard with dummies.
      base.Fn(lambda x: np.concatencate([x, x], axis=4))
    # Check that this layer works when n_out is set.
    layer = base.Fn(lambda x: np.concatenate([x, x], axis=4), n_out=1)
    input_signature = ShapeDtype((2, 1, 2, 2, 3))
    expected_shape = (2, 1, 2, 2, 6)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

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

      def backward(self, inputs, output, ct, weights, state, new_state,
                   **kwargs):
        return (np.zeros_like(ct), ())

    layer = IdWithZeroGrad()
    rng = math.random.get_prng(0)
    input_signature = ShapeDtype((9, 17))
    random_input = math.random.uniform(rng, input_signature.shape,
                                       minval=-1.0, maxval=1.0)
    layer.init(input_signature)
    f = lambda x: np.mean(layer(x))
    grad = math.grad(f)(random_input)
    self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
    self.assertEqual(sum(sum(grad * grad)), 0.0)  # Each one is 0.

  def test_custom_id_grad(self):

    class IdWithIdGrad(base.Layer):

      def forward(self, x, weights):
        return x

      @property
      def has_backward(self):
        return True

      def backward(self, inputs, output, ct, weights, state, new_state,
                   **kwargs):
        return (inputs, ())

    layer = IdWithIdGrad()
    rng = math.random.get_prng(0)
    input_signature = ShapeDtype((9, 17))
    random_input = math.random.uniform(rng, input_signature.shape,
                                       minval=-1.0, maxval=1.0)
    layer.init(input_signature)
    f = lambda x: np.mean(layer(x))
    grad = math.grad(f)(random_input)
    self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
    self.assertEqual(sum(sum(grad)), sum(sum(random_input)))  # Same as input.

  def test_accelerated_forward_called_twice(self):

    class Constant(base.Layer):

      def new_weights(self, input_signature):
        return 123

      def forward(self, inputs, weights, **kwargs):
        return weights

    layer = Constant()
    layer.init(input_signature=ShapeDtype(()))
    layer(0, n_accelerators=1)
    layer(0, n_accelerators=1)

  def test_custom_name(self):
    layer = base.Layer()
    self.assertIn('Layer', str(layer))
    self.assertNotIn('CustomLayer', str(layer))

    layer = base.Layer(name='CustomLayer')
    self.assertIn('CustomLayer', str(layer))

    # pylint: disable=no-value-for-parameter,invalid-name
    @base.layer()
    def DefaultDecoratorLayer(x, **unused_kwargs):
      return x

    layer = DefaultDecoratorLayer()
    self.assertIn('DefaultDecoratorLayer', str(layer))

    @base.layer(name='CustomDecoratorLayer')
    def NotDefaultDecoratorLayer(x, **unused_kwargs):
      return x

    layer = NotDefaultDecoratorLayer()
    self.assertIn('CustomDecoratorLayer', str(layer))
    # pylint: enable=no-value-for-parameter,invalid-name


if __name__ == '__main__':
  absltest.main()
