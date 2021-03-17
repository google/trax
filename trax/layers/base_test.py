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
"""Tests for Trax base layer classes and generic layer-creating functions."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from trax import fastmath
from trax import shapes
from trax.fastmath import numpy as jnp
import trax.layers as tl

BACKENDS = [fastmath.Backend.JAX, fastmath.Backend.TFNP]
CUSTOM_GRAD_BACKENDS = [fastmath.Backend.JAX]  # TODO(afrozm): del after TF 2.3


class BaseLayerTest(parameterized.TestCase):

  def test_call_raises_error(self):
    layer = tl.Layer()
    x = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    with self.assertRaisesRegex(tl.LayerError, 'NotImplementedError'):
      _ = layer(x)

  def test_set_weighs_raises_error(self):
    layer = tl.Layer()
    layer.weights = 1.0  # can assign weights
    with self.assertRaisesRegex(ValueError, 'weighs'):
      layer.weighs = 1.0  # cannot assign weighs

  def test_forward_raises_error(self):
    layer = tl.Layer()
    x = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    with self.assertRaises(NotImplementedError):
      _ = layer.forward(x)

  def test_init_returns_empty_weights_and_state(self):
    layer = tl.Layer()
    input_signature = shapes.ShapeDtype((2, 5))
    weights, state = layer.init(input_signature)
    self.assertEmpty(weights)
    self.assertEmpty(state)

  def test_output_signature_no_weights(self):
    shape_2_3_5 = shapes.ShapeDtype((2, 3, 5))
    input_signature = (shape_2_3_5, shape_2_3_5)
    layer = tl.Fn('2in1out', lambda x, y: x + y)
    output_signature = layer.output_signature(input_signature)
    self.assertEqual(output_signature, shape_2_3_5)

    shape_5_7 = shapes.ShapeDtype((5, 7))
    input_signature = shape_5_7
    layer = tl.Fn('1in3out', lambda x: (x, 2 * x, 3 * x), n_out=3)
    output_signature = layer.output_signature(input_signature)
    self.assertEqual(output_signature, (shape_5_7, shape_5_7, shape_5_7))

  # TODO(jonni): Define/test behavior of output signature for layers w/weights.

  @parameterized.named_parameters(
      [('_' + b.value, b) for b in CUSTOM_GRAD_BACKENDS])
  def test_custom_zero_grad(self, backend):

    class IdWithZeroGrad(tl.Layer):

      def forward(self, x):
        return x

      @property
      def has_backward(self):
        return True

      def backward(self, inputs, output, grad, weights, state, new_state, rng):
        return (jnp.zeros_like(grad), ())

    with fastmath.use_backend(backend):
      layer = IdWithZeroGrad()
      rng = fastmath.random.get_prng(0)
      input_signature = shapes.ShapeDtype((9, 17))
      random_input = fastmath.random.uniform(
          rng, input_signature.shape, minval=-1.0, maxval=1.0)
      layer.init(input_signature)
      f = lambda x: jnp.mean(layer(x))
      grad = fastmath.grad(f)(random_input)
      self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
      self.assertEqual(sum(sum(grad * grad)), 0.0)  # Each one is 0.

  @parameterized.named_parameters(
      [('_' + b.value, b) for b in CUSTOM_GRAD_BACKENDS])
  def test_custom_id_grad(self, backend):

    class IdWithIdGrad(tl.Layer):

      def forward(self, x):
        return x

      @property
      def has_backward(self):
        return True

      def backward(self, inputs, output, grad, weights, state, new_state, rng):
        return (inputs, ())

    with fastmath.use_backend(backend):
      layer = IdWithIdGrad()
      rng = fastmath.random.get_prng(0)
      input_signature = shapes.ShapeDtype((9, 17))
      random_input = fastmath.random.uniform(
          rng, input_signature.shape, minval=-1.0, maxval=1.0)
      layer.init(input_signature)
      f = lambda x: jnp.mean(layer(x))
      grad = fastmath.grad(f)(random_input)
      self.assertEqual(grad.shape, (9, 17))  # Gradient for each input.
      self.assertEqual(sum(sum(grad)), sum(sum(random_input)))  # Same as input.

  def test_weights_and_state_signature(self):

    class MyLayer(tl.Layer):

      def init_weights_and_state(self, input_signature):
        self.weights = jnp.zeros((2, 3))
        self.state = jnp.ones(input_signature.shape)

      def forward(self, inputs):
        return self.weights + self.state

    layer = MyLayer()
    w, s = layer.weights_and_state_signature(jnp.zeros((3, 4)))
    self.assertEqual(w.shape, (2, 3))
    self.assertEqual(s.shape, (3, 4))

  def test_custom_name(self):
    layer = tl.Layer()
    self.assertIn('Layer', str(layer))
    self.assertNotIn('CustomLayer', str(layer))

    layer = tl.Layer(name='CustomLayer')
    self.assertIn('CustomLayer', str(layer))


class PureLayerTest(absltest.TestCase):

  def test_forward(self):
    layer = tl.PureLayer(lambda x: 2 * x)

    # Use Layer.__call__.
    in_0 = np.array([1, 2])
    out_0 = layer(in_0, weights=jnp.zeros((2, 3)))
    self.assertEqual(out_0.tolist(), [2, 4])
    self.assertEmpty(layer.weights)

    # Use PureLayer.forward.
    in_1 = np.array([3, 4])
    out_1 = layer.forward(in_1)
    self.assertEqual(out_1.tolist(), [6, 8])

    # Use Layer.pure_fn
    in_2 = np.array([5, 6])
    out_2, _ = layer.pure_fn(in_2, tl.EMPTY_WEIGHTS, tl.EMPTY_WEIGHTS, None)
    self.assertEqual(out_2.tolist(), [10, 12])


class FnTest(absltest.TestCase):

  def test_bad_f_has_default_arg(self):
    with self.assertRaisesRegex(ValueError, 'default arg'):
      _ = tl.Fn('', lambda x, sth=None: x)

  def test_bad_f_has_keyword_arg(self):
    with self.assertRaisesRegex(ValueError, 'keyword arg'):
      _ = tl.Fn('', lambda x, **kwargs: x)

  def test_bad_f_has_variable_arg(self):
    with self.assertRaisesRegex(ValueError, 'variable arg'):
      _ = tl.Fn('', lambda *args: args[0])

  def test_forward(self):
    layer = tl.Fn(
        'SumAndMax', lambda x0, x1: (x0 + x1, jnp.maximum(x0, x1)), n_out=2)

    x0 = np.array([1, 2, 3, 4, 5])
    x1 = np.array([10, 20, 30, 40, 50])

    y0, y1 = layer((x0, x1))
    self.assertEqual(y0.tolist(), [11, 22, 33, 44, 55])
    self.assertEqual(y1.tolist(), [10, 20, 30, 40, 50])

    y2, y3 = layer.forward((x0, x1))
    self.assertEqual(y2.tolist(), [11, 22, 33, 44, 55])
    self.assertEqual(y3.tolist(), [10, 20, 30, 40, 50])

    (y4, y5), state = layer.pure_fn((x0, x1), tl.EMPTY_WEIGHTS, tl.EMPTY_STATE,
                                    None)
    self.assertEqual(y4.tolist(), [11, 22, 33, 44, 55])
    self.assertEqual(y5.tolist(), [10, 20, 30, 40, 50])
    self.assertEqual(state, tl.EMPTY_STATE)

  def test_weights_state(self):
    layer = tl.Fn(
        '2in2out',
        lambda x, y: (x + y, jnp.concatenate([x, y], axis=0)),
        n_out=2)
    layer.init_weights_and_state(None)
    self.assertEmpty(layer.weights)
    self.assertEmpty(layer.state)


if __name__ == '__main__':
  absltest.main()
