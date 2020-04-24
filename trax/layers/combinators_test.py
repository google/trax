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
"""Tests for combinator layers."""

from absl.testing import absltest
from trax.layers import activation_fns
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import normalization
from trax.math import numpy as np
from trax.shapes import ShapeDtype


def divide_by(val):
  """Returns a simple division layer with n_in == 1 and n_out == 1."""
  return base.Fn('divide_by', lambda x: x / val)


class CombinatorLayerTest(absltest.TestCase):

  def test_serial_no_op(self):
    layer = cb.Serial(None)
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_no_op_list(self):
    layer = cb.Serial([])
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_one_in_one_out(self):
    layer = cb.Serial(divide_by(3.0))
    input_signature = ShapeDtype((3, 2))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_div_div(self):
    layer = cb.Serial(divide_by(2.0), divide_by(5.0))
    input_signature = ShapeDtype((3, 2))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_dup_dup(self):
    layer = cb.Serial(cb.Dup(), cb.Dup())
    input_signature = ShapeDtype((3, 2))
    expected_shape = ((3, 2), (3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_with_side_outputs_div_div(self):
    def some_layer():
      return cb.Parallel(divide_by(2.0), divide_by(5.0))
    layer = cb.SerialWithSideOutputs([some_layer(), some_layer()])
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 2)),
                       ShapeDtype((5, 2)))
    expected_shape = ((3, 2), (4, 2), (5, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_custom_name(self):
    layer = cb.Serial(cb.Dup(), cb.Dup())
    self.assertIn('Serial', str(layer))

    layer = cb.Serial(cb.Dup(), cb.Dup(), name='Branch')
    self.assertIn('Branch', str(layer))

  def test_branch_noop_dup(self):
    layer = cb.Branch([], cb.Dup())
    input_signature = ShapeDtype((3, 2))
    expected_shape = ((3, 2), (3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_add_div(self):
    layer = cb.Branch(cb.Add(), divide_by(0.5))
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((3, 2)))
    expected_shape = ((3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_one_layer(self):
    layer = cb.Branch(divide_by(0.5))
    input_signature = ShapeDtype((3, 2))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_name(self):
    layer = cb.Branch(cb.Add(), divide_by(0.5))
    self.assertIn('Branch', str(layer))

  def test_select_computes_n_in(self):
    layer = cb.Select([0, 0])
    self.assertEqual(layer.n_in, 1)
    layer = cb.Select([1, 0])
    self.assertEqual(layer.n_in, 2)
    layer = cb.Select([2])
    self.assertEqual(layer.n_in, 3)

  def test_select_given_n_in(self):
    layer = cb.Select([0], n_in=2)
    self.assertEqual(layer.n_in, 2)
    layer = cb.Select([0], n_in=3)
    self.assertEqual(layer.n_in, 3)

  def test_select_first_of_3(self):
    layer = cb.Select([0], n_in=3)
    input_signature = (
        ShapeDtype((3, 2)), ShapeDtype((4, 7)), ShapeDtype((11, 13)))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_select_second_of_3(self):
    layer = cb.Select([1], n_in=3)
    input_signature = (
        ShapeDtype((3, 2)), ShapeDtype((4, 7)), ShapeDtype((11, 13)))
    expected_shape = (4, 7)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_dup_dup(self):
    layer = cb.Parallel(cb.Dup(), cb.Dup())
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (3, 2), (4, 7), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_div_div(self):
    layer = cb.Parallel(divide_by(0.5), divide_by(3.0))
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_no_ops(self):
    layer = cb.Parallel([], None)
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_custom_name(self):
    layer = cb.Parallel(cb.Dup(), cb.Dup())  # pylint: disable=no-value-for-parameter
    self.assertIn('Parallel', str(layer))

    layer = cb.Parallel(cb.Dup(), cb.Dup(), name='DupDup')  # pylint: disable=no-value-for-parameter
    self.assertIn('DupDup', str(layer))

  def test_concatenate(self):
    x0 = np.array([[1, 2, 3],
                   [4, 5, 6]])
    x1 = np.array([[10, 20, 30],
                   [40, 50, 60]])

    layer0 = cb.Concatenate(axis=0)
    y = layer0([x0, x1])
    self.assertEqual(y.tolist(), [[1, 2, 3],
                                  [4, 5, 6],
                                  [10, 20, 30],
                                  [40, 50, 60]])

    layer1 = cb.Concatenate(axis=1)
    y = layer1([x0, x1])
    self.assertEqual(y.tolist(), [[1, 2, 3, 10, 20, 30],
                                  [4, 5, 6, 40, 50, 60]])

    layer2 = cb.Concatenate(n_items=3)
    y = layer2([x0, x1, x0])
    self.assertEqual(y.tolist(), [[1, 2, 3, 10, 20, 30, 1, 2, 3],
                                  [4, 5, 6, 40, 50, 60, 4, 5, 6]])

    self.assertEqual(repr(layer0), 'Concatenate_axis0_in2')
    self.assertEqual(repr(layer1), 'Concatenate_axis1_in2')
    self.assertEqual(repr(layer2), 'Concatenate_axis-1_in3')

  def test_drop(self):
    layer = cb.Drop()
    input_signature = ShapeDtype((3, 2))
    expected_shape = ()
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_dup(self):
    layer = cb.Dup()
    input_signature = ShapeDtype((3, 2))
    expected_shape = ((3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_swap(self):
    layer = cb.Swap()
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((4, 7), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_scan_basic(self):
    def add():  # pylint: disable=invalid-name
      def f(x, carry):
        res = x + carry
        return res, res  # output and carry are the same
      return base.Fn('add', f, n_out=2)

    scan_layer = cb.Scan(add())
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((2, 7)))
    expected_shape = ((3, 2, 7), (2, 7))
    output_shape = base.check_shape_agreement(scan_layer, input_signature)
    self.assertEqual(output_shape, expected_shape)
    inp = (np.array([1, 2, 3]), np.array(0))
    o, v = scan_layer(inp)
    self.assertEqual(int(v), 6)
    self.assertEqual([int(x) for x in o], [1, 3, 6])

  def test_scan_axis1(self):
    def add():  # pylint: disable=invalid-name
      def f(x, carry):
        res = x + carry
        return res, res  # output and carry are the same
      return base.Fn('add', f, n_out=2)

    scan = cb.Scan(add(), axis=1)
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((3, 7)))
    expected_shape = ((3, 2, 7), (3, 7))
    output_shape = base.check_shape_agreement(scan, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_scan_multiinput(self):
    def foo():  # pylint: disable=invalid-name
      def f(a, b, carry):
        return a + b, b, carry + 1
      return base.Fn('foo', f, n_out=2)

    scan = cb.Scan(foo(), axis=1)
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((3, 2, 7)),
                       ShapeDtype((3, 7)))
    expected_shape = ((3, 2, 7), (3, 2, 7), (3, 7))
    output_shape = base.check_shape_agreement(scan, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_scan_nocarry(self):
    def addone():  # pylint: disable=invalid-name
      return base.Fn('addone', lambda x: x + 1)

    scan_layer = cb.Scan(addone(), n_carry=0)
    input_signature = ShapeDtype((3, 2, 7))
    expected_shape = (3, 2, 7)
    output_shape = base.check_shape_agreement(scan_layer, input_signature)
    self.assertEqual(output_shape, expected_shape)
    inp = np.array([1, 2, 3])
    o = scan_layer(inp)
    self.assertEqual([int(x) for x in o], [2, 3, 4])

  def test_input_signatures_serial(self):
    layer = cb.Serial(divide_by(2.0), divide_by(5.0))
    self.assertIsNone(layer.input_signature)

    layer._set_input_signature_recursive(ShapeDtype((3, 2)))
    self.assertEqual(layer.input_signature, ShapeDtype((3, 2)))
    self.assertLen(layer.sublayers, 2)
    for sublayer in layer.sublayers:
      self.assertEqual(sublayer.input_signature, ShapeDtype((3, 2)))

  def test_input_signatures_serial_batch_norm(self):
    # Include a layer that actively uses state.
    input_signature = ShapeDtype((3, 28, 28))
    batch_norm = normalization.BatchNorm()
    relu = activation_fns.Relu()
    batch_norm_and_relu = cb.Serial(batch_norm, relu)
    batch_norm_and_relu.init(input_signature)

    # Check for correct shapes entering and exiting the batch_norm layer.
    # And the code should run without errors.
    batch_norm_and_relu._set_input_signature_recursive(input_signature)
    self.assertEqual(batch_norm.input_signature, input_signature)
    self.assertEqual(relu.input_signature, input_signature)

  def test_input_signatures_parallel(self):
    layer = cb.Parallel(divide_by(0.5), divide_by(3.0))
    self.assertIsNone(layer.input_signature)

    layer._set_input_signature_recursive((ShapeDtype((3, 2)),
                                          ShapeDtype((4, 7))))
    self.assertEqual(layer.input_signature,
                     (ShapeDtype((3, 2)), ShapeDtype((4, 7))))
    self.assertLen(layer.sublayers, 2)
    sublayer_0, sublayer_1 = layer.sublayers
    self.assertEqual(sublayer_0.input_signature, ShapeDtype((3, 2)))
    self.assertEqual(sublayer_1.input_signature, ShapeDtype((4, 7)))

  def test_state_parallel(self):
    model = cb.Parallel(core.Dense(3), core.Dense(5))
    self.assertIsInstance(model.state, tuple)
    self.assertLen(model.state, 2)

  def test_state_serial(self):
    model = cb.Serial(core.Dense(4), core.Dense(5), core.Dense(7))
    self.assertIsInstance(model.state, tuple)
    self.assertLen(model.state, 3)

  def test_weights_parallel(self):
    model = cb.Parallel(core.Dense(3), core.Dense(5))
    self.assertIsInstance(model.weights, tuple)
    self.assertLen(model.weights, 2)

  def test_weights_serial(self):
    model = cb.Serial(core.Dense(4), core.Dense(5), core.Dense(7))
    self.assertIsInstance(model.weights, tuple)
    self.assertLen(model.weights, 3)

  def test_set_rng_serial_recurse_two_levels(self):
    dense_00 = core.Dense(2)
    dense_01 = core.Dense(2)
    dense_10 = core.Dense(2)
    dense_11 = core.Dense(2)
    layer = cb.Serial(
        cb.Serial(dense_00, dense_01),
        cb.Serial(dense_10, dense_11),
    )
    input_signature = ShapeDtype((1, 2))

    _, _ = layer.init(input_signature)
    weights = layer.weights
    dense_00_w, dense_00_b = weights[0][0]
    dense_01_w, dense_01_b = weights[0][1]
    dense_10_w, dense_10_b = weights[1][0]
    dense_11_w, dense_11_b = weights[1][1]

    # Setting rng's recursively during init should yield differing weights.
    self.assertFalse(np.array_equal(dense_00_w, dense_01_w))
    self.assertFalse(np.array_equal(dense_00_b, dense_01_b))
    self.assertFalse(np.array_equal(dense_10_w, dense_11_w))
    self.assertFalse(np.array_equal(dense_10_b, dense_11_b))


if __name__ == '__main__':
  absltest.main()
