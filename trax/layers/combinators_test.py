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

"""Tests for combinator layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from trax.backend import numpy as np
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import normalization
from trax.shapes import ShapeDtype


class CombinatorLayerTest(absltest.TestCase):

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
    layer = cb.Serial(core.Div(divisor=2.0))
    input_signature = ShapeDtype((3, 2))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_div_div(self):
    layer = cb.Serial(core.Div(divisor=2.0), core.Div(divisor=5.0))
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
      return cb.Parallel(core.Div(divisor=2.0), core.Div(divisor=5.0))
    layer = cb.SerialWithSideOutputs([some_layer(), some_layer()])
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 2)),
                       ShapeDtype((5, 2)))
    expected_shape = ((3, 2), (4, 2), (5, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_dup_dup(self):
    layer = cb.Parallel(cb.Dup(), cb.Dup())
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((4, 7)))
    expected_shape = ((3, 2), (3, 2), (4, 7), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_div_div(self):
    layer = cb.Parallel(core.Div(divisor=0.5), core.Div(divisor=3.0))
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

  def test_branch_noop_dup(self):
    layer = cb.Branch([], cb.Dup())
    input_signature = ShapeDtype((3, 2))
    expected_shape = ((3, 2), (3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_add_div(self):
    layer = cb.Branch(cb.Add(), core.Div(divisor=0.5))
    input_signature = (ShapeDtype((3, 2)), ShapeDtype((3, 2)))
    expected_shape = ((3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_scan_basic(self):
    @base.layer(n_in=2, n_out=2)
    def add(x, **unused_kwargs):
      res = x[0] + x[1]
      return res, res
    scan_layer = cb.Scan(add())  # pylint: disable=no-value-for-parameter
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((2, 7)))
    expected_shape = ((3, 2, 7), (2, 7))
    output_shape = base.check_shape_agreement(scan_layer, input_signature)
    self.assertEqual(output_shape, expected_shape)
    inp = (np.array([1, 2, 3]), np.array(0))
    o, v = scan_layer(inp)
    self.assertEqual(int(v), 6)
    self.assertEqual([int(x) for x in o], [1, 3, 6])

  def test_scan_axis1(self):
    @base.layer(n_in=2, n_out=2)
    def add(x, **unused_kwargs):
      res = x[0] + x[1]
      return res, res
    scan = cb.Scan(add(), axis=1)  # pylint: disable=no-value-for-parameter
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((3, 7)))
    expected_shape = ((3, 2, 7), (3, 7))
    output_shape = base.check_shape_agreement(scan, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_scan_multiinput(self):
    @base.layer(n_in=3, n_out=2)
    def foo(x, **unused_kwargs):
      a, b, carry = x
      return a + b, b, carry + 1
    scan = cb.Scan(foo(), axis=1)  # pylint: disable=no-value-for-parameter
    input_signature = (ShapeDtype((3, 2, 7)), ShapeDtype((3, 2, 7)),
                       ShapeDtype((3, 7)))
    expected_shape = ((3, 2, 7), (3, 2, 7), (3, 7))
    output_shape = base.check_shape_agreement(scan, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_fn_layer_example(self):
    layer = cb.Fn(lambda x, y: (x + y, np.concatenate([x, y], axis=0)))
    input_signature = (ShapeDtype((2, 7)), ShapeDtype((2, 7)))
    expected_shape = ((2, 7), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)
    inp = (np.array([2]), np.array([3]))
    x, xs = layer(inp)
    self.assertEqual(int(x), 5)
    self.assertEqual([int(y) for y in xs], [2, 3])

  def test_fn_layer_fails_wrong_f(self):
    with self.assertRaisesRegexp(ValueError, 'default arg'):
      cb.Fn(lambda x, sth=None: x)
    with self.assertRaisesRegexp(ValueError, 'keyword arg'):
      cb.Fn(lambda x, **kwargs: x)

  def test_fn_layer_varargs_n_in(self):
    with self.assertRaisesRegexp(ValueError, 'variable arg'):
      cb.Fn(lambda *args: args[0])
    # Check that varargs work when n_in is set.
    id_layer = cb.Fn(lambda *args: args[0], n_in=1)
    input_signature = ShapeDtype((2, 7))
    expected_shape = (2, 7)
    output_shape = base.check_shape_agreement(id_layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_fn_layer_difficult_n_out(self):
    with self.assertRaisesRegexp(ValueError, 'n_out'):
      # Determining the output of this layer is hard with dummies.
      cb.Fn(lambda x: np.concatencate([x, x], axis=4))
    # Check that this layer works when n_out is set.
    layer = cb.Fn(lambda x: np.concatenate([x, x], axis=4), n_out=1)
    input_signature = ShapeDtype((2, 1, 2, 2, 3))
    expected_shape = (2, 1, 2, 2, 6)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_input_signatures_serial(self):
    layer = cb.Serial(core.Div(divisor=2.0), core.Div(divisor=5.0))
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
    relu = core.Relu()
    batch_norm_and_relu = cb.Serial(batch_norm, relu)
    batch_norm_and_relu.init(input_signature)

    # Check for correct shapes entering and exiting the batch_norm layer.
    # And the code should run without errors.
    batch_norm_and_relu._set_input_signature_recursive(input_signature)
    self.assertEqual(batch_norm.input_signature, input_signature)
    self.assertEqual(relu.input_signature, input_signature)

  def test_input_signatures_parallel(self):
    layer = cb.Parallel(core.Div(divisor=0.5), core.Div(divisor=3.0))
    self.assertIsNone(layer.input_signature)

    layer._set_input_signature_recursive((ShapeDtype((3, 2)),
                                          ShapeDtype((4, 7))))
    self.assertEqual(layer.input_signature,
                     (ShapeDtype((3, 2)), ShapeDtype((4, 7))))
    self.assertLen(layer.sublayers, 2)
    sublayer_0, sublayer_1 = layer.sublayers
    self.assertEqual(sublayer_0.input_signature, ShapeDtype((3, 2)))
    self.assertEqual(sublayer_1.input_signature, ShapeDtype((4, 7)))


if __name__ == '__main__':
  absltest.main()
