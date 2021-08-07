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
"""Tests for combinator layers."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from trax import fastmath
from trax import shapes
import trax.layers as tl


def DivideBy(val):  # pylint: disable=invalid-name
  """Returns a simple division layer with n_in == 1 and n_out == 1."""
  return tl.Fn('DivideBy', lambda x: x / val)


def ReturnConst(val):  # pylint: disable=invalid-name
  """Returns a simple const layer with n_in == 0 and n_out == 1."""
  return tl.Fn('ReturnConst', lambda: val)


def SmallerThan(val):  # pylint: disable=invalid-name
  """Checks if the input is smaller than certain value."""
  return tl.Fn('SmallerThan', lambda x: x < val)


# TODO(jonni): Consider a more generic home for this utiliity function.
def as_list(outputs):
  """Converts layer outputs to a nested list, for easier equality testing.

  Args:
    outputs: A tensor or tuple/list of tensors coming from the forward
        application of a layer. Each tensor is NumPy ndarray-like, which
        complicates simple equality testing (e.g., via `assertEquals`):
        such tensors require equality testing to use either `all` (all
        elements match) or `any` (at least one element matches), which is not
        directly supported in absltest.

  Returns:
    A nested list structure containing all the output values, but now directly
    testable using `assertEquals`.
  """
  if isinstance(outputs, (list, tuple)):
    return [as_list(y) for y in outputs]
  else:
    return outputs.tolist()


class SerialTest(absltest.TestCase):

  def test_none_is_no_op(self):
    layer = tl.Serial(None)
    xs = [np.array([1, 2, 3, 4]),
          np.array([10, 20, 30])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[1, 2, 3, 4],
                                   [10, 20, 30]])

  def test_empty_list_is_no_op(self):
    layer = tl.Serial([])
    xs = [np.array([1, 2, 3, 4]),
          np.array([10, 20, 30])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[1, 2, 3, 4],
                                   [10, 20, 30]])

  def test_one_in_one_out(self):
    layer = tl.Serial(DivideBy(3))
    x = np.array([3, 6, 9, 12])
    y = layer(x)
    self.assertEqual(as_list(y), [1, 2, 3, 4])

  def test_zero_in_one_out(self):
    layer = tl.Serial(ReturnConst(np.array([3, 4, 5, 6])))
    y = layer(())
    self.assertEqual(as_list(y), [3, 4, 5, 6])

  def test_one_in_two_out(self):
    layer = tl.Serial(DivideBy(3),
                      ReturnConst(np.array([3, 4, 5, 6])))
    x = np.array([3, 6, 9, 12])
    y = layer(x)
    self.assertEqual(as_list(y), [[3, 4, 5, 6],
                                  [1, 2, 3, 4]])

  def test_const_div(self):
    layer = tl.Serial(ReturnConst(np.array([3, 6, 9, 12])),
                      DivideBy(3))
    y = layer(())
    self.assertEqual(as_list(y), [1, 2, 3, 4])

  def test_div_div(self):
    layer = tl.Serial(DivideBy(2.0), DivideBy(5.0))
    x = np.array([10, 20, 30])
    y = layer(x)
    self.assertEqual(as_list(y), [1, 2, 3])

  def test_dup_dup(self):
    layer = tl.Serial(tl.Dup(), tl.Dup())
    x = np.array([1, 2, 3])
    ys = layer(x)
    self.assertEqual(as_list(ys), [[1, 2, 3],
                                   [1, 2, 3],
                                   [1, 2, 3]])

  def test_default_name(self):
    layer = tl.Serial(tl.Dup(), tl.Dup())
    self.assertIn('Serial', str(layer))

  def test_custom_name(self):
    layer = tl.Serial(tl.Dup(), tl.Dup(), name='Branch')
    self.assertIn('Branch', str(layer))

  def test_weights(self):
    model = tl.Serial(tl.Dense(4), tl.Dense(5), tl.Dense(7))
    self.assertIsInstance(model.weights, tuple)
    self.assertLen(model.weights, 3)

  def test_flat_weights_and_state(self):
    model = tl.Serial(tl.Dup(), tl.Dense(5), tl.Serial(tl.Dense(7), tl.Dup()))
    sample_input_signature = shapes.signature(np.zeros((2, 3)))
    model.init(sample_input_signature)
    flat_weights, flat_state = tl.flatten_weights_and_state(
        model.weights, model.state)
    # Model has 2 pairs of trainable weights: (w, b) for the 2 dense layers.
    # So after making them flat, there are 4 trainable weights.
    self.assertLen(flat_weights, 4)
    self.assertEmpty(flat_state)
    model2 = tl.Serial(tl.Dense(5), tl.Dup(), tl.Dense(7))
    sig = model2.weights_and_state_signature(sample_input_signature)
    weights2, state2 = tl.unflatten_weights_and_state(
        flat_weights, flat_state, sig)
    model2.weights = weights2
    model2.state = state2
    self.assertLen(model2.weights, 3)
    self.assertEqual(model.weights[1], model2.weights[0])
    self.assertEqual(model.weights[2][0], model2.weights[2])

  def test_flat_weights_and_state_shared(self):
    shared = tl.Dense(5)
    model = tl.Serial(tl.Dense(5), shared, tl.Serial(shared, tl.Dup()))
    sample_input_signature = shapes.signature(np.zeros((2, 3)))
    model.init(sample_input_signature)
    flat_weights, flat_state = tl.flatten_weights_and_state(
        model.weights, model.state)
    # Model has 2 pairs of trainable weights: (w, b) for the 2 dense layers.
    # So after making them flat, there are 4 trainable weights.
    self.assertLen(flat_weights, 4)
    self.assertEmpty(flat_state)
    model2 = tl.Serial(tl.Dense(5), tl.Dup(), tl.Dense(5))
    sig = model2.weights_and_state_signature(sample_input_signature)
    weights2, state2 = tl.unflatten_weights_and_state(
        flat_weights, flat_state, sig)
    model2.weights = weights2
    model2.state = state2
    self.assertLen(model2.weights, 3)
    self.assertEqual(model.weights[0], model2.weights[0])
    self.assertEqual(model.weights[1], model2.weights[2])

  def test_assign_sublayer_weights(self):
    layer = tl.Dense(5, use_bias=False)
    model = tl.Serial(tl.Serial(layer, tl.Dense(6)), tl.Dense(7))
    sample_input = np.array([1, 2, 3, 4, 5])
    weights, _ = model.init(shapes.signature(sample_input))
    new_layer_weights = np.random.uniform(weights[0][0].shape)
    layer.weights = new_layer_weights
    self.assertIs(model.weights[0][0], new_layer_weights)

  def test_shared_weights(self):
    layer = tl.Dense(5)
    model = tl.Serial(layer, layer)
    sample_input = np.array([1, 2, 3, 4, 5])
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1], tl.GET_WEIGHTS_FROM_CACHE)

  def test_shared_weights_nested(self):
    layer = tl.Dense(5)
    model = tl.Serial(layer, tl.Serial(layer))
    sample_input = np.array([1, 2, 3, 4, 5])
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1][0], tl.GET_WEIGHTS_FROM_CACHE)

  def test_shared_weights_double_nested(self):
    layer = tl.Dense(5)
    model = tl.Serial(tl.Serial(layer), tl.Serial(layer))
    sample_input = np.array([1, 2, 3, 4, 5])
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1][0], tl.GET_WEIGHTS_FROM_CACHE)

  def test_shared_weights_for_shared_serial(self):
    layer = tl.Serial(tl.Dense(5), tl.Dense(5))
    model = tl.Serial(layer, layer)
    sample_input = np.array([1, 2, 3, 4, 5])
    # Init gives weights reflecting weight sharing.
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIsNot(weights[0], tl.GET_WEIGHTS_FROM_CACHE)
    self.assertIs(weights[1], tl.GET_WEIGHTS_FROM_CACHE)
    # Forward pass runs successfully.
    y = model(sample_input)
    self.assertEqual(y.shape, (5,))

  def test_state(self):
    model = tl.Serial(tl.Dense(4), tl.Dense(5), tl.Dense(7))
    self.assertIsInstance(model.state, tuple)
    self.assertLen(model.state, 3)

  def test_set_rng_recurse_two_levels(self):
    dense_00 = tl.Dense(2)
    dense_01 = tl.Dense(2)
    dense_10 = tl.Dense(2)
    dense_11 = tl.Dense(2)
    layer = tl.Serial(
        tl.Serial(dense_00, dense_01),
        tl.Serial(dense_10, dense_11),
    )
    input_signature = shapes.ShapeDtype((1, 2))

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


class ParallelTest(absltest.TestCase):

  def test_dup_dup(self):
    layer = tl.Parallel(tl.Dup(), tl.Dup())
    xs = [np.array([1, 2, 3]),
          np.array([10, 20])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[1, 2, 3],
                                   [1, 2, 3],
                                   [10, 20],
                                   [10, 20]])

  def test_div_div(self):
    layer = tl.Parallel(DivideBy(0.5), DivideBy(3.0))
    xs = [np.array([1, 2, 3]),
          np.array([30, 60])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[2, 4, 6],
                                   [10, 20]])

  def test_two_no_ops(self):
    layer = tl.Parallel([], None)
    xs = [np.array([1, 2, 3]),
          np.array([10, 20])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[1, 2, 3],
                                   [10, 20]])

  def test_default_name(self):
    layer = tl.Parallel(tl.Dup(), tl.Dup())
    self.assertIn('Parallel', str(layer))

  def test_custom_name(self):
    layer = tl.Parallel(tl.Dup(), tl.Dup(), name='DupDup')
    self.assertIn('DupDup', str(layer))

  def test_weights(self):
    model = tl.Parallel(tl.Dense(3), tl.Dense(5))
    self.assertIsInstance(model.weights, tuple)
    self.assertLen(model.weights, 2)

  def test_shared_weights(self):
    layer = tl.Dense(5)
    model = tl.Parallel(layer, layer)
    sample_input = (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1], tl.GET_WEIGHTS_FROM_CACHE)

  def test_shared_weights_nested(self):
    layer = tl.Dense(5)
    model = tl.Parallel([layer, tl.Dense(2)],
                        [layer, tl.Dense(2)])
    sample_input = (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1][0], tl.GET_WEIGHTS_FROM_CACHE)

  def test_shared_weights_for_shared_parallel(self):
    layer = tl.Parallel(tl.Dense(5), tl.Dense(7))
    model = tl.Parallel(layer, layer)
    sample_input = [
        np.array([1, 2, 3]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300]),
        np.array([1000, 2000, 3000]),
    ]
    # Init gives weights reflecting weight sharing.
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIsNot(weights[0], tl.GET_WEIGHTS_FROM_CACHE)
    self.assertIs(weights[1], tl.GET_WEIGHTS_FROM_CACHE)
    # Forward pass runs successfully.
    y0, y1, y2, y3 = model(sample_input)
    self.assertEqual(y0.shape, (5,))
    self.assertEqual(y1.shape, (7,))
    self.assertEqual(y2.shape, (5,))
    self.assertEqual(y3.shape, (7,))

  def test_state(self):
    model = tl.Parallel(tl.Dense(3), tl.Dense(5))
    self.assertIsInstance(model.state, tuple)
    self.assertLen(model.state, 2)


class ConcatenateTest(absltest.TestCase):

  def test_n_in_n_out(self):
    layer = tl.Concatenate()
    self.assertEqual(layer.n_in, 2)
    self.assertEqual(layer.n_out, 1)

  def test_with_defaults(self):
    layer = tl.Concatenate()  # Default n_items=2, axis=-1
    xs = [np.array([[1, 2, 3],
                    [4, 5, 6]]),
          np.array([[10, 20, 30],
                    [40, 50, 60]])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[1, 2, 3, 10, 20, 30],
                                   [4, 5, 6, 40, 50, 60]])

  def test_axis_0(self):
    layer = tl.Concatenate(axis=0)
    xs = [np.array([[1, 2, 3],
                    [4, 5, 6]]),
          np.array([[10, 20, 30],
                    [40, 50, 60]])]
    y = layer(xs)
    self.assertEqual(as_list(y), [[1, 2, 3],
                                  [4, 5, 6],
                                  [10, 20, 30],
                                  [40, 50, 60]])

  def test_axis_1(self):
    layer = tl.Concatenate(axis=1)
    xs = [np.array([[1, 2, 3],
                    [4, 5, 6]]),
          np.array([[10, 20, 30],
                    [40, 50, 60]])]
    y = layer(xs)
    self.assertEqual(as_list(y), [[1, 2, 3, 10, 20, 30],
                                  [4, 5, 6, 40, 50, 60]])

  def test_n_items_is_not_default(self):
    layer = tl.Concatenate(n_items=3)
    xs = [np.array([[1, 2, 3],
                    [4, 5, 6]]),
          np.array([[10, 20, 30],
                    [40, 50, 60]]),
          np.array([[100, 200, 300],
                    [400, 500, 600]])]
    y = layer(xs)
    self.assertEqual(y.shape, (2, 9))
    self.assertEqual(as_list(y), [[1, 2, 3, 10, 20, 30, 100, 200, 300],
                                  [4, 5, 6, 40, 50, 60, 400, 500, 600]])

  def test_repr(self):
    layer = tl.Concatenate()
    self.assertEqual(repr(layer), 'Concatenate_in2')

    layer = tl.Concatenate(axis=0)
    self.assertEqual(repr(layer), 'Concatenate_axis0_in2')

    layer = tl.Concatenate(axis=1)
    self.assertEqual(repr(layer), 'Concatenate_axis1_in2')

    layer = tl.Concatenate(n_items=3)
    self.assertEqual(repr(layer), 'Concatenate_in3')


class BranchTest(absltest.TestCase):

  def test_noop_dup(self):
    layer = tl.Branch([], tl.Dup())
    x = np.array([1, 2, 3])
    ys = layer(x)
    self.assertEqual(as_list(ys), [[1, 2, 3],
                                   [1, 2, 3],
                                   [1, 2, 3]])

  def test_add_div(self):
    layer = tl.Branch(tl.Add(), DivideBy(0.5))
    xs = [np.array([1, 2, 3]),
          np.array([10, 20, 30])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[11, 22, 33],
                                   [2, 4, 6]])

  def test_one_sublayer(self):
    layer = tl.Branch(DivideBy(0.5))
    x = np.array([1, 2, 3])
    ys = layer(x)
    self.assertEqual(as_list(ys), [2, 4, 6])

  def test_default_name(self):
    layer = tl.Branch(tl.Add(), DivideBy(0.5))
    self.assertIn('Branch', str(layer))

  def test_printing_sublayers(self):
    layer = tl.Branch(tl.Add(), tl.Add())
    expected_result = 'Branch_in2_out2[\n  Add_in2\n  Add_in2\n]'
    self.assertEqual(expected_result, str(layer))


class SelectTest(absltest.TestCase):

  def test_computes_n_in(self):
    layer = tl.Select([0, 0])
    self.assertEqual(layer.n_in, 1)

    layer = tl.Select([1, 0])
    self.assertEqual(layer.n_in, 2)

    layer = tl.Select([2])
    self.assertEqual(layer.n_in, 3)

  def test_given_n_in(self):
    layer = tl.Select([0], n_in=2)
    self.assertEqual(layer.n_in, 2)

    layer = tl.Select([0], n_in=3)
    self.assertEqual(layer.n_in, 3)

  def test_first_of_3(self):
    layer = tl.Select([0], n_in=3)
    xs = [np.array([1, 2, 3]),
          np.array([10, 20]),
          np.array([100])]
    y = layer(xs)
    self.assertEqual(as_list(y), [1, 2, 3])

  def test_second_of_3(self):
    layer = tl.Select([1], n_in=3)
    xs = [np.array([1, 2, 3]),
          np.array([10, 20]),
          np.array([100])]
    y = layer(xs)
    self.assertEqual(as_list(y), [10, 20])


class DropTest(absltest.TestCase):

  def test_drop(self):
    layer = tl.Drop()
    x = np.array([1, 2, 3])
    y = layer(x)
    self.assertEqual(as_list(y), [])


class SwapTest(absltest.TestCase):

  def test_swap(self):
    layer = tl.Swap()
    xs = [np.array([1, 2, 3]),
          np.array([10, 20, 30])]
    ys = layer(xs)
    self.assertEqual(as_list(ys), [[10, 20, 30],
                                   [1, 2, 3]])


class ChunkTest(absltest.TestCase):

  def test_chunk(self):
    layer = tl.Dense(4)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    layer.init(x)
    y = layer(x)
    z = tl.Chunk(layer, 1)(x)
    self.assertLess(np.sum((y - z)**2), 1e-5)  # y == z upto numerics

  def test_chunk_uneven_numbers(self):
    layer = tl.Dense(4)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    layer.init(x)
    y = layer(x)
    z = tl.Chunk(layer, 3)(x)  # By default it should just pass
    self.assertLess(np.sum((y - z)**2), 1e-5)  # y == z upto numerics
    chunk_with_test = tl.Chunk(layer, 3, pass_unchunkable=False)
    self.assertRaises(tl.LayerError, lambda: chunk_with_test(x))


class SerialWithSideOutputsTest(absltest.TestCase):

  def test_serial_with_side_outputs_div_div(self):
    def some_layer():
      return tl.Parallel(DivideBy(2.0), DivideBy(5.0))
    layer = tl.SerialWithSideOutputs([some_layer(), some_layer()])
    xs = (np.array([1, 2, 3]),
          np.array([10, 20, 30, 40, 50]),
          np.array([100, 200]))
    ys = layer(xs)
    output_shapes = [y.shape for y in ys]
    self.assertEqual(output_shapes, [(3,), (5,), (2,)])


BACKENDS = [fastmath.Backend.JAX]


@parameterized.named_parameters(
    ('_' + b.value, b) for b in BACKENDS)
class ScanTest(parameterized.TestCase):

  def _AddWithCarry(self):  # pylint: disable=invalid-name
    del self
    def f(x, carry):
      res = x + carry
      return res, res  # output and carry are the same
    return tl.Fn('AddWithCarry', f, n_out=2)

  def test_default_axis(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.Scan(self._AddWithCarry())
      xs = [
          np.array([[0, 1, 2, 3],
                    [0, 10, 20, 30],
                    [0, 100, 200, 300]]),
          np.array([9000, 8000, 7000, 6000])
      ]
      ys = layer(xs)
      self.assertEqual(as_list(ys),
                       [[[9000, 8001, 7002, 6003],
                         [9000, 8011, 7022, 6033],
                         [9000, 8111, 7222, 6333]
                        ],
                        [9000, 8111, 7222, 6333]
                       ])

  def test_axis_1(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.Scan(self._AddWithCarry(), axis=1)
      xs = [
          np.array([[0, 1, 2, 3],
                    [0, 10, 20, 30],
                    [0, 100, 200, 300]]),
          np.array([9000,
                    8000,
                    7000])
      ]
      ys = layer(xs)
      self.assertEqual(as_list(ys),
                       [[[9000, 9001, 9003, 9006],
                         [8000, 8010, 8030, 8060],
                         [7000, 7100, 7300, 7600]
                        ],
                        [9006,
                         8060,
                         7600]
                       ])

  def test_predict(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.Scan(self._AddWithCarry(), axis=1, mode='predict')
      xs = [np.array([[0, 1, 2]]),
            np.array([90])]
      ys = layer(xs)
      self.assertEqual(as_list(ys),
                       [[[90, 91, 93]],
                        [93]])
      xs = [np.array([[3, 4]]),
            np.array([90])]
      ys = layer(xs)
      self.assertEqual(as_list(ys),
                       [[[96, 100]],
                        [100]])

  def test_multi_input(self, backend):
    def _MultiInputFn():  # pylint: disable=invalid-name
      def f(a, b, carry):
        return a + b, b, carry + 1
      return tl.Fn('MultiInputFn', f, n_out=2)

    with fastmath.use_backend(backend):
      layer = tl.Scan(_MultiInputFn(), axis=1)
      xs = [
          np.array([[0, 1, 2],
                    [0, 10, 20]]),
          np.array([[4, 5, 6],
                    [40, 50, 60]]),
          np.array([9000,
                    8000])
      ]
      ys = layer(xs)
      self.assertEqual(as_list(ys),
                       [[[4, 6, 8],
                         [40, 60, 80]],
                        [[4, 5, 6],
                         [40, 50, 60]],
                        [9003,
                         8003]
                       ])

  def test_no_carry(self, backend):
    def _AddOne():  # pylint: disable=invalid-name
      return tl.Fn('AddOne', lambda x: x + 1)

    with fastmath.use_backend(backend):
      layer = tl.Scan(_AddOne(), n_carry=0)
      x = np.array([[1, 3, 7],
                    [10, 30, 70]])
      y = layer(x)
      self.assertEqual(as_list(y), [[2, 4, 8],
                                    [11, 31, 71]])


class CondTest(absltest.TestCase):

  def test_basic_true(self):
    cond = ReturnConst(True)
    true = ReturnConst([2])
    false = ReturnConst([5])
    layer = tl.Cond(cond, true, false)
    layer.init(())
    xs = tuple()
    ys = layer(xs)
    self.assertEqual(as_list(ys), 2)

  def test_basic_false(self):
    cond = ReturnConst(False)
    true = ReturnConst([2])
    false = ReturnConst([5])
    layer = tl.Cond(cond, true, false)
    layer.init(())
    xs = tuple()
    ys = layer(xs)
    self.assertEqual(as_list(ys), 5)

  def test_complex_blocks(self):
    cond = ReturnConst(True)
    true = DivideBy(2.)
    false = DivideBy(4.)
    layer = tl.Cond(cond, true, false)
    xs = [np.arange(5).astype(np.float32)]
    layer.init(shapes.signature(xs))
    ys = layer(xs)
    self.assertEqual(as_list(ys), [0., 0.5, 1.0, 1.5, 2.0])

  def test_condition_func_true(self):
    cond = SmallerThan(3.0)
    true = DivideBy(2.)
    false = DivideBy(4.)
    layer = tl.Cond(cond, true, false)
    xs = (np.array(2.), np.array([4., 12.]))
    layer.init(shapes.signature(xs))
    ys = layer(xs)
    self.assertEqual(as_list(ys), [2., 6.])

  def test_condition_func_false(self):
    cond = SmallerThan(3.0)
    true = DivideBy(2.)
    false = DivideBy(4.)
    layer = tl.Cond(cond, true, false)
    xs = (np.array(4.), np.array([4., 12.]))
    layer.init(shapes.signature(xs))
    ys = layer(xs)
    self.assertEqual(as_list(ys), [1., 3.])

  def test_condition_func_default_false(self):
    cond = SmallerThan(3.0)
    true = DivideBy(2.)
    layer = tl.Cond(cond, true)
    xs = (np.array(4.), np.array([4., 12.]))
    layer.init(shapes.signature(xs))
    ys = layer(xs)
    self.assertEqual(as_list(ys), [4., 12.])

  def test_exception_n_out(self):
    cond = SmallerThan(3.0)
    true = DivideBy(2.)
    false = tl.Dup()
    self.assertRaises(ValueError, lambda: tl.Cond(cond, true, false))

  def test_exception_n_in(self):
    cond = SmallerThan(3.0)
    true = ReturnConst(2.)
    false = DivideBy(2.)
    self.assertRaises(ValueError, lambda: tl.Cond(cond, true, false))

  def test_exception_run1(self):
    # We expect exactly one input.
    cond = SmallerThan(3.0)
    true = ReturnConst(2.)
    false = ReturnConst(5.)
    def init_and_run(layer, xs):
      layer.init(shapes.signature(xs))
      layer(xs)
    # It will pass with one input.
    xs = np.array(4.)
    layer = tl.Cond(cond, true, false)
    init_and_run(layer, xs)
    # It will fail with zero or two inputs.
    for xs in ((), (np.array(4.), np.array([4., 12.]))):
      layer = tl.Cond(cond, true, false)
      # pylint: disable=cell-var-from-loop
      self.assertRaises(Exception, lambda: init_and_run(layer, xs))

  def test_exception_run2(self):
    # We expect exactly two inputs.
    cond = SmallerThan(3.0)
    true = DivideBy(2.)
    false = DivideBy(5.)
    def init_and_run(layer, xs):
      layer.init(shapes.signature(xs))
      layer(xs)
    # It will pass with two inputs.
    xs = (np.array(4.), np.array([4., 12.]))
    layer = tl.Cond(cond, true, false)
    init_and_run(layer, xs)
    # It will fail with zero or one input.
    for xs in ((), (np.array(4.))):
      # pylint: disable=cell-var-from-loop
      self.assertRaises(Exception, lambda: init_and_run(layer, xs))

  def test_weights_and_state(self):
    cond = SmallerThan(3.0)
    true = tl.Dense(5)
    false = tl.Dense(5)
    different = tl.Dense(5)
    layer = tl.Cond(cond, true, false)
    xs = (np.array(2.), np.array([0., 1., 2.]))
    layer.init(shapes.signature(xs))

    # weights
    self.assertEqual(as_list(layer.weights),
                     as_list((cond.weights, true.weights, false.weights)))
    self.assertNotEqual(as_list(true.weights), as_list(false.weights))
    self.assertNotEqual(as_list(true.weights), as_list(different.weights))

    false.weights = true.weights
    self.assertEqual(as_list(layer.weights),
                     as_list((cond.weights, true.weights, true.weights)))

    layer.weights = (cond.weights, true.weights, different.weights)
    self.assertEqual(as_list(layer.weights),
                     as_list((cond.weights, true.weights, different.weights)))
    # state
    self.assertEqual(as_list(layer.state),
                     as_list((cond.state, true.state, false.state)))
    # just check if simple assignments (setter from base.Layer) work correctly
    # with Cond.init_weights_and_state ; all states are empty so there is no
    # point in checking equality
    false.state = true.state
    layer.state = (cond.state, true.state, different.state)


class BatchLeadingAxesTest(absltest.TestCase):

  def _Id3Dim(self):  # pylint: disable=invalid-name
    del self
    def f(x):
      assert len(x.shape) == 3
      return x
    return tl.Fn('Id3Dim', f, n_out=1)

  def test_2axes(self):
    layer = tl.BatchLeadingAxes(self._Id3Dim(), n_last_axes_to_keep=2)
    ys = layer(np.zeros((3, 4, 5)))
    self.assertEqual(ys.shape, (3, 4, 5))
    ys = layer(np.zeros((2, 3, 4, 5)))
    self.assertEqual(ys.shape, (2, 3, 4, 5))
    ys = layer(np.zeros((1, 2, 3, 4, 5)))
    self.assertEqual(ys.shape, (1, 2, 3, 4, 5))


class BidirectionalTest(absltest.TestCase):

  def test_dimensionality(self):
    x = np.ones((2, 3, 8))
    layer = tl.Bidirectional(tl.GRU(n_units=8))
    input_signature = shapes.signature(x)
    _, _ = layer.init(input_signature)
    yhat = layer(x)

    self.assertEqual(yhat.shape, (2, 3, 8 + 8))


if __name__ == '__main__':
  absltest.main()
