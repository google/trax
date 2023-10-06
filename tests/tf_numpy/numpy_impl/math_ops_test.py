# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for tf numpy mathematical methods."""
import itertools
from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import array_ops
from trax.tf_numpy.numpy_impl import arrays
from trax.tf_numpy.numpy_impl import math_ops


class MathTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.array_transforms = [
            lambda x: x,  # Identity,
            tf.convert_to_tensor,
            np.array,
            lambda x: np.array(x, dtype=np.float32),
            lambda x: np.array(x, dtype=np.float64),
            array_ops.array,
            lambda x: array_ops.array(x, dtype=np.float32),
            lambda x: array_ops.array(x, dtype=np.float64),
        ]
        self.types = [np.int32, np.int64, np.float32, np.float64]

    def _testBinaryOp(
        self,
        math_fun,
        np_fun,
        name,
        operands=None,
        extra_operands=None,
        check_promotion=True,
        check_promotion_result_type=True,
    ):
        def run_test(a, b):
            for fn in self.array_transforms:
                arg1 = fn(a)
                arg2 = fn(b)
                self.match(
                    math_fun(arg1, arg2),
                    np_fun(arg1, arg2),
                    msg="{}({}, {})".format(name, arg1, arg2),
                )
            # Tests type promotion
            for type_a in self.types:
                for type_b in self.types:
                    if not check_promotion and type_a != type_b:
                        continue
                    arg1 = array_ops.array(a, dtype=type_a)
                    arg2 = array_ops.array(b, dtype=type_b)
                    self.match(
                        math_fun(arg1, arg2),
                        np_fun(arg1, arg2),
                        msg="{}({}, {})".format(name, arg1, arg2),
                        check_dtype=check_promotion_result_type,
                    )

        if operands is None:
            operands = [
                (5, 2),
                (5, [2, 3]),
                (5, [[2, 3], [6, 7]]),
                ([1, 2, 3], 7),
                ([1, 2, 3], [5, 6, 7]),
            ]
        for operand1, operand2 in operands:
            run_test(operand1, operand2)
        if extra_operands is not None:
            for operand1, operand2 in extra_operands:
                run_test(operand1, operand2)

    def testDot(self):
        extra_operands = [
            ([1, 2], [[5, 6, 7], [8, 9, 10]]),
            (
                np.arange(2 * 3 * 5).reshape([2, 3, 5]).tolist(),
                np.arange(5 * 7 * 11).reshape([7, 5, 11]).tolist(),
            ),
        ]
        return self._testBinaryOp(
            math_ops.dot, np.dot, "dot", extra_operands=extra_operands
        )

    def testMinimum(self):
        # The numpy version has strange result type when promotion happens,
        # so set check_promotion_result_type to False.
        return self._testBinaryOp(
            math_ops.minimum, np.minimum, "minimum", check_promotion_result_type=False
        )

    def testMaximum(self):
        # The numpy version has strange result type when promotion happens,
        # so set check_promotion_result_type to False.
        return self._testBinaryOp(
            math_ops.maximum, np.maximum, "maximum", check_promotion_result_type=False
        )

    def testMatmul(self):
        operands = [([[1, 2]], [[3, 4, 5], [6, 7, 8]])]
        return self._testBinaryOp(
            math_ops.matmul, np.matmul, "matmul", operands=operands
        )

    def testMatmulError(self):
        with self.assertRaisesRegex(ValueError, r""):
            math_ops.matmul(
                array_ops.ones([], np.int32), array_ops.ones([2, 3], np.int32)
            )
        with self.assertRaisesRegex(ValueError, r""):
            math_ops.matmul(
                array_ops.ones([2, 3], np.int32), array_ops.ones([], np.int32)
            )

    def _testUnaryOp(self, math_fun, np_fun, name):
        def run_test(a):
            for fn in self.array_transforms:
                arg1 = fn(a)
                self.match(
                    math_fun(arg1), np_fun(arg1), msg="{}({})".format(name, arg1)
                )

        run_test(5)
        run_test([2, 3])
        run_test([[2, -3], [-6, 7]])

    def testLog(self):
        self._testUnaryOp(math_ops.log, np.log, "log")

    def testExp(self):
        self._testUnaryOp(math_ops.exp, np.exp, "exp")

    def testTanh(self):
        self._testUnaryOp(math_ops.tanh, np.tanh, "tanh")

    def testSqrt(self):
        self._testUnaryOp(math_ops.sqrt, np.sqrt, "sqrt")

    def match(self, actual, expected, msg="", check_dtype=True):
        self.assertIsInstance(actual, arrays.ndarray)
        if check_dtype:
            self.assertEqual(
                actual.dtype,
                expected.dtype,
                "Dtype mismatch.\nActual: {}\nExpected: {}\n{}".format(
                    actual.dtype, expected.dtype, msg
                ),
            )
        self.assertEqual(
            actual.shape,
            expected.shape,
            "Shape mismatch.\nActual: {}\nExpected: {}\n{}".format(
                actual.shape, expected.shape, msg
            ),
        )
        self.assertAllClose(actual.tolist(), expected.tolist())

    def testArgsort(self):
        self._testUnaryOp(math_ops.argsort, np.argsort, "argsort")

        # Test stability
        r = np.arange(100)
        a = np.zeros(100)
        np.testing.assert_equal(math_ops.argsort(a, kind="stable"), r)

    def testArgMaxArgMin(self):
        data = [
            0,
            5,
            [1],
            [1, 2, 3],
            [[1, 2, 3]],
            [[4, 6], [7, 8]],
            [[[4, 6], [9, 10]], [[7, 8], [12, 34]]],
        ]
        for fn, d in itertools.product(self.array_transforms, data):
            arr = fn(d)
            self.match(math_ops.argmax(arr), np.argmax(arr))
            self.match(math_ops.argmin(arr), np.argmin(arr))
            if hasattr(arr, "shape"):
                ndims = len(arr.shape)
            else:
                ndims = array_ops.array(arr, copy=False).ndim
            if ndims == 0:
                # Numpy flattens the scalar ndarray and treats it as a 1-d array of
                # size 1.
                ndims = 1
            for axis in range(-ndims, ndims):
                self.match(math_ops.argmax(arr, axis=axis), np.argmax(arr, axis=axis))
                self.match(math_ops.argmin(arr, axis=axis), np.argmin(arr, axis=axis))

    @parameterized.parameters([False, True])
    def testIsCloseEqualNan(self, equal_nan):
        a = np.asarray([1, 1, np.nan, 1, np.nan], np.float32)
        b = np.asarray([1, 2, 1, np.nan, np.nan], np.float32)
        self.match(
            math_ops.isclose(a, b, equal_nan=equal_nan),
            np.isclose(a, b, equal_nan=equal_nan),
        )

    def testAverageWrongShape(self):
        with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError, r""):
            math_ops.average(np.ones([2, 3]), weights=np.ones([2, 4]))
        with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError, r""):
            math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([2, 4]))
        with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError, r""):
            math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([]))
        with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError, r""):
            math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([5]))

    def testClip(self):
        def run_test(arr, *args, **kwargs):
            check_dtype = kwargs.pop("check_dtype", True)
            for fn in self.array_transforms:
                arr = fn(arr)
                self.match(
                    math_ops.clip(arr, *args, **kwargs),
                    np.clip(arr, *args, **kwargs),
                    check_dtype=check_dtype,
                )

        # NumPy exhibits weird typing behavior when a/a_min/a_max are scalars v/s
        # lists, e.g.,
        #
        # np.clip(np.array(0, dtype=np.int32), -5, 5).dtype == np.int64
        # np.clip(np.array([0], dtype=np.int32), -5, 5).dtype == np.int32
        # np.clip(np.array([0], dtype=np.int32), [-5], [5]).dtype == np.int64
        #
        # So we skip matching type. In tf-numpy the type of the output array is
        # always the same as the input array.
        run_test(0, -1, 5, check_dtype=False)
        run_test(-1, -1, 5, check_dtype=False)
        run_test(5, -1, 5, check_dtype=False)
        run_test(-10, -1, 5, check_dtype=False)
        run_test(10, -1, 5, check_dtype=False)
        run_test(10, None, 5, check_dtype=False)
        run_test(10, -1, None, check_dtype=False)
        run_test([0, 20, -5, 4], -1, 5, check_dtype=False)
        run_test([0, 20, -5, 4], None, 5, check_dtype=False)
        run_test([0, 20, -5, 4], -1, None, check_dtype=False)
        run_test([0.5, 20.2, -5.7, 4.4], -1.5, 5.1, check_dtype=False)

        run_test([0, 20, -5, 4], [-5, 0, -5, 0], [0, 5, 0, 5], check_dtype=False)
        run_test([[1, 2, 3], [4, 5, 6]], [2, 0, 2], 5, check_dtype=False)
        run_test([[1, 2, 3], [4, 5, 6]], 0, [5, 3, 1], check_dtype=False)

    def testPtp(self):
        def run_test(arr, *args, **kwargs):
            for fn in self.array_transforms:
                arg = fn(arr)
                self.match(
                    math_ops.ptp(arg, *args, **kwargs), np.ptp(arg, *args, **kwargs)
                )

        run_test([1, 2, 3])
        run_test([1.0, 2.0, 3.0])
        run_test([[1, 2], [3, 4]], axis=1)
        run_test([[1, 2], [3, 4]], axis=0)
        run_test([[1, 2], [3, 4]], axis=-1)
        run_test([[1, 2], [3, 4]], axis=-2)

    def testLinSpace(self):
        array_transforms = [
            lambda x: x,  # Identity,
            tf.convert_to_tensor,
            np.array,
            lambda x: np.array(x, dtype=np.float32),
            lambda x: np.array(x, dtype=np.float64),
            array_ops.array,
            lambda x: array_ops.array(x, dtype=np.float32),
            lambda x: array_ops.array(x, dtype=np.float64),
        ]

        def run_test(start, stop, **kwargs):
            for fn1 in array_transforms:
                for fn2 in array_transforms:
                    arg1 = fn1(start)
                    arg2 = fn2(stop)
                    self.match(
                        math_ops.linspace(arg1, arg2, **kwargs),
                        np.linspace(arg1, arg2, **kwargs),
                        msg="linspace({}, {})".format(arg1, arg2),
                    )

        run_test(0, 1)
        run_test(0, 1, num=10)
        run_test(0, 1, endpoint=False)
        run_test(0, -1)
        run_test(0, -1, num=10)
        run_test(0, -1, endpoint=False)

    def testLogSpace(self):
        array_transforms = [
            lambda x: x,  # Identity,
            tf.convert_to_tensor,
            np.array,
            lambda x: np.array(x, dtype=np.float32),
            lambda x: np.array(x, dtype=np.float64),
            array_ops.array,
            lambda x: array_ops.array(x, dtype=np.float32),
            lambda x: array_ops.array(x, dtype=np.float64),
        ]

        def run_test(start, stop, **kwargs):
            for fn1 in array_transforms:
                for fn2 in array_transforms:
                    arg1 = fn1(start)
                    arg2 = fn2(stop)
                    self.match(
                        math_ops.logspace(arg1, arg2, **kwargs),
                        np.logspace(arg1, arg2, **kwargs),
                        msg="logspace({}, {})".format(arg1, arg2),
                    )

        run_test(0, 5)
        run_test(0, 5, num=10)
        run_test(0, 5, endpoint=False)
        run_test(0, 5, base=2.0)
        run_test(0, -5)
        run_test(0, -5, num=10)
        run_test(0, -5, endpoint=False)
        run_test(0, -5, base=2.0)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
