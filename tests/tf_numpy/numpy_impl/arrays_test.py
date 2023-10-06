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

"""Tests for ndarray."""
from collections import abc

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import arrays

# Required for operator overloads
from trax.tf_numpy.numpy_impl import math_ops  # pylint: disable=unused-import


t2a = arrays.tensor_to_ndarray


class ArrayTest(tf.test.TestCase):
    def testDtype(self):
        a = t2a(tf.zeros(shape=[1, 2], dtype=tf.int64))
        self.assertIs(a.dtype.type, np.int64)
        self.assertAllEqual(0, a.dtype.type(0))

    def testAstype(self):
        a = t2a(tf.convert_to_tensor(value=1.1, dtype=tf.float32)).astype(np.int32)
        self.assertIs(a.dtype.type, np.int32)
        self.assertAllEqual(1, a)
        a = t2a(tf.convert_to_tensor(value=[0.0, 1.1], dtype=tf.float32)).astype(
            np.bool_
        )
        self.assertIs(a.dtype.type, np.bool_)
        self.assertAllEqual([False, True], a)

    def testNeg(self):
        a = t2a(tf.convert_to_tensor(value=[1.0, 2.0]))
        self.assertAllEqual([-1.0, -2.0], -a)

    def _testBinOp(self, a, b, out, f, types=None):
        a = t2a(tf.convert_to_tensor(value=a, dtype=np.int32))
        b = t2a(tf.convert_to_tensor(value=b, dtype=np.int32))
        if not isinstance(out, arrays.ndarray):
            out = t2a(tf.convert_to_tensor(value=out, dtype=np.int32))
        if types is None:
            types = [
                [np.int32, np.int32, np.int32],
                [np.int64, np.int32, np.int64],
                [np.int32, np.int64, np.int64],
                [np.float32, np.int32, np.float64],
                [np.int32, np.float32, np.float64],
                [np.float32, np.float32, np.float32],
                [np.float64, np.float32, np.float64],
                [np.float32, np.float64, np.float64],
            ]
        for a_type, b_type, out_type in types:
            o = f(a.astype(a_type), b.astype(b_type))
            self.assertIs(o.dtype.type, out_type)
            self.assertAllClose(out.astype(out_type), o)

    def testAdd(self):
        self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: a.__add__(b))

    def testRadd(self):
        self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: b.__radd__(a))

    def testSub(self):
        self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: a.__sub__(b))

    def testRsub(self):
        self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: b.__rsub__(a))

    def testMul(self):
        self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: a.__mul__(b))

    def testRmul(self):
        self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: b.__rmul__(a))

    def testPow(self):
        self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: a.__pow__(b))

    def testRpow(self):
        self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: b.__rpow__(a))

    _truediv_types = [
        [np.int32, np.int32, np.float64],
        [np.int64, np.int32, np.float64],
        [np.int32, np.int64, np.float64],
        [np.float32, np.int32, np.float64],
        [np.int32, np.float32, np.float64],
        [np.float32, np.float32, np.float32],
        [np.float64, np.float32, np.float64],
        [np.float32, np.float64, np.float64],
    ]

    def testTruediv(self):
        self._testBinOp(
            [3, 5],
            [2, 4],
            t2a(tf.convert_to_tensor(value=[1.5, 1.25])),
            lambda a, b: a.__truediv__(b),
            types=self._truediv_types,
        )

    def testRtruediv(self):
        self._testBinOp(
            [3, 5],
            [2, 4],
            t2a(tf.convert_to_tensor(value=[1.5, 1.25])),
            lambda a, b: b.__rtruediv__(a),
            types=self._truediv_types,
        )

    def _testCmp(self, a, b, out, f):
        a = t2a(tf.convert_to_tensor(value=a, dtype=np.int32))
        b = t2a(tf.convert_to_tensor(value=b, dtype=np.int32))
        types = [
            [np.int32, np.int32],
            [np.int64, np.int32],
            [np.int32, np.int64],
            [np.float32, np.int32],
            [np.int32, np.float32],
            [np.float32, np.float32],
            [np.float64, np.float32],
            [np.float32, np.float64],
        ]
        for a_type, b_type in types:
            o = f(a.astype(a_type), b.astype(b_type))
            self.assertAllEqual(out, o)

    def testLt(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [True, False, False], lambda a, b: a.__lt__(b)
        )

    def testLe(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [True, True, False], lambda a, b: a.__le__(b)
        )

    def testGt(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [False, False, True], lambda a, b: a.__gt__(b)
        )

    def testGe(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [False, True, True], lambda a, b: a.__ge__(b)
        )

    def testEq(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [False, True, False], lambda a, b: a.__eq__(b)
        )

    def testNe(self):
        self._testCmp(
            [1, 2, 3], [3, 2, 1], [True, False, True], lambda a, b: a.__ne__(b)
        )

    def testInt(self):
        v = 10
        u = int(t2a(tf.convert_to_tensor(value=v)))
        self.assertIsInstance(u, int)
        self.assertAllEqual(v, u)

    def testFloat(self):
        v = 21.32
        u = float(t2a(tf.convert_to_tensor(value=v)))
        self.assertIsInstance(u, float)
        self.assertAllClose(v, u)

    def testBool(self):
        b = bool(t2a(tf.convert_to_tensor(value=10)))
        self.assertIsInstance(b, bool)
        self.assertTrue(b)
        self.assertFalse(bool(t2a(tf.convert_to_tensor(value=0))))
        self.assertTrue(bool(t2a(tf.convert_to_tensor(value=0.1))))
        self.assertFalse(bool(t2a(tf.convert_to_tensor(value=0.0))))

    def testHash(self):
        a = t2a(tf.convert_to_tensor(value=10))
        self.assertNotIsInstance(a, abc.Hashable)
        with self.assertRaisesWithPredicateMatch(TypeError, r"unhashable type"):
            hash(a)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
