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

"""Tests for trax.shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.shapes import ShapeDtype


class ShapesTest(absltest.TestCase):

  def test_constructor_and_read_properties(self):
    sd = ShapeDtype((2, 3), np.int32)
    self.assertEqual(sd.shape, (2, 3))
    self.assertEqual(sd.dtype, np.int32)

  def test_default_dtype_is_float32(self):
    sd = ShapeDtype((2, 3))
    self.assertEqual(sd.shape, (2, 3))
    self.assertEqual(sd.dtype, np.float32)

  def test_signature_on_ndarray(self):
    array = np.array([[2, 3, 5, 7],
                      [11, 13, 17, 19]],
                     dtype=np.int16)
    sd = shapes.signature(array)
    self.assertEqual(sd.shape, (2, 4))
    self.assertEqual(sd.dtype, np.int16)

  def test_shape_dtype_repr(self):
    sd = ShapeDtype((2, 3))
    repr_string = '{}'.format(sd)
    self.assertEqual(repr_string,
                     "ShapeDtype{shape:(2, 3), dtype:<class 'numpy.float32'>}")

  def test_splice_signatures(self):
    sd1 = ShapeDtype((1,))
    sd2 = ShapeDtype((2,))
    sd3 = ShapeDtype((3,))
    sd4 = ShapeDtype((4,))
    sd5 = ShapeDtype((5,))

    # Signatures can be ShapeDtype instances, tuples of 2+ ShapeDtype instances,
    # or empty tuples.
    sig1 = sd1
    sig2 = (sd2, sd3, sd4)
    sig3 = ()
    sig4 = sd5
    spliced = shapes.splice_signatures(sig1, sig2, sig3, sig4)
    self.assertEqual(spliced, (sd1, sd2, sd3, sd4, sd5))

  def test_len_signature(self):
    """Signatures of all sizes should give correct length when asked."""
    x1 = np.array([1, 2, 3])
    x2 = np.array([10, 20, 30])
    inputs0 = ()
    inputs1 = x1  # NOT in a tuple
    inputs2 = (x1, x2)

    sig0 = shapes.signature(inputs0)
    sig1 = shapes.signature(inputs1)
    sig2 = shapes.signature(inputs2)

    # pylint: disable=g-generic-assert
    self.assertEqual(len(sig0), 0)
    self.assertEqual(len(sig1), 1)
    self.assertEqual(len(sig2), 2)
    # pylint: enable=g-generic-assert


if __name__ == '__main__':
  absltest.main()
