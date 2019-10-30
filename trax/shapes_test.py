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

"""Tests for trax.shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp

from trax.shapes import ShapeDtype
from trax.shapes import signature


class ShapesTest(absltest.TestCase):

  def test_constructor_and_read_properties(self):
    sd = ShapeDtype((2, 3), onp.int32)
    self.assertEqual(sd.shape, (2, 3))
    self.assertEqual(sd.dtype, onp.int32)

  def test_default_dtype_is_float32(self):
    sd = ShapeDtype((2, 3))
    self.assertEqual(sd.shape, (2, 3))
    self.assertEqual(sd.dtype, onp.float32)

  def test_signature_on_ndarray(self):
    array = onp.array([[2, 3, 5, 7],
                       [11, 13, 17, 19]],
                      dtype=onp.int16)
    sd = signature(array)
    self.assertEqual(sd.shape, (2, 4))
    self.assertEqual(sd.dtype, onp.int16)

  def test_shape_dtype_repr(self):
    sd = ShapeDtype((2, 3))
    repr_string = '{}'.format(sd)
    self.assertEqual(repr_string,
                     "ShapeDtype{shape:(2, 3), dtype:<class 'numpy.float32'>}")


if __name__ == '__main__':
  absltest.main()
