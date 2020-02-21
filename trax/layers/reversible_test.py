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
"""Tests for reversible layers."""

from absl.testing import absltest
from trax.layers import base
from trax.layers import reversible
from trax.shapes import ShapeDtype


class ReversibleLayerTest(absltest.TestCase):

  def test_reversible_swap(self):
    layer = reversible.ReversibleSwap()
    input_signature = (ShapeDtype((2, 3)), ShapeDtype((3, 3)))
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(final_shape, ((3, 3), (2, 3)))


if __name__ == '__main__':
  absltest.main()
