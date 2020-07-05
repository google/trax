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
from absl.testing import parameterized
import numpy as np

from trax import fastmath
import trax.layers as tl


BACKENDS = ['jax', 'tf']


class ReversibleLayerTest(parameterized.TestCase):

  @parameterized.named_parameters([('_' + b, b) for b in BACKENDS])
  def test_reversible_swap(self, backend_name):
    with fastmath.use_backend(backend_name):
      layer = tl.ReversibleSwap()
      xs = [np.array([1, 2]), np.array([10, 20])]
      ys = layer(xs)
      self.assertEqual(tl.to_list(ys), [[10, 20], [1, 2]])


if __name__ == '__main__':
  absltest.main()
