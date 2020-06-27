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
"""Tests for trax.layers.research.position_encodings."""

import functools
import absl.testing.absltest as unittest
import numpy as np
import parameterized

from trax import fastmath
import trax.layers.research.position_encodings as pe


@parameterized.parameterized_class([
    # {'Encoding': pe.FixedBasePositionalEncoding},
    {'Encoding': pe.InfinitePositionalEncoding},
    {'Encoding': functools.partial(
        pe.InfinitePositionalEncoding, affine=False)},
    {'Encoding': functools.partial(
        pe.TimeBinPositionalEncoding, time_bin_length=5)},
])
class PositionEncodingsTest(unittest.TestCase):
  """Position encodings conform to the position encodings protocol."""

  @parameterized.parameterized.expand([
      (1, 100, 8),  # typical
      (1, 1, 8),  # short
      (1, 100, 1),  # narrow
      (2, 100, 8),  # batched
  ])
  def test_training(self, n, t, c):
    encoding = self.Encoding()
    input_ntc = np.random.randn(n, t, c)
    encoding.init(input_ntc)
    output_ntc = encoding(input_ntc)
    self.assertEqual(output_ntc.shape, input_ntc.shape)
    self.assertTrue(np.not_equal(output_ntc, input_ntc).any())

  @parameterized.parameterized.expand([
      (1, 100, 8),  # typical
      (1, 100, 1),  # narrow
      (2, 100, 8),  # batched
  ])
  def test_inference(self, n, t, c):
    # Get the eval mode outputs:
    encoding = self.Encoding(mode='eval')
    input_ntc = np.random.randn(n, t, c)
    rng = fastmath.random.get_prng(1234)
    encoding.init(input_ntc, rng=rng)
    output_ntc = encoding(input_ntc)

    is_random = self.Encoding == pe.InfinitePositionalEncoding

    # Get the predict mode outputs:
    encoding_pred = self.Encoding(mode='predict')
    encoding_pred.init(input_ntc[:, 0:1, :], rng=rng)
    output_ntc0 = encoding_pred(input_ntc[:, 0:1, :])
    if not is_random:
      np.testing.assert_allclose(output_ntc0, output_ntc[:, 0:1, :], atol=1e-4)

    output_ntc1 = encoding_pred(input_ntc[:, 1:2, :])
    if not is_random:
      np.testing.assert_allclose(output_ntc1, output_ntc[:, 1:2, :], atol=1e-4)

    output_ntc2 = encoding_pred(input_ntc[:, 2:3, :])
    if not is_random:
      np.testing.assert_allclose(output_ntc2, output_ntc[:, 2:3, :], atol=1e-4)


if __name__ == '__main__':
  unittest.main()
