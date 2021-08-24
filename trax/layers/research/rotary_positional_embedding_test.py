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
"""Tests for trax.layers.research.rotary_positional_embedding."""

from absl.testing import absltest
import numpy as np
from trax.layers.research import rotary_positional_embedding as rotary_pe


class RelAttentionTest(absltest.TestCase):

  def test_rotary_monotonicity(self):
    layer = rotary_pe.Rotate()
    batch_size = 1
    seq_len = 32
    d_model = 512
    shape = (batch_size, seq_len, d_model)
    q, k = np.ones(shape).astype(np.float32), np.ones(shape).astype(np.float32)
    q, k = layer(q), layer(k)

    self.assertEqual(q.dtype, np.float32)
    self.assertEqual(q.shape, shape)

    # Test monotonicity of the resulting dot_product for the two first tokens
    # in close proximity
    dot_product = np.einsum('bnd, bmd -> bnm', q, k)

    self.assertTrue((dot_product[0, 0, :9] > dot_product[0, 0, 1:10]).all())
    self.assertTrue((dot_product[0, 1, 1:10] > dot_product[0, 1, 2:11]).all())


if __name__ == '__main__':
  absltest.main()
