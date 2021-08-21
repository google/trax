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
"""Tests for trax.layers.relattention."""

from absl.testing import absltest
import numpy as np

import trax.layers as tl
import trax.layers.research.rel_attention as ra


class RelAttentionTest(absltest.TestCase):

  def test_fast_shift_matrix(self):
    layer = ra._fast_matrix_shift
    x = np.array([[[[-3., -2., -1., 0.], [-3., -2., -1.,
                                          0.], [-3., -2., -1., 0.],
                    [-3., -2., -1., 0.]]]]).astype(np.float32)

    y = layer(x)
    self.assertEqual(y.dtype, np.float32)
    self.assertEqual(
        tl.to_list(y), [[[[0., 0., -3., -2.], [-1., 0., 0., -3.],
                          [-2., -1., 0., 0.], [-3., -2., -1., 0.]]]])

if __name__ == '__main__':
  absltest.main()
