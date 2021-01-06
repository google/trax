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
"""Tests for MLP."""

from absl.testing import absltest
import numpy as np

from trax import fastmath
from trax import shapes
from trax.models import mlp


class MLPTest(absltest.TestCase):

  def test_mlp_forward_shape(self):
    model = mlp.MLP(layer_widths=(32, 16, 8))
    x = np.ones((7, 28, 28, 3)).astype(np.float32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (7, 8))



if __name__ == '__main__':
  absltest.main()
