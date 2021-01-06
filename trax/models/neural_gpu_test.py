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
"""Tests for trax.models.neural_gpu."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models import neural_gpu


class NeuralGPUTest(absltest.TestCase):

  def test_ngpu(self):
    model = neural_gpu.NeuralGPU(d_feature=30, steps=4, vocab_size=22)
    x = np.ones((3, 5, 7)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, 7, 22))


if __name__ == '__main__':
  absltest.main()
