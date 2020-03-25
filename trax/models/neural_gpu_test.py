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
"""Tests for trax.models.neural_gpu."""

from absl.testing import absltest
import numpy as onp
from trax.layers import base
from trax.models import neural_gpu
from trax.shapes import ShapeDtype


class NeuralGPUTest(absltest.TestCase):

  def test_ngpu(self):
    vocab_size = 2
    input_signature = ShapeDtype((3, 5, 7), onp.int32)
    model = neural_gpu.NeuralGPU(d_feature=30, steps=4, vocab_size=vocab_size)
    final_shape = base.check_shape_agreement(model, input_signature)
    self.assertEqual((3, 5, 7, vocab_size), final_shape)


if __name__ == '__main__':
  absltest.main()
