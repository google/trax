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
"""Tests for trax.models.atari_cnn."""

import functools
import operator as op
import numpy as np
from tensorflow import test
from trax.models import atari_cnn
from trax.shapes import ShapeDtype


class AtariCnnTest(test.TestCase):

  def test_computes(self):
    hidden_size = (4, 4)
    output_size = 6
    model = atari_cnn.AtariCnn(
        hidden_sizes=hidden_size, output_size=output_size)
    B, T, OBS = 2, 2, (28, 28, 3)  # pylint: disable=invalid-name
    input_signature = ShapeDtype((1, 1) + OBS)
    _, _ = model.init(input_signature)
    x = np.arange(B * (T + 1) * functools.reduce(op.mul, OBS)).reshape(
        B, T + 1, *OBS)
    y = model(x)
    self.assertEqual((B, T + 1, output_size), y.shape)


class FrameStackMLPTest(test.TestCase):

  def test_computes(self):
    hidden_size = (4, 4)
    output_size = 6
    model = atari_cnn.FrameStackMLP(
        hidden_sizes=hidden_size, output_size=output_size)
    B, T, OBS = 2, 2, 3  # pylint: disable=invalid-name
    input_signature = ShapeDtype((1, 1, OBS))
    _, _ = model.init(input_signature)
    x = np.arange(B * (T + 1) * OBS).reshape(B, T + 1, OBS)
    y = model(x)
    self.assertEqual((B, T + 1, output_size), y.shape)


if __name__ == '__main__':
  test.main()
