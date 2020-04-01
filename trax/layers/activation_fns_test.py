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
"""Tests for activation function layers."""

from absl.testing import absltest
import numpy as np
from trax.layers import activation_fns


class ActivationFnsTest(absltest.TestCase):

  def test_relu(self):
    layer = activation_fns.Relu()
    x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
    self.assertEqual([0.0, 0.0, 0.0, 2.0, 3.0, 5.0], list(layer(x)))

  def test_parametric_relu(self):
    layer = activation_fns.ParametricRelu(a=.25)
    x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
    self.assertEqual([0.0, 0.0, 0.0, .5, .75, 1.25], list(layer(x)))

  def test_leaky_relu(self):
    layer = activation_fns.LeakyRelu(a=.125)
    x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
    self.assertEqual([-.25, -.125, 0.0, 2.0, 3.0, 5.0], list(layer(x)))

  def test_hard_sigmoid(self):
    layer = activation_fns.HardSigmoid()
    x = np.array([-1.5, -.5, -.25, 0.0, .25, .5, 1.5])
    self.assertEqual([0.0, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], list(layer(x)))

  def test_hard_tanh(self):
    layer = activation_fns.HardTanh()
    x = np.array([-1.5, -.5, -.25, 0.0, .25, .5, 1.5])
    self.assertEqual([-1.0, -.5, -.25, 0.0, .25, .5, 1.0], list(layer(x)))


if __name__ == '__main__':
  absltest.main()
