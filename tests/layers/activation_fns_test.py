# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for activation function layers."""

import numpy as np

from absl.testing import absltest

import trax.layers as tl


class ActivationFnsTest(absltest.TestCase):
    def test_relu(self):
        layer = tl.Relu()
        x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
        y = layer(x)
        self.assertEqual(tl.to_list(y), [0.0, 0.0, 0.0, 2.0, 3.0, 5.0])

    def test_parametric_relu(self):
        layer = tl.ParametricRelu(a=0.25)
        x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
        y = layer(x)
        self.assertEqual(tl.to_list(y), [0.0, 0.0, 0.0, 0.5, 0.75, 1.25])

    def test_leaky_relu(self):
        layer = tl.LeakyRelu(a=0.125)
        x = np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 5.0])
        y = layer(x)
        self.assertEqual(tl.to_list(y), [-0.25, -0.125, 0.0, 2.0, 3.0, 5.0])

    def test_hard_sigmoid(self):
        layer = tl.HardSigmoid()
        x = np.array([-1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5])
        y = layer(x)
        self.assertEqual(tl.to_list(y), [0.0, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])

    def test_hard_tanh(self):
        layer = tl.HardTanh()
        x = np.array([-1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5])
        y = layer(x)
        self.assertEqual(tl.to_list(y), [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])


if __name__ == "__main__":
    absltest.main()
