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
"""Tests for rnn layers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import fastmath
from trax import shapes
import trax.layers as tl


BACKENDS = [fastmath.Backend.JAX]


@parameterized.named_parameters(
    ('_' + b.value, b) for b in BACKENDS)
class RnnTest(parameterized.TestCase):

  def test_conv_gru_cell(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.ConvGRUCell(9, kernel_size=(3, 3))
      x = np.ones((8, 1, 7, 9))
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_gru_cell(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.GRUCell(9)
      xs = [np.ones((8, 7, 9)), np.ones((8, 7, 9))]
      _, _ = layer.init(shapes.signature(xs))
      ys = layer(xs)
      self.assertEqual([y.shape for y in ys], [(8, 7, 9), (8, 7, 9)])

  def test_lstm_cell(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.LSTMCell(9)
      xs = [np.ones((8, 9)), np.ones((8, 18))]
      _, _ = layer.init(shapes.signature(xs))
      ys = layer(xs)
      self.assertEqual([y.shape for y in ys], [(8, 9), (8, 18)])

  def test_sru(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.SRU(7)
      x = np.ones((8, 9, 7), np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_names(self, backend):
    with fastmath.use_backend(backend):
      layer = tl.LSTM(3)
      self.assertEqual('LSTM_3', str(layer))
      layer = tl.GRU(5)
      self.assertEqual('GRU_5', str(layer))
      layer = tl.SRU(7)
      self.assertEqual('SRU_7', str(layer))


if __name__ == '__main__':
  absltest.main()
