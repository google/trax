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
"""Tests for rnn layers."""

from absl.testing import absltest
from trax.layers import base
from trax.layers import rnn
from trax.shapes import ShapeDtype


class RnnLayerTest(absltest.TestCase):

  def _test_cell_runs(self, layer, input_signature, output_shape):
    final_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, final_shape)

  def test_conv_gru_cell(self):
    self._test_cell_runs(
        rnn.ConvGRUCell(9, kernel_size=(3, 3)),
        input_signature=ShapeDtype((8, 1, 7, 9)),
        output_shape=(8, 1, 7, 9))

  def test_gru_cell(self):
    self._test_cell_runs(
        rnn.GRUCell(9),
        input_signature=(ShapeDtype((8, 7, 9)), ShapeDtype((8, 7, 9))),
        output_shape=((8, 7, 9), (8, 7, 9)))

  def test_lstm_cell(self):
    self._test_cell_runs(
        rnn.LSTMCell(9),
        input_signature=(ShapeDtype((8, 9)), ShapeDtype((8, 18))),
        output_shape=((8, 9), (8, 18)))

  def test_sru(self):
    self._test_cell_runs(
        rnn.SRU(7),
        input_signature=ShapeDtype((8, 9, 7)),
        output_shape=(8, 9, 7))


if __name__ == '__main__':
  absltest.main()
