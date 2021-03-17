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
"""Tests for Residual Shuffle-Exchange Networks."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models.research import rse


class RSETest(absltest.TestCase):

  def test_rsu_forward_shape(self):
    batch_size = 3
    seq_len = 32
    d_model = 17
    model = rse.ResidualSwitchUnit(
        d_model=d_model, dropout=0.1, mode='train')
    x = np.ones((batch_size, seq_len, d_model)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (batch_size, seq_len, d_model))

  def test_shuffle_layer(self):
    shuffle_layer = rse.ShuffleLayer()
    x = np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]])
    print(x.shape)
    _, _ = shuffle_layer.init(shapes.signature(x))
    y = shuffle_layer(x)
    expected_output = np.array([[[0], [2], [4], [6], [1], [3], [5], [7]]])
    self._assert_equal_tensors(y, expected_output)

  def test_shuffle_layer_log_times_is_identity(self):
    seq_len = 8
    d_model = 17
    shuffle_layer = rse.ShuffleLayer()
    x = _input_with_indice_as_values(seq_len, d_model)
    _, _ = shuffle_layer.init(shapes.signature(x))
    y = x
    for _ in range(np.int(np.log2(seq_len))):
      y = shuffle_layer(y)
    self._assert_equal_tensors(x, y)

  def test_reverse_shuffle_layer(self):
    reverse_shuffle_layer = rse.ReverseShuffleLayer()
    x = np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]])
    print(x.shape)
    _, _ = reverse_shuffle_layer.init(shapes.signature(x))
    y = reverse_shuffle_layer(x)
    expected_output = np.array([[[0], [4], [1], [5], [2], [6], [3], [7]]])
    self._assert_equal_tensors(y, expected_output)

  def test_reverse_shuffle_layer_log_times_is_identity(self):
    seq_len = 8
    d_model = 17
    reverse_shuffle_layer = rse.ReverseShuffleLayer()
    x = _input_with_indice_as_values(seq_len, d_model)
    _, _ = reverse_shuffle_layer.init(shapes.signature(x))
    y = x
    for _ in range(np.int(np.log2(seq_len))):
      y = reverse_shuffle_layer(y)
    self._assert_equal_tensors(x, y)

  def test_rse_forward_shape(self):
    vocab_size = 12
    seq_len = 32
    model = rse.ResidualShuffleExchange(
        vocab_size=vocab_size, d_model=17, dropout=0.1, input_dropout=0.05,
        mode='train')
    x = np.ones((3, seq_len)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, seq_len, vocab_size))

  def _assert_equal_tensors(self, x, y):
    self.assertEqual(y.shape, x.shape)
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          self.assertEqual(
              x[i][j][k], y[i][j][k],
              f'Tensors differ on index [{i}][{j}][{k}].')


def _input_with_indice_as_values(length, dim):
  """Retuns np.array of size (1, length, dim) where x[0, a, b] = a."""
  positions = []
  for i in range(length):
    positions.append([i] * dim)
  positions_input = np.array(positions)
  positions_input = np.expand_dims(positions_input, axis=0)
  return positions_input


if __name__ == '__main__':
  absltest.main()
