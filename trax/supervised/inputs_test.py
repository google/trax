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
"""Tests for trax.supervised.inputs."""

from absl.testing import absltest
import numpy as np
from trax.supervised import inputs


class InputsTest(absltest.TestCase):

  def test_batch_data(self):
    dataset = ((i, i+1) for i in range(10))
    batches = inputs.batch_data(dataset, 10)
    batch = next(batches)
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (10,))

  def test_pad_to_max_dims(self):
    tensors1 = [np.zeros((3, 10)), np.ones((3, 10))]
    padded1 = inputs.pad_to_max_dims(tensors1)
    self.assertEqual(padded1.shape, (2, 3, 10))
    tensors2 = [np.zeros((2, 10)), np.ones((3, 9))]
    padded2 = inputs.pad_to_max_dims(tensors2)
    self.assertEqual(padded2.shape, (2, 3, 10))
    tensors3 = [np.zeros((8, 10)), np.ones((8, 9))]
    padded3 = inputs.pad_to_max_dims(tensors3, 12)
    self.assertEqual(padded3.shape, (2, 8, 12))
    tensors4 = [np.zeros((2, 10)), np.ones((3, 9))]
    padded4 = inputs.pad_to_max_dims(tensors4, 12)
    self.assertEqual(padded4.shape, (2, 4, 12))


if __name__ == '__main__':
  absltest.main()
