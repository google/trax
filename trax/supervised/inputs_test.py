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
from absl.testing import parameterized
import numpy as np
from trax.supervised import inputs


class InputsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -5),
  )
  def test_shuffle_data_raises_error_queue_size(self, queue_size):
    samples = iter(range(10))
    with self.assertRaises(ValueError):
      _ = list(inputs.shuffle_data(samples, queue_size))

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
      ('twenty', 20),
  )
  def test_shuffle_data_queue_size(self, queue_size):
    samples = iter(range(100, 200))
    shuffled_stream = inputs.shuffle_data(samples, queue_size)
    first_ten = [next(shuffled_stream) for _ in range(10)]

    # Queue size limits how far ahead/upstream the current sample can reach.
    self.assertLess(first_ten[0], 100 + queue_size)
    self.assertLess(first_ten[3], 103 + queue_size)
    self.assertLess(first_ten[9], 109 + queue_size)

    unshuffled_first_ten = list(range(100, 110))
    if queue_size == 1:  # Degenerate case: no shuffling can happen.
      self.assertEqual(first_ten, unshuffled_first_ten)
    if queue_size > 1:
      self.assertNotEqual(first_ten, unshuffled_first_ten)

  @parameterized.named_parameters(
      ('qsize_100_n_001', 100, 1),
      ('qsize_100_n_099', 100, 99),
      ('qsize_100_n_100', 100, 100),
      ('qsize_100_n_101', 100, 101),
      ('qsize_100_n_199', 100, 199),
  )
  def test_shuffle_data_yields_all_samples(self, queue_size, n_samples):
    samples = iter(range(n_samples))
    shuffled_stream = inputs.shuffle_data(samples, queue_size)
    self.assertLen(list(shuffled_stream), n_samples)

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

  def test_pad_to_max_dims_boundary_list(self):
    tensors = [np.zeros((1, 15, 31)), np.ones((2, 10, 35)), np.ones((4, 2, 3))]
    padded_tensors = inputs.pad_to_max_dims(tensors, boundary=(None, 15, 20))
    # no boundary, only max in the first dim, 15 is already the max len in
    # second dim, last dim padded to multiple of 20.
    # The outer dim is the batch here.
    self.assertEqual(padded_tensors.shape, (3, 4, 15, 40))

  def test_pad_to_max_dims_strict_pad_on_len(self):
    tensors = [np.ones((15,)), np.ones((12,)), np.ones((14,))]
    padded_tensors = inputs.pad_to_max_dims(
        tensors, boundary=10, strict_pad_on_len=True)
    self.assertEqual(padded_tensors.shape, (3, 20))

  def test_bucket_by_length(self):
    def fake_generator(length, num_examples=1):
      for _ in range(num_examples):
        yield (np.ones((length,)), np.ones((length,)))

    def length_function(example):
      return max(example[0].shape[0], example[1].shape[0])

    batches = list(inputs.bucket_by_length(fake_generator(5, 6),
                                           length_function,
                                           [20 + 1],
                                           [2],
                                           strict_pad_on_len=True))

    # We'll get three batches of 2 examples each.
    self.assertLen(batches, 3)
    self.assertIsInstance(batches[0], tuple)
    self.assertLen(batches[0], 2)
    self.assertEqual((2, 20), batches[0][0].shape)
    self.assertEqual((2, 20), batches[0][1].shape)

if __name__ == '__main__':
  absltest.main()
