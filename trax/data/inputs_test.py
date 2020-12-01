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
from trax import data


class InputsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -5),
  )
  def test_shuffle_data_raises_error_queue_size(self, queue_size):
    samples = iter(range(10))
    with self.assertRaises(ValueError):
      _ = list(data.shuffle(samples, queue_size))

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
      ('twenty', 20),
  )
  def test_shuffle_data_queue_size(self, queue_size):
    samples = iter(range(100, 200))
    shuffled_stream = data.shuffle(samples, queue_size)
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
    shuffled_stream = data.shuffle(samples, queue_size)
    self.assertLen(list(shuffled_stream), n_samples)

  def test_batch_data(self):
    dataset = ((i, i+1) for i in range(10))
    batches = data.batch(dataset, 10)
    batch = next(batches)
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (10,))

  def test_batch_exception_size(self):
    dataset = ((i, i + 1) for i in range(10))
    with self.assertRaises(ValueError):
      batches = data.batch(dataset, 0)
      next(batches)

  def test_serial(self):
    dataset = lambda _: ((i, i+1) for i in range(10))
    batches = data.Serial(dataset, data.Shuffle(3), data.Batch(10))
    batch = next(batches())
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (10,))

  def test_serial_composes(self):
    """Check that data.Serial works inside another data.Serial."""
    dataset = lambda _: ((i, i+1) for i in range(10))
    serial1 = data.Serial(dataset, data.Shuffle(3))
    batches = data.Serial(serial1, data.Batch(10))
    batch = next(batches())
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (10,))

  def test_serial_with_python(self):
    dataset = lambda _: ((i, i+1) for i in range(10))
    batches = data.Serial(
        dataset,
        lambda g: map(lambda x: (x[0], x[1] + 1), g),
        lambda g: filter(lambda x: x[0] % 2 == 1, g),
        data.Batch(2)
    )
    batch = next(batches())
    self.assertLen(batch, 2)
    (xs, ys) = batch
    # First tuple after filtering is (1, 3) = (1, 2+1).
    self.assertEqual(xs[0], 1)
    self.assertEqual(ys[0], 3)
    # Second tuple after filtering is (3, 5).
    self.assertEqual(xs[1], 3)
    self.assertEqual(ys[1], 5)

  def test_pad_to_max_dims(self):
    tensors1 = [np.zeros((3, 10)), np.ones((3, 10))]
    padded1 = data.inputs.pad_to_max_dims(tensors1)
    self.assertEqual(padded1.shape, (2, 3, 10))
    tensors2 = [np.zeros((2, 10)), np.ones((3, 9))]
    padded2 = data.inputs.pad_to_max_dims(tensors2)
    self.assertEqual(padded2.shape, (2, 3, 10))
    tensors3 = [np.zeros((8, 10)), np.ones((8, 9))]
    padded3 = data.inputs.pad_to_max_dims(tensors3, 12)
    self.assertEqual(padded3.shape, (2, 12, 12))
    tensors4 = [np.zeros((2, 10)), np.ones((3, 9))]
    padded4 = data.inputs.pad_to_max_dims(tensors4, 12)
    self.assertEqual(padded4.shape, (2, 4, 12))

  def test_pad_to_length(self):
    tensors1 = [(np.zeros((5)), np.ones((3)))]
    pad_to_length_function1 = data.inputs.PadToLength(len_map={0: 10,
                                                               1: 11},
                                                      pad_value={0: 0,
                                                                 1: 1})
    padded1 = next(pad_to_length_function1(tensors1))
    self.assertEqual(padded1[0].shape, (10,))
    self.assertEqual(padded1[1].shape, (11,))

    tensors2 = [(np.zeros((15)), np.ones((20)))]
    pad_to_length_function2 = data.inputs.PadToLength(len_map={0: 10,
                                                               1: 10},
                                                      pad_value={0: 0,
                                                                 1: 1},
                                                      multiple=True)
    padded2 = next(pad_to_length_function2(tensors2))
    self.assertEqual(padded2[0].shape, (20,))
    self.assertEqual(padded2[1].shape, (20,))

  def test_concatenate_lm_input(self):
    tensors1 = [(np.zeros((5)), np.ones((3)))]

    lm_input_function1 = data.inputs.ConcatenateToLMInput(pad_to_length=10)
    lm_input_1 = next(lm_input_function1(tensors1))
    self.assertEqual(lm_input_1[0].shape, (10,))
    self.assertEqual(lm_input_1[1].shape, (10,))
    self.assertEqual(lm_input_1[2].shape, (10,))
    self.assertEqual(lm_input_1[2].all(),
                     np.array([[0., 0., 0., 0., 0.,
                                1., 1., 1., 0., 0.]]).all())

    tensors2 = [(np.zeros((5)), np.ones((3)))]
    lm_input_function2 = data.inputs.ConcatenateToLMInput()
    lm_input_2 = next(lm_input_function2(tensors2))
    self.assertEqual(lm_input_2[0].shape, (8,))
    self.assertEqual(lm_input_2[1].shape, (8,))
    self.assertEqual(lm_input_2[2].shape, (8,))
    self.assertEqual(lm_input_2[2].all(),
                     np.array([[0., 0., 0., 0., 0.,
                                1., 1., 1.]]).all())

  def test_truncate_to_length(self):
    tensors1 = [(np.zeros((1, 5)), np.ones((1, 5)))]

    truncate_to_length_function1 = data.inputs.TruncateToLength()
    truncated1 = next(truncate_to_length_function1(tensors1))
    self.assertEqual(truncated1[0].shape, (1, 5))
    self.assertEqual(truncated1[1].shape, (1, 5))

    truncate_to_length_function2 = data.inputs.TruncateToLength({0: (1, 3),
                                                                 1: (1, 2)})
    truncated2 = next(truncate_to_length_function2(tensors1))
    self.assertEqual(truncated2[0].shape, (1, 3))
    self.assertEqual(truncated2[1].shape, (1, 2))

    truncate_to_length_function3 = data.inputs.TruncateToLength({0: (1, 77),
                                                                 1: (1, 88)})
    truncated3 = next(truncate_to_length_function3(tensors1))
    self.assertEqual(truncated3[0].shape, (1, 5))
    self.assertEqual(truncated3[1].shape, (1, 5))

  def test_filter_empty_examples(self):
    tensors1 = [(np.zeros((0,)), np.ones((1, 5))),
                (np.zeros((1, 5)), np.ones((1, 5)))]

    filter_empty_examples_function1 = data.inputs.FilterEmptyExamples()
    filtered1 = next(filter_empty_examples_function1(tensors1))
    self.assertEqual(filtered1[0].shape, (1, 5))
    self.assertEqual(filtered1[1].shape, (1, 5))

    filter_empty_examples_function2 = data.inputs.FilterEmptyExamples(axes=[1])
    filtered2 = next(filter_empty_examples_function2(tensors1))
    self.assertEqual(filtered2[0].shape, (0,))
    self.assertEqual(filtered2[1].shape, (1, 5))

  def test_append_value(self):
    tensors1 = [(np.zeros((1, 5)), np.ones((1, 5)))]

    append_value_function1 = data.inputs.AppendValue()
    unmodified = next(append_value_function1(tensors1))
    self.assertEqual(unmodified[0].shape, (1, 5))
    self.assertEqual(unmodified[1].shape, (1, 5))

    append_value_function2 = data.inputs.AppendValue({0: [[5]],
                                                      1: [[4]]})
    appended = next(append_value_function2(tensors1))
    self.assertEqual(appended[0].shape, (1, 6))
    self.assertEqual(appended[0].all(),
                     np.array([[0., 0., 0., 0., 0., 5.]]).all())
    self.assertEqual(appended[1].shape, (1, 6))
    self.assertEqual(appended[1].all(),
                     np.array([[1., 1., 1., 1., 1., 4.]]).all())

  def test_pad_to_max_dims_boundary_list(self):
    tensors = [np.zeros((1, 15, 31)), np.ones((2, 10, 35)), np.ones((4, 2, 3))]
    padded_tensors = data.inputs.pad_to_max_dims(
        tensors, boundary=(None, 15, 20))
    # no boundary, only max in the first dim, 15 is already the max len in
    # second dim, last dim padded to multiple of 20.
    # The outer dim is the batch here.
    self.assertEqual(padded_tensors.shape, (3, 4, 15, 40))

  def test_pad_to_max_dims_strict_pad_on_len(self):
    tensors = [np.ones((15,)), np.ones((12,)), np.ones((14,))]
    padded_tensors = data.inputs.pad_to_max_dims(
        tensors, boundary=10, strict_pad_on_len=True)
    self.assertEqual(padded_tensors.shape, (3, 20))

  def test_bucket_by_length(self):
    def fake_generator(length, num_examples=1):
      for _ in range(num_examples):
        yield (np.ones((length,)), np.ones((length,)))

    def length_function(example):
      return max(example[0].shape[0], example[1].shape[0])

    batches = list(data.bucket_by_length(fake_generator(5, 6),
                                         length_function,
                                         [20],
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
