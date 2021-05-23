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
"""Tests for trax.data.inputs."""

import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from trax import data

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')


def _spm_path():
  return os.path.join(_TESTDATA, 'sentencepiece.model')


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

  def test_batch_data_padding(self):
    dataset = (([1] * (10 - i), i+1) for i in range(10))
    batches = data.batch(dataset, 10)
    batch = next(batches)
    self.assertEqual(batch[0].shape, (10, 10))
    self.assertTrue(np.array_equal(batch[0][-1], np.asarray([1] + 9 * [0])))

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

  def test_count_and_skip(self):
    dataset = lambda _: ((i, i+1) for i in range(10))
    examples = data.Serial(dataset, data.CountAndSkip('toy_data'))
    ex_generator = examples()
    ex1 = next(ex_generator)
    self.assertEqual(ex1, (0, 1))
    self.assertEqual(data.inputs.data_counters['toy_data'], 1)
    ex2 = next(ex_generator)
    self.assertEqual(ex2, (1, 2))
    self.assertEqual(data.inputs.data_counters['toy_data'], 2)
    ex3 = next(examples())  # new generator, will skip
    self.assertEqual(ex3, (2, 3))
    self.assertEqual(data.inputs.data_counters['toy_data'], 3)
    data.inputs.data_counters['toy_data'] = 0  # reset
    ex4 = next(examples())  # new generator, was reset
    self.assertEqual(ex4, (0, 1))
    self.assertEqual(data.inputs.data_counters['toy_data'], 1)

  def test_parallel(self):
    """Basic test of the parallel ccmbinator."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 20))
    parallel = data.Parallel([dataset1, dataset2])
    generator = parallel()

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 10)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 11)
    self.assertEqual(next(generator), 2)
    self.assertEqual(next(generator), 12)

  def test_parallel_with_gen_not_none(self):
    """Test of the parallel ccmbinator with a not none generator."""
    dataset1 = lambda _: (i for i in range(10))
    dataset2 = lambda _: (i for i in range(10, 20))
    parallel = data.Parallel([dataset1, dataset2])

    def test_generator():
      yield 0

    generator = parallel(gen=test_generator)

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 10)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 11)
    self.assertEqual(next(generator), 2)
    self.assertEqual(next(generator), 12)

  def test_parallel_with_weights(self):
    """Test of the parallel ccmbinator with weights."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 20))
    parallel = data.Parallel([dataset1, dataset2], counters=(2, 1))
    generator = parallel()

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 10)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 11)
    self.assertEqual(next(generator), 2)
    self.assertEqual(next(generator), 3)
    self.assertEqual(next(generator), 12)
    self.assertEqual(next(generator), 4)
    self.assertEqual(next(generator), 5)
    self.assertEqual(next(generator), 13)

  def test_parallel_with_weights_and_minimum(self):
    """Test of the parallel ccmbinator with weights and minimum."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 110))
    parallel = data.Parallel([dataset1, dataset2],
                             counters=(10, 100),
                             reweight_by_minimum=True)
    generator = parallel()

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 10)
    self.assertEqual(next(generator), 11)
    self.assertEqual(next(generator), 12)
    self.assertEqual(next(generator), 13)
    self.assertEqual(next(generator), 14)
    self.assertEqual(next(generator), 15)
    self.assertEqual(next(generator), 16)
    self.assertEqual(next(generator), 17)
    self.assertEqual(next(generator), 18)
    self.assertEqual(next(generator), 19)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 20)
    self.assertEqual(next(generator), 21)
    self.assertEqual(next(generator), 22)
    self.assertEqual(next(generator), 23)
    self.assertEqual(next(generator), 24)
    self.assertEqual(next(generator), 25)
    self.assertEqual(next(generator), 26)
    self.assertEqual(next(generator), 27)
    self.assertEqual(next(generator), 28)
    self.assertEqual(next(generator), 29)
    self.assertEqual(next(generator), 2)

  def test_parallel_with_gradual_reweighting(self):
    """Test of the parallel ccmbinator with weights and minimum."""
    dataset1 = lambda: (i for i in itertools.cycle(range(1)))
    dataset2 = lambda: (i for i in itertools.cycle(range(10, 30)))
    dataset3 = lambda: (i for i in itertools.cycle(range(30, 70)))
    parallel = data.Parallel([dataset2, dataset1, dataset3],
                             counters=(20, 1, 40),
                             gradually_reweight=True)
    generator = parallel()

    for _ in range(3):
      self.assertEqual(next(generator), 0)
      for i in range(20):
        self.assertEqual(next(generator), 10 + i)
        self.assertEqual(next(generator), 30 + 2 * i)
        self.assertEqual(next(generator), 30 + 2 * i + 1)

  def test_parallel_with_gradual_reweighting_remainders(self):
    """Test of the parallel ccmbinator with weights and minimum."""
    dataset1 = lambda: (i for i in itertools.cycle(range(1)))
    dataset2 = lambda: (i for i in itertools.cycle(range(10, 30)))
    dataset3 = lambda: (i for i in itertools.cycle(range(30, 80)))
    parallel = data.Parallel([dataset2, dataset1, dataset3],
                             counters=(20, 1, 50),
                             gradually_reweight=True,
                             use_remainders=True)
    generator = parallel()

    for _ in range(3):
      self.assertEqual(next(generator), 0)
      for i in range(20):
        self.assertEqual(next(generator), 10 + i)
        self.assertEqual(next(generator), 30 + 2 * i)
        self.assertEqual(next(generator), 30 + 2 * i + 1)
      # Here we process the remainder from dataset 3:
      for i in range(10):
        self.assertEqual(next(generator), 70 + i)

  def test_parallel_with_gradual_reweighting_remainders_big(self):
    """Test of the parallel ccmbinator with weights and minimum."""
    dataset1 = lambda: (i for i in itertools.cycle(range(1)))
    dataset2 = lambda: (i for i in itertools.cycle(range(10, 30)))
    dataset3 = lambda: (i for i in itertools.cycle(range(30, 80)))
    dataset4 = lambda: (i for i in itertools.cycle(range(100, 220)))
    parallel = data.Parallel([dataset2, dataset1, dataset4, dataset3],
                             counters=(20, 1, 120, 50),
                             gradually_reweight=True,
                             use_remainders=True)
    generator = parallel()

    for _ in range(3):
      self.assertEqual(next(generator), 0)
      for i in range(20):
        self.assertEqual(next(generator), 10 + i)
        for j in range(2):
          self.assertEqual(next(generator), 30 + 2 * i + j)
          for k in range(2):
            self.assertEqual(next(generator), 100 + 2 * 2 * i + 2 * j + k)
      # Here we process the remainder from datasets 3 and 4:
      for i in range(10):
        self.assertEqual(next(generator), 70 + i)
      for i in range(40):
        self.assertEqual(next(generator), 180 + i)

  def test_parallel_with_weights_three_datasets(self):
    """Check that data.Serial works inside another data.Serial."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 20))
    dataset3 = lambda: (i for i in range(20, 30))
    parallel = data.Parallel(
        [dataset1, dataset2, dataset3], counters=(2, 1, 3))
    generator = parallel()

    self.assertEqual(next(generator), 0)    # (1,0,0)
    self.assertEqual(next(generator), 10)   # (1,1,0)
    self.assertEqual(next(generator), 20)   # (1,1,1)
    self.assertEqual(next(generator), 1)    # (2,1,1)
    self.assertEqual(next(generator), 21)   # (2,1,2)
    self.assertEqual(next(generator), 22)   # (2,1,3)
    self.assertEqual(next(generator), 2)    # (1,0,0)
    self.assertEqual(next(generator), 11)   # (1,1,0)
    self.assertEqual(next(generator), 23)   # (1,1,1)
    self.assertEqual(next(generator), 3)    # (2,1,1)
    self.assertEqual(next(generator), 24)   # (2,1,2)
    self.assertEqual(next(generator), 25)   # (2,1,3)
    self.assertEqual(next(generator), 4)    # (1,0,0)

  def test_stack_parallel(self):
    """Test of stacked parallel ccmbinators."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 20))
    dataset3 = lambda: (i for i in range(20, 30))
    parallel_lev0 = data.Parallel([dataset1, dataset2])
    parallel_lev1 = data.Parallel([parallel_lev0, dataset3])
    generator = parallel_lev1()

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 20)
    self.assertEqual(next(generator), 10)
    self.assertEqual(next(generator), 21)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 22)
    self.assertEqual(next(generator), 11)
    self.assertEqual(next(generator), 23)
    self.assertEqual(next(generator), 2)
    self.assertEqual(next(generator), 24)
    self.assertEqual(next(generator), 12)

  def test_parallel_with_zero_counters(self):
    """Test of stacked parallel ccmbinators."""
    dataset1 = lambda: (i for i in range(10))
    dataset2 = lambda: (i for i in range(10, 20))
    dataset3 = lambda: (i for i in range(20, 30))
    parallel = data.Parallel([dataset1, dataset2, dataset3], counters=[1, 0, 1])
    generator = parallel()

    self.assertEqual(next(generator), 0)
    self.assertEqual(next(generator), 20)
    self.assertEqual(next(generator), 1)
    self.assertEqual(next(generator), 21)
    self.assertEqual(next(generator), 2)
    self.assertEqual(next(generator), 22)
    self.assertEqual(next(generator), 3)
    self.assertEqual(next(generator), 23)

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

  def test_truncate_to_length_no_arg(self):
    """Tests that a no-arg call leaves shapes unchanged."""
    def data_stream():
      while True:
        yield (np.zeros((1, 5)), np.ones((1, 5)))
    stream_fn = data.inputs.TruncateToLength()
    y0, y1 = next(stream_fn(data_stream()))
    self.assertEqual(y0.shape, (1, 5))
    self.assertEqual(y1.shape, (1, 5))

  @parameterized.named_parameters(
      ('none', None, ((1, 5), (1, 5))),
      ('large_values', {0: (1, 77), 1: (1, 88)}, ((1, 5), (1, 5))),
      ('small_values', {0: (1, 3), 1: (1, 2)}, ((1, 3), (1, 2))),
  )
  def test_truncate_to_length_len_map(self, len_map, out_shapes):
    """Tests that truncation occurs when len_map values are small enough."""
    def data_stream():
      while True:
        yield (np.zeros((1, 5)), np.ones((1, 5)))
    stream_fn = data.inputs.TruncateToLength(len_map=len_map)
    y0, y1 = next(stream_fn(data_stream()))
    self.assertEqual(y0.shape, out_shapes[0])
    self.assertEqual(y1.shape, out_shapes[1])

  def test_truncate_to_length_questionable_behavior(self):
    # Use of np.reshape in TruncateToLength allows non-truncation results
    # without warning. As long as the target shape (len_map value) is
    # lexicographically prior to the data shape, then np.reshape can happen,
    # even if it results in *adding* values to the overall array.
    #
    # This test passes as a marker of the questionable behavior, and should
    # *fail* -- and then be removed -- when the function is
    # clarified/re-implemented.
    #
    # TODO(jonni): Determine desired behavior, and fit implementation to it.
    x = np.arange(21).reshape((1, 21, 1))
    def data_stream():
      while True:
        yield x
    stream_fn = data.inputs.TruncateToLength(len_map={0: (1, 4, 6)})
    (y,) = next(stream_fn(data_stream()))
    self.assertEqual(y.shape, (1, 4, 6))
    self.assertEqual(y[0, 3, 1], 19)
    self.assertEqual(y[0, 3, 2], 20)  # end of original values [0..20]
    self.assertEqual(y[0, 3, 3], 0)  # added value
    self.assertEqual(y[0, 3, 4], 1)  # added value
    self.assertEqual(y[0, 3, 5], 2)  # added value

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

  @parameterized.named_parameters(
      ('encdec_on', True),
      ('encdec_off', False),
  )
  def test_addition_inputs_exceptions(self, encdec):
    vocab_size = 5
    batch_size = 256
    seq_length = 64
    # Check if max/min lengths are validated for train stream
    with self.assertRaises(ValueError):
      inputs = data.inputs.addition_inputs(
          vocab_size=vocab_size,
          batch_size=batch_size,
          train_length=2,
          eval_min_length=1,
          eval_max_length=seq_length,
          pad_to_multiple=seq_length,
          encdec=encdec)
      train_stream = inputs.train_stream(n_devices=1)
      for _ in range(10):
        next(train_stream)

    # Check if max/min lengths are validated for eval stream
    with self.assertRaises(ValueError):
      inputs = data.inputs.addition_inputs(
          vocab_size=vocab_size,
          batch_size=batch_size,
          train_length=seq_length,
          eval_min_length=1,
          eval_max_length=seq_length,
          pad_to_multiple=seq_length,
          encdec=True)
      eval_stream = inputs.eval_stream(n_devices=1)
      for _ in range(10):
        next(eval_stream)

  def test_addition_inputs_constraints(self):
    vocab_size = 5
    batch_size = 256
    seq_length = 64
    inputs = data.inputs.addition_inputs(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=seq_length,
        eval_min_length=seq_length,
        eval_max_length=seq_length,
        pad_to_multiple=seq_length,
        encdec=True)

    # Check if max length is respected for train stream
    train_stream = inputs.train_stream(n_devices=1)
    for _ in range(10):
      x, y, weights = next(train_stream)
      self.assertEqual(x.shape[1], seq_length)
      self.assertEqual(y.shape[1], seq_length)
      self.assertEqual(weights.shape[1], seq_length)

    # Check if max length is respected for eval stream
    eval_stream = inputs.eval_stream(n_devices=1)
    for _ in range(10):
      x, y, weights = next(eval_stream)
      self.assertEqual(x.shape[1], seq_length)
      self.assertEqual(y.shape[1], seq_length)
      self.assertEqual(weights.shape[1], seq_length)

  def _get_span_lengths(self, x):
    span_lengths = []
    curr_len = 0
    for i in range(1, len(x)):
      # 1 -> 0
      if x[i] == 0 and x[i - 1] == 1:
        span_lengths.append(curr_len)
        curr_len = 0
      # 1 -> 1 or 0 -> 1
      elif ((x[i] == 1 and x[i - 1] == 1) or
            (x[i] == 1 and x[i - 1] == 0)):
        curr_len += 1
    if curr_len != 0:
      span_lengths.append(curr_len)
    return span_lengths

  def test_random_spans_noise_mask(self):
    length = 100
    noise_density = 0.15
    mean_noise_span_length = 3.0

    # Take 5 random seed1, seed2 values.
    for seed in np.random.randint(0, 100, (5, 2)):
      is_noise = data.random_spans_noise_mask(length,
                                              noise_density,
                                              mean_noise_span_length,
                                              seed1=seed[0],
                                              seed2=seed[1])
      is_noise = is_noise.astype(np.int32)
      # noise_density fraction of tokens are produced
      self.assertEqual(np.sum(is_noise), noise_density * length)
      # Get span lengths and make sure the average is what we expect.
      actual_span_lengths = self._get_span_lengths(is_noise)
      average_span_length = (
          sum(actual_span_lengths) / len(actual_span_lengths))
      self.assertEqual(mean_noise_span_length, average_span_length)

  def test_process_c4_with_span_corruption(self):
    def process_c4_with_span_corruption(spm_path=None,
                                        extra_ids=0,
                                        train=False,
                                        max_length=100,
                                        noise_density=0.15,
                                        mean_noise_span_length=3.0,
                                        seed1=None,
                                        seed2=None):
      return data.Serial(
          data.TFDS(
              'c4/en:2.3.0', data_dir=_TESTDATA, keys=('text',), train=train),
          data.SentencePieceTokenize(spm_path=spm_path, extra_ids=extra_ids),
          data.generate_sequential_chunks(max_length=max_length),
          data.generate_random_noise_mask(
              noise_density=noise_density,
              mean_noise_span_length=mean_noise_span_length,
              seed1=seed1, seed2=seed2),
          data.consume_noise_mask(vocab_size=32000 + extra_ids),
          data.FilterEmptyExamples(),
          data.AppendValue(val={0: [1], 1: [1]}),
          data.PadToLength(len_map={0: 100, 1: 30}, pad_value={0: 0, 1: 0}),
          data.AddLossWeights(id_to_mask=0),
          data.Batch(batch_size=2)
          )

    gen = process_c4_with_span_corruption(
        spm_path=_spm_path(), seed1=0, seed2=1)

    examples = []
    for i, ex in enumerate(gen()):
      if i == 100:
        break
      examples.append(ex)

    self.assertLen(examples, 100)
    example = examples[0]

    batched_input, batched_output, batched_loss_weights = example

    self.assertSequenceEqual(
        batched_input.tolist(),
        # pylint: disable=bad-continuation,bad-whitespace
        [[   37,  2335,   113,  3977,   227,  7306,    45,     3,     9,
           4716,   147,     8,    71,  2658,    65,   118,  4313,    38,
              3,     9, 13065,    32, 31999,     9,  5704,    26,   109,
              6,  6862,     6,  4728,    45,     8,  3796, 24093, 11834,
           4716,    30,     8,  1379,    13, 31998,   130,   718,    12,
              8, 24124,  1343,   300,  4357,  1714, 31997,  1373,    47,
          16487,  3168,    16,   321,  7943,     5,     3,  4868,  3856,
           5700,    75,     7,   200,  2231,     6, 11163,     9,     6,
            113,    47,  5330,    45, 14354,     6,    47, 31996, 20721,
           3654,    44,     8,  3112,     5, 14599,    11,  8067, 31995,
              1,     0,     0,     0,     0,     0,     0,     0,     0,
              0],
         [  277,   828,    43,  5899,    46,    16, 10952,   139,   160,
           1687,    56,   539,    30,  2875,    41, 31122,  2307,   137,
           2702,  2780,    15,     7, 31999,    44,     8,  3112,    11,
             30,   569,   783,     5,     3, 17701,     6,  2194,    26,
             23,  1336,  6321,  1694,    30, 31998,   196,    56,  1852,
           1423,    25,     5,    27,   183,  8032, 31997,   217,   149,
           1513,    11,  2238,    25,  1800,     5,    96,  2703,    44,
           3065, 12537, 11163,     9,   535,    71,  9363, 14886,   646,
             44,     8,  3112,   243, 23281,    12,     8, 31996,   346,
            402,    17,    99,    83,    11,   773,  3668,  1280, 31995,
              1,     0,     0,     0,     0,     0,     0,     0,     0,
              0]]
        # pylint: enable=bad-continuation,bad-whitespace
        )

    self.assertSequenceEqual(
        batched_output.tolist(),
        # pylint: disable=bad-continuation,bad-whitespace
        [[31999,  1639,     7, 15480,     5, 11163, 31998,  2083,  9997,
           5076, 31997,   265,    11,     8, 31996,     3, 31995,  1343,
           2487,   106,     1,     0,     0,     0,     0,     0,     0,
              0,     0,     0],
         [31999,    12,     8, 15480,   130,   646, 31998,  1376,    10,
             96, 31997,    62,   410,    59, 31996,    96, 31995,    94,
            608,    10,     1,     0,     0,     0,     0,     0,     0,
              0,     0,     0]]
        # pylint: enable=bad-continuation,bad-whitespace
        )

    self.assertSequenceEqual(
        batched_loss_weights.tolist(),
        # pylint: disable=bad-continuation,bad-whitespace
        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        # pylint: enable=bad-continuation,bad-whitespace
        )

  def test_prefix_lm_last_output_batch_is_short(self):
    prefix_lm_fn = data.PrefixLM(input_length=2, output_length=3)
    examples = list(prefix_lm_fn([[1, 2, 3, 4, 5, 6, 7, 8]]))
    self.assertSequenceEqual(([1, 2], [3, 4, 5]), examples[0])
    self.assertSequenceEqual(([6, 7], [8]), examples[1])
    self.assertLen(examples, 2)

  def test_prefix_lm_last_input_batch_is_short(self):
    prefix_lm_fn = data.PrefixLM(input_length=2, output_length=3)
    examples = list(prefix_lm_fn([[1, 2, 3, 4, 5, 6]]))
    self.assertSequenceEqual(([1, 2], [3, 4, 5]), examples[0])
    self.assertLen(examples, 1)

  def test_prefix_lm_last_input_batch_exists_but_no_output(self):
    prefix_lm_fn = data.PrefixLM(input_length=2, output_length=3)
    examples = list(prefix_lm_fn([[1, 2, 3, 4, 5, 6, 7]]))
    self.assertSequenceEqual(([1, 2], [3, 4, 5]), examples[0])
    self.assertLen(examples, 1)

  def test_unbatch(self):
    unbatch_fn = data.UnBatch()
    batched_inputs = [
        # First batch - 3 examples
        (np.arange(3*2).reshape(3, -1),
         np.arange(3*3).reshape(3, -1),
         np.arange(3*4).reshape(3, -1)),
        # Second batch - 4 examples
        (np.arange(4*2).reshape(4, -1),
         np.arange(4*3).reshape(4, -1),
         np.arange(4*4).reshape(4, -1)),
    ]
    examples = list(unbatch_fn(batched_inputs))
    self.assertLen(examples, 3 + 4)

  def test_sine_shape(self):
    inputs = data.sine_inputs(batch_size=3, length=5)
    train_batch = next(inputs.train_stream(n_devices=1))
    eval_batch = next(inputs.eval_stream(n_devices=1))
    # (observations, actions, observations, mask)
    self.assertLen(train_batch, 4)
    self.assertLen(eval_batch, 4)
    for (x, y) in zip(train_batch, eval_batch):
      self.assertEqual(x.shape, (3, 5))
      self.assertEqual(y.shape, (3, 5))


if __name__ == '__main__':
  absltest.main()
