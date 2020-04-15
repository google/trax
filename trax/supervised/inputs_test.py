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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from trax.supervised import inputs


pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')


def test_dataset_ints(lengths):
  """Create a test dataset of int64 tensors of shape [length]."""
  def generator():
    """Sample generator of sequences of shape [length] of type int64."""
    for length in lengths:
      x = np.zeros([length], dtype=np.int64)
      yield (x, x)  # Inputs and targets are the same here.
  types = (tf.int64, tf.int64)
  shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
  return tf.data.Dataset.from_generator(
      generator, output_types=types, output_shapes=shapes)


class InputsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def test_batch_fn(self):
    dataset = test_dataset_ints([32])
    dataset = dataset.repeat(10)
    batches = inputs.batch_fn(
        dataset, True, ([None], [None]), 1, batch_size=10)
    count = 0
    for example in tfds.as_numpy(batches):
      count += 1
      self.assertEqual(example[0].shape[0], 10)  # Batch size = 10.
    self.assertEqual(count, 1)  # Just one batch here.

  def test_batch_fn_n_devices(self):
    dataset = test_dataset_ints([32])
    dataset = dataset.repeat(9)
    batches = inputs.batch_fn(
        dataset, True, ([None], [None]), 9, batch_size=10)
    count = 0
    for example in tfds.as_numpy(batches):
      count += 1
      # Batch size adjusted to be divisible by n_devices.
      self.assertEqual(example[0].shape[0], 9)
    self.assertEqual(count, 1)  # Just one batch here.

  def test_c4_preprocess(self):
    def load_c4_dataset(split='train'):
      dataset = tfds.load(
          name='c4', split=split, data_dir=_TESTDATA, shuffle_files=False)
      return dataset.map(lambda example: (example, example['text']))

    count = 0
    for example in tfds.as_numpy(load_c4_dataset()):
      count += 1
      # Targets are NOT in the example.
      self.assertNotIn('targets', example[0])

    proc_dataset, proc_shapes = inputs.c4_preprocess(
        load_c4_dataset(), False, ({}, ()), 2048)

    # The target shape.
    self.assertEqual(proc_shapes[1], (None,))

    proc_count = 0
    for example in tfds.as_numpy(proc_dataset):
      proc_count += 1
      ex = example[0]
      # Targets are in the example.
      self.assertIn('targets', ex)
      self.assertEqual(ex['targets'].dtype, np.int64)

    # Both the original and filtered datasets have examples.
    self.assertGreater(count, 0)
    self.assertGreater(proc_count, 0)

    # Because we filter out some entries on length.
    self.assertLess(proc_count, count)

  def test_c4(self):
    gin.bind_parameter(
        'shuffle_and_batch_data.preprocess_fun', inputs.c4_preprocess)
    gin.bind_parameter('c4_preprocess.max_target_length', 2048)

    gin.bind_parameter('batch_fn.batch_size_per_device', 8)
    gin.bind_parameter('batch_fn.eval_batch_size', 8)
    gin.bind_parameter('batch_fn.max_eval_length', 2048)
    gin.bind_parameter('batch_fn.buckets', ([2049], [8, 1]))

    # Just make sure this doesn't throw.
    _ = inputs.inputs(
        'c4', data_dir=_TESTDATA, input_name='targets', target_name='text')


if __name__ == '__main__':
  tf.test.main()
