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


def _test_dataset_ints(lengths):
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

  def test_batch_data(self):
    dataset = ((i, i+1) for i in range(10))
    batches = inputs.batch_data(dataset, 10)
    batch = next(batches)
    self.assertEqual(len(batch), 2)
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

  def test_c4_preprocess(self):
    def load_c4_dataset(split='train'):
      dataset = tfds.load(
          name='c4', split=split, data_dir=_TESTDATA, shuffle_files=False)
      return dataset.map(lambda example: (example, example['text']))

    def examine_processed_dataset(proc_dataset):
      count = 0
      lengths = []
      for example in tfds.as_numpy(proc_dataset):
        count += 1
        ex = example[0]
        # Targets are in the example.
        self.assertIn('targets', ex)
        self.assertEqual(ex['targets'].dtype, np.int64)
        lengths.append(len(ex['targets']))
      return count, lengths

    unfiltered_count = 0
    for example in tfds.as_numpy(load_c4_dataset()):
      unfiltered_count += 1
      # Targets are NOT in the example.
      self.assertNotIn('targets', example[0])

    proc_dataset, proc_shapes = inputs.c4_preprocess(
        load_c4_dataset(), False, ({}, ()), 2048)

    # The target shape.
    self.assertEqual(proc_shapes[1], (None,))

    # `examine_processed_dataset` has some asserts in it.
    proc_count, char_lengths = examine_processed_dataset(proc_dataset)

    # Both the original and filtered datasets have examples.
    self.assertGreater(unfiltered_count, 0)
    self.assertGreater(proc_count, 0)

    # Because we filter out some entries on length.
    self.assertLess(proc_count, unfiltered_count)

    # Preprocess using the sentencepiece model in testdata.
    spc_proc_dataset, _ = inputs.c4_preprocess(
        load_c4_dataset(), False, ({}, ()), 2048,
        tokenization='spc',
        spm_path=os.path.join(_TESTDATA, 'sentencepiece.model'))

    spc_proc_count, spc_lengths = examine_processed_dataset(spc_proc_dataset)

    # spc shortens the target sequence a lot, should be almost equal to
    # unfiltered
    self.assertLessEqual(proc_count, spc_proc_count)
    self.assertEqual(unfiltered_count, spc_proc_count)

    # Assert all spc_lengths are lesser than their char counterparts.
    for spc_len, char_len in zip(spc_lengths, char_lengths):
      self.assertLessEqual(spc_len, char_len)

  def test_c4(self):
    gin.bind_parameter(
        'shuffle_and_batch_data.preprocess_fun', inputs.c4_preprocess)
    gin.bind_parameter('c4_preprocess.max_target_length', 2048)
    gin.bind_parameter('c4_preprocess.tokenization', 'spc')
    gin.bind_parameter('c4_preprocess.spm_path',
                       os.path.join(_TESTDATA, 'sentencepiece.model'))

    gin.bind_parameter('batch_fn.batch_size_per_device', 8)
    gin.bind_parameter('batch_fn.eval_batch_size', 8)
    gin.bind_parameter('batch_fn.max_eval_length', 2048)
    gin.bind_parameter('batch_fn.buckets', ([2049], [8, 1]))

    # Just make sure this doesn't throw.
    _ = inputs.inputs(
        'c4', data_dir=_TESTDATA, input_name='targets', target_name='text')


if __name__ == '__main__':
  tf.test.main()
