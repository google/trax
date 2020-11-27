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
"""Tests for trax.data.bert_inputs."""

import os

from functools import partial

from absl.testing import absltest

import numpy as np

from trax.data import bert_inputs
from trax.models.research.bert import download_model_if_model_name


class BertInputsTest(absltest.TestCase):
  def setUp(self):
    model_dir, model_filename = download_model_if_model_name('bert-base-uncased')
    self.model_dir = model_dir
    self.model_filename = model_filename

  def test_model_download(self):
    self.assertLen(os.listdir(self.model_dir), 5)

  def test_bert_tokenizer(self):
    test_cases = [
        ['Jack collected five apples this evening',
          [2990, 5067, 2274, 18108, 2023, 3944]],
        ['This is just a random sentence written in order to test tokenizer',
          [2023, 2003, 2074, 1037, 6721, 6251,2517, 1999, 2344, 2000, 3231, 19204, 17629]],
        ['trax is awesome',
          [19817, 8528, 2003, 12476]],
        ['neglecting discrepancy can suffice for indistinguishable vanity',
          [19046, 2075, 5860, 2890, 9739, 5666, 2064, 10514, 26989, 3401,
           2005, 27427, 2923, 2075, 27020, 25459, 2571, 18736]]
    ]
    tokenizer = bert_inputs.bert_tokenizer(model_name='bert-base-uncased')
    for sentence, token_ids in test_cases:
        tokens = tokenizer.tokenize(sentence)

        self.assertEqual(token_ids, tokenizer.convert_tokens_to_ids(tokens))
        self.assertEqual(tokens, tokenizer.convert_ids_to_tokens(token_ids))

  def test_bert_preprocess(self):
    batch = {
      'idx': np.arange(0, 2),
      'key_a': np.array(['I am', 'b']),
      'key_b': np.array(['e', 'f']),
      'label': np.array([0, 1])
    }
    MAX_LEN = 6
    exp_batch_keyb_result = (
      np.array([[101, 1045, 2572, 102, 1041, 102], # expected token_ids
               [101, 1038, 102, 1042, 102, 0]]),
      np.array([[0, 0, 0, 0, 1, 1], # expected segment_ids
              [0, 0, 0, 1, 1, 0]]),
      np.array([[True, True, True, True, True, True], # expected token masks
               [True, True, True, True, True, False]]),
      np.array([0, 1]), # expected labels
      np.array([1., 1.]) # expected sentence weights
    )
    exp_batch_nokeyb_result = (
      np.array([[101, 1045, 2572, 102, 0, 0],
               [101, 1038, 102, 0, 0, 0]]),
      np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
      np.array([[True, True, True, True, False, False],
               [True, True, True, False, False, False]]),
      np.array([0, 1]),
      np.array([1., 1.])
    )

    tokenizer = bert_inputs.bert_tokenizer(model_name='bert-base-uncased')

    batch_keyb_result = bert_inputs.bert_preprocess(batch, tokenizer, 'key_a', 'key_b', max_len=MAX_LEN)
    for exp_el_keyb_result, el_keyb_result in zip(exp_batch_keyb_result, batch_keyb_result):
        self.assertTrue((exp_el_keyb_result == el_keyb_result).all())

    del batch['key_b']
    batch_nokeyb_result = bert_inputs.bert_preprocess(batch, tokenizer, 'key_a', max_len=MAX_LEN)
    for exp_el_nokeyb_result, el_nokeyb_result in zip(exp_batch_nokeyb_result, batch_nokeyb_result):
        self.assertTrue((exp_el_nokeyb_result == el_nokeyb_result).all())

  def test_glue_inputs(self):
    exp_example_shape_dtype = ([(16, 512), (16, 512), (16, 512), (16,), (16,)],
                               [np.dtype('int32'),
                                np.dtype('int32'),
                                np.dtype('bool'),
                                np.dtype('int64'),
                                np.dtype('float64')])

    tokenizer_fn = partial(bert_inputs.bert_tokenizer, model_name='bert-base-uncased')
    # model name is passed to bert_tokenizer through gin, this tests whether bert_preprocess handles function calls

    stream = bert_inputs.bert_glue_inputs('glue/cola', tokenizer=tokenizer_fn)
    self.assertEqual(exp_example_shape_dtype, stream.example_shape_dtype)
    sample_batch = next(stream.eval_stream(1))
    self.assertEqual(4384, sample_batch[0][1, 2])
    self.assertEqual(2984, sample_batch[0][2, 1])
    self.assertTrue((np.zeros(exp_example_shape_dtype[0][1]) == sample_batch[1]).all())


if __name__ == '__main__':
  absltest.main()

