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
"""data processing for BERT.

For now, this file only supports fine-tuning bert-base-uncased on GLUE.

TODO(afrozm): Move this into data/
"""
import functools

import gin
import numpy as onp

import tensorflow_datasets as tfds
from trax.data.inputs import Inputs


def _tfds_stream(n_devices,
                 dataset_name,
                 split,
                 batch_size,
                 data_dir,
                 shuffle_files,
                 shuffle_buffer_size,
                 batch_shuffle_size,
                 preprocess_fun,
                 repeat=True):
  """Streams batches of examples from tfds, with pure-python preprocessing."""
  # TODO(piotrekp1): delete if switched to data_streams
  if batch_size % n_devices != 0:
    raise ValueError(f'Batch size ({batch_size}) not divisible'
                     ' by number of devices ({n_devices})')
  ds = tfds.load(
      name=dataset_name,
      split=split,
      data_dir=data_dir,
      shuffle_files=shuffle_files)
  if repeat:
    ds = ds.repeat()
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.batch(batch_size)
  if batch_shuffle_size is not None:
    ds = ds.shuffle(batch_shuffle_size)

  for batch in tfds.as_numpy(ds):
    if preprocess_fun is not None:
      yield preprocess_fun(batch)
    else:
      yield batch


@gin.configurable
def tfds_inputs(
    dataset_name,
    preprocess_fun,
    batch_size,
    eval_batch_size=None,
    data_dir=None,
    train_split=tfds.Split.TRAIN,
    eval_split=tfds.Split.VALIDATION,
    shuffle_buffer_size=1024,
    batch_shuffle_size=128,
):
  """Tensorflow Datasets input pipeline, with pure-python preprocessing."""
  if eval_batch_size is None:
    eval_batch_size = batch_size
  return Inputs(
      train_stream=functools.partial(
          _tfds_stream,
          dataset_name=dataset_name,
          split=train_split,
          batch_size=batch_size,
          data_dir=data_dir,
          shuffle_files=True,
          shuffle_buffer_size=shuffle_buffer_size,
          batch_shuffle_size=batch_shuffle_size,
          preprocess_fun=preprocess_fun,
      ),
      eval_stream=functools.partial(
          _tfds_stream,
          dataset_name=dataset_name,
          split=eval_split,
          batch_size=eval_batch_size,
          data_dir=data_dir,
          shuffle_files=False,
          shuffle_buffer_size=None,
          batch_shuffle_size=None,
          preprocess_fun=preprocess_fun,
      ),
  )


@gin.configurable
def bert_tokenizer(vocab_path=None):
  """Constructs a BERT tokenizer."""
  # This import is from https://github.com/google-research/bert which is not
  # listed as a dependency in trax.
  # TODO(piotrekp1): using SubwordTextEncoder instead after fixing the
  # differences
  from bert.tokenization.bert_tokenization import FullTokenizer  # pylint: disable=g-import-not-at-top
  if vocab_path is None:
    raise ValueError('vocab_path is required to construct the BERT tokenizer.')
  tokenizer = FullTokenizer(vocab_path, do_lower_case=True)
  return tokenizer


def bert_preprocess(batch, tokenizer, key_a, key_b=None, max_len=128):
  """Tokenize and convert text to model inputs in a BERT format."""
  batch_size = batch['idx'].shape[0]
  input_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
  type_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
  for i in range(batch_size):
    sentence_a = batch[key_a][i]
    tokens_a = [101] + tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(sentence_a)) + [102]

    if key_b is not None:
      sentence_b = batch[key_b][i]
      tokens_b = tokenizer.convert_tokens_to_ids(
          tokenizer.tokenize(sentence_b)) + [102]
    else:
      tokens_b = []

    ex_input_ids = (tokens_a + tokens_b)[:max_len]
    ex_type_ids = ([0] * len(tokens_a) + [1] * len(tokens_b))[:max_len]

    input_ids[i, :len(ex_input_ids)] = ex_input_ids
    type_ids[i, :len(ex_type_ids)] = ex_type_ids
  return input_ids, type_ids, input_ids > 0, batch['label'], onp.ones(
      batch_size)


@gin.configurable
def glue_inputs(dataset_name=gin.REQUIRED,
                batch_size=16,
                eval_batch_size=None,
                data_dir=None,
                max_len=128,
                tokenizer=bert_tokenizer):
  """Input pipeline for fine-tuning BERT on GLUE tasks."""
  if callable(tokenizer):  # If we pass a function, e.g., through gin, call it.
    tokenizer = bert_tokenizer()

  eval_split = tfds.Split.VALIDATION
  if dataset_name == 'glue/mnli':
    eval_split = 'validation_matched'
    # TODO(kitaev): Support diagnostic dataset (AX)

  keys_lookup = {
      'glue/cola': ('sentence', None),
      'glue/sst2': ('sentence', None),
      'glue/mrpc': ('sentence1', 'sentence2'),
      'glue/qqp': ('question1', 'question2'),
      'glue/stsb': ('sentence1', 'sentence2'),
      'glue/mnli': ('premise', 'hypothesis'),  # TODO(kitaev): swap the two?
      'glue/qnli': ('question', 'sentence'),  # TODO(kitaev) swap the two?
      'glue/rte': ('sentence1', 'sentence2'),
      'glue/wnli': ('sentence1', 'sentence2'),
  }

  key_a, key_b = keys_lookup[dataset_name]

  preprocess_fn = functools.partial(
      bert_preprocess,
      tokenizer=tokenizer,
      key_a=key_a,
      key_b=key_b,
      max_len=max_len)
  return tfds_inputs(  # TODO(piotrekp1): use data_streams instead
      dataset_name=dataset_name,
      preprocess_fun=preprocess_fn,
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      data_dir=data_dir,
      train_split=tfds.Split.TRAIN,
      eval_split=eval_split)


# TODO(piotrekp1): add glue evaluation
