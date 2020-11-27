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
"""
data processing for BERT.

For now, this file only supports fine-tuning bert-base-uncased on GLUE.
"""

import os

import functools
import collections

import gin
import numpy as onp

import tensorflow_datasets as tfds
import tensorflow as tf

from bert import tokenization


from trax.data.inputs import Inputs
from trax.models.research.bert import download_model_if_model_name # needed for downloading tokenizer vocab


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
  ds = tfds.load(name=dataset_name, split=split, data_dir=data_dir,
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


@gin.configurable()
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


class BertLegacyTokenizer:
  # todo (piotrekp1): remove after training bert models from scratch in trax
  """
  Wrap up on the methods from bert to work in tf2 environment in order to use with models pretrained in other
  libraries. For models created from scratch use data.text_encoder.SubwordTextEncoder instead.
  For example:
    SubwordTextEncoder will tokenize "trax" into ['tr', 'ax', '_'],
    whereas original bert will tokenize it into ['tr', '##ax'],
    another example is '235235', which will be tokenized into ['235', '235', '_'] by subword text encoder
    and into ['235', '##2', '##35'] by original bert tokenizer
  It is made for compatibility with bert models trained in other libraries
  """
  def __init__(self, vocab_path=None, model_name=None, do_lower_case=True):
    if vocab_path is None and model_name is None:
      raise ValueError('either vocab_path or model_name is required to construct the BERT tokenizer.')
    if vocab_path is None:
        # get vocab_path from model name
        model_path, _ = download_model_if_model_name(model_name)
        vocab_path = os.path.join(model_path, 'vocab.txt')

    self.vocab = self.load_vocab(vocab_path)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)

  @classmethod
  def load_vocab(cls, vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = cls.convert_to_unicode(reader.readline())
        if not token:
          break
        token = token.strip()
        vocab[token] = index
        index += 1
    return vocab

  @classmethod
  def convert_to_unicode(cls, text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab[token] for token in tokens]

  def convert_ids_to_tokens(self, ids):
    return [self.inv_vocab[id] for id in ids]

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def encode(self, text):
    return self.convert_tokens_to_ids(self.tokenize(text))


def bert_preprocess(batch, tokenizer, key_a, key_b=None, max_len=128):
  """Tokenize and convert text to model inputs in a BERT format."""
  batch_size = batch['idx'].shape[0]
  input_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
  type_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
  for i in range(batch_size):
    sentence_a = batch[key_a][i]
    tokens_a = [101] + tokenizer.encode(sentence_a) + [102]

    if key_b is not None:
      sentence_b = batch[key_b][i]
      tokens_b = tokenizer.encode(sentence_b) + [102]
    else:
      tokens_b = []

    ex_input_ids = (tokens_a + tokens_b)[:max_len]
    ex_type_ids = ([0] * len(tokens_a) + [1] * len(tokens_b))[:max_len]

    input_ids[i, :len(ex_input_ids)] = ex_input_ids
    type_ids[i, :len(ex_type_ids)] = ex_type_ids
  return input_ids, type_ids, input_ids > 0, batch['label'], onp.ones(batch_size)


def bert_glue_inputs(dataset_name=gin.REQUIRED,
                     batch_size=16,
                     eval_batch_size=None,
                     data_dir=None,
                     max_len=512,
                     tokenizer=BertLegacyTokenizer):
  """Input pipeline for fine-tuning BERT on GLUE tasks."""
  if callable(tokenizer):  # If we pass a function, e.g., through gin, call it.
    tokenizer = tokenizer()

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

  preprocess_fn = functools.partial(bert_preprocess,
                                    tokenizer=tokenizer,
                                    key_a=key_a,
                                    key_b=key_b,
                                    max_len=max_len)
  return tfds_inputs( # TODO(piotrekp1): use data_streams instead
    dataset_name=dataset_name,
    preprocess_fun=preprocess_fn,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    data_dir=data_dir,
    train_split=tfds.Split.TRAIN,
    eval_split=eval_split
  )

# TODO(piotrekp1): add glue evaluation
