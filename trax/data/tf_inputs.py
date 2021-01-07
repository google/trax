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
"""Trax TF input pipeline."""

import collections
import functools
import os
import random
import json

from absl import logging

import gin
import numpy as np

import t5.data
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from trax import fastmath
from trax.data import text_encoder

# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
# TODO(lukaszkaiser): can we improve efficiency, should that be changed?
_MAX_SKIP_EXAMPLES = 1e5


def no_preprocess(dataset, training):
  del training
  return dataset


def t2t_problems():
  # Load t2t problems on request only, this should save some import time.
  from tensor2tensor import problems_colab as t2tp  # pylint: disable=g-import-not-at-top
  return t2tp


@gin.configurable()
def data_streams(dataset_name,
                 data_dir=None,
                 preprocess_fn=no_preprocess,
                 bare_preprocess_fn=None,
                 shuffle_buffer_size=1024,
                 eval_holdout_size=0,
                 input_name=None,
                 target_name=None):
  """Make data streams for TF datasets.

  Args:
    dataset_name: a TFDS or T2T dataset name. If it's a T2T dataset name, prefix
      with 't2t_'.
    data_dir: data directory.
    preprocess_fn: function to use for pre-processing after appending targets to
      inputs.
    bare_preprocess_fn: function to use for pre-processing before appending
      targets to inputs.
    shuffle_buffer_size: size of the shuffle buffer.
    eval_holdout_size: float from 0 to <1; if >0 use this much of training data
      for evaluation (instead of looking for a pre-specified VALIDATION split).
    input_name: optional, name of the inputs from the dictionary.
    target_name: optional, name of the outputs either from the dictionary or as
      a result of post-processing.

  Returns:
    A pair of python streams, one for training and one for eval.
  """
  data_dir = download_and_prepare(dataset_name, data_dir)

  cache = []

  def stream(which):
    """Create the stream, cache TF streams if needed."""
    if not cache:
      cache.append(
          _train_and_eval_streams(dataset_name, data_dir, preprocess_fn,
                                  bare_preprocess_fn, shuffle_buffer_size,
                                  eval_holdout_size, input_name, target_name))

    (train_ds, eval_ds, input_name_c) = cache[0]
    dataset = eval_ds if which == 'eval' else train_ds
    return dataset_to_stream(dataset, input_name_c)

  train_stream = lambda: stream('train')
  eval_stream = lambda: stream('eval')
  return train_stream, eval_stream


def dataset_to_stream(dataset, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  # All input-pipeline processing should be on CPU.
  for example in fastmath.dataset_as_numpy(dataset):
    features = example[0]
    inp, out = features[input_name], example[1]
    mask = features['mask'] if 'mask' in features else None
    # Some accelerators don't handle uint8 well, cast to int.
    if isinstance(inp, np.uint8):
      inp = inp.astype(np.int32)
    if isinstance(out, np.uint8):
      out = out.astype(np.int32)
    yield (inp, out) if mask is None else (inp, out, mask)


def _train_and_eval_streams(dataset, data_dir, preprocess_fn,
                            bare_preprocess_fn, shuffle_buffer_size,
                            eval_holdout_size, input_name, target_name):
  """Return train and eval batches with input name and shape."""
  (train_data, eval_data,
   keys) = _train_and_eval_dataset(dataset, data_dir, eval_holdout_size)
  # If provided select input_name/target_name else fall back to keys if that is
  # available, else [None].
  input_names = ([input_name] if input_name is not None else
                 keys[0] if keys is not None else [None])
  target_names = ([target_name] if target_name is not None else
                  keys[1] if keys is not None else [None])

  train_batches = _shuffle_data(train_data, target_names, True,
                                shuffle_buffer_size, preprocess_fn,
                                bare_preprocess_fn)
  eval_batches = _shuffle_data(eval_data, target_names, False,
                               shuffle_buffer_size, preprocess_fn,
                               bare_preprocess_fn)
  return (train_batches, eval_batches, input_names[0])


def _shuffle_data(dataset, target_names, training, shuffle_buffer_size,
                  preprocess_fn, bare_preprocess_fn):
  """Shuffle the given dataset and run pre-processing."""

  def append_targets(example):
    """Append targets to the example dictionary. Needed for Keras."""
    if len(target_names) == 1:
      return (example, example[target_names[0]])
    targets = {}
    for name in target_names:
      targets[name] = example[name]
    return (example, targets)

  # `bare_preprocess_fn` is called before appending targets etc.
  if bare_preprocess_fn is not None:
    dataset = bare_preprocess_fn(dataset, training)
  dataset = dataset.map(append_targets)
  # TODO(pkozakowski): Repeat both the training and evaluation set, so we don't
  # have incomplete batches during evaluation. This will be a problem when we
  # add an option to evaluate on the whole dataset, then we'll need to think of
  # a different solution.
  dataset = dataset.repeat()
  if training:
    # Skip a random fraction at the beginning of the stream.  The skip is
    # essential for synchronous highly-parallel training to avoid multiple
    # replicas reading the same data in lock-step.
    dataset = dataset.skip(random.randint(0, _MAX_SKIP_EXAMPLES))
  dataset = preprocess_fn(dataset, training)
  dataset = dataset.shuffle(shuffle_buffer_size)
  return dataset.prefetch(8)


def _train_and_eval_dataset(dataset_name,
                            data_dir,
                            eval_holdout_size,
                            train_shuffle_files=True,
                            eval_shuffle_files=False):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the dataset; if it starts with 't2t_'
      then we'll search T2T Problem registry for it, otherwise we assume it is a
      dataset from TFDS and load it from there.
    data_dir: directory where the data is located.
    eval_holdout_size: float from 0 to <1; if >0 use this much of training data
      for evaluation (instead of looking for a pre-specified VALIDATION split).
    train_shuffle_files: Boolean determining whether or not to shuffle the train
      files at startup. Set to False if you want data determinism.
    eval_shuffle_files: Boolean determining whether or not to shuffle the test
      files at startup. Set to False if you want data determinism.

  Returns:
    a 4-tuple consisting of:
     * the train tf.Dataset
     * the eval tf.Dataset
     * information about features: a python dictionary with feature names
         as keys and an object as value that provides .shape and .n_classes.
     * supervised_keys: information what's the input and what's the target,
         ie., a pair of lists with input and target feature names.
  """
  if dataset_name.startswith('t2t_'):
    return _train_and_eval_dataset_v1(dataset_name[4:], data_dir,
                                      train_shuffle_files, eval_shuffle_files)
  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  info = dataset_builder.info
  splits = dataset_builder.info.splits
  if tfds.Split.TRAIN not in splits:
    raise ValueError('To train we require a train split in the dataset.')
  train_split = tfds.Split.TRAIN
  if eval_holdout_size > 0:
    holdout_percentage = int(eval_holdout_size * 100.0)
    train_percentage = 100 - holdout_percentage
    train_split = f'train[:{train_percentage}%]'
    eval_split = f'train[{train_percentage}%:]'
  elif dataset_name == 'glue/mnli':
    eval_split = 'validation_matched'
    # TODO(kitaev): Support diagnostic dataset (AX)
  else:
    if tfds.Split.VALIDATION not in splits and 'test' not in splits:
      raise ValueError('We require a validation or test split in the dataset.')
    eval_split = tfds.Split.VALIDATION
    if tfds.Split.VALIDATION not in splits:
      eval_split = tfds.Split.TEST
  train = tfds.load(
      name=dataset_name,
      split=train_split,
      data_dir=data_dir,
      shuffle_files=train_shuffle_files)
  valid = tfds.load(
      name=dataset_name,
      split=eval_split,
      data_dir=data_dir,
      shuffle_files=eval_shuffle_files)
  keys = None
  if info.supervised_keys:
    keys = ([info.supervised_keys[0]], [info.supervised_keys[1]])
  return train, valid, keys


@gin.configurable()
def TFDS(  # pylint: disable=invalid-name
    dataset_name,
    data_dir=None,
    tfds_preprocess_fn=None,
    keys=None,
    train=True,
    eval_holdout_size=0):
  """Returns an iterator of numpy arrays representing the dataset."""
  data_dir = download_and_prepare(dataset_name, data_dir)

  (train_data, eval_data, _) = _train_and_eval_dataset(dataset_name, data_dir,
                                                       eval_holdout_size)
  dataset = train_data if train else eval_data
  dataset = dataset if tfds_preprocess_fn is None else tfds_preprocess_fn(
      dataset)

  def select_from(example):
    return tuple(example[k] for k in keys)

  dataset = dataset.map(select_from)
  dataset = dataset.repeat()

  def gen(generator=None):
    del generator
    for example in fastmath.dataset_as_numpy(dataset):
      yield example

  return gen


def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ['inputs', 'targets']
  return {f: example[f] for f in feature_list if f in example}


def _eager_dataset_iterator(dataset):
  for item in dataset:
    flat = tf.nest.flatten(item)
    flat = [el.numpy() for el in flat]
    yield tf.nest.pack_sequence_as(item, flat)


def _train_and_eval_dataset_v1(problem_name, data_dir, train_shuffle_files,
                               eval_shuffle_files):
  """Return train and evaluation datasets, feature info and supervised keys."""
  with tf.device('cpu:0'):
    problem = t2t_problems().problem(problem_name)
    hparams = None
    if problem_name == 'video_bair_robot_pushing':
      hparams = problem.get_hparams()
      bair_robot_pushing_hparams(hparams)
    train_dataset = problem.dataset(
        tf.estimator.ModeKeys.TRAIN,
        data_dir,
        shuffle_files=train_shuffle_files,
        hparams=hparams)
    train_dataset = train_dataset.map(_select_features)
    eval_dataset = problem.dataset(
        tf.estimator.ModeKeys.EVAL,
        data_dir,
        shuffle_files=eval_shuffle_files,
        hparams=hparams)
    eval_dataset = eval_dataset.map(_select_features)
    # TODO(lukaszkaiser): remove this need for one example, just input_key.
    examples = list(tfds.as_numpy(train_dataset.take(1)))
  # We use 'inputs' as input except for purely auto-regressive tasks like
  # language models where 'targets' are used as input_key.
  input_key = 'inputs' if 'inputs' in examples[0] else 'targets'
  supervised_keys = ([input_key], ['targets'])
  return train_dataset, eval_dataset, supervised_keys


# Tokenization.
def tokenize(stream,
             keys=None,
             vocab_type='subword',
             vocab_file=None,
             vocab_dir=None,
             n_reserved_ids=0):
  """Tokenize examples from the stream.

  This function assumes that `stream` generates either strings or tuples/dicts
  containing strings at some `keys`. This function maps these strings to
  numpy arrays of integers -- the tokenized version of each string.

  Args:
    stream: A python generator yielding strings, tuples or dicts.
    keys: which keys of the tuple/dict to tokenize (by default: all)
    vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
    vocab_file: Name of the vocabulary file.
    vocab_dir: Directory which contains the vocabulary file.
    n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
      This is common for example when reserving the 0 for padding and 1 for EOS,
      but it's only needed if these symbols are not already included (and thus
      reserved) in the vocab_file.

  Yields:
    Examples from stream with strings at `keys` replaced by np.arrays of
    integers -- the tokenized version of these strings.
  """
  vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
  for example in stream:
    if isinstance(example, (list, tuple)):
      new_example = []
      for i, x in enumerate(example):
        if keys is None or i in keys:
          new_example.append(np.array(vocab.encode(x)) + n_reserved_ids)
        else:
          new_example.append(x)
      yield tuple(new_example)
    elif isinstance(example, dict):
      new_example = {}
      for k in example:
        if keys is None or k in keys:
          new_example[k] = np.array(vocab.encode(example[k])) + n_reserved_ids
        else:
          new_example[k] = example[k]
      yield new_example
    else:
      yield np.array(vocab.encode(example)) + n_reserved_ids


@gin.configurable()
def Tokenize(  # pylint: disable=invalid-name
    keys=None,
    vocab_type='subword',  # pylint: disable=invalid-name
    vocab_file=None,
    vocab_dir=None,
    n_reserved_ids=0):
  """Returns a function that maps text to integer arrays; see `tokenize`."""
  return lambda g: tokenize(  # pylint: disable=g-long-lambda
      g,
      keys=keys,
      vocab_type=vocab_type,
      vocab_file=vocab_file,
      vocab_dir=vocab_dir,
      n_reserved_ids=n_reserved_ids)


def detokenize(x,
               vocab_type='subword',
               vocab_file=None,
               vocab_dir=None,
               n_reserved_ids=0):
  """Maps integer arrays to text; the opposite of `tokenize`.

  In many cases (all char- and subword-type vocabularies and most sentencepiece
  ones) the tokenization is invertible, so detokenize(tokenize(x)) = x. In some
  more rare cases this can remove some spacing, but it is still often useful
  to run detokenize to get a readable version for a tokenized string.

  Args:
    x: a list or numpy array of integers.
    vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
    vocab_file: Name of the vocabulary file.
    vocab_dir: Directory which contains the vocabulary file.
    n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
      This is common for example when reserving the 0 for padding and 1 for EOS,
      but it's only needed if these symbols are not already included (and thus
      reserved) in the vocab_file.

  Returns:
    A string corresponding to the de-tokenized version of x.
  """
  vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
  x_unreserved = np.array(x) - n_reserved_ids
  return str(vocab.decode(x_unreserved.tolist()))


def _to_unicode(s):
  # Errors of the casting are ignored (e.g. sequences not allowed by UTF-8),
  # in order not to stay with incomplete examples (with empty values).
  return str(s, encoding='utf-8', errors='ignore')


def ConvertToUnicode(keys=None):  # pylint: disable=invalid-name
  """Converts to Unicode UTF-8 elements of an example.

  Useful for when TFDS outputs byte arrays. All of the errors of the conversion
  are ignored.

  Args:
    keys: tuple/list of example dimensions to convert.

  Returns:
    Function converting chosen elements of an example to UTF-8.
  """

  def _convert_to_unicode_str(stream, keys=None):
    for example in stream:
      if isinstance(example, (list, tuple)):
        new_example = []
        for i, x in enumerate(example):
          if keys is None or i in keys:
            new_example.append(_to_unicode(x))
          else:
            new_example.append(x)
        yield tuple(new_example)
      elif isinstance(example, dict):
        new_example = {}
        for k in example:
          if keys is None or k in keys:
            new_example[k] = _to_unicode(example[k])
          else:
            new_example[k] = example[k]
        yield new_example
      else:
        yield _to_unicode(example)

  return lambda g: _convert_to_unicode_str(g, keys)


def vocab_size(vocab_type='subword',
               vocab_file=None,
               vocab_dir=None,
               n_reserved_ids=0):
  """Returns the size of the vocabulary (number of symbols used).

  This function can be used to set the size of the final layers of a model that
  needs to predict symbols from a given vocabulary. More precisely, if this
  function returns N then the last layer size should be set to at least N (it
  can be more). Note that this function does take reserved IDs into account.

  Args:
    vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
    vocab_file: Name of the vocabulary file.
    vocab_dir: Directory which contains the vocabulary file.
    n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused.

  Returns:
    An integer, the number of symbols used (including reserved IDs).
  """
  vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
  return vocab.vocab_size + n_reserved_ids


def _get_vocab(vocab_type='subword', vocab_file=None, vocab_dir=None):
  """Gets the vocabulary object for tokenization; see tokenize for details."""
  if vocab_type not in [
      'char', 'subword', 'sentencepiece', 'bert', 'bert-lowercase'
  ]:
    raise ValueError(
        'vocab_type must be "subword", "char", "sentencepiece", "bert" or "bert-lowercase" '
        f'but got {vocab_type}')

  if vocab_type == 'char':
    # Note that we set num_reserved_ids=0 below. We could instead pass
    # the value n_reserved_ids from tokenize here -- ByteTextEncoder does
    # exactly the same thing as tokenize above, ie., adds num_reserved_ids.
    return text_encoder.ByteTextEncoder(num_reserved_ids=0)

  vocab_dir = vocab_dir or 'gs://trax-ml/vocabs/'
  path = os.path.join(vocab_dir, vocab_file)

  if vocab_type == 'subword':
    return text_encoder.SubwordTextEncoder(path)

  if vocab_type == 'bert':
    return text_encoder.BertEncoder(path, do_lower_case=False)

  if vocab_type == 'bert-lowercase':
    return text_encoder.BertEncoder(path, do_lower_case=True)

  assert vocab_type == 'sentencepiece'
  return t5.data.SentencePieceVocabulary(sentencepiece_model_file=path,
                                         extra_ids=0)


# Makes the function accessible in gin configs, even with all args denylisted.
@gin.configurable(denylist=['dataset', 'training'])
def cifar10_no_augmentation_preprocess(dataset, training):
  del training

  def cast_image(features, targets):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features, targets

  dataset = dataset.map(cast_image)
  return dataset


def _cifar_augment_image(image):
  """Image augmentation suitable for CIFAR-10/100.

  As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).

  Args:
    image: a Tensor.

  Returns:
    Tensor of the same shape as image.
  """
  image = tf.image.resize_with_crop_or_pad(image, 40, 40)
  image = tf.image.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  return image


# Makes the function accessible in gin configs, even with all args denylisted.
@gin.configurable(denylist=['dataset', 'training'])
def cifar10_augmentation_preprocess(dataset, training):
  """Preprocessing for cifar10 with augmentation (see below)."""

  def augment(features, targets):
    features['image'] = _cifar_augment_image(features['image'])
    return features, targets

  def cast_image(features, targets):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features, targets

  if training:
    dataset = dataset.map(augment)
  dataset = dataset.map(cast_image)
  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def cifar10_augmentation_flatten_preprocess(dataset,
                                            training,
                                            predict_image_train_weight=0.01):
  """Preprocessing for cifar10 that flattens it and appends targets."""

  def augment(features, targets):
    features['image'] = _cifar_augment_image(features['image'])
    return features, targets

  def flatten_image(features, targets):
    """Flatten the image."""
    img = features['image']
    flat = tf.cast(tf.reshape(img, [-1]), tf.int64)
    tgt = tf.expand_dims(targets, axis=0)
    flat_with_target = tf.concat([flat, tgt], axis=0)
    new_features = {}
    new_features['image'] = flat_with_target
    predict_image_weight = predict_image_train_weight if training else 0.0
    mask_begin = tf.ones_like(flat)
    mask_begin = tf.cast(mask_begin, tf.float32) * predict_image_weight
    mask_end = tf.cast(tf.ones_like(tgt), tf.float32)
    new_features['mask'] = tf.concat([mask_begin, mask_end], axis=0)
    return new_features, flat_with_target

  if training:
    dataset = dataset.map(augment)
  dataset = dataset.map(flatten_image)

  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def concat_preprocess(dataset, training, pad_symbol=0):
  """Pre-processing function that concatenates input and target for LM."""
  del training

  def concat(features, targets):
    inp = features['inputs']
    pad = tf.expand_dims(tf.zeros_like(inp[0]) + pad_symbol, axis=0)
    concat = tf.concat([pad, inp, pad, targets], axis=0)
    # Note: we're updating existing features dictionary here, so make sure
    # it is not re-used in some other ways outside of this function.
    features['inputs'] = concat
    return features, concat

  dataset = dataset.map(concat)
  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def squeeze_targets_preprocess(dataset, training):
  """Pre-processing function that squeezes last axis of targets."""
  del training

  def squeeze(features, targets):
    if targets.shape[-1] == 1:
      targets = tf.squeeze(targets, axis=-1)
    return features, targets

  dataset = dataset.map(squeeze)
  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def lm1b_preprocess(dataset,
                    training,
                    max_target_length=-1,
                    max_eval_target_length=-1):
  """Preprocessing for LM1B: filter out targets exceeding maximum length."""

  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)

  def eval_target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_eval_target_length + 1)

  if max_target_length > 0 and training:
    dataset = dataset.filter(target_right_length)

  if max_eval_target_length > 0 and not training:
    dataset = dataset.filter(eval_target_right_length)

  return dataset


# TODO(lukaszkaiser): find a single more abstract way of text pre-processing.
@gin.configurable(denylist=['dataset', 'training'])
def wmt_preprocess(dataset, training, max_length=-1, max_eval_length=-1):
  """Preprocessing for LM1B: filter out targets exceeding maximum length."""

  def train_right_length(example, target):
    l = tf.maximum(tf.shape(example['inputs'])[0], tf.shape(target)[0])
    return tf.less(l, max_length + 1)

  def eval_right_length(example, target):
    l = tf.maximum(tf.shape(example['inputs'])[0], tf.shape(target)[0])
    return tf.less(l, max_eval_length + 1)

  if max_length > 0 and training:
    dataset = dataset.filter(train_right_length)

  if max_eval_length > 0 and not training:
    dataset = dataset.filter(eval_right_length)

  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def wmt_concat_preprocess(dataset, training, max_length=-1, max_eval_length=-1):
  """Preprocessing for WMT: filter exceeding maximum length and concatenate."""
  dataset = wmt_preprocess(dataset, training, max_length, max_eval_length)

  def concat_and_add_mask(features, targets):
    inp = features['inputs']
    pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
    concat = tf.concat([inp, pad, targets], axis=0)
    mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
    features['inputs'] = concat
    features['mask'] = mask
    return features, concat

  dataset = dataset.map(concat_and_add_mask)
  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def lm_token_preprocessing(dataset, training):
  """Concatenates inputs, 0, targets, with masking only for targets."""
  del training

  def concat_and_add_mask(x):
    inp = x['inputs']
    targets = x['targets']
    pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
    concat = tf.concat([inp, pad, targets], axis=0)
    mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
    x['inputs'] = concat
    x['targets'] = concat
    x['mask'] = mask
    return x

  dataset = dataset.map(concat_and_add_mask)
  return dataset


@gin.configurable(denylist=['hparams'])
def bair_robot_pushing_hparams(hparams=None,
                               video_num_input_frames=1,
                               video_num_target_frames=15):
  if hparams is not None:
    hparams.video_num_input_frames = video_num_input_frames
    hparams.video_num_target_frames = video_num_target_frames
  else:
    return video_num_input_frames, video_num_target_frames


@gin.configurable(denylist=['dataset', 'training'])
def bair_robot_pushing_preprocess(dataset, training):
  """Pre-processing function that concatenates input and target frames."""
  del training

  def concat_and_add_mask(features, targets):
    """Concatenate input and output frames to form a language modeling setup."""
    inp = features['inputs']
    concat = tf.concat([inp, targets], axis=0)
    mask = tf.concat([tf.zeros_like(inp), tf.ones_like(targets)], axis=0)
    concat = tf.reshape(concat, (-1,))
    mask = tf.reshape(mask, (-1,))
    concat = tf.cast(concat, tf.int32)
    mask = tf.cast(mask, tf.float32)
    features['inputs'] = features['targets'] = concat
    features['mask'] = mask
    return features, concat

  dataset = dataset.map(concat_and_add_mask)
  return dataset


DEFAULT_SPM_PATH = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'  # GCS


@gin.configurable(denylist=['dataset', 'training'])
def c4_preprocess(dataset,
                  training,
                  max_target_length=-1,
                  tokenization=None,
                  spm_path=None):
  """Pre-processing function for C4 dataset."""
  del training

  def unicode_decode_chars(features, targets):
    targets = tf.strings.unicode_decode(features['text'], 'UTF-8')
    targets = tf.cast(targets, tf.int64)
    features['targets'] = targets
    features['inputs'] = targets
    return (features, targets)

  def spc_tokenize(tokenizer, features, targets):
    del targets
    tokenized_text = tokenizer.tokenize(features['text'])
    features['targets'] = tf.cast(tokenized_text, tf.int64)
    features['inputs'] = features['targets']
    return features, features['targets']

  if tokenization == 'spc':
    spm_path = spm_path or t5.data.DEFAULT_SPM_PATH
    with tf.compat.v1.gfile.GFile(spm_path, 'rb') as f:
      spc_model = f.read()
    tokenizer = tf_text.SentencepieceTokenizer(model=spc_model)
    dataset = dataset.map(functools.partial(spc_tokenize, tokenizer))
  else:
    dataset = dataset.map(unicode_decode_chars)

  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)

  if max_target_length > 0:
    dataset = dataset.filter(target_right_length)

  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def c4_bare_preprocess_fn(dataset,
                          training=True,
                          spm_path=None,
                          copy_pretokenized=True,
                          sequence_length=None):
  """Returns a dataset that contains 'inputs' and 'targets' from C4."""
  # Set target key to be equal to the text content.
  dataset = t5.data.preprocessors.rekey(
      dataset, key_map={
          'targets': 'text',
          'inputs': None
      })

  # Vocabulary for tokenization.
  extra_ids = 0
  vocab = t5.data.SentencePieceVocabulary(
      sentencepiece_model_file=spm_path or t5.data.DEFAULT_SPM_PATH,
      extra_ids=extra_ids)
  feature = t5.data.Feature(vocab)
  output_features = {'targets': feature, 'inputs': feature}

  # Tokenize the targets.
  keys = output_features

  def encode_string_features_fn(features):
    """Encode all specified feature that are strings and return a dictionary.

    Args:
      features: a dictionary

    Returns:
      a dictionary
    """
    ret = {}
    for k, v in features.items():
      if k in keys and v.dtype == tf.string:
        if copy_pretokenized:
          ret['%s_pretokenized' % k] = v
        v = tf.cast(output_features[k].vocabulary.encode_tf(v), tf.int64)
      ret[k] = v
    return ret

  dataset = dataset.map(
      encode_string_features_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Preprocess the tokens - the exact preprocessors are set via gin.
  dataset = t5.data.preprocessors.unsupervised(
      dataset, sequence_length=sequence_length, output_features=output_features)

  # Add EOS.
  dataset = add_eos_to_output_features(dataset, training)

  # Truncate and then pad the examples -- all examples have the same shape.
  dataset = truncate_dataset_on_len(dataset, training, sequence_length, True)
  dataset = pad_dataset_to_length(dataset, training, sequence_length)

  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def filter_dataset_on_len(dataset,
                          training,
                          len_map=None,
                          filter_on_eval=False):
  """Filters a dataset of lengths given in `len_map`.

  Args:
    dataset: `tf.data.Dataset` the dataset to filter.
    training: bool, true if we are in training mode.
    len_map: optional dict of str to (int, int). We filter examples where a
      feature's size is beyond the specified bounds. Ex:
      {'inputs': (1, 512), 'targets': (64, 128)} will keep only those examples
        where 1 <= len(inputs) <= 512 and 64 <= len(targets) <= 128.
    filter_on_eval: bool if true, we will filter in eval mode also.

  Returns:
    a filtered `tf.data.Dataset`.
  """
  if (len_map is None) or (not training and not filter_on_eval):
    return dataset

  assert isinstance(len_map, dict)
  for k, bounds in len_map.items():
    # pylint: disable=cell-var-from-loop
    # TODO(afrozm): Investigate `cell-var-from-loop` - since this is WAI and
    # there is a test too.
    def within_bounds(x, key, len_bounds):
      size = tf.shape(x[key])[0]
      min_len, max_len = len_bounds
      return (min_len <= size) and (size <= max_len)

    dataset = dataset.filter(lambda x: within_bounds(x, k, bounds))
    # pylint: enable=cell-var-from-loop

  return dataset


@gin.configurable(denylist=['dataset', 'training'])
def truncate_dataset_on_len(dataset,
                            training,
                            len_map=None,
                            truncate_on_eval=False):
  """Truncates features in an example to lengths given in `len_map`.

  Args:
    dataset: `tf.data.Dataset` the dataset to filter.
    training: bool, true if we are in training mode.
    len_map: optional dict of str to int, we truncate examples where a feature's
      size is beyond the max. Ex: {'inputs': 512, 'targets': 64} will truncate
        examples to be within those bounds.
    truncate_on_eval: bool if true, we will truncate in eval mode also.

  Returns:
    a filtered `tf.data.Dataset`.
  """
  if (len_map is None) or (not training and not truncate_on_eval):
    return dataset

  assert isinstance(len_map, dict)

  def truncate_example(x):
    for key, max_len in len_map.items():
      x_len = tf.shape(x[key])[0]
      if x_len > max_len:
        x[key] = x[key][:max_len, ...]
    return x

  return dataset.map(truncate_example)


@gin.configurable(denylist=['dataset', 'training'])
def pad_dataset_to_length(dataset, training, len_map=None):
  """Pad features less than specified length to specified length."""
  del training
  if len_map is None:
    return dataset

  def pad_to_len(x):
    for key, max_len in len_map.items():
      x_shape = tf.shape(x[key])
      x_len = x_shape[0]
      if x_len < max_len:
        pad_shape = [
            max_len - x_len,
        ]
        zeros = tf.zeros(pad_shape, dtype=x[key].dtype)
        x[key] = tf.concat([x[key], zeros], 0)
    return x

  return dataset.map(pad_to_len)


@gin.configurable(denylist=['dataset', 'training'])
def add_eos_to_output_features(dataset,
                               training,
                               output_features='targets',
                               eos=1):
  """Adds `EOS` to all features in `output_features`."""
  del training
  if not isinstance(output_features, (list, tuple)):
    output_features = [output_features]

  def add_eos(x):
    for output_feature in output_features:
      x[output_feature] = tf.concat([x[output_feature], [eos]], axis=0)
    return x

  return dataset.map(add_eos)


@gin.configurable(denylist=['dataset', 'training'])
def generic_text_dataset_preprocess_fn(dataset,
                                       training=True,
                                       text_preprocess_fns=None,
                                       token_preprocess_fns=None,
                                       spm_path=None,
                                       copy_pretokenized=False,
                                       debug_print_examples=False,
                                       debug_print_examples_rate=0.01):
  """Pre-processes, tokenizes and post-processes a `tf.data.Dataset`.

  Args:
    dataset: `tf.data.Dataset` to process.
    training: boolean, set to True if training, False otherwise.
    text_preprocess_fns: None or list of callables: `tf.data.Dataset`, bool ->
      `tf.data.Dataset` this operates before tokenization. Typically used to
      select which fields we want to learn over or change something into "text
      to text" form.
    token_preprocess_fns: None or list of callables: `tf.data.Dataset`, bool ->
      `tf.data.Dataset`, this operates after tokenization. Since this can view
      the tokenized fields, this can be used to filter on length etc.
    spm_path: None or str, path to a sentencepiece model to use for tokenization
      by default uses the 32k vocabulary from T5.
    copy_pretokenized: bool, if True retains the original fields after
      tokenization.
    debug_print_examples: bool, if True this prints examples to the logging
      stream for inspection, both before and after tokenization.
    debug_print_examples_rate: float, [0, 1.0], on average this fraction of
      dataset examples will be printed out in each phase i.e. pre and post
      tokenization.

  Returns:
    a `tf.data.Dataset` with all the preprocessing and tokenization performed.
  """

  # The assumption is that `text_preprocess_fns` finally gives us a dataset
  # which has `inputs` and `targets`.
  if text_preprocess_fns is not None:
    for text_preprocess_fn in text_preprocess_fns:
      dataset = text_preprocess_fn(dataset, training)

  # Print debugging examples if needed before tokenization.
  if debug_print_examples:

    def print_examples(x):
      if np.random.uniform() < debug_print_examples_rate:
        tf.print(x, output_stream=logging.info)
      return x

    dataset = dataset.map(print_examples)

  # Vocabulary for tokenization.
  extra_ids = 0
  vocab = t5.data.SentencePieceVocabulary(
      sentencepiece_model_file=spm_path or t5.data.DEFAULT_SPM_PATH,
      extra_ids=extra_ids)
  feature = t5.data.Feature(vocab)
  output_features = {'targets': feature, 'inputs': feature}

  # Tokenize the inputs and targets.
  dataset = t5.data.preprocessors.tokenize(
      dataset, output_features, copy_pretokenized=copy_pretokenized)

  # Apply the token-preprocessors.
  if token_preprocess_fns is not None:
    for token_preprocess_fn in token_preprocess_fns:
      dataset = token_preprocess_fn(dataset, training)

  if debug_print_examples:

    def print_examples_and_shapes(x):
      if np.random.uniform() < debug_print_examples_rate:
        tf.print(
            {
                'inputs_shape': tf.size(x['inputs']),
                'targets_shape': tf.size(x['targets']),
                'inputs': x['inputs'],
                'targets': x['targets'],
            },
            output_stream=logging.info)
      return x

    dataset = dataset.map(print_examples_and_shapes)

  return dataset


@gin.configurable
def get_t5_preprocessor_by_name(name=None, fn_kwargs=None):
  """Returns a closure of any T5 preprocessor function with its arguments.

  The main use-case is to use this (with gin scopes) to make any preprocessor
  function available in a gin file to configure and use.

  See: `TFInputs.test_gin_configurable_preprocessors`

  Args:
    name: str, name of the preprocessor function to configure.
    fn_kwargs: optional dictionary, the arguments to configure, these will be
      partially applied to the function given by `name`.

  Returns:
    a closure of the preprocessor function along with its arguments, this
    function takes two arguments only, dataset and boolean training and ignores
    the training and calls the t5 processor with the dataset (and closed over
    arguments only).
  """

  assert name is not None
  f = getattr(t5.data.preprocessors, name)
  if fn_kwargs is not None:
    f = functools.partial(f, **fn_kwargs)
  return lambda ds, unused_training: f(ds)


def download_and_prepare(dataset_name, data_dir):
  """Downloads and prepares T2T or TFDS dataset.

  Args:
    dataset_name: tfds dataset or t2t problem name prefixed by 't2t_'.
    data_dir: location of existing dataset or None.

  Returns:
    data_dir: path string of downloaded data.
  """
  if not data_dir:
    data_dir = os.path.expanduser('~/tensorflow_datasets/')
    dl_dir = os.path.join(data_dir, 'download')
    logging.info(
        'No dataset directory provided. '
        'Downloading and generating dataset for %s inside data directory %s '
        'For large datasets it is better to prepare datasets manually!',
        dataset_name, data_dir)
    if dataset_name.startswith('t2t_'):
      # Download and run dataset generator for T2T problem.
      data_dir = os.path.join(data_dir, dataset_name)
      tf.io.gfile.makedirs(data_dir)
      tf.io.gfile.makedirs(dl_dir)
      t2t_problems().problem(dataset_name[len('t2t_'):]).generate_data(
          data_dir, dl_dir)
    else:
      # Download and prepare TFDS dataset.
      tfds_builder = tfds.builder(dataset_name)
      tfds_builder.download_and_prepare(download_dir=dl_dir)
  else:
    data_dir = os.path.expanduser(data_dir)
  return data_dir


def BertSingleSentenceInputs(batch,  # pylint: disable=invalid-name
                             labeled=True,
                             cls_id=101,
                             sep_id=102):
  """Prepares inputs for BERT: add [SEP], [CLS] and create embeddings."""
  if labeled:
    for sent1, label in batch:
      value_vector = np.concatenate(([cls_id], sent1, [sep_id]))
      segment_embs = np.zeros(sent1.shape[0] + 2, dtype=np.int32)
      yield value_vector, segment_embs, segment_embs, label, np.int32(1)
  else:
    for (sent1,) in batch:  # row is a tuple with 1 element
      value_vector = np.concatenate(([cls_id], sent1, [sep_id]))
      segment_embs = np.zeros(sent1.shape[0] + 2, dtype=np.int32)
      yield value_vector, segment_embs, segment_embs


def BertDoubleSentenceInputs(batch,  # pylint: disable=invalid-name
                             labeled=True,
                             cls_id=101,
                             sep_id=102):
  """Prepares inputs for BERT models by adding [SEP] and [CLS] tokens and creating segment embeddings."""
  if labeled:
    for sent1, sent2, label in batch:
      value_vector = np.concatenate(
          ([cls_id], sent1, [sep_id], sent2, [sep_id]))

      segment_embs = np.zeros(
          sent1.shape[0] + sent2.shape[0] + 3, dtype=np.int32)
      second_sent_start = sent1.shape[0] + 2
      segment_embs[second_sent_start:] = 1
      yield value_vector, segment_embs, segment_embs, label, np.int32(1)
  else:
    for sent1, sent2 in batch:
      value_vector = np.concatenate(
          ([cls_id], sent1, [sep_id], sent2, [sep_id]))

      segment_embs = np.zeros(
          sent1.shape[0] + sent2.shape[0] + 3, dtype=np.int32)
      second_sent_start = sent1.shape[0] + 2
      segment_embs[second_sent_start:] = 1
      yield value_vector, segment_embs, segment_embs


def CreateBertInputs(double_sentence=True,  # pylint: disable=invalid-name
                     labeled=True,
                     cls_id=101,
                     sep_id=102):
  bert_inputs_fn = BertDoubleSentenceInputs if double_sentence else BertSingleSentenceInputs
  return functools.partial(
      bert_inputs_fn, labeled=labeled, cls_id=cls_id, sep_id=sep_id)


def mask_random_tokens(batch,
                       explicit_vocab_size=30522,
                       masking_prob=0.15,
                       cls_id=101,
                       sep_id=102,
                       mask_id=103,
                       vocab_start_id=999):
  """Prepares input for the masking task.

  Preparation consist in masking masking_prob percentage of non-special tokens
  at each input row; round(masking_prob * num_nonspecial_tokens) random tokens
  are selected out of which each token is either
  - replaced with [MASK] token with 80% probability,
  - replaced with random token with 10% probability,
  - or unchanged with 10%.
  The implentation is based on
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L342

  Examples:
  - batch is a stream with each row having tuple (token_ids,). Function yields
  rows of form (modified_token_ids, original_tokens, token_weights), where
  modified_token_ids have [MASK] tokens or random tokens according to the
  procedure described above.
  - batch is a stream with each row having tuple (token_ids, segment_embeddings,
  nsp_label, nsp_weight).Function yields rows of form (modified_token_ids,
  segment_embeddings, nsp_label, nsp_weight, original_tokens, token_weights).

  Args:
    batch: stream of inputs. Each row in the stream is a tuple which first
      element is an array of tokens
    explicit_vocab_size: the total size of the vocabulary.
    masking_prob: Determines percent of non-special tokens to be selected for
      masking.
    cls_id: id of the special CLS token.
    sep_id: id of the special SEP token.
    mask_id: id of the special MASK token.
    vocab_start_id: id of first non-special token in the vocabulary.

  Yields:
    a stream with tokens masked for MLM training and 2 appended arrays:
      - original tokens: a copy of original tokens used as a label for mlm
      training
      - token_weights: weights distributed uniformly over selected tokens (sum
      is 1). Other tokens have 0 weight.
  """
  for token_ids, *row_rest in batch:
    original_tokens = token_ids.copy()

    # choose tokens for prediction. Chooses 0.15 of
    # all non-special tokens
    is_special_token = np.logical_or(token_ids == cls_id,
                                     token_ids == sep_id)  # CLS and SEP tokens
    is_special_token = np.logical_or(is_special_token,
                                     token_ids == 0)  # padding
    viable_ids = np.arange(token_ids.shape[0])[~is_special_token]
    num_to_sample = round(masking_prob * viable_ids.shape[0])
    if num_to_sample == 0:
      # sentence is too short to select given percentage of tokens to mask
      continue
    candidate_ids = np.random.choice(viable_ids, num_to_sample, replace=False)

    # create weights
    token_weights = np.zeros(token_ids.shape)
    token_weights[candidate_ids] = 1 / candidate_ids.shape[0]

    prob_scores = np.random.random(candidate_ids.shape)

    # change 80 % of tokens to [MASK]
    mask_token_ids = candidate_ids[prob_scores < 0.8]
    token_ids[mask_token_ids] = mask_id

    # change 10% of tokens to random token
    random_token_ids = candidate_ids[(0.8 <= prob_scores) & (prob_scores < 0.9)]
    token_ids[random_token_ids] = np.random.randint(vocab_start_id,
                                                    explicit_vocab_size,
                                                    random_token_ids.shape[0])

    # rest (10%) is left unchaged
    yield (token_ids, *row_rest, original_tokens, token_weights)


def BertNextSentencePredictionInputs(dataset_name,  # pylint: disable=invalid-name
                                     data_dir=None,
                                     text_key='text',
                                     train=True,
                                     shuffle_size=50000):
  """Defines a stream for the next sentence prediction task."""
  stream = TFDS(
      dataset_name,
      data_dir=data_dir,
      tfds_preprocess_fn=functools.partial(
          t5.data.preprocessors.next_sentence_prediction,
          text_key=text_key,
          label_sentences=True,
          buffer_size=shuffle_size),
      keys=['inputs', 'targets'],
      train=train)

  def split_stream(generator=None):
    # split string with 'sentence1:' and 'sentence2:' into two separate strings
    for text, target in stream(generator):
      text_str = str(text)[:-1]  # removes last '"' which is always at the end
      sentences = text_str.split('sentence1: ')[1].split(' sentence2: ')
      if len(sentences) != 2:
        # 'sentence2:' appeared in the text and got mixed up with the label
        continue
      sent1, sent2 = sentences
      yield sent1, sent2, target == 'next'

  return split_stream


def LoadJSONRows(dir_path=gin.REQUIRED, fname=gin.REQUIRED):
  path = os.path.join(dir_path, fname)
  with open(path, 'r') as f:
    # loads everything into memory
    ds = [json.loads(line) for line in f.readlines()]

  def iterator(generator=None):
    while True:
      for el in ds:
        yield el
  return iterator


def AQuAExtendedQuestion():
  """
  Merges question with possible options into one string and
  returns tuple with 3 elements: extended string, rationale and correct answer
  """
  def aqua_extended_question(ds=None):
    for datapoint in ds:
      ext_question = datapoint['question'] + ' ; ' + ' ; '.join(datapoint['options'])
      correct_answer_label = ord(datapoint['correct']) - ord('A')
      yield ext_question, datapoint['rationale'], correct_answer_label
  return aqua_extended_question


def NROP(ROP_prob=0.5, split_token='\n'):
  """
  Creates task for predicting correct order of reasoning in rationale
  Rationale is string with steps separated by '\n'.
  """
  def nrop(ds=None):
    for ext_question, rationale, *_ in ds:
      if np.random.random() < ROP_prob:
        steps = rationale.split(split_token)
        if len(steps) >= 2:
          swap_id = np.random.randint(0, len(steps) - 1)
          steps[swap_id], steps[swap_id + 1] = steps[swap_id + 1], steps[swap_id]
          rationale = '\n'.join(steps)
          label = 0
        else:  # rationale is too short to swap
          label = 1
      else:
        label = 1
      yield ext_question, rationale, label
  return nrop


def AQuAFT():
  def aquaft(ds=None):
    for ext_question, _, label in ds:
      yield ext_question, label
  return aquaft


def CorpusToRandomChunks(dataset_name, num_tokens=512, train=True):  # pylint: disable=invalid-name
  return TFDS(
      dataset_name,
      tfds_preprocess_fn=functools.partial(
          t5.data.preprocessors.random_split_text,
          max_words_per_segment=num_tokens),
      train=train,
      keys=['text'])


@gin.configurable()
def get_glue_key(task_name=gin.REQUIRED):
  """Get glue key from the task name."""
  ext_task_name = task_name if task_name.startswith(
      'glue') else f'glue/{task_name}'
  try:
    glue_keys = {
        'glue/cola': ('sentence',),
        'glue/sst2': ('sentence',),
        'glue/mrpc': ('sentence1', 'sentence2'),
        'glue/qqp': ('question1', 'question2'),
        'glue/stsb': ('sentence1', 'sentence2'),
        'glue/mnli': ('premise', 'hypothesis'),
        'glue/qnli': ('question', 'sentence'),
        'glue/rte': ('sentence1', 'sentence2'),
        'glue/wnli': ('sentence1', 'sentence2'),
    }
    return (*glue_keys[ext_task_name], 'label')
  except KeyError:
    raise KeyError(
        f'Wrong task name entered, available glue tasks: {list(glue_keys.keys())}. Entered: {task_name}'
    )


def get_glue_t5_labels(dataset_name):
  """Get glue labels for T5 from the task name."""
  ext_task_name = dataset_name if dataset_name.startswith(
      'glue') else f'glue/{dataset_name}'
  try:
    # Labels inferred from the T5 paper: https://arxiv.org/pdf/1910.10683.pdf
    glue_t5_labels = {
        'glue/cola': ('unacceptable', 'acceptable'),
        'glue/sst2': ('negative', 'positive'),
        'glue/mrpc': ('not_equivalent', 'equivalent'),
        'glue/qqp': ('not_duplicate', 'duplicate'),
        # Requires processing of floats
        # 'glue/stsb': ('sentence1', 'sentence2'),
        'glue/mnli': ('entailment', 'neutral', 'contradiction'),
        'glue/qnli': ('entailment', 'not_entailment'),
        'glue/rte': ('entailment', 'not_entailment'),
        # Used for evaluation and for training of T5.
        # As explained in Section 2.4 of https://arxiv.org/pdf/1910.10683.pdf
        # it has an overlap with WSC from Super-GLUE.
        # 'glue/wnli': ('sentence1', 'sentence2'),
    }
    return glue_t5_labels[ext_task_name]
  except KeyError:
    raise KeyError(
        f'Wrong task name entered, available glue tasks: {list(glue_t5_labels.keys())}. Entered: {dataset_name}'
    )


def get_t5_splits(dataset_name, train=True):
  """Get splits for glue tasks."""
  # Splits listed in https://www.tensorflow.org/datasets/catalog/glue
  glue_t5_labels = collections.defaultdict(lambda: ('train', 'validation'))
  glue_t5_labels['glue/mnli'] = ('train', 'validation_matched')
  if train:
    return glue_t5_labels[dataset_name][0]
  else:
    return glue_t5_labels[dataset_name][1]


def CreateT5GlueInputs(  # pylint: disable=invalid-name
    dataset_name='glue/qnli',
    split=None,
    train=True,
    label_names=('entailment', 'not_entailment')):
  """Prepares glue inputs for T5 models using standard T5 preprocessor."""

  label_names = get_glue_t5_labels(dataset_name)
  if not split:
    split = get_t5_splits(dataset_name, train)
  benchmark_name = dataset_name.split('/')[1]
  dataset = tfds.load(name=dataset_name, split=split)
  proc_dataset = generic_text_dataset_preprocess_fn(
      dataset,
      spm_path=t5.data.DEFAULT_SPM_PATH,
      text_preprocess_fns=[
          lambda ds, training: t5.data.preprocessors.glue(  # pylint: disable=g-long-lambda
              ds,
              benchmark_name=benchmark_name,
              label_names=label_names)
      ],
      copy_pretokenized=True,
      debug_print_examples=True,
      debug_print_examples_rate=0.05)

  def t5_yield_examples(generator=None):
    del generator
    while True:
      for example in proc_dataset:
        input_values = example['inputs']
        target_values = example['targets']
        yield (fastmath.numpy.array(input_values),
               fastmath.numpy.array(target_values),
               fastmath.numpy.array([1] * len(target_values)))

  return t5_yield_examples
