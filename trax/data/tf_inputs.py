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
"""TensorFlow data sources and associated prepocessing functions."""

import functools
import itertools
import json
import math
import os
import random
import re

from absl import logging
import gin
import jax
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from trax import data
from trax import fastmath
from trax import layers as tl
from trax import supervised
from trax.data import debug_data_pipeline
from trax.data import text_encoder
from trax.fastmath import numpy as jnp

# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
# TODO(lukaszkaiser): can we improve efficiency, should that be changed?
_MAX_SKIP_EXAMPLES = 1e5


def t5_data():
  """Get the T5 data module if available."""
  module = None
  try:
    import t5.data  # pylint: disable=g-import-not-at-top
    module = t5.data
  except AttributeError as e:
    logging.error('pip install t5')
    raise e
  return module


def no_preprocess(dataset, training):
  del training
  return dataset


def t2t_problems():
  # Load t2t problems on request only, this should save some import time.
  from tensor2tensor import problems_colab as t2tp  # pylint: disable=g-import-not-at-top
  return t2tp


# TODO(jonni): Rename function to better match its return values.
@gin.configurable(module='trax.data')
def data_streams(dataset_name,
                 data_dir=None,
                 preprocess_fn=no_preprocess,
                 bare_preprocess_fn=None,
                 shuffle_buffer_size=1024,
                 eval_holdout_size=0,
                 input_name=None,
                 target_name=None):
  """Creates `(train, eval)` data sources from ``dataset_name``.

  Args:
    dataset_name: Name of dataset belonging to TFDS or T2T. T2T dataset names
      must start with ``'t2t_'``.
    data_dir: Directory where the data is located.
    preprocess_fn: Function to use for pre-processing after appending targets to
      inputs.
    bare_preprocess_fn: Function to use for pre-processing before appending
      targets to inputs.
    shuffle_buffer_size: Size of the shuffle buffer.
    eval_holdout_size: If greater than 0, specifies a fraction of training data
      to siphon off and use as eval data, in place of an separate eval split.
    input_name: Name of the inputs from the dictionary.
    target_name: Name of the outputs either from the dictionary or as a result
      of post-processing.

  Returns:
    A pair of functions, `(f, g)` for use as data sources; call `f()` to get an
    iterator of training data samples, and call `g()` to get an iterator of eval
    data samples.
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
                            eval_shuffle_files=False,
                            use_alt_eval=False,
                            subsplit=None):
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
    use_alt_eval: If True, use the dataset's alternate/secondary eval split;
      else use the dataset's default/only eval split. Currently, only the
      `glue/mnli` dataset provides an alternate eval split, and this arg is
      ignored for other datasets.
    subsplit: a pair of floats (x, y), both in [0, 1], saying which part of the
      full training dataset we should return (default: all of it, [0, 1]).

  Returns:
    a 4-tuple consisting of:
     * the train tf.Dataset
     * the eval tf.Dataset
     * information about features: a python dictionary with feature names
         as keys and an object as value that provides .shape and .n_classes.
     * supervised_keys: information what's the input and what's the target,
         ie., a pair of lists with input and target feature names.
  """
  logging.info('Building TF data pipeline for %s', dataset_name)
  if dataset_name.startswith('t2t_'):
    return _train_and_eval_dataset_v1(dataset_name[4:], data_dir,
                                      train_shuffle_files, eval_shuffle_files)
  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  info = dataset_builder.info
  splits = dataset_builder.info.splits
  if dataset_name != 'c4/multilingual' and tfds.Split.TRAIN not in splits:
    raise ValueError('To train we require a train split in the dataset.')
  train_split = tfds.Split.TRAIN if dataset_name != 'c4/multilingual' else 'en'
  eval_split = None
  train_examples = info.splits[train_split].num_examples
  eval_holdout_examples = int(train_examples * eval_holdout_size)
  if eval_holdout_examples > 0 or subsplit is not None:
    if subsplit is None:
      subsplit = (0, 1)
    n_train = train_examples - eval_holdout_examples
    train_start = int(n_train * subsplit[0])
    train_end = int(n_train * subsplit[1])
    if train_end - train_start < 1:
      raise ValueError('Requested train subsplit has no examples: '
                       'n_train %d subsplit %s' % (n_train, subsplit))
    # Eval holdout examples from the end of the training set.
    if eval_holdout_examples > 0:
      eval_split = f'{train_split}[-{eval_holdout_examples}:]'
    # Shard the training set for this host.
    train_split = f'{train_split}[{train_start}:{train_end}]'

  if dataset_name == 'glue/mnli':
    eval_split = (
        'validation_mismatched' if use_alt_eval else 'validation_matched')
  elif dataset_name == 'c4/multilingual':
    eval_split = 'en-validation'
  elif eval_split is None:
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


# TODO(jonni): Consider renaming this function.
@gin.configurable(module='trax.data')
def TFDS(  # pylint: disable=invalid-name
    dataset_name,
    data_dir=None,
    tfds_preprocess_fn=None,
    keys=None,
    train=True,
    use_alt_eval=False,
    shuffle_train=True,
    host_id=None,
    n_hosts=None,
    eval_holdout_size=0):
  """Creates a data source from TensorFlow dataset ``dataset_name``.

  Args:
    dataset_name: Name of the dataset, as registered in TensorFlow datasets
      (e.g., ``'glue/mnli'``).
    data_dir: Directory where the data is located.
    tfds_preprocess_fn: If specified, function that applies to items in raw
      dataset (before selecting specific features).
    keys: Tuple of dataset-specific strings that select features from the
      dataset.
    train: If True, select the training split from the dataset; else select an
      eval split.
    use_alt_eval: If True, and if ``train`` is False, select the dataset's
      alternate eval split if it has one (or fall back to the dataset's only
      eval split). This currently affects only the `glue/mnli` dataset.
    shuffle_train: If True, have TensorFlow pre-shuffle the training data; else
      receive training data in deterministic sequence.
    host_id: Integer id used for tracking data subsplits, in cases where
      ``n_hosts`` > 1.
    n_hosts: If greater than 1, prepare data subsplits for the given number of
      hosts.
    eval_holdout_size: If greater than 0, specifies a fraction of training data
      to siphon off and use as eval data, in place of an separate eval split.

  Returns:
    A function `f` for use as a training or eval data source; call `f()` to get
    an iterator of data samples.
  """
  data_dir = download_and_prepare(dataset_name, data_dir)

  host_id = jax.process_index() if host_id is None else host_id
  n_hosts = n_hosts or jax.host_count()
  if n_hosts > 1:
    subsplit = (host_id / n_hosts, (host_id + 1) / n_hosts)
  else:
    subsplit = None
  train_data, eval_data, _ = (
      _train_and_eval_dataset(dataset_name,
                              data_dir,
                              eval_holdout_size,
                              train_shuffle_files=shuffle_train,
                              use_alt_eval=use_alt_eval,
                              subsplit=subsplit))
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
@debug_data_pipeline.debug_pipeline
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
      output = tuple(new_example)
      yield output
    elif isinstance(example, dict):
      new_example = {}
      for k in example:
        if keys is None or k in keys:
          new_example[k] = np.array(vocab.encode(example[k])) + n_reserved_ids
        else:
          new_example[k] = example[k]
      yield new_example
    else:
      output = np.array(vocab.encode(example)) + n_reserved_ids
      yield output


@gin.configurable(module='trax.data')
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


@gin.configurable(module='trax.data')
def ConvertToUnicode(keys=None):  # pylint: disable=invalid-name
  """Converts to Unicode UTF-8 elements of an example.

  Useful for when TFDS outputs byte arrays. All of the errors of the conversion
  are ignored.

  Args:
    keys: tuple/list of example dimensions to convert.

  Returns:
    Function converting chosen elements of an example to UTF-8.
  """

  @debug_data_pipeline.debug_pipeline
  def _convert_to_unicode_str(stream):
    for example in stream:
      if isinstance(example, (list, tuple)):
        new_example = []
        for i, x in enumerate(example):
          if keys is None or i in keys:
            new_example.append(_to_unicode(x))
          else:
            new_example.append(x)
        output = tuple(new_example)
        yield output
      elif isinstance(example, dict):
        new_example = {}
        for k in example:
          if keys is None or k in keys:
            new_example[k] = _to_unicode(example[k])
          else:
            new_example[k] = example[k]
        yield new_example
      else:
        output = _to_unicode(example)
        yield output

  return _convert_to_unicode_str


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


def _get_vocab(vocab_type='subword', vocab_file=None, vocab_dir=None,
               extra_ids=0):
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
  return t5_data().SentencePieceVocabulary(sentencepiece_model_file=path,
                                           extra_ids=extra_ids)


# Makes the function accessible in gin configs, even with all args denylisted.
@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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
@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
def downsampled_imagenet_flatten_bare_preprocess(dataset, training):
  """Preprocessing for downsampled_imagenet.

  Args:
    dataset: the dataset.
    training: unused option.

  Returns:
    Flattened dataset.

  Preprocessing for downsampled_imagenet 32x32 and 64x64 generation from
  http://arxiv.org/abs/1601.06759 (page 8).
  """
  del training

  def flatten_image(features):
    img = features['image']
    flat = tf.cast(tf.reshape(img, [-1]), tf.int64)

    new_features = {'image': flat}
    return new_features

  return dataset.map(flatten_image)


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
def squeeze_targets_preprocess(dataset, training):
  """Pre-processing function that squeezes last axis of targets."""
  del training

  def squeeze(features, targets):
    if targets.shape[-1] == 1:
      targets = tf.squeeze(targets, axis=-1)
    return features, targets

  dataset = dataset.map(squeeze)
  return dataset


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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
@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['hparams'])
def bair_robot_pushing_hparams(hparams=None,
                               video_num_input_frames=1,
                               video_num_target_frames=15):
  if hparams is not None:
    hparams.video_num_input_frames = video_num_input_frames
    hparams.video_num_target_frames = video_num_target_frames
  else:
    return video_num_input_frames, video_num_target_frames


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


def sentencepiece_tokenize(stream, spm_path=None, extra_ids=0):
  """Sentencepiece tokenization."""
  spm_path = spm_path or t5_data().DEFAULT_SPM_PATH
  vocab_file = os.path.basename(spm_path)
  vocab_dir = os.path.dirname(spm_path)
  vocab = _get_vocab(vocab_type='sentencepiece',
                     vocab_file=vocab_file,
                     vocab_dir=vocab_dir,
                     extra_ids=extra_ids)
  for example in stream:
    # example could either be str or (str,)
    if isinstance(example, tuple):
      example = example[0]
    yield np.array(vocab.encode(example))


@gin.configurable(module='trax.data')
def SentencePieceTokenize(  # pylint: disable=invalid-name
    spm_path=None,
    extra_ids=0):
  """Returns a function that maps text to integer arrays."""
  return lambda g: sentencepiece_tokenize(  # pylint: disable=g-long-lambda
      g,
      spm_path=spm_path,
      extra_ids=extra_ids)


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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
    spm_path = spm_path or t5_data().DEFAULT_SPM_PATH
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
def c4_bare_preprocess_fn(dataset,
                          training=True,
                          spm_path=None,
                          copy_pretokenized=True,
                          sequence_length=None):
  """Returns a dataset that contains 'inputs' and 'targets' from C4."""
  # Set target key to be equal to the text content.
  dataset = t5_data().preprocessors.rekey(
      dataset, key_map={
          'targets': 'text',
          'inputs': None
      })

  # Vocabulary for tokenization.
  extra_ids = 0
  vocab = t5_data().SentencePieceVocabulary(
      sentencepiece_model_file=spm_path or t5_data().DEFAULT_SPM_PATH,
      extra_ids=extra_ids)
  feature = t5_data().Feature(vocab)
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
  dataset = t5_data().preprocessors.unsupervised(
      dataset, sequence_length=sequence_length, output_features=output_features)

  # Add EOS.
  dataset = add_eos_to_output_features(dataset, training)

  # Truncate and then pad the examples -- all examples have the same shape.
  dataset = truncate_dataset_on_len(dataset, training, sequence_length, True)
  dataset = pad_dataset_to_length(dataset, training, sequence_length)

  return dataset


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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


@gin.configurable(module='trax.data', denylist=['dataset', 'training'])
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
  vocab = t5_data().SentencePieceVocabulary(
      sentencepiece_model_file=spm_path or t5_data().DEFAULT_SPM_PATH,
      extra_ids=extra_ids)
  feature = t5_data().Feature(vocab)
  output_features = {'targets': feature, 'inputs': feature}

  # Tokenize the inputs and targets.
  dataset = t5_data().preprocessors.tokenize(
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


@gin.configurable(module='trax.data')
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
  f = getattr(t5_data().preprocessors, name)
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


@gin.configurable(module='trax.data')
def CreateBertInputs(double_sentence=True,  # pylint: disable=invalid-name
                     labeled=True,
                     cls_id=101,
                     sep_id=102):
  bert_inputs_fn = BertDoubleSentenceInputs if double_sentence else BertSingleSentenceInputs
  return functools.partial(
      bert_inputs_fn, labeled=labeled, cls_id=cls_id, sep_id=sep_id)


@gin.configurable(module='trax.data')
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


@gin.configurable(module='trax.data')
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
          t5_data().preprocessors.next_sentence_prediction,
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


@gin.configurable(module='trax.data')
def CorpusToRandomChunks(dataset_name, num_tokens=512, train=True):  # pylint: disable=invalid-name
  return TFDS(
      dataset_name,
      tfds_preprocess_fn=functools.partial(
          t5_data().preprocessors.random_split_text,
          max_words_per_segment=num_tokens),
      train=train,
      keys=['text'])


_GLUE_KEYS = {
    'cola': ('sentence',),
    'sst2': ('sentence',),
    'mrpc': ('sentence1', 'sentence2'),
    'qqp': ('question1', 'question2'),
    'stsb': ('sentence1', 'sentence2'),
    'mnli': ('premise', 'hypothesis'),
    'qnli': ('question', 'sentence'),
    'rte': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}


# Labels inferred from the T5 paper: https://arxiv.org/pdf/1910.10683.pdf
_GLUE_LABELS = {
    'cola': ('unacceptable', 'acceptable'),
    'sst2': ('negative', 'positive'),
    'mrpc': ('not_equivalent', 'equivalent'),
    'qqp': ('not_duplicate', 'duplicate'),
    'stsb': ('sentence1', 'sentence2'),
    'mnli': ('entailment', 'neutral', 'contradiction'),
    'qnli': ('entailment', 'not_entailment'),
    'rte': ('entailment', 'not_entailment'),
    'wnli': ('sentence1', 'sentence2'),
}

# Defining separate <Foo>TrainStream and <Foo>EvalStream functions (below)
# makes gin configuration expressions more direct. A single gin line can
# configure each; for example:
#
#   BertGlueTrainStream.benchmark= 'mnli'
#   BertGlueEvalStream.benchmark = 'mnli'


# pylint: disable=invalid-name
@gin.configurable(module='trax.data')
def BertGlueTrainStream(benchmark=gin.REQUIRED):
  """Returns a Bert-preprocessed training stream for ``benchmark``.

  Args:
    benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
        ``'mnli'``, ``'rte'``.
  """
  return _BertGlueDataStream(benchmark + '_t')


# GLUE evals need special handling because one eval in particular, MNLI, has
# two different eval sets: "matched" and "mismatched". The code in this module
# distinguishes between the two using the suffixes '_e' versus '_e2',
# respectively.
def _ensure_eval_suffix(benchmark):
  """Returns a string ending in an eval suffix; adds ``'_e'`` suffix if needed.

  Args:
    benchmark: Name of a benchmark or task, that might already include an
        eval-indicating suffix (``'_e'`` or ``'_e2'``).
  """
  if benchmark.endswith('_e') or benchmark.endswith('_e2'):
    return benchmark
  else:
    return benchmark + '_e'


@gin.configurable(module='trax.data')
def BertGlueEvalStream(benchmark=gin.REQUIRED):
  """Returns a Bert-preprocessed eval data stream for ``benchmark``.

  Args:
    benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
        ``'mnli'``, ``'rte'``. If the benchmark includes an alternate
        eval (e.g., MNLI's "mismatched" eval/validation split), you can
        specify it with an ``'_e2'`` suffix, e.g., ``'mnli_e2'``.
  """
  return _BertGlueDataStream(_ensure_eval_suffix(benchmark))


def _BertGlueDataStream(benchmark_id):
  """Returns a Bert-preprocessed data stream for ``benchmark_id``.

  Args:
    benchmark_id: String that indicates the name and data split of a GLUE
        benchmark. Data splits are indicated as underscore suffixes, e.g.,
        ``'cola_t'`` (Cola benchmark, training split), ``'rte_e'`` (RTE
        benchmark, eval/validation split), and ``'mnli_e2'`` (MNLI benchmark,
        alternate "mismatched" eval/validation split).
  """
  benchmark_id = _ensure_eval_suffix(benchmark_id)
  benchmark, split = benchmark_id.rsplit('_', 1)
  glue_data = TFDS(f'glue/{benchmark}',
                   keys=_GLUE_KEYS[benchmark],
                   train=(split == 't'),
                   use_alt_eval=(split == 'e2'))
  return data.Serial(
      glue_data,
      data.Tokenize(),
      data.CreateBertInputs(),
      data.Shuffle(),
      data.PadToLength(),
      data.TruncateToLength(),
      data.Batch(),
  )


@gin.configurable(module='trax.data')
def T5GlueTrainStream(benchmark=gin.REQUIRED):
  """Returns a T5-preprocessed training data stream for ``benchmark``.

  Args:
    benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
        ``'mnli'``, ``'rte'``.
  """
  return _T5GlueDataStream(benchmark + '_t')


@gin.configurable(module='trax.data')
def T5GlueTrainStreamsParallel(benchmark_list=gin.REQUIRED,
                               counters=None,
                               reweight_by_minimum=False,
                               gradually_reweight=False):
  """Returns a parallel set of training streams, based on ``benchmark_list``.

  Args:
    benchmark_list: List of simple lower-case names of GLUE benchmarks, e.g.,
        ``'cola'``, ``'mnli'``, ``'rte'``.
    counters: a list of counters to be passed to data.Parallel, e.g.,
    [8551, 392702, 2490] would be a reasonable counterpart to
    benchmark_list = ["cola", "mnli", "rte"], see
    https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/glue_utils.py#L42
    for more details on counters.
    reweight_by_minimum: divide by the minimal counter.
    gradually_reweight: a more refined reweighting policy, see inputs.py
      for more details.
  """
  stream_list = list(map(T5GlueTrainStream, benchmark_list))
  return data.Parallel(
      stream_list,
      counters=counters,
      reweight_by_minimum=reweight_by_minimum,
      gradually_reweight=gradually_reweight)()


@gin.configurable(module='trax.data')
def T5GlueEvalStream(benchmark=gin.REQUIRED):
  """Returns a T5-preprocessed eval data stream for ``benchmark``.

  Args:
    benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
        ``'mnli'``, ``'rte'``. If the benchmark includes an alternate
        eval (e.g., MNLI's "mismatched" eval/validation split), you can
        specify it with an ``'_e2'`` suffix, e.g., ``'mnli_e2'``.
  """
  return _T5GlueDataStream(_ensure_eval_suffix(benchmark))


@gin.configurable(module='trax.data')
def T5GlueEvalStreamsParallel(benchmark_list=gin.REQUIRED):
  """Returns a parallel set of T5 eval streams, based on ``benchmark_list``.

  Args:
    benchmark_list: List of strings, each of which is a simple lower-case name
        of a GLUE benchmark, e.g., ``'cola'``, ``'mnli'``, ``'rte'``. If a
        benchmark includes an alternate eval (e.g., MNLI's "mismatched"
        eval/validation split), you can specify it with an ``'_e2'`` suffix,
        e.g., ``'mnli_e2'``.
  """
  stream_list = list(map(T5GlueEvalStream, benchmark_list))
  return data.Parallel(stream_list)()


def _T5GlueDataStream(benchmark_id, t5_tokenization=False):
  """Returns a T5-preprocessed data stream for ``benchmark_id``.

  Args:
    benchmark_id: String that indicates the name and data split of a GLUE
        benchmark. Data splits are indicated as underscore suffixes, e.g.,
        ``'cola_t'`` (Cola benchmark, training split), ``'rte_e'`` (RTE
        benchmark, eval/validation split), and ``'mnli_e2'`` (MNLI benchmark,
        alternate "mismatched" eval/validation split).
    t5_tokenization: if true, then use t5_tokenization.
  """
  return data.Serial(
      _t5_glue_data_split(benchmark_id)
      if t5_tokenization else _t5_glue_data_split_no_token(benchmark_id),
      data.Tokenize(),
      data.Shuffle(),
      data.PadToLength(),
      data.TruncateToLength(),
      data.Batch(),
  )


@gin.configurable(module='trax.data')
def T5GlueEvalTasks(benchmark_list=gin.REQUIRED):
  """Returns a list of T5 GLUE eval tasks, based on ``benchmark_list``.

  Args:
    benchmark_list: List of strings, each of which indicates the name and
        data split of a GLUE benchmark. Data splits are indicated as underscore
        suffixes, e.g., ``'cola_t'`` (Cola benchmark, training split),
        ``'rte_e'`` (RTE benchmark, eval/validation split), and ``'mnli_e2'``
        (MNLI alternate "mismatched" eval/validation split).
  """
  task_list = list(map(_T5GlueEvalTask, benchmark_list))
  return task_list


def _T5GlueEvalTask(benchmark_id):
  """Returns a T5 GLUE eval task, based on ``benchmark_id``."""
  eval_data = T5GlueEvalStream(benchmark_id)
  benchmark_id = _ensure_eval_suffix(benchmark_id)
  metrics = [tl.WeightedCategoryAccuracy(), tl.SequenceAccuracy()]
  benchmark, split = benchmark_id.rsplit('_', 1)
  if benchmark == 'cola':
    name_upper = 'Cola'
  elif benchmark == 'mnli':
    name_upper = 'MNLI_matched' if split == 'e' else 'MNLI_mismatched'
  else:
    name_upper = benchmark.upper()
  return supervised.training.EvalTask(
      eval_data(),
      metrics,
      metric_names=[f'{name_upper} accuracy',
                    f'{name_upper} sequence accuracy'])


def _t5_glue_data_split_no_token(benchmark_id):
  """Returns a GLUE data split prepared with the standard T5 preprocessor."""
  benchmark, split = _t5_glue_benchmark_and_split(benchmark_id)
  dataset = tfds.load(name=f'glue/{benchmark}', split=split)
  processed_dataset = t5_data().preprocessors.glue(  # pylint: disable=g-long-lambda
      dataset,
      benchmark_name=benchmark,
      label_names=_GLUE_LABELS[benchmark])

  def stream_of_inputs_targets_weights(generator=None):
    del generator
    while True:
      for example in processed_dataset:
        input_values = example['inputs'].numpy()
        target_values = example['targets'].numpy()
        yield (input_values,
               target_values,
               jnp.array([1] * len(target_values)))

  return stream_of_inputs_targets_weights


def _t5_glue_data_split(benchmark_id):
  """Returns a GLUE data split prepared with the standard T5 preprocessor."""
  benchmark, split = _t5_glue_benchmark_and_split(benchmark_id)
  dataset = tfds.load(name=f'glue/{benchmark}', split=split)
  processed_dataset = generic_text_dataset_preprocess_fn(
      dataset,
      spm_path=t5_data().DEFAULT_SPM_PATH,
      text_preprocess_fns=[
          lambda ds, training: t5_data().preprocessors.glue(  # pylint: disable=g-long-lambda
              ds,
              benchmark_name=benchmark,
              label_names=_GLUE_LABELS[benchmark])
      ],
      copy_pretokenized=True,
      debug_print_examples=True,
      debug_print_examples_rate=0.05)
  dataset_as_numpy = tfds.as_numpy(processed_dataset)

  def stream_of_inputs_targets_weights(generator=None):
    del generator
    while True:
      for example in dataset_as_numpy:
        input_values = example['inputs']
        target_values = example['targets']
        yield (jnp.array(input_values),
               jnp.array(target_values),
               jnp.array([1] * len(target_values)))

  return stream_of_inputs_targets_weights


def _t5_glue_benchmark_and_split(benchmark_id):
  benchmark, mode = benchmark_id.rsplit('_', 1)
  if mode == 't':
    split = 'train'
  elif benchmark == 'mnli':
    split = 'validation_mismatched' if mode == 'e2' else 'validation_matched'
  else:
    split = 'validation'
  return benchmark, split
# pylint: enable=invalid-name


def compute_single_result(op_name, num_args):
  """An implementation of the most popular ops from the MathQA dataset."""
  # See https://gitlab.cs.washington.edu/amini91/mathqa-categorization/
  # and specfically line 142 and following in new_DataStructure.py
  # for an implementation which covers more details.
  if op_name == 'add':
    return num_args[0] + num_args[1]
  elif op_name == 'circle_arc':
    return num_args[0] / 360 * math.pi * 2 * num_args[1]
  elif op_name == 'circle_area':
    return math.pi * num_args[0]**2
  elif op_name == 'circle_sector_area':
    return num_args[1] / 360 * math.pi * (num_args[0]**2)
  elif op_name == 'circumface':
    return 2 * math.pi * num_args[0]
  elif op_name == 'choose':
    # Older versions of scipy may require scipy.misc.comb.
    return scipy.special.comb(num_args[0], num_args[1])  # pylint: disable=unreachable
  elif op_name == 'cosine':
    return math.cos(num_args[0])
  elif op_name == 'cube_edge_by_volume':
    return num_args[0]**(1 / 3)
  elif op_name == 'combined_work':
    return 1 / (
        min(num_args[0], 1 / num_args[0]) + min(num_args[1], 1 / num_args[1]))
  elif op_name == 'count_interval':
    return num_args[0] - num_args[1] + 1
  elif op_name == 'diagonal':
    return math.sqrt(num_args[0]**2 + num_args[1]**2)
  elif op_name == 'divide' or op_name == 'speed':
    if num_args[1] != 0:
      return num_args[0] / num_args[1]
    else:
      return 0
  elif op_name == 'factorial':
    return math.factorial(min(15, int(num_args[0])))
  elif op_name == 'floor':
    return math.floor(num_args[0])
  elif op_name == 'find_work':
    return 1 / (
        max(
            min(num_args[0], 1 / num_args[0]), min(
                num_args[1], 1 / num_args[1])) - min(
                    min(num_args[0], 1 / num_args[0]),
                    min(num_args[1], 1 / num_args[1])))
  elif op_name == 'from_percent':
    return num_args[0] / 100
  elif op_name == 'gain_percent':
    return 100 + num_args[0]
  elif op_name == 'gcd':
    return scipy.gcd(int(num_args[0]), int(num_args[1]))
  elif op_name == 'inverse':
    if num_args[0] != 0:
      return 1 / num_args[0]
    else:
      return 0
  elif op_name == 'lcm':
    return scipy.lcm(int(num_args[0]), int(num_args[1]))
  elif op_name == 'log':
    return math.log(max(1e-5, num_args[0]), 2)
  elif op_name == 'loss_percent':
    return 100 - num_args[0]
  elif op_name == 'max':
    return max(num_args[0], num_args[1])
  elif op_name == 'multiply':
    return num_args[0] * num_args[1]
  elif op_name == 'negate_percent':
    return 100 - num_args[0]
  elif op_name == 'negate':
    return -num_args[0]
  elif op_name == 'original_price_before_loss':
    return num_args[1] * 100 / (100 + 1e-5 - num_args[0])
  elif op_name == 'original_price_before_gain':
    return num_args[1] * 100 / (100 + num_args[0])
  elif op_name == 'permutation':
    n, m = min(num_args[0], num_args[1]), max(num_args[0], num_args[1])
    return math.factorial(int(m)) / math.factorial(int(m - n))
  elif op_name == 'power':
    return num_args[0]**min(num_args[1], 5)
  elif op_name == 'percent':
    return num_args[0] / 100 * num_args[1]
  elif op_name == 'price_after_gain' or op_name == 'p_after_gain':
    return (1 + num_args[0] / 100) * num_args[1]
  elif op_name == 'price_after_loss' or op_name == 'price_after_loss':
    return (1 - num_args[0] / 100) * num_args[1]
  elif op_name == 'quadrilateral_area':
    return num_args[0] * (num_args[1] + num_args[2]) / 2
  elif op_name == 'reminder':
    return num_args[0] % num_args[1]
  elif op_name == 'rectangle_area':
    return num_args[0] * num_args[1]
  elif op_name == 'rectangle_perimeter':
    return 2 * (num_args[0] + num_args[1])
  elif op_name == 'rhombus_area':
    return num_args[0] * num_args[1] / 2
  elif op_name == 'sine':
    return math.sin(num_args[0])
  elif op_name == 'sqrt':
    return math.sqrt(max(0, num_args[0]))
  elif op_name == 'subtract':
    return num_args[0] - num_args[1]
  elif op_name == 'square_edge_by_perimeter':
    return num_args[0] / 4
  elif op_name == 'square_edge_by_area':
    return math.sqrt(num_args[0])
  elif op_name == 'square_area':
    return num_args[0]**2
  elif op_name == 'surface_cube':
    return 6 * num_args[0]**2
  elif op_name == 'surface_rectangular_prism':
    return 2 * (
        num_args[0] * num_args[1] + num_args[0] * num_args[2] +
        num_args[1] * num_args[2])
  elif op_name == 'semi_circle_perimiter':
    return math.pi * num_args[0] + 2 * num_args[0]
  elif op_name == 'square_perimeter' or op_name == 'rhombus_perimeter':
    return 4 * num_args[0]
  elif op_name == 'surface_sphere':
    return 4 * math.pi * num_args[0]**2
  elif op_name == 'speed_ratio_steel_to_stream':
    return (num_args[0] + num_args[1]) / (num_args[0] - num_args[1])
  elif op_name == 'speed_in_still_water':
    return (num_args[0] + num_args[1]) / 2
  elif op_name == 'stream_speed':
    return (num_args[0] - num_args[1]) / 2
  elif op_name == 'trapezium_area':
    return num_args[0] * (num_args[1] + num_args[2]) / 2
  elif op_name == 'triangle_area':
    return num_args[0] * num_args[1] / 2
  elif op_name == 'triangle_perimeter':
    return num_args[0] + num_args[1] + num_args[2]
  elif op_name == 'triangle_area_three_edges':
    # Heron's formula
    s = (num_args[0] + num_args[1] + num_args[2]) / 2
    return math.sqrt(
        max(0,
            s * (s - num_args[0]) * (s - num_args[1]) * (s - num_args[2])))
  elif op_name == 'union_prob':
    return num_args[0] + num_args[1] - num_args[2]
  elif op_name == 'negate_prob':
    return 1 - num_args[0]
  elif op_name == 'volume_cube':
    return num_args[0]**3
  elif op_name == 'volume_cone':
    return math.pi * num_args[0]**2 * num_args[1] / 3
  elif op_name == 'volume_cylinder':
    return math.pi * num_args[0]**2 * num_args[1]
  elif op_name == 'volume_rectangular_prism':
    return num_args[0] * num_args[1] * num_args[2]
  elif op_name == 'volume_sphere':
    return 4 / 3 * math.pi * num_args[0]**3


def compute_result(list_op, list_num):
  """Python execution of MathQA ops."""
  # The last of temporary results is the final answer.
  temporary_results = []
  for op in list_op:
    op_name = op.split('(')[0]
    start_bracket = op.find('(')
    end_bracket = op.find(')')
    op_args = op[start_bracket + 1:end_bracket].split(',')
    num_args = []
    for arg in op_args:
      # The hash stands for a number stored in temporary_results.
      # For example #2 refers to the third temporary result.
      if arg[0] == '#':
        temp_index = int(
            re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                       arg)[0])
        num_args.append(temporary_results[temp_index])
      # The n prefix stands for numbers which listed in list_num -
      # originally they were contained in the text.
      elif arg[0] == 'n':
        n_index = int(
            re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                       arg)[0])
        num_args.append(list_num[n_index])
      elif arg[0] == 'c':
        if arg == 'const_pi':
          constant = math.pi
        elif arg == 'const_deg_to_rad':
          constant = math.pi / 180
        else:
          consts = re.findall(
              r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', arg)
          if len(consts) == 1:
            constant = float(consts[0])
          else:
            constant1 = float(consts[0])
            constant2 = float('0.' + consts[1])
            constant = constant1 + constant2
        num_args.append(constant)
    temporary_results.append(compute_single_result(op_name, num_args))
  return temporary_results


def single_op_to_python_command(op_name, num_args):
  """An implementation of the most popular ops from the MathQA dataset."""
  # See https://gitlab.cs.washington.edu/amini91/mathqa-categorization/
  # and specfically line 142 and following in new_DataStructure.py
  # for an implementation which covers more details.
  if op_name == 'add':
    return '{} + {}'.format(num_args[0], num_args[1])
  elif op_name == 'circle_arc':
    return '{} / 360 * math.pi * 2 * {}'.format(num_args[0], num_args[1])
  elif op_name == 'circle_area':
    return 'math.pi * {}**2'.format(num_args[0])
  elif op_name == 'circle_sector_area':
    return '{} / 360 * math.pi * ({}**2)'.format(num_args[1], num_args[0])
  elif op_name == 'circumface':
    return '2 * math.pi * {}'.format(num_args[0])
  elif op_name == 'choose':
    # Older versions of scipy may require scipy.misc.comb.
    return 'scipy.special.comb({}, {})'.format(num_args[0], num_args[1])  # pylint: disable=unreachable
  elif op_name == 'cosine':
    return 'math.cos({})'.format(num_args[0])
  elif op_name == 'cube_edge_by_volume':
    return '{}**(1 / 3)'.format(num_args[0])
  elif op_name == 'combined_work':
    return '1 / (min({}, 1 / {}) + min({}, 1 / {}))'.format(
        num_args[0], num_args[0], num_args[1], num_args[1])
  elif op_name == 'count_interval':
    return '{} - {} + 1'.format(num_args[0], num_args[1])
  elif op_name == 'diagonal':
    return 'math.sqrt({}**2 + {}**2)'.format(num_args[0], num_args[1])
  elif op_name == 'divide' or op_name == 'speed':
    # safe divide
    if num_args[1] != 0:
      return '{} / {}'.format(num_args[0], num_args[1])
    else:
      return '0'
  elif op_name == 'factorial':
    return 'math.factorial(min(15, int({})))'.format(num_args[0])
  elif op_name == 'floor':
    return 'math.floor({})'.format(num_args[0])
  elif op_name == 'find_work':
    return ('1 / (max(min({}, 1 / {}), min({}, 1 / {})) - min(min({}, 1 / {}), '
            'min({}, 1 / {})))').format(num_args[0], num_args[0], num_args[1],
                                        num_args[1], num_args[0], num_args[0],
                                        num_args[1], num_args[1])
  elif op_name == 'from_percent':
    return '{} / 100'.format(num_args[0])
  elif op_name == 'gain_percent':
    return '100 + {}'.format(num_args[0])
  elif op_name == 'gcd':
    return 'scipy.gcd(int({}), int({}))'.format(num_args[0], num_args[1])
  elif op_name == 'inverse':
    # safe inverse
    if num_args[0] != 0:
      return '1 / {}'.format(num_args[0])
    else:
      return '0'
  elif op_name == 'lcm':
    return 'scipy.lcm(int({}), int({}))'.format(num_args[0], num_args[1])
  elif op_name == 'log':
    return 'math.log(max(1e-5, {}), 2)'.format(num_args[0])
  elif op_name == 'loss_percent':
    return '100 - {}'.format(num_args[0])
  elif op_name == 'max':
    return 'max({},{})'.format(num_args[0], num_args[1])
  elif op_name == 'multiply':
    return '{} * {}'.format(num_args[0], num_args[1])
  elif op_name == 'negate_percent':
    return '100 - {}'.format(num_args[0])
  elif op_name == 'negate':
    return '-{}'.format(num_args[0])
  elif op_name == 'original_price_before_loss':
    return '{} * 100 / (100 + 1e-5 - {})  # original price before loss'.format(
        num_args[1], num_args[0])
  elif op_name == 'original_price_before_gain':
    return '{} * 100 / (100 + {})  # original_price_before gain'.format(
        num_args[1], num_args[0])
  elif op_name == 'permutation':
    return ('math.factorial(int(max({}, {}))) / math.factorial(int(max({}, {}) '
            '- min({}, {})))  # find all permutations').format(
                num_args[0], num_args[1], num_args[0], num_args[1], num_args[0],
                num_args[1])
  elif op_name == 'power':
    return '{}**min({}, 5)'.format(num_args[0], num_args[1])
  elif op_name == 'percent':
    return '{} / 100 * {}'.format(num_args[0], num_args[1])
  elif op_name == 'price_after_gain' or op_name == 'p_after_gain':
    return '(1 + {} / 100) * {}'.format(num_args[0], num_args[1])
  elif op_name == 'price_after_loss' or op_name == 'price_after_loss':
    return '(1 - {} / 100) * {}'.format(num_args[0], num_args[1])
  elif op_name == 'quadrilateral_area':
    return '{} * ({} + {}) / 2  # quadrilateral area'.format(
        num_args[0], num_args[1], num_args[2])
  elif op_name == 'reminder':
    return '{} % {}'.format(num_args[0], num_args[1])
  elif op_name == 'rectangle_area':
    return '{} * {}  # area of rectangle'.format(num_args[0], num_args[1])
  elif op_name == 'rectangle_perimeter':
    return '2 * ({} + {})  # perimetere of rectangle'.format(
        num_args[0], num_args[1])
  elif op_name == 'rhombus_area':
    return '{} * {} / 2'.format(num_args[0], num_args[1])
  elif op_name == 'sine':
    return 'math.sin({})'.format(num_args[0])
  elif op_name == 'sqrt':
    return 'math.sqrt(max(0, {}))'.format(num_args[0])
  elif op_name == 'subtract':
    return '{} - {}'.format(num_args[0], num_args[1])
  elif op_name == 'square_edge_by_perimeter':
    return '{} / 4. # square edge given perimeter'.format(num_args[0])
  elif op_name == 'square_edge_by_area':
    return 'math.sqrt({})  # square edge given area'.format(num_args[0])
  elif op_name == 'square_area':
    return '{}**2'.format(num_args[0])
  elif op_name == 'surface_cube':
    return '6 * {}**2  # surface of a cube'.format(num_args[0])
  elif op_name == 'surface_rectangular_prism':
    return '2 * ({} * {} + {} * {} + {} * {})  # surface of a rectangular prism'.format(
        num_args[0], num_args[1], num_args[0], num_args[2], num_args[1],
        num_args[2])
  elif op_name == 'semi_circle_perimiter':
    return 'math.pi * {} + 2 * {}  # perimeter of a semi-circle'.format(
        num_args[0], num_args[0])
  elif op_name == 'square_perimeter' or op_name == 'rhombus_perimeter':
    return '4 * {}'.format(num_args[0])
  elif op_name == 'surface_sphere':
    return '4 * math.pi * {}**2'.format(num_args[0])
  elif op_name == 'speed_ratio_steel_to_stream':
    return '({} + {}) / ({} - {})'.format(num_args[0], num_args[1], num_args[0],
                                          num_args[1])
  elif op_name == 'speed_in_still_water':
    return '{} + {} / 2'.format(num_args[0], num_args[1])
  elif op_name == 'stream_speed':
    return '{} - {} / 2'.format(num_args[0], num_args[1])
  elif op_name == 'trapezium_area':
    return '{} * ({} + {}) / 2'.format(num_args[0], num_args[1], num_args[2])
  elif op_name == 'triangle_area':
    return '{} * {} / 2'.format(num_args[0], num_args[1])
  elif op_name == 'triangle_perimeter':
    return '{} + {} + {}  # perimeter of a triangle'.format(
        num_args[0], num_args[1], num_args[2])
  elif op_name == 'triangle_area_three_edges':
    return ("(lambda s, a, b, c: math.sqrt(max(0, s * (s - a) * (s - b) * (s - "
            "c))))(({} + {} + {}) / 2, {}, {}, {})  # Heron's formula").format(
                num_args[0], num_args[1], num_args[2], num_args[0], num_args[1],
                num_args[2])
  elif op_name == 'union_prob':
    return '{} + {} - {}'.format(num_args[0], num_args[1], num_args[2])
  elif op_name == 'negate_prob':
    return '1 - {}'.format(num_args[0])
  elif op_name == 'volume_cube':
    return '{}**3'.format(num_args[0])
  elif op_name == 'volume_cone':
    return 'math.pi * {}**2 * {} / 3'.format(num_args[0], num_args[1])
  elif op_name == 'volume_cylinder':
    return 'math.pi * {}**2 * {}'.format(num_args[0], num_args[1])
  elif op_name == 'volume_rectangular_prism':
    return '{} * {} * {}'.format(num_args[0], num_args[1], num_args[2])
  elif op_name == 'volume_sphere':
    return '4 / 3 * math.pi * {}**3'.format(num_args[0])


def compute_program(list_op):
  """Python execution of MathQA ops."""
  # The last of temporary results is the final answer.
  temporary_results = []
  num_op = 0
  for op in list_op:
    op_name = op.split('(')[0]
    start_bracket = op.find('(')
    end_bracket = op.find(')')
    op_args = op[start_bracket + 1:end_bracket].split(',')
    num_args = []
    for arg in op_args:
      # The hash stands for a number stored in temporary_results.
      # For example #2 refers to the third temporary result.
      if arg[0] == '#':
        temp_index = int(
            re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                       arg)[0])
        num_args.append('t{}'.format(temp_index))
      # The n prefix stands for numbers which listed in list_num -
      # originally they were contained in the text.
      elif arg[0] == 'n':
        # n_index = int(
        #     re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
        #                arg)[0])
        num_args.append(arg)
      elif arg[0] == 'c':
        if arg == 'const_pi':
          constant = math.pi
        elif arg == 'const_deg_to_rad':
          constant = math.pi / 180
        else:
          consts = re.findall(
              r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', arg)
          if len(consts) == 1:
            constant = float(consts[0])
          else:
            constant1 = float(consts[0])
            constant2 = float('0.' + consts[1])
            constant = constant1 + constant2
        num_args.append(str(constant))
    temporary_result = 't{} = {}'.format(
        num_op, single_op_to_python_command(op_name, num_args))
    temporary_results.append(temporary_result)
    num_op += 1
  return temporary_results


def compute_nums(question):
  """Finds numbers in a string and convert them to floats."""
  # The funny looking replace is needed to deal with numbers such as 4,000
  # TODO(henrykm) deal with numbers written as words "one", "two", ...
  return [
      float(num.replace(',', '')) for num in re.findall(
          r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', question)
  ]


def compute_ops(linear_formula):
  list_op = linear_formula.split('|')
  # In some cases the list of operations contains a superflous last element,
  # namely an empty string.
  if not list_op[-1]:
    list_op = list_op[:-1]
  return list_op


def process_single_mathqa_example(example):
  """Execute a single example and verify coherence of a MathQA problem.

  Args:
    example: a dictionary with the following fields: Problem - a natural
      language formulation of the problem Rationale - a natural language
      solution of the problem options - five possible answers ( a) b) c) d) and
      e) ) correct - the letter representing the correct answer
      annotated_formula - formula representing the full solution linear_formula
      - a string of operations separated by the | character, e.g.
      multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|
      multiply(#2,const_100)|divide(#3,#1)| category - a natural language
      description of the category to which a given problem belongs.

  Returns:
    answer_num: numerical answer contained in the example
    python_result: numerical answers computed in Python, including intermediate
      results. The answer_num should be close python_result[-1]
    list_op: list of arithmetic operations
    list_num: list of identified numbers in the text
  """
  question = example['Problem']
  list_num = compute_nums(question)
  list_op = compute_ops(example['linear_formula'])
  answers = example['options']
  correct_answer = example['correct']
  index = answers.find('{} )'.format(correct_answer))
  answer_string = re.findall(
      r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', answers[index:])
  # The if statement deals with empty lists - they are needed to treat
  # a correct non-numerical answer e) None of the above. Here we do not want
  # non-numerical answers, hence we return None.
  if answer_string:
    answer_num = float(
        re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                   answers[index:])[0].replace(',', ''))
  else:
    return None
  # The if statements below deals with answers written as fractions e.g.
  # a ) 1 / 2 , b ) 1 / 3 , c ) 1 / 5 , d ) 10 / 30 , e ) 2 / 5 ?
  index_end_of_answer = index + len(str(answer_num)) + 3
  if index_end_of_answer < len(answers) and answers[index_end_of_answer] == '/':
    answer_denom = float(
        re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                   answers[index_end_of_answer:])[0].replace(',', ''))
    answer_num /= answer_denom
  python_result = compute_result(list_op, list_num)
  python_program = compute_program(list_op)
  return answer_num, python_result, python_program, list_op, list_num


def convert_float_to_mathqa(number):
  floor = int(float(number))
  if floor == number:
    return 'const_' + str(floor)
  else:
    return 'const_' + str(floor) + '_' + str(number)[len(str(floor)) + 1:]


def convert_to_subtract(const_string):
  return 'subtract({},const_0)'.format(const_string)


def execute_mathqa_dsl_program(problem, dsl_code):
  """Executes the DSL code for a given problem.

  Args:
    problem: problem formulation (needed to get parameters).
    dsl_code: DSL code.

  Returns:
    the result of executing of the DSL code.
  """
  n0_loc = problem.find('n0')
  list_num = compute_nums(problem[n0_loc:])
  # The list contains _all_ numbers in the string, hence in particular
  # for n0 = 2.0 n1 = 3.0 we are getting list_num = [0.0, 2.0, 1.0, 3.0],
  # so that below we are filtering the odd occurrences.
  assert len(list_num) % 2 == 0
  list_num = [list_num[2 * i + 1] for i in range(int(len(list_num) / 2))]

  # dsl_code is a list of strings; since all DSL programs are single liners,
  # we need to guess the correct line. For now we use the same location as in
  # in the ground truth examples, that is the first line.
  list_op = compute_ops(dsl_code[0])

  try:
    results = compute_result(list_op, list_num)[-1]
  except:  # pylint: disable=bare-except
    results = None
  return results


def is_number(s):
  try:
    float(s)
    return True
  except:  # pylint: disable=bare-except
    return False


def execute_mathqa_program(problem, program):
  """Executes the DSL code for a given problem.

  Args:
    problem: problem formulation (not needed, but we want the same API as
      in the DSL case).
    program: Python code.

  Returns:
    the result of executing of the Python code.
  """
  del problem  # problem only needed in the DSL version.
  # Programs are lists of strings. We need to concatenate them in order to exec.
  program = '\n'.join(program)
  var_dict = {}
  try:
    # The logic of this is the following: if exec with timeout is working
    # without exceptions, then we can call exec again and gather the variables.
    exec(program, globals(), var_dict)  # pylint: disable=exec-used
    if 'answer' in var_dict and is_number(var_dict['answer']):
      return float(var_dict['answer'])
    else:
      return None
  except:  # pylint: disable=bare-except
    return None


@gin.configurable(module='trax.data')
def CreateMathQAInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    test=False,
    challenge=False,
    tolerance=0.01,
    cumulative=True,
    python_code=False,
    full_dict=False,
    partial_results=True,
    nlp_rationale=False,
    correct_answer=False,
    answer_in_mathqa_format=True,
    correct_answer_given_reasoning=False,
    category=False,
    order_prediction=False,
    reduced_operation_name=True,
    qed=False):
  """Prepares MathQA inputs.

  The generation procedure leaves a lot parameters to be set by the user.
  Currently we support only correct examples in the following sense:
  python execution agrees with the declared answer up to 1%.

  According to this criterion wrong examples such as
  problem: calculate 85184 ÷ ? = 352
  operations ['multiply(n0,n1)']
  are ignored (this should be divide(n0,n1) in this case).

  Args:
    dataset_path: a path with the MathQA dataset.
    train: if True, then generate training examples; if train, test and
      challenge are set to False generate validation examples.
    test: if train is set to False and test is set to True,
      then generate test examples.
    challenge: if train and test are set to False and challenge is set to True,
      then generate challenge examples.
    tolerance: if for a given example relative difference between Python result
      and the result declared in the dataset exceeds the level, then the example
      is dropped; tolerances ranging from 0.1 to 0.001 yield from 18K to 21K
      examples.
    cumulative: if set to True, then generate examples in the format input -
      problem + numbers + op1 + op2 + op3 target - op4 If set to False, then
      examples are in the format input - problem + numbers target - all
      operations.
    python_code: if set to True, then generates python code instead of
      MathQA commands.
    full_dict: if set to True, then Python examples are returned together with
      the DSL code and the NLP rationale.
    partial_results: if set to True, then partial results will be reported as
      part of the input, e.g. input - problem + numbers + op1 + #1 + op2 + #2 +
      op3 + #3, target - op4, where #k is the partial results from operation
      opk. Activated only in cumulative set to True.
    nlp_rationale: if set to True, then input is the problem and the target is
      the nlp rationale.
    correct_answer: if set to True, then input is the problem plus all possible
      answers and the target is the correct answer.
    answer_in_mathqa_format: if set to True, then convert numerical answer to
      the MathQA format and wrap it in the subtract operation.
      E.g. "3.13" is converted to "subtract(const_3_13,const_0)".
    correct_answer_given_reasoning: if set to True, then input is the problem
      plus linear formula plus all possible answers and the target is the
      correct answer.
    category: if set to True, then input is the problem and the target is its
      category.
    order_prediction: if set to True, then input is the problem and a list of
      all operations; with probability 0.5 two operations are swapped; the task
      consists in detecting whether the operations were swapped. See the
      order prediction task in CreateAquaInputs in this file.
    reduced_operation_name: If set to True, then in order prediction consider
      only the operation token without parameterers.
    qed: if set to True, then the reasoning is finished with an additional
      operation qed.

  Returns:
    mathqa_yield_examples: a generator of MathQA examples; the generator yields
    non-tokenized examples - they can be further processed using for example
    the tokenize function from this module
  """
  if train:
    dataset_path = os.path.join(dataset_path, 'train.json')
  elif test:
    dataset_path = os.path.join(dataset_path, 'test.json')
  elif challenge:
    dataset_path = os.path.join(dataset_path, 'challenge_test.json')
  else:
    dataset_path = os.path.join(dataset_path, 'dev.json')
  # Opening with GFile allows to use remotely stored files, e.g.
  # in a gs bucket.
  dataset_handle = tf.io.gfile.GFile(dataset_path, 'r')
  dataset = json.load(dataset_handle)

  def mathqa_yield_examples(generator=None):
    del generator
    while True:
      for example in itertools.cycle(dataset):
        result = process_single_mathqa_example(example)
        # TODO(henrykm): Remove the first two ifs.
        if not result:
          continue
        answer_num, python_result, python_program, list_op, list_num = result
        if not answer_num or not python_result[-1]:
          continue
        if qed:
          list_op.append('qed')
        if math.isclose(answer_num, python_result[-1], rel_tol=tolerance):
          input_prefix = example['Problem']
          for i in range(len(list_num)):
            input_prefix += ' n{} = {}'.format(i, list_num[i])
          if cumulative:
            for i in range(len(list_op)):
              input_values = input_prefix
              target_values = list_op[i]
              input_prefix += ' ' + list_op[i]
              if partial_results:
                input_prefix += ' #{} = {}'.format(i, answer_num)
              yield input_values, target_values, np.array([1] *
                                                          len(target_values))
          elif python_code:
            input_values = '# ' + input_prefix
            target_values = ''
            for command in python_program:
              if 'math' in command:
                target_values += 'import math\n'
                break
            for command in python_program:
              if 'scipy' in command:
                target_values += 'import scipy\n'
                break
            for i in range(len(list_num)):
              target_values += 'n{} = {}\n'.format(i, list_num[i])
            target_values += '\n'.join(python_program[:-1])
            final_line = python_program[-1].split('=')[1]
            target_values += '\nanswer ={}'.format(final_line)
            var_dict = {}
            # We generate a python code and want to check whether the answer
            # is coorect.
            exec(target_values, globals(), var_dict)  # pylint: disable=exec-used
            if math.isclose(answer_num, var_dict['answer'], rel_tol=tolerance):
              if full_dict:
                yield input_values, target_values, example[
                    'linear_formula'], example['Rationale']
              else:
                yield input_values, target_values, np.array([1] *
                                                            len(target_values))
          elif nlp_rationale:
            input_values = 'infer full rationale: ' + input_prefix
            target_values = example['Rationale']
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
          elif correct_answer:
            input_values = 'infer correct answer: ' + input_prefix
            input_values += ' ' + example['options']
            if answer_in_mathqa_format:
              target_values = str(answer_num)
              target_values = convert_to_subtract(
                  convert_float_to_mathqa(target_values))
            else:
              target_values = example['correct']
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
          elif correct_answer_given_reasoning:
            input_values = 'infer correct answer given reasoning: ' + input_prefix
            input_values += ' ' + ' '.join(list_op) + ' ' + example['options']
            target_values = example['correct']
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
          elif category:
            input_values = 'infer category: ' + input_prefix
            target_values = example['category']
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
          elif order_prediction:
            if np.random.uniform() < 0.5 and len(list_op) >= 2:
              idx = range(len(list_op))
              i1, i2 = random.sample(idx, 2)
              list_op[i1], list_op[i2] = list_op[i2], list_op[i1]
              target_values = 'not_ordered'
            else:
              target_values = 'ordered'
            if reduced_operation_name:
              list_op = [op.split('(')[0] for op in list_op]
            input_values = 'order prediction: ' + input_prefix + ' ' + ' '.join(
                list_op)
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
          else:
            input_values = 'infer full calculation: ' + input_prefix
            target_values = example['linear_formula']
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))

  return mathqa_yield_examples


@gin.configurable(module='trax.data')
def CreateAquaInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    cumulative=False,
    rationale=False,
    correct_answer=False,
    correct_answer_given_reasoning=False,
    partial_reasoning=True,
    order_prediction=False):
  """Prepares Aqua inputs.

  Args:
    dataset_path: a path with the Aqua dataset.
    train: if True, then generate training examples, otherwhise generate
      validation examples (the dataset has also a test set).
    cumulative: if set to True, then generate examples in the format input -
      problem + step1 + step3 + step3 target - step4 If set to False, then
      examples are in the format input - problem, target - all operations.
    rationale: if set to True, then input is the problem and the target is the
      rationale.
    correct_answer: if set to True, then input is the problem plus all possible
      answers and the target is the correct answer.
    correct_answer_given_reasoning: if set to True, then input is the problem
      plus reasoning (aka rationale) plus all possible answers and the target is
      the correct answer.
    partial_reasoning: an additional option related to
      correct_answer_given_reasoning; if set to True, then we take a random
      prefix of the reasoning.
    order_prediction: if set to True, then input is the problem and a list of
      all operations; with probability 0.5 two operations are swapped; the task
      consists in detecting whether the operations were swapped. A similar
      additional task was considered in https://arxiv.org/pdf/1909.11942.pdf and
        in a recent work of Piotr Piękos, henrykm@ and mateuszm@.

  Returns:
    aqua_yield_examples: a generator of Aqua examples; the generator yields
    non-tokenized examples - they can be further processed using for example
    the tokenize function from this module
  """
  if train:
    dataset_path = os.path.join(dataset_path, 'train.json')
  else:
    dataset_path = os.path.join(dataset_path, 'dev.json')
  # Opening with GFile allows to use remotely stored files, e.g.
  # in a gs bucket.
  dataset_handle = tf.io.gfile.GFile(dataset_path, 'r')
  dataset = []
  for line in dataset_handle:
    dataset.append(json.loads(line))

  def aqua_yield_examples(generator=None):
    del generator
    while True:
      for example in itertools.cycle(dataset):
        input_prefix = example['question']
        steps = example['rationale'].split('\n')
        if cumulative:
          for i in range(len(steps)):
            input_values = 'infer cumulative rationale: ' + input_prefix
            target_values = steps[i]
            input_prefix += ' ' + steps[i]
            yield input_values, target_values, np.array([1] *
                                                        len(target_values))
        elif rationale:
          input_values = 'infer full rationale: ' + input_prefix
          target_values = example['rationale']
          yield input_values, target_values, np.array([1] * len(target_values))
        elif correct_answer:
          input_values = 'infer correct answer: ' + input_prefix
          input_values += ' ' + ' '.join(example['options'])
          target_values = example['correct']
          yield input_values, target_values, np.array([1] * len(target_values))
        elif correct_answer_given_reasoning:
          input_values = 'infer correct answer given reasoning: ' + input_prefix
          if partial_reasoning:
            reasoning_list = example['rationale'].split('\n')
            reasoning_list = reasoning_list[0:np.random
                                            .randint(0, len(reasoning_list))]
            reasoning = '\n'.join(reasoning_list)
          else:
            reasoning = example['rationale']
          input_values += ' ' + example['rationale'] + ' ' + ' '.join(
              example['options'])
          target_values = example['correct']
          yield input_values, target_values, np.array([1] * len(target_values))
        elif order_prediction:
          if np.random.uniform() < 0.5 and len(steps) >= 2:
            idx = range(len(steps))
            i1, i2 = random.sample(idx, 2)
            steps[i1], steps[i2] = steps[i2], steps[i1]
            target_values = 'not_ordered'
          else:
            target_values = 'ordered'
          input_values = 'order prediction: ' + input_prefix + ' ' + '\n'.join(
              steps)
          yield input_values, target_values, np.array([1] * len(target_values))
        else:
          raise ValueError(
              'One of the boolean parameters of the Aqua generator must be set to True.'
          )

  return aqua_yield_examples


@gin.configurable(module='trax.data')
def CreateDropInputs(  # pylint: disable=invalid-name
    train=True, mathqa_format=False):
  """Prepares Drop inputs.

  Args:
    train: if True, then generate training examples, otherwhise generate
      validation examples (the dataset has also a test set).
    mathqa_format: if True, then floats in targets are converted to the
      the MathQA convention and wrapped in the subtract operation.
      E.g. "3.13" is converted to "subtract(const_3_13,const_0)".

  Returns:
    drop_yield_examples: a generator of Drop examples; the generator yields
    non-tokenized examples - they can be further processed using for example
    the tokenize function from this module
  """
  if train:
    dataset = tfds.load(name='drop', split='train')
  else:
    dataset = tfds.load(name='drop', split='dev')
  dataset = tfds.as_numpy(dataset)

  def drop_yield_examples(generator=None):
    del generator
    while True:
      for example in itertools.cycle(dataset):
        input_values = 'drop question: ' + example['passage'].decode(
            'utf-8') + ' ' + example['question'].decode('utf-8')
        target_values = example['answer'].decode('utf-8')
        # Apparently the dataset has some empty "target values" -
        # when such a value is encountered, the Tokenizer decides to assign
        # to it a float32 tensor and the training fails.
        if not target_values:
          continue
        if mathqa_format:
          if target_values.replace('.', '', 1).isdigit():
            target_values = convert_to_subtract(
                convert_float_to_mathqa(target_values))
        yield input_values, target_values, np.array(
            [1] * len(target_values), dtype=np.int32)

  return drop_yield_examples


@gin.configurable(module='trax.data')
def CreateAnnotatedDropInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    single_file=True,
    unique=False,
    total_number_of_samples=None,
    percentile=1.):
  r"""Prepares annotated Drop inputs.

  Example of an annotated input which can be used with this interface:

  {
    'passage': 'The Armenian Prelature of Cyprus was established in 973 by
    Catholicos Khatchig I. Historically, the Prelature has been under the
    jurisdiction of the Catholicosate of the Great House of Cilicia, while today
    it is the oldest theme that falls under its jurisdiction. Since 2014 the
    Prelate, a Catholicosal Vicar General, has been Archbishop Nareg Alemezian.
    The parish priest in Nicosia is Fr. Momik Habeshian, while the parish priest
    in Larnaca and Limassol is Fr. Mashdots Ashkarian. For centuries, the
    Prelature building was located within the Armenian compound in Victoria
    street in walled Nicosia; when that area was taken over by Turkish-Cypriot
    extremists in 1963-1964, the Prelature was temporarily housed in Aram
    Ouzounian street and, later on, in Kyriakos Matsis street in Ayios
    Dhometios. Thanks to the efforts of Bishop Zareh Aznavorian and with
    financial aid from the Evangelical Church of Westphalia, the new Prelature
    building was erected in 1983, next to the Virgin Mary church and the Nareg
    school in Nicosia, by architects Athos Dikaios & Alkis Dikaios; it was
    officially inaugurated on 4 March 1984, during the pastoral visit of
    Catholicos Karekin II. By initiative of Archbishop Varoujan Hergelian, in
    1998 the basement of the building was renovated and the "Vahram Utidjian"
    Hall was formed; previously a store room, it became a reality from the
    proceeds of the auction in 1994 of the art collection that Vahram Utidjian
    had donated to the Prelature in 1954. It was inaugurated on 3 February 1999
    by Catholicos Aram I; numerous charity, communal and cultural events take
    place there. The Prelature\'s consistory houses a collection of
    ecclesiastical relics, some of which were previously in the old Virgin Mary
    church or the Magaravank.',
    'question': 'How many years after the Vahram Utidjian was donated to the
    Prelature was it sold at an auction?',
    'answer': 40,
    'calculation': 'subtract(n8,n9)'
  }

  In this example the calculation is formulated using the notation from the
  MathQA dataset, but this is not required. subtract(n8,n9) means that the
  answer 40 can be obtained through the substraction of the 9th and and the 10th
  number in the input. The input consists of the passage concatened with the
  question. The annotations can be generated using, for example, a method
  from the paper https://arxiv.org/abs/1909.00109.

  Args:
    dataset_path: a path with the Aqua dataset.
    train: if True, then generate training examples, otherwhise generate
      validation examples (the dataset has also a test set).
    single_file: if True, then look just for one file. If False, read all
      json files in a given directory and assume that each file contains one
      example. Applied only to training data.
    unique: if set to True, then the generator will provide at most one question
      per passage.
    total_number_of_samples: if set to a positive integer, then the total number
      of unique samples will be bounded total_number_of_samples.
    percentile: the percentile of the train dataset used for training; default
      set to 1., though setting to a lower value can be interesting when
      combined train is combined with another source of data.

  Returns:
    drop_annotated_yield_examples: a generator of annotated Drop examples;
    the generator yields non-tokenized examples - they can be further processed
    using for example the tokenize function from this module.
  """
  if train:
    if single_file:
      dataset_path = os.path.join(dataset_path, 'train_annotated.json')
  else:
    dataset_path = os.path.join(dataset_path, 'dev_annotated.json')

  def load_dataset():
    dataset = []
    if single_file:
      # Opening with GFile allows to use remotely stored files, e.g.
      # in a gs bucket.
      dataset_handle = tf.io.gfile.GFile(dataset_path, 'r')
      for line in dataset_handle:
        dataset.append(json.loads(line))
    else:
      all_files = tf.io.gfile.listdir(dataset_path)
      for filename in all_files:
        if 'json' in filename:
          print('Loading data from file {}'.format(filename))
          with tf.io.gfile.GFile(os.path.join(dataset_path, filename)) as f:
            for line in f:
              dataset.append(json.loads(line))
    print('The total size of the dataset {}'.format(len(dataset)))
    return dataset[:int(len(dataset) * percentile)]

  def drop_annotated_yield_examples(generator=None):
    del generator
    while True:
      passages = set()
      unique_examples = set()
      # Notice that below we enable a poor man RL loop
      # aka the DAgger algorithm: https://arxiv.org/pdf/1011.0686.pdf
      # tl;dr: after parsing all examples we re-load the dataset - this
      # may become handy if a prediction service generates new examples.
      dataset = load_dataset()
      for example in dataset:
        # If total_number_of_samples is not None and we have reached this
        # number of samples, then we re-load the dataset.
        if total_number_of_samples:
          if len(unique_examples) >= total_number_of_samples:
            break
        # Do we have a pre-calculated input in the example?
        if 'input' in example.keys():
          question = example['input']
          # Remove the old prompt
          question = question[question.find(':') + 2:]
        else:
          # If input is not present, then we expect that this is an
          # original drop example.
          if unique and example['passage'] in passages:
            continue
          passages.add(example['passage'])
          question = example['passage'] + ' ' + example['question']
          list_num = [
              float(num.replace(',', '').rstrip('.').lstrip('.'))  # pylint: disable=g-complex-comprehension
              for num in re.findall(
                  r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                  question)
          ]
          for i in range(len(list_num)):
            question += ' n{} = {}'.format(i, list_num[i])
        input_values = 'drop annotated question: ' + question
        target_values = example['calculation']
        unique_examples.add((input_values, target_values))
        yield input_values, target_values, np.array(
            [1] * len(target_values), dtype=np.int32)

  return drop_annotated_yield_examples
