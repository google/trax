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

"""Trax input pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random

from absl import logging

import gin
import numpy as onp

from tensor2tensor import problems_colab as t2t_problems
import tensorflow as tf   # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_datasets as tfds
from trax import math
from trax.math import numpy as np


class Inputs(object):
  """Inputs bundle.

  Inputs bundle holds input streams and shapes for a training run.
  It contains stream-creating functions that return python generators
  of (input_batch, target_batch) tuples.

  * train_stream: training data that will be used for training
      may include all the augmentation or selection the training wants
      the shape of examples is [batch_fn.batch_size, ...]
  * train_eval_stream: training data used for evaluation
      examples from training data but usually without augmentation
      the shape of examples is [batch_fn.eval_batch_size, ...]
  * eval_stream: evaluation data stream
      examples from evaluation data, usually without augmentation
      the shape of examples is [batch_fn.eval_batch_size, ...]
  * input_shape: the shape of inputs
      the [...] above, without batch size
  * input_dtype: the data type of inputs
  * target_shape: the shape of targets
      the [...] above, without batch size
  * target_dtype: the data type of targets
  """

  def __init__(self, train_stream, eval_stream=None, train_eval_stream=None):
    """Initialize a new set of inputs.

    Args:
      train_stream: a function taking n_devices (an int) and returning
        a python generator of training batches.
      eval_stream: a function taking n_devices (an int) and returning
        a python generator of validation batches;
        if None, then the training generator will be used for evaluation.
      train_eval_stream: a function taking n_devices (an int) and returning
        a python generator of batches from
        the training set used for evaluation (if None, use train_stream).
    """
    self._train_stream = train_stream
    self._eval_stream = eval_stream or self._train_stream

    # TODO(lukaszkaiser): should we get rid of this one day?
    self._train_eval_stream = train_eval_stream or self._train_stream

    # Peek into the train stream to get an example shape.
    example_train_batch = next(train_stream(1))
    self._input_shape = tuple(example_train_batch[0].shape)[1:]
    self._input_dtype = example_train_batch[0].dtype
    self._target_shape = tuple(example_train_batch[-1].shape)[1:]
    self._target_dtype = example_train_batch[-1].dtype
    self._example_shape = [x.shape for x in example_train_batch]
    self._example_dtype = [x.dtype for x in example_train_batch]

  def train_stream(self, n_devices):
    return self._train_stream(n_devices)

  def eval_stream(self, n_devices):
    return self._eval_stream(n_devices)

  def train_eval_stream(self, n_devices):
    return self._train_stream(n_devices)

  @property
  def input_shape(self):
    """Example input shape, without batch dimension."""
    return self._input_shape

  @property
  def target_shape(self):
    """Example target shape, without batch dimension."""
    return self._target_shape

  @property
  def input_dtype(self):
    """Dtype of the input."""
    return self._input_dtype

  @property
  def target_dtype(self):
    """Dtype of the target."""
    return self._target_dtype

  @property
  def example_shape_dtype(self):
    """Shape and Dtype of an example batch."""
    return self._example_shape, self._example_dtype


# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
# TODO(lukaszkaiser): can we improve efficiency, should that be changed?
_MAX_SKIP_EXAMPLES = 1e5


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
      t2t_problems.problem(
          dataset_name[len('t2t_'):]).generate_data(data_dir, dl_dir)
    else:
      # Download and prepare TFDS dataset.
      tfds_builder = tfds.builder(dataset_name)
      tfds_builder.download_and_prepare(download_dir=dl_dir)
  else:
    data_dir = os.path.expanduser(data_dir)
  return data_dir


@gin.configurable()
def inputs(dataset_name, data_dir=None, input_name=None, target_name=None):
  """Make Inputs for built-in datasets.

  Args:
    dataset_name: a TFDS or T2T dataset name. If it's a T2T dataset name, prefix
      with 't2t_'.
    data_dir: data directory.
    input_name: optional, name of the inputs from the dictionary.
    target_name: optional, name of the outputs either from the dictionary or as
      a result of post-processing.

  Returns:
    trax.inputs.Inputs
  """
  data_dir = download_and_prepare(dataset_name, data_dir)

  cache = {}
  def stream(n_devices, which):
    """Create the stream, cache TF streams if needed."""
    if n_devices not in cache:
      cache[n_devices] = _train_and_eval_batches(
          dataset_name, data_dir, input_name, target_name, n_devices)

    (train_batches, train_eval_batches, eval_batches,
     input_name_c) = cache[n_devices]
    dataset = train_batches
    if which == 'eval':
      dataset = eval_batches
    if which == 'train_eval':
      dataset = train_eval_batches
    return dataset_to_stream(dataset, input_name_c)

  return Inputs(train_stream=lambda n: stream(n, 'train'),
                train_eval_stream=lambda n: stream(n, 'train_eval'),
                eval_stream=lambda n: stream(n, 'eval'))


@gin.configurable()
def random_inputs(
    input_shape=gin.REQUIRED, input_dtype=np.int32, input_range=(0, 255),
    output_shape=gin.REQUIRED, output_dtype=np.int32, output_range=(0, 9)):
  """Make random Inputs for debugging.

  Args:
    input_shape: the shape of inputs (including batch dimension).
    input_dtype: the type of the inputs (int32 by default).
    input_range: the range of inputs (defaults to (0, 255)).
    output_shape: the shape of outputs (including batch dimension).
    output_dtype: the type of the outputs (int32 by default).
    output_range: the range of outputs (defaults to (0, 9)).

  Returns:
    trax.inputs.Inputs
  """
  def random_minibatches(n_devices):
    """Generate a stream of random mini-batches."""
    assert input_range[0] % n_devices == 0
    if input_dtype in [np.float16, np.float32, np.float64]:
      rand = onp.random.uniform
    else:
      rand = onp.random.random_integers
    while True:
      inp = rand(input_range[0], input_range[1], input_shape)
      inp = inp.astype(input_dtype)
      out = rand(output_range[0], output_range[1], output_shape)
      out = out.astype(output_dtype)
      yield inp, out

  return Inputs(random_minibatches)


def _pad_to_multiple_of(x, y, axis):
  """Pads x to multiple of y on the given axis."""
  pad_len = onp.ceil(x.shape[axis] / float(y)) * y
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, int(pad_len - x.shape[axis]))
  return onp.pad(x, pad_widths, mode='constant',
                 constant_values=x.dtype.type(0))


@gin.configurable()
def sequence_copy_inputs(
    vocab_size=gin.REQUIRED, batch_size=gin.REQUIRED, train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED, eval_max_length=gin.REQUIRED, reverse=False,
    pad_to_multiple=32):
  """Inputs for the sequence copy problem: 0w0w for w in [1..vocab_size-1]*.

  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    train_length: maximum length of w for training.
    eval_min_length: minimum length of w for eval.
    eval_max_length : maximum length of w for eval.
    reverse: bool (optional, false by default): reverse the second sequence.
    pad_to_multiple: int, pad length to be multiple of this number.

  Returns:
    trax.inputs.Inputs
  """
  def random_minibatches(length_list):
    """Generate a stream of random mini-batches."""
    while True:
      length = random.choice(length_list)
      assert length % 2 == 0
      w_length = (length // 2) - 1
      w = onp.random.randint(low=1, high=vocab_size-1,
                             size=(batch_size, w_length))
      zero = onp.zeros([batch_size, 1], onp.int32)
      loss_weights = onp.concatenate([onp.zeros((batch_size, w_length+2)),
                                      onp.ones((batch_size, w_length))], axis=1)
      if reverse:
        x = onp.concatenate([zero, w, zero, np.flip(w, axis=1)], axis=1)
      else:
        x = onp.concatenate([zero, w, zero, w], axis=1)
      x = _pad_to_multiple_of(x, pad_to_multiple, 1)
      loss_weights = _pad_to_multiple_of(loss_weights, pad_to_multiple, 1)
      yield (x, x, loss_weights)  # Here inputs and targets are the same.

  train_lengths = [2*(i+2) for i in range(train_length - 1)]
  eval_lengths = [2*(i+1) for i in range(eval_min_length, eval_max_length)]
  return Inputs(
      train_stream=lambda _: random_minibatches(train_lengths),
      eval_stream=lambda _: random_minibatches(eval_lengths)
  )


def lower_endian_to_number(l, base):
  """Helper function: convert a list of digits in the given base to a number."""
  return sum([d * (base**i) for i, d in enumerate(l)])


def number_to_lower_endian(n, base):
  """Helper function: convert a number to a list of digits in the given base."""
  if n < base:
    return [n]
  return [n % base] + number_to_lower_endian(n // base, base)


def random_number_lower_endian(length, base):
  """Helper function: generate a random number as a lower-endian digits list."""
  if length == 1:  # Last digit can be 0 only if length is 1.
    return [onp.random.randint(base)]
  prefix = [onp.random.randint(base) for _ in range(length - 1)]
  return prefix + [onp.random.randint(base - 1) + 1]  # Last digit is not 0.


@gin.configurable()
def addition_inputs(
    vocab_size=gin.REQUIRED, batch_size=gin.REQUIRED, train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED, eval_max_length=gin.REQUIRED,
    pad_to_multiple=32):
  """Inputs for the add problem: <S>x+y<S>(x+y).

  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    train_length: maximal length of w for training.
    eval_min_length: minimal length of w for eval.
    eval_max_length: maximal length of w for eval.
    pad_to_multiple: int, pad length to be multiple of this number.

  Returns:
    trax.inputs.Inputs
  """
  base = vocab_size - 3  # We use 0 to pad, base+1 as "+" and base+2 as "<S>".
  def single_example(max_length, min_length):
    """Generate a stream of random mini-batches."""
    add_len = (min_length - 1) // 2
    l1 = onp.random.randint((max_length - add_len + 1) // 2) + add_len
    l2 = onp.random.randint(max_length - l1 - 1) + 1
    n1 = random_number_lower_endian(l1, base)
    n2 = random_number_lower_endian(l2, base)
    result = lower_endian_to_number(n1, base) + lower_endian_to_number(
        n2, base)
    inp = n1 + [base] + n2
    tgt = number_to_lower_endian(result, base)
    x = [base+2] + [i+1 for i in inp] + [base+2] + [i+1 for i in tgt]
    weights = ([0] * (len(inp) + 2)) + ([1] * len(tgt))
    return (x, weights)

  def batches(max_length, min_length):
    """Batches of examples."""
    if max_length < 3:
      raise ValueError('Maximum length must be at least 3.')
    while True:
      res = [single_example(max_length, min_length) for _ in range(batch_size)]
      l = max([len(x[0]) for x in res])
      xs = onp.array([x[0] + [0] * (l - len(x[0])) for x in res])
      ws = onp.array([x[1] + [0] * (l - len(x[1])) for x in res],
                     dtype=onp.float32)
      xs = _pad_to_multiple_of(xs, pad_to_multiple, 1)
      ws = _pad_to_multiple_of(ws, pad_to_multiple, 1)
      yield (xs, xs, ws)

  return Inputs(
      train_stream=lambda _: batches(train_length, 3),
      eval_stream=lambda _: batches(eval_max_length, eval_min_length)
  )


def dataset_to_stream(dataset, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  for example in math.dataset_as_numpy(dataset):
    features = example[0]
    inp, out = features[input_name], example[1]
    mask = features['mask'] if 'mask' in features else None
    # All input-pipeline processing should be on CPU.
    with tf.device('cpu:0'):
      # Some accelerators don't handle uint8 well, cast to int.
      if isinstance(inp, np.uint8):
        inp = inp.astype(np.int32)
      if isinstance(out, np.uint8):
        out = out.astype(np.int32)
      if len(out.shape) > 1 and out.shape[-1] == 1:
        out = np.squeeze(out, axis=-1)
    yield (inp, out) if mask is None else (inp, out, mask)


@gin.configurable(whitelist=['train_shuffle_files', 'eval_shuffle_files',
                             'eval_holdout_size'])
def train_and_eval_dataset(dataset_name, data_dir, eval_holdout_size=0,
                           train_shuffle_files=True, eval_shuffle_files=False):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the dataset; if it starts with 't2t_'
      then we'll search T2T Problem registry for it, otherwise we assume it
      is a dataset from TFDS and load it from there.
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
    train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:train_percentage])
    eval_split = tfds.Split.TRAIN.subsplit(tfds.percent[train_percentage:])
  else:
    if tfds.Split.VALIDATION not in splits and 'test' not in splits:
      raise ValueError('We require a validation or test split in the dataset.')
    eval_split = tfds.Split.VALIDATION
    if tfds.Split.VALIDATION not in splits:
      eval_split = tfds.Split.TEST
  train = tfds.load(
      name=dataset_name, split=train_split, data_dir=data_dir,
      shuffle_files=train_shuffle_files)
  valid = tfds.load(
      name=dataset_name, split=eval_split, data_dir=data_dir,
      shuffle_files=eval_shuffle_files)
  keys = None
  if info.supervised_keys:
    keys = ([info.supervised_keys[0]], [info.supervised_keys[1]])
  return train, valid, info.features, keys


def _make_info(shape_list, n_classes, dtype):
  """Create an info-like tuple for feature given some shapes and vocab size."""
  feature_info = collections.namedtuple(
      'FeatureInfo', ['shape', 'n_classes', 'dtype'])
  cur_shape = list(shape_list[0])
  # We need to merge the provided shapes, put None where they disagree.
  for shape in shape_list:
    if len(shape) != len(cur_shape):
      raise ValueError('Shapes need to have the same number of dimensions.')
    for i in range(len(shape)):
      if cur_shape[i] is not None:
        if shape[i] != cur_shape[i]:
          cur_shape[i] = None
  return feature_info(cur_shape, n_classes, dtype)


def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ['inputs', 'targets']
  return {f: example[f] for f in feature_list if f in example}


def _eager_dataset_iterator(dataset):
  for item in dataset:
    flat = tf.nest.flatten(item)
    flat = [el.numpy() for el in flat]
    yield tf.nest.pack_sequence_as(item, flat)


def _train_and_eval_dataset_v1(problem_name, data_dir,
                               train_shuffle_files, eval_shuffle_files):
  """Return train and evaluation datasets, feature info and supervised keys."""
  with tf.device('cpu:0'):
    problem = t2t_problems.problem(problem_name)
    hparams = None
    if problem_name == 'video_bair_robot_pushing':
      hparams = problem.get_hparams()
      bair_robot_pushing_hparams(hparams)
    train_dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, data_dir,
                                    shuffle_files=train_shuffle_files,
                                    hparams=hparams)
    train_dataset = train_dataset.map(_select_features)
    eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir,
                                   shuffle_files=eval_shuffle_files,
                                   hparams=hparams)
    eval_dataset = eval_dataset.map(_select_features)
    hparams = problem.get_hparams()
    # We take a few training examples to guess the shapes.
    input_shapes, target_shapes, examples = [], [], []
    if tf.executing_eagerly():
      for example in _eager_dataset_iterator(train_dataset.take(3)):
        examples.append(example)
    else:
      example_tensor = train_dataset.make_one_shot_iterator().get_next()
      sess = tf.Session()
      example1 = sess.run(example_tensor)
      example2 = sess.run(example_tensor)
      example3 = sess.run(example_tensor)
      examples = [example1, example2, example3]
  # We use 'inputs' as input except for purely auto-regressive tasks like
  # language models where 'targets' are used as input_key.
  input_key = 'inputs' if 'inputs' in examples[0] else 'targets'
  supervised_keys = ([input_key], ['targets'])
  for example in examples:
    input_shapes.append(list(example[input_key].shape))
    target_shapes.append(list(example['targets'].shape))
  input_vocab_size = hparams.vocab_size[input_key]
  target_vocab_size = hparams.vocab_size['targets']
  input_dtype = examples[0][input_key].dtype
  target_dtype = examples[0]['targets'].dtype
  input_info = _make_info(input_shapes, input_vocab_size, input_dtype)
  target_info = _make_info(target_shapes, target_vocab_size, target_dtype)
  info = {input_key: input_info, 'targets': target_info}
  return train_dataset, eval_dataset, info, supervised_keys


@gin.configurable(blacklist=['dataset', 'training', 'shapes', 'n_devices'])
def batch_fn(dataset, training, shapes, n_devices,
             batch_size_per_device=32, batch_size=None, eval_batch_size=32,
             bucket_length=32, buckets=None,
             buckets_include_inputs_in_length=False,
             batch_shuffle_size=128, max_eval_length=None):
  """Batching function."""
  # Batch size is batch_size_per_device * n_devices unless given directly.
  batch_size = batch_size or batch_size_per_device * n_devices
  # If bucketing is not specified, check if target shapes are variable.
  cur_batch_size = batch_size if training else eval_batch_size
  # Make cur_batch_size divisible by n_devices.
  cur_batch_size = max(cur_batch_size // n_devices, 1) * n_devices
  # Create heuristic buckets is none are specified.
  if buckets is None:
    variable_target_shapes = False
    target_shape = shapes[1]
    for dim in target_shape:
      if dim is None:
        variable_target_shapes = True
    logging.info('Heuristically setting bucketing to %s based on shapes '
                 'of target tensors.', variable_target_shapes)
    if variable_target_shapes:
      bucket_boundaries = [bucket_length // 4, bucket_length // 2,
                           bucket_length, bucket_length * 2,
                           bucket_length * 4, bucket_length * 8,
                           bucket_length * 16]
      if not training:
        max_eval_length = max_eval_length or bucket_length * 32
        bucket_boundaries[-1] = max_eval_length
      # We will pad to boundaries which pads to bucket_boundary - 1: add 1 here.
      bucket_boundaries = [b + 1 for b in bucket_boundaries]
      bucket_batch_sizes = [cur_batch_size * 4, cur_batch_size * 2,
                            cur_batch_size, cur_batch_size // 2,
                            cur_batch_size // 4, cur_batch_size // 8,
                            cur_batch_size // 16, 1]
      if not training:
        bucket_batch_sizes[-2] = cur_batch_size // max_eval_length
      # Make batch sizes divisible by n_devices.
      bucket_batch_sizes = [max(b // n_devices, 1) * n_devices
                            for b in bucket_batch_sizes]
      buckets = (bucket_boundaries, bucket_batch_sizes)

  if buckets:
    logging.info('Bucketing with buckets %s.', str(buckets))
    def example_length(example_inputs, target):
      """The length function used by bucket_by_sequence_length to bucket."""
      other_length = 0
      if buckets_include_inputs_in_length:
        other_length = tf.shape(example_inputs['inputs'])[0]
      return tf.maximum(tf.shape(target)[0], other_length)
    boundaries, batch_sizes = buckets
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        example_length, boundaries, batch_sizes,
        pad_to_bucket_boundary=True))
  else:
    dataset = dataset.padded_batch(cur_batch_size, shapes)
  if training:
    return dataset.shuffle(batch_shuffle_size)
  return dataset


# Makes the function accessible in gin configs, even with all args blacklisted.
@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def cifar10_no_augmentation_preprocess(dataset, training, shapes):
  del training

  def cast_image(features, targets):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features, targets

  dataset = dataset.map(cast_image)
  return dataset, shapes


def _cifar_augment_image(image):
  """Image augmentation suitable for CIFAR-10/100.

  As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).

  Args:
    image: a Tensor.
  Returns:
    Tensor of the same shape as image.
  """
  image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
  image = tf.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  return image


# Makes the function accessible in gin configs, even with all args blacklisted.
@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def cifar10_augmentation_preprocess(dataset, training, shapes):
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
  return dataset, shapes


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def cifar10_augmentation_flatten_preprocess(dataset, training, shapes,
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

  inp_shape, _ = shapes
  img_shape = inp_shape['image']
  flat_shape = img_shape[0] * img_shape[1] * img_shape[2] + 1
  new_shapes = {'image': (flat_shape,), 'mask': (flat_shape,)}
  return dataset, (new_shapes, (flat_shape,))


def no_preprocess(dataset, training, shapes):
  del training
  return dataset, shapes


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def concat_preprocess(dataset, training, shapes, pad_symbol=0):
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
  return dataset, shapes


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def lm1b_preprocess(dataset, training, shapes,
                    max_target_length=-1, max_eval_target_length=-1):
  """Preprocessing for LM1B: filter out targets exceeding maximum length."""

  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)

  def eval_target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_eval_target_length + 1)

  if max_target_length > 0 and training:
    dataset = dataset.filter(target_right_length)

  if max_eval_target_length > 0 and not training:
    dataset = dataset.filter(eval_target_right_length)

  return dataset, shapes


# TODO(lukaszkaiser): find a single more abstract way of text pre-processing.
@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def wmt_preprocess(dataset, training, shapes,
                   max_length=-1, max_eval_length=-1):
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

  return dataset, shapes


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def wmt_concat_preprocess(dataset, training, shapes,
                          max_length=-1, max_eval_length=-1):
  """Preprocessing for WMT: filter exceeding maximum length and concatenate."""
  dataset, shapes = wmt_preprocess(
      dataset, training, shapes, max_length, max_eval_length)

  def concat_and_add_mask(features, targets):
    inp = features['inputs']
    pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
    concat = tf.concat([inp, pad, targets], axis=0)
    mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
    features['inputs'] = concat
    features['mask'] = mask
    return features, concat

  dataset = dataset.map(concat_and_add_mask)
  return dataset, shapes


@gin.configurable(blacklist=['hparams'])
def bair_robot_pushing_hparams(
    hparams=None, video_num_input_frames=1, video_num_target_frames=15
    ):
  if hparams is not None:
    hparams.video_num_input_frames = video_num_input_frames
    hparams.video_num_target_frames = video_num_target_frames
  else:
    return video_num_input_frames, video_num_target_frames


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def bair_robot_pushing_preprocess(dataset, training, shapes):
  """Pre-processing function that concatenates input and target frames."""
  del training, shapes

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

  video_num_input_frames, video_num_target_frames = bair_robot_pushing_hparams()
  video_num_frames = video_num_input_frames + video_num_target_frames
  new_shape = (video_num_frames * 64 * 64 * 3,)
  new_shapes = {
      'inputs': new_shape,
      'targets': new_shape,
      'mask': new_shape,
  }

  return dataset, (new_shapes, new_shape)


@gin.configurable(whitelist=['preprocess_fun', 'shuffle_buffer_size'])
def shuffle_and_batch_data(dataset,
                           target_names,
                           features_info,
                           training,
                           n_devices,
                           shuffle_buffer_size=1024,
                           preprocess_fun=no_preprocess):
  """Shuffle and batch the given dataset."""
  def append_targets(example):
    """Append targets to the example dictionary. Needed for Keras."""
    if len(target_names) == 1:
      return (example, example[target_names[0]])
    targets = {}
    for name in target_names:
      targets[name] = example[name]
    return (example, targets)
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
  shapes = {k: features_info[k].shape for k in features_info}
  shapes = (shapes, shapes[target_names[0]])
  dataset, shapes = preprocess_fun(dataset, training, shapes)
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = batch_fn(dataset, training, shapes, n_devices)
  return dataset.prefetch(2)


def _train_and_eval_batches(
    dataset, data_dir, input_name, target_name, n_devices):
  """Return train and eval batches with input name and shape."""
  (train_data, eval_data, features_info, keys) = train_and_eval_dataset(
      dataset, data_dir)
  if keys is not None:
    input_names, target_names = keys[0], keys[1]
  else:
    input_names, target_names = [input_name], [target_name]
  train_batches = shuffle_and_batch_data(
      train_data, target_names, features_info, training=True,
      n_devices=n_devices)
  train_eval_batches = shuffle_and_batch_data(  # Data for eval-on-train
      train_data, target_names, features_info, training=False,
      n_devices=n_devices)
  eval_batches = shuffle_and_batch_data(
      eval_data, target_names, features_info, training=False,
      n_devices=n_devices)
  input_name = input_name or input_names[0]
  return (train_batches, train_eval_batches, eval_batches, input_name)


@gin.configurable(blacklist=['dataset', 'training', 'shapes'])
def c4_preprocess(dataset, training, shapes, max_target_length=-1):
  """Pre-processing function for C4 dataset."""
  del training
  def unicode_decode_chars(features, targets):
    targets = tf.strings.unicode_decode(features['text'], 'UTF-8')
    targets = tf.cast(targets, tf.int64)
    features['targets'] = targets
    features['inputs'] = targets
    return (features, targets)

  dataset = dataset.map(unicode_decode_chars)

  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)

  if max_target_length > 0:
    dataset = dataset.filter(target_right_length)

  new_shapes = shapes[0]
  new_shapes['inputs'] = (None,)
  new_shapes['targets'] = (None,)
  return dataset, (new_shapes, new_shapes['targets'])
