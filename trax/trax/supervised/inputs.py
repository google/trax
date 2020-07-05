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
"""Trax input pipeline."""

import math
import random

from absl import logging

import gin
import numpy as np

from trax.fastmath import numpy as jnp


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
    if not callable(train_stream):
      raise ValueError('Trax Inputs should be initialized with a function. '
                       'Did you forget the n_devices argument? If your inputs '
                       'do not use it, try lambda _: [your-inputs].')

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


# Batching and input pipeline creation helpers.


@gin.configurable()
def batcher(data_streams=gin.REQUIRED, variable_shapes=True,
            batch_size_per_device=32, batch_size=None, eval_batch_size=32,
            bucket_length=32, buckets=None,
            buckets_include_inputs_in_length=False,
            batch_shuffle_size=None, max_eval_length=None,
            # TODO(afrozm): Unify padding logic.
            id_to_mask=None, strict_pad_on_len=False):
  """Batcher: create trax Inputs from single-example data-streams."""
  # TODO(lukaszkaiser, jonni): revisit arguments, their semantics and naming.
  # For now leaving the arguments as in batch_fn to reduce gin config changes.
  if callable(data_streams):  # If we pass a function, e.g., through gin, call.
    train_stream, eval_stream = data_streams()
  else:
    train_stream, eval_stream = data_streams
  # pylint: disable=g-long-lambda
  batch_train_stream = lambda n_devices: batch_fn(
      train_stream(), True, n_devices, variable_shapes,
      batch_size_per_device, batch_size, eval_batch_size,
      bucket_length, buckets, buckets_include_inputs_in_length,
      batch_shuffle_size, max_eval_length, id_to_mask, strict_pad_on_len)
  batch_eval_stream = lambda n_devices: batch_fn(
      eval_stream(), False, n_devices, variable_shapes,
      batch_size_per_device, batch_size, eval_batch_size,
      bucket_length, buckets, buckets_include_inputs_in_length,
      batch_shuffle_size, max_eval_length, id_to_mask, strict_pad_on_len)
  batch_train_eval_stream = lambda n_devices: batch_fn(
      train_stream(), False, n_devices, variable_shapes,
      batch_size_per_device, batch_size, eval_batch_size,
      bucket_length, buckets, buckets_include_inputs_in_length,
      batch_shuffle_size, max_eval_length, id_to_mask, strict_pad_on_len)
  # pylint: enable=g-long-lambda
  return Inputs(train_stream=batch_train_stream,
                eval_stream=batch_eval_stream,
                train_eval_stream=batch_train_eval_stream)


def batch_fn(dataset, training, n_devices, variable_shapes,
             batch_size_per_device=32, batch_size=None, eval_batch_size=32,
             bucket_length=32, buckets=None,
             buckets_include_inputs_in_length=False,
             batch_shuffle_size=None, max_eval_length=None,
             id_to_mask=None, strict_pad_on_len=False):
  """Batching function."""
  # TODO(lukaszkaiser, jonni): revisit arguments, their semantics and naming.
  # After that, create a proper doc-string; we may also not need to pass both
  # training and eval arguments here, as batcher calls the function separately
  # now and it's not under gin-config any more -- consider reducing args.
  batch_size = batch_size or batch_size_per_device * n_devices
  # If bucketing is not specified, check if target shapes are variable.
  cur_batch_size = batch_size if training else eval_batch_size
  # Make cur_batch_size divisible by n_devices.
  cur_batch_size = max(cur_batch_size // n_devices, 1) * n_devices
  # Create heuristic buckets if none are specified.
  if buckets is None:
    logging.info('Heuristically setting bucketing to %s based on shapes '
                 'of target tensors.', variable_shapes)
    if variable_shapes:
      bucket_boundaries = [bucket_length // 4, bucket_length // 2,
                           bucket_length, bucket_length * 2,
                           bucket_length * 4, bucket_length * 8,
                           bucket_length * 16]
      if not training:
        max_eval_length = max_eval_length or bucket_length * 32
        # Set last bucket boundary to be max_eval_length, cut off boundaries
        # that are larger than this.
        bucket_boundaries = (
            [b for b in bucket_boundaries if b < max_eval_length] +
            [max_eval_length]
        )
        bucket_boundaries.append(max_eval_length)
      # We will pad to boundaries which pads to bucket_boundary - 1: add 1 here.
      bucket_boundaries = [b + 1 for b in bucket_boundaries]
      bucket_batch_sizes = [cur_batch_size * 4, cur_batch_size * 2,
                            cur_batch_size, cur_batch_size // 2,
                            cur_batch_size // 4, cur_batch_size // 8,
                            cur_batch_size // 16, 1]
      if not training:
        # The last bucket batch size is always 1, but the one-but-last is
        # sized to accomodate the final length = bucket_boundaries[-1], which
        # we changed for eval above -- so adjusting here too.

        # Resize if needed, since bucket_batch_sizes may not be the same size
        # anymore.
        bucket_batch_sizes = bucket_batch_sizes[:len(bucket_boundaries)] + [1]
        bucket_batch_sizes[-2] = cur_batch_size // max_eval_length
      # Make batch sizes divisible by n_devices.
      bucket_batch_sizes = [max(b // n_devices, 1) * n_devices
                            for b in bucket_batch_sizes]
      buckets = (bucket_boundaries, bucket_batch_sizes)

  if buckets:
    logging.info('Bucketing with buckets %s.', str(buckets))
    def example_length(x):
      """The length function used by bucket_by_sequence_length to bucket."""
      # The input x is a tuple to go on the stack, typically either
      # (input, target) or (input, target, mask).
      example_inputs, target = x[0], x[1]
      # Length is the shape of axis 0 here (no batch yet).
      other_length = 0  # We include input length only if asked.
      if buckets_include_inputs_in_length:
        other_length = example_inputs.shape[0]
      return max(target.shape[0], other_length)
    boundaries, batch_sizes = buckets
    dataset = bucket_by_length(
        dataset, example_length, boundaries, batch_sizes, strict_pad_on_len)
  else:
    logging.info('Not Bucketing cur_batch_size %d.', cur_batch_size)
    dataset = batch_data(dataset, cur_batch_size)
  if training and batch_shuffle_size is not None:
    dataset = shuffle_data(dataset, batch_shuffle_size)
  return add_loss_weights(dataset, id_to_mask)


def shuffle_data(samples, queue_size):
  """Shuffles a sample stream using a random-out next-in queue of given size.

  Args:
    samples: Stream of samples for eventual use as training data or eval data.
    queue_size: Minimum number of samples within which the streamed shuffling
        takes place.

  Yields:
    Shuffled stream of samples, ready for further processing, e.g., grouping
    into batches.
  """
  if queue_size < 1:
    raise ValueError(f'Arg queue_size ({queue_size}) is less than 1.')
  if queue_size == 1:
    logging.warning('Queue size of 1 results in no shuffling.')
  queue = []
  try:
    # Prep: fill the queue.
    for _ in range(queue_size):
      queue.append(next(samples))

    # Core streaming shuffle: yield sample from random location in queue, then
    # fill that location with new sample from input stream.
    for sample in samples:
      i = np.random.randint(queue_size)
      yield queue[i]
      queue[i] = sample
  except StopIteration:
    # Only get here if the initial queue fill fails.
    logging.warning(
        'Not enough samples (%d) to fill initial queue (size %d).',
        len(queue), queue_size)

  # No new samples coming in; shuffle and drain the queue.
  np.random.shuffle(queue)
  for sample in queue:
    yield sample


def batch_data(generator, batch_size):
  """Batch and pad generator as in tf.data.Dataset.padded_batch."""
  buf = []
  # TODO(lukaszkaiser): convert to ValueError
  assert batch_size > 0, f'Batch size must be positive, but is {batch_size}'
  for example in generator:
    buf.append(example)  # Examples are tuples of tensors.
    if len(buf) == batch_size:
      # buf is a list of tuples, e.g., [(in1, tgt1), (in2, tgt2), (in3, tgt3)]
      # batch is a tuple of arrays: ([in1, in2, in3], [tgt1, tgt2, tgt3])
      batch = tuple(np.stack(x) for x in zip(*buf))
      # Note that it's the same shape as each example with added batch dim.
      yield batch
      buf = []


def pad_to_max_dims(tensors, boundary=None, strict_pad_on_len=False):
  """Pad a tuple of tensors to a joint dimension and return their batch.

  For example, a pair of tensors of shape (2, 10) and (3, 9) will be padded
  to (3, 10) both and the returned tensor will have shape (2, 3, 10).

  When boundary is specified, we try to pad all unknown dimensions to boundary
  if possible, which can help reduce the number of different shapes occuring
  in the tensors and speed up XLA compilation. So, for example, a pair of
  tensors of shapes (8, 10), (8, 9) with boundary=12 will be padded to (8, 12).

  One special case occurs when boundary is much higher than the padding length
  that we'd use without boundary. For example, tensors (2, 10) and (3, 9) with
  boundary=12 could end up padded to (12, 12), but this is very wasteful in
  the first dimension. In that case, we will use the closest power-of-2 instead
  of the boundary, so the we will end up padding to (4, 12) instead of (12, 12).

  Args:
    tensors: a tuple or list of tensors to pad
    boundary: int or None; if given, expand the padded dimensions to this size
    strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
      strictly as a multiple of boundary.

  Returns:
    a tensor, the tensors padded together
  """
  # TODO(afrozm): Unify this later.
  if ((boundary is not None) and
      (strict_pad_on_len or isinstance(boundary, (list, tuple)))):
    ndim = tensors[0].ndim
    if not isinstance(boundary, (list, tuple)):
      boundary = [boundary] * ndim

    if ndim != len(boundary):
      raise ValueError(f'ndim != len(boundary) - '
                       f'ndim({ndim}) vs boundary({boundary}) '
                       f'len(boundary) = {len(boundary)}.')

    max_len_per_dim = [0] * ndim
    for tensor in tensors:
      max_len_per_dim = [
          max(e, s) for e, s in zip(tensor.shape, max_len_per_dim)]

    # Round everything up to a multiple of boundary in the respective dimension.
    len_per_dim = [
        max_len_per_dim[i] if not b else b * math.ceil(max_len_per_dim[i] / b)
        for i, b in enumerate(boundary)]

    padded_tensors = [
        np.pad(t, [(0, len_per_dim[i] - t.shape[i]) for i in range(ndim)],
               mode='constant', constant_values=t.dtype.type(0))
        for t in tensors]

    return np.stack(padded_tensors)

  max_len_to_pad = []
  padding_needed = False
  dim = len(tensors[0].shape)
  for i in range(dim):
    max_len = max([t.shape[i] for t in tensors])
    min_len = min([t.shape[i] for t in tensors])
    if max_len == min_len:  # No padding needed.
      max_len_to_pad.append(max_len)
    elif boundary is None:
      max_len_to_pad.append(max_len)
      padding_needed = True
    else:
      padding_needed = True
      cur_boundary = max(max_len, boundary)
      if 2 * max_len < cur_boundary:
        cur_boundary = 2**int(np.ceil(np.log2(max_len)))
      max_len_to_pad.append(cur_boundary)
  if not padding_needed:
    return np.stack(tensors)
  padded_tensors = []
  for t in tensors:
    pad_widths = [(0, max_len_to_pad[i] - t.shape[i]) for i in range(dim)]
    padded_t = np.pad(t, pad_widths, mode='constant',
                      constant_values=t.dtype.type(0))
    padded_tensors.append(padded_t)
  return np.stack(padded_tensors)


def bucket_by_length(generator, length_fn, boundaries, batch_sizes,
                     strict_pad_on_len=False):
  """Bucket by length, like tf.data.experimental.bucket_by_sequence_length.

  Args:
    generator: python generator to draw data from.
    length_fn: a function taking the example and returning the length.
    boundaries: a list of bucket boundaries.
    batch_sizes: a list of batch sizes.
    strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
      strictly as a multiple of boundary.

  Yields:
    An input batch, which comes from one of the buckets.
  """
  buckets = [[] for _ in range(len(batch_sizes))]
  boundaries = boundaries + [math.inf]  # Max boundary is unlimited.
  for example in generator:
    length = length_fn(example)
    # `bucket_idx` will always be < len(boundaries), since boundaries is right
    # padded by `math.inf`.
    bucket_idx = min([i for i, b in enumerate(boundaries) if length < b])
    buckets[bucket_idx].append(example)
    if len(buckets[bucket_idx]) == batch_sizes[bucket_idx]:
      batch = zip(*buckets[bucket_idx])
      boundary = boundaries[bucket_idx] - 1
      boundary = None if boundary == math.inf else boundary
      padded_batch = tuple(
          pad_to_max_dims(x, boundary, strict_pad_on_len) for x in batch)
      yield padded_batch
      buckets[bucket_idx] = []


def add_loss_weights(generator, id_to_mask=None):
  """Add weights to inputs without weights and masks by id if requested.

  The generator stream is augmented in the following way:

  - If the stream consists of pairs `(inputs, targets)`, a loss mask is added
    that is creates as a tensor of ones of the same shape as targets.
  - If `id_to_mask` is not `None`, and the stream (after the previous point)
    has triples `(inputs, targets, weights)`, the weights are multipled by a
    0/1 mask that is 0 iff targets is equal to `id_to_mask` (1 otherwise).

  Args:
    generator: Stream of tuples.
    id_to_mask: If not None, int-valued id that represents padding, as opposed
        to true target id's.

  Yields:
    Examples from the augmented stream.
  """
  for example in generator:
    if len(example) > 3 or len(example) < 2:
      assert id_to_mask is None, 'Cannot automatically mask this stream.'
      yield example
    else:
      if len(example) == 2:
        weights = np.ones_like(example[1]).astype(np.float32)
      else:
        weights = example[2].astype(np.float32)
      mask = 1.0 - np.equal(example[1], id_to_mask).astype(np.float32)
      weights *= mask
      yield (example[0], example[1], weights)


# Example input functions.


@gin.configurable()
def random_inputs(
    input_shape=gin.REQUIRED, input_dtype=jnp.int32, input_range=(0, 255),
    output_shape=gin.REQUIRED, output_dtype=jnp.int32, output_range=(0, 9)):
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
    if input_dtype in [jnp.float16, jnp.float32, jnp.float64]:
      rand = np.random.uniform
    else:
      rand = np.random.random_integers
    while True:
      inp = rand(input_range[0], input_range[1], input_shape)
      inp = inp.astype(input_dtype)
      out = rand(output_range[0], output_range[1], output_shape)
      out = out.astype(output_dtype)
      yield inp, out

  return Inputs(random_minibatches)


def _pad_to_multiple_of(x, y, axis):
  """Pads x to multiple of y on the given axis."""
  pad_len = np.ceil(x.shape[axis] / float(y)) * y
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, int(pad_len - x.shape[axis]))
  return np.pad(x, pad_widths, mode='constant',
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
      w = np.random.randint(low=1, high=vocab_size-1,
                            size=(batch_size, w_length))
      zero = np.zeros([batch_size, 1], np.int32)
      loss_weights = np.concatenate([np.zeros((batch_size, w_length+2)),
                                     np.ones((batch_size, w_length))], axis=1)
      if reverse:
        x = np.concatenate([zero, w, zero, jnp.flip(w, axis=1)], axis=1)
      else:
        x = np.concatenate([zero, w, zero, w], axis=1)
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
    return [np.random.randint(base)]
  prefix = [np.random.randint(base) for _ in range(length - 1)]
  return prefix + [np.random.randint(base - 1) + 1]  # Last digit is not 0.


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
    l1 = np.random.randint((max_length - add_len + 1) // 2) + add_len
    l2 = np.random.randint(max_length - l1 - 1) + 1
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
      xs = np.array([x[0] + [0] * (l - len(x[0])) for x in res])
      ws = np.array([x[1] + [0] * (l - len(x[1])) for x in res],
                    dtype=np.float32)
      xs = _pad_to_multiple_of(xs, pad_to_multiple, 1)
      ws = _pad_to_multiple_of(ws, pad_to_multiple, 1)
      yield (xs, xs, ws)

  return Inputs(
      train_stream=lambda _: batches(train_length, 3),
      eval_stream=lambda _: batches(eval_max_length, eval_min_length)
  )
