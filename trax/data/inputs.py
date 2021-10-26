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
"""Data sources and input processing.

Trax authors recommend constructing input pipelines using layer-like functions
and combinators. For example, following is an input pipeline for training
sentiment analysis tasks on the IMDB dataset::

  from trax import data

  inputs = data.Serial(
    data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),
    data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    data.Shuffle(),
    data.FilterByLength(max_length=2048, length_keys=[0]),
    data.BucketByLength(boundaries=[  32, 128, 512, 2048],
                        batch_sizes=[128,  32,   8,    2, 1],
                        length_keys=[0]),
    data.AddLossWeights()
  )

Each of these functions creates a Python generator of tuples of data arrays.
For example::

  data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

creates a generator of examples (tuples of NumPy :py:class:`ndarray` objects)
from the TFDS imdb_reviews dataset, see here:
https://www.tensorflow.org/datasets/catalog/imdb_reviews

As you can see on the website above, this dataset has 'text' and 'label' fields
and we create tuples containing the text and the label from the training split
by specifying keys=('text', 'label'), train=True.

Other functions, like ``Tokenize`` and ``Shuffle``, take a generator and output
another generator, in this way converting tuples into other tuples or mixing
the training stream. For example, ``Tokenize(..., keys=[0])`` tokenizes the
first element of a tuple -- converting it from text to a NumPy integer array.
And ``Shuffle`` randomizes the order of examples.

Note that all elements in the data pipeline are just functions on generators,
so you can use Python's `map` and `filter` and other native functions too.
For example, you can create an input pipeline for a language model reading
lines from `my_file.txt` as follows::

  inputs = data.Serial(
    lambda _: open('my_file.txt'),
    lambda g: map(lambda line: line.strip(), g),
    data.Tokenize(vocab_file='en_8k.subword'),
    lambda g: filter(lambda x: x.shape[0] < 513, g),  # At most 512 tokens.
    data.Shuffle(),
    lambda g: map(lambda x: (x, x)),  # Language models have inputs = targets.
    data.BucketByLength(boundaries=[  32, 64, 128, 256, 512],
                        batch_sizes=[ 32, 16,  8,    4,   2, 1]),
    data.AddLossWeights(id_to_mask=0)
  )

"""

import math
import multiprocessing.dummy as mp  # using threads for now
import os
import pickle
import random
import time

from absl import logging
import gin
import jax
import numpy as np
import tensorflow as tf

from trax import fastmath
from trax import shapes
from trax.data import debug_data_pipeline
from trax.fastmath import numpy as jnp


def Serial(*fns):  # pylint: disable=invalid-name
  """Combines generator functions into one that runs them serially."""
  def composed_fns(generator=None):
    for f in fastmath.tree_flatten(fns):
      generator = f(generator)
    return generator
  return composed_fns


# TODO(jonni): Rename to Blend/Merge/Mix/Interleave/...?
def Parallel(  # pylint: disable=invalid-name
    fns=None,
    counters=None,
    reweight_by_minimum=False,
    gradually_reweight=False,
    use_remainders=False):
  """Combines generator functions into one that runs them in parallel.

  Args:
    fns: a sequence of datasets which are combined in parallel.
    counters: a sequence of ints with same length as fns, please see comments on
      its use below.
    reweight_by_minimum: if set to True, then we re-weight every counter by the
      minimal counter. E.g. counters (10000, 100000) are translated to (1, 10)
      and hence for every 10 examples from the second dataset we are getting
      1 example from the first dataset. Without reweighting first we would see
      20 examples from the first and second dataset and then 90 thousand eamples
      only from the first dataset.
    gradually_reweight: if set to True, then we loop through the generators
      using a recursive rule defined in emit_examples. First we sort generators
      by the counters. If we have datasets with counters 1, 20, 40
      (after sorting) then we yield examples (a(b c^2)^20)^*, where examples of
      type a come from the first dataset, of type b from the second and of type
      c from the third. The exponents are obtained through divisions of
      subsequent counters.
    use_remainders: if set to True as weell as gradually_reweight is set to
      True and counters are 1, 20, 45 then after dealing with all examples in
      the format (a(b c^2)^20)^*, the generator yields the remaining 5 examples
      from the dataset with counter 45.
  Returns:
    parallel_generator: the generator yields samples according to given;
    if counters are not given then samples are genereted uniformly.

  Example 1:

    gen = data.Parallel([dataset1, dataset2, dataset3], counters=(2, 1, 3))

  defines a generator that yields 33% examples from dataset1, 16% examples from
  dataset2 and 50% examples from dataset3.

  Example 2:

    gen = data.Parallel([dataset1, dataset2, dataset3], counters=(20, 50, 30))

  defines a generator that yields 20% examples from dataset1, 50% examples from
  dataset2 and 30% examples from dataset3.
  """

  if counters:
    assert len(counters) == len(fns)
    # Remove generators with zero counters
    counters = list(counters)
    fns = list(fns)
    non_zeros = [j for j in range(len(counters)) if counters[j] != 0]
    counters = [counters[j] for j in non_zeros]
    fns = [fns[j] for j in non_zeros]
  else:
    counters = [1] * len(fns)

  if reweight_by_minimum:
    counters = [math.floor(counter / min(counters)) for counter in counters]

  def emit_examples(sorted_counters_with_gens, prev_counter):
    if sorted_counters_with_gens:
      _, counter, generator = sorted_counters_with_gens[0]
      repeats = math.floor(counter / prev_counter)
      for _ in range(repeats):
        yield next(generator)
        yield from emit_examples(sorted_counters_with_gens[1:], counter)

  def parallel_generator(gen=None):
    # If gradually_reweight is set to False then
    # current_counters are increased step by step; they are reset to 0s when
    # current_counters[idx] == counters[idx] for all idx. See
    # test_parallel_with_weights_three_datasets for an example of how
    # current_counters are changed during computation.
    # If gradually_reweight is set to False then we loop using a
    # recursive rule defined in emit_examples.

    generators = []
    for f in fns:
      if gen:
        generators.append(f(gen))
      else:
        # This handles the case when the function f cannot be
        # called on None.
        generators.append(f())

    if gradually_reweight:
      counters_with_gens = zip(range(len(generators)), counters, generators)
      sorted_counters_with_gens = sorted(counters_with_gens, key=lambda x: x[1])
      while True:
        yield from emit_examples(sorted_counters_with_gens, min(counters))
        if use_remainders:
          # Below we are dealing with remainders.
          fractions = []
          for i in range(len(sorted_counters_with_gens)):
            _, counter, generator = sorted_counters_with_gens[i]
            processed = 1
            for fraction in fractions:
              processed *= fraction
            remainder = counter - processed
            for _ in range(remainder):
              yield next(generator)
            if i < len(sorted_counters_with_gens) - 1:
              _, next_counter, _ = sorted_counters_with_gens[i + 1]
              fractions.append(math.floor(next_counter / counter))
    else:
      current_counters = [0] * len(generators)
      while True:
        for idx, generator in enumerate(generators):
          if current_counters[idx] < counters[idx]:
            current_counters[idx] += 1
            # instead of checking current_counters[idx] == counters[idx] for
            # all idx, we check the equivalent condition:
            if sum(current_counters) == sum(counters):
              current_counters = [0] * len(generators)
            yield next(generator)

  return parallel_generator


@gin.configurable(module='trax.data')
def Shuffle(queue_size=1024):  # pylint: disable=invalid-name
  """Returns a shuffle function with the given queue size."""
  return lambda g: shuffle(g, queue_size)


@gin.configurable(module='trax.data')
def Batch(batch_size):  # pylint: disable=invalid-name
  """Returns a batching function with given batch size."""
  return lambda g: batch(g, batch_size)


@gin.configurable(module='trax.data')
def Dup():  # pylint: disable=invalid-name
  """Duplicates (copies) the top element (inputs).

  The generator stream is augmented in the following way:

  - If the stream consists of a single element `(inputs, )`,
    the inputs simply get copied to `(inputs, inputs)`.
  - If the stream consists of multiple elements, for example
    `(inputs, weights)`, the rest of elements get moved toward
    the right side `(inputs, inputs, weights)`.

  Returns:
    the duplicating function.
  """
  def _copy(xs):
    x, *rest = xs
    return (x, x, *rest)
  return lambda g: map(lambda x: _copy(x), g)  # pylint: disable=unnecessary-lambda


@gin.configurable(module='trax.data')
def FilterEmptyExamples(axes=None, debug=False):  # pylint: disable=invalid-name
  """Filters empty examples.

  Filters any example that has an array of size (0,) (if axes=None).
  Alternatively, checks only axes provided in `axes' list. Contrary to
  FilterByLength used with several elements with length_axis, here the example
  would be filtered if ANY of the dimensions listed in `axes' contains an empty
  array.

  Args:
    axes: list of indices to check, if None, all of them.
    debug: If true, emits a log everytime we filter out an empty example.

  Returns:
    Function filtering empty examples.
  """
  def _filter_examples(generator):
    for example in generator:
      correct = True
      for i, unused_tuple_element in enumerate(example):
        if axes is None or i in axes:
          if example[i].shape == (0,):
            correct = False
            break
      if correct:
        yield example
      elif debug:
        logging.info('Filtered example: %r', example)
  return _filter_examples


@gin.configurable(module='trax.data')
def FilterByLength(max_length, min_length=0,  # pylint: disable=invalid-name
                   length_keys=None, length_axis=0):
  """Returns a function that filters out examples by length.

  Args:
    max_length: int. If not None, indicates maximum length.
    min_length: int. If not None, indicates minimum length.
    length_keys: (list) which example keys to take into account.
    length_axis: which shape axis to take into account.
  Returns:
    a function that filters out examples by length.
  """

  assert max_length is not None or min_length is not None
  length_keys = length_keys or [0, 1]
  length_fn = lambda x: _length_fn(x, length_axis, length_keys)
  def filtered(gen):
    for example in gen:
      example_len = length_fn(example)

      # Checking max length boundary.
      if max_length is not None:
        if example_len > max_length:
          continue
      # Checking min length boundary.
      if min_length is not None:
        if example_len < min_length:
          continue
      # Within bounds.
      yield example
  return filtered


@gin.configurable(module='trax.data')
def TruncateToLength(len_map=None):  # pylint: disable=invalid-name
  """Returns a stream function that resizes items as specified by ``len_map``.

  Args:
    len_map: Dictionary that specifies maximum shapes for potentially multiple
        features per stream item. For example, given a stream of tokenized
        string pairs, one could enforce a maximum length of 256 tokens for each
        string by using ``len_map={0: (256,), 1: (256,)}``.
  """
  @debug_data_pipeline.debug_pipeline
  def _truncate_to_length(generator):
    for example in generator:
      if isinstance(example, np.ndarray):
        example = (example,)
      if isinstance(example, (list, tuple)):
        example = list(example)
        if len_map is not None:
          for key, max_len in len_map.items():
            example_len = example[key].shape
            if example_len > max_len:
              example[key] = np.resize(example[key], max_len)
        output = tuple(example)
      else:
        output = None
        raise ValueError(f'Unknown example type: {example}')
      yield output

  return _truncate_to_length


@gin.configurable(module='trax.data')
def PadToLength(  # pylint: disable=invalid-name
    len_map=None, pad_value=0, multiple=False):
  """Pads the values to lengths given in `len_map'.

  len_map contains a dictionary of example keys to dimension sizes.

  Args:
    len_map: dict of int to int, we pad examples to lengths
      given by the values of the dict. If multiple is True, the dimensions are
      padded to multiple of this value.
    pad_value: dict of int to int. The value gets applied to
      constant_values on numpy.pad per given dimension.
    multiple: boolean. If False, pads to the value of len_map. If True, pads to
      closest multiple of value of len_map.
  Returns:
    Function to pad examples to given lengths.
  """
  @debug_data_pipeline.debug_pipeline
  def _pad_to_length(generator):
    for example in generator:
      if isinstance(example, (list, tuple)):
        example = list(example)
        for key, value in len_map.items():
          array_length = example[key].shape[0]
          if multiple:
            padding_len = array_length - ((array_length // value) * value)
          else:
            padding_len = max([0, value-example[key].shape[0]])
          example[key] = np.pad(example[key],
                                pad_width=(0, padding_len),
                                mode='constant',
                                constant_values=pad_value[key])
        output = tuple(example)
      else:
        if not isinstance(example, np.ndarray):
          raise ValueError(f'example isn\'t nparray, but should be: {example}')
        array_length = example.shape[0]
        if multiple:
          padding_len = (
              array_length - ((array_length // len_map[0]) * len_map[0]))
        else:
          padding_len = max(0, len_map[0] - array_length)
        output = np.pad(example,
                        pad_width=(0, padding_len),
                        mode='constant',
                        constant_values=pad_value[0])
      yield output
  if len_map is None:
    raise ValueError('len_map parameter should be provided.')
  return _pad_to_length


@gin.configurable(module='trax.data')
def BucketByLength(boundaries, batch_sizes,  # pylint: disable=invalid-name
                   length_keys=None, length_axis=0, strict_pad_on_len=False):
  """Returns a function for bucketing inputs, see `bucket_by_length`."""
  length_keys = length_keys or [0, 1]
  # In all cases so far, we use a length function of the following form.
  length_fn = lambda x: _length_fn(x, length_axis, length_keys)
  return lambda g: bucket_by_length(  # pylint: disable=g-long-lambda
      g, length_fn, boundaries, batch_sizes, strict_pad_on_len)


@gin.configurable(module='trax.data')
def MLM(vocab_size=None,  # pylint:disable=invalid-name
        max_length=None,
        noise_density=0.15,
        mean_noise_span_length=3.0):
  """Pipeline that just does MLM."""
  return Serial(
      # Generate sequential chunks.
      generate_sequential_chunks(max_length=max_length),
      # Generate mask and chunk.
      generate_random_noise_mask(
          noise_density=noise_density,
          mean_noise_span_length=mean_noise_span_length),
      # Consume mask and chunk to give (input, targets).
      consume_noise_mask(vocab_size=vocab_size),
  )


@gin.configurable(module='trax.data')
def PrefixLM(input_length=128, output_length=512):  # pylint:disable=invalid-name
  """Chunks examples so as to make inputs/outputs of specified lenghts."""
  def _f(generator):
    for example in generator:
      n_tokens = len(example)
      # Iterate:
      # |--------|<---- input_length ---->|<- output_length ->|--------------|
      # ^        ^                        ^                   ^
      # |        |                        |                   |
      # 0        input_begin_idx          input_end_idx       output_end_idx
      input_begin_idx = 0
      # While you can make an input batch, keep going.
      while input_begin_idx + input_length < n_tokens:
        input_end_idx = input_begin_idx + input_length
        output_end_idx = min(input_end_idx + output_length, n_tokens)
        yield (example[input_begin_idx:input_end_idx],
               example[input_end_idx:output_end_idx])
        # Update the indices.
        input_begin_idx = output_end_idx
  return _f


@gin.configurable(module='trax.data')
def ConcatenateToLMInput(pad_to_length=None):  # pylint: disable=invalid-name
  """Prepares the input needed for training of Language Models.

  Each example needs to contain two elements (input and target).
  Input is concatenated to target and, if pad_to_length is given, padded to
  length provided.
  The loss_weights indicates only the target, without input nor padding.

  Args:
    pad_to_length: int, total length of padding of input and target arrays.
  Returns:
    Function to return input for a LM.
  """
  @debug_data_pipeline.debug_pipeline
  def _concatenate_to_lm_input(generator):
    for example in generator:
      if isinstance(example, (list, tuple)) and (len(example) == 2):
        concatenated = np.concatenate((example[0], example[1]), axis=-1)
        loss_weights = np.concatenate((np.zeros_like(example[0]),
                                       np.ones_like(example[1])))
        if pad_to_length is not None:
          padding_len = pad_to_length - (
              example[0].shape[0] + example[1].shape[0])
          if padding_len < 0:
            raise ValueError(
                'Example lengths '
                f'({example[0].shape[0]}, {example[1].shape[0]}) '
                f'longer than pad_to_length ({pad_to_length}).')
          loss_weights = np.pad(loss_weights, (0, padding_len), 'constant')
          concatenated = np.pad(concatenated, (0, padding_len), 'constant')
        output = (concatenated, concatenated, loss_weights)
      elif isinstance(example, (list, tuple)) and (len(example) == 1):
        # Make x into (x, x)
        output = (example[0], example[0])
      elif isinstance(example, np.ndarray):
        # Make x into (x, x)
        output = (example, example)
      else:
        output = None
        raise ValueError(f'Unknown input to ConcatenateToLMInput: {example}')
      yield output

  return _concatenate_to_lm_input


@gin.configurable(module='trax.data')
def CastTo(dtype=np.int32, indices=(0, 1,), debug=False):  # pylint: disable=invalid-name
  """Casts the given indices to the given dtype."""
  def _cast_fn(generator):
    debug_count = 0
    for example in generator:
      debug_count += 1
      assert isinstance(example, tuple)
      example = list(example)
      dtype_mismatch = False
      original_index_and_dtype = []
      for i in range(len(example)):
        if i not in indices:
          continue
        original_type = example[i].dtype
        if original_type != dtype:
          if not (original_type == np.int64 and dtype == np.int32):
            # Downcasting from np.int64 to np.int32 is OK
            original_index_and_dtype.append((i, original_type))
          example[i] = example[i].astype(dtype)
          dtype_mismatch = True
      if debug and dtype_mismatch and original_index_and_dtype:
        logging.info('dtype mismatch in example[%d] = %r was earlier: %r',
                     debug_count, example, original_index_and_dtype)
      yield tuple(example)
  return _cast_fn


@gin.configurable(module='trax.data')
def AppendValue(val=None):  # pylint: disable=invalid-name
  """Appends values provided in 'val` to inputs.

  val are keyed by example keys, its values contain appended tensors.

  Args:
    val: dict of int to tensors. Specific keys get the tensors specified in
      values appended.
  Returns:
    Funtion to append tensors to examples.
  """
  @debug_data_pipeline.debug_pipeline
  def _append_value(generator):
    for example in generator:
      if isinstance(example, tuple):
        example = list(example)
        if val is not None:
          for key, value in val.items():
            example[key] = np.append(example[key], value, -1)
        output = tuple(example)
      else:
        if not isinstance(example, np.ndarray):
          raise ValueError(f'example isn\'t nparray, but should be: {example}')
        output = np.append(example, val[0])
      yield output

  return _append_value


@gin.configurable(module='trax.data')
def AddLossWeights(id_to_mask=None):  # pylint: disable=invalid-name
  """Returns a function to add loss weights; see `add_loss_weights`."""
  return lambda g: add_loss_weights(g, id_to_mask=id_to_mask)


@gin.configurable(module='trax.data')
def UnBatch():  # pylint: disable=invalid-name
  """Returns a function which unbatches."""
  def _unbatch(generator):
    for batched_example in generator:
      # batched_example is usually like:
      # (batched_inputs, batched_outputs) or
      # (batched_inputs, batched_outputs, batched_weights)
      assert isinstance(batched_example, tuple)
      # assert all lengths are the same.
      batch_sizes = list(set(map(lambda example: example.shape[0],
                                 batched_example)))
      assert len(batch_sizes) == 1
      # Now unbatch examples.
      for example_idx in range(batch_sizes[0]):
        yield tuple(map(lambda x: x[example_idx], batched_example))  # pylint: disable=cell-var-from-loop
  return _unbatch


@gin.configurable(module='trax.data')
def Prefetch(n_prefetch=2):  # pylint: disable=invalid-name
  """Pre-fetches a number of examples from generator in a separate process."""
  def prefetch(generator):
    in_q, out_q = mp.Queue(), mp.Queue()
    p = mp.Process(target=_generator_process, args=(generator, in_q, out_q))
    for _ in range(n_prefetch):
      in_q.put(None)
    p.start()
    while True:
      yield out_q.get()
      in_q.put(None)
  return prefetch


@gin.configurable(module='trax.data')
def UniformlySeek(name=None, host_id=None, n_hosts=None, dataset_size=None):  # pylint: disable=invalid-name
  """Sets each host at (dataset_size/n_hosts)-th of the dataset."""
  if not dataset_size:
    dataset_size = 2 ** 18  # 512 * 512
    logging.error(
        'No dataset size given to Uniformly seek, assuming: %d', dataset_size)
  assert name
  host_id = jax.process_index() if host_id is None else host_id
  n_hosts = n_hosts or jax.host_count()
  each_host = int(dataset_size / n_hosts)
  def _f(generator):
    # Each host seeks to the appropriate point in the dataset.
    num_to_seek = int(host_id * each_host)
    start_time = time.time()
    logging.info('Dataset[%s] host_id[%d] is seeking to position[%d]',
                 name, host_id, num_to_seek)
    for _ in range(num_to_seek):
      next(generator)
    logging.info('Dataset[%s] host_id[%d] reached position[%d]. '
                 'Time taken [%s] seconds',
                 name, host_id, num_to_seek, time.time() - start_time)
    for example in generator:
      yield example
  return _f


@gin.configurable(module='trax.data')
def CountAndSkip(name):  # pylint: disable=invalid-name
  """Returns a function that counts and skips examples (see above)."""
  return lambda g: count_and_skip(g, name)


@gin.configurable(module='trax.data')
def Log(n_steps_per_example=1, only_shapes=True):  # pylint: disable=invalid-name
  """Creates a logging component of the input pipeline."""
  def log(stream):
    counter = 0
    for example in stream:
      item_to_log = example
      if only_shapes:
        item_to_log = fastmath.nested_map(shapes.signature, example)
      if counter % n_steps_per_example == 0:
        logging.info(str(item_to_log))
        print(item_to_log)
      counter += 1
      yield example
  return log


def shuffle(samples, queue_size):
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


def batch(generator, batch_size):
  """Batch and pad generator as in tf.data.Dataset.padded_batch."""
  if batch_size <= 0:
    raise ValueError(f'Batch size must be positive, but is {batch_size}.')
  buf = []
  i = 0
  for example in generator:
    buf.append(example)  # Examples are tuples of tensors.
    if len(buf) == batch_size:
      # buf is a list of tuples, e.g., [(in1, tgt1), (in2, tgt2), (in3, tgt3)]
      # batch is a tuple of arrays: ([in1, in2, in3], [tgt1, tgt2, tgt3])
      try:
        batched_example = tuple(
            pad_to_max_dims([np.asarray(tensor) for tensor in x])
            for x in zip(*buf))
      except ValueError as e:
        for j in range(len(buf)):
          logging.error('Batch[%d][%d] input shape: %r output shape: %r',
                        i, j, buf[j][0].shape, buf[j][1].shape)
        for j in range(len(buf)):
          logging.error('Batch[%d][%d] input: %r', i, j, buf[j][0])
          logging.error('Batch[%d][%d] output: %r', i, j, buf[j][1])
        raise e
      i += 1
      yield batched_example
      buf = []


def pad_to_max_dims(tensors, boundary=None, strict_pad_on_len=False):
  """Pad a tuple of tensors to a joint dimension and return their batch.

  For example, a pair of tensors of shape (2, 10) and (3, 9) will be padded
  to (3, 10) both and the returned tensor will have shape (2, 3, 10).

  When boundary is specified, we try to pad all unknown dimensions to boundary
  if possible, which can help reduce the number of different shapes occurring
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
    if max_len == min_len and max_len == boundary:  # No padding needed.
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

  This function draws examples from the provided `generator` and puts an
  example into a bucket depending on `l = length_fn(example)`. Which bucket
  is used depends on between which `boundaries` is l. When a bucket reaches
  its batch size, as specified by `batch_sizes`, generates a batch of
  padded examples from this bucket.

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
    bucket_idx = min([i for i, b in enumerate(boundaries) if length <= b])
    buckets[bucket_idx].append(example)
    if len(buckets[bucket_idx]) == batch_sizes[bucket_idx]:
      batched = zip(*buckets[bucket_idx])
      boundary = boundaries[bucket_idx]
      boundary = None if boundary == math.inf else boundary
      padded_batch = tuple(
          pad_to_max_dims(x, boundary, strict_pad_on_len) for x in batched)
      yield padded_batch
      buckets[bucket_idx] = []


@debug_data_pipeline.debug_pipeline
def add_loss_weights(generator, id_to_mask=None):
  """Add weights to inputs without weights and masks by id if requested.

  The generator stream is augmented in the following way:

  - If the stream consists of pairs `(inputs, targets)`, a loss mask is added
    that is creates as a tensor of ones of the same shape as targets.
  - If `id_to_mask` is not `None`, and the stream (after the previous point)
    has triples `(inputs, targets, weights)`, the weights are multiplied by a
    0/1 mask that is 0 iff targets is equal to `id_to_mask` (1 otherwise).

  Args:
    generator: Stream of tuples.
    id_to_mask: If not None, int-valued id that represents padding, as opposed
        to true target IDs.

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
      output = (example[0], example[1], weights)
      yield output


@gin.configurable(module='trax.data')
def generate_random_noise_mask(noise_density=0.15,
                               mean_noise_span_length=3.0,
                               seed1=None,
                               seed2=None):
  """Returns a function that generates a random noise mask."""
  def _f(generator):
    for example in generator:
      length = len(example)
      noise_mask = random_spans_noise_mask(
          length, noise_density=noise_density,
          mean_noise_span_length=mean_noise_span_length,
          seed1=seed1, seed2=seed2, example=example)
      yield (example, noise_mask)
  return _f


@gin.configurable(module='trax.data')
def consume_noise_mask(vocab_size=32100):
  """Consumes (tokens, noise mask) and returns (inputs, targets)."""
  def _noise_span_to_unique_sentinel(tokens, noise_mask):
    prev_token_is_noise = np.pad(
        noise_mask[:-1], [1, 0], mode='constant', constant_values=False)
    first_noise_tokens = np.logical_and(noise_mask,
                                        np.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)
    sentinel = vocab_size - np.cumsum(first_noise_tokens)
    tokens = np.where(first_noise_tokens, sentinel, tokens)
    return tokens[np.logical_not(subsequent_noise_tokens)]

  def _f(generator):
    for tokens, noise_mask in generator:
      # Returns inputs and targets.
      yield (_noise_span_to_unique_sentinel(tokens, noise_mask),
             _noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask)))
  return _f


@gin.configurable(module='trax.data')
def generate_sequential_chunks(max_length=None):
  """Returns a function that generates chunks of atmost max_length length."""
  def _f(generator):
    for example in generator:
      n_tokens = len(example)
      if n_tokens <= max_length:
        yield example
      else:
        n_segments = int(math.ceil(float(n_tokens) / float(max_length)))
        for i in range(n_segments):
          start = max_length * i
          end = min(start + max_length, n_tokens)
          yield example[start:end]
  return _f


@gin.configurable(module='trax.data')
def addition_input_stream(
    vocab_size=gin.REQUIRED, batch_size=gin.REQUIRED, min_length=gin.REQUIRED,
    max_length=gin.REQUIRED, pad_to_multiple=32, encdec=False):
  """Data stream for the add problem: <S>x+y<S>(x+y).

  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    min_length: minimal length of w.
    max_length: maximal length of w.
    pad_to_multiple: int, pad length to be multiple of this number.
    encdec: bool, if True return encoder-decoder style inputs (default: False)

  Returns:
    python generator of tuples of data examples
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
    if encdec:
      x = [i + 1 for i in inp]
      y = [i + 1 for i in tgt]
      weights = [1] * len(tgt)
      candidate_example = (np.array(x), np.array(y), np.array(weights))
      if any(len(sample) > max_length for sample in candidate_example):
        # sample too long, try again
        return single_example(max_length, min_length)
      return (np.array(x), np.array(y), np.array(weights))
    else:
      x = [base+2] + [i+1 for i in inp] + [base+2] + [i+1 for i in tgt]
      weights = ([0] * (len(inp) + 2)) + ([1] * len(tgt))
      return (np.array(x), np.array(x), np.array(weights))

  def batches(max_length, min_length):
    """Batches of examples."""
    if max_length < 3 or min_length < 3:
      raise ValueError('Maximum/minimum length must be at least 3.')
    while True:
      ex = [single_example(max_length, min_length) for _ in range(batch_size)]
      padded_batch = [pad_to_max_dims(x, boundary=pad_to_multiple,
                                      strict_pad_on_len=True)
                      for x in zip(*ex)]
      yield tuple(padded_batch)

  return batches(max_length, min_length)


# This is a straightforward translation of T5's random_spans_noise_mask.
def random_spans_noise_mask(length,
                            noise_density=0.15,
                            mean_noise_span_length=3.0,
                            seed1=None,
                            seed2=None,
                            example=None):
  """Computes span corruption masks given input parameters."""
  # Passing this in case if we want to use for debugging/logging
  del example
  orig_length = length
  # increase length to avoid degeneracy
  length = max(length, 2)
  num_noise_tokens = int(round(length * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
  num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = max(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens

  # Pick the lengths of the noise spans and the non-noise spans
  def randomly_segment(num_items, num_segments, seed):
    x = np.arange(num_items - 1) < num_segments - 1
    # Set random seed if passed (only in tests for now).
    if seed is not None:
      np.random.seed(seed)
    np.random.shuffle(x)
    first_in_segment = np.pad(x, (1, 0), mode='constant')
    segment_id = np.cumsum(first_in_segment)

    y = np.roll(segment_id, 1)
    y[0] = 0
    idxs = np.pad(np.squeeze(np.argwhere(segment_id - y), axis=1),
                  (1, 0),
                  mode='constant')
    segment_lengths = np.add.reduceat(np.ones_like(segment_id), idxs, axis=0)
    return segment_lengths

  noise_span_lengths = randomly_segment(
      num_noise_tokens, num_noise_spans, seed1)
  nonnoise_span_lengths = randomly_segment(
      num_nonnoise_tokens, num_noise_spans, seed2)
  interleaved_span_lengths = np.reshape(
      np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
      [num_noise_spans * 2])
  span_starts = np.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = np.zeros(length)  # all 0s to begin with
  span_start_indicator[span_starts] = 1
  span_num = np.cumsum(span_start_indicator)
  is_noise = np.equal(span_num % 2, 1)
  return is_noise[:orig_length]


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


data_counters = {}  # Used by {load,save}_data_counters and count_and_skip


def count_and_skip(generator, name):
  """Count the number of items in the generator, skip already counted ones.

  This function counts the number of processed examples and puts it into
  the global variable `counters`. This variable can be saved and restored,
  and if restored, this function will skip examples until the restored counter
  is reached. When the data generator is deterministic, this allows to restore
  the data reading process from a checkpoint.

  Args:
    generator: generator for examples in the dataset.
    name: string, a unique id that we use to count the examples

  Yields:
    The examples from generator but first skip the number specified in the
    global variable counters[name] and next increment this variable every
    time a new example appears.
  """
  global data_counters
  local_counter = 0
  for example in generator:
    local_counter += 1
    # This check must be inside the loop due to asynchronous initializations.
    if name not in data_counters:
      data_counters[name] = 0
    if local_counter > data_counters[name]:
      data_counters[name] += 1
      yield example


def save_data_counters(output_dir, host_id=None):
  """Checkpoint data counters."""
  global data_counters
  host_id = jax.process_index() if host_id is None else host_id
  fname = os.path.join(output_dir, 'data_counters%d.pkl' % host_id)
  with tf.io.gfile.GFile(fname, 'wb') as f:
    pickle.dump(data_counters, f)


def load_data_counters(output_dir, host_id=None):
  """Checkpoint data counters."""
  global data_counters
  host_id = jax.process_index() if host_id is None else host_id
  fname = os.path.join(output_dir, 'data_counters%d.pkl' % host_id)
  if not tf.io.gfile.exists(fname):
    logging.info('Did not load data counters as %s does not exist.', fname)
    return
  with tf.io.gfile.GFile(fname, 'rb') as f:
    obj = pickle.load(f)
  data_counters = obj


def _generator_process(generator, in_q, out_q):
  for example in generator:
    in_q.get()
    out_q.put(example)


def _buckets_for_length(bucket_length, batch_size, max_eval_length, n_devices,
                        training):
  """Creates heuristically a set of bucket boundaries and sizes.

  The middle boundary is set to `bucket_length` and the corresponding batch
  size is set to `batch_size`. We also create buckets of 1/2 and 1/4 length
  with 2x and 4x batch size, and buckets of 2x and 4x and larger length with
  1/2 and 1/4 batch size respectively, and batch size 1 for the final one.

  Args:
    bucket_length: the length of the middle bucket.
    batch_size: the batch size for the middle bucket.
    max_eval_length: the longest bucket length if training=False.
    n_devices: number of devices, batch sizes are divisible by that.
    training: bool, whether we are training or evaluating.

  Returns:
    a pair of lists of integers, (bucket_boundaries, bucket_batch_sizes).
  """
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
  bucket_batch_sizes = [batch_size * 4, batch_size * 2,
                        batch_size, batch_size // 2,
                        batch_size // 4, batch_size // 8,
                        batch_size // 16, 1]
  if not training:
    # The last bucket batch size is always 1, but the one-but-last is
    # sized to accommodate the final length = bucket_boundaries[-1], which
    # we changed for eval above -- so adjusting here too.

    # Resize if needed, since bucket_batch_sizes may not be the same size
    # anymore.
    bucket_batch_sizes = bucket_batch_sizes[:len(bucket_boundaries)] + [1]
    bucket_batch_sizes[-2] = batch_size // max_eval_length
  # Make batch sizes divisible by n_devices.
  bucket_batch_sizes = [max(b // n_devices, 1) * n_devices
                        for b in bucket_batch_sizes]
  return (bucket_boundaries, bucket_batch_sizes)


def _length_fn(example, length_axis, length_keys):
  """Length is the maximum of shape on length_axis over length_keys."""
  if isinstance(example, (list, tuple)):
    return max([example[i].shape[length_axis] for i in length_keys])
  return example.shape[length_axis]


# ########################################################################
# Inputs class used by Trainer, and associated helper functions.
#
# Note: In the planned move from Trainer to Loop, the Inputs class should be
# deprecated and finally removed.


class Inputs:
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


# Batching and Inputs creation helpers.


@gin.configurable(module='trax.data')
def make_inputs(train_stream=gin.REQUIRED, eval_stream=None):
  """Create Inputs from two streams; mostly for use in gin configs."""
  if isinstance(train_stream, (list, tuple)):
    train_stream = Serial(train_stream)()
  if isinstance(eval_stream, (list, tuple)):
    eval_stream = Serial(eval_stream)()
  eval_stream_fn = None if eval_stream is None else lambda _: eval_stream
  return Inputs(train_stream=lambda _: train_stream,
                eval_stream=eval_stream_fn)


@gin.configurable(module='trax.data')
def make_additional_stream(stream=gin.REQUIRED):
  """Create a stream mostly for use in gin configs for additional tasks."""
  return Serial(stream)()


@gin.configurable(module='trax.data')
def make_parallel_stream(streams=gin.REQUIRED, counters=None):
  """Create a parallel stream for use in gin configs for additional tasks."""
  return Parallel(streams, counters=counters)()


@gin.configurable(module='trax.data')
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
      buckets = _buckets_for_length(
          bucket_length, cur_batch_size, max_eval_length, n_devices, training)

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
    dataset = batch(dataset, cur_batch_size)
  if training and batch_shuffle_size is not None:
    dataset = shuffle(dataset, batch_shuffle_size)
  return add_loss_weights(dataset, id_to_mask)


# Example input functions.


@gin.configurable(module='trax.data')
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


@gin.configurable(module='trax.data')
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


@gin.configurable(module='trax.data')
def simple_sequence_copy_inputs(
    vocab_size=gin.REQUIRED, batch_size=gin.REQUIRED, train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED, eval_max_length=gin.REQUIRED,
    pad_to_multiple=32):
  """Inputs for the sequence copy problem: w for w in [1..vocab_size-1]*.

  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    train_length: maximum length of w for training.
    eval_min_length: minimum length of w for eval.
    eval_max_length : maximum length of w for eval.
    pad_to_multiple: int, pad length to be multiple of this number.

  Returns:
    trax.inputs.Inputs
  """
  def random_minibatches(length_list):
    """Generate a stream of random mini-batches."""
    while True:
      length = random.choice(length_list)
      x = np.random.randint(low=1, high=vocab_size-1,
                            size=(batch_size, length))
      loss_weights = np.ones((batch_size, length))
      x = _pad_to_multiple_of(x, pad_to_multiple, 1)
      loss_weights = _pad_to_multiple_of(loss_weights, pad_to_multiple, 1)
      yield (x, x, loss_weights)  # Here inputs and targets are the same.

  train_lengths = list(range(1, train_length + 1))
  eval_lengths = list(range(eval_min_length, eval_max_length + 1))
  return Inputs(
      train_stream=lambda _: random_minibatches(train_lengths),
      eval_stream=lambda _: random_minibatches(eval_lengths)
  )


@gin.configurable(module='trax.data')
def addition_inputs(
    vocab_size=gin.REQUIRED, batch_size=gin.REQUIRED, train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED, eval_max_length=gin.REQUIRED,
    pad_to_multiple=32, encdec=False):
  """Inputs for the add problem: <S>x+y<S>(x+y).

  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    train_length: maximal length of w for training.
    eval_min_length: minimal length of w for eval.
    eval_max_length: maximal length of w for eval.
    pad_to_multiple: int, pad length to be multiple of this number.
    encdec: bool, if True return encoder-decoder style inputs (default: False)

  Returns:
    trax.inputs.Inputs
  """
  train_stream = addition_input_stream(
      vocab_size, batch_size, 3, train_length, pad_to_multiple, encdec)
  eval_stream = addition_input_stream(
      vocab_size, batch_size, eval_min_length, eval_max_length, pad_to_multiple,
      encdec)
  return Inputs(
      train_stream=lambda _: train_stream,
      eval_stream=lambda _: eval_stream
  )


@gin.configurable(module='trax.data')
def sine_inputs(
    batch_size=gin.REQUIRED,
    length=gin.REQUIRED,
    max_phase=(2 * math.pi),
    min_period=0.1,
    max_period=10.0,
):
  """Sinusoids of random period and phase.

  Args:
    batch_size (int): Number of examples in a batch.
    length (int): Length of each sequence.
    max_phase (float): Maximum phase of the sinusoids.
    min_period (float): Minimum period of the sinusoids.
    max_period (float): Maximum period of the sinusoids.

  Returns:
    trax.inputs.Inputs
  """
  def random_series():
    while True:
      phase = np.random.uniform(0, max_phase)
      period = np.exp(np.random.uniform(np.log(min_period), np.log(max_period)))
      x = np.arange(length)
      yield np.sin((x - phase) / period)

  def random_minibatches(_):
    minibatch = []
    for series in random_series():
      minibatch.append(series)
      if len(minibatch) == batch_size:
        obs = np.stack(minibatch)
        minibatch.clear()
        act = np.zeros_like(obs, dtype=np.int32)
        mask = np.ones_like(obs)
        yield (obs, act, obs, mask)

  return Inputs(train_stream=random_minibatches, eval_stream=random_minibatches)


def _pad_to_multiple_of(x, y, axis):
  """Pads x to multiple of y on the given axis."""
  pad_len = np.ceil(x.shape[axis] / float(y)) * y
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, int(pad_len - x.shape[axis]))
  return np.pad(x, pad_widths, mode='constant',
                constant_values=x.dtype.type(0))
