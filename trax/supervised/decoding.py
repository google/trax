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

"""Decoding with Trax models."""

import numpy as np
from trax import fastmath
from trax import layers as tl


def autoregressive_sample_stream(model, inputs=None,
                                 batch_size=1, temperature=1.0,
                                 start_id=0, accelerate=True,
                                 eval_mode=False, eval_min_length=1):
  """Yields samples from `model`, in autoregressive language model fashion.

  This function uses `model` to generate outputs one position at a time, with
  access to inputs for the current position and all preceding positions. The
  new output becomes the next position's input, and further calls to
  `autoregressive_sample_stream` repeat the process for successive positions
  indefinitely.

  Inputs and outputs always come in batches, even if size 1. If `inputs` is
  present, it must have shape (`batch_size`, inputs_sequence_length), and each
  output in the stream has shape (`batch_size`, 1).

  Args:
    model: A layer object (subclass of `trax.layers.Layer`) created in
        `'predict'` mode and initialized from trained weights. The model
        must have a structure that allows it to run as an autoregressive
        one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`),
        except if `eval_mode` is set -- any model can be sampled then,
        but the sampling process may be much slower.
    inputs: Sequence of symbols the model sees as input the first time it
        generates an output. If None, the model generates the first output
        based on just the start symbol.
    batch_size: Number of sequences to generate in parallel as a batch.
    temperature: Parameter that controls the sharpness of the softmax that
        feeds the sampling process. Values range from 0.0 (all probability mass
        goes to one candidate; like an argmax) to positive infinity (all
        candidates have equal probability).
    start_id: Integer representing the start symbol for the autoregressive
        process, or array of shape (`batch_size`, 1) of such integers.
    accelerate: If True, create an accelerated version of `model` and use it
        for generating outputs.
    eval_mode: If True, assume the model is created in `eval` mode and sample
        by collecting all previous outputs and passing the whole tensor.
    eval_min_length: If set, the minimum length to pad to in eval mode.

  Yields:
    Tensor of integers with shape (`batch_size`, 1), representing the batch of
    outputs for the next position in the stream.
  """
  if inputs is not None and inputs.shape[0] != batch_size:
    raise ValueError(f'Inputs batch size ({inputs.shape[0]}) does not match '
                     f'batch_size arg ({batch_size}.')

  fast_model = tl.Accelerate(model) if accelerate else model
  if np.isscalar(start_id):
    start_symbol = np.full((batch_size, 1), start_id, dtype=np.int32)
  else:
    start_symbol = start_id
  if model.n_in == 1 and inputs is not None:
    current_symbols = np.concatenate([start_symbol, inputs], axis=1)
  else:
    current_symbols = start_symbol

  if eval_mode:
    # no start symbol needed in eval mode
    current_symbols = current_symbols[:, 1:]

  while True:
    # Pad inputs to power-of-2 length if needed.
    if eval_mode:
      # one extra symbol as an initial one will be added
      l = max(eval_min_length, current_symbols.shape[1] + 1)
      pad_len = int(2**np.ceil(np.log2(l))) - current_symbols.shape[1]
      unpadded_symbols = current_symbols
      current_symbols = np.pad(
          current_symbols, [[0, 0], [0, pad_len]], mode='constant')
      last_index = -pad_len  # no -1 as the starting one will be added
    else:
      last_index = -1
    # Run the model.
    if model.n_in > 1 and inputs is not None:
      logits = fast_model((inputs, current_symbols))[0]
    else:
      logits = fast_model(current_symbols)
    logits = tl.log_softmax(logits[:, last_index, :])
    sample = tl.logsoftmax_sample(logits, temperature=temperature)
    yield sample
    if eval_mode:
      current_symbols = np.concatenate(
          [unpadded_symbols, sample[:, None]], axis=1)
    else:
      # NOTE: Because the model is autoregressive and in 'predict' mode, its
      # history is cached in the model state and the next input is the single
      # symbol just sampled.
      current_symbols = sample[:, None]


def autoregressive_sample(model, inputs=None,
                          batch_size=1, temperature=1.0,
                          start_id=0, eos_id=1, max_length=100,
                          accelerate=True, eval_mode=False, eval_min_length=1):
  """Returns a batch of sequences created by autoregressive sampling.

  This function uses `model` to generate outputs one position at a time, with
  access to inputs for the current position and all preceding positions. The
  new output becomes the next position's input, and this loop repeats until
  either the model outputs the `eos_id` value or the output sequence reaches
  `max_length` items.

  Args:
    model: A layer object (subclass of `trax.layers.Layer`) created in
        `'predict'` mode and initialized from trained weights. The model
        must have a structure that allows it to run as autoregressive
        one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`),
        except if `eval_mode` is set -- any model can be sampled then,
        but the sampling process may be much slower.
    inputs: Sequence of symbols the model sees as input the first time it
        generates an output. If None, the model must generate the first output
        with no input to guide it.
    batch_size: Number of sequences to generate in parallel as a batch.
    temperature: Parameter that controls the sharpness of the softmax that
        feeds the sampling process. Values range from 0.0 (all probability mass
        goes to one candidate; like an argmax) to positive infinity (all
        candidates have equal probability).
    start_id: The start symbol (ID/integer) for the autoregressive process,
        or array of shape (`batch_size`, 1) of such integers.
    eos_id: The end-of-sequence symbol (ID/integer) for the autoregressive
        process.
    max_length: Maximum length for generated sequences.
    accelerate: If True, create an accelerated version of `model` and use it
        for generating outputs.
    eval_mode: If True, assume the model is created in `eval` mode and sample
        by collecting all previous outputs and passing the whole tensor.
    eval_min_length: If set, the minimum length to pad to in eval mode.

  Returns:
    Tensor of integers with shape (`batch_size`, output_length) representing
    a batch of output sequences. output_length is the maximum length of the
    output sequences, where each sequence can be no longer than `max_length`.
  """
  result = []
  eos_seen = []
  counter = 0
  for sample in autoregressive_sample_stream(
      model, inputs, batch_size=batch_size, temperature=temperature,
      start_id=start_id, accelerate=accelerate, eval_mode=eval_mode,
      eval_min_length=eval_min_length):
    sample = sample[:, None]
    result.append(sample)
    counter += 1
    if counter >= max_length:
      return np.concatenate(result, axis=1)
    # Check at which batch positions have we already encountered EOS.
    for j in range(batch_size):
      if int(sample[j, 0]) == eos_id:
        eos_seen.append(j)
    # If EOS has been seen on all positions, stop.
    if all([j in eos_seen for j in range(batch_size)]):
      return np.concatenate(result, axis=1)
  return np.concatenate(result, axis=1)


def beam_search(model, inputs=None, batch_size=1, n_beams=2, start_id=0,
                eos_id=1, max_length=100, length_penalty=1.0, accelerate=True):
  """Returns a batch of n_beams-sequences created by beam search.

  This function uses `model` to generate outputs one position at a time, with
  access to inputs for the current position and all preceding positions. The
  new output becomes the next position's input, and this loop repeats until
  either the model outputs the `eos_id` value or the output sequence reaches
  `max_length` items -- but keeping n_beams top beams.

  Args:
    model: A layer object (subclass of `trax.layers.Layer`) created in
        `'predict'` mode and initialized from trained weights. The model
        must have a structure that allows it to run as autoregressive
        one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`).
    inputs: Sequence of symbols the model sees as input the first time it
        generates an output. If None, the model must generate the first output
        with no input to guide it.
    batch_size: Number of sequences to generate in parallel as a batch.
    n_beams: How many beams to consider at the same time.
    start_id: The start symbol (ID/integer) for the autoregressive process,
        or array of shape (`batch_size`, 1) of such integers.
    eos_id: The end-of-sequence symbol (ID/integer) for the autoregressive
        process.
    max_length: Maximum length for generated sequences.
    length_penalty: Factor alpha in calculating the length penalty for beams.
    accelerate: If True, create an accelerated version of `model` and use it
        for generating outputs.

  Returns:
    Tensor of integers with shape (`batch_size`, n_beams, output_length) with
    a batch of output sequences. output_length is the maximum length of the
    output sequences, where each sequence can be no longer than `max_length`.
  """
  del eos_id, length_penalty  # TODO(lukaszkaiser): add length penalty, eos
  assert batch_size == 1, 'Batch size > 1 not supported yet'
  if inputs is not None and inputs.shape[0] != batch_size:
    raise ValueError(f'Inputs batch size ({inputs.shape[0]}) does not match '
                     f'batch_size arg ({batch_size}.')

  fast_model = tl.Accelerate(model) if accelerate else model
  if np.isscalar(start_id):
    start_symbol = np.full((batch_size, 1), start_id, dtype=np.int32)
  else:
    start_symbol = start_id
  if model.n_in == 1 and inputs is not None:
    current_symbols = np.concatenate([start_symbol, inputs], axis=1)
  else:
    current_symbols = start_symbol

  beams = [current_symbols for _ in range(n_beams)]
  results = [([], 0.0) for _ in range(n_beams)]
  states = [fast_model.state for _ in range(n_beams)]
  top_k = [None] * n_beams
  counter = 0
  while counter < max_length:
    counter += 1
    # Run the model on all beams, collect states and top_k for each beam.
    for beam_id in range(n_beams if counter > 1 else 1):
      fast_model.state = states[beam_id]
      if model.n_in > 1 and inputs is not None:
        logits = fast_model((inputs, beams[beam_id]))[0]
      else:
        logits = fast_model(beams[beam_id])
      logits = tl.log_softmax(logits[:, -1, :])
      states[beam_id] = fast_model.state
      top_k[beam_id] = fastmath.top_k(logits, k=n_beams)

    # Select new beams.
    cur_values = []  # will hold triples (sum-of-logprobs, beam-id, symbol)
    for beam_id in range(n_beams if counter > 1 else 1):
      for k in range(n_beams):
        values, symbols = top_k[beam_id]
        value, symbol = values[:, k], symbols[:, k]
        cur_values.append((results[beam_id][1] + value, beam_id, symbol))
    cur_values.sort(key=lambda x: -x[0][0])  # x[0][0] as batch_size=1
    # Collect top beams to the new states and results.
    new_results, new_states, new_beams = [], [], []
    for (value, beam_id, symbol) in cur_values[:n_beams]:
      new_results.append((results[beam_id][0] + [symbol], value))
      new_states.append(states[beam_id])  # copy?
      new_beams.append(symbol[:, None])
    results, states, beams = new_results, new_states, new_beams

  return [(np.stack(r, axis=-1), v) for (r, v) in results]
