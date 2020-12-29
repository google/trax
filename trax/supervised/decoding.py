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

"""Decoding with Trax models."""

import numpy as np
from trax import layers as tl


def autoregressive_sample_stream(model, inputs=None,
                                 batch_size=1, temperature=1.0,
                                 start_id=0, accelerate=True):
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
        one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`).
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

  while True:
    if model.n_in > 1 and inputs is not None:
      logits = fast_model((inputs, current_symbols))[0]
    else:
      logits = fast_model(current_symbols)
    sample = tl.logsoftmax_sample(logits[:, -1, :], temperature=temperature)
    yield sample
    # NOTE: Because the model is autoregressive and in 'predict' mode, its
    # history is cached in the model state and the next input is the single
    # symbol just sampled.
    current_symbols = sample[:, None]


def autoregressive_sample(model, inputs=None,
                          batch_size=1, temperature=1.0,
                          start_id=0, eos_id=1, max_length=100,
                          accelerate=True):
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
        one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`).
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
      start_id=start_id, accelerate=accelerate):
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
