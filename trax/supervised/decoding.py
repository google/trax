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

"""Helper functions for decoding using Trax models, esp. autoregressive ones."""

import numpy as np
from trax import layers as tl


def autoregressive_sample_stream(model, inputs=None,
                                 batch_size=1, temperature=1.0,
                                 start_id=0, accelerate=True):
  """Stream aturegressive samples from the provided model.

  Note that the provided model should be an autoregressive model initialized
  in 'predict' mode. In this mode, a model takes the outputs it is generating
  one-by-one (instead of taking them all at once, as, e.g., during training).
  Model state is used to store the intermediate information needed, and usually
  the model perfoms inference in this mode faster than in 'eval' mode.

  Args:
    model: instance of trax.Layer, the model to sample from (at mode='predict')
    inputs: optional tensor [batch_size, M]: inputs to provide to the model;
      for language models (with n_in=1) we use inputs as prefix to the model
    batch_size: how many batches to sample (default: 1)
    temperature: sampling temperature (default: 1.0)
    start_id: int, id for the start symbol fed at the beginning (default: 1)
    accelerate: whether to accelerate the model before decoding (default: True)

  Yields:
    Tensor of ints of shape [batch_size] containing subsequent
    autoregressive samples from the model.
  """
  if inputs is not None and inputs.shape[0] != batch_size:
    raise ValueError(f'Inputs batch size {inputs.shape[0]} != {batch_size}.')
  fast_model = tl.Accelerate(model) if accelerate else model
  cur_symbol = np.full((batch_size, 1), start_id, dtype=np.int32)
  if inputs is not None and model.n_in == 1:  # use inputs as prefix
    cur_symbol = np.concatenate([cur_symbol, inputs], axis=1)
  while True:
    model_input = cur_symbol
    if inputs is not None and model.n_in > 1:
      model_input = (inputs, cur_symbol)
    logits = fast_model(model_input)
    if inputs is not None and model.n_in > 1:
      logits = logits[0]  # Pick first element from model output (a pair here)
    sample = tl.logsoftmax_sample(logits[:, -1, :], temperature=temperature)
    yield sample
    # Note: we're using 'predict' mode autoregressive models here, so history
    # is caches in the model state and we are only feeding one symbol next.
    cur_symbol = sample[:, None]


def autoregressive_sample(model, inputs=None,
                          batch_size=1, temperature=1.0,
                          start_id=0, eos_id=1, max_length=100,
                          accelerate=True):
  """Perform aturegressive sampling from the provided model.

  Note that the provided model should be an autoregressive model initialized
  in 'predict' mode. In this mode, a model takes the outputs it is generating
  one-by-one (instead of taking them all at once, as, e.g., during training).
  Model state is used to store the intermediate information needed, and usually
  the model perfoms inference in this mode faster than in 'eval' mode.

  Args:
    model: instance of trax.Layer, the model to sample from (at mode='predict')
    inputs: optional tensor [batch_size, M]: inputs to provide to the model;
      for language models (with n_in=1) we use inputs as prefix to the model
    batch_size: how many batches to sample (default: 1)
    temperature: sampling temperature (default: 1.0)
    start_id: int, id for the start symbol fed at the beginning (default: 1)
    eos_id: int, id of the end-of-sequence symbol used to stop (default: 1)
    max_length: maximum length to sample (default: 100)
    accelerate: whether to accelerate the model before decoding (default: True)

  Returns:
    a tensor of ints of shape [batch_size, N] with N <= max_length containing
    the autoregressively sampled output from the model
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
