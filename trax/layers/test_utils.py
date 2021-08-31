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

"""Utility functions for testing."""

import copy
import functools

import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes


def test_eval_is_deterministic(inp, model_fn, message=''):
  """Utility method for testing if eval mode is deterministic.

  Args:
    inp: input fed to the model. It can be a tensor, or a tuple of tensors.
    model_fn: function creating a model after calling with `mode` argument.
    message: Optional message to show when outputs of eval/predict mode don't
      match.
  """
  with fastmath.use_backend(fastmath.Backend.JAX):
    model_eval1 = model_fn(mode='eval')
    model_eval2 = model_fn(mode='eval')

    input_signature = shapes.signature(inp)
    model_eval1.init(input_signature)
    model_eval2.init(input_signature)
    model_eval1.save_to_file('/tmp/unique_weights')
    model_eval2.init_from_file('/tmp/unique_weights', weights_only=True,
                               input_signature=input_signature)

    rng = fastmath.random.get_prng(0)
    output_eval1 = model_eval1(inp, rng=rng)
    if not isinstance(output_eval1, (tuple, list)):
      # We will automatically check each and every tensor returned.
      output_eval1 = [output_eval1]

    output_eval2 = model_eval2(inp, rng=rng)
    if not isinstance(output_eval2, (tuple, list)):
      # We will automatically check each and every tensor returned.
      output_eval2 = [output_eval2]

    np.testing.assert_equal(len(output_eval1), len(output_eval2))
    for out1, out2 in zip(output_eval1, output_eval2):
      np.testing.assert_array_almost_equal(
          out1,
          out2,
          decimal=5,
          err_msg='Non-deterministic.{}'.format(message))


def test_eval_equals_predict(inp, model_fn, seq_axis=1, seq_tensor=None,
                             init_tokens=3, message=''):
  """Utility method for testing equivalence of predict and eval modes.

  Args:
    inp: input fed to the model. It can be a tensor, or a tuple of tensors.
    model_fn: function creating a model after calling with `mode` argument.
    seq_axis: axis of sequence_length. In predict mode we iterate over this
      axis. By default `1`, which is 2nd dimension.
    seq_tensor: if `inp` is a tuple, `seq_tensor` is an index of an input tensor
      in this tuple on which we iterate the sequence.
    init_tokens: how many tokens should be passed to the first `predict` call.
    message: Optional message to show when outputs of eval/predict mode don't
      match.
  """
  with fastmath.use_backend(fastmath.Backend.JAX):
    model_eval = model_fn(mode='eval')
    model_predict = model_fn(mode='predict')

    input_signature = shapes.signature(inp)
    model_eval.init(input_signature)
    model_predict.init(input_signature)
    model_eval.save_to_file('/tmp/unique_weights')
    model_predict.init_from_file('/tmp/unique_weights', weights_only=True,
                                 input_signature=input_signature)

    rng = fastmath.random.get_prng(0)
    output_eval = model_eval(inp, rng=rng)
    if not isinstance(output_eval, (tuple, list)):
      # We will automatically check each and every tensor returned.
      output_eval = [output_eval]

    if seq_tensor is None:
      length = inp.shape[seq_axis]
    else:
      length = inp[seq_tensor].shape[seq_axis]

    assert length >= init_tokens + 2  # Required to properly test predict mode.
    indices_list = [(0, init_tokens)] + [(i, i+1)
                                         for i in range(init_tokens, length)]

    for indices in indices_list:
      start, end = indices
      if seq_tensor is None:
        new_inp = inp.take(indices=np.arange(start, end), axis=seq_axis)
      else:
        new_inp = list(inp)
        new_inp[seq_tensor] = new_inp[seq_tensor].take(
            indices=np.arange(start, end), axis=seq_axis)

      output_predict = model_predict(new_inp, rng=rng)
      if not isinstance(output_predict, (tuple, list)):
        # We will automatically check each and every tensor returned.
        output_predict = [output_predict]

      np.testing.assert_equal(len(output_predict), len(output_eval))
      for outp, oute in zip(output_predict, output_eval):
        np.testing.assert_array_almost_equal(
            oute.take(indices=np.arange(start, end), axis=seq_axis),
            outp.take(indices=np.arange(0, end-start), axis=seq_axis),
            decimal=5,
            err_msg='Error on element {} out of {}.{}'.format(indices, length,
                                                              message))


def test_eval_equals_predict_configs(inp, model_fn, configs, seq_axis=1,
                                     seq_tensor=None, message=''):
  """Utility method for testing equivalence of predict and eval modes.

  This function iterates over a list of dictionaries `confis`, and runs the test
  on models with each configuration.

  Args:
    inp: input fed to the model. It can be a tensor, or a tuple of tensors.
    model_fn: function creating a model after calling with `mode` argument.
    configs: List of dictionaries, which contain configs to be fed into
      `model_fn`.
    seq_axis: axis of sequence_length. In predict mode we iterate over this
      axis. By default `1`, which is 2nd dimension.
    seq_tensor: if `inp` is a tuple, `seq_tensor` is an index of an input tensor
      in this tuple on which we iterate the sequence.
    message: Optional message to show when outputs of eval/predict mode don't
      match.
  """
  for config in configs:
    model_fn_configured = functools.partial(model_fn, **config)
    test_eval_equals_predict(inp, model_fn_configured, seq_axis=seq_axis,
                             seq_tensor=seq_tensor,
                             message=' Config: {}.{}'.format(config, message))


def test_eval_equals_predict_discrete(
    model_fn, vocab_size=10, length=5, batch_size=3
):
  """Tests the equivalence of eval and predict modes for discrete models."""
  with fastmath.use_backend(fastmath.Backend.JAX):
    model_slow = model_fn(mode='eval', vocab_size=vocab_size)
    model_fast = model_fn(mode='predict', vocab_size=vocab_size)
    rng = fastmath.random.get_prng(0)
    input_signature = shapes.ShapeDtype((batch_size, 1), np.int32)
    # Given the same rng, both models initialize with the same parameters.
    model_slow.init(input_signature, rng)
    model_fast.init(input_signature, rng)

    buf = np.zeros((batch_size, length), dtype=np.int32)
    next_sym = np.zeros((batch_size, 1), dtype=np.int32)

    for index in range(length):
      logits_slow = model_slow(buf, rng=rng)
      logits_fast = model_fast(next_sym, rng=rng)
      np.testing.assert_array_almost_equal(
          logits_slow[:, index, :], logits_fast[:, 0, :],
          decimal=5,
      )
      next_sym = np.random.randint(vocab_size, size=(batch_size, 1))
      buf[:, index] = next_sym[:, 0]


class MockTransformerLM(tl.Layer):
  r"""Mock TransformerLM for testing autoregressive sampling routines.

  Mimics the behavior of a perfectly-trained, deterministic TransformerLM.
  Allows to specify the \sigma^* -> \sigma function implemented by the model
  and to make assertions about the input sequence passed to the model.

  Supports two modes: stateful "predict" for fast inference, and stateless
  non-"predict" ("train", "eval" etc).

  Useful for testing any logic that relies on autoregressive sampling, as it
  removes the additional layer of complexity related to training a model or
  maintaining a pretrained one. Makes the tests run MUCH faster.

  Does not support acceleration. Do not wrap in tl.Accelerate().
  """

  def __init__(self, sequence_fn, mode, vocab_size):
    super().__init__()

    self._sequence_fn = sequence_fn
    self._mode = mode
    self._vocab_size = vocab_size

    self._prediction_buffers = None

  @property
  def state(self):
    return copy.deepcopy(self._prediction_buffers)

  @state.setter
  def state(self, state):
    self._prediction_buffers = copy.deepcopy(state)

  def _output_symbol_predict(self, input_symbols, prediction_buffer):
    prediction_buffer.extend(input_symbols)
    output_symbol = self._sequence_fn(np.array(prediction_buffer))
    return np.array([output_symbol])

  def _output_symbols_eval(self, input_symbols, prediction_buffer):
    del prediction_buffer

    # Add a leading 0 token to imitate ShiftRight.
    input_symbols = np.concatenate(([0], input_symbols))

    # Call sequence_fn repeatedly along the input sequence.
    return np.array([
        self._sequence_fn(input_symbols[:end])
        for end in range(1, len(input_symbols))
    ])

  def _symbols_to_logits(self, symbols):
    # Assert that symbols are discrete.
    assert np.issubdtype(symbols.dtype, np.integer)
    # Assert that 0 <= symbols < vocab_size.
    np.testing.assert_array_less(-1, symbols)
    np.testing.assert_array_less(symbols, self._vocab_size)

    # Return almost-determinisitc logits:
    # e^1000 / (e^1000 + vocab_size) ~= 1
    return tl.one_hot(symbols, n_categories=self._vocab_size) * 1000.0

  def __call__(self, inputs, rng=None):
    del rng

    assert inputs.ndim == 2, (
        'The input sequences should have exactly two axes.'
    )

    if self._prediction_buffers is None:
      # Initialize the buffer.
      batch_size = inputs.shape[0]
      # [[]] * batch_size would create multiple references to the same
      # list, and we want separate lists.
      self._prediction_buffers = [[] for _ in range(batch_size)]

    if self._mode == 'predict':
      output_fn = self._output_symbol_predict
    else:
      output_fn = self._output_symbols_eval

    # Calculate the output separately for each sequence in the batch.
    output_symbols = np.array([
        output_fn(input_seq, pred_buffer)
        for (input_seq, pred_buffer) in zip(
            inputs, self._prediction_buffers
        )
    ])
    return self._symbols_to_logits(output_symbols)

  def assert_prediction_buffers_equal(self, expected_buffers):
    if self._prediction_buffers is None:
      batch_size = expected_buffers.shape[0]
      actual_buffers = np.empty((batch_size, 0))
    else:
      actual_buffers = np.array(self._prediction_buffers)

    np.testing.assert_array_equal(actual_buffers, expected_buffers)
