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

"""Utility functions for testing.
"""
import functools
import numpy as np

from trax import fastmath
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
                             message=''):
  """Utility method for testing equivalence of predict and eval modes.

  Args:
    inp: input fed to the model. It can be a tensor, or a tuple of tensors.
    model_fn: function creating a model after calling with `mode` argument.
    seq_axis: axis of sequence_length. In predict mode we iterate over this
      axis. By default `1`, which is 2nd dimension.
    seq_tensor: if `inp` is a tuple, `seq_tensor` is an index of an input tensor
      in this tuple on which we iterate the sequence.
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

    for index in range(length):
      if seq_tensor is None:
        new_inp = inp.take(indices=range(index, index+1), axis=seq_axis)
      else:
        new_inp = list(inp)
        new_inp[seq_tensor] = new_inp[seq_tensor].take(
            indices=range(index, index+1), axis=seq_axis)

      output_predict = model_predict(new_inp, rng=rng)
      if not isinstance(output_predict, (tuple, list)):
        # We will automatically check each and every tensor returned.
        output_predict = [output_predict]

      np.testing.assert_equal(len(output_predict), len(output_eval))
      for outp, oute in zip(output_predict, output_eval):
        np.testing.assert_array_almost_equal(
            oute.take(indices=index, axis=seq_axis),
            outp.take(indices=0, axis=seq_axis),
            decimal=5,
            err_msg='Error on element {} out of {}.{}'.format(index, length,
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
