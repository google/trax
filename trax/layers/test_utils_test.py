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
"""Tests for trax.layers.test_utils."""

import functools

from absl.testing import absltest
import numpy as np

from trax.layers import test_utils
from trax.supervised import decoding


def arithmetic_sequence(input_seq, limit=10):
  # Increment the last symbol. Wrap to [0, 10).
  return (input_seq[-1] + 1) % limit


class TestUtilsTest(absltest.TestCase):

  def test_mock_transformer_lm_eval_equals_predict(self):
    model_fn = functools.partial(
        test_utils.MockTransformerLM,
        sequence_fn=arithmetic_sequence,
        vocab_size=10,
    )
    test_utils.test_eval_equals_predict_discrete(model_fn, vocab_size=10)

  def test_mock_transformer_lm_decodes_arithmetic_sequence(self):
    model = test_utils.MockTransformerLM(
        sequence_fn=arithmetic_sequence,
        vocab_size=10,
        mode='predict',
    )
    output = decoding.autoregressive_sample(
        model, max_length=5, start_id=0, eos_id=-1, accelerate=False
    )

    # Sequence including the leading 0 and the last predicted symbol.
    full_seq = list(range(6))
    # decoding.autoregressive_sample doesn't return the leading 0.
    np.testing.assert_array_equal(output, [full_seq[1:]])
    # The prediction buffers don't include the last predicted symbol.
    model.assert_prediction_buffers_equal([full_seq[:-1]])

  def test_mock_transformer_lm_rewinds(self):
    model = test_utils.MockTransformerLM(
        sequence_fn=arithmetic_sequence,
        vocab_size=10,
        mode='predict',
    )
    sample_3 = functools.partial(
        decoding.autoregressive_sample,
        max_length=3,
        eos_id=-1,
        accelerate=False,
    )

    # Generate the 3 initial symbols.
    init_output = sample_3(model, start_id=0)
    np.testing.assert_array_equal(init_output, [[1, 2, 3]])
    state = model.state

    # Generate the next 3 symbols.
    next_output = sample_3(model, start_id=init_output[0, -1])
    np.testing.assert_array_equal(next_output, [[4, 5, 6]])

    # Rewind and generate the last 3 symbols again.
    model.state = state
    next_output = sample_3(model, start_id=init_output[0, -1])
    np.testing.assert_array_equal(next_output, [[4, 5, 6]])

    # Check the buffers.
    model.assert_prediction_buffers_equal([[0, 1, 2, 3, 4, 5]])


if __name__ == '__main__':
  absltest.main()
