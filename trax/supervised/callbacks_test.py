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

"""Tests for trax.supervised.callbacks."""

import functools
import io
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import gym
import numpy as np

from trax import models
from trax import test_utils
from trax.data import inputs
from trax.layers import test_utils as tl_test_utils
from trax.rl import serialization_utils
from trax.rl import space_serializer
from trax.supervised import callbacks
from trax.supervised import lr_schedules
from trax.supervised import trainer_lib
from trax.supervised import training


def random_inputs(seq_len, batch_size):
  def stream_fn(num_devices):
    del num_devices
    while True:
      x = np.random.uniform(size=(batch_size, seq_len))
      y = np.random.uniform(size=(batch_size, seq_len))
      mask = np.ones_like(x).astype(np.float32)
      yield (x, y, x, mask)

  return inputs.Inputs(
      train_stream=stream_fn,
      eval_stream=stream_fn,
  )


def make_multibonacci_modulo(history_length, limit):
  """Creates a function that generates the Multibonacci sequence modulo n."""
  def sequence_fn(seq):
    return np.sum(seq[-history_length:]) % limit
  return sequence_fn


def generate_trajectory(sequence_fn, space, n_steps):
  """Generates random actions and observations that follow sequence_fn."""
  act = [space.sample() for _ in range(n_steps)]
  obs = [space.sample()]

  for (o, a) in zip(
      obs,
      act[:-1],  # Don't generate the last observation.
  ):
    context = list(np.array([o, a]).flatten())
    symbols = []
    for _ in range(np.array(o).size):
      symbol = sequence_fn(context + symbols)
      symbols.append(symbol)
    obs.append(np.reshape(symbols, space.shape))

  obs = np.array([obs])
  act = np.array([act])
  return (obs, act)


def make_singleton_eval_task(observations, actions):
  """Creates an EvalTask with just one example."""
  mask = np.ones(observations.shape[:2])
  def data():
    while True:
      yield (observations, actions, observations, mask)

  return training.EvalTask(
      labeled_data=data(),
      metrics=[],
  )


def make_serialized_model(seq_model, space, vocab_size):
  srl = space_serializer.create(space, vocab_size)
  return serialization_utils.SerializedModel(
      functools.partial(seq_model, vocab_size=vocab_size),
      observation_serializer=srl,
      action_serializer=srl,
      significance_decay=0.7,
  )


class CallbacksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_serialized_model_evaluation(self, mock_stdout):
    precision = 1
    vocab_size = 2
    srl = space_serializer.BoxSpaceSerializer(
        space=gym.spaces.Box(shape=(), low=0.0, high=1.0),
        vocab_size=vocab_size,
        precision=precision,
    )

    def inner_model(mode):
      return models.TransformerLM(
          mode=mode,
          vocab_size=vocab_size,
          d_model=2,
          d_ff=4,
          n_layers=1,
          n_heads=1,
      )

    serialized_model_fn = functools.partial(
        serialization_utils.SerializedModel,
        inner_model,
        observation_serializer=srl,
        action_serializer=srl,
        significance_decay=0.7,
    )
    eval_callback = functools.partial(
        callbacks.SerializedModelEvaluation, eval_at=5
    )

    output_dir = self.create_tempdir().full_path
    trainer_lib.train(
        output_dir=output_dir,
        model=serialized_model_fn,
        inputs=functools.partial(random_inputs, seq_len=4, batch_size=64),
        lr_schedule_fn=functools.partial(lr_schedules.constant, 0.01),
        callbacks=[eval_callback],
        steps=10,
    )
    self.assertTrue(_has_metric('pred_error', mock_stdout))

  @parameterized.product(
      context_lengths=((2,), (1, 3)),
      horizon_lengths=((1,), (1, 2)),
  )
  def test_srl_eval_feeds_correct_sequence(
      self, context_lengths, horizon_lengths
  ):
    vocab_size = 10
    n_steps = 5

    multibonacci_modulo = make_multibonacci_modulo(2, vocab_size)
    space = gym.spaces.Discrete(n=vocab_size)
    (obs, act) = generate_trajectory(multibonacci_modulo, space, n_steps)
    eval_task = make_singleton_eval_task(obs, act)
    seq_model = functools.partial(
        tl_test_utils.MockTransformerLM,
        sequence_fn=multibonacci_modulo,
    )
    serialized_model = make_serialized_model(seq_model, space, vocab_size)
    callback = callbacks.SerializedModelEvaluation(
        loop=None,
        eval_task=eval_task,
        model=serialized_model,
        context_lengths=context_lengths,
        horizon_lengths=horizon_lengths,
        accelerate_model=False,
    )
    callback.evaluate(weights=None)

    expected_seq = np.zeros(2 * n_steps + 1)
    expected_seq[1::2] = obs
    expected_seq[2::2] = act
    seen_len = (context_lengths[-1] + horizon_lengths[-1]) * 2
    callback.predict_model.assert_prediction_buffers_equal(
        [expected_seq[:seen_len]]
    )

  @parameterized.named_parameters(('one_symbol', 1), ('two_symbols', 2))
  def test_srl_eval_reports_zero_error_for_perfect_model(self, precision):
    vocab_size = 100
    n_steps = 5

    multibonacci_modulo = make_multibonacci_modulo(2 * precision, vocab_size)
    space = gym.spaces.MultiDiscrete(nvec=([vocab_size] * precision))
    (obs, act) = generate_trajectory(multibonacci_modulo, space, n_steps)
    eval_task = make_singleton_eval_task(obs, act)
    seq_model = functools.partial(
        tl_test_utils.MockTransformerLM,
        sequence_fn=multibonacci_modulo,
    )
    serialized_model = make_serialized_model(seq_model, space, vocab_size)
    callback = callbacks.SerializedModelEvaluation(
        loop=None,
        eval_task=eval_task,
        model=serialized_model,
        context_lengths=(1,),
        horizon_lengths=(4,),
        accelerate_model=False,
    )
    metrics = callback.evaluate(weights=None)
    error = next(
        value for (name, value) in metrics.items() if 'pred_error' in name
    )
    assert error == 0


def _has_metric(metric_name, stdout):
  log = stdout.getvalue()
  metric_logs = [line for line in log.split('\n') if metric_name in line]
  return bool(metric_logs)


if __name__ == '__main__':
  absltest.main()
