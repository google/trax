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

"""Tests for trax.supervised.callbacks."""

import functools
import io
from unittest import mock

from absl.testing import absltest
import gym
import numpy as np

from trax import models
from trax.data import inputs
from trax.rl import serialization_utils
from trax.rl import space_serializer
from trax.supervised import callbacks
from trax.supervised import lr_schedules
from trax.supervised import trainer_lib


def dummy_inputs(seq_len, batch_size):
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


class CallbacksTest(absltest.TestCase):

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

    def model(mode):
      return serialization_utils.SerializedModel(
          inner_model(mode),
          observation_serializer=srl,
          action_serializer=srl,
          significance_decay=0.7,
      )

    eval_callback = functools.partial(
        callbacks.SerializedModelEvaluation,
        model=inner_model('predict'),
        observation_serializer=srl,
        action_serializer=srl,
        eval_at=5,
    )

    output_dir = self.create_tempdir().full_path
    trainer_lib.train(
        output_dir=output_dir,
        model=model,
        inputs=functools.partial(dummy_inputs, seq_len=4, batch_size=64),
        lr_schedule_fn=functools.partial(lr_schedules.constant, 0.01),
        callbacks=[eval_callback],
        steps=10,
    )
    self.assertTrue(_has_metric('pred_error', mock_stdout))


def _has_metric(metric_name, stdout):
  log = stdout.getvalue()
  metric_logs = [line for line in log.split('\n') if metric_name in line]
  return bool(metric_logs)


if __name__ == '__main__':
  absltest.main()
