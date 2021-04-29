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
"""Tests for trax.rl.value_tasks."""

from absl.testing import absltest
import numpy as np

from trax import layers as tl
from trax import models
from trax import optimizers as opt
from trax.rl import advantages
from trax.rl import distributions
from trax.rl import policy_tasks
from trax.rl import task as rl_task
from trax.rl import value_tasks
from trax.supervised import lr_schedules
from trax.supervised import training


class ValueTasksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._model_fn = lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
        tl.Dense(64), tl.Relu(), tl.Dense(1)
    )
    self._task = rl_task.RLTask(
        'CartPole-v0', gamma=0.5, max_steps=10, initial_trajectories=100
    )
    self._trajectory_batch_stream = self._task.trajectory_batch_stream(
        batch_size=256, epochs=[-1], max_slice_length=2
    )

  def _value_error(self, value_fn):
    errors = []
    for _ in range(10):
      batch = next(self._trajectory_batch_stream)
      values = value_fn(batch)
      errors.append(np.mean((values - batch.return_) ** 2))
    return np.mean(errors)

  def test_value_tasks_smoke(self):
    # Smoke test for train + eval.
    train_model = self._model_fn(mode='train')
    eval_model = self._model_fn(mode='eval')
    train_task = value_tasks.ValueTrainTask(
        self._trajectory_batch_stream,
        optimizer=opt.Adam(),
        lr_schedule=lr_schedules.constant(1e-3),
        advantage_estimator=advantages.td_k(gamma=self._task.gamma, margin=1),
        model=train_model,
        target_model=eval_model,
    )
    eval_task = value_tasks.ValueEvalTask(train_task)
    loop = training.Loop(
        model=train_model,
        eval_model=eval_model,
        tasks=[train_task],
        eval_tasks=[eval_task],
        eval_at=(lambda _: True),
    )
    loop.run(n_steps=1)

  def test_value_error_high_without_syncs(self):
    train_model = self._model_fn(mode='train')
    eval_model = self._model_fn(mode='eval')
    train_task = value_tasks.ValueTrainTask(
        self._trajectory_batch_stream,
        optimizer=opt.Adam(),
        lr_schedule=lr_schedules.constant(1e-3),
        advantage_estimator=advantages.td_k(gamma=self._task.gamma, margin=1),
        model=train_model,
        target_model=eval_model,
        # Synchronize just once, at the end of training.
        sync_at=(lambda step: step == 100),
    )
    loop = training.Loop(
        model=train_model,
        eval_model=eval_model,
        tasks=[train_task],
    )

    # Assert that before training, the error is high.
    error_before = self._value_error(train_task.value)
    self.assertGreater(error_before, 2.0)

    loop.run(n_steps=100)

    # Assert that after training, the error is smaller, but still high.
    error_after = self._value_error(train_task.value)

    self.assertLess(error_after, 2.0)
    self.assertGreater(error_after, 0.8)

  def test_value_error_low_with_syncs(self):
    min_error = np.inf
    for _ in range(5):
      train_model = self._model_fn(mode='train')
      eval_model = self._model_fn(mode='eval')
      train_task = value_tasks.ValueTrainTask(
          self._trajectory_batch_stream,
          optimizer=opt.Adam(),
          lr_schedule=lr_schedules.constant(1e-3),
          advantage_estimator=advantages.td_k(gamma=self._task.gamma, margin=1),
          model=train_model,
          target_model=eval_model,
          # Synchronize often throughout training.
          sync_at=(lambda step: step % 10 == 0),
      )
      loop = training.Loop(
          model=train_model,
          eval_model=eval_model,
          tasks=[train_task],
      )

      # Assert that before training, the error is high.
      error_before = self._value_error(train_task.value)
      self.assertGreater(error_before, 2.0)

      loop.run(n_steps=100)

      # Assert that after training, the error is small.
      error_after = self._value_error(train_task.value)

      if error_after < 0.8:
        return

      min_error = min(min_error, error_after)

    self.fail(f'Even after 5 trials, min error_after({min_error}) is not < 0.8')

  def test_integration_with_policy_tasks(self):
    # Integration test for policy + value training and eval.
    optimizer = opt.Adam()
    lr_schedule = lr_schedules.constant(1e-3)
    advantage_estimator = advantages.td_k(gamma=self._task.gamma, margin=1)
    policy_dist = distributions.create_distribution(self._task.action_space)
    body = lambda mode: tl.Dense(64)
    train_model = models.PolicyAndValue(policy_dist, body=body)
    eval_model = models.PolicyAndValue(policy_dist, body=body)

    head_selector = tl.Select([1])
    value_train_task = value_tasks.ValueTrainTask(
        self._trajectory_batch_stream,
        optimizer,
        lr_schedule,
        advantage_estimator,
        model=train_model,
        target_model=eval_model,
        head_selector=head_selector,
    )
    value_eval_task = value_tasks.ValueEvalTask(
        value_train_task, head_selector=head_selector
    )

    # Drop the value head - just tl.Select([0]) would pass it, and it would
    # override the targets.
    head_selector = tl.Select([0], n_in=2)
    policy_train_task = policy_tasks.PolicyTrainTask(
        self._trajectory_batch_stream,
        optimizer,
        lr_schedule,
        policy_dist,
        advantage_estimator,
        # Plug a trained critic as our value estimate.
        value_fn=value_train_task.value,
        head_selector=head_selector,
    )
    policy_eval_task = policy_tasks.PolicyEvalTask(
        policy_train_task, head_selector=head_selector
    )

    loop = training.Loop(
        model=train_model,
        eval_model=eval_model,
        tasks=[policy_train_task, value_train_task],
        eval_tasks=[policy_eval_task, value_eval_task],
        eval_at=(lambda _: True),
        # Switch the task every step.
        which_task=(lambda step: step % 2),
    )
    # Run for a couple of steps to make sure there are a few task switches.
    loop.run(n_steps=10)


if __name__ == '__main__':
  absltest.main()
