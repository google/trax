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

# Lint as: python3
"""Tests for RL training."""

import functools

from absl.testing import absltest

from trax import lr_schedules
from trax import models
from trax import optimizers as opt
from trax.models import atari_cnn
from trax.rl import actor_critic
from trax.rl import task as rl_task



class AtariTest(absltest.TestCase):

  def test_boxing(self):
    """Test-runs PPO on Boxing."""
    env = environments.load_from_settings(
        platform='atari',
        settings={
            'levelName': 'boxing',
            'interleaved_pixels': True,
            'zero_indexed_actions': True
        })
    env = atari_wrapper.AtariWrapper(environment=env, num_stacked_frames=1)

    task = rl_task.RLTask(
        env, initial_trajectories=20, dm_suite=True, max_steps=200)

    body = lambda mode: atari_cnn.AtariCnnBody()

    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)

    # pylint: disable=g-long-lambda
    lr_value = lambda h: lr_schedules.MultifactorSchedule(
        h, constant=3e-3, warmup_steps=100, factors='constant * linear_warmup')
    lr_policy = lambda h: lr_schedules.MultifactorSchedule(
        h, constant=1e-3, warmup_steps=100, factors='constant * linear_warmup')
    # pylint: enable=g-long-lambda

    trainer = actor_critic.PPOTrainer(
        task,
        n_shared_layers=0,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr_value,
        value_batch_size=1,
        value_train_steps_per_epoch=1,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr_policy,
        policy_batch_size=1,
        policy_train_steps_per_epoch=1,
        collect_per_epoch=10)
    trainer.run(2)
    # Make sure that we test everywhere at least for 2 epochs, beucase
    # the first epoch is different
    self.assertEqual(2, trainer.current_epoch)


if __name__ == '__main__':
  absltest.main()
