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

import jax

from trax import lr_schedules
from trax import math
from trax import models
from trax import optimizers as opt
from trax import test_utils
from trax.rl import actor_critic_joint
from trax.rl import task as rl_task
# from trax.tf_numpy import numpy as tf_np



class ActorCriticJointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_jointa2ctrainer_atari(self):
    """Test-runs joint A2C on boxing."""

    # tf_np.set_allow_float64(FLAGS.tf_allow_float64)

    env = environments.load_from_settings(
        platform='atari',
        settings={
            'levelName': 'boxing',
            'interleaved_pixels': True,
            'zero_indexed_actions': True
        })
    env = atari_wrapper.AtariWrapper(environment=env, num_stacked_frames=1)

    task = rl_task.RLTask(
        env, initial_trajectories=0, dm_suite=True, max_steps=20)

    body = lambda mode: models.AtariCnnBody()
    joint_model = functools.partial(models.PolicyAndValue, body=body)

    lr = lambda h: lr_schedules.MultifactorSchedule(  # pylint: disable=g-long-lambda
        h, constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')

    trainer = actor_critic_joint.A2CJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.RMSProp,
        lr_schedule=lr,
        batch_size=2,
        train_steps_per_epoch=1,
        collect_per_epoch=1)

    math.disable_jit()
    with jax.disable_jit():
      trainer.run(1)
      trainer.close()
    self.assertEqual(1, trainer.current_epoch)

if __name__ == '__main__':
  absltest.main()
