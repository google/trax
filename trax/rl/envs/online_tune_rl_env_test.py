# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Tests for trax.rl.online_tune_rl_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow import test
from tensorflow.compat.v1.io import gfile
from trax import models
from trax.rl import ppo_trainer
from trax.rl.envs import online_tune_rl_env


class OnlineTuneRLTest(test.TestCase):

  @staticmethod
  def _create_env(output_dir):
    return online_tune_rl_env.OnlineTuneRLEnv(
        trainer_class=functools.partial(
            ppo_trainer.PPO,
            policy_and_value_model=functools.partial(
                models.FrameStackMLP, hidden_sizes=()
            ),
            max_timestep=1,
            max_timestep_eval=1,
        ),
        env_name='CartPole-v0',
        env_kwargs={'max_timestep': 1, 'resize': False},
        train_batch_size=1,
        eval_batch_size=1,
        train_epochs=1,
        output_dir=output_dir,
    )

  def test_runs(self):
    env = self._create_env(output_dir=self.get_temp_dir())
    obs = env.reset()
    self.assertEqual(obs.shape, env.observation_space.shape)
    (obs, _, _, _) = env.step(env.action_space.sample())
    self.assertEqual(obs.shape, env.observation_space.shape)

  def test_creates_new_trajectory_dirs(self):
    output_dir = self.get_temp_dir()
    env = self._create_env(output_dir=output_dir)
    self.assertEqual(set(gfile.listdir(output_dir)), set())
    env.reset()
    self.assertEqual(set(gfile.listdir(output_dir)), {'0'})
    env.reset()
    self.assertEqual(set(gfile.listdir(output_dir)), {'0', '1'})


if __name__ == '__main__':
  test.main()
