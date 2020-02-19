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
"""Tests for trax.rl.awr_trainer."""

import contextlib
import functools
import tempfile
from absl.testing import absltest
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.rl import gym_utils
from tensorflow.compat.v1 import test
from tensorflow.compat.v1.io import gfile
from trax import layers
from trax import optimizers
from trax.rl import awr_trainer


class AwrTrainerTest(absltest.TestCase):

  def get_wrapped_env(self,
                      name='CartPole-v0',
                      max_episode_steps=10,
                      batch_size=2):
    wrapper_fn = functools.partial(
        gym_utils.gym_env_wrapper,
        **{
            'rl_env_max_episode_steps': max_episode_steps,
            'maxskip_env': False,
            'rendered_env': False,
            'rendered_env_resize_to': None,  # Do not resize frames
            'sticky_actions': False,
            'output_dtype': None,
            'num_actions': None,
        })

    return gym_env_problem.GymEnvProblem(
        base_env_name=name,
        batch_size=batch_size,
        env_wrapper_fn=wrapper_fn,
        discrete_rewards=False)

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=test.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  def _make_trainer(self,
                    train_env,
                    eval_env,
                    output_dir,
                    num_samples_to_collect=20,
                    replay_buffer_sample_size=50,
                    model=None,
                    optimizer=None,
                    max_timestep=None,
                    **kwargs):
    if model is None:
      # pylint: disable=g-long-lambda
      model = lambda: layers.Serial(
          layers.Dense(32),
          layers.Relu(),
      )
      # pylint: enable=g-long-lambda

    if optimizer is None:
      optimizer = functools.partial(optimizers.SGD, 5e-5)
    return awr_trainer.AwrTrainer(
        train_env=train_env,
        eval_env=eval_env,
        policy_and_value_model=model,
        policy_and_value_optimizer=optimizer,
        num_samples_to_collect=num_samples_to_collect,
        replay_buffer_sample_size=replay_buffer_sample_size,
        actor_optimization_steps=2,
        critic_optimization_steps=2,
        output_dir=output_dir,
        random_seed=0,
        max_timestep=max_timestep,
        boundary=20,
        actor_loss_weight=1.0,
        entropy_bonus=0.01,
        **kwargs)

  def test_training_loop_cartpole(self):
    with self.tmp_dir() as output_dir:
      trainer = self._make_trainer(
          train_env=self.get_wrapped_env('CartPole-v0', 10),
          eval_env=self.get_wrapped_env('CartPole-v0', 10),
          output_dir=output_dir,
          num_samples_to_collect=20,
          max_timestep=20,
          replay_buffer_sample_size=50,
      )
      trainer.training_loop(n_epochs=2)


if __name__ == '__main__':
  absltest.main()
