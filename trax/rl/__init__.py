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

"""Trax RL library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from trax.rl import actor_critic
from trax.rl import actor_critic_joint
from trax.rl import simulated_env_problem
from trax.rl import training


def configure_rl(*args, **kwargs):
  kwargs['module'] = 'trax.rl'
  return gin.external_configurable(*args, **kwargs)


def configure_simulated_env_problem(*args, **kwargs):
  kwargs['blacklist'] = [
      'batch_size', 'observation_space', 'action_space', 'reward_range',
      'discrete_rewards', 'history_stream', 'output_dir']
  return configure_rl(*args, **kwargs)


# pylint: disable=invalid-name
RawSimulatedEnvProblem = configure_simulated_env_problem(
    simulated_env_problem.RawSimulatedEnvProblem)
SerializedSequenceSimulatedEnvProblem = configure_simulated_env_problem(
    simulated_env_problem.SerializedSequenceSimulatedEnvProblem)

cartpole_done_fn = configure_rl(simulated_env_problem.cartpole_done_fn)
cartpole_reward_fn = configure_rl(simulated_env_problem.cartpole_reward_fn)
acrobot_done_fn = configure_rl(simulated_env_problem.acrobot_done_fn)
acrobot_reward_fn = configure_rl(simulated_env_problem.acrobot_reward_fn)
onlinetune_done_fn = configure_rl(simulated_env_problem.onlinetune_done_fn)
onlinetune_reward_fn = configure_rl(simulated_env_problem.onlinetune_reward_fn)


PolicyGradientTrainer = configure_rl(
    training.PolicyGradientTrainer, blacklist=['task', 'output_dir'])
ActorCriticTrainer = configure_rl(
    actor_critic.ActorCriticTrainer, blacklist=['task'])
A2CTrainer = configure_rl(
    actor_critic.A2CTrainer,
    blacklist=['task', 'output_dir'])
AWRTrainer = configure_rl(
    actor_critic.AWRTrainer, blacklist=['task', 'output_dir'])
AWRJointTrainer = configure_rl(
    actor_critic_joint.AWRJointTrainer, blacklist=['task', 'output_dir'])
PPOTrainer = configure_rl(
    actor_critic.PPOTrainer, blacklist=['task', 'output_dir'])
PPOJointTrainer = configure_rl(
    actor_critic_joint.PPOJointTrainer, blacklist=['task', 'output_dir'])
