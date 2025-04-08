# coding=utf-8
# Copyright 2022 The Trax Authors.
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

import gin

from learning.reinforcement import (
    actor_critic,
    actor_critic_joint,
    serialization_utils,
    training,
)


def configure_rl(*args, **kwargs):
    kwargs["module"] = "trax.reinforcement"
    kwargs["denylist"] = ["task", "output_dir"]
    return gin.external_configurable(*args, **kwargs)


gin.enter_interactive_mode()


@gin.configurable(module="trax.reinforcement")
def every(n_steps):
    """Returns True every n_steps, for use as *_at functions in various places."""
    return lambda step: step % n_steps == 0


A2C = configure_rl(actor_critic.A2C)
AWR = configure_rl(actor_critic.AWR)
LoopAWR = configure_rl(actor_critic.LoopAWR)
PPO = configure_rl(actor_critic.PPO)
SamplingAWR = configure_rl(actor_critic.SamplingAWR)

A2CJoint = configure_rl(actor_critic_joint.A2CJoint)
AWRJoint = configure_rl(actor_critic_joint.AWRJoint)
PPOJoint = configure_rl(actor_critic_joint.PPOJoint)

PolicyGradient = configure_rl(training.PolicyGradient)
ExpertIteration = configure_rl(training.ExpertIteration)
DQN = configure_rl(training.DQN)

TimeSeriesModel = gin.external_configurable(
    serialization_utils.TimeSeriesModel, module="trax.reinforcement"
)
