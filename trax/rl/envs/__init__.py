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

"""Trax RL environments library."""

import gin
from trax.rl.envs import data_envs


def configure_rl_env(*args, **kwargs):
  kwargs['module'] = 'trax.rl.envs'
  return gin.external_configurable(*args, **kwargs)


copy_stream = configure_rl_env(data_envs.copy_stream)
SequenceDataEnv = configure_rl_env(data_envs.SequenceDataEnv)
