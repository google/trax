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
"""Loop callbacks.

Callbacks can be used to customize the behavior of `supervised.training.Loop`
to accomodate a variety of use-cases.

Examples include:
  - custom evaluation schemes
  - logging metrics to external servers
  - sending model checkpoints to external servers
  - updating the target network in RL algorithms and other non-stationary
    problems
"""


class TrainingStepCallback:
  """Callback triggered before and after a training step."""

  def __init__(self, loop):
    """Initializes the callback with a `supervised.training.Loop` instance."""
    self._loop = loop

  def call_at(self, step):
    """Returns whether the callback should be called at a given step."""
    raise NotImplementedError

  def on_step_begin(self, step):
    """Called by Loop before training steps, when call_at returned True."""
    raise NotImplementedError

  def on_step_end(self, step):
    """Called by Loop after training steps, when call_at returned True."""
    raise NotImplementedError
