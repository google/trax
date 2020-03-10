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

from absl.testing import absltest
from trax.rl import task as rl_task


class TaskTest(absltest.TestCase):

  def test_task_sampling(self):
    """Trains a policy on cartpole."""
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      tr1.extend(0, 0, 0, 0)
    tr1.extend(0, 0, 0, 200)
    tr2 = rl_task.Trajectory(101)
    tr2.extend(0, 0, 0, 200)
    task = rl_task.RLTask(
        'CartPole-v0', initial_trajectories=[tr1, tr2])
    stream = task.trajectory_stream(max_slice_length=1)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert len(next_slice) == 1
      slices.append(next_slice.last_observation)
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 1 sampling from 0x100, 101 uniformly.
    assert mean_obs < 31  # Sampling 101 even 3 times is unlikely.
    self.assertLen(slices, 10)


if __name__ == '__main__':
  absltest.main()
