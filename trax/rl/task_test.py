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

import os
from absl.testing import absltest
import gym
import numpy as np
from trax import test_utils
from trax.rl import task as rl_task


class DummyEnv(object):
  """Dummy Env class for testing."""

  @property
  def action_space(self):
    return gym.spaces.Discrete(2)

  @property
  def observation_space(self):
    return gym.spaces.Box(-2, 2, shape=(2,))

  def reset(self):
    return np.ones((2,))

  def step(self, action):
    del action
    return np.ones((2,)), 0.0, False, None


class TaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_task_random_initial_trajectories_and_max_steps(self):
    """Test generating initial random trajectories, stop at max steps."""
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=1, max_steps=9)
    stream = task.trajectory_stream(max_slice_length=1)
    next_slice = next(stream)
    self.assertLen(next_slice, 1)
    self.assertEqual(next_slice.last_observation.shape, (2,))

  def test_task_save_init(self):
    """Test saving and re-initialization."""
    task1 = rl_task.RLTask(DummyEnv(), initial_trajectories=13,
                           max_steps=9, gamma=0.9)
    self.assertLen(task1.trajectories[0], 13)
    self.assertEqual(task1.max_steps, 9)
    self.assertEqual(task1.gamma, 0.9)
    temp_file = os.path.join(self.create_tempdir().full_path, 'task.pkl')
    task1.save_to_file(temp_file)
    task2 = rl_task.RLTask(DummyEnv(), initial_trajectories=3,
                           max_steps=19, gamma=1.0)
    self.assertLen(task2.trajectories[0], 3)
    self.assertEqual(task2.max_steps, 19)
    self.assertEqual(task2.gamma, 1.0)
    task2.init_from_file(temp_file)
    self.assertLen(task2.trajectories[0], 13)
    self.assertEqual(task2.max_steps, 9)
    self.assertEqual(task2.gamma, 0.9)

  def test_task_epochs_index_minusone(self):
    """Test that the epoch index -1 means last epoch and updates to it."""
    elem = np.zeros((2,))
    tr1 = rl_task.Trajectory(elem)
    tr1.extend(0, 0, 0, True, elem)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(epochs=[-1], max_slice_length=1)
    next_slice = next(stream)
    self.assertLen(next_slice, 1)
    self.assertEqual(next_slice.last_observation[0], 0)
    task.collect_trajectories((lambda _: (0, 0)), 1)
    next_slice = next(stream)
    self.assertLen(next_slice, 1)
    self.assertEqual(next_slice.last_observation[0], 1)

  def test_trajectory_stream_shape(self):
    """Test the shape yielded by trajectory stream."""
    elem = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(elem)
    tr1.extend(0, 0, 0, True, elem)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(max_slice_length=1)
    next_slice = next(stream)
    self.assertLen(next_slice, 1)
    self.assertEqual(next_slice.last_observation.shape, (12, 13))

  def test_trajectory_stream_long_slice(self):
    """Test trajectory stream with slices of longer length."""
    elem = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(elem)
    tr1.extend(0, 0, 0, False, elem)
    tr1.extend(0, 0, 0, True, elem)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(max_slice_length=2)
    next_slice = next(stream)
    self.assertLen(next_slice, 2)
    self.assertEqual(next_slice.last_observation.shape, (12, 13))

  def test_trajectory_stream_final_state(self):
    """Test trajectory stream with and without the final state."""
    tr1 = rl_task.Trajectory(0)
    tr1.extend(0, 0, 0, True, 1)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)

    # Stream of slices without the final state.
    stream1 = task.trajectory_stream(
        max_slice_length=1, include_final_state=False)
    for _ in range(10):
      next_slice = next(stream1)
      self.assertLen(next_slice, 1)
      self.assertEqual(next_slice.last_observation, 0)

    # Stream of slices with the final state.
    stream2 = task.trajectory_stream(
        max_slice_length=1, include_final_state=True)
    all_sum = 0
    for _ in range(100):
      next_slice = next(stream2)
      self.assertLen(next_slice, 1)
      all_sum += next_slice.last_observation
    self.assertEqual(min(all_sum, 1), 1)  # We've seen the end at least once.

  def test_trajectory_stream_sampling_uniform(self):
    """Test if the trajectory stream samples uniformly."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      tr1.extend(0, 0, 0, False, 0)
    tr1.extend(0, 0, 0, True, 200)
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    tr2.extend(0, 0, 0, True, 200)
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by slice, not by trajectory.
    stream = task.trajectory_stream(max_slice_length=1)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert len(next_slice) == 1
      slices.append(next_slice.last_observation)
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 1 sampling from 0x100, 101 uniformly.
    self.assertLess(mean_obs, 31)  # Sampling 101 even 3 times is unlikely.
    self.assertLen(slices, 10)

  def test_trajectory_stream_sampling_by_trajectory(self):
    """Test if the trajectory stream samples by trajectory."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      tr1.extend(0, 0, 0, False, 0)
    tr1.extend(0, 0, 0, True, 200)
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    tr2.extend(0, 0, 0, True, 200)
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by trajectory.
    stream = task.trajectory_stream(
        max_slice_length=1, sample_trajectories_uniformly=True)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert len(next_slice) == 1
      slices.append(next_slice.last_observation)
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 50, sampling from {0, 101} uniformly.
    # Sampling 101 < 2 times has low probability (but it possible, flaky test).
    self.assertGreater(mean_obs, 20)
    self.assertLen(slices, 10)

  def test_trajectory_stream_margin(self):
    """Test trajectory stream with an added margin."""
    tr1 = rl_task.Trajectory(0)
    tr1.extend(0, 0, 0, False, 1)
    tr1.extend(1, 2, 3, True, 1)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)

    # Stream of slices without the final state.
    stream1 = task.trajectory_stream(
        max_slice_length=3, margin=2, include_final_state=False)
    got_done = False
    for _ in range(10):
      next_slice = next(stream1)
      self.assertLen(next_slice, 3)
      if next_slice.timesteps[0].done:
        for i in range(1, 3):
          self.assertTrue(next_slice.timesteps[i].done)
          self.assertFalse(next_slice.timesteps[i].mask)
        got_done = True
    # Assert that we got a done somewhere, otherwise the test is not trigerred.
    # Not getting done has low probability (1/2^10) but is possible, flaky test.
    self.assertTrue(got_done)


if __name__ == '__main__':
  absltest.main()
