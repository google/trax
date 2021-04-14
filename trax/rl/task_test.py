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

# Lint as: python3
"""Tests for RL training."""

import os
from absl.testing import absltest
import gym
import numpy as np
from trax import test_utils
from trax.rl import task as rl_task


class DummyEnv:
  """Dummy Env class for testing."""

  observation_space = gym.spaces.Box(-2, 2, shape=(2,))
  action_space = gym.spaces.Discrete(2)

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
    self.assertEqual(next_slice.observations.shape, (1, 2))

  def test_time_limit_terminates_epsiodes(self):
    """Test that episodes are terminated upon reaching `time_limit` steps."""
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=3, max_steps=10, time_limit=10
    )
    trajectories = task.trajectories[0]  # Get trajectories from epoch 0.
    self.assertLen(trajectories, 3)
    for trajectory in trajectories:
      self.assertTrue(trajectory.done)
      # max_steps + 1 (the initial observation doesn't count).
      self.assertLen(trajectory, 11)

  def test_max_steps_doesnt_terminate_epsiodes(self):
    """Test that episodes are not terminated upon reaching `max_steps` steps."""
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=2, max_steps=5, time_limit=10
    )
    trajectories = task.trajectories[0]  # Get trajectories from epoch 0.
    self.assertLen(trajectories, 2)
    # The trajectory should be cut in half. The first half should not be "done".
    self.assertFalse(trajectories[0].done)
    self.assertLen(trajectories[0], 6)  # max_steps + 1
    # The second half should be "done".
    self.assertTrue(trajectories[1].done)
    self.assertLen(trajectories[1], 6)  # max_steps + 1

  def test_collects_specified_number_of_interactions(self):
    """Test that the specified number of interactions are collected."""
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=0, max_steps=3, time_limit=20
    )
    task.collect_trajectories(policy=(lambda _: (0, 0)), n_interactions=10)
    trajectories = task.trajectories[1]  # Get trajectories from epoch 1.
    n_interactions = 0
    for trajectory in trajectories:
      n_interactions += len(trajectory) - 1
    self.assertEqual(n_interactions, 10)

  def test_collects_specified_number_of_trajectories(self):
    """Test that the specified number of interactions are collected."""
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=0, max_steps=3, time_limit=20
    )
    task.collect_trajectories(policy=(lambda _: (0, 0)), n_trajectories=3)
    trajectories = task.trajectories[1]  # Get trajectories from epoch 1.
    self.assertLen(trajectories, 3)

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
    obs = np.zeros((2,))
    tr1 = rl_task.Trajectory(obs)
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=obs
    )
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(epochs=[-1], max_slice_length=1)
    next_slice = next(stream)
    np.testing.assert_equal(next_slice.observations, np.zeros((1, 2)))
    task.collect_trajectories(policy=(lambda _: (0, 0)), n_trajectories=1)
    next_slice = next(stream)
    np.testing.assert_equal(next_slice.observations, np.ones((1, 2)))

  def test_trajectory_stream_shape(self):
    """Test the shape yielded by trajectory stream."""
    obs = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(obs)
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=obs
    )
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(max_slice_length=1)
    next_slice = next(stream)
    self.assertEqual(next_slice.observations.shape, (1, 12, 13))

  def test_trajectory_stream_long_slice(self):
    """Test trajectory stream with slices of longer length."""
    obs = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(obs)
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=False, new_observation=obs
    )
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=obs
    )
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_stream(max_slice_length=2)
    next_slice = next(stream)
    self.assertEqual(next_slice.observations.shape, (2, 12, 13))

  def test_trajectory_stream_sampling_uniform(self):
    """Test if the trajectory stream samples uniformly."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      tr1.extend(
          action=0, dist_inputs=0, reward=0, done=False, new_observation=0
      )
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=200
    )
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    tr2.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=200
    )
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by slice, not by trajectory.
    stream = task.trajectory_stream(max_slice_length=1)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert next_slice.observations.shape[0] == 1
      slices.append(next_slice.observations[-1])
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 1 sampling from 0x100, 101 uniformly.
    self.assertLess(mean_obs, 31)  # Sampling 101 even 3 times is unlikely.
    self.assertLen(slices, 10)

  def test_trajectory_stream_sampling_by_trajectory(self):
    """Test if the trajectory stream samples by trajectory."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      tr1.extend(
          action=0, dist_inputs=0, reward=0, done=False, new_observation=0
      )
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=200
    )
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    tr2.extend(
        action=0, dist_inputs=0, reward=0, done=True, new_observation=200
    )
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by trajectory.
    stream = task.trajectory_stream(
        max_slice_length=1, sample_trajectories_uniformly=True)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert next_slice.observations.shape[0] == 1
      slices.append(next_slice.observations[-1])
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 50, sampling from {0, 101} uniformly.
    # Sampling 101 < 2 times has low probability (but it possible, flaky test).
    self.assertGreater(mean_obs, 20)
    self.assertLen(slices, 10)

  def test_trajectory_stream_margin(self):
    """Test trajectory stream with an added margin."""
    tr1 = rl_task.Trajectory(0)
    # TODO(pkozakowski): Add an util for extending trajectories with dummy
    # timesteps.
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=False, new_observation=1
    )
    tr1.extend(
        action=0, dist_inputs=0, reward=0, done=False, new_observation=1
    )
    tr1.extend(
        action=1, dist_inputs=2, reward=3, done=True, new_observation=1
    )
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)

    # Stream of slices without the final state.
    stream1 = task.trajectory_stream(max_slice_length=4, margin=3)
    got_done = False
    for _ in range(20):
      next_slice = next(stream1)
      self.assertEqual(next_slice.observations.shape, (4,))
      if next_slice.dones[0]:
        # In the slice, first we have the last timestep in the actual
        # trajectory, so observation = 1.
        # Then comes the first timestep in the margin, which has the final
        # observation from the trajectory: observation = 1.
        # The remaining timesteps have 0 observations.
        np.testing.assert_array_equal(next_slice.observations, [1, 1, 0, 0])
        # In the margin, done = True and mask = 0.
        for i in range(1, next_slice.observations.shape[0]):
          self.assertTrue(next_slice.dones[i])
          self.assertFalse(next_slice.mask[i])
        got_done = True
    # Assert that we got a done somewhere, otherwise the test is not triggered.
    # Not getting done has low probability (1/2^20) but is possible, flaky test.
    self.assertTrue(got_done)

  def test_trajectory_batch_stream_shape(self):
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=1, max_steps=10)
    batch_stream = task.trajectory_batch_stream(
        batch_size=3, min_slice_length=4, max_slice_length=4
    )
    batch = next(batch_stream)
    self.assertEqual(batch.observations.shape, (3, 4, 2))


if __name__ == '__main__':
  absltest.main()
