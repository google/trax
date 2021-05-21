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
    self._step = 0
    return np.ones((2,))

  def step(self, action):
    del action
    info = {
        'control_mask': self._step % 2 == 0,
        'discount_mask': self._step % 3 == 0,
    }
    self._step += 1
    return np.ones((2,)), 0.0, False, info


class TaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def _extend(
      self, trajectory, action=0, dist_inputs=0, reward=0, done=False,
      new_observation=0,
  ):
    trajectory.extend(
        action=action, dist_inputs=dist_inputs, reward=reward, done=done,
        new_observation=new_observation,
    )

  def test_trajectory_len(self):
    """Test that trajectory length is equal to the number of observations."""
    tr = rl_task.Trajectory(observation=0)
    for _ in range(5):
      self._extend(tr)
    self.assertLen(tr, 6)

  def test_empty_trajectory_last_observation(self):
    """Test that last_observation is the one passed in __init__."""
    tr = rl_task.Trajectory(observation=123)
    self.assertEqual(tr.last_observation, 123)

  def test_nonempty_trajectory_last_observation(self):
    """Test that last_observation is the one passed in the last extend()."""
    tr = rl_task.Trajectory(observation=123)
    for _ in range(5):
      self._extend(tr)
    self._extend(tr, new_observation=321)
    self.assertEqual(tr.last_observation, 321)

  def test_trajectory_done_get_and_set(self):
    """Test that we can get and set the `done` flag of a trajectory."""
    tr = rl_task.Trajectory(observation=123)
    self._extend(tr)
    self.assertFalse(tr.done)
    tr.done = True
    self.assertTrue(tr.done)

  def test_trajectory_suffix_len(self):
    """Test that a trajectory suffix has the correct length."""
    tr = rl_task.Trajectory(observation=0)
    for _ in range(5):
      self._extend(tr)
    tr_suffix = tr.suffix(length=3)
    self.assertLen(tr_suffix, 3)

  def test_trajectory_suffix_observations(self):
    """Test that a trajectory suffix has the correct observations."""
    tr = rl_task.Trajectory(observation=0)
    for obs in range(1, 6):
      self._extend(tr, new_observation=obs)
    tr_suffix = tr.suffix(length=4)
    self.assertEqual([ts.observation for ts in tr_suffix.timesteps], [2, 3, 4])
    self.assertEqual(tr_suffix.last_observation, 5)

  def test_trajectory_to_np_shape(self):
    """Test that the shape of a to_np result matches the trajectory length."""
    tr = rl_task.Trajectory(observation=np.zeros((2, 3)))
    for _ in range(5):
      self._extend(tr, new_observation=np.zeros((2, 3)))
    tr_np = tr.to_np()
    self.assertEqual(tr_np.observation.shape, (len(tr), 2, 3))
    self.assertEqual(tr_np.action.shape, (len(tr),))

  def test_trajectory_to_np_shape_after_extend(self):
    """Test that the shape of a to_np result grows after calling extend()."""
    tr = rl_task.Trajectory(observation=0)
    for _ in range(5):
      self._extend(tr)
    len_before = tr.to_np().observation.shape[0]
    self._extend(tr)
    len_after = tr.to_np().observation.shape[0]
    self.assertEqual(len_after, len_before + 1)

  def test_trajectory_to_np_observations(self):
    """Test that to_np returns correct observations."""
    tr = rl_task.Trajectory(observation=0)
    for obs in range(1, 3):
      self._extend(tr, new_observation=obs)
    tr_np = tr.to_np()
    np.testing.assert_array_equal(tr_np.observation, [0, 1, 2])

  def test_trajectory_to_np_adds_margin(self):
    """Test that to_np adds a specified margin."""
    tr = rl_task.Trajectory(observation=2)
    for _ in range(2):
      self._extend(tr, new_observation=2)
    tr_np = tr.to_np(margin=2)
    np.testing.assert_array_equal(tr_np.observation, [2, 2, 2, 0])
    np.testing.assert_array_equal(tr_np.mask, [1, 1, 0, 0])

  def test_trajectory_to_np_without_margin_cuts_last_observation(self):
    """Test that to_np with margin=0 cuts the last observation."""
    tr = rl_task.Trajectory(observation=0)
    for obs in range(1, 4):
      self._extend(tr, new_observation=obs)
    tr_np = tr.to_np(margin=0)
    np.testing.assert_array_equal(tr_np.observation, [0, 1, 2])

  def test_task_random_initial_trajectories_and_max_steps(self):
    """Test generating initial random trajectories, stop at max steps."""
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=1, max_steps=9)
    stream = task.trajectory_slice_stream(max_slice_length=1)
    next_slice = next(stream)
    self.assertEqual(next_slice.observation.shape, (1, 2))

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
    self._extend(tr1, new_observation=obs, done=True)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_slice_stream(epochs=[-1], max_slice_length=1)
    next_slice = next(stream)
    np.testing.assert_equal(next_slice.observation, np.zeros((1, 2)))
    task.collect_trajectories(policy=(lambda _: (0, 0)), n_trajectories=1)
    next_slice = next(stream)
    np.testing.assert_equal(next_slice.observation, np.ones((1, 2)))

  def test_trajectory_slice_stream_shape(self):
    """Test the shape yielded by trajectory stream."""
    obs = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(obs)
    self._extend(tr1, new_observation=obs, done=True)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_slice_stream(max_slice_length=1)
    next_slice = next(stream)
    self.assertEqual(next_slice.observation.shape, (1, 12, 13))

  def test_trajectory_slice_stream_long_slice(self):
    """Test trajectory stream with slices of longer length."""
    obs = np.zeros((12, 13))
    tr1 = rl_task.Trajectory(obs)
    self._extend(tr1, new_observation=obs)
    self._extend(tr1, new_observation=obs, done=True)
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)
    stream = task.trajectory_slice_stream(max_slice_length=2)
    next_slice = next(stream)
    self.assertEqual(next_slice.observation.shape, (2, 12, 13))

  def test_trajectory_slice_stream_sampling_uniform(self):
    """Test if the trajectory stream samples uniformly."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      self._extend(tr1)
    self._extend(tr1, new_observation=200, done=True)
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    self._extend(tr2, new_observation=200, done=True)
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by slice, not by trajectory.
    stream = task.trajectory_slice_stream(max_slice_length=1)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert next_slice.observation.shape[0] == 1
      slices.append(next_slice.observation[-1])
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 1 sampling from 0x100, 101 uniformly.
    self.assertLess(mean_obs, 31)  # Sampling 101 even 3 times is unlikely.
    self.assertLen(slices, 10)

  def test_trajectory_slice_stream_sampling_by_trajectory(self):
    """Test if the trajectory stream samples by trajectory."""
    # Long trajectory of 0s.
    tr1 = rl_task.Trajectory(0)
    for _ in range(100):
      self._extend(tr1)
    self._extend(tr1, new_observation=200, done=True)
    # Short trajectory of 101.
    tr2 = rl_task.Trajectory(101)
    self._extend(tr2, new_observation=200, done=True)
    task = rl_task.RLTask(
        DummyEnv(), initial_trajectories=[tr1, tr2], max_steps=9)

    # Stream of both. Check that we're sampling by trajectory.
    stream = task.trajectory_slice_stream(
        max_slice_length=1, sample_trajectories_uniformly=True)
    slices = []
    for _ in range(10):
      next_slice = next(stream)
      assert next_slice.observation.shape[0] == 1
      slices.append(next_slice.observation[-1])
    mean_obs = sum(slices) / float(len(slices))
    # Average should be around 50, sampling from {0, 101} uniformly.
    # Sampling 101 < 2 times has low probability (but it possible, flaky test).
    self.assertGreater(mean_obs, 20)
    self.assertLen(slices, 10)

  def test_trajectory_slice_stream_margin(self):
    """Test trajectory stream with an added margin."""
    tr1 = rl_task.Trajectory(0)
    self._extend(tr1, new_observation=1)
    self._extend(tr1, new_observation=1)
    self._extend(
        tr1, new_observation=1, action=1, dist_inputs=2, reward=3, done=True
    )
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=[tr1], max_steps=9)

    # Stream of slices without the final state.
    stream1 = task.trajectory_slice_stream(max_slice_length=4, margin=3)
    got_done = False
    for _ in range(20):
      next_slice = next(stream1)
      self.assertEqual(next_slice.observation.shape, (4,))
      if next_slice.done[0]:
        # In the slice, first we have the last timestep in the actual
        # trajectory, so observation = 1.
        # Then comes the first timestep in the margin, which has the final
        # observation from the trajectory: observation = 1.
        # The remaining timesteps have 0 observations.
        np.testing.assert_array_equal(next_slice.observation, [1, 1, 0, 0])
        # In the margin, done = True and mask = 0.
        for i in range(1, next_slice.observation.shape[0]):
          self.assertTrue(next_slice.done[i])
          self.assertFalse(next_slice.mask[i])
        got_done = True
    # Assert that we got a done somewhere, otherwise the test is not triggered.
    # Not getting done has low probability (1/2^20) but is possible, flaky test.
    self.assertTrue(got_done)

  def test_trajectory_batch_stream_propagates_env_info(self):
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=1, max_steps=4)
    stream = task.trajectory_batch_stream(batch_size=1, max_slice_length=4)
    tr_slice = next(stream)
    # control_mask = step % 2 == 0, discount_mask = step % 3 == 0.
    np.testing.assert_array_equal(
        tr_slice.env_info.control_mask, [[1, 0, 1, 0]]
    )
    np.testing.assert_array_equal(
        tr_slice.env_info.discount_mask, [[1, 0, 0, 1]]
    )

  def test_trajectory_batch_stream_shape(self):
    task = rl_task.RLTask(DummyEnv(), initial_trajectories=1, max_steps=10)
    batch_stream = task.trajectory_batch_stream(
        batch_size=3, min_slice_length=4, max_slice_length=4
    )
    batch = next(batch_stream)
    self.assertEqual(batch.observation.shape, (3, 4, 2))


if __name__ == '__main__':
  absltest.main()
