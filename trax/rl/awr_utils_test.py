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
"""Tests for trax.rl.awr_utils."""

from absl.testing import absltest
import numpy as onp
from tensor2tensor.envs import trajectory
from trax.rl import awr_utils
from trax.rl.trajectory import replay_buffer


class AwrUtilsTest(absltest.TestCase):

  def get_random_trajectory(self,
                            max_time_step=None,
                            obs_shape=(2, 2)) -> trajectory.Trajectory:
    t = trajectory.Trajectory()
    max_time_step = max_time_step or onp.random.randint(2, 10)
    for i in range(max_time_step):
      r = float(i)
      t.add_time_step(
          observation=onp.random.uniform(size=obs_shape),
          done=False,
          raw_reward=r,
          processed_reward=r,
          action=int(onp.random.choice(10, ())),
          info={
              replay_buffer.ReplayBuffer.LOGPS_KEY_TRAJ:
                  float(onp.random.uniform(low=-10, high=0))
          })
    t.change_last_time_step(done=True)
    return t

  def test_padding(self):
    l = [onp.ones(n) for n in [2, 3, 4]]
    pad_back, pad_back_mask = awr_utils.pad_array_to_length(l, 5)
    onp.testing.assert_array_equal(
        pad_back,
        onp.array([
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
        ]))
    onp.testing.assert_array_equal(pad_back, pad_back_mask)

    pad_front, pad_front_mask = awr_utils.pad_array_to_length(l, 5, False)
    onp.testing.assert_array_equal(
        pad_front,
        onp.array([
            [0., 0., 0., 1., 1.],
            [0., 0., 1., 1., 1.],
            [0., 1., 1., 1., 1.],
        ]))
    onp.testing.assert_array_equal(pad_front, pad_front_mask)

  def test_replay_buffer_to_padded_observations(self):
    traj_lengths = [10, 15, 17]
    obs_shape = (3, 4)
    t_final = 20  # lowest multiple of 10 that is sufficient.
    trajs = [
        self.get_random_trajectory(max_time_step=l, obs_shape=obs_shape)
        for l in traj_lengths
    ]
    rb = replay_buffer.ReplayBuffer(2 * sum(traj_lengths))
    for traj in trajs:
      rb.store(traj)

    padded_obs, mask = awr_utils.replay_buffer_to_padded_observations(
        rb, None, None)
    self.assertEqual((len(traj_lengths), t_final) + obs_shape, padded_obs.shape)
    # pylint: disable=line-too-long
    self.assertTrue(
        all((mask == onp.array([[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0.
        ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 0., 0., 0., 0., 0.
                                ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 1., 1., 0., 0., 0.
                                ]])).flatten()))
    # pylint: enable=line-too-long

    t_final = 6 * 3  # 18 is enough to cover everything.
    padded_obs, mask = awr_utils.replay_buffer_to_padded_observations(
        rb, None, 6)
    self.assertEqual((len(traj_lengths), t_final) + obs_shape, padded_obs.shape)
    # pylint: disable=line-too-long
    self.assertTrue(
        all((mask == onp.array([[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
            0.
        ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 0., 0., 0.
                                ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 1., 1., 0.
                                ]])).flatten()))
    # pylint: enable=line-too-long

  def test_replay_buffer_to_padded_rewards(self):
    traj_lengths = [10, 15, 17]
    obs_shape = (3, 4)
    t_final = 20  # lowest multiple of 10 that is sufficient.
    trajs = [
        self.get_random_trajectory(max_time_step=l, obs_shape=obs_shape)
        for l in traj_lengths
    ]
    rb = replay_buffer.ReplayBuffer(2 * sum(traj_lengths))
    for traj in trajs:
      rb.store(traj)

    idx = rb.get_unrolled_indices()
    padded_rewards, mask = awr_utils.replay_buffer_to_padded_rewards(
        rb, idx, t_final - 1)
    # pylint: disable=line-too-long
    self.assertTrue(
        all((padded_rewards == onp.array([[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.
        ],
                                          [
                                              1., 2., 3., 4., 5., 6., 7., 8.,
                                              9., 10., 11., 12., 13., 14., 0.,
                                              0., 0., 0., 0.
                                          ],
                                          [
                                              1., 2., 3., 4., 5., 6., 7., 8.,
                                              9., 10., 11., 12., 13., 14., 15.,
                                              16., 0., 0., 0.
                                          ]])).flatten()))
    self.assertTrue(
        all((mask == onp.array([[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.
        ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 0., 0., 0., 0., 0.
                                ],
                                [
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 1., 0., 0., 0.
                                ]])).flatten()))
    # pylint: enable=line-too-long

    # import pdb; pdb.set_trace()

  def test_compute_td_lambda_return(self):
    rewards = onp.array([1, 2, 4, 8, 16])
    value_preds = onp.array([1, 2, 4, 8, 16, 32])
    gamma = 0.5
    td_lambda = 0.5
    td_lambda_returns = awr_utils.compute_td_lambda_return(
        rewards, value_preds, gamma, td_lambda)
    onp.testing.assert_array_equal(
        onp.array([2, 5, 11, 20, 32]), td_lambda_returns)

  def test_batched_compute_td_lambda_return(self):
    rewards = onp.array([
        [1, 2, 4, 8, 16, 0, 0],
        [1, 2, 4, 8, 0, 0, 0],
    ])
    rewards_mask = onp.array(rewards > 0).astype(onp.int32)
    value_preds = onp.array([
        [1, 2, 4, 8, 16, 32, 0, 0],
        [1, 2, 4, 8, 16, 0, 0, 0],
    ])
    value_preds_mask = onp.array(value_preds > 0).astype(onp.int32)
    gamma = 0.5
    td_lambda = 0.5
    list_td_lambda_returns = awr_utils.batched_compute_td_lambda_return(
        rewards, rewards_mask, value_preds, value_preds_mask, gamma, td_lambda)
    onp.testing.assert_array_equal(
        onp.array([2, 5, 11, 20, 32]), list_td_lambda_returns[0])
    onp.testing.assert_array_equal(
        onp.array([2, 5, 10, 16]), list_td_lambda_returns[1])

  def test_critic_loss(self):
    pass


if __name__ == '__main__':
  absltest.main()
