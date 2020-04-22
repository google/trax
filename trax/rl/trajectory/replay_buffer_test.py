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
"""Tests for trax.rl.trajectory.replay_buffer."""

from absl.testing import absltest
import numpy as np
from tensor2tensor.envs import trajectory
from trax.rl.trajectory import replay_buffer


class ReplayBufferTest(absltest.TestCase):

  def get_random_trajectory(self,
                            max_time_step=None,
                            obs_shape=(2, 2)) -> trajectory.Trajectory:
    t = trajectory.Trajectory()
    max_time_step = max_time_step or np.random.randint(2, 10)
    for _ in range(max_time_step):
      r = float(np.random.uniform(size=()))
      t.add_time_step(
          observation=np.random.uniform(size=obs_shape),
          done=False,
          raw_reward=r,
          processed_reward=r,
          action=int(np.random.choice(10, ())),
          info={
              replay_buffer.ReplayBuffer.LOGPS_KEY_TRAJ:
                  float(np.random.uniform(low=-10, high=0))
          })
    t.change_last_time_step(done=True)
    return t

  def test_add_three_trajectories(self):
    n1 = 10
    t1 = self.get_random_trajectory(max_time_step=n1)

    n2 = 5
    t2 = self.get_random_trajectory(max_time_step=n2)

    # Make a buffer of just sufficient size to hold these two.
    rb = replay_buffer.ReplayBuffer(n1 + n2)

    # import pdb; pdb.set_trace()

    start_index_t1 = rb.store(t1)

    # Stored at the beginning.
    self.assertEqual(0, start_index_t1)
    # One path stored in total.
    self.assertEqual(1, rb.num_paths)
    # Total number of states stored, ever.
    self.assertEqual(n1, rb.total_count)
    # The current number of states stored.
    self.assertEqual(n1, rb.get_current_size())

    start_index_t2 = rb.store(t2)

    # Stored right afterwards
    self.assertEqual(n1, start_index_t2)
    self.assertEqual(2, rb.num_paths)
    self.assertEqual(n1 + n2, rb.total_count)
    self.assertEqual(n1 + n2, rb.get_current_size())

    # We now make a path that is smaller than n1.
    # Since there is no more space in the buffer, t1 will be ejected.
    # t2 will remain there.

    n3 = 6
    assert n3 < n1
    t3 = self.get_random_trajectory(max_time_step=n3)
    start_index_t3 = rb.store(t3)
    self.assertEqual(0, start_index_t3)
    self.assertEqual(2, rb.num_paths)
    self.assertEqual(n1 + n2 + n3, rb.total_count)
    self.assertEqual(n2 + n3, rb.get_current_size())

    # So the first n3 rb.buffers[replay_buffer.ReplayBuffer.PATH_START_KEY] will
    # be 0, the next n1 - n3 will be -1, and the rest will be start_index_t2.
    path_start_array = ([0] * n3) + ([-1] * (n1 - n3)) + ([start_index_t2] * n2)

    np.testing.assert_array_equal(
        path_start_array, rb.buffers[replay_buffer.ReplayBuffer.PATH_START_KEY])

    # The unrolled indices will be first t2s indices, then t3s.
    unrolled_indices = [start_index_t2 + x for x in range(n2)
                       ] + [start_index_t3 + x for x in range(n3)]

    np.testing.assert_array_equal(unrolled_indices, rb.get_unrolled_indices())

    invalid_indices = [start_index_t3 + n3 - 1, start_index_t2 + n2 - 1]

    # Let's sample a really large sample.
    n = 1000
    sample_valid_indices = rb.sample(n, filter_end=True)
    sample_all_indices = rb.sample(n, filter_end=False)

    self.assertNotIn(invalid_indices[0], sample_valid_indices)
    self.assertNotIn(invalid_indices[1], sample_valid_indices)

    # Holds w.h.p.
    self.assertIn(invalid_indices[0], sample_all_indices)
    self.assertIn(invalid_indices[1], sample_all_indices)

  def test_valid_indices(self):
    lens = [10, 5, 7]
    rb = replay_buffer.ReplayBuffer(lens[0] + lens[1])
    trajs = [self.get_random_trajectory(max_time_step=l) for l in lens]
    for traj in trajs:
      rb.store(traj)

    # Now the buffer looks like [traj3 <gap> traj2]
    self.assertLess(rb.buffer_head, rb.buffer_tail)

    idx, valid_mask, valid_idx = rb.get_valid_indices()

    np.testing.assert_array_equal(
        idx, np.array([10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6]))

    np.testing.assert_array_equal(
        valid_idx,
        np.array([[10, 0], [11, 1], [12, 2], [13, 3], [0, 5], [1, 6], [2, 7],
                  [3, 8], [4, 9], [5, 10]],))

    np.testing.assert_array_equal(
        valid_mask,
        np.array([
            True, True, True, True, False, True, True, True, True, True, True,
            False
        ]))

  def test_iteration(self):
    lens = [10, 5, 7]
    rb = replay_buffer.ReplayBuffer(lens[0] + lens[1])
    trajs = [self.get_random_trajectory(max_time_step=l) for l in lens]

    # buffer will have traj0 only.
    rb.store(trajs[0])
    idx = rb.get_unrolled_indices()
    start_end_pairs = [
        (idx[p[0]], idx[p[1] - 1]) for p in rb.iterate_over_paths(idx)
    ]
    self.assertEqual([(0, 9)], start_end_pairs)

    # buffer will have traj0 and traj1.
    rb.store(trajs[1])
    idx = rb.get_unrolled_indices()
    start_end_pairs = [
        (idx[p[0]], idx[p[1] - 1]) for p in rb.iterate_over_paths(idx)
    ]
    self.assertEqual([(0, 9), (10, 14)], start_end_pairs)

    # buffer will have traj1 and traj2, traj0 is booted out.
    rb.store(trajs[2])
    idx = rb.get_unrolled_indices()
    start_end_pairs = [
        (idx[p[0]], idx[p[1] - 1]) for p in rb.iterate_over_paths(idx)
    ]
    self.assertEqual([(10, 14), (0, 6)], start_end_pairs)


if __name__ == "__main__":
  absltest.main()
