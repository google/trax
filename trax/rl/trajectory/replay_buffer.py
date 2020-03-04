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
"""Replay buffer, focusing on sampling states."""

from typing import List

import numpy as np
from tensor2tensor.envs import trajectory


class ReplayBuffer:
  """A replay buffer with state (not trajectory) oriented focus."""

  TERMINATE_KEY = 'terminate'
  PATH_START_KEY = 'path_start'
  PATH_END_KEY = 'path_end'

  OBSERVATIONS_KEY = 'observations'
  ACTIONS_KEY = 'actions'
  # TODO(afrozm): Maybe rename to LOGPS_KEY_TRAJ?
  LOGPS_KEY = 'logps'
  REWARDS_KEY = 'rewards'

  LOGPS_KEY_TRAJ = 'log_prob_actions'
  INVALID_IDX = -1

  def __init__(self, buffer_size: int):
    # TODO(afrozm): Rename to max_num_states or something better.
    assert buffer_size > 0
    self.buffer_size = buffer_size
    self.total_count = 0
    self.buffer_head = 0
    self.buffer_tail = self.INVALID_IDX
    self.num_paths = 0

    # TODO(afrozm): Flatten this dictionary, too confusing now.
    self.buffers = None

    self.clear()
    return

  def sample(self, n, filter_end=True):
    """Sample `n` observations from the replay buffer."""
    curr_size = self.get_current_size()
    assert curr_size > 0

    # Select the indices to sample from.
    if filter_end:
      idxs_to_sample_from = self.get_valid_idx()
    else:
      idxs_to_sample_from = self.get_unrolled_indices()

    return np.random.choice(idxs_to_sample_from, size=n, replace=True)

  def get(self, key, idx):
    return self.buffers[key][idx]

  def get_all(self, key):
    return self.buffers[key]

  def get_unrolled_indices(self):
    indices = None
    if self.buffer_tail == self.INVALID_IDX:
      indices = []
    elif self.buffer_tail < self.buffer_head:
      indices = list(range(self.buffer_tail, self.buffer_head))
    else:
      indices = list(range(self.buffer_tail, self.buffer_size))
      indices += list(range(0, self.buffer_head))
    return indices

  def get_path_start(self, idx):
    return self.buffers[self.PATH_START_KEY][idx]

  def get_path_end(self, idx):
    return self.buffers[self.PATH_END_KEY][idx]

  def get_subpath_indices(self, idx: int) -> List[int]:
    """Get indices of path that starts at `idx`."""
    start_idx = idx
    end_idx = self.get_path_end(idx)

    if start_idx <= end_idx:
      path_indices = list(range(start_idx, end_idx + 1))
    else:
      path_indices = list(range(start_idx, self.buffer_size))
      path_indices += list(range(0, end_idx + 1))

    return path_indices

  def get_pathlen(self, idx):
    """Returns lengths of paths at indices `idx`."""
    is_array = isinstance(idx, np.ndarray) or isinstance(idx, list)
    if not is_array:
      idx = [idx]

    n = len(idx)
    start_idx = self.get_path_start(idx)
    end_idx = self.get_path_end(idx)
    pathlen = np.empty(n, dtype=int)

    for i in range(n):
      curr_start = start_idx[i]
      curr_end = end_idx[i]
      if curr_start < curr_end:
        curr_len = curr_end - curr_start
      else:
        curr_len = self.buffer_size - curr_start + curr_end
      pathlen[i] = curr_len

    if not is_array:
      pathlen = pathlen[0]

    return pathlen

  def is_valid_path(self, idx):
    """Returns if `idx` is part of a valid path."""
    start_idx = self.get_path_start(idx)
    valid = start_idx != self.INVALID_IDX
    return valid

  def store(self, path: trajectory.Trajectory):
    """Stores a given trajectory in the replay buffer."""
    if not self._path_is_valid(path) or not self._path_check_values(path):
      return self.INVALID_IDX

    n = self._path_length(path)
    if n <= 0:
      return self.INVALID_IDX

    (observations_np, actions_np, rewards_np, raw_rewards_np,
     info_np_dict) = path.as_numpy

    del raw_rewards_np

    return self.store_np(n, observations_np, actions_np, rewards_np,
                         info_np_dict)

  def store_np(self, n, observations_np, actions_np, rewards_np, info_np_dict):
    """Stores a given trajectory in numpy form in the replay buffer."""
    if self.buffers is None:
      # Since obervations_np's shape is (T,) + OBS, similarly for the rest.
      state_shape = observations_np.shape[1:]
      state_dtype = observations_np.dtype

      actions_shape = actions_np.shape[1:]
      actions_dtype = actions_np.dtype

      self.init_buffers(state_shape, state_dtype, actions_shape, actions_dtype)

    idx = self._request_idx(n + 1)
    self._store_path(observations_np, actions_np, rewards_np, info_np_dict, idx)

    self.num_paths += 1
    self.total_count += n + 1
    return idx[0]

  def clear(self):
    self.buffer_head = 0
    self.buffer_tail = self.INVALID_IDX
    self.num_paths = 0
    return

  def get_prev_idx(self, idx) -> int:
    prev_idx = idx - 1
    prev_idx[prev_idx < 0] += self.buffer_size
    is_start = self.is_path_start(idx)
    prev_idx[is_start] = idx[is_start]
    return prev_idx

  def get_next_idx(self, idx) -> int:
    next_idx = np.mod(idx + 1, self.buffer_size)
    is_end = self.is_path_end(idx)
    next_idx[is_end] = idx[is_end]
    return next_idx

  def is_terminal_state(self, idx) -> bool:
    terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
    terminate = terminate_flags != 0
    is_end = self.is_path_end(idx)
    terminal_state = np.logical_and(terminate, is_end)
    return terminal_state

  def check_terminal_flag(self, idx, flag) -> bool:
    terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
    terminate = (terminate_flags == flag.value)
    return terminate

  def is_path_start(self, idx) -> bool:
    is_end = self.buffers[self.PATH_START_KEY][idx] == idx
    return is_end

  def is_path_end(self, idx) -> bool:
    is_end = self.buffers[self.PATH_END_KEY][idx] == idx
    return is_end

  def get_current_size(self) -> int:
    if self.buffer_tail == self.INVALID_IDX:
      return 0
    elif self.buffer_tail < self.buffer_head:
      return self.buffer_head - self.buffer_tail
    else:
      return self.buffer_size - self.buffer_tail + self.buffer_head

  def get_valid_idx(self) -> List[int]:
    """Returns indices that aren't the end states."""
    valid_idx = np.argwhere(
        self.buffers[self.PATH_START_KEY] != self.INVALID_IDX)
    is_end = self.is_path_end(valid_idx)
    valid_idx = valid_idx[np.logical_not(is_end)]
    return valid_idx

  def init_buffers(self, state_shape, state_dtype, actions_shape,
                   actions_dtype):
    """Initialize the buffers."""

    self.buffers = dict()
    self.buffers[self.PATH_START_KEY] = self.INVALID_IDX * np.ones(
        self.buffer_size, dtype=int)
    self.buffers[self.PATH_END_KEY] = self.INVALID_IDX * np.ones(
        self.buffer_size, dtype=int)
    self.buffers[self.TERMINATE_KEY] = np.zeros(
        shape=[self.buffer_size], dtype=int)

    # states
    self.buffers[self.OBSERVATIONS_KEY] = np.zeros(
        [self.buffer_size] + list(state_shape), dtype=state_dtype)

    # actions
    self.buffers[self.ACTIONS_KEY] = np.zeros(
        [self.buffer_size] + list(actions_shape), dtype=actions_dtype)

    # logps & rewards
    self.buffers[self.LOGPS_KEY] = np.zeros(self.buffer_size, dtype=np.float32)
    self.buffers[self.REWARDS_KEY] = np.zeros(
        self.buffer_size, dtype=np.float32)

    return

  def get_valid_indices(self):
    """Returns an array of valid (non-terminal) indices and masks."""
    idx = np.array(self.get_unrolled_indices())

    end_mask = self.is_path_end(idx)
    valid_mask = np.logical_not(end_mask)
    valid_idx = idx[valid_mask]
    valid_idx = np.column_stack([valid_idx, np.nonzero(valid_mask)[0]])

    # `idx` is an array of all filled positions in the buffer.
    # `valid_idx`'s first column are all the valid positions (non-ending) and
    # the second column is its index in the `idx` array.
    return idx, valid_mask, valid_idx

  def iterate_over_paths(self, idx=None):
    """Iterates over valid paths if `idx` is None, else on given paths."""
    if idx is None:
      idx = self.get_unrolled_indices()

    idx = np.asarray(idx)
    assert len(idx.shape) == 1
    n = len(idx)

    # `*_i` are indices into idx
    # `*_idx` are indices into rb.buffers
    start_i = 0
    while start_i < n:
      start_idx = idx[start_i]
      path_length = self.get_pathlen(start_idx)
      end_i = start_i + path_length
      end_idx = idx[end_i]

      # Consistency check.
      assert start_idx == self.get_path_start(start_idx)
      assert end_idx == self.get_path_end(start_idx)

      yield (start_i, end_i + 1)

      # Go to the starting of the next path.
      start_i = end_i + 1

  def _request_idx(self, n):
    """Returns an index capable of storing a trajectory of path length `n`."""
    assert n + 1 < self.buffer_size  # bad things can happen if path is too long

    remainder = n
    idx = []

    start_idx = self.buffer_head
    while remainder > 0:
      end_idx = np.minimum(start_idx + remainder, self.buffer_size)
      remainder -= (end_idx - start_idx)

      free_idx = list(range(start_idx, end_idx))
      self._free_idx(free_idx)
      idx += free_idx
      start_idx = 0

    self.buffer_head = (self.buffer_head + n) % self.buffer_size
    return idx

  def _free_idx(self, idx):
    """Free the whole path at `idx` in the buffer."""
    assert idx[0] <= idx[-1]
    n = len(idx)
    if self.buffer_tail != self.INVALID_IDX:
      update_tail = ((idx[0] <= idx[-1]) and
                     (idx[0] <= self.buffer_tail) and
                     (idx[-1] >= self.buffer_tail))
      update_tail |= idx[0] > idx[-1] and (idx[0] <= self.buffer_tail or
                                           idx[-1] >= self.buffer_tail)

      if update_tail:
        i = 0
        while i < n:
          curr_idx = idx[i]
          if self.is_valid_path(curr_idx):
            start_idx = self.get_path_start(curr_idx)
            end_idx = self.get_path_end(curr_idx)
            pathlen = self.get_pathlen(curr_idx)

            if start_idx < end_idx:
              self.buffers[self.PATH_START_KEY][start_idx:end_idx +
                                                1] = self.INVALID_IDX
            else:
              self.buffers[self.PATH_START_KEY][start_idx:self
                                                .buffer_size] = self.INVALID_IDX
              self.buffers[self.PATH_START_KEY][0:end_idx +
                                                1] = self.INVALID_IDX

            self.num_paths -= 1
            i += pathlen + 1
            self.buffer_tail = (end_idx + 1) % self.buffer_size
          else:
            i += 1
    else:
      self.buffer_tail = idx[0]
    return

  def _store_path(self, states_np, actions_np, rewards_np, info_np_dict,
                  idx: List[int]):
    """Store the given trajectory in the replay buffer."""
    n = actions_np.shape[0]

    assert len(states_np) == n + 1
    assert len(actions_np) == n
    assert len(rewards_np) == n
    logp_np = info_np_dict[self.LOGPS_KEY_TRAJ]
    assert len(logp_np) == n
    if len(logp_np.shape) > 1:
      assert logp_np.shape[1] == 1
      # Sometimes we can get something like (n+1, 1, #actions)
      logp_np = np.squeeze(logp_np, axis=1)
      # Then extract logps only for the actions we carried out.
      logp_np = np.squeeze(logp_np[np.arange(n)[None, :], actions_np])

    self.buffers[self.OBSERVATIONS_KEY][idx[:n + 1]] = [x for x in states_np]
    self.buffers[self.ACTIONS_KEY][idx[:n]] = [x for x in actions_np]
    self.buffers[self.REWARDS_KEY][idx[:n]] = [x for x in rewards_np]
    self.buffers[self.LOGPS_KEY][idx[:n]] = [x for x in logp_np]

    self.buffers[self.TERMINATE_KEY][idx] = 0  # path.terminate.value
    self.buffers[self.PATH_START_KEY][idx] = idx[0]
    self.buffers[self.PATH_END_KEY][idx] = idx[-1]
    return

  def _path_length(self, path: trajectory.Trajectory) -> int:
    # Returns the number of actions, which is 1 less than number of time-steps.
    return path.num_time_steps - 1

  def _path_is_valid(self, path: trajectory.Trajectory) -> bool:
    # The internals of the path are always consistent.
    del path
    return True

  def _path_check_values(self, path: trajectory.Trajectory):
    # Is any value infinity?
    obs_np, actions_np, rewards_np, raw_rewards_np, info_np = path.as_numpy
    del info_np

    def check_all_finite(np_obj):
      return np.isfinite(np_obj).all()

    return all(
        check_all_finite(x)
        for x in [obs_np, actions_np, rewards_np, raw_rewards_np])
