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
"""Classes for defining RL tasks in Trax."""

import collections
import gin
import gym
import numpy as np


class _TimeStep(object):
  """A single step of interaction with a RL environment.

  TimeStep stores a single step in the trajectory of an RL run:
  * state (same as observation) at the beginning of the step
  * action that was takes (or None if none taken yet)
  * reward gotten when the action was taken (or None if action wasn't taken)
  * log-probability of the action taken (or None if not specified)
  * discounted return from that state (includes the reward from this step)
  """

  def __init__(self, state, action=None, reward=None, log_prob=None):
    self.state = state
    self.action = action
    self.reward = reward
    self.log_prob = log_prob
    self.discounted_return = None


class Trajectory(object):
  """A trajectory of interactions with a RL environment.

  Trajectories are created when interacting with a RL environment. They can
  be prolonged and sliced and when completed, allow to re-calculate returns.
  """

  def __init__(self, state):
    # TODO(lukaszkaiser): add support for saving and loading trajectories,
    # reuse code from base_trainer.dump_trajectories and related functions.
    if state is not None:
      self._timesteps = [_TimeStep(state)]

  def __len__(self):
    return len(self._timesteps)

  def __str__(self):
    return str([(ts.state, ts.action, ts.reward) for ts in self._timesteps])

  def __getitem__(self, key):
    t = Trajectory(None)
    t._timesteps = self._timesteps[key]  # pylint: disable=protected-access
    return t

  @property
  def total_return(self):
    """Sum of all rewards in this trajectory."""
    return sum([t.reward or 0.0 for t in self._timesteps])

  @property
  def last_state(self):
    """Return the last state in this trajectory."""
    last_timestep = self._timesteps[-1]
    return last_timestep.state

  def extend(self, action, log_prob, reward, new_state):
    """Take action in the last state, getting reward and going to new state."""
    last_timestep = self._timesteps[-1]
    last_timestep.action = action
    last_timestep.log_prob = log_prob
    last_timestep.reward = reward
    new_timestep = _TimeStep(new_state)
    self._timesteps.append(new_timestep)

  def calculate_returns(self, gamma):
    """Calculate discounted returns."""
    ret = 0.0
    for timestep in reversed(self._timesteps):
      cur_reward = timestep.reward or 0.0
      ret = gamma * ret + cur_reward
      timestep.discounted_return = ret

  def _default_timestep_to_np(self, ts):
    """Default way to convert timestep to numpy."""
    if ts.action is None:
      return (np.array(ts.state, dtype=np.float32), None, None, None, None)
    # Currently we add 1 to actions because discrete actions in envs are given
    # as integers from 0, ..., n_actions - 1. But when batching multiple
    # trajectories we pad with 0, so it is useful to shift the real actions
    # to be from 1 to n_actions and reserve 0 for padding.
    # TODO(lukaszkaiser): revisit the action padding issue, continuous actions.
    return (np.array(ts.state, dtype=np.float32),
            np.array(ts.action + 1, dtype=np.int32),  # Shift by 1 for padding.
            np.array(ts.log_prob, dtype=np.float32),
            np.array(ts.reward, dtype=np.float32),
            np.array(ts.discounted_return, dtype=np.float32))

  def to_np(self, timestep_to_np=None):
    """Create a tuple of numpy arrays from a given trajectory."""
    observations, actions, logps, rewards, returns = [], [], [], [], []
    timestep_to_np = timestep_to_np or self._default_timestep_to_np
    for timestep in self._timesteps:
      if timestep.action is None:
        obs = timestep_to_np(timestep)[0]
        observations.append(obs[None, ...])
      else:
        (obs, act, logp, rew, ret) = timestep_to_np(timestep)
        observations.append(obs[None, ...])
        actions.append(act[None, ...])
        logps.append(logp[None, ...])
        rewards.append(rew[None, ...])
        returns.append(ret[None, ...])
    # TODO(lukaszkaiser): use np.stack instead?
    return (np.concatenate(observations, axis=0),
            np.concatenate(actions, axis=0),
            np.concatenate(logps, axis=0),
            np.concatenate(rewards, axis=0),
            np.concatenate(returns, axis=0))


def _zero_pad(x, pad, axis):
  """Helper for np.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return np.pad(x, pad_widths, mode='constant',
                constant_values=x.dtype.type(0))


def _random_policy(n_actions):
  return lambda _: (np.random.randint(n_actions), np.log(1 / float(n_actions)))


@gin.configurable()
class RLTask:
  """A RL task: environment and a collection of trajectories."""

  def __init__(self, env=gin.REQUIRED, initial_trajectories=1, gamma=0.99,
               max_steps=None, timestep_to_np=None):
    r"""Configures a RL task.

    Args:
      env: Environment confirming to the gym.Env interface or a string,
        in which case `gym.make` will be called on this string to create an env.
      initial_trajectories: either a list of Trajectories to use as at start
        or an int, in which case that many trajectories are
        collected using a random policy to play in env.
      gamma: float: discount factor for calculating returns.
      max_steps: Optional int: stop all trajectories at that many steps.
      timestep_to_np: a function that turns a timestep into a numpy array
        (ie., a tensor); if None, we just use the state of the timestep to
        represent it, but other representations (such as embeddings that include
        actions or serialized representations) can be passed here.

    """
    if isinstance(env, str):
      env = gym.make(env)
    self._env = env
    self._max_steps = max_steps
    self._gamma = gamma
    # TODO(lukaszkaiser): find a better way to pass initial trajectories,
    # whether they are an explicit list, a file, or a number of random ones.
    if isinstance(initial_trajectories, int):
      initial_trajectories = [self.play(_random_policy(self.n_actions))
                              for _ in range(initial_trajectories)]
    self._timestep_to_np = timestep_to_np
    # Stored trajectories are indexed by epoch and within each epoch they
    # are stored in the order of generation so we can implement replay buffers.
    # TODO(lukaszkaiser): use dump_trajectories from BaseTrainer to allow
    # saving and reading trajectories from disk.
    self._trajectories = collections.defaultdict(list)
    self._trajectories[0] = initial_trajectories

  @property
  def max_steps(self):
    return self._max_steps

  @property
  def n_actions(self):
    return self._env.action_space.n

  @property
  def state_shape(self):
    return self._env.observation_space.shape

  def play(self, policy):
    """Play an episode in env taking actions according to the given policy.

    Environment is first reset and an from then on, a game proceeds. At each
    step, the policy is asked to choose an action and the environment moves
    forward. A Trajectory is created in that way and returns when the episode
    finished, which is either when env returns `done` or max_steps is reached.

    Args:
      policy: a function taking a Trajectory and returning an action (int).

    Returns:
      a completed trajectory that was just played.
    """
    terminal = False
    cur_step = 0
    cur_trajectory = Trajectory(self._env.reset())
    while not terminal and cur_step < self.max_steps:
      action, log_prob = policy(cur_trajectory)
      state, reward, terminal, _ = self._env.step(action)
      cur_trajectory.extend(action, log_prob, reward, state)
      cur_step += 1
    cur_trajectory.calculate_returns(self._gamma)
    return cur_trajectory

  def collect_trajectories(self, policy, n, epoch_id=1):
    """Collect n trajectories in env playing the given policy."""
    new_trajectories = [self.play(policy) for _ in range(n)]
    self._trajectories[epoch_id].extend(new_trajectories)
    returns = [t.total_return for t in new_trajectories]
    return sum(returns) / float(len(returns))

  def trajectory_stream(self, epochs=None, max_slice_length=None):
    """Return a stream of random trajectory slices from the specified epochs.

    Args:
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return

    Yields:
      random trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    # TODO(lukaszkaiser): add option to sample from n last trajectories.
    def n_slices(t):
      """How many slices of length upto max_slice_length in a trajectory."""
      if not max_slice_length:
        return 1
      # A trajectory [a, b, c, end_state] will have 2 proper slices of length 2:
      # the slice [a, b] and the one [b, c].
      return max(1, len(t) - max_slice_length)

    while True:
      all_epochs = list(self._trajectories.keys())
      max_epoch = max(all_epochs) + 1
      epochs = epochs or all_epochs
      epochs = [ep % max_epoch for ep in epochs]  # So -1 means "last".
      # TODO(lukaszkaiser): the code below can probably be better using
      # np.random.choice(..., p=probabilities) and sampling like this.
      slices = [[n_slices(t) for t in self._trajectories[ep]] for ep in epochs]
      slices_per_epoch = [sum(s) for s in slices]
      slice_id = np.random.randint(sum(slices_per_epoch))  # Which slice?
      # We picked a trajectory slice, which epoch and trajectory is it in?
      slices_per_epoch_sums = np.array(slices_per_epoch).cumsum()
      epoch_id = min([i for i in range(len(epochs))
                      if slices_per_epoch_sums[i] >= slice_id])
      slice_in_epoch_id = slices_per_epoch_sums[epoch_id] - slice_id
      slices_in_epoch_sums = np.array(slices[epoch_id]).cumsum()
      trajectory_id = min([i for i in range(len(self._trajectories[epoch_id]))
                           if slices_in_epoch_sums[i] >= slice_in_epoch_id])
      trajectory = self._trajectories[epoch_id][trajectory_id]
      slice_start = slices_in_epoch_sums[trajectory_id] - slice_in_epoch_id
      slice_end = slice_start + (max_slice_length or len(trajectory))
      slice_end = min(slice_end, len(trajectory) - 1)
      yield trajectory[slice_start:slice_end]

  def batches_stream(self, batch_size, epochs=None, max_slice_length=None):
    """Return a stream of trajectory batches from the specified epochs.

    This function returns a stream of tuples of numpy arrays (tensors).
    If tensors have different lengths, they will be padded by 0.

    Args:
      batch_size: the size of the batches to return
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return

    Yields:
      batches of trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    def pad(tensor_list):
      max_len = max([t.shape[0] for t in tensor_list])
      pad_len = 2**int(np.ceil(np.log2(max_len)))
      return np.array([_zero_pad(t, (0, pad_len - t.shape[0]), axis=0)
                       for t in tensor_list])
    cur_batch = []
    for t in self.trajectory_stream(epochs, max_slice_length):
      cur_batch.append(t)
      if len(cur_batch) == batch_size:
        obs, act, logp, rew, ret = zip(*[t.to_np(self._timestep_to_np)
                                         for t in cur_batch])
        yield pad(obs), pad(act), pad(logp), pad(rew), pad(ret)
        cur_batch = []
