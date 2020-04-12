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
import os

import gin
import gym
import numpy as np

from trax.supervised import trainer_lib


class _TimeStep(object):
  """A single step of interaction with a RL environment.

  TimeStep stores a single step in the trajectory of an RL run:
  * observation (same as observation) at the beginning of the step
  * action that was takes (or None if none taken yet)
  * reward gotten when the action was taken (or None if action wasn't taken)
  * log-probability of the action taken (or None if not specified)
  * discounted return from that state (includes the reward from this step)
  """

  def __init__(self, observation, action=None, reward=None, log_prob=None):
    self.observation = observation
    self.action = action
    self.reward = reward
    self.log_prob = log_prob
    self.discounted_return = None


# Tuple for representing trajectories and batches of them in numpy; immutable.
TrajectoryNp = collections.namedtuple('TrajectoryNp', [
    'observations',
    'actions',
    'log_probs',
    'rewards',
    'returns',
    'mask'
])


class Trajectory(object):
  """A trajectory of interactions with a RL environment.

  Trajectories are created when interacting with a RL environment. They can
  be prolonged and sliced and when completed, allow to re-calculate returns.
  """

  def __init__(self, observation):
    # TODO(lukaszkaiser): add support for saving and loading trajectories,
    # reuse code from base_trainer.dump_trajectories and related functions.
    if observation is not None:
      self._timesteps = [_TimeStep(observation)]

  def __len__(self):
    return len(self._timesteps)

  def __str__(self):
    return str([(ts.observation, ts.action, ts.reward)
                for ts in self._timesteps])

  def __repr__(self):
    return repr([(ts.observation, ts.action, ts.reward)
                 for ts in self._timesteps])

  def __getitem__(self, key):
    t = Trajectory(None)
    t._timesteps = self._timesteps[key]  # pylint: disable=protected-access
    return t

  @property
  def timesteps(self):
    return self._timesteps

  @property
  def total_return(self):
    """Sum of all rewards in this trajectory."""
    return sum([t.reward or 0.0 for t in self._timesteps])

  @property
  def last_observation(self):
    """Return the last observation in this trajectory."""
    last_timestep = self._timesteps[-1]
    return last_timestep.observation

  def extend(self, action, log_prob, reward, new_observation):
    """Take action in the last state, getting reward and going to new state."""
    last_timestep = self._timesteps[-1]
    last_timestep.action = action
    last_timestep.log_prob = log_prob
    last_timestep.reward = reward
    new_timestep = _TimeStep(new_observation)
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
      return (np.array(ts.observation),
              None, None, None, None)
    return (np.array(ts.observation),
            np.array(ts.action),
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
        observations.append(obs)
      else:
        (obs, act, logp, rew, ret) = timestep_to_np(timestep)
        observations.append(obs)
        actions.append(act)
        logps.append(logp)
        rewards.append(rew)
        returns.append(ret)
    def stack(x):
      if not x:
        return None
      return np.stack(x, axis=0)
    return TrajectoryNp(stack(observations),
                        stack(actions),
                        stack(logps),
                        stack(rewards),
                        stack(returns),
                        None)


def play(env, policy, dm_suite=False, max_steps=None):
  """Play an episode in env taking actions according to the given policy.

  Environment is first reset and an from then on, a game proceeds. At each
  step, the policy is asked to choose an action and the environment moves
  forward. A Trajectory is created in that way and returns when the episode
  finished, which is either when env returns `done` or max_steps is reached.

  Args:
    env: the environment to play in, conforming to gym.Env or
      DeepMind suite interfaces.
    policy: a function taking a Trajectory and returning a pair consisting
      of an action (int or float) and the confidence in that action (float,
      defined as the log of the probability of taking that action).
    dm_suite: whether we are using the DeepMind suite or the gym interface
    max_steps: for how many steps to play.

  Returns:
    a completed trajectory that was just played.
  """
  terminal = False
  cur_step = 0
  if dm_suite:
    cur_trajectory = Trajectory(env.reset().observation)
    while not terminal and (max_steps is None or cur_step < max_steps):
      action, log_prob = policy(cur_trajectory)
      observation = env.step(action)
      cur_trajectory.extend(action, log_prob,
                            observation.reward,
                            observation.observation)
      cur_step += 1
      terminal = observation.step_type.last()
  else:
    cur_trajectory = Trajectory(env.reset())
    while not terminal and (max_steps is None or cur_step < max_steps):
      action, log_prob = policy(cur_trajectory)
      observation, reward, terminal, _ = env.step(action)
      cur_trajectory.extend(action, log_prob, reward, observation)
      cur_step += 1
  return cur_trajectory


def _zero_pad(x, pad, axis):
  """Helper for np.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return np.pad(x, pad_widths, mode='constant',
                constant_values=x.dtype.type(0))


def _random_policy(action_space):
  # TODO(pkozakowski): Make returning the log probabilities optional.
  # Returning 1 as a log probability is a temporary hack.
  return lambda _: (action_space.sample(), 1.0)


def _sample_proportionally(inputs, weights):
  """Sample an element from the inputs list proportionally to weights.

  Args:
    inputs: a list, we will return one element of this list.
    weights: a list of numbers of the same length as inputs; we will sample
      the k-th input with probability weights[k] / sum(weights).

  Returns:
    an element from inputs.
  """
  l = len(inputs)
  if l != len(weights):
    raise ValueError(f'Inputs and weights must have the same length, but do not'
                     f': {l} != {len(weights)}')
  weights_sum = float(sum(weights))
  norm_weights = [w / weights_sum for w in weights]
  idx = np.random.choice(l, p=norm_weights)
  return inputs[int(idx)]


@gin.configurable()
class RLTask:
  """A RL task: environment and a collection of trajectories."""

  def __init__(self, env=gin.REQUIRED, initial_trajectories=1, gamma=0.99,
               dm_suite=False, max_steps=None,
               timestep_to_np=None, num_stacked_frames=1):
    r"""Configures a RL task.

    Args:
      env: Environment confirming to the gym.Env interface or a string,
        in which case `gym.make` will be called on this string to create an env.
      initial_trajectories: either a dict or list of Trajectories to use
        at start or an int, in which case that many trajectories are
        collected using a random policy to play in env.
      gamma: float: discount factor for calculating returns.
      dm_suite: whether we are using the DeepMind suite or the gym interface
      max_steps: Optional int: stop all trajectories at that many steps.
      timestep_to_np: a function that turns a timestep into a numpy array
        (ie., a tensor); if None, we just use the state of the timestep to
        represent it, but other representations (such as embeddings that include
        actions or serialized representations) can be passed here.
      num_stacked_frames: the number of stacked frames for Atari.
    """
    if isinstance(env, str):
      self._env_name = env
      if dm_suite:
        env = environments.load_from_settings(
            platform='atari',
            settings={
                'levelName': env,
                'interleaved_pixels': True,
                'zero_indexed_actions': True
            })
        env = atari_wrapper.AtariWrapper(environment=env,
                                         num_stacked_frames=num_stacked_frames)
      else:
        env = gym.make(env)
    else:
      self._env_name = type(env).__name__
    self._env = env
    self._dm_suite = dm_suite
    self._max_steps = max_steps
    self._gamma = gamma
    # TODO(lukaszkaiser): find a better way to pass initial trajectories,
    # whether they are an explicit list, a file, or a number of random ones.
    if isinstance(initial_trajectories, int):
      initial_trajectories = [self.play(_random_policy(self.action_space))
                              for _ in range(initial_trajectories)]
    if isinstance(initial_trajectories, list):
      initial_trajectories = {0: initial_trajectories}
    self._timestep_to_np = timestep_to_np
    # Stored trajectories are indexed by epoch and within each epoch they
    # are stored in the order of generation so we can implement replay buffers.
    # TODO(lukaszkaiser): use dump_trajectories from BaseTrainer to allow
    # saving and reading trajectories from disk.
    self._trajectories = collections.defaultdict(list)
    self._trajectories.update(initial_trajectories)
    # When we repeatedly save, trajectories for many epochs do not change, so
    # we don't need to save them again. This keeps track which are unchanged.
    self._saved_epochs_unchanged = []

  @property
  def env(self):
    return self._env

  @property
  def env_name(self):
    return self._env_name

  @property
  def max_steps(self):
    return self._max_steps

  @property
  def gamma(self):
    return self._gamma

  @property
  def action_space(self):
    if self._dm_suite:
      return gym.spaces.Discrete(self._env.action_spec().num_values)
    else:
      return self._env.action_space

  def observation_shape(self):
    if self._dm_suite:
      return self._env.observation_spec().shape
    else:
      return self._env.observation_space.shape

  @property
  def trajectories(self):
    return self._trajectories

  @property
  def timestep_to_np(self):
    return self._timestep_to_np

  @timestep_to_np.setter
  def timestep_to_np(self, ts):
    self._timestep_to_np = ts

  def _epoch_filename(self, base_filename, epoch):
    """Helper function: file name for saving the given epoch."""
    # If base is /foo/task.pkl, we save epoch 1 under /foo/task_epoch1.pkl.
    filename, ext = os.path.splitext(base_filename)
    return filename + '_epoch' + str(epoch) + ext

  def init_from_file(self, file_name):
    """Initialize this task from file."""
    dictionary = trainer_lib.unpickle_from_file(file_name, gzip=False)
    self._max_steps = dictionary['max_steps']
    self._gamma = dictionary['gamma']
    epochs_to_load = dictionary['all_epochs']
    for epoch in epochs_to_load:
      trajectories = trainer_lib.unpickle_from_file(
          self._epoch_filename(file_name, epoch), gzip=True)
      self._trajectories[epoch] = trajectories
    self._saved_epochs_unchanged = epochs_to_load

  def save_to_file(self, file_name):
    """Save this task to file."""
    # Save trajectories from new epochs first.
    epochs_to_save = [e for e in self._trajectories.keys()
                      if e not in self._saved_epochs_unchanged]
    for epoch in epochs_to_save:
      trainer_lib.pickle_to_file(self._trajectories[epoch],
                                 self._epoch_filename(file_name, epoch),
                                 gzip=True)
    # Now save the list of epochs (so the trajectories are already there,
    # even in case of preemption).
    dictionary = {'max_steps': self._max_steps,
                  'gamma': self._gamma,
                  'all_epochs': list(self._trajectories.keys())}
    trainer_lib.pickle_to_file(dictionary, file_name, gzip=False)

  def play(self, policy):
    """Play an episode in env taking actions according to the given policy."""
    cur_trajectory = play(self._env, policy, self._dm_suite, self._max_steps)
    cur_trajectory.calculate_returns(self._gamma)
    return cur_trajectory

  def collect_trajectories(self, policy, n, epoch_id=1):
    """Collect n trajectories in env playing the given policy."""
    new_trajectories = [self.play(policy) for _ in range(n)]
    self._trajectories[epoch_id].extend(new_trajectories)
    # Mark that epoch epoch_id has changed.
    if epoch_id in self._saved_epochs_unchanged:
      self._saved_epochs_unchanged = [e for e in self._saved_epochs_unchanged
                                      if e != epoch_id]
    # Calculate returns.
    returns = [t.total_return for t in new_trajectories]
    return sum(returns) / float(len(returns))

  def n_trajectories(self, epochs=None):
    all_epochs = list(self._trajectories.keys())
    epoch_indices = epochs or all_epochs
    return sum([len(self._trajectories[ep]) for ep in epoch_indices])

  def n_interactions(self, epochs=None):
    all_epochs = list(self._trajectories.keys())
    epoch_indices = epochs or all_epochs
    return sum([sum([len(traj) for traj in self._trajectories[ep]])
                for ep in epoch_indices])

  def trajectory_stream(self, epochs=None, max_slice_length=None,
                        include_final_state=False,
                        sample_trajectories_uniformly=False):
    """Return a stream of random trajectory slices from the specified epochs.

    Args:
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return
      include_final_state: whether to include slices with the final state of
        the trajectory which may have no action and reward
      sample_trajectories_uniformly: whether to sample trajectories uniformly,
       or proportionally to the number of slices in each trajectory (default)

    Yields:
      random trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    # TODO(lukaszkaiser): add option to sample from n last trajectories.
    end_offset = 0 if include_final_state else 1
    def n_slices(t):
      """How many slices of length upto max_slice_length in a trajectory."""
      if not max_slice_length:
        return 1
      # A trajectory [a, b, c, end_state] will have 2 slices of length 2:
      # the slice [a, b] and the one [b, c], with end_offset; 3 without.
      return max(1, len(t) - max_slice_length + 1 - end_offset)

    while True:
      all_epochs = list(self._trajectories.keys())
      max_epoch = max(all_epochs) + 1
      # Bind the epoch indices to a new name so they can be recalculated every
      # epoch.
      epoch_indices = epochs or all_epochs
      epoch_indices = [
          ep % max_epoch for ep in epoch_indices
      ]  # So -1 means "last".

      # Sample an epoch proportionally to number of slices in each epoch.
      if len(epoch_indices) == 1:  # Skip this step if there's just 1 epoch.
        epoch_id = epoch_indices[0]
      else:
        slices_per_epoch = [sum([n_slices(t) for t in self._trajectories[ep]])
                            for ep in epoch_indices]
        epoch_id = _sample_proportionally(epoch_indices, slices_per_epoch)
      epoch = self._trajectories[epoch_id]

      # Sample a trajectory proportionally to number of slices in each one.
      if sample_trajectories_uniformly:
        slices_per_trajectory = [1] * len(epoch)
      else:
        slices_per_trajectory = [n_slices(t) for t in epoch]
      trajectory = _sample_proportionally(epoch, slices_per_trajectory)

      # Sample a slice from the trajectory.
      slice_start = np.random.randint(n_slices(trajectory))
      slice_end = slice_start + (max_slice_length or len(trajectory))
      slice_end = min(slice_end, len(trajectory) - end_offset)
      yield trajectory[slice_start:slice_end]

  def trajectory_batch_stream(self, batch_size, epochs=None,
                              max_slice_length=None,
                              include_final_state=False,
                              sample_trajectories_uniformly=False):
    """Return a stream of trajectory batches from the specified epochs.

    This function returns a stream of tuples of numpy arrays (tensors).
    If tensors have different lengths, they will be padded by 0.

    Args:
      batch_size: the size of the batches to return
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return
      include_final_state: whether to include slices with the final state of
        the trajectory which may have no action and reward
      sample_trajectories_uniformly: whether to sample trajectories uniformly,
       or proportionally to the number of slices in each trajectory (default)

    Yields:
      batches of trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    def pad(tensor_list):
      max_len = max([t.shape[0] for t in tensor_list])
      min_len = min([t.shape[0] for t in tensor_list])
      if max_len == min_len:  # No padding needed.
        return np.array(tensor_list)

      pad_len = 2**int(np.ceil(np.log2(max_len)))
      return np.array([_zero_pad(t, (0, pad_len - t.shape[0]), axis=0)
                       for t in tensor_list])
    cur_batch = []
    for t in self.trajectory_stream(
        epochs, max_slice_length,
        include_final_state, sample_trajectories_uniformly):
      cur_batch.append(t)
      if len(cur_batch) == batch_size:
        obs, act, logp, rew, ret, _ = zip(*[t.to_np(self._timestep_to_np)
                                            for t in cur_batch])
        # Where act, logp, rew and ret will usually have the following shape:
        # [batch_size, trajectory_length-1], which we call [B, L-1].
        # Observations are more complex and will usuall be [B, L] + S where S
        # is the shape of the observation space (self.observation_shape).
        yield TrajectoryNp(
            pad(obs), pad(act), pad(logp), pad(rew), pad(ret),
            pad([np.ones(a.shape[:1]) for a in act]))
        cur_batch = []
