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
"""Classes for defining RL tasks in Trax."""

import collections
import os

import gin
import gym
import numpy as np

from trax import fastmath
from trax.rl import advantages
from trax.supervised import training



# TimeStepBatch stores a single step in the trajectory of an RL run, or
# a sequence of timesteps (trajectory slice), or a batch of such sequences.
# Fields:
# * `observation` at the beginning of the step
# * `action` that was taken
# * `reward` gotten when the action was taken (or None if action wasn't taken)
# * `done` - whether the trajectory has finished in this step
# * `mask` - padding mask
# * `return_` - discounted return from this state (includes the current reward);
#       `None` if it hasn't been computed yet
# * `dist_inputs` - parameters of the policy distribution, stored by some
#       RL algortihms
# TODO(pkozakowski): Generalize `dist_inputs` to `agent_info` - a namedtuple
# storing agent-specific data.
TimeStepBatch = collections.namedtuple('TimeStepBatch', [
    'observation',
    'action',
    'reward',
    'done',
    'mask',
    'dist_inputs',
    'env_info',
    'return_',
])


# EnvInfo stores additional information returned by
# `trax.rl.envs.SequenceDataEnv`. In those environments, one timestep
# corresponds to one token in the sequence. While the environment is emitting
# observation tokens, actions taken by the agent don't matter. Actions can also
# span multiple tokens, but the discount should only be applied once.
# Fields:
# * `control_mask` - mask determining whether the last interaction was
#       controlled, so whether the action performed by the agent mattered;
#       can be used to mask policy and value loss; negation can be used to mask
#       world model observation loss; defaults to 1 - all actions matter
# * `discount_mask` - mask determining whether the discount should be applied to
#       the current reward; defaults to 1 - all rewards are discounted
EnvInfo = collections.namedtuple('EnvInfo', ['control_mask', 'discount_mask'])
EnvInfo.__new__.__defaults__ = (1, 1)


# `env_info` and `return_` can be omitted in `TimeStepBatch`.
TimeStepBatch.__new__.__defaults__ = (EnvInfo(), None,)


class Trajectory:
  """A trajectory of interactions with a RL environment.

  Trajectories are created when interacting with an RL environment. They can
  be prolonged and sliced and when completed, allow to re-calculate returns.
  """

  def __init__(self, observation):
    # TODO(lukaszkaiser): add support for saving and loading trajectories,
    # reuse code from base_trainer.dump_trajectories and related functions.
    self._last_observation = observation
    self._timesteps = []
    self._timestep_batch = None
    self._cached_to_np_args = None

  def __len__(self):
    """Returns the number of observations in the trajectory."""
    # We always have 1 more of observations than of everything else.
    return len(self._timesteps) + 1

  def __repr__(self):
    return repr({
        'timesteps': self._timesteps,
        'last_observation': self._last_observation,
    })

  def suffix(self, length):
    """Returns a `Trajectory` with the last `length` observations."""
    assert length >= 1
    t = Trajectory(self._last_observation)
    t._timesteps = self._timesteps[-(length - 1):]  # pylint: disable=protected-access
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
    return self._last_observation

  @property
  def done(self):
    """Returns whether the trajectory is finished."""
    if not self._timesteps:
      return False
    return self._timesteps[-1].done

  @done.setter
  def done(self, done):
    """Sets the `done` flag in the last timestep."""
    if not self._timesteps:
      raise ValueError('No interactions yet in the trajectory.')
    self._timesteps[-1] = self._timesteps[-1]._replace(done=done)

  def extend(self, new_observation, mask=1, **kwargs):
    """Take action in the last state, getting reward and going to new state."""
    self._timesteps.append(TimeStepBatch(
        observation=self._last_observation, mask=mask, **kwargs
    ))
    self._last_observation = new_observation

  def calculate_returns(self, gamma):
    """Calculate discounted returns."""
    rewards = np.array([ts.reward for ts in self._timesteps])
    discount_mask = np.array([
        ts.env_info.discount_mask for ts in self._timesteps
    ])
    gammas = advantages.mask_discount(gamma, discount_mask)
    returns = advantages.discounted_returns(rewards, gammas)
    for (i, return_) in enumerate(returns):
      self._timesteps[i] = self._timesteps[i]._replace(return_=return_)

  def _default_timestep_to_np(self, ts):
    """Default way to convert timestep to numpy."""
    return fastmath.nested_map(np.array, ts)

  def to_np(self, margin=1, timestep_to_np=None):
    """Create a tuple of numpy arrays from a given trajectory.

    Args:
        margin (int): Number of dummy timesteps past the trajectory end to
            include. By default we include 1, which contains the last
            observation.
        timestep_to_np (callable or None): Optional function
            TimeStepBatch[Any] -> TimeStepBatch[np.array], converting the
            timestep data into numpy arrays.

    Returns:
        TimeStepBatch, where all fields have shape
        (len(self) + margin - 1, ...).
    """
    timestep_to_np = timestep_to_np or self._default_timestep_to_np
    args = (margin, timestep_to_np)

    # Return the cached result if the arguments agree and the trajectory has not
    # grown.
    if self._timestep_batch:
      result_length = len(self) + margin - 1
      length_ok = self._timestep_batch.observation.shape[0] == result_length
      if args == self._cached_to_np_args and length_ok:
        return self._timestep_batch

    # observation, action, reward, etc.
    fields = TimeStepBatch._fields
    # List of timestep data for each field.
    data_lists = TimeStepBatch(**{field: [] for field in fields})
    for timestep in self._timesteps:
      timestep_np = timestep_to_np(timestep)
      # Append each field of timestep_np to the appropriate list.
      for field in fields:
        getattr(data_lists, field).append(getattr(timestep_np, field))
    # Append the last observation.
    data_lists.observation.append(self._last_observation)

    # TODO(pkozakowski): The case len(obs) == 1 is for handling
    # "dummy trajectories", that are only there to determine data shapes. Check
    # if they're still required.
    if len(data_lists.observation) > 1:
      # Extend the trajectory with a given margin - this is to make sure that
      # the networks always "see" the "done" states in the training data, even
      # when a suffix is added to the trajectory slice for better estimation of
      # returns.
      # We set `mask` to 0, so the added timesteps don't influence the loss. We
      # set `done` to True for easier implementation of advantage estimators.
      # The rest of the fields don't matter, so we set them to 0 for easy
      # debugging (unless they're None). The list of observations is longer, so
      # we pad it with margin - 1.
      data_lists.mask.extend([0] * margin)
      data_lists.done.extend([True] * margin)
      data_lists.observation.extend(
          [np.zeros_like(data_lists.observation[-1])] * (margin - 1)
      )
      for field in set(fields) - {'mask', 'done', 'observation'}:
        l = getattr(data_lists, field)
        filler = None if l[-1] is None else np.zeros_like(l[-1])
        l.extend([filler] * margin)

      # Trim the observations to have the same length as the rest of the fields.
      # This is not be the case when margin=0.
      if margin == 0:
        data_lists.observation.pop()

    def stack(x):
      if not x:
        return None
      return fastmath.nested_stack(x)

    # Stack the data_lists into numpy arrays.
    timestep_batch = TimeStepBatch(*map(stack, data_lists))

    self._timestep_batch = timestep_batch
    self._cached_to_np_args = args

    return timestep_batch


def play(env, policy, dm_suite=False, max_steps=None, last_observation=None):
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
    last_observation: last observation from a previous trajectory slice, used to
      begin a new one. Controls whether we reset the environment at the
      beginning - if `None`, resets the env and starts the slice from the
      observation got from reset().

  Returns:
    a completed trajectory slice that was just played.
  """
  done = False
  cur_step = 0
  if last_observation is None:
    # TODO(pkozakowski): Make a Gym wrapper over DM envs to get rid of branches
    # like that.
    last_observation = env.reset().observation if dm_suite else env.reset()
  cur_trajectory = Trajectory(last_observation)
  while not done and (max_steps is None or cur_step < max_steps):
    action, dist_inputs = policy(cur_trajectory)
    step = env.step(action)
    if dm_suite:
      (observation, reward, done) = (
          step.observation, step.reward, step.step_type.last()
      )
      info = {}
    else:
      (observation, reward, done, info) = step

    # Make an EnvInfo out of the supported keys in the info dict.
    env_info = EnvInfo(**{
        key: value for (key, value) in info.items()
        if key in EnvInfo._fields
    })
    cur_trajectory.extend(
        action=action,
        dist_inputs=dist_inputs,
        reward=reward,
        done=done,
        new_observation=observation,
        env_info=env_info,
    )
    cur_step += 1
  return cur_trajectory


def _zero_pad(x, pad, axis):
  """Helper for np.pad with 0s for single-axis case."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = pad  # Padding on axis.
  return np.pad(x, pad_widths, mode='constant',
                constant_values=x.dtype.type(0))


def _random_policy(action_space):
  return lambda _: (action_space.sample(), None)


def _sample_proportionally(inputs, weights):
  """Sample an element from the inputs list proportionally to weights.

  Args:
    inputs: a list, we will return one element of this list.
    weights: a sequence of numbers of the same length as inputs; we will sample
      the k-th input with probability weights[k] / sum(weights).

  Returns:
    an element from inputs.
  """
  l = len(inputs)
  weights = np.array(weights)
  if l != len(weights):
    raise ValueError(f'Inputs and weights must have the same length, but do not'
                     f': {l} != {len(weights)}')
  norm_weights = weights / np.sum(weights)
  # TODO(pkozakowski): Currently this is O(n). It can be sped up to O(log n) by
  # storing CDF and binsearching on it.
  idx = np.random.choice(l, p=norm_weights)
  return inputs[int(idx)]


def _n_slices(trajectory, max_slice_length, margin):
  """How many slices of length upto max_slice_length in a trajectory."""
  # TODO(lukaszkaiser): add option to sample from n last trajectories.
  if not max_slice_length:
    return 1
  # A trajectory [a, b, c, end_state] will have 2 slices of length 2:
  # the slice [a, b] and the one [b, c], with margin=0; 3 with margin=1.
  return max(1, len(trajectory) + margin - max_slice_length)


@gin.configurable
class RLTask:
  """A RL task: environment and a collection of trajectories."""

  def __init__(self, env=gin.REQUIRED,
               initial_trajectories=1,
               gamma=0.99,
               dm_suite=False,
               random_starts=True,
               max_steps=None,
               time_limit=None,
               timestep_to_np=None,
               num_stacked_frames=1,
               n_replay_epochs=1):
    r"""Configures a RL task.

    Args:
      env: Environment confirming to the gym.Env interface or a string,
        in which case `gym.make` will be called on this string to create an env.
      initial_trajectories: either a dict or list of Trajectories to use
        at start or an int, in which case that many trajectories are
        collected using a random policy to play in env. It can be also a string
        and then it should direct to the location where previously recorded
        trajectories are stored.
      gamma: float: discount factor for calculating returns.
      dm_suite: whether we are using the DeepMind suite or the gym interface
      random_starts: use random starts for training of Atari agents.
      max_steps: optional int: cut all trajectory slices at that many steps.
        The trajectory will be continued in the next epochs, up to `time_limit`.
      time_limit: optional int: stop all trajectories after that many steps (or
        after getting "done"). If `None`, use the same value as `max_steps`.
      timestep_to_np: a function that turns a timestep into a numpy array
        (ie., a tensor); if None, we just use the state of the timestep to
        represent it, but other representations (such as embeddings that include
        actions or serialized representations) can be passed here.
      num_stacked_frames: the number of stacked frames for Atari.
      n_replay_epochs: the size of the replay buffer expressed in epochs.
    """
    if isinstance(env, str):
      self._env_name = env
      if dm_suite:
        eval_env = None
        env = None
      else:
        env = gym.make(self._env_name)
        eval_env = gym.make(self._env_name)
    else:
      self._env_name = type(env).__name__
      eval_env = env
    self._env = env
    self._eval_env = eval_env
    self._dm_suite = dm_suite
    self._max_steps = max_steps
    if time_limit is None:
      time_limit = max_steps
    self._time_limit = time_limit
    self._gamma = gamma
    self._initial_trajectories = initial_trajectories
    self._last_observation = None
    self._n_steps_left = time_limit
    # Example trajectory for determining input/output shapes of the networks.
    self._example_trajectory = self.play(
        _random_policy(self.action_space), only_eval=True
    )
    # TODO(lukaszkaiser): find a better way to pass initial trajectories,
    # whether they are an explicit list, a file, or a number of random ones.
    if isinstance(initial_trajectories, int):
      initial_trajectories = [
          self.play(_random_policy(self.action_space))
          for _ in range(initial_trajectories)
      ]
    if isinstance(initial_trajectories, str):
      initial_trajectories = self.load_initial_trajectories_from_path(
          initial_trajectories_path=initial_trajectories)
    if isinstance(initial_trajectories, list):
      if initial_trajectories:
        initial_trajectories = {0: initial_trajectories}
      else:
        initial_trajectories = {}
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
    self._n_replay_epochs = n_replay_epochs
    self._n_trajectories = 0
    self._n_interactions = 0

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

  @property
  def observation_space(self):
    """Returns the env's observation space in a Gym interface."""
    if self._dm_suite:
      return gym.spaces.Box(
          shape=self._env.observation_spec().shape,
          dtype=self._env.observation_spec().dtype,
          low=float('-inf'),
          high=float('+inf'),
      )
    else:
      return self._env.observation_space

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

  def set_n_replay_epochs(self, n_replay_epochs):
    self._n_replay_epochs = n_replay_epochs

  def load_initial_trajectories_from_path(self,
                                          initial_trajectories_path,
                                          dictionary_file='trajectories.pkl',
                                          start_epoch_to_load=0):
    """Initialize trajectories task from file."""
    # We assume that this is a dump generated by Trax
    dictionary_file = os.path.join(initial_trajectories_path, dictionary_file)
    dictionary = training.unpickle_from_file(dictionary_file, gzip=False)
    # TODO(henrykm): as currently implemented this accesses only
    # at most the last n_replay_epochs - this should be more flexible
    epochs_to_load = dictionary['all_epochs'][start_epoch_to_load:]

    all_trajectories = []
    for epoch in epochs_to_load:
      trajectories = training.unpickle_from_file(
          self._epoch_filename(dictionary_file, epoch), gzip=True)
      all_trajectories += trajectories
    return all_trajectories

  def init_from_file(self, file_name):
    """Initialize this task from file."""
    dictionary = training.unpickle_from_file(file_name, gzip=False)
    self._n_trajectories = dictionary['n_trajectories']
    self._n_interactions = dictionary['n_interactions']
    self._max_steps = dictionary['max_steps']
    self._gamma = dictionary['gamma']
    epochs_to_load = dictionary['all_epochs'][-self._n_replay_epochs:]

    for epoch in epochs_to_load:
      trajectories = training.unpickle_from_file(
          self._epoch_filename(file_name, epoch), gzip=True)
      self._trajectories[epoch] = trajectories
    self._saved_epochs_unchanged = epochs_to_load

  def save_to_file(self, file_name):
    """Save this task to file."""
    # Save trajectories from new epochs first.
    epochs_to_save = [e for e in self._trajectories
                      if e not in self._saved_epochs_unchanged]
    for epoch in epochs_to_save:
      training.pickle_to_file(self._trajectories[epoch],
                              self._epoch_filename(file_name, epoch),
                              gzip=True)
    # Now save the list of epochs (so the trajectories are already there,
    # even in case of preemption).
    dictionary = {'n_interactions': self._n_interactions,
                  'n_trajectories': self._n_trajectories,
                  'max_steps': self._max_steps,
                  'gamma': self._gamma,
                  'all_epochs': list(self._trajectories.keys())}
    training.pickle_to_file(dictionary, file_name, gzip=False)

  def play(self, policy, max_steps=None, only_eval=False):
    """Play an episode in env taking actions according to the given policy."""
    if max_steps is None:
      max_steps = self._max_steps
    if only_eval:
      cur_trajectory = play(
          self._eval_env, policy, self._dm_suite,
          # Only step up to the time limit.
          max_steps=min(max_steps, self._time_limit),
          # Always reset at the beginning of an eval episode.
          last_observation=None,
      )
    else:
      cur_trajectory = play(
          self._env, policy, self._dm_suite,
          # Only step up to the time limit, used up by all slices of the same
          # trajectory.
          max_steps=min(max_steps, self._n_steps_left),
          # Pass the environmnent state between play() calls, so one episode can
          # span multiple training epochs.
          # NOTE: Cutting episodes between epochs may hurt e.g. with
          # Transformers.
          # TODO(pkozakowski): Join slices together if this becomes a problem.
          last_observation=self._last_observation,
      )
      # Update the number of steps left to reach the time limit.
      # NOTE: This should really be done as an env wrapper.
      # TODO(pkozakowski): Do that once we wrap the DM envs in Gym interface.
      # The initial reset doesn't count.
      self._n_steps_left -= len(cur_trajectory) - 1
      assert self._n_steps_left >= 0
      if self._n_steps_left == 0:
        cur_trajectory.done = True
      # Pass the last observation between trajectory slices.
      if cur_trajectory.done:
        self._last_observation = None
        # Reset the time limit.
        self._n_steps_left = self._time_limit
      else:
        self._last_observation = cur_trajectory.last_observation

    cur_trajectory.calculate_returns(self._gamma)
    return cur_trajectory

  def collect_trajectories(
      self, policy,
      n_trajectories=None,
      n_interactions=None,
      only_eval=False,
      max_steps=None,
      epoch_id=1,
  ):
    """Collect experience in env playing the given policy."""
    max_steps = max_steps or self.max_steps
    if n_trajectories is not None:
      new_trajectories = [self.play(policy, max_steps=max_steps,
                                    only_eval=only_eval)
                          for _ in range(n_trajectories)]
    elif n_interactions is not None:
      new_trajectories = []
      while n_interactions > 0:
        traj = self.play(policy, max_steps=min(n_interactions, max_steps))
        new_trajectories.append(traj)
        n_interactions -= len(traj) - 1  # The initial reset does not count.
    else:
      raise ValueError(
          'Either n_trajectories or n_interactions must be defined.'
      )

    # Calculate returns.
    returns = [t.total_return for t in new_trajectories]
    if returns:
      mean_returns = sum(returns) / float(len(returns))
    else:
      mean_returns = 0

    # If we're only evaluating, we're done, return the average.
    if only_eval:
      return mean_returns
    # Store new trajectories.
    if new_trajectories:
      self._trajectories[epoch_id].extend(new_trajectories)

    # Mark that epoch epoch_id has changed.
    if epoch_id in self._saved_epochs_unchanged:
      self._saved_epochs_unchanged = [e for e in self._saved_epochs_unchanged
                                      if e != epoch_id]

    # Remove epochs not intended to be in the buffer
    current_trajectories = {
        key: value for key, value in self._trajectories.items()
        if key >= epoch_id - self._n_replay_epochs}
    self._trajectories = collections.defaultdict(list)
    self._trajectories.update(current_trajectories)

    # Update statistics.
    self._n_trajectories += len(new_trajectories)
    self._n_interactions += sum([len(traj) for traj in new_trajectories])

    return mean_returns

  def n_trajectories(self, epochs=None):
    # TODO(henrykm) support selection of epochs if really necessary (will
    # require a dump of a list of lengths in save_to_file
    del epochs
    return self._n_trajectories

  def n_interactions(self, epochs=None):
    # TODO(henrykm) support selection of epochs if really necessary (will
    # require a dump of a list of lengths in save_to_file
    del epochs
    return self._n_interactions

  def _random_slice(self, trajectory, max_slice_length, margin):
    """Returns a random TimeStepBatch slice from a given trajectory."""
    # Sample a slice from the trajectory.
    slice_start = np.random.randint(
        _n_slices(trajectory, max_slice_length, margin)
    )

    # Convert the whole trajectory to Numpy while adding the margin. The
    # result is cached, so we don't have to repeat this for every sample.
    timestep_batch = trajectory.to_np(margin, self._timestep_to_np)

    # Slice and yield the result.
    slice_end = slice_start + (
        max_slice_length or timestep_batch.observation.shape[0]
    )
    return fastmath.nested_map(
        lambda x: x[slice_start:slice_end], timestep_batch
    )

  def _trajectory_stream(self, epochs=None, max_slice_length=None,
                         sample_trajectories_uniformly=False, margin=0):
    """Return a stream of random trajectory slices from the specified epochs.

    Args:
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return
      sample_trajectories_uniformly: whether to sample trajectories uniformly,
        or proportionally to the number of slices in each trajectory (default)
      margin: number of extra steps after "done" that should be included in
        slices, so that networks see the terminal states in the training data

    Yields:
      random trajectory slices sampled uniformly from all slices of length
      up to max_slice_length in all specified epochs
    """
    # {int: array[int]};
    # epoch_to_ns_slices[epoch][i] = n_slices(self._trajectories[epoch][i])
    # It stores arrays for faster sampling.
    epoch_to_ns_slices = {}
    # {int: int};
    # epoch_to_total_n_slices[epoch] = sum(epoch_to_ns_slices[epoch])
    epoch_to_total_n_slices = {}
    # [int]: list of epoch indices to sample from.
    epoch_indices = []
    # epoch_to_total_n_slices filtered using epoch_indices. It's an array for
    # faster sampling.
    sampling_epoch_weights = None

    def new_epoch(epoch_id):
      """Updates the lists defined above to include the new epoch."""
      all_epochs = list(self._trajectories.keys())
      max_epoch = max(all_epochs) + 1

      # Calculate the numbers of slices for the new epoch.
      epoch_to_ns_slices[epoch_id] = np.array([
          _n_slices(trajectory, max_slice_length, margin)
          for trajectory in self._trajectories[epoch_id]
      ])
      epoch_to_total_n_slices[epoch_id] = np.sum(
          epoch_to_ns_slices[epoch_id]
      )

      # Update the indices of epochs to sample from.
      new_epoch_indices = epochs or all_epochs
      new_epoch_indices = [
          # So -1 means "last".
          ep % max_epoch for ep in new_epoch_indices
      ]
      # Remove duplicates and consider only epochs where some trajectories
      # were recorded and that we have processed in new_epoch.
      new_epoch_indices = [
          epoch_id for epoch_id in set(new_epoch_indices)
          if self._trajectories[epoch_id] and epoch_id in epoch_to_ns_slices
      ]
      epoch_indices[:] = new_epoch_indices

      nonlocal sampling_epoch_weights
      sampling_epoch_weights = np.array(list(
          epoch_to_total_n_slices[ep] for ep in epoch_indices
      ))

    while True:
      # If we haven't collected any trajectories yet, yield an example
      # trajectory. It's needed to determine the input/output shapes of
      # networks.
      if not self._trajectories:
        yield self._example_trajectory
        continue

      # Catch up if we have a new epoch or we've restarted the experiment.
      for epoch_id in self._trajectories.keys() - epoch_to_ns_slices.keys():  # pylint:disable=g-builtin-op
        new_epoch(epoch_id)

      # Sample an epoch proportionally to number of slices in each epoch.
      epoch_id = _sample_proportionally(epoch_indices, sampling_epoch_weights)
      epoch = self._trajectories[epoch_id]

      # Sample a trajectory proportionally to number of slices in each one.
      if sample_trajectories_uniformly:
        slices_per_trajectory = np.ones((len(epoch),))
      else:
        slices_per_trajectory = epoch_to_ns_slices[epoch_id]
      trajectory = _sample_proportionally(epoch, slices_per_trajectory)

      yield trajectory

  def trajectory_slice_stream(
      self,
      epochs=None,
      max_slice_length=None,
      sample_trajectories_uniformly=False,
      margin=0,
      trajectory_stream_preprocessing_fn=None,
  ):
    """Return a stream of random trajectory slices from the specified epochs.

    Args:
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return
      sample_trajectories_uniformly: whether to sample trajectories uniformly,
        or proportionally to the number of slices in each trajectory (default)
      margin: number of extra steps after "done" that should be included in
        slices, so that networks see the terminal states in the training data
      trajectory_stream_preprocessing_fn: function to apply to the trajectory
        stream before batching; can be used e.g. to filter trajectories

    Yields:
      random trajectory slices sampled uniformly from all slices of length
      up to max_slice_length in all specified epochs
    """
    trajectory_stream = self._trajectory_stream(
        epochs=epochs,
        max_slice_length=max_slice_length,
        sample_trajectories_uniformly=sample_trajectories_uniformly,
        margin=margin,
    )

    if trajectory_stream_preprocessing_fn is not None:
      trajectory_stream = trajectory_stream_preprocessing_fn(trajectory_stream)

    for trajectory in trajectory_stream:
      yield self._random_slice(trajectory, max_slice_length, margin)

  def trajectory_batch_stream(
      self,
      batch_size,
      epochs=None,
      max_slice_length=None,
      min_slice_length=None,
      margin=0,
      sample_trajectories_uniformly=False,
      trajectory_stream_preprocessing_fn=None,
  ):
    """Return a stream of trajectory batches from the specified epochs.

    This function returns a stream of tuples of numpy arrays (tensors).
    If tensors have different lengths, they will be padded by 0.

    Args:
      batch_size: the size of the batches to return
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return
      min_slice_length: minimum length of the slices of trajectories to return
      margin: number of extra steps after "done" that should be included in
        slices, so that networks see the terminal states in the training data
      sample_trajectories_uniformly: whether to sample trajectories uniformly,
        or proportionally to the number of slices in each trajectory (default)
      trajectory_stream_preprocessing_fn: function to apply to the trajectory
        stream before batching; can be used e.g. to filter trajectories

    Yields:
      batches of trajectory slices sampled uniformly from all slices of length
      at least min_slice_length and up to max_slice_length in all specified
      epochs
    """
    def pad(tensor_list):
      # Replace Nones with valid tensors.
      not_none_tensors = [t for t in tensor_list if t is not None]
      assert not_none_tensors, 'All tensors to pad are None.'
      prototype = np.zeros_like(not_none_tensors[0])
      tensor_list = [t if t is not None else prototype for t in tensor_list]

      max_len = max([t.shape[0] for t in tensor_list])
      if min_slice_length is not None:
        max_len = max(max_len, min_slice_length)
      min_len = min([t.shape[0] for t in tensor_list])
      if max_len == min_len:  # No padding needed.
        return np.array(tensor_list)

      pad_len = 2**int(np.ceil(np.log2(max_len)))
      return np.array([_zero_pad(t, (0, pad_len - t.shape[0]), axis=0)
                       for t in tensor_list])

    trajectory_slice_stream = self.trajectory_slice_stream(
        epochs=epochs,
        max_slice_length=max_slice_length,
        sample_trajectories_uniformly=sample_trajectories_uniformly,
        margin=margin,
        trajectory_stream_preprocessing_fn=trajectory_stream_preprocessing_fn,
    )

    cur_batch = []
    for t in trajectory_slice_stream:
      cur_batch.append(t)
      if len(cur_batch) == batch_size:
        # Make a nested TimeStepBatch of lists out of a list of TimeStepBatches.
        timestep_batch = fastmath.nested_zip(cur_batch)
        # Actions, rewards and returns in the trajectory slice have shape
        # [batch_size, trajectory_length], which we denote as [B, L].
        # Observations are more complex: [B, L] + S, where S is the shape of the
        # observation space (self.observation_space.shape).
        # We stop the recursion at level 1, so we pass lists of arrays into
        # pad().
        yield fastmath.nested_map(pad, timestep_batch, level=1)
        cur_batch = []
