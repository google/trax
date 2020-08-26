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
"""Classes for RL training in Trax."""

import os
import pickle
import time

import gin
import tensorflow as tf

from trax import jaxboard
from trax import supervised
from trax.rl import normalization  # So gin files see it. # pylint: disable=unused-import
from trax.rl import task as rl_task


class Agent:
  """Abstract class for RL agents, presenting the required API."""

  def __init__(self, task: rl_task.RLTask,
               n_trajectories_per_epoch=None,
               n_interactions_per_epoch=None,
               n_eval_episodes=0,
               eval_steps=None,
               only_eval=False,
               output_dir=None,
               timestep_to_np=None):
    """Configures the Agent.

    Note that subclasses can have many more arguments, which will be configured
    using defaults and gin. But task and output_dir are passed explicitly.

    Args:
      task: RLTask instance, which defines the environment to train on.
      n_trajectories_per_epoch: How many new trajectories to collect in each
        epoch.
      n_interactions_per_epoch: How many interactions to collect in each epoch.
      n_eval_episodes: Number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only.
      eval_steps: an optional list of max_steps to use for evaluation
        (defaults to task.max_steps).
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      output_dir: Path telling where to save outputs such as checkpoints.
      timestep_to_np: Timestep-to-numpy function to override in the task.
    """
    assert bool(n_trajectories_per_epoch) != bool(n_interactions_per_epoch), (
        'Exactly one of n_trajectories_per_epoch or n_interactions_per_epoch '
        'should be specified.'
    )
    self._epoch = 0
    self._task = task
    self._eval_steps = eval_steps or [task.max_steps]
    if timestep_to_np is not None:
      self._task.timestep_to_np = timestep_to_np
    self._n_trajectories_per_epoch = n_trajectories_per_epoch
    self._n_interactions_per_epoch = n_interactions_per_epoch
    self._only_eval = only_eval
    self._output_dir = output_dir
    self._avg_returns = []
    self._n_eval_episodes = n_eval_episodes
    self._avg_returns_temperature0 = {step: [] for step in self._eval_steps}
    self._sw = None
    if output_dir is not None:
      self._sw = jaxboard.SummaryWriter(os.path.join(output_dir, 'rl'))

  @property
  def current_epoch(self):
    """Returns current step number in this training session."""
    return self._epoch

  @property
  def task(self):
    """Returns the task."""
    return self._task

  @property
  def avg_returns(self):
    return self._avg_returns

  def save_gin(self):
    assert self._output_dir is not None
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    if self._sw:
      self._sw.text('gin_config',
                    jaxboard.markdownify_operative_config_str(config_str))

  def save_to_file(self, file_name='rl.pkl',
                   task_file_name='trajectories.pkl'):
    """Save current epoch number and average returns to file."""
    assert self._output_dir is not None
    task_path = os.path.join(self._output_dir, task_file_name)
    self._task.save_to_file(task_path)
    file_path = os.path.join(self._output_dir, file_name)
    dictionary = {'epoch': self._epoch,
                  'avg_returns': self._avg_returns,
                  'avg_returns_temperature0': self._avg_returns_temperature0}
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      pickle.dump(dictionary, f)

  def init_from_file(self, file_name='rl.pkl',
                     task_file_name='trajectories.pkl'):
    """Initialize epoch number and average returns from file."""
    assert self._output_dir is not None
    task_path = os.path.join(self._output_dir, task_file_name)
    if tf.io.gfile.exists(task_path):
      self._task.init_from_file(task_path)
    file_path = os.path.join(self._output_dir, file_name)
    if not tf.io.gfile.exists(file_path):
      return
    with tf.io.gfile.GFile(file_path, 'rb') as f:
      dictionary = pickle.load(f)
    self._epoch = dictionary['epoch']
    self._avg_returns = dictionary['avg_returns']
    self._avg_returns_temperature0 = dictionary['avg_returns_temperature0']

  def _collect_trajectories(self):
    return self.task.collect_trajectories(
        self.policy,
        n_trajectories=self._n_trajectories_per_epoch,
        n_interactions=self._n_interactions_per_epoch,
        only_eval=self._only_eval,
        epoch_id=self._epoch
    )

  def policy(self, trajectory, temperature=1.0):
    """Policy function that allows to play using this trainer.

    Args:
      trajectory: an instance of trax.rl.task.Trajectory
      temperature: temperature used to sample from the policy (default=1.0)

    Returns:
      a pair (action, dist_inputs) where action is the action taken and
      dist_inputs is the parameters of the policy distribution, that will later
      be used for training.
    """
    raise NotImplementedError

  def train_epoch(self):
    """Trains this Agent for one epoch -- main RL logic goes here."""
    raise NotImplementedError

  def run(self, n_epochs=1, n_epochs_is_total_epochs=False):
    """Runs this loop for n epochs.

    Args:
      n_epochs: Stop training after completing n steps.
      n_epochs_is_total_epochs: if True, consider n_epochs as the total
        number of epochs to train, including previously trained ones
    """
    if self._output_dir is not None:
      self.init_from_file()
    n_epochs_to_run = n_epochs
    if n_epochs_is_total_epochs:
      n_epochs_to_run -= self._epoch
    cur_n_interactions = 0
    for _ in range(n_epochs_to_run):
      self._epoch += 1
      cur_time = time.time()
      self.train_epoch()
      supervised.trainer_lib.log(
          'RL training took %.2f seconds.' % (time.time() - cur_time))
      cur_time = time.time()
      avg_return = self._collect_trajectories()
      self._avg_returns.append(avg_return)
      if self._n_trajectories_per_epoch:
        supervised.trainer_lib.log(
            'Collecting %d episodes took %.2f seconds.'
            % (self._n_trajectories_per_epoch, time.time() - cur_time))
      else:
        supervised.trainer_lib.log(
            'Collecting %d interactions took %.2f seconds.'
            % (self._n_interactions_per_epoch, time.time() - cur_time))
      supervised.trainer_lib.log(
          'Average return in epoch %d was %.2f.' % (self._epoch, avg_return))
      if self._n_eval_episodes > 0:
        for steps in self._eval_steps:
          avg_return_temperature0 = self.task.collect_trajectories(
              lambda x: self.policy(x, temperature=0.0),
              n_trajectories=self._n_eval_episodes,
              max_steps=steps, only_eval=True)
          self._avg_returns_temperature0[steps].append(avg_return_temperature0)
          supervised.trainer_lib.log(
              'Avg return with temperature 0 at %d steps in epoch %d was %.2f.'
              % (steps, self._epoch, avg_return_temperature0))
      if self._sw is not None:
        self._sw.scalar('timing/collect', time.time() - cur_time,
                        step=self._epoch)
        self._sw.scalar('rl/avg_return', avg_return, step=self._epoch)
        if self._n_eval_episodes > 0:
          for steps in self._eval_steps:
            self._sw.scalar('rl/avg_return_temperature0_steps%d' % steps,
                            self._avg_returns_temperature0[steps][-1],
                            step=self._epoch)
        self._sw.scalar('rl/n_interactions', self.task.n_interactions(),
                        step=self._epoch)
        self._sw.scalar('rl/n_interactions_per_second',
                        (self.task.n_interactions() - cur_n_interactions)/ \
                        (time.time() - cur_time),
                        step=self._epoch)
        cur_n_interactions = self.task.n_interactions()
        self._sw.scalar('rl/n_trajectories', self.task.n_trajectories(),
                        step=self._epoch)
        self._sw.flush()
      if self._output_dir is not None and self._epoch == 1:
        self.save_gin()
      if self._output_dir is not None:
        self.save_to_file()

  def close(self):
    if self._sw is None:
      return
    self._sw.close()
    self._sw = None


def remaining_evals(cur_step, epoch, train_steps_per_epoch, evals_per_epoch):
  """Helper function to calculate remaining evaluations for a trainer.

  Args:
    cur_step: current step of the supervised trainer
    epoch: current epoch of the RL trainer
    train_steps_per_epoch: supervised trainer steps per RL epoch
    evals_per_epoch: supervised trainer evals per RL epoch

  Returns:
    number of remaining evals to do this epoch

  Raises:
    ValueError if the provided numbers indicate a step mismatch
  """
  if epoch < 1:
    raise ValueError('Epoch must be at least 1, got %d' % epoch)
  prev_steps = (epoch - 1) * train_steps_per_epoch
  done_steps_this_epoch = cur_step - prev_steps
  if done_steps_this_epoch < 0:
    raise ValueError('Current step (%d) < previously done steps (%d).'
                     % (cur_step, prev_steps))
  train_steps_per_eval = train_steps_per_epoch // evals_per_epoch
  if done_steps_this_epoch % train_steps_per_eval != 0:
    raise ValueError('Done steps (%d) must divide train steps per eval (%d).'
                     % (done_steps_this_epoch, train_steps_per_eval))
  return evals_per_epoch - (done_steps_this_epoch // train_steps_per_eval)
