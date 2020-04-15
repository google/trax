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

import functools
import os
import pickle
import time
import numpy as np
import tensorflow as tf

from trax import jaxboard
from trax import lr_schedules as lr
from trax import supervised
from trax.rl import distributions
from trax.rl import task as rl_task


class RLTrainer:
  """Abstract class for RL Trainers, presenting the required API."""

  def __init__(self, task: rl_task.RLTask, collect_per_epoch=None,
               output_dir=None, timestep_to_np=None):
    """Configures the RL Trainer.

    Note that subclasses can have many more arguments, which will be configured
    using defaults and gin. But task and output_dir are passed explicitly.

    Args:
      task: RLTask instance, which defines the environment to train on.
      collect_per_epoch: How many new trajectories to collect in each epoch.
      output_dir: Path telling where to save outputs such as checkpoints.
      timestep_to_np: Timestep-to-numpy function to override in the task.
    """
    self._epoch = 0
    self._task = task
    if timestep_to_np is not None:
      self._task.timestep_to_np = timestep_to_np
    self._collect_per_epoch = collect_per_epoch
    self._output_dir = output_dir
    self._avg_returns = []
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

  def save_to_file(self, file_name='rl.pkl',
                   task_file_name='trajectories.pkl'):
    """Save current epoch number and average returns to file."""
    assert self._output_dir is not None
    task_path = os.path.join(self._output_dir, task_file_name)
    self._task.save_to_file(task_path)
    file_path = os.path.join(self._output_dir, file_name)
    dictionary = {'epoch': self._epoch,
                  'avg_returns': self._avg_returns}
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

  def policy(self, trajectory):
    """Policy function that allows to play using this trainer.

    Args:
      trajectory: an instance of trax.rl.task.Trajectory

    Returns:
      a pair (action, log_prob) where action is the action taken and log_prob
      is the probability assigned to this action (for future use, can be None).
    """
    raise NotImplementedError

  def train_epoch(self):
    """Trains this RL Trainer for one epoch -- main RL logic goes here."""
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
    for _ in range(n_epochs_to_run):
      self._epoch += 1
      cur_time = time.time()
      self.train_epoch()
      supervised.trainer_lib.log(
          'RL training took %.2f seconds.' % (time.time() - cur_time))
      cur_time = time.time()
      avg_return = self.task.collect_trajectories(
          self.policy, self._collect_per_epoch, self._epoch)
      self._avg_returns.append(avg_return)
      supervised.trainer_lib.log(
          'Collecting %d episodes took %.2f seconds.'
          % (self._collect_per_epoch, time.time() - cur_time))
      supervised.trainer_lib.log(
          'Average return in epoch %d was %.2f.' % (self._epoch, avg_return))
      if self._sw is not None:
        self._sw.scalar('timing/collect', time.time() - cur_time,
                        step=self._epoch)
        self._sw.scalar('rl/avg_return', avg_return, step=self._epoch)
        self._sw.scalar('rl/n_interactions', self.task.n_interactions(),
                        step=self._epoch)
        self._sw.scalar('rl/n_trajectories', self.task.n_trajectories(),
                        step=self._epoch)
        self._sw.flush()
      if self._output_dir is not None:
        self.save_to_file()

  def close(self):
    if self._sw is not None:
      self._sw.close()
      self._sw = None


class PolicyTrainer(RLTrainer):
  """Trainer that uses a deep learning model for policy.

  Many deep RL methods, such as policy gradeints (reinforce) or actor-critic
  ones fall into this category, so a lot of classes will be subclasses of this
  one. But some methods only have a value or Q function, these are different.
  """

  def __init__(self, task, policy_model=None, policy_optimizer=None,
               policy_lr_schedule=lr.MultifactorSchedule, policy_batch_size=64,
               policy_train_steps_per_epoch=500, policy_evals_per_epoch=1,
               policy_eval_steps=1, collect_per_epoch=50,
               max_slice_length=1, output_dir=None):
    """Configures the policy trainer.

    Args:
      task: RLTask instance, which defines the environment to train on.
      policy_model: Trax layer, representing the policy model.
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      policy_optimizer: the optimizer to use to train the policy model.
      policy_lr_schedule: learning rate schedule to use to train the policy.
      policy_batch_size: batch size used to train the policy model.
      policy_train_steps_per_epoch: how long to train policy in each RL epoch.
      policy_evals_per_epoch: number of policy trainer evaluations per RL epoch
          - only affects metric reporting.
      policy_eval_steps: number of policy trainer steps per evaluation - only
          affects metric reporting.
      collect_per_epoch: how many trajectories to collect per epoch.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
    """
    super(PolicyTrainer, self).__init__(
        task, collect_per_epoch=collect_per_epoch, output_dir=output_dir)
    self._policy_batch_size = policy_batch_size
    self._policy_train_steps_per_epoch = policy_train_steps_per_epoch
    self._policy_evals_per_epoch = policy_evals_per_epoch
    self._policy_eval_steps = policy_eval_steps
    self._collect_per_epoch = collect_per_epoch
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)

    # Inputs to the policy model are produced by self._policy_batches_stream.
    self._policy_inputs = supervised.Inputs(
        train_stream=lambda _: self.policy_batches_stream())

    policy_model = functools.partial(
        policy_model,
        policy_distribution=self._policy_dist,
    )

    # This is the policy Trainer that will be used to train the policy model.
    # * inputs to the trainer come from self.policy_batches_stream
    # * outputs, targets and weights are passed to self.policy_loss
    self._policy_trainer = supervised.Trainer(
        model=policy_model,
        optimizer=policy_optimizer,
        lr_schedule=policy_lr_schedule,
        loss_fn=self.policy_loss,
        inputs=self._policy_inputs,
        output_dir=output_dir,
        metrics={'policy_loss': self.policy_loss})
    self._policy_eval_model = policy_model(mode='eval')
    policy_batch = next(self.policy_batches_stream())
    self._policy_eval_model.init(policy_batch)

  @property
  def policy_loss(self):
    """Policy loss."""
    return NotImplementedError

  def policy_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    return NotImplementedError

  def policy(self, trajectory):
    """Chooses an action to play after a trajectory."""
    model = self._policy_eval_model
    model.weights = self._policy_trainer.model_weights
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observations[None, ...], n_accelerators=1)
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred)
    log_prob = self._policy_dist.log_prob(pred, sample)
    return (sample.copy(), log_prob.copy())

  def train_epoch(self):
    """Trains RL for one epoch."""
    # When restoring, calculate how many evals are remaining.
    n_evals = remaining_evals(
        self._policy_trainer.step,
        self._epoch,
        self._policy_train_steps_per_epoch,
        self._policy_evals_per_epoch)
    for _ in range(n_evals):
      self._policy_trainer.train_epoch(
          self._policy_train_steps_per_epoch // self._policy_evals_per_epoch,
          self._policy_eval_steps)

  def close(self):
    self._policy_trainer.close()
    super().close()


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


class PolicyGradientTrainer(PolicyTrainer):
  """Trains a policy model using policy gradient on the given RLTask."""

  @property
  def policy_loss(self):
    """Policy loss."""
    return distributions.LogLoss(distribution=self._policy_dist)

  def policy_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._policy_batch_size,
        epochs=[-1],
        max_slice_length=self._max_slice_length,
        sample_trajectories_uniformly=True):
      ret = np_trajectory.returns
      ret = (ret - np.mean(ret)) / np.std(ret)  # Normalize returns.
      # We return a triple (observations, actions, normalized returns) which is
      # later used by the model as (inputs, targets, loss weights).
      yield (np_trajectory.observations, np_trajectory.actions, ret)
