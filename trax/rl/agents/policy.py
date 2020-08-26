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
"""Policy-based RL agents."""

import functools

import numpy as np

from trax import data
from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax import supervised
from trax.rl import distributions
from trax.rl import normalization  # So gin files see it. # pylint: disable=unused-import
from trax.rl.agents import base
from trax.supervised import lr_schedules as lr


class PolicyAgent(base.Agent):
  """Agent that uses a deep learning model for policy.

  Many deep RL methods, such as policy gradient (REINFORCE) or actor-critic fall
  into this category, so a lot of classes will be subclasses of this one. But
  some methods only have a value or Q function, these are different.
  """

  def __init__(self, task, policy_model=None, policy_optimizer=None,
               policy_lr_schedule=lr.multifactor, policy_batch_size=64,
               policy_train_steps_per_epoch=500, policy_evals_per_epoch=1,
               policy_eval_steps=1, n_eval_episodes=0,
               only_eval=False, max_slice_length=1, output_dir=None, **kwargs):
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
      n_eval_episodes: number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
      **kwargs: arguments for the superclass Agent.
    """
    super().__init__(
        task,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        **kwargs
    )
    self._policy_batch_size = policy_batch_size
    self._policy_train_steps_per_epoch = policy_train_steps_per_epoch
    self._policy_evals_per_epoch = policy_evals_per_epoch
    self._policy_eval_steps = policy_eval_steps
    self._only_eval = only_eval
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)

    # Inputs to the policy model are produced by self._policy_batches_stream.
    self._policy_inputs = data.inputs.Inputs(
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
        lr_schedule=policy_lr_schedule(),
        loss_fn=self.policy_loss,
        inputs=self._policy_inputs,
        output_dir=output_dir,
        metrics=self.policy_metrics,
    )
    self._policy_collect_model = tl.Accelerate(
        policy_model(mode='collect'), n_devices=1)
    policy_batch = next(self.policy_batches_stream())
    self._policy_collect_model.init(shapes.signature(policy_batch))
    self._policy_eval_model = tl.Accelerate(
        policy_model(mode='eval'), n_devices=1)  # Not collecting stats
    self._policy_eval_model.init(shapes.signature(policy_batch))
    if self._task._initial_trajectories == 0:
      self._task.remove_epoch(0)
      self._collect_trajectories()

  @property
  def policy_loss(self):
    """Policy loss."""
    return NotImplementedError

  @property
  def policy_metrics(self):
    return {'policy_loss': self.policy_loss}

  def policy_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    return NotImplementedError

  def policy(self, trajectory, temperature=1.0):
    """Chooses an action to play after a trajectory."""
    model = self._policy_collect_model
    if temperature != 1.0:  # When evaluating (t != 1.0), don't collect stats
      model = self._policy_eval_model
      model.state = self._policy_collect_model.state
    model.replicate_weights(self._policy_trainer.model_weights)
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observations[None, ...])
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred, temperature=temperature)
    result = (sample, pred)
    if fastmath.is_backend(fastmath.Backend.JAX):
      result = fastmath.nested_map(lambda x: x.copy(), result)
    return result

  def train_epoch(self):
    """Trains RL for one epoch."""
    # When restoring, calculate how many evals are remaining.
    n_evals = base.remaining_evals(
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


class PolicyGradient(PolicyAgent):
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
