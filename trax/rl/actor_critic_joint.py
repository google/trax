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

from trax import layers as tl
from trax import lr_schedules as lr
from trax import supervised
from trax.math import numpy as jnp
from trax.rl import actor_critic
from trax.rl import distributions
from trax.rl import training as rl_training


class ActorCriticJointTrainer(rl_training.RLTrainer):
  """Trains a joint policy-and-value model using actor-critic methods."""

  def __init__(self, task, joint_model=None,
               optimizer=None, lr_schedule=lr.MultifactorSchedule,
               batch_size=64, train_steps_per_epoch=500,
               supervised_evals_per_epoch=1, supervised_eval_steps=1,
               collect_per_epoch=50, max_slice_length=1, output_dir=None):
    """Configures the joint trainer.

    Args:
      task: RLTask instance, which defines the environment to train on.
      joint_model: Trax layer, representing the joint policy and value model.
      optimizer: the optimizer to use to train the joint model.
      lr_schedule: learning rate schedule to use to train the joint model/.
      batch_size: batch size used to train the joint model.
      train_steps_per_epoch: how long to train the joint model in each RL epoch.
      supervised_evals_per_epoch: number of value trainer evaluations per RL
          epoch - only affects metric reporting.
      supervised_eval_steps: number of value trainer steps per evaluation -
          only affects metric reporting.
      collect_per_epoch: how many trajectories to collect per epoch.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
    """
    super(ActorCriticJointTrainer, self).__init__(
        task, collect_per_epoch=collect_per_epoch, output_dir=output_dir)
    self._batch_size = batch_size
    self._train_steps_per_epoch = train_steps_per_epoch
    self._supervised_evals_per_epoch = supervised_evals_per_epoch
    self._supervised_eval_steps = supervised_eval_steps
    self._collect_per_epoch = collect_per_epoch
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)
    self._total_samples = 0
    self._lr_schedule = lr_schedule
    self._optimizer = optimizer

    # Inputs to the joint model are produced by self.batches_stream.
    self._inputs = supervised.Inputs(
        train_stream=lambda _: self.batches_stream())

    self._joint_model = functools.partial(
        joint_model,
        policy_distribution=self._policy_dist,
    )

    # This is the joint Trainer that will be used to train the policy model.
    # * inputs to the trainer come from self.batches_stream
    # * outputs are passed to self._joint_loss
    self._trainer = supervised.Trainer(
        model=self._joint_model,
        optimizer=self._optimizer,
        lr_schedule=self._lr_schedule,
        loss_fn=self.joint_loss,
        inputs=self._inputs,
        output_dir=output_dir,
        metrics={'joint_loss': self.joint_loss,
                 'advantage_mean': self.advantage_mean,
                 'value_loss': self.value_loss,
                 'log_probs_mean': self.log_probs_mean,
                 'preferred_move': self.preferred_move})
    self._eval_model = self._joint_model(mode='eval')
    example_batch = next(self.batches_stream())
    self._eval_model.init(example_batch)

  def batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    return NotImplementedError

  @property
  def joint_loss(self):
    """Joint policy and value loss layer."""
    return NotImplementedError

  @property
  def advantage_mean(self):
    """Mean of advantages."""
    @tl.layer(n_in=3, n_out=1)
    def AdvantageMean(x, **unused_kwargs):
      """Definition of the mean of advantages."""
      _, values, returns = x
      advantages = returns - values
      return jnp.mean(advantages)
    return AdvantageMean

  @property
  def value_loss(self):
    """Value loss - so far generic for all A2C."""
    @tl.layer(n_in=3, n_out=1)
    def ValueLoss(x, **unused_kwargs):
      """Definition of the Proximal Policy Optimization loss."""
      _, values, returns = x
      advantages = returns - values
      l2_value_loss = jnp.sum(advantages**2) * self._value_loss_coeff
      return l2_value_loss
    return ValueLoss

  @property
  def log_probs_mean(self):
    """Mean of log_probs aka dist_inputs."""
    # PPO is a widely used actor-critic RL algorithm.
    @tl.layer(n_in=2, n_out=1)
    def LogProbsMean(x, **unused_kwargs):
      """Definition of the Proximal Policy Optimization loss."""
      dist_inputs, _ = x
      return jnp.mean(dist_inputs)
    return LogProbsMean

  @property
  def preferred_move(self):
    """Preferred move - the mean of selected moves."""
    @tl.layer(n_in=2, n_out=1)
    def PreferredMove(x, **unused_kwargs):
      """Definition of the preferred move."""
      dist_inputs, _ = x
      preferred_moves = self._policy_dist.sample(dist_inputs, temperature=0.0)
      return jnp.mean(preferred_moves)
    return PreferredMove

  def policy(self, trajectory):
    """Chooses an action to play after a trajectory."""
    model = self._eval_model
    model.weights = self._trainer.model_weights
    # The two lines below along with the copying
    # before return make the TPU happy
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observations[None, ...], n_accelerators=1)[0]
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred)
    log_prob = self._policy_dist.log_prob(pred, sample)
    return (sample.copy(), log_prob.copy())

  def train_epoch(self):
    """Trains RL for one epoch."""
    for _ in range(self._supervised_evals_per_epoch):
      self._trainer.train_epoch(
          self._train_steps_per_epoch // self._supervised_evals_per_epoch,
          self._supervised_eval_steps,
      )


class PPOJointTrainer(ActorCriticJointTrainer):
  """The Proximal Policy Optimization Algorithm aka PPO.

  Trains policy and value models using the PPO algortithm.
  """

  # TODO(henrykm): make on_policy more generic
  # (currently epochs are passed manually)
  on_policy = True

  def __init__(self, task, epsilon=0.2, value_loss_coeff=0.1,
               entropy_coeff=0.01, **kwargs):
    """Configures the PPO Trainer."""
    self._epsilon = epsilon
    self._value_loss_coeff = value_loss_coeff
    self._entropy_coeff = entropy_coeff
    super(PPOJointTrainer, self).__init__(task, **kwargs)
    self._trainer = supervised.Trainer(
        model=self._joint_model,
        optimizer=self._optimizer,
        lr_schedule=self._lr_schedule,
        loss_fn=self.joint_loss,
        inputs=self._inputs,
        output_dir=self._output_dir,
        metrics={'joint_loss': self.joint_loss,
                 'advantage_mean': self.advantage_mean,
                 'value_loss': self.value_loss,
                 'log_probs_mean': self.log_probs_mean,
                 'entropy_loss': self.entropy_loss,
                 'probs_ratio_mean': self.probs_ratio_mean,
                 'preferred_move': self.preferred_move})

  def batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._batch_size, max_slice_length=self._max_slice_length, epochs=[-1]):
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      self._total_samples += np_trajectory.observations.shape[0]
      if self._sw is not None:
        self._sw.scalar('total_samples', self._total_samples)
      yield (np_trajectory.observations,         # Inputs to the value model.
             np_trajectory.returns[:, :, None],
             np_trajectory.actions,
             np_trajectory.log_probs,
             np_trajectory.mask)

  # TODO(henrykm): Use the layers defined above
  @property
  def joint_loss(self):
    """Joint policy and value loss."""
    @tl.layer(n_in=6, n_out=1)
    def PPOJointLoss(x, **unused_kwargs):
      """Definition of the Proximal Policy Optimization loss."""
      dist_inputs, values, returns, actions, old_log_probs, mask = x
      del mask  # TODO(lukaszkaiser): make PPO work with Transformer
      new_log_probs = self._policy_dist.log_prob(dist_inputs, actions)

      advantages = returns - values
      l2_value_loss = jnp.sum(advantages**2) * self._value_loss_coeff

      # Old log probs have an undesirable extra dimension which we remove here
      old_log_probs = jnp.array(old_log_probs.squeeze(axis=-1),
                                dtype=jnp.float32)
      new_log_probs = jnp.array(new_log_probs.squeeze(axis=-1))

      # The ratio between new_probs and old_probs expressed
      # using log_probs and exponentaion
      probs_ratio = jnp.exp(new_log_probs - old_log_probs)
      unclipped_objective = probs_ratio * advantages
      clipped_objective = jnp.clip(probs_ratio,
                                   1 - self._epsilon,
                                   1 + self._epsilon) * advantages
      ppo_objective = jnp.minimum(unclipped_objective, clipped_objective)

      entropy_loss = self._policy_dist.entropy(new_log_probs) *\
          self._entropy_coeff

      return -ppo_objective.mean() + l2_value_loss - entropy_loss
    return PPOJointLoss

  @property
  def probs_ratio_mean(self):
    """Joint policy and value loss layer."""
    @tl.layer(n_in=5, n_out=1)
    def ProbsRatioMean(x, **unused_kwargs):
      """Probability Ratio Mean from the PPO algorithm."""
      dist_inputs, _, _, actions, old_log_probs = x
      new_log_probs = self._policy_dist.log_prob(dist_inputs, actions)

      # Old log probs have an undesirable extra dimension which we remove here
      old_log_probs = jnp.array(old_log_probs.squeeze(axis=-1),
                                dtype=jnp.float32)
      new_log_probs = jnp.array(new_log_probs.squeeze(axis=-1))

      # The ratio between new_probs and old_probs expressed
      # using log_probs and exponentaion
      probs_ratio = jnp.exp(new_log_probs - old_log_probs)
      return jnp.mean(probs_ratio)
    return ProbsRatioMean

  @property
  def entropy_loss(self):
    """Entropy layer."""
    @tl.layer(n_in=4, n_out=1)
    def EntropyLoss(x, **unused_kwargs):
      """Definition of the Entropy Layer."""
      dist_inputs, _, _, actions = x
      new_log_probs = self._policy_dist.log_prob(dist_inputs, actions)
      new_log_probs = jnp.array(new_log_probs.squeeze(axis=-1))
      entropy_loss = self._policy_dist.entropy(new_log_probs) *\
          self._entropy_coeff
      return entropy_loss
    return EntropyLoss

  # TODO(henrykm) Add more debug PPO metrics
  @property
  def unclipped_objective_mean(self):
    """Unclipped objective - make sense for PPO."""
    return NotImplementedError

  @property
  def clipped_objective_mean(self):
    """Clipped objective - make sense for PPO."""
    return NotImplementedError

  @property
  def ppo_objective_mean(self):
    """PPO objective - make sense for PPO."""
    return NotImplementedError


class AWRJointTrainer(ActorCriticJointTrainer):
  """Trains a joint policy-and-value model using AWR."""

  # TODO(henrykm): value_loss_coeff looks like a common parameter
  def __init__(self, task, value_loss_coeff=0.1, beta=1.0, w_max=20.0,
               **kwargs):
    """Configures the joint AWR Trainer."""
    self._beta = beta
    self._w_max = w_max
    self._value_loss_coeff = value_loss_coeff
    super(AWRJointTrainer, self).__init__(task, **kwargs)

  def batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._batch_size, max_slice_length=self._max_slice_length):
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (np_trajectory.observations,         # Inputs to the value model.
             np_trajectory.returns[:, :, None],  # Targets: regress to returns.
             np_trajectory.actions,              # Policy targets: actions.
             np_trajectory.mask)                 # Padding mask.

  @property
  def joint_loss(self):
    """Joint policy and value loss."""
    @tl.layer(n_in=5, n_out=1)
    def AWRJointLoss(x, **unused_kwargs):  # pylint: disable=invalid-name
      preds, values, returns, actions, mask = x
      advantages = jnp.squeeze(returns - values, axis=-1)
      logps = self._policy_dist.log_prob(preds, actions)
      awr_loss = actor_critic.AWRLoss(beta=self._beta, w_max=self._w_max)(
          (logps, advantages, jnp.zeros_like(logps), mask))
      l2_value_loss = jnp.sum((returns - values)**2) * self._value_loss_coeff
      return awr_loss + l2_value_loss
    return AWRJointLoss
