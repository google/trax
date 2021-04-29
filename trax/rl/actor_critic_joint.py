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
"""Classes for RL training in Trax."""

import functools

from trax import data
from trax import layers as tl
from trax import supervised
from trax.fastmath import numpy as jnp
from trax.fastmath import stop_gradient
from trax.rl import actor_critic
from trax.rl import distributions
from trax.rl import rl_layers
from trax.rl import training as rl_training
from trax.supervised import lr_schedules as lr


# pylint: disable=g-long-lambda
class ActorCriticJointAgent(rl_training.Agent):
  """Trains a joint policy-and-value model using actor-critic methods."""

  def __init__(self, task,
               joint_model=None,
               optimizer=None,
               lr_schedule=lr.multifactor,
               batch_size=64,
               train_steps_per_epoch=500,
               supervised_evals_per_epoch=1,
               supervised_eval_steps=1,
               n_trajectories_per_epoch=50,
               max_slice_length=1,
               normalize_advantages=True,
               output_dir=None,
               n_replay_epochs=1):
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
      n_trajectories_per_epoch: how many trajectories to collect per epoch.
      max_slice_length: the maximum length of trajectory slices to use.
      normalize_advantages: if True, then normalize advantages - currently
          implemented only in PPO.
      output_dir: Path telling where to save outputs (evals and checkpoints).
      n_replay_epochs: how many last epochs to take into the replay buffer;
           > 1 only makes sense for off-policy algorithms.
    """
    super().__init__(
        task,
        n_trajectories_per_epoch=n_trajectories_per_epoch,
        output_dir=output_dir,
    )
    self._batch_size = batch_size
    self._train_steps_per_epoch = train_steps_per_epoch
    self._supervised_evals_per_epoch = supervised_evals_per_epoch
    self._supervised_eval_steps = supervised_eval_steps
    self._n_trajectories_per_epoch = n_trajectories_per_epoch
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)
    self._lr_schedule = lr_schedule()
    self._optimizer = optimizer
    self._normalize_advantages = normalize_advantages
    self._n_replay_epochs = n_replay_epochs
    self._task.set_n_replay_epochs(n_replay_epochs)

    # Inputs to the joint model are produced by self.batches_stream.
    self._inputs = data.inputs.Inputs(
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
                 'advantage_norm': self.advantage_norm,
                 'value_loss': self.value_loss,
                 'explained_variance': self.explained_variance,
                 'log_probs_mean': self.log_probs_mean,
                 'preferred_move': self.preferred_move})
    self._eval_model = tl.Accelerate(
        self._joint_model(mode='eval'), n_devices=1)
    example_batch = next(self.batches_stream())
    self._eval_model.init(example_batch)

  def close(self):
    self._trainer.close()
    super().close()

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
    def f(dist_inputs, values, returns):
      del dist_inputs
      return jnp.mean(returns - values)
    return tl.Fn('AdvantageMean', f)

  @property
  def advantage_norm(self):
    """Norm of advantages."""
    def f(dist_inputs, values, returns):
      del dist_inputs
      return jnp.linalg.norm(returns - values)
    return tl.Fn('AdvantageNorm', f)

  @property
  def value_loss(self):
    """Value loss - so far generic for all A2C."""
    def f(dist_inputs, values, returns):
      del dist_inputs
      return rl_layers.ValueLoss(values, returns, self._value_loss_coeff)
    return tl.Fn('ValueLoss', f)

  @property
  def explained_variance(self):
    """Explained variance metric."""
    def f(dist_inputs, values, returns):
      del dist_inputs
      return rl_layers.ExplainedVariance(values, returns)
    return tl.Fn('ExplainedVariance', f)

  @property
  def log_probs_mean(self):
    """Mean of log_probs aka dist_inputs."""
    def f(dist_inputs, values):
      del values
      return jnp.mean(dist_inputs)
    return tl.Fn('LogProbsMean', f)

  @property
  def preferred_move(self):
    """Preferred move - the mean of selected moves."""
    def f(dist_inputs, values):
      del values
      return rl_layers.PreferredMove(dist_inputs, self._policy_dist.sample)
    return tl.Fn('PreferredMove', f)

  def policy(self, trajectory, temperature=1.0):
    """Chooses an action to play after a trajectory."""
    model = self._eval_model
    model.replicate_weights(self._trainer.model_weights)
    # The two lines below along with the copying
    # before return make the TPU happy
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observation[None, ...])[0]
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred, temperature=temperature)
    return (sample.copy(), pred.copy())

  def train_epoch(self):
    """Trains RL for one epoch."""
    n_evals = rl_training.remaining_evals(
        self._trainer.step,
        self._epoch,
        self._train_steps_per_epoch,
        self._supervised_evals_per_epoch)
    for _ in range(n_evals):
      self._trainer.train_epoch(
          self._train_steps_per_epoch // self._supervised_evals_per_epoch,
          self._supervised_eval_steps)


class PPOJoint(ActorCriticJointAgent):
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
    super().__init__(task, **kwargs)
    self._trainer = supervised.Trainer(
        model=self._joint_model,
        optimizer=self._optimizer,
        lr_schedule=self._lr_schedule,
        loss_fn=self.joint_loss,
        inputs=self._inputs,
        output_dir=self._output_dir,
        metrics={'joint_loss': self.joint_loss,
                 'advantage_mean': self.advantage_mean,
                 'advantage_norm': self.advantage_norm,
                 'value_loss': self.value_loss,
                 'explained_variance': self.explained_variance,
                 'log_probs_mean': self.log_probs_mean,
                 'entropy_loss': self.entropy_loss,
                 'probs_ratio_mean': self.probs_ratio_mean,
                 'unclipped_objective_mean': self.unclipped_objective_mean,
                 'clipped_objective_mean': self.clipped_objective_mean,
                 'ppo_objective_mean': self.ppo_objective_mean,
                 'clip_fraction': self.clip_fraction,
                 'preferred_move': self.preferred_move,
                 'approximate_kl_divergence': self.approximate_kl_divergence})

  def batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._batch_size, max_slice_length=self._max_slice_length, epochs=[-1]):
      if np_trajectory.dist_inputs is not None:
        old_dist_inputs = np_trajectory.dist_inputs
      else:
        old_dist_inputs = jnp.zeros(
            np_trajectory.reward.shape + (self._policy_dist.n_inputs,)
        )
      old_log_probs = self._policy_dist.log_prob(
          old_dist_inputs, np_trajectory.action
      )
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (np_trajectory.observation,         # Inputs to the value model.
             np_trajectory.return_[:, :, None],
             np_trajectory.done[:, :, None],
             np_trajectory.reward[:, :, None],
             np_trajectory.action,
             old_log_probs,
             np_trajectory.mask)

  @property
  def joint_loss(self):
    """Joint policy and value loss."""
    def f(dist_inputs, values, returns, dones, rewards,
          actions, old_log_probs, mask):
      """Definition of the Proximal Policy Optimization loss."""
      del mask  # TODO(lukaszkaiser): make PPO work with Transformer
      # We have dist_inputs of the shape float32[128,1,18]
      assert len(dist_inputs.shape) == 3, (
          f'dist_inputs.shape was {dist_inputs.shape}'
          f'but expected length of the tensor shape is 3')
      # values of the shape float32[128,1,1]
      # returns of the shape float32[128,1,1]
      # dones of the shape int32[128,1,1]
      # rewards of the shape float32[128,1,1]
      # and old_log_probs of the shape float32[128,1]
      assert values.shape == returns.shape, (
          f'values.shape was {values.shape}'
          f'returns.shape was {returns.shape}')
      assert values.shape == dones.shape, (
          f'values.shape was {values.shape}'
          f'returns.shape was {dones.shape}')
      assert rewards.shape == dones.shape, (
          f'values.shape was {values.shape}'
          f'returns.shape was {dones.shape}')
      assert returns.shape[0:2] == old_log_probs.shape, (
          f'returns.shape was {returns.shape}'
          f'old_log_probs.shape was {old_log_probs.shape}')

      # actions is a tensor of the shape int32[128,1] in the case
      # of discrete actions and float32[128,1,6] in the case of
      # half-cheetah and other continuous actions
      # actions agree with returns/values on the first two coordinates
      # meaning batch and time
      assert actions.shape[0:2] == returns.shape[0:2], (
          f'actions.shape was {actions.shape} and '
          f'returns.shape was {returns.shape}')

      ppo_objective = rl_layers.PPOObjective(
          dist_inputs, stop_gradient(values), returns, dones, rewards,
          actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob,
          epsilon=self._epsilon,
          normalize_advantages=self._normalize_advantages)

      # we insist that ppo_objective is a vector of shape [128,1]
      assert len(ppo_objective.shape) == 2, (
          f'ppo_objective was {ppo_objective}')
      # which agrees with returns/values/actions on the first two coordinates
      assert ppo_objective.shape[0:2] == values.shape[0:2], (
          f'ppo_objective.shape was {ppo_objective.shape} and '
          f'values.shape was {values.shape}')

      entropy_loss = rl_layers.EntropyLoss(
          dist_inputs,
          distribution=self._policy_dist,
          coeff=self._entropy_coeff,
      )

      assert jnp.ndim(entropy_loss) == 0, f'entropy_loss was {entropy_loss}'

      l2_value_loss = rl_layers.ValueLoss(
          values, returns, value_loss_coeff=self._value_loss_coeff)

      assert jnp.ndim(l2_value_loss) == 0, f'l2_value_loss was {l2_value_loss}'

      return -ppo_objective.mean() + l2_value_loss - entropy_loss

    return tl.Fn('PPOJointLoss', f)

  # pylint: disable=invalid-name
  @property
  def probs_ratio_mean(self):
    """Joint policy and value loss layer."""
    def ProbsRatioMean(dist_inputs, actions, old_log_probs):
      """Probability Ratio Mean from the PPO algorithm."""
      probs_ratio = rl_layers.ProbsRatio(
          dist_inputs, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
      return jnp.mean(probs_ratio)

    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      del values, returns, dones, rewards
      return ProbsRatioMean(dist_inputs, actions, old_log_probs)
    return tl.Fn('ProbsRatioMean', f)

  @property
  def clip_fraction(self):
    """Joint policy and value loss layer."""
    def ClipFraction(dist_inputs, actions, old_log_probs):
      """Probability Ratio Mean from the PPO algorithm."""
      probs_ratio = rl_layers.ProbsRatio(
          dist_inputs, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
      return jnp.mean(jnp.abs(probs_ratio - 1) > self._epsilon)

    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      del values, returns, dones, rewards
      return ClipFraction(dist_inputs, actions, old_log_probs)
    return tl.Fn('ClipFraction', f)
  # pylint: enable=invalid-name

  @property
  def entropy_loss(self):
    """Entropy layer."""
    def f(dist_inputs, values, returns, dones, rewards, actions):
      del values, returns, dones, rewards, actions
      return rl_layers.EntropyLoss(
          dist_inputs,
          distribution=self._policy_dist,
          coeff=self._entropy_coeff,
      )
    return tl.Fn('EntropyLoss', f)

  @property
  def approximate_kl_divergence(self):
    """Approximate KL divergence."""
    def f(dist_inputs, values, returns, dones, rewards,
          actions, old_log_probs):
      del values, returns, dones, rewards
      return rl_layers.ApproximateKLDivergence(
          dist_inputs,
          actions,
          old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
    return tl.Fn('ApproximateKLDivergence', f)

  @property
  def unclipped_objective_mean(self):
    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      """Unclipped objective Mean from the PPO algorithm."""
      del dones, rewards
      advantages = returns - values
      probs_ratio = rl_layers.ProbsRatio(
          dist_inputs, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
      # advantages are of the shape [128,1,1]
      # and probs_ratio are of the shape [128,1]
      advantages = advantages.squeeze(axis=2)
      unclipped_objective = rl_layers.UnclippedObjective(
          probs_ratio, advantages)
      return jnp.mean(unclipped_objective)

    return tl.Fn('UnclippedObjectiveMean', f)

  @property
  def clipped_objective_mean(self):
    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      """Clipped objective from the PPO algorithm."""
      del dones, rewards
      advantages = returns - values
      probs_ratio = rl_layers.ProbsRatio(
          dist_inputs, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
      # advantages are of the shape [128,1,1]
      # and probs_ratio are of the shape [128,1]
      advantages = advantages.squeeze(axis=2)
      clipped_objective = rl_layers.ClippedObjective(
          probs_ratio, advantages, epsilon=self._epsilon)
      return jnp.mean(clipped_objective)

    return tl.Fn('ClippedObjectiveMean', f)

  @property
  def ppo_objective(self):
    """PPO objective with local parameters."""
    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      return rl_layers.PPOObjective(
          dist_inputs, values, returns, dones, rewards, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob,
          epsilon=self._epsilon,
          normalize_advantages=self._normalize_advantages)
    return tl.Fn('PPOObjective', f)

  @property
  def ppo_objective_mean(self):
    """PPO objective mean."""
    def f(dist_inputs, values, returns, dones, rewards, actions, old_log_probs):
      """Clipped objective from the PPO algorithm."""
      ppo_objective = rl_layers.PPOObjective(
          dist_inputs, values, returns, dones, rewards, actions, old_log_probs,
          log_prob_fun=self._policy_dist.log_prob,
          epsilon=self._epsilon,
          normalize_advantages=self._normalize_advantages)
      return jnp.mean(ppo_objective)
    return tl.Fn('PPOObjectiveMean', f)


class A2CJoint(ActorCriticJointAgent):
  """The A2C algorithm.

  Trains policy and value models using the A2C algortithm.
  """

  on_policy = True

  def __init__(self, task, value_loss_coeff=0.1,
               entropy_coeff=0.01, **kwargs):
    """Configures the A2C Trainer."""
    self._value_loss_coeff = value_loss_coeff
    self._entropy_coeff = entropy_coeff
    super().__init__(task, **kwargs)
    self._trainer = supervised.Trainer(
        model=self._joint_model,
        optimizer=self._optimizer,
        lr_schedule=self._lr_schedule,
        loss_fn=self.joint_loss,
        inputs=self._inputs,
        output_dir=self._output_dir,
        metrics={'joint_loss': self.joint_loss,
                 'advantage_mean': self.advantage_mean,
                 'advantage_norm': self.advantage_norm,
                 'value_loss': self.value_loss,
                 'explained_variance': self.explained_variance,
                 'log_probs_mean': self.log_probs_mean,
                 'entropy_loss': self.entropy_loss,
                 'a2c_objective_mean': self.a2c_objective_mean,
                 'approximate_kl_divergence': self.approximate_kl_divergence,
                 'preferred_move': self.preferred_move})

  def batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._batch_size, max_slice_length=self._max_slice_length, epochs=[-1]):
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (np_trajectory.observation,         # Inputs to the value model.
             np_trajectory.return_[:, :, None],
             np_trajectory.done[:, :, None],
             np_trajectory.reward[:, :, None],
             np_trajectory.action,
             jnp.zeros_like(np_trajectory.mask),
             np_trajectory.mask)

  @property
  def joint_loss(self):
    """Joint policy and value loss."""
    def f(dist_inputs, values, returns, dones, rewards,
          actions, old_log_probs, mask):
      """Definition of the A2C loss."""
      del old_log_probs

      # Typically we have dist_inputs of the shape float32[128,1,18]
      assert len(dist_inputs.shape) == 3, (
          f'dist_inputs.shape was {dist_inputs.shape} '
          f'but expected length of the tensor shape is 3')
      # values of the shape float32[128,1,1]
      # returns of the shape float32[128,1,1]
      assert values.shape == returns.shape, (
          f'values.shape was {values.shape}'
          f'returns.shape was (returns.shape)')
      # actions of the shape int32[128,1] in the case of discrete actions
      # and float32[128,1,6] in the case of of half-cheetah
      # actions agree with returns/values on the first two coordinates
      assert actions.shape[0:2] == returns.shape[0:2], (
          f'actions.shape was {actions.shape}'
          f'returns.shape was (returns.shape)')
      # and mask of the shape float32[128,1]
      assert len(mask.shape) == 2, f'mask.shape was {mask.shape}'
      # which agrees with returns/values/actions on the first two coordinates
      assert mask.shape[0:2] == returns.shape[0:2], (
          f'mask.shape was {mask.shape}'
          f'returns.shape was (returns.shape)')

      a2c_objective = rl_layers.A2CObjective(
          dist_inputs,
          stop_gradient(values),
          returns, dones, rewards, actions, mask,
          log_prob_fun=self._policy_dist.log_prob,
          normalize_advantages=self._normalize_advantages)

      # we insist that a2c_objective is a scalar
      assert jnp.ndim(a2c_objective) == 0, f'a2c_objective was {a2c_objective}'

      entropy_loss = rl_layers.EntropyLoss(
          dist_inputs,
          distribution=self._policy_dist,
          coeff=self._entropy_coeff,
      )

      assert jnp.ndim(entropy_loss) == 0, f'entropy_loss was {entropy_loss}'

      l2_value_loss = rl_layers.ValueLoss(
          values, returns, value_loss_coeff=self._value_loss_coeff)

      assert jnp.ndim(l2_value_loss) == 0, f'l2_value_loss was {l2_value_loss}'

      combined_loss = a2c_objective + l2_value_loss - entropy_loss

      return combined_loss

    return tl.Fn('A2CJointLoss', f, n_out=1)

  @property
  def entropy_loss(self):
    """Entropy layer."""
    def f(dist_inputs, values, returns, dones, rewards, actions):
      del values, returns, dones, rewards, actions
      return rl_layers.EntropyLoss(
          dist_inputs,
          distribution=self._policy_dist,
          coeff=self._entropy_coeff,
      )
    return tl.Fn('EntropyLoss', f)

  @property
  def approximate_kl_divergence(self):
    """Approximate KL divergence."""
    def f(dist_inputs, values, returns, dones, rewards,
          actions, old_log_probs):
      del values, returns, dones, rewards
      return rl_layers.ApproximateKLDivergence(
          dist_inputs,
          actions,
          old_log_probs,
          log_prob_fun=self._policy_dist.log_prob)
    return tl.Fn('ApproximateKLDivergence', f)

  @property
  def a2c_objective(self):
    """A2C objective with local parameters."""
    return tl.Fn(
        'A2CObjective',
        lambda dist_inputs, values, returns, dones, rewards, actions, \
        old_log_probs, mask: rl_layers.A2CObjective(
            dist_inputs,
            values,
            returns,
            dones,
            rewards,
            actions,
            mask,
            log_prob_fun=self._policy_dist.log_prob,
            normalize_advantages=self._normalize_advantages),
        n_out=1)

  @property
  def a2c_objective_mean(self):
    """A2C objective mean."""
    def f(dist_inputs, values, returns, dones, rewards,
          actions, old_log_probs, mask):
      """A2C objective mean."""
      # TODO(henrykm): include dones, rewards
      del old_log_probs
      a2c_objective = rl_layers.A2CObjective(
          dist_inputs, values, returns, dones, rewards, actions, mask,
          log_prob_fun=self._policy_dist.log_prob,
          normalize_advantages=self._normalize_advantages)
      return jnp.mean(a2c_objective)
    return tl.Fn('A2CObjectiveMean', f, n_out=1)


class AWRJoint(ActorCriticJointAgent):
  """Trains a joint policy-and-value model using AWR."""

  # TODO(henrykm): value_loss_coeff looks like a common parameter
  def __init__(self, task, value_loss_coeff=0.1, beta=1.0, w_max=20.0,
               thresholds=None, **kwargs):
    """Configures the joint AWR Trainer."""
    self._beta = beta
    self._w_max = w_max
    self._thresholds = thresholds
    self._value_loss_coeff = value_loss_coeff
    super().__init__(task, **kwargs)

  def batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._batch_size, max_slice_length=self._max_slice_length):
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (np_trajectory.observation,         # Inputs to the value model.
             np_trajectory.return_[:, :, None],  # Targets: regress to returns.
             np_trajectory.action,              # Policy targets: actions.
             np_trajectory.mask)                 # Padding mask.

  @property
  def joint_loss(self):
    """Joint policy and value loss."""

    def f(preds, values, returns, actions, mask):
      advantages = jnp.squeeze(returns - stop_gradient(values), axis=-1)
      logps = self._policy_dist.log_prob(preds, actions)
      awr_loss = actor_critic.AWRLoss(
          beta=self._beta, w_max=self._w_max, thresholds=self._thresholds)(
              (logps, advantages, jnp.zeros_like(logps), mask))
      l2_value_loss = jnp.mean((returns - values)**2) * self._value_loss_coeff
      return awr_loss + l2_value_loss
    return tl.Fn('AWRJointLoss', f)
