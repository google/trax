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
import numpy as np

from trax import layers as tl
from trax import lr_schedules as lr
from trax import shapes
from trax import supervised
from trax.math import numpy as jnp
from trax.rl import computation_utils
from trax.rl import distributions
from trax.rl import training as rl_training


class ActorCriticTrainer(rl_training.PolicyTrainer):
  """Trains policy and value models using actor-critic methods.

  Attrs:
    on_policy (bool): Whether the algorithm is on-policy. Used in the data
      generators. Should be set in derived classes.
  """

  on_policy = None

  def __init__(self, task,
               value_model=None,
               value_optimizer=None,
               value_lr_schedule=lr.MultifactorSchedule,
               value_batch_size=64,
               value_train_steps_per_epoch=500,
               n_shared_layers=0,
               added_policy_slice_length=0,
               **kwargs):  # Arguments of PolicyTrainer come here.
    """Configures the actor-critic Trainer.

    Args:
      task: RLTask instance to use
      value_model: the model to use for the value function
      value_optimizer: the optimizer to train the value model
      value_lr_schedule: lr schedule for value model training
      value_batch_size: batch size for value model training
      value_train_steps_per_epoch: how many steps are we using to
        train the value model in each epoch
      n_shared_layers: how many layers to share between value and
        policy models
      added_policy_slice_length: how much longer should slices of
        trajectories be for policy than for value training; this
        is useful for TD calculations and only affect the length
        of elements produced for policy batches; value batches
        have maximum length set by max_slice_length in **kwargs
     **kwargs: arguments for PolicyTrainer super-class
    """
    self._n_shared_layers = n_shared_layers
    self._value_batch_size = value_batch_size
    self._value_train_steps_per_epoch = value_train_steps_per_epoch

    # The 2 below will be initalized in super.__init__ anyway, but are needed
    # to construct value batches which are needed before PolicyTrainer init
    # since policy input creation calls the value model -- hence this code.
    self._task = task
    self._max_slice_length = kwargs.get('max_slice_length', None)
    self._added_policy_slice_length = added_policy_slice_length

    # Initialize training of the value function.
    value_output_dir = kwargs.get('output_dir', None)
    if value_output_dir is not None:
      value_output_dir = os.path.join(value_output_dir, 'value')
    self._value_inputs = supervised.Inputs(
        train_stream=lambda _: self.value_batches_stream())
    self._value_trainer = supervised.Trainer(
        model=value_model,
        optimizer=value_optimizer,
        lr_schedule=value_lr_schedule,
        loss_fn=tl.L2Loss,
        inputs=self._value_inputs,
        output_dir=value_output_dir,
        metrics={'value_loss': tl.L2Loss},
        has_weights=True)
    self._value_eval_model = value_model(mode='eval')
    value_batch = next(self.value_batches_stream())
    self._value_eval_model.init(value_batch)

    # Initialize policy training.
    super(ActorCriticTrainer, self).__init__(task, **kwargs)

  def value_batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    for np_trajectory in self._task.trajectory_batch_stream(
        self._value_batch_size, max_slice_length=self._max_slice_length):
      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (np_trajectory.observations,         # Inputs to the value model.
             np_trajectory.returns[:, :, None],  # Targets: regress to returns.
             np_trajectory.mask[:, :, None])     # Mask to zero-out padding.

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TrajectoryNp and values.

    Args:
      trajectory: a TrajectoryNp, the trajectory to create inputs from
      values: a numpy array: value function computed on trajectory

    Returns:
      a tuple of numpy arrays of the form (inputs, x1, x2, ...) that will be
      passed to the policy model; policy model will compute outputs from
      inputs and (outputs, x1, x2, ...) will be passed to self.policy_loss
      which should be overridden accordingly.
    """
    return NotImplementedError

  def policy_batches_stream(self):
    """Use the RLTask self._task to create inputs to the policy model."""
    if self.on_policy:
      epochs = [-1]
    else:
      epochs = None
    # Maximum slice length for policy is max_slice_len + the added policy len.
    max_slice_length = self._max_slice_length + self._added_policy_slice_length
    for np_trajectory in self._task.trajectory_batch_stream(
        self._policy_batch_size,
        epochs=epochs,
        max_slice_length=max_slice_length,
        include_final_state=(max_slice_length > 1)):
      value_model = self._value_eval_model
      value_model.weights = self._value_trainer.model_weights
      values = value_model(np_trajectory.observations, n_accelerators=1)
      shapes.assert_shape_equals(
          values, (self._policy_batch_size, max_slice_length, 1))
      values = np.squeeze(values, axis=2)  # Remove the singleton depth dim.
      yield self.policy_inputs(np_trajectory, values)

  def train_epoch(self):
    """Trains RL for one epoch."""
    self._value_trainer.train_epoch(self._value_train_steps_per_epoch, 1)
    if self._n_shared_layers > 0:  # Copy value weights to policy trainer.
      _copy_model_weights(0, self._n_shared_layers,
                          self._value_trainer, self._policy_trainer)
    self._policy_trainer.train_epoch(self._policy_train_steps_per_epoch, 1)
    if self._n_shared_layers > 0:  # Copy policy weights to value trainer.
      _copy_model_weights(0, self._n_shared_layers,
                          self._policy_trainer, self._value_trainer)


def _copy_model_weights(start, end, from_trainer, to_trainer,  # pylint: disable=invalid-name
                        copy_optimizer_slots=True):
  """Copy model weights[start:end] from from_trainer to to_trainer."""
  from_weights = from_trainer.model_weights
  to_weights = to_trainer.model_weights
  shared_weights = from_weights[start:end]
  to_weights[start:end] = shared_weights
  to_trainer.model_weights = to_weights
  if copy_optimizer_slots:
    # TODO(lukaszkaiser): make a nicer API in Trainer to support this.
    # Currently we use the hack below. Note [0] since that's the model w/o loss.
    # pylint: disable=protected-access
    from_slots = from_trainer._opt_state.slots[0][start:end]
    to_slots = to_trainer._opt_state.slots[0]
    # The lines below do to_slots[start:end] = from_slots, but on tuples.
    new_slots = to_slots[:start] + from_slots[start:end] + to_slots[end:]
    new_slots = tuple([new_slots] + list(to_trainer._opt_state.slots[1:]))
    to_trainer._opt_state = to_trainer._opt_state._replace(slots=new_slots)
    # pylint: enable=protected-access


class AWRTrainer(ActorCriticTrainer):
  """Trains policy and value models using AWR."""

  on_policy = False

  def __init__(self, task, beta=1.0, w_max=20.0, **kwargs):
    """Configures the AWR Trainer."""
    self._beta = beta
    self._w_max = w_max
    super(AWRTrainer, self).__init__(task, **kwargs)

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TrajectoryNp and values."""
    # How much TD to use is determined by the added policy slice length,
    # as the policy batches need to be this much longer to calculate TD.
    td = self._added_policy_slice_length
    advantage = computation_utils.calculate_advantage(
        trajectory.rewards, trajectory.returns, values, self._task.gamma, td)
    awr_weights = np.minimum(np.exp(advantage / self._beta), self._w_max)
    # Observations should be the same length as awr_weights - so if we are
    # using td_advantage, we need to cut td-man out from the end.
    obs = trajectory.observations
    obs = obs[:, :-td] if td > 0 else obs
    act = trajectory.actions
    act = act[:, :-td] if td > 0 else act
    assert len(awr_weights.shape) == 2  # [batch_size, length]
    assert act.shape[0:2] == awr_weights.shape
    assert obs.shape[0:2] == awr_weights.shape
    return (obs, act, awr_weights)

  @property
  def policy_loss(self):
    """Policy loss."""
    return functools.partial(
        distributions.LogLoss, distribution=self._policy_dist)


class AdvantageActorCriticTrainer(ActorCriticTrainer):
  """The Advantage Actor Critic Algorithm aka A2C.

  Trains policy and value models using the A2C algortithm.
  """

  on_policy = True

  def __init__(self, task, **kwargs):
    """Configures the a2c Trainer."""
    super(AdvantageActorCriticTrainer, self).__init__(task, **kwargs)

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TrajectoryNp and values."""
    td = self._added_policy_slice_length
    advantage = computation_utils.calculate_advantage(
        trajectory.rewards, trajectory.returns, values, self._task.gamma, td)
    return (
        trajectory.observations,
        trajectory.actions,
        advantage)

  @property
  def policy_loss(self):
    """Policy loss."""
    return functools.partial(
        distributions.LogLoss, distribution=self._policy_dist)


@tl.layer(n_in=4, n_out=1)
def PPOLoss(x, distribution, epsilon, **unused_kwargs):
  """Definition of the Proximal Policy Optimization loss."""
  (dist_inputs, actions, advantages, old_log_probs) = x
  new_log_probs = distribution.log_prob(dist_inputs, actions)

  # Old log probs have an undesirable extra dimension which we remove here
  old_log_probs = old_log_probs.squeeze(axis=-1)

  # The ratio between new_probs and old_probs expressed
  # using log_probs and exponentaion
  probs_ratio = jnp.exp(new_log_probs - old_log_probs)
  unclipped_objective = probs_ratio * advantages
  clipped_objective = jnp.clip(probs_ratio,
                               1 - epsilon,
                               1 + epsilon)
  ppo_objective = jnp.minimum(unclipped_objective, clipped_objective)

  return -ppo_objective.mean()


class PPOTrainer(ActorCriticTrainer):
  """The Proximal Policy Optimization Algorithm aka PPO.

  Trains policy and value models using the PPO algortithm.
  """

  on_policy = True

  def __init__(self, task, epsilon=0.1, **kwargs):
    """Configures the PPO Trainer."""
    self._epsilon = epsilon
    super(PPOTrainer, self).__init__(task, **kwargs)

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TrajectoryNp and values."""
    td = self._added_policy_slice_length
    advantage = computation_utils.calculate_advantage(
        trajectory.rewards, trajectory.returns, values, self._task.gamma, td)

    return (
        trajectory.observations,
        trajectory.actions,
        advantage,
        trajectory.log_probs)

  @property
  def policy_loss(self):
    """Policy loss."""
    return functools.partial(
        PPOLoss, distribution=self._policy_dist, epsilon=self._epsilon
    )


# The DirectAdvantageActorCriticTrainerLoss and
# the following DirectAdvantageActorCriticTrainer are
# intended to show a direct definition of
# one of the simplest actor-critic losses and trainers.
@tl.layer(n_in=3, n_out=1)
def DirectAdvantageActorCriticTrainerLoss(x, **kwargs):
  """Definition of the Advantage Actor Critic loss."""
  del kwargs
  (new_log_probs, actions, advantages) = x

  flat_actions = np.reshape(actions, -1)
  flat_new_log_probs = np.reshape(new_log_probs, (actions.size, -1))
  action_probs = flat_new_log_probs[np.arange(actions.size),
                                    flat_actions.astype(int)]
  return -(action_probs * advantages).mean()


class DirectAdvantageActorCriticTrainer(ActorCriticTrainer):
  """A direct implementation of AdvantageActorCriticTrainer.

  Trains policy and value models using the A2C algortithm.
  """

  on_policy = True

  def __init__(self, task, **kwargs):
    """Configures the A2C Trainer."""
    super(DirectAdvantageActorCriticTrainer, self).__init__(task, **kwargs)
    assert isinstance(self._policy_dist, distributions.Categorical), (
        'This Trainer works only with categorical distributions.'
    )

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TrajectoryNp and values."""
    td = self._added_policy_slice_length
    advantage = computation_utils.calculate_advantage(
        trajectory.rewards, trajectory.returns, values, self._task.gamma, td)

    return (
        trajectory.observations,
        trajectory.actions,
        advantage)

  @property
  def policy_loss(self):
    """Policy loss."""
    return DirectAdvantageActorCriticTrainerLoss
