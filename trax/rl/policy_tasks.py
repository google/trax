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
"""Policy network training tasks.

Policy tasks encapsulate the training process of a policy network into a simple,
replaceable component. To implement a policy-based Agent using policy tasks:

  1. Subclass the base Agent class.
  2. In __init__(), initialize the policy training and evaluation tasks, and
     a trax.supervised.training.Loop instance using them.
  3. In train_epoch(), call the Loop to train the network.
  4. In policy(), call network_policy() defined in this module.
"""

import numpy as np

from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.rl import distributions
from trax.supervised import training


class PolicyTrainTask(training.TrainTask):
  """Task for policy training.

  Trains the policy based on action advantages.
  """

  def __init__(
      self,
      trajectory_batch_stream,
      optimizer,
      lr_schedule,
      policy_distribution,
      advantage_estimator,
      value_fn,
      weight_fn=(lambda x: x),
      advantage_normalization=True,
      advantage_normalization_epsilon=1e-5,
      head_selector=(),
  ):
    """Initializes PolicyTrainTask.

    Args:
      trajectory_batch_stream: Generator of trax.rl.task.TimeStepBatch.
      optimizer: Optimizer for network training.
      lr_schedule: Learning rate schedule for network training.
      policy_distribution: Distribution over actions.
      advantage_estimator: Function
        (rewards, returns, values, dones) -> advantages, created by one of the
        functions from trax.rl.advantages.
      value_fn: Function TimeStepBatch -> array (batch_size, seq_len)
        calculating the baseline for advantage calculation. Can be used to
        implement actor-critic algorithms, by substituting a call to the value
        network as value_fn.
      weight_fn: Function float -> float to apply to advantages. Examples:
        - A2C: weight_fn = id
        - AWR: weight_fn = exp
        - behavioral cloning: weight_fn(_) = 1
      advantage_normalization: Whether to normalize advantages.
      advantage_normalization_epsilon: Epsilon to use then normalizing
        advantages.
      head_selector: Layer to apply to the network output to select the value
        head. Only needed in multitask training. By default, use a no-op layer,
        signified by an empty sequence of layers, ().
    """
    self.trajectory_batch_stream = trajectory_batch_stream
    self._value_fn = value_fn
    self._advantage_estimator = advantage_estimator
    self._weight_fn = weight_fn
    self._advantage_normalization = advantage_normalization
    self._advantage_normalization_epsilon = advantage_normalization_epsilon
    self.policy_distribution = policy_distribution

    labeled_data = map(self.policy_batch, trajectory_batch_stream)
    sample_batch = self.policy_batch(
        next(trajectory_batch_stream), shape_only=True
    )
    loss_layer = distributions.LogLoss(distribution=policy_distribution)
    loss_layer = tl.Serial(head_selector, loss_layer)
    super().__init__(
        labeled_data, loss_layer, optimizer,
        sample_batch=sample_batch,
        lr_schedule=lr_schedule,
        loss_name='policy_loss',
    )

  def calculate_advantages(self, trajectory_batch, shape_only=False):
    (batch_size, seq_len) = trajectory_batch.observation.shape[:2]
    assert trajectory_batch.action.shape[:2] == (batch_size, seq_len)
    assert trajectory_batch.mask.shape == (batch_size, seq_len)
    if shape_only:
      values = np.zeros((batch_size, seq_len))
    else:
      # Compute the value, i.e. baseline in advantage computation.
      values = np.array(self._value_fn(trajectory_batch))
      assert values.shape == (batch_size, seq_len)
    # Compute the advantages using the chosen advantage estimator.
    return self._advantage_estimator(
        rewards=trajectory_batch.reward,
        returns=trajectory_batch.return_,
        dones=trajectory_batch.done,
        values=values,
        discount_mask=trajectory_batch.env_info.discount_mask,
    )

  def calculate_weights(self, advantages):
    """Calculates advantage-based weights for log loss in policy training."""
    if self._advantage_normalization:
      # Normalize advantages.
      advantages -= jnp.mean(advantages)
      advantage_std = jnp.std(advantages)
      advantages /= advantage_std + self._advantage_normalization_epsilon
    weights = self._weight_fn(advantages)
    assert weights.shape == advantages.shape
    return weights

  def trim_and_mask_batch(self, trajectory_batch, advantages):
    (batch_size, seq_len) = trajectory_batch.observation.shape[:2]
    adv_seq_len = advantages.shape[1]
    # The advantage sequence should be shorter by the margin. Margin is the
    # number of timesteps added to the trajectory slice, to make the advantage
    # estimation more accurate. adv_seq_len determines the length of the target
    # sequence, and is later used to trim the inputs and targets in the training
    # batch. Example for margin 2:
    # observations.shape == (4, 5, 6)
    # rewards.shape == values.shape == (4, 5)
    # advantages.shape == (4, 3)
    assert adv_seq_len <= seq_len
    assert advantages.shape == (batch_size, adv_seq_len)
    # Trim observations, actions and mask to match the target length.
    observations = trajectory_batch.observation[:, :adv_seq_len]
    actions = trajectory_batch.action[:, :adv_seq_len]
    mask = trajectory_batch.mask[:, :adv_seq_len]
    # Apply the control mask, so we only compute policy loss for controllable
    # timesteps.
    mask *= trajectory_batch.env_info.control_mask[:, :adv_seq_len]
    return (observations, actions, mask)

  def policy_batch(self, trajectory_batch, shape_only=False):
    """Computes a policy training batch based on a trajectory batch.

    Args:
      trajectory_batch: trax.rl.task.TimeStepBatch with a batch of trajectory
        slices. Elements should have shape (batch_size, seq_len, ...).
      shape_only: Whether to return dummy zero arrays of correct shape. Useful
        for initializing models.

    Returns:
      Triple (observations, actions, weights), where weights are the
      advantage-based weights for the policy loss. Shapes:
      - observations: (batch_size, seq_len) + observation_shape
      - actions: (batch_size, seq_len) + action_shape
      - weights: (batch_size, seq_len)
    """
    advantages = self.calculate_advantages(
        trajectory_batch, shape_only=shape_only
    )
    (observations, actions, mask) = self.trim_and_mask_batch(
        trajectory_batch, advantages
    )
    weights = self.calculate_weights(advantages) * mask / jnp.sum(mask)
    return (observations, actions, weights)


class PolicyEvalTask(training.EvalTask):
  """Task for policy evaluation."""

  def __init__(self, train_task, n_eval_batches=1, head_selector=()):
    """Initializes PolicyEvalTask.

    Args:
      train_task: PolicyTrainTask used to train the policy network.
      n_eval_batches: Number of batches per evaluation.
      head_selector: Layer to apply to the network output to select the value
        head. Only needed in multitask training.
    """
    self._train_task = train_task
    self._policy_dist = train_task.policy_distribution
    labeled_data = map(self._eval_batch, train_task.trajectory_batch_stream)
    sample_batch = self._eval_batch(
        next(train_task.trajectory_batch_stream), shape_only=True
    )
    # TODO(pkozakowski): Implement more metrics.
    metrics = {
        'policy_entropy': self.entropy_metric,
    }
    metrics.update(self.advantage_metrics)
    metrics.update(self.weight_metrics)
    metrics = {
        name: tl.Serial(head_selector, metric)
        for (name, metric) in metrics.items()
    }
    (metric_names, metric_layers) = zip(*metrics.items())
    # Select the appropriate head for evaluation.
    super().__init__(
        labeled_data, metric_layers,
        sample_batch=sample_batch,
        metric_names=metric_names,
        n_eval_batches=n_eval_batches,
    )

  def _eval_batch(self, trajectory_batch, shape_only=False):
    advantages = self._train_task.calculate_advantages(
        trajectory_batch, shape_only=shape_only
    )
    (observations, actions, mask) = self._train_task.trim_and_mask_batch(
        trajectory_batch, advantages
    )
    return (observations, actions, advantages, mask)

  @property
  def entropy_metric(self):
    def Entropy(policy_inputs, actions, advantages, mask):
      del actions, advantages, mask
      return jnp.mean(self._policy_dist.entropy(policy_inputs))
    return tl.Fn('Entropy', Entropy)

  @property
  def advantage_metrics(self):
    def make_metric(aggregate_fn):  # pylint: disable=invalid-name
      def AdvantageMetric(policy_inputs, actions, advantages, mask):
        del policy_inputs, actions, mask
        return aggregate_fn(advantages)
      return tl.Fn('AdvantageMetric', AdvantageMetric)
    return {
        'advantage_' + name: make_metric(fn) for (name, fn) in [
            ('mean', jnp.mean),
            ('std', jnp.std),
        ]
    }

  @property
  def weight_metrics(self):
    def make_metric(aggregate_fn):  # pylint: disable=invalid-name
      def WeightMetric(policy_inputs, actions, advantages, mask):
        del policy_inputs, actions, mask
        weights = self._train_task.calculate_weights(advantages)
        return aggregate_fn(weights)
      return tl.Fn('WeightMetric', WeightMetric)
    return {  # pylint: disable=g-complex-comprehension
        'weight_' + name: make_metric(fn) for (name, fn) in [
            ('mean', jnp.mean),
            ('std', jnp.std),
            ('min', jnp.min),
            ('max', jnp.max),
        ]
    }
