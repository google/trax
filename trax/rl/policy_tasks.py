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
"""Policy network training tasks.

Policy tasks encapsulate the training process of a policy network into a simple,
replaceable component. To implement a policy-based Agent using policy tasks:

  1. Subclass the base Agent class.
  2. In __init__(), initialize the policy training and evaluation tasks, and
     a trax.supervised.training.Loop instance using them.
  3. In train_epoch(), call the Loop to train the network.
  4. In policy(), call network_policy() defined in this module.
"""

import functools

import numpy as np

from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.rl import distributions
from trax.supervised import training


class PolicyTrainTask(training.TrainTask):
  """Task for policy training.

  Trains the policy based on action advantages.
  """

  # TODO(pkozakowski): Find a way to decrease the number of arguments.
  def __init__(
      self,
      trajectory_batch_stream,
      optimizer,
      lr_schedule,
      policy_distribution,
      advantage_estimator,
      value_fn,
      gamma,
      margin,
      weight_fn=(lambda x: x),
      advantage_normalization=True,
      advantage_normalization_epsilon=1e-5,
  ):
    """Initializes PolicyTrainTask.

    Args:
      trajectory_batch_stream: Generator of trax.rl.task.TrajectoryNp.
      optimizer: Optimizer for network training.
      lr_schedule: Learning rate schedule for network training.
      policy_distribution: Distribution over actions.
      advantage_estimator: Advantage estimator from trax.rl.advantages.
      value_fn: Function TrajectoryNp -> array (batch_size, seq_len) calculating
        the baseline for advantage calculation. Can be used to implement
        actor-critic algorithms, by substituting a call to the value network
        as value_fn.
      gamma: Discount factor.
      margin: The number of extra timesteps to use when calculating TD returns.
      weight_fn: Function float -> float to apply to advantages. Examples:
        - A2C: weight_fn = id
        - AWR: weight_fn = exp
        - behavioral cloning: weight_fn(_) = 1
      advantage_normalization: Whether to normalize advantages.
      advantage_normalization_epsilon: Epsilon to use then normalizing
        advantages.
    """
    self._value_fn = value_fn
    self._advantage_estimator = functools.partial(
        advantage_estimator, gamma=gamma, n_extra_steps=margin
    )
    self._weight_fn = weight_fn
    self._advantage_normalization = advantage_normalization
    self._advantage_normalization_epsilon = advantage_normalization_epsilon
    self.policy_distribution = policy_distribution

    labeled_data = map(self.policy_batch, trajectory_batch_stream)
    loss_layer = distributions.LogLoss(distribution=policy_distribution)
    super().__init__(
        labeled_data, loss_layer, optimizer,
        lr_schedule=lr_schedule,
    )

  def policy_batch(self, trajectory_batch):
    """Computes a policy training batch based on a trajectory batch.

    Args:
      trajectory_batch: trax.rl.task.TrajectoryNp with a batch of trajectory
        slices. Elements should have shape (batch_size, seq_len, ...).

    Returns:
      Triple (observations, actions, weights), where weights are the
      advantage-based weights for the policy loss. Shapes:
      - observations: (batch_size, seq_len) + observation_shape
      - actions: (batch_size, seq_len) + action_shape
      - weight: (batch_size, seq_len)
    """
    (batch_size, seq_len) = trajectory_batch.observations.shape[:2]
    assert trajectory_batch.actions.shape[:2] == (batch_size, seq_len)
    # Compute the value, i.e. baseline in advantage computation.
    values = np.array(self._value_fn(trajectory_batch))
    assert values.shape == (batch_size, seq_len)
    # Compute the advantages using the chosen advantage estimator.
    advantages = self._advantage_estimator(
        rewards=trajectory_batch.rewards,
        returns=trajectory_batch.returns,
        dones=trajectory_batch.dones,
        values=values,
    )
    assert advantages.shape == (batch_size, seq_len)
    if self._advantage_normalization:
      # Normalize advantages.
      advantages -= np.mean(advantages)
      advantages /= (np.std(advantages) + self._advantage_normalization_epsilon)
    # Compute advantage-based weights for the log loss in policy training.
    weights = self._weight_fn(advantages)
    assert weights.shape == (batch_size, seq_len)
    return (trajectory_batch.observations, trajectory_batch.actions, weights)


class PolicyEvalTask(training.EvalTask):
  """Task for policy evaluation."""

  def __init__(self, train_task, n_eval_batches=1):
    """Initializes PolicyEvalTask.

    Args:
      train_task: PolicyTrainTask used to train the policy network.
      n_eval_batches: Number of batches per evaluation.
    """
    self._policy_dist = train_task.policy_distribution
    # TODO(pkozakowski): Implement more metrics.
    metrics = [self.entropy_metric]
    super().__init__(
        train_task.labeled_data, metrics, n_eval_batches=n_eval_batches
    )

  @property
  def entropy_metric(self):
    def Entropy(policy_inputs, actions, weights):
      del actions
      del weights
      return jnp.mean(self._policy_dist.entropy(policy_inputs))
    return tl.Fn('Entropy', Entropy)
