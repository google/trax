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
"""Value network training tasks."""

import copy

import numpy as np

from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.supervised import training


class ValueTrainTask(training.TrainTask):
  """Task for value training."""

  def __init__(
      self,
      trajectory_batch_stream,
      optimizer,
      lr_schedule,
      advantage_estimator,
      model,
      target_model=None,
      target_scale=1.0,
      sync_at=(lambda step: step % 100 == 0),
      loss_layer=None,
      head_selector=(),
  ):
    """Initializes ValueTrainTask.

    Args:
      trajectory_batch_stream: Generator of trax.rl.task.TimeStepBatch.
      optimizer: Optimizer for network training.
      lr_schedule: Learning rate schedule for network training.
      advantage_estimator: Function
        (rewards, returns, values, dones) -> advantages, created by one of the
        functions from trax.rl.advantages.
      model: Model being trained, used to synchronize weights of the target
        model.
      target_model: Model for calculating TD targets. If `None`, use `model`.
      target_scale: Multiplier for the targets. Useful for rescaling the targets
        to a more reasonable range for model training.
      sync_at: Function step -> bool, indicating when to synchronize the target
        network with the trained network. This is necessary for training the
        network on bootstrapped targets, e.g. using TD-k.
      loss_layer: The value loss layer. The default is L2 loss.
      head_selector: Layer to apply to the network output to select the value
        head. Only needed in multitask training.
    """
    self._trajectory_batch_stream = trajectory_batch_stream
    self._advantage_estimator = advantage_estimator
    self._target_scale = target_scale

    self._synced = False
    def sync_also_on_initial_batch(step):
      return sync_at(step) or not self._synced
    self._sync_at = sync_also_on_initial_batch

    self._head_selector = head_selector

    def attach_head(model):
      return tl.Serial(model, self._head_selector)
    self._train_model = attach_head(model)
    if target_model is None:
      target_model = model
    # TODO(pkozakowski): Use target_model.clone() once it's implemented.
    self._target_model = attach_head(copy.deepcopy(target_model))

    # Count the steps, so we know when to synchronize the target network.
    self._step = 0
    def labeled_data():
      for trajectory_batch in self._trajectory_batch_stream:
        self._step += 1
        yield self.value_batch(trajectory_batch)
    sample_batch = self.value_batch(
        next(trajectory_batch_stream), shape_only=True
    )
    if loss_layer is None:
      loss_layer = tl.L2Loss()
    loss_layer = tl.Serial(head_selector, loss_layer)
    super().__init__(
        labeled_data(), loss_layer, optimizer,
        sample_batch=sample_batch,
        lr_schedule=lr_schedule,
        loss_name='value_loss',
    )

  @property
  def trajectory_batch_stream(self):
    return self._trajectory_batch_stream

  def _sync_target_model(self):
    self._target_model.weights = self._train_model.weights
    self._target_model.state = self._train_model.state
    self._synced = True

  def value_batch(self, trajectory_batch, shape_only=False):
    """Computes a value training batch based on a trajectory batch.

    Args:
      trajectory_batch: trax.rl.task.TimeStepBatch with a batch of trajectory
        slices. Elements should have shape (batch_size, seq_len, ...).
      shape_only: Whether to return dummy zero arrays of correct shape. Useful
        for initializing models.

    Returns:
      Triple (observations, targets, weights), where targets are the target
      values for network training and weights are used for masking in loss
      computation. Shapes:
      - observations: (batch_size, seq_len) + observation_shape
      - targets: (batch_size, seq_len)
      - weights: (batch_size, seq_len)
    """
    if self._sync_at(self._step) and not shape_only:
      self._sync_target_model()

    (batch_size, seq_len) = trajectory_batch.observation.shape[:2]
    assert trajectory_batch.action.shape[:2] == (batch_size, seq_len)
    assert trajectory_batch.mask.shape == (batch_size, seq_len)
    # Compute the value from the target network.
    values = np.array(self.value(trajectory_batch, shape_only=shape_only))
    assert values.shape == (batch_size, seq_len)
    # Compute the advantages - the TD errors of the target network.
    advantages = self._advantage_estimator(
        rewards=trajectory_batch.reward,
        returns=trajectory_batch.return_,
        dones=trajectory_batch.done,
        values=values,
        discount_mask=trajectory_batch.env_info.discount_mask,
    )
    adv_seq_len = advantages.shape[1]
    # The advantage sequence should be shorter by the margin. For more details,
    # see the comment in policy_tasks.PolicyTrainTask.policy_batch.
    assert adv_seq_len <= seq_len
    assert advantages.shape == (batch_size, adv_seq_len)
    # Compute the targets based on the target values and their TD errors. The
    # network gives perfect predictions when targets == values, so the
    # advantages are zero.
    targets = (values[:, :adv_seq_len] + advantages) * self._target_scale
    # Trim observations and the mask to match the target length.
    observations = trajectory_batch.observation[:, :adv_seq_len]
    mask = trajectory_batch.mask[:, :adv_seq_len]
    # Add a singleton depth dimension to the targets and the mask.
    targets = targets[:, :, None]
    mask = mask[:, :, None]
    return (observations, targets, mask)

  def value(self, trajectory_batch, shape_only=False):
    """Computes values of states in a given batch of trajectory slices.

    Can be passed as value_fn to PolicyTrainTask to implement a critic baseline
    for advantage calculation.

    Args:
      trajectory_batch: Batch of trajectory slices to compute values for.
      shape_only: Whether to return dummy zero arrays of correct shape. Useful
        for initializing models.

    Returns:
      Array of values of all states in `trajectory_batch`.
    """
    if shape_only:
      # The target model hasn't been initialized yet, and we are asked for the
      # initial, sample batch. Only shape matters here, so just return zeros.
      return np.zeros(trajectory_batch.observation.shape[:2])

    if not self._synced:
      self._sync_target_model()

    values = self._target_model(trajectory_batch.observation)
    # Squeeze the singleton depth axis.
    return np.squeeze(values, axis=-1) / self._target_scale


class ValueEvalTask(training.EvalTask):
  """Task for value evaluation."""

  def __init__(self, train_task, n_eval_batches=1, head_selector=()):
    """Initializes ValueEvalTask.

    Args:
      train_task: ValueTrainTask used to train the policy network.
      n_eval_batches: Number of batches per evaluation.
      head_selector: Layer to apply to the network output to select the value
        head. Only needed in multitask training.
    """
    labeled_data = map(
        train_task.value_batch, train_task.trajectory_batch_stream
    )
    metrics = [tl.L2Loss(), self.l1_loss]
    metric_names = ['value_l2', 'value_l1']
    # Select the appropriate head for evaluation.
    metrics = [tl.Serial(head_selector, metric) for metric in metrics]
    super().__init__(
        labeled_data, metrics,
        sample_batch=train_task.sample_batch,
        metric_names=metric_names,
        n_eval_batches=n_eval_batches,
    )

  @property
  def l1_loss(self):
    def loss(values, targets, weights):
      return jnp.sum(jnp.abs(values - targets) * weights) / jnp.sum(weights)
    return tl.Fn('L1Loss', loss)
