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
"""Classes for supervised learning/training in Trax.

Trax provides classes for training supervised models:

  - Loop: Core training loop for an n-step training session, starting from
    random initialization.

  - TrainTask: Labeled data + feedback mechanism (loss function w/ optimizer)
    for modifying a model's weights.

  - Optimizer: How to compute model weight updates using loss-derived gradients.
    May contain state ("slots", 1-1 with model weights) that accumulates across
    training steps. (This class is defined in the optimizers package.)

  - EvalTask: How and when to measure model performance as a function of
    training step number.
"""

from absl import logging
import numpy as np

from trax import layers as tl
from trax import math
from trax import shapes


class Loop:
  """Training loop that can be run for n steps to train a supervised model.

  A typical training session randomly initializes a model and evolves its
  weights via feedback from a supervised task, looping through batches of
  labeled data. A training loop can also be configured to run periodic evals
  and save intermediate checkpoints.
  """

  def __init__(self, model, task, output_dir=None, eval_task=None,
               checkpoint_at=None):
    """Configures a training `Loop`, including a random initialization.

    Args:
      model: Trax layer.
      task: TrainTask instance.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both eval_task and checkpoint_at are None.
      eval_task: EvalTask instance or None. If None, don't do any evals.
      checkpoint_at: Function or None. Function (integer --> boolean) says,
          for step n, whether that step should have its checkpoint saved. If
          None, don't save any checkpoints.
    """
    self._model = model
    self._task = task
    self._output_dir = output_dir
    self._eval_task = eval_task
    self._checkpoint_at = checkpoint_at
    self._eval_at = None if eval_task is None else eval_task.eval_at
    self._step = None

    # TODO(jonni): Decide how to control when __init__ includes initialization.
    _, _ = model.init(task.input_signature)
    _, _ = task.optimizer.tree_init(model.weights)

  def run(self, n_steps=1):
    """Runs this training loop for n steps.

    Args:
      n_steps: Stop training after completing n steps.
    """
    model, eval_task = self._model, self._eval_task

    for step_i in range(1, n_steps + 1):
      self._step = step_i
      self._run_one_step()
      if self._eval_at is not None and self._eval_at(step_i):
        eval_task.run(model, step_i)
      if self._checkpoint_at is not None and self._checkpoint_at(step_i):
        self._save_checkpoint()

  def current_step(self):
    """Returns current step number in this training session."""
    return self._step

  def _run_one_step(self):
    """Updates model weights and optimizer slots by running one step/batch."""
    optimizer = self._task.optimizer
    # TODO(jonni): figure out why JAX tracer needs the following line.
    weights = self._model.weights
    opt_params = optimizer._init_opt_params  # pylint: disable=protected-access
    batch = self._task.next_batch()
    model_with_loss = tl.Serial(self._model, self._task.loss_layer)
    loss_as_fn_of_weights = lambda w: model_with_loss(batch, weights=w)
    gradients = math.grad(loss_as_fn_of_weights)(model_with_loss.weights)
    self._model.weights, optimizer.slots = optimizer.tree_update(
        self.current_step(), gradients, weights, optimizer.slots, opt_params)

  def _log_step(self, msg):
    """Logs message, labeled with the current training step number."""
    # TODO(jonni): Is direct print() is better for command-line use?
    logging.info(f'Step {self.current_step()}: {msg}')

  def _save_checkpoint(self):
    """Saves checkpoint to disk for the current training step."""
    raise NotImplementedError


class TrainTask:
  """A supervised task (labeled data + feedback mechanism) for training."""

  def __init__(self, labeled_data, loss_layer, optimizer):
    r"""Configures a training task.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ inputs (NumPy ndarrays) followed by an ndarray of target values.
      loss_layer: Layer that computes a scalar value (the "loss") by comparing
          model output $$\hat{y}=f(x)$$ to the target $$y$$.
      optimizer: Optimizer object that computes model weight updates from
          loss-function gradients.
    """
    self._labeled_data = labeled_data
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._input_signature = shapes.signature(self._next_input())

  @property
  def loss_layer(self):
    return self._loss_layer

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def input_signature(self):
    return self._input_signature

  def next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  def _next_input(self):
    inputs_plus_label = self.next_batch()
    inputs = inputs_plus_label[:-1]
    if not inputs:
      raise ValueError(f'Inputs is empty.')
    return inputs[0] if len(inputs) == 1 else inputs


class EvalTask:
  """Labeled data plus scalar functions for (periodically) measuring a model.

  An eval task specifies how (`labeled_data` + `metrics`) and when (`eval_at`)
  to measure a model as it is training. The variance of each scalar output is
  reduced by measuring over multiple (`eval_N`) batches and reporting the
  average from those measurements.
  """

  def __init__(self, labeled_data, metrics,
               names=None, eval_at=None, eval_N=10):
    r"""Configures an eval task: named metrics run with a given data source.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ data tensors (NumPy ndarrays) followed by 1 label (target value)
          tensor.
      metrics: List of layers; each computes a scalar value per batch by
          comparing model output $$\hat{y}=f(x)$$ to the target $$y$$.
      names: List of names, one for each item in `metrics`, in matching order,
          to be used when recording/reporting eval output. If None, generate
          default names: 'metric_0', 'metric_1', ...
      eval_at: Function (integer --> boolean) that says, for training step n,
          whether that step should run the evals. If None, run evals just once,
          on step 1.
      eval_N: Integer N that specifies how many eval batches to run; the eval
          output is then the average of the scalar outputs from the N batches.
    """
    self._labeled_data = labeled_data
    self._metrics = metrics
    self._names = names or self._default_names()
    self._eval_at = eval_at if eval_at is not None else _step_1_only
    self._eval_N = eval_N  # pylint: disable=invalid-name
    self._check_init_values()

  @property
  def eval_at(self):
    return self._eval_at

  def run(self, model, step_n=None):
    """Runs and records all the metrics in this eval task.

    Args:
      model: Layer, typically in the midst of being trained.
      step_n: Current training step number for the model. Can be `None`, e.g.,
          for a model not currently being trained.
    """
    model_with_metrics = tl.Serial(model, tl.Branch(*self._metrics))
    n_metrics = len(self._metrics)
    n_batches = self._eval_N

    sums = np.zeros((n_metrics,))
    for _ in range(self._eval_N):
      batch = self._next_batch()
      sums += model_with_metrics(batch)
    averages = sums / n_batches
    for average_value, name in zip(averages, self._names):
      self._record(average_value, name, step_n)

  def _default_names(self):
    return [f'metric_{i}' for i in range(len(self._metrics))]

  def _check_init_values(self):
    if len(self._metrics) != len(self._names):
      raise ValueError(
          f'number of metrics ({len(self._metrics)}) does not equal '
          f'number of names ({len(self._names)})')

  def _next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  def _record(self, value, name, step_n):
    logging.info(f'Eval at step {step_n}: {name} = {value}')


def _never(*args):
  """Returns False for all step numbers."""
  del args
  return False


def _step_1_only(step_n):
  """Returns true for step 1 only."""
  return step_n == 1
