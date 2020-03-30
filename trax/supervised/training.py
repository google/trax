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
  """Loop that can run for a given number of steps to train a supervised model.

  The typical supervised training process randomly initializes a model and
  updates its weights via feedback (loss-derived gradients) from a training
  task, by looping through batches of labeled data. A training loop can also
  be configured to run periodic evals and save intermediate checkpoints.

  For speed, the implementation takes advantage of JAX's composable function
  transformations (specifically, `jit` and `grad`). It creates JIT-compiled
  pure functions derived from variants of the core model; schematically:

    - training variant: jit(grad(pure_function(model+loss)))
    - evals variant: jit(pure_function(model+evals))

  In training or during evals, these variants are called with explicit
  arguments for all relevant input data, model weights/state, optimizer slots,
  and random number seeds:

    - batch: labeled data
    - model weights/state: trainable weights and input-related state (e.g., as
      used by batch norm)
    - optimizer slots: weights in the optimizer that evolve during the training
      process
    - random number seeds: JAX PRNG keys that enable high-quality, distributed,
      repeatable generation of pseudo-random numbers
  """

  def __init__(self, model, task, eval_task=None, output_dir=None,
               checkpoint_at=None):
    """Configures a training `Loop`, including a random initialization.

    Args:
      model: Trax layer, representing the core model to be trained. Loss
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      task: TrainTask instance, which defines the training data, loss function,
          and optimizer to be used in this training loop.
      eval_task: EvalTask instance or None. If None, don't do any evals.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both `eval_task` and `checkpoint_at` are None.
      checkpoint_at: Function (integer --> boolean) telling, for step n, whether
          that step should have its checkpoint saved. If None, don't save any
          checkpoints.
    """
    self._task = task
    self._model_in_training = tl.Serial(model, task.loss_layer)
    self._eval_task = eval_task
    self._output_dir = output_dir
    self._checkpoint_at = checkpoint_at or _never
    self._step = None

    batch_signature = shapes.signature(task.sample_batch)
    # Initialize the model and the optimizer; discard the return values
    # (model weights/state, optimizer slots/params), since they're available
    # from the model and optimizer objects.
    _, _ = self._model_in_training.init(batch_signature)
    _, _ = task.optimizer.tree_init(self._model_in_training.weights)

    self._gradients_and_state_fn = (
        math.jit(math.grad(self._model_in_training.pure_fn,
                           argnums=1,  # arg1 of pure_fn: weights
                           has_aux=True)))  # return (gradients, state)

    if eval_task is not None:
      model_with_metrics = _model_with_metrics(model, eval_task)
      self._eval_weights = model_with_metrics.weights[1]  # just the eval part
      self._eval_state = model_with_metrics.state[1]  # just the eval part
      self._metrics_fn = math.jit(model_with_metrics.pure_fn)

  def run(self, n_steps=1):
    """Runs this training loop for n steps.

    Optionally runs evals and saves checkpoints at specified points.

    Args:
      n_steps: Stop training after completing n steps.
    """
    # Extract key values (weights, state, slots) and update them in each loop.
    weights = self._model_in_training.weights
    state = self._model_in_training.state
    slots = self._task.optimizer.slots
    for step_i in range(1, n_steps + 1):
      self._step = step_i
      weights, state, slots = self._run_one_step(weights, state, slots)
      if self._eval_at(step_i):
        self._run_evals(weights, state)
      if self._checkpoint_at(step_i):
        self._save_checkpoint(weights, state, slots)

    # Store the final values back into their respective objects, for testing
    # or other inspection/use.
    self._model_in_training.weights = weights
    self._model_in_training.state = state
    self._task.optimizer.slots = slots

  def current_step(self):
    """Returns current step number in this training session."""
    return self._step

  def new_rng(self):
    """Returns a new single-use random number generator (JAX PRNG key)."""
    return self._model_in_training.new_rng()

  def _run_one_step(self, weights, state, slots):
    """Updates model weights/state and optimizer slots by running one step.

    Args:
      weights: Weights from model being trained.
      state: State (non-weight parameters) from model being trained.
      slots: Updatable weights for the optimizer in this training loop.

    Returns:
      Tuple (weights, state, slots) with new values from one step of training.
    """
    step = self.current_step()
    batch = self._task.next_batch()
    optimizer = self._task.optimizer
    opt_params = optimizer._init_opt_params  # pylint: disable=protected-access

    gradients, state = (
        self._gradients_and_state_fn(batch, weights, state, self.new_rng()))
    weights, slots = (
        optimizer.tree_update(step, gradients, weights, slots, opt_params))
    return weights, state, slots

  def _run_evals(self, weights, state):
    """Runs and records evals for this training session.

    Args:
      weights: Current weights from model in training.
      state: Current state from model in training.
    """
    eval_task = self._eval_task
    model_weights = weights[0]  # exclude weights from the loss layer
    model_state = state[0]  # exclude state from the loss layer
    metrics_weights = (model_weights, self._eval_weights)
    metrics_state = (model_state, self._eval_state)

    n_batches = eval_task._eval_N  # pylint: disable=protected-access
    n_metrics = len(eval_task.metrics)
    sums = np.zeros((n_metrics,))
    for _ in range(n_batches):
      rng = self.new_rng()
      batch = eval_task.next_batch()
      metric_values, _ = (
          self._metrics_fn(batch, metrics_weights, metrics_state, rng))
      sums += metric_values
    averages = sums / n_batches
    for name, average_value in zip(eval_task.names, averages):
      logging.info('Eval at step %d: %s = %f',
                   self.current_step(), name, average_value)

  def _eval_at(self, step_n):
    """Returns True for training step n if evals should be run for that step."""
    return self._eval_task is not None and self._eval_task.eval_at(step_n)

  def _log_step(self, msg):
    """Logs message, labeled with the current training step number."""
    # TODO(jonni): Is direct print() is better for command-line use?
    logging.info('Step %d: %s', self.current_step(), msg)

  def _save_checkpoint(self, weights, state, slots):
    """Saves checkpoint to disk for the current training step.

    Args:
      weights: Weights from model being trained.
      state: State (non-weight parameters) from model being trained.
      slots: Updatable weights for the optimizer in this training loop.
    """
    raise NotImplementedError


def _model_with_metrics(model, eval_task):
  """Returns a model+metrics layer built on an already initialized model.

  Args:
    model: Layer with initialized weights and state.
    eval_task: EvalTask instance.

  Returns:
    An initialized, combined model+metrics layer, preserving the initialization
    of `model`.
  """
  # TODO(jonni): Redo this function as part of an initialization refactor?
  metrics_layer = tl.Branch(*eval_task.metrics)
  data_signature = shapes.signature(eval_task.sample_batch[:-1])
  label_signature = shapes.signature(eval_task.sample_batch[-1])
  metrics_input_signature = (
      shapes.splice_signatures(model.output_signature(data_signature),
                               label_signature))
  _, _ = metrics_layer.init(metrics_input_signature)

  model_with_metrics = tl.Serial(model, metrics_layer)
  model_with_metrics._rng = model.new_rng()  # pylint: disable=protected-access
  return model_with_metrics


class TrainTask:
  """A supervised task (labeled data + feedback mechanism) for training."""

  def __init__(self, labeled_data, loss_layer, optimizer):
    r"""Configures a training task.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ data (input value) tensors followed by 1 label (target value)
          tensor.  All tensors are NumPy ndarrays or their JAX counterparts.
      loss_layer: Layer that computes a scalar value (the "loss") by comparing
          model output $$\hat{y}=f(x)$$ to the target $$y$$.
      optimizer: Optimizer object that computes model weight updates from
          loss-function gradients.
    """
    self._labeled_data = labeled_data
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._sample_batch = next(labeled_data)

  @property
  def labeled_data(self):
    return self._labeled_data

  @property
  def sample_batch(self):
    return self._sample_batch

  def next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  @property
  def loss_layer(self):
    return self._loss_layer

  @property
  def optimizer(self):
    return self._optimizer


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

    self._sample_batch = next(labeled_data)
    self._check_init_values()

  @property
  def labeled_data(self):
    return self._labeled_data

  @property
  def sample_batch(self):
    return self._sample_batch

  def next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  @property
  def metrics(self):
    return self._metrics

  @property
  def names(self):
    return self._names

  @property
  def eval_at(self):
    return self._eval_at

  def _default_names(self):
    return [f'metric_{i}' for i in range(len(self._metrics))]

  def _check_init_values(self):
    if len(self._metrics) != len(self._names):
      raise ValueError(
          f'Number of metrics ({len(self._metrics)}) does not equal '
          f'number of names ({len(self._names)}).')


def _never(*args):
  """Returns False for all step numbers."""
  del args
  return False


def _step_1_only(step_n):
  """Returns true for step 1 only."""
  return step_n == 1
