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
"""Simplified API (under development) for supervised learning/training in Trax.

Trax authors expect that this module will replace `trainer_lib.Trainer`.

Key classes:

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
import collections
import contextlib
import functools
import gzip as gzip_lib
import os
import pickle
import random
import sys
import time

from absl import logging
import gin
import jax
import numpy as np
import tensorflow as tf

from trax import fastmath
from trax import jaxboard
from trax import layers as tl
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.fastmath import random as jax_random


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

  def __init__(self, model, tasks, eval_model=None, eval_tasks=None,
               output_dir=None, checkpoint_at=None, eval_at=None,
               n_devices=None, random_seed=None):
    """Configures a training `Loop`, including a random initialization.

    Args:
      model: Trax layer, representing the core model to be trained. Loss
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      tasks: List of TrainTask instances, which define the training data, loss
          function, and optimizer to be used in respective tasks in this
          training loop.
      eval_model: Optional Trax layer, representing model used for evaluation,
        e.g., with dropout turned off. If None, the training model (model)
        will be used.
      eval_tasks: List of EvalTask instances or None. If None, don't do any
        evals.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both `eval_task` and `checkpoint_at` are None.
      checkpoint_at: Function (integer --> boolean) telling, for step n, whether
          that step should have its checkpoint saved. If None, the default is
          periodic checkpointing at `task.n_steps_per_checkpoint`.
      eval_at: Function (integer --> boolean) that says, for training step n,
          whether that step should run evals. If None, run when checkpointing.
      n_devices: integer or None, the number of devices for this computation.
      random_seed: the random seed to use; time/os dependent if None (default).
    """
    self._is_chief, self._n_hosts, self._n_devices, self._rng = (
        init_host_and_devices(n_devices, random_seed))

    # Handle single task case without lists too.
    if not isinstance(tasks, (list, tuple)):
      tasks = [tasks]

    assert len(tasks) == 1, 'Multitask training not supported yet.'
    task = tasks[0]
    if eval_tasks is None:
      eval_task = None
    else:
      assert len(eval_tasks) == 1, 'Multitask training not supported yet.'
      eval_task = eval_tasks[0]

    self._task = task
    self._model = model
    self._eval_model = eval_model or model
    default_at = (
        _at_step_1_and_every_nth_step(self._task.n_steps_per_checkpoint))
    if output_dir is not None:
      self._output_dir = os.path.expanduser(output_dir)
      tf.io.gfile.makedirs(self._output_dir)
    else:
      self._output_dir = None

    # Prepare training components.
    self._step = 0
    self._checkpoint_at = checkpoint_at or default_at
    self._batch_signature = shapes.signature(self._task.sample_batch)
    self._model_in_training = tl.Serial(self._model, self._task.loss_layer)

    # Initialize using the given random seed.
    # NOTE: If `random_seed` is `None` then `self._rng` will be different on
    # different hosts, leading to different weights on the different hosts.
    self._model_in_training.rng = self.new_rng()
    self._model_in_training.init(self._batch_signature)
    self._eval_model.rng = self.new_rng()
    self._eval_model.init(self._batch_signature)

    # To handle the above case (i.e. random_seed = None), we psum the weights
    # and state and average them.
    # NOTE: This adds time (how much?) so we prefer not to do it if it is
    # unnecessary, i.e. random_seed was set.
    if random_seed is None and self._n_hosts > 1:
      logging.info('Syncing weights/state across %d hosts.', self._n_hosts)

      if logging.vlog_is_on(1):
        logging.info(
            'Input training weights shape: %s',
            fastmath.nested_map(lambda x: x.shape,
                                self._model_in_training.weights))
        logging.info('Input training weights: %s',
                     self._model_in_training.weights)
        logging.info('Input training state: %s', self._model_in_training.state)
        logging.info('Input eval weights: %s', self._eval_model.weights)
        logging.info('Input eval state: %s', self._eval_model.state)

      (self._model_in_training.weights, self._model_in_training.state,
       self._eval_model.weights, self._eval_model.state) = self._unreplicate(
           _make_weights_and_state_same_across_hosts(
               self._for_n_devices(
                   (self._model_in_training.weights,
                    self._model_in_training.state, self._eval_model.weights,
                    self._eval_model.state))))

      if logging.vlog_is_on(1):
        logging.info(
            'Output training weights shape: %s',
            fastmath.nested_map(lambda x: x.shape,
                                self._model_in_training.weights))
        logging.info('Output training weights: %s',
                     self._model_in_training.weights)
        logging.info('Output training state: %s', self._model_in_training.state)
        logging.info('Output eval weights: %s', self._eval_model.weights)
        logging.info('Output eval state: %s', self._eval_model.state)

    self._task.optimizer.tree_init(self._model_in_training.weights)

    # Signature:
    # (batch, weights, state, rng) -> ((loss, state), gradients)
    self._forward_and_backward_fn = (
        fastmath.value_and_grad(
            self._model_in_training.pure_fn,
            argnums=1,  # arg1 of pure_fn: weights
            has_aux=True))  # return (loss, state), gradients

    # Signature:
    # (weights, slots), step, opt_params, batch, state, rng ->
    # (weights, slots), state, stats
    self._accelerated_update_fn = (
        _accelerate_update_fn(
            self._forward_and_backward_fn,
            self._task.optimizer,
            n_devices=self.n_devices,
            accelerate=True,
        )
    )

    # Restore from checkpoint if there is one.
    self.load_checkpoint()

    # Prepare eval components.
    if eval_task is None:
      self._eval_at = _never
    else:
      self._eval_task = eval_task
      self._eval_at = eval_at or default_at
      metric_name_lengths = [len(name) for name in self._eval_task.metric_names]
      self._rjust_len = max(
          [len(self._task.loss_layer.name)] + metric_name_lengths)
      model_with_metrics = (
          _model_with_metrics(self._eval_model, self._eval_task))
      # Keep self._eval_{weights/state} replicated.
      self._eval_weights = self._for_n_devices(
          model_with_metrics.weights[1])  # just the eval part
      self._eval_state = self._for_n_devices(
          model_with_metrics.state[1])  # just the eval part
      self._metrics_fn = _accelerate_model_with_metrics(
          model_with_metrics, self.n_devices)
      if self._output_dir is None:
        _log('Will not write evaluation metrics, because output_dir is None.')

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
    opt_params = self._task.optimizer.opt_params

    # weights, state, slots need to be replicated if needed.
    weights, state, slots, opt_params = self._for_n_devices(
        (weights, state, slots, opt_params))

    with self._open_summary_writers() as (train_summary_writer,
                                          eval_summary_writer):
      loss_acc, step_acc = 0.0, 0
      start_time = time.time()
      optimizer_metrics_acc = collections.defaultdict(float)
      for _ in range(n_steps):
        self._step += 1
        loss, weights, state, slots, optimizer_metrics = self._run_one_step(
            weights, state, slots, opt_params)

        # optimizer_metrics and loss are replicated on self.n_devices, a few
        # metrics are replicated (ex: gradients_l2, weights_l2) - i.e. they are
        # the same across devices, whereas some (ex: loss) aren't because they
        # are different on different devices (due to different data).
        # Taking the average does the correct thing in both the cases.
        #
        # NOTE: Only the weights and gradients are synced across the hosts. This
        # implies the loss here is averaged from this hosts' devices and not
        # across all hosts.
        optimizer_metrics, loss = fastmath.nested_map(
            jnp.mean, (optimizer_metrics, loss))

        loss_acc += loss
        step_acc += 1
        for metric_name, value in optimizer_metrics.items():
          optimizer_metrics_acc[metric_name] += value
        if self._checkpoint_at(self.step):
          self.save_checkpoint(weights, state, slots)
        if self._eval_at(self.step):
          elapsed_time = time.time() - start_time
          self._model_in_training.weights = weights
          self._model_in_training.state = state
          self._eval_model.weights = self._model.weights
          self._log_training_progress(
              total_loss=loss_acc, n_steps=step_acc, elapsed_time=elapsed_time,
              optimizer_metrics=optimizer_metrics_acc,
              summary_writer=train_summary_writer)
          self.run_evals(weights, state, eval_summary_writer)
          loss_acc, step_acc = 0.0, 0
          start_time = time.time()
          optimizer_metrics_acc = collections.defaultdict(float)

    # Store the final values back into their respective objects, for testing
    # or other inspection/use.

    # We keep the standard model weights/state unreplicated and
    # `tl.Accelerate(model)` will carry the replicated weights/state.
    # TODO(afrozm): Try to use `tl.Accelerate(model)` everywhere in the Loop.
    self._model_in_training.weights = self._unreplicate(weights)
    self._model_in_training.state = self._unreplicate(state)
    self._task.optimizer.slots = self._unreplicate(slots)
    self._task.optimizer.opt_params = self._unreplicate(opt_params)
    self._eval_model.weights = self._model.weights

  @property
  def step(self):
    """Returns current step number in this training session."""
    return self._step

  @property
  def n_devices(self):
    """Returns the number of devices to be used in this computation."""
    return self._n_devices

  @property
  def is_chief(self):
    """Returns true if this Loop is the chief."""
    return self._is_chief

  @property
  def model(self):
    """Returns the model that is training."""
    return self._model

  @property
  def eval_model(self):
    """Returns the model used for evaluation."""
    return self._eval_model

  def new_rng(self):
    """Returns a new single-use random number generator (JAX PRNG key)."""
    self._rng, rng = fastmath.random.split(self._rng)
    return rng

  def _for_n_devices(self, x):
    """Replicates/broadcasts `x` for n devices if `self.n_devicess > 1`."""
    return tl.for_n_devices(x, self.n_devices)

  def _unreplicate(self, x):
    if self.n_devices == 1:
      return x

    unreplicate_fn = lambda x: x[0]
    return fastmath.nested_map(unreplicate_fn, x)

  def _reshape_by_device(self, x):
    if self.n_devices == 1:
      return x
    return tl.reshape_by_device(x, self.n_devices)

  def _run_one_step(self, weights, state, slots, opt_params):
    """Updates model weights/state and optimizer slots by running one step.

    Args:
      weights: Weights from model being trained.
      state: State (non-weight parameters) from model being trained.
      slots: Updatable weights for the optimizer in this training loop.
      opt_params: Dictionary of optimizer (hyper)parameters,
        e.g. learning rate, momentum.

    Returns:
      Tuple (loss, weights, state, slots, stats) with new values from one step
      of training, where stats are current optimizer statistics.
    """
    step = self.step
    # Update the learning rate.
    opt_params['learning_rate'] = self._for_n_devices(
        self._task.learning_rate(step))

    batch = self._task.next_batch()
    # batch needs to be split across the local devices -- the difference
    # between _for_n_devices and _reshape_by_device is that the latter splits
    # the batch dim to batch // n_devices, vs _for_n_devices
    # broadcasts/replicates to n_devices dimension.
    batch = self._reshape_by_device(batch)

    rng = self.new_rng()
    if self.n_devices > 1:
      rng = jnp.stack(jax_random.split(rng, self.n_devices))

    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      # Prints every power of two, if debugging is enabled.
      logging.info('step[%d]', step)
      # logging.info('batch[%s]', batch)
      logging.info('opt_params[%s]', opt_params)
      logging.info('weights[%s]', weights)

    # NOTE: stats is a replicated dictionary of key to jnp arrays.
    (weights, slots), state, stats = (
        self._accelerated_update_fn(
            (weights, slots), step, opt_params, batch, state, rng)
        )

    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      logging.info('updated weights[%s]', weights)
      logging.info('stats[%s]', stats)

    return stats['loss'], weights, state, slots, stats

  def _log_training_progress(self, total_loss, n_steps, elapsed_time,
                             optimizer_metrics, summary_writer):
    """Logs training related metrics.

    Logs:
     * current learning rate,
     * steps per second,
     * average training loss,
     * average metrics returned from the optimizer
    to the provided summary writer. Training loss is also logged to stdout.

    Args:
      total_loss: Total training loss accumulated over n_steps training steps.
      n_steps: Number of steps over which the metrics were accumulated.
      elapsed_time: Time of execusion of n_steps training steps.
      optimizer_metrics: Dict from optimizer metric name to metric values.
      summary_writer: Jaxboard summary writer for saving provided metrics.
    """
    loss_name = self._task.loss_layer.name
    # only here do avoid potential divide-by-0
    n_steps = max(1, n_steps)
    _log('')  # Separator for visibility on terminals.
    self._log_step('Ran %d train steps in %0.2f secs' % (n_steps, elapsed_time))
    self._log_scalars(
        {loss_name: total_loss / float(n_steps)},
        summary_writer, 'metrics/', 'train')
    if self.step == 1:
      self._save_gin(summary_writer)
    train_parameters = {
        'learning_rate': self._task.learning_rate(self.step),
        'steps per second': n_steps / elapsed_time,
    }
    # Average optimizer_metrics over n_steps.
    optimizer_metrics = {k: v / n_steps for k, v in optimizer_metrics.items()}
    train_parameters.update(optimizer_metrics)
    self._log_scalars(
        train_parameters, summary_writer, 'training/', 'train', stdout=False)

  def _save_gin(self, summary_writer=None):
    """"Saves the operative gin config."""
    if not self.is_chief:
      return
    assert self._output_dir is not None
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    if summary_writer is not None:
      summary_writer.text('gin_config',
                          jaxboard.markdownify_operative_config_str(config_str))

  # TODO(afrozm): Fix multi-host evals, right now the reported numbers in the
  #   summary writer are only from the chief and not averaged across hosts.
  def run_evals(self, weights=None, state=None, summary_writer=None):
    """Runs and records evals for this training session.

    Args:
      weights: Current weights from model in training.
      state: Current state from model in training.
      summary_writer: Jaxboard summary writer to log metrics.
    """

    # If weights and state are provided, they are used as is, otherwise we get
    # them from the training model (they are stored unreplicated) and replicate
    # them. Replication will only happen if necessary i.e. self.n_devices > 1.
    weights = (
        weights if weights is not None else self._for_n_devices(
            self._model_in_training.weights))
    state = (
        state if state is not None else self._for_n_devices(
            self._model_in_training.state))

    # From the above weights and state, create the weights and state of the
    # eval model.
    model_weights = weights[0]  # exclude weights from the loss layer
    model_state = state[0]  # exclude state from the loss layer

    # self._eval_{weights/state} are already replicated.
    metrics_weights = (model_weights, self._eval_weights)
    metrics_state = (model_state, self._eval_state)

    eval_task = self._eval_task
    n_batches = eval_task.n_eval_batches
    n_metrics = len(eval_task.metrics)
    sums = np.zeros((n_metrics,))
    for _ in range(n_batches):
      rng = self.new_rng()
      batch = eval_task.next_batch()
      metric_values, _ = (
          self._metrics_fn(batch, metrics_weights, metrics_state, rng))
      sums += metric_values
    averages = sums / n_batches
    all_metrics = dict(zip(eval_task.metric_names, averages))
    self._log_scalars(all_metrics, summary_writer, 'metrics/', 'eval')

  def _log_scalars(self, scalars, summary_writer, scalar_prefix, log_prefix,
                   stdout=True):
    """Logs and saves provided metrics.

    Args:
      scalars: Dict from metric name to metric value.
      summary_writer: Jaxboard summary writer.
      scalar_prefix: String appended in front of summary_writer entries.
      log_prefix: String appended in front of logs.
      stdout: Boolean saying if logs should be logged to stdout as well.
    """
    should_write_summaries = self.is_chief and summary_writer is not None
    for name, value in scalars.items():
      self._log_step(
          '%s %s | % .8f' %
          (log_prefix.ljust(5), name.rjust(self._rjust_len), value),
          stdout=stdout)
      if should_write_summaries:
        full_name = scalar_prefix + name
        summary_writer.scalar(full_name, value, self.step)
    if should_write_summaries:
      summary_writer.flush()

  def _log_step(self, msg, stdout=True):
    """Logs message, labeled with the current training step number."""
    _log('Step % 6d: %s' % (self.step, msg), stdout=stdout)

  def save_checkpoint(self, weights=None, state=None, slots=None):
    """Saves checkpoint to disk for the current training step.

    Args:
      weights: Weights from model being trained.
      state: State (non-weight parameters) from model being trained.
      slots: Updatable weights for the optimizer in this training loop.
    """
    if not self.is_chief:
      return
    if self._output_dir is None:
      _log('Did not save checkpoint as output_dir is None', stdout=False)
      return
    weights = self._model_in_training.weights if weights is None else weights
    state = self._model_in_training.state if state is None else state
    slots = self._task.optimizer.slots if slots is None else slots
    flat_weights, flat_state = tl.flatten_weights_and_state(weights, state)
    d = {
        'step': self.step,
        'flat_weights': flat_weights,
        'flat_state': flat_state,
        'slots': slots,
        'input_signature': self._batch_signature,
        'version_timestamp': 'Jun-29-2020'  # To update in the future if needed.
    }
    ckpt_file = os.path.join(self._output_dir, 'model.pkl.gz')
    pickle_to_file(d, ckpt_file, gzip=True)

  def load_checkpoint(self, directory=None, filename=None):
    """Loads model weights and step from a checkpoint on disk.

    Args:
      directory: Directory with the checkpoint (self._output_dir by default).
      filename: Checkpoint file name (model.pkl.gz by default).
    """
    directory = directory or self._output_dir
    if directory is None:
      _log('Not loading as both directory and output_dir are None.',
           stdout=False)
      return
    filename = filename or 'model.pkl.gz'
    path = os.path.join(directory, filename)
    if not tf.io.gfile.exists(path):
      _log(f'Not loading as checkpoint file does not exist: {path}.',
           stdout=False)
      return
    d = unpickle_from_file(path, gzip=True)
    self._step = d['step']
    self._task.optimizer.slots = d['slots']
    # TODO(lukaszkaiser): this call will read the file again, optimize it.
    self._model_in_training.init_from_file(path)
    self._eval_model.weights = self._model.weights

  @contextlib.contextmanager
  def _open_summary_writers(self):
    """Opens the Jaxboard summary writers wrapped by context manager.

    Yields:
      Tuple (train_summary_writer, eval_summary_writer) of Jaxboard summary
      writers wrapped by the GeneratorContextManager object.
      If there was no output_dir provided, yields (None, None).
    """
    if self._output_dir is not None:
      _log('Training and evaluation metrics will be written in %s.' %
           self._output_dir, stdout=False)
      train_summary_writer = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'train'))
      eval_summary_writer = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'eval'))
      try:
        yield train_summary_writer, eval_summary_writer
      finally:
        train_summary_writer.close()
        eval_summary_writer.close()
        _log('Training and evaluation metrics were written in %s.' %
             self._output_dir, stdout=False)
    else:
      yield None, None


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
  eval_data_signature = shapes.signature(eval_task.sample_batch)
  metrics_input_signature = model.output_signature(eval_data_signature)
  _, _ = metrics_layer.init(metrics_input_signature)

  # TODO(afrozm): Should we set model_with_metrics._rng, tl.Serial will assign
  #  one in any case. But its weights aren't used, so no harm in either case.
  model_with_metrics = tl.Serial(model, metrics_layer)
  return model_with_metrics


class TrainTask:
  """A supervised task (labeled data + feedback mechanism) for training."""

  def __init__(self, labeled_data, loss_layer, optimizer, lr_schedule=None,
               n_steps_per_checkpoint=100):
    r"""Configures a training task.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ data (input value) tensors followed by 1 label (target value)
          tensor.  All tensors are NumPy ndarrays or their JAX counterparts.
      loss_layer: Layer that computes a scalar value (the "loss") by comparing
          model output :math:`\hat{y}=f(x)` to the target :math:`y`.
      optimizer: Optimizer object that computes model weight updates from
          loss-function gradients.
      lr_schedule: Learning rate schedule, a function step -> learning_rate.
      n_steps_per_checkpoint: How many steps to run between checkpoints.
    """
    self._labeled_data = labeled_data
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._lr_schedule = lr_schedule
    self._sample_batch = next(labeled_data)
    self._n_steps_per_checkpoint = n_steps_per_checkpoint

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
  def n_steps_per_checkpoint(self):
    return self._n_steps_per_checkpoint

  @property
  def optimizer(self):
    return self._optimizer

  def learning_rate(self, step):
    """Return the learning rate for the given step."""
    if self._lr_schedule is not None:
      with fastmath.use_backend('numpy'):
        return self._lr_schedule(step)
    params = self._optimizer._init_opt_params  # pylint: disable=protected-access
    return params['learning_rate']


class EvalTask:
  """Labeled data plus scalar functions for (periodically) measuring a model.

  An eval task specifies how (`labeled_data` + `metrics`) and with what
  precision (`n_eval_batches`) to measure a model as it is training.
  The variance of each scalar output is reduced by measuring over multiple
  (`n_eval_batches`) batches and reporting the average from those measurements.
  """

  def __init__(self, labeled_data, metrics,
               metric_names=None, n_eval_batches=1):
    r"""Configures an eval task: named metrics run with a given data source.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ data tensors (NumPy ndarrays) followed by 1 label (target value)
          tensor.
      metrics: List of layers; each computes a scalar value per batch by
          comparing model output :math:`\hat{y}=f(x)` to the target :math:`y`.
      metric_names: List of names, one for each item in `metrics`, in matching
           order, to be used when recording/reporting eval output. If None,
           generate default names using layer names from metrics.
      n_eval_batches: Integer N that specifies how many eval batches to run;
          the output is then the average of the outputs from the N batches.
    """
    self._labeled_data = labeled_data
    self._metrics = metrics
    self._metric_names = metric_names or self._default_names()
    self._n_eval_batches = n_eval_batches  # pylint: disable=invalid-name

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
  def metric_names(self):
    return self._metric_names

  @property
  def n_eval_batches(self):
    return self._n_eval_batches

  def _default_names(self):
    return [m.name for m in self._metrics]

  def _check_init_values(self):
    if len(self._metrics) != len(self._metric_names):
      raise ValueError(
          f'Number of metrics ({len(self._metrics)}) does not equal '
          f'number of metric names ({len(self._metric_names)}).')


def _never(*args):
  """Returns False for all step numbers."""
  del args
  return False


def _at_step_1_and_every_nth_step(period):
  """A function that's true at 1 and n when n % period == 0."""
  def _at_1_and_periodically(step_n):
    return (step_n == 1) or (step_n > 0 and (step_n % period == 0))
  return _at_1_and_periodically


def _log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)
    sys.stdout.flush()


def pickle_to_file(obj, file_path, gzip=False):
  """Pickle obj to file_path with gzipping and failure protection."""
  # Pickle to tmp file and overwrite to prevent writing partial files.
  tmp_file_path = file_path + '._tmp_'
  with tf.io.gfile.GFile(tmp_file_path, 'wb') as f:
    if not gzip:
      pickle.dump(obj, f)
    else:
      with gzip_lib.GzipFile(fileobj=f, compresslevel=2) as gzipf:
        pickle.dump(obj, gzipf)
  # Moving a file is much less error-prone than pickling large files.
  tf.io.gfile.rename(tmp_file_path, file_path, overwrite=True)


def unpickle_from_file(file_path, gzip=False):
  """Unpickle obj from file_path with gzipping."""
  with tf.io.gfile.GFile(file_path, 'rb') as f:
    if not gzip:
      obj = pickle.load(f)
    else:
      with gzip_lib.GzipFile(fileobj=f, compresslevel=2) as gzipf:
        obj = pickle.load(gzipf)
  return obj


def _init_random_number_generators(seed=None):
  """Initializes random generators for Python, NumPy, TensorFlow, and JAX."""
  # Seed Python random (None as seed is okay), then use it to seed the others.
  random.seed(seed)
  if seed is None:
    seed = random.randint(0, 2**31 - 1)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  return jax_random.get_prng(seed)


def init_host_and_devices(n_devices=None, random_seed=None):
  """Initializes host and device attributes for this trainer.

  Args:
    n_devices: Number of devices this trainer will use. If `None`, get the
        number from the backend.
    random_seed: Random seed as the starting point for all random numbers used
        by the trainer. If `None`, calculate one from system time and host id.

  Returns:
    is_chief: True if this trainer has special chief responsibilities.
    host_count: Number of hosts in this computation.
    n_devices: The passed in value of n_devices or a computed default (for this
      host).
    random_seed: The passed in value of random_seed or a computed default.
  """
  if fastmath.backend_name() == 'jax':
    host_id = jax.host_id()
    host_count = jax.host_count()
  else:
    host_id = 0
    host_count = 1
  is_chief = (host_id == 0)

  logging.info('Initializing hosts and devices: host_id %d, host_count %d, '
               'is_chief %d', host_id, host_count, is_chief)

  device_count = fastmath.device_count()
  n_devices = n_devices or device_count
  # TODO(lukaszkaiser): remove this restriction when possible.
  if n_devices != device_count and fastmath.backend_name() == 'jax':
    raise ValueError('JAX cannot work yet with n_devices != all devices: '
                     '%d != %d' % (n_devices, device_count))

  if random_seed is None and host_count > 1:
    random_seed = int(1e6 * (host_id + time.time())) % 2**32
  return (is_chief, host_count, n_devices,
          _init_random_number_generators(random_seed))


# Returns a function with the following signature:
# (weights, slots), step, opt_params, batch, state, rng ->
# (weights, slots), state, stats
def _accelerate_update_fn(forward_and_backward_fn,
                          optimizer,
                          n_devices,
                          accelerate=True):
  """Accelerate the given forward_and_backward_fn function."""
  if n_devices == 1:
    def single_device_update_fn(
        weights_and_slots, step, opt_params, batch, state, rng):
      weights, slots = weights_and_slots
      (loss, state), gradients = forward_and_backward_fn(
          batch, weights, state, rng)
      weights, slots, stats = optimizer.tree_update(
          step, gradients, weights, slots, opt_params)
      stats['loss'] = loss
      return (weights, slots), state, stats
    if accelerate:
      # TODO(afrozm): Find out the status of buffer donation on GPUs, then do
      #  donate_argnums=(0,).
      single_device_update_fn = fastmath.jit(single_device_update_fn)
    return single_device_update_fn

  # More than one device (core), i.e. all of TPU configurations etc.
  assert n_devices > 1, f'{n_devices} should be greater than 1.'

  @functools.partial(fastmath.pmap, axis_name='batch', donate_argnums=(0,))
  def _multi_device_update_fn(
      weights_and_slots, step, opt_params, batch, state, rng):
    # We assume all tensors have the first dimension = n_devices.
    weights, slots = weights_and_slots
    (loss, state), gradients = forward_and_backward_fn(
        batch, weights, state, rng)

    # gradients now need to be summed over all the devices across different host
    # machines, n_devices is only the number of devices on *this* host machine.
    gradients = fastmath.psum(gradients, 'batch')
    n_devices_total = fastmath.psum(jnp.array(1.0), 'batch')
    # Average across hosts.
    gradients = jax.tree_util.tree_map(lambda g: g / n_devices_total, gradients)

    weights, slots, stats = optimizer.tree_update(
        step, gradients, weights, slots, opt_params)
    stats['loss'] = loss
    return (weights, slots), state, stats

  def multi_device_update_fn(
      weights_and_slots, step, opt_params, batch, state, rng):
    # Need to replicate step to n_devices leading dimension.
    return _multi_device_update_fn(weights_and_slots,
                                   jnp.repeat(step, n_devices), opt_params,
                                   batch, state, rng)

  return multi_device_update_fn


def _accelerate_model_with_metrics(model_with_metrics, n_devices,
                                   accelerate=True, do_mean=True):
  if not accelerate:
    return model_with_metrics.pure_fn

  return tl.jit_forward(model_with_metrics.pure_fn, n_devices, do_mean=do_mean)


@functools.partial(fastmath.pmap, axis_name='devices', donate_argnums=(0,))
def _make_weights_and_state_same_across_hosts(weights_and_state):
  """Makes train and eval model's weights and state the same across hosts."""

  # We assume that they have been already replicated, i.e the leading axis is
  # self._n_devices

  # This is the total number of devices across all hosts.
  n_devices_total = fastmath.psum(jnp.array(1.0), 'devices')

  # This sums up the weights and state across all devices.
  # NOTE: There will not be any leading axis remaining because we psum
  # over it.
  weights_and_state = fastmath.psum(weights_and_state, 'devices')

  # We finally take the average over all devices.
  weights_and_state = jax.tree_util.tree_map(
      lambda ws: ws / n_devices_total, weights_and_state)

  return weights_and_state
