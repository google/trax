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
import psutil
import tensorflow as tf

from trax import fastmath
from trax import jaxboard
from trax import layers as tl
from trax import optimizers
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.fastmath import random as jax_random


_Evaluator = collections.namedtuple(
    '_Evaluator', ['weights', 'state', 'metrics_fn']
)


class Loop:
  """Loop that can run for a given number of steps to train a supervised model.

  Can train the model on multiple tasks by interleaving updates according to the
  which_task() argument.

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

  def __init__(
      self,
      model,
      tasks,
      eval_model=None,
      eval_tasks=None,
      output_dir=None,
      checkpoint_at=None,
      permanent_checkpoint_at=None,
      eval_at=None,
      which_task=None,
      n_devices=None,
      random_seed=None,
      loss_chunk_size=0,
      use_memory_efficient_trainer=False,
      callbacks=None,
  ):
    """Configures a training `Loop`, including a random initialization.

    Args:
      model: Trax layer, representing the core model to be trained. Loss
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      tasks: List of TrainTask instances, which define the training data, loss
          function, and optimizer to be used in respective tasks in this
          training loop. It can also be a single TrainTask instance which is
          treated in the same way as a singleton list.
      eval_model: Optional Trax layer, representing model used for evaluation,
        e.g., with dropout turned off. If None, the training model (model)
        will be used.
      eval_tasks: List of EvalTask instances which define how to evaluate
        the model: which validation data to use and which metrics to report.
        Evaluation on each of the tasks and will run and be reported separately
        which allows to score a model on different subtasks. This argument can
        also be None, in which case no evals will be run, or a single
        EvalTask, which wil be treated in the same way as a singleton list.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both `eval_task` and `checkpoint_at` are None.
      checkpoint_at: Function (integer --> boolean) telling, for step n, whether
          that step should have its checkpoint saved. If None, the default is
          periodic checkpointing at `task.n_steps_per_checkpoint`.
      permanent_checkpoint_at: Function (integer --> boolean) telling,
          for step n, whether that step should have its checkpoint saved
          permanently. If None, the default is periodic checkpointing at
          `task.n_steps_per_permanent_checkpoint`.
      eval_at: Function (integer --> boolean) that says, for training step n,
          whether that step should run evals. If None, run when checkpointing.
      which_task: Function (integer --> integer) indicating which task should be
          used at which training step. Can be set to None in single-task
          training.
      n_devices: integer or None, the number of devices for this computation.
      random_seed: the random seed to use; time/os dependent if None (default).
      loss_chunk_size: int, if > 0 use chunks of this size to make loss
        computation more more memory-efficient.
      use_memory_efficient_trainer: whether to use a special memory-efficient
        trainer.
      callbacks: List of subclasses of StepCallback to call on training
        steps.
    """
    self._is_chief, self._n_hosts, self._n_devices, self._rng = (
        init_host_and_devices(n_devices, random_seed))

    # Handle single task case without lists too.
    if not isinstance(tasks, (list, tuple)):
      tasks = [tasks]

    if not tasks:
      raise ValueError('Must provide at least one training task.')
    if eval_tasks is None:
      eval_tasks = []
      eval_at = _never
    else:
      if not isinstance(eval_tasks, (list, tuple)):
        eval_tasks = [eval_tasks]

    self._tasks = tasks
    self._model = model
    self._eval_model = eval_model or model

    self._use_memory_efficient_trainer = use_memory_efficient_trainer
    self._loss_chunk_size = loss_chunk_size
    # TODO(lukaszkaiser): can we have different eval models and save memory?
    if use_memory_efficient_trainer:
      assert len(tasks) == 1, 'only single task supported for now'
      assert len(eval_tasks) < 2, 'a most 1 eval task supported for now'
      self._eval_model = model

    default_at = _at_step_1_and_every_nth_step(tasks[0].n_steps_per_checkpoint)
    permanent_default_at = _at_step_1_and_every_nth_step(
        tasks[0].n_steps_per_permanent_checkpoint)
    if output_dir is not None:
      self._output_dir = os.path.expanduser(output_dir)
      tf.io.gfile.makedirs(self._output_dir)
    else:
      self._output_dir = None

    # Prepare training components.
    self._step = 0
    self._checkpoint_at = checkpoint_at or default_at
    self._permanent_checkpoint_at = (
        permanent_checkpoint_at or permanent_default_at)
    if which_task is None:
      if len(tasks) > 1:
        raise ValueError('Must provide which_task for multitask training.')
      which_task = lambda _: 0
    self._which_task = which_task

    # Initialize using the given random seed.
    # NOTE: If `random_seed` is `None` then `self._rng` will be different on
    # different hosts, leading to different weights on the different hosts.
    self._batch_signature = shapes.signature(tasks[0].sample_batch)
    self._model.rng = self.new_rng()
    # In the memory-efficient case, we initialize in init_trainer.
    if not use_memory_efficient_trainer:
      if _is_uninitialized(self._model):
        self._model.init(self._batch_signature)
      self._eval_model.rng = self.new_rng()
      if _is_uninitialized(self._eval_model):
        self._eval_model.init(self._batch_signature)

    # To handle the above case (i.e. random_seed = None), we psum the weights
    # and state and average them.
    # NOTE: This adds time (how much?) so we prefer not to do it if it is
    # unnecessary, i.e. random_seed was set.
    # NOTE: Averaging the weights across devices can screw up the initial weight
    # statistics.
    # TODO(pkozakowski): Broadcast from one of the devices instead?
    # TODO(lukaszkaiser): make it work for the memory-efficient trainer too.
    if (random_seed is None and self._n_hosts > 1 and
        not use_memory_efficient_trainer):
      logging.info('Syncing weights/state across %d hosts.', self._n_hosts)
      self._sync_weights_and_state_across_hosts()

    # Create the optimizer for the training loss function.
    self._trainer_per_task = tuple(self._init_trainer(task) for task in tasks)
    self.load_checkpoint()

    # Prepare eval components.
    self._eval_at = eval_at or default_at
    self._eval_tasks = eval_tasks
    loss_names = [task.loss_layer.name for task in self._tasks]
    metric_names = [
        name  # pylint: disable=g-complex-comprehension
        for eval_task in self._eval_tasks
        for name in eval_task.metric_names
    ]
    self._rjust_len = max(map(len, loss_names + metric_names))
    self._evaluator_per_task = tuple(
        self._init_evaluator(eval_task) for eval_task in self._eval_tasks)

    if self._output_dir is None:
      _log('Will not write evaluation metrics, because output_dir is None.')

    def task_output_dir(task_index, task_list):
      if self._output_dir is not None:
        if len(task_list) < 2:
          output_dir = self._output_dir
        else:
          output_dir = os.path.join(self._output_dir, str(task_index))
        tf.io.gfile.makedirs(output_dir)
        return output_dir
      else:
        return None
    self._output_dir_per_eval_task = [
        task_output_dir(i, eval_tasks) for i in range(len(eval_tasks))]
    self._output_dir_per_train_task = [
        task_output_dir(i, tasks) for i in range(len(tasks))]

    callbacks = callbacks or []
    self._callbacks = [
        callback_class(self) for callback_class in callbacks
    ]

  def _init_trainer(self, task):
    """Initializes the per-task trainer."""
    # Build the per-task model, sharing weights with other tasks.
    if not self._use_memory_efficient_trainer:
      model_in_training = _model_with_ends(
          self._model,
          [task.loss_layer],
          shapes.signature(task.sample_batch)
      )
      task.optimizer.tree_init(model_in_training.weights)
      return optimizers.Trainer(model_in_training, task.optimizer)
    # In the memory-efficient path, we initialize the model here.
    blocks, loss_layer = optimizers.trainer.extract_reversible_blocks(
        [self._model, task.loss_layer], loss_chunk_size=self._loss_chunk_size)
    rng = self._model.rng
    sig = shapes.signature(task.sample_batch)
    optimizers.trainer.init_reversible_blocks(blocks, loss_layer, sig, rng)
    # TODO(lukaszkaiser): here optimizer is a function, revisit this.
    return optimizers.ReversibleSerialTrainer(
        blocks, loss_layer, task.optimizer)

  def _init_evaluator(self, eval_task):
    """Initializes the per-task evaluator."""
    model_with_metrics = _model_with_metrics(
        self._eval_model, eval_task)
    if self._use_memory_efficient_trainer:
      return _Evaluator(
          weights=model_with_metrics.weights[1],
          state=model_with_metrics.state[1],
          metrics_fn=_accelerate_model_with_metrics(model_with_metrics, 0)
      )
    else:
      return _Evaluator(
          # Replicate the eval part of weights and state.
          weights=self._for_n_devices(model_with_metrics.weights[1]),
          state=self._for_n_devices(model_with_metrics.state[1]),
          metrics_fn=_accelerate_model_with_metrics(
              model_with_metrics, self.n_devices)
      )

  def _sync_weights_and_state_across_hosts(self):
    """Sync weights and state across all the hosts in the computation."""

    if logging.vlog_is_on(1):
      logging.debug(
          'Input training weights shape: %s',
          fastmath.nested_map(lambda x: x.shape,
                              self._model.weights))
      logging.debug('Input training weights: %s', self._model.weights)
      logging.debug('Input training state: %s', self._model.state)
      logging.debug('Input eval weights: %s', self._eval_model.weights)
      logging.debug('Input eval state: %s', self._eval_model.state)

    (self._model.weights, self._model.state,
     self._eval_model.weights, self._eval_model.state) = self._unreplicate(
         _make_weights_and_state_same_across_hosts(
             self._for_n_devices(
                 (self._model.weights, self._model.state,
                  self._eval_model.weights,
                  self._eval_model.state))))

    if logging.vlog_is_on(1):
      logging.debug(
          'Output training weights shape: %s',
          fastmath.nested_map(lambda x: x.shape, self._model.weights))
      logging.debug('Output training weights: %s', self._model.weights)
      logging.debug('Output training state: %s', self._model.state)
      logging.debug('Output eval weights: %s', self._eval_model.weights)
      logging.debug('Output eval state: %s', self._eval_model.state)

  def run(self, n_steps=1):
    """Runs this training loop for n steps.

    Optionally runs evals and saves checkpoints at specified points.

    Args:
      n_steps: Stop training after completing n steps.
    """
    with self._open_summary_writers() as (
        train_summary_writers, eval_summary_writers):
      process = psutil.Process(os.getpid())
      loss_acc, step_acc = 0.0, 0
      start_time = time.time()
      optimizer_metrics_acc = collections.defaultdict(float)
      for i in range(n_steps):
        prev_task_index = self._which_task(self._step)
        self._step += 1
        task_index = self._which_task(self._step)
        task_changed = task_index != prev_task_index
        loss, optimizer_metrics = self._run_one_step(task_index, task_changed)

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
            functools.partial(tl.mean_or_pmean, self._n_devices),
            (optimizer_metrics, loss))

        loss_acc += loss
        step_acc += 1
        for metric_name, value in optimizer_metrics.items():
          optimizer_metrics_acc[metric_name] += value

        if self._checkpoint_at(self.step):
          self.save_checkpoint()
        if self._permanent_checkpoint_at(self.step):
          self.save_checkpoint(permanent=True)
        if self._eval_at(self.step):
          logging.info('cpu memory use (MB): %.2f',
                       process.memory_info().rss / float(1024*1024))
          elapsed_time = time.time() - start_time
          self._eval_model.weights = self._model.weights
          self._log_training_progress(
              task=self._tasks[task_index],
              total_loss=loss_acc,
              n_steps=step_acc,
              elapsed_time=elapsed_time,
              optimizer_metrics=optimizer_metrics_acc,
              summary_writer=train_summary_writers[task_index],
          )
          self.run_evals(eval_summary_writers)
          loss_acc, step_acc = 0.0, 0
          start_time = time.time()
          optimizer_metrics_acc = collections.defaultdict(float)

    # Store the final values back into their respective objects, for testing
    # or other inspection/use.

    # We keep the standard model weights/state unreplicated and
    # `tl.Accelerate(model)` will carry the replicated weights/state.
    # TODO(afrozm): Try to use `tl.Accelerate(model)` everywhere in the Loop.
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
  def tasks(self):
    """Returns the training tasks."""
    return self._tasks

  @property
  def eval_model(self):
    """Returns the model used for evaluation."""
    return self._eval_model

  @property
  def eval_tasks(self):
    """Returns the evaluation tasks."""
    return self._eval_tasks

  @property
  def output_dir(self):
    """Returns the output directory."""
    return self._output_dir

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

  def _run_one_step(self, task_index, task_changed):
    """Updates model weights/state and optimizer slots by running one step.

    Args:
      task_index (int): Index of the task to train on.
      task_changed (bool): Whether the state has changed since the last step.

    Returns:
      Tuple (loss, stats) with loss value from one step
      of training and stats, the current optimizer statistics.
    """
    step = self.step
    for callback in self._callbacks:
      if callback.call_at(step):
        callback.on_step_begin(step)

    learning_rate = self._tasks[task_index].learning_rate(step)
    batch = self._tasks[task_index].next_batch()
    rng = self.new_rng()
    trainer = self._trainer_per_task[task_index]
    if task_changed:
      # Re-replicate weights and state to synchronize them between tasks.
      model = trainer.model_with_loss
      trainer.accelerated_model_with_loss.replicate_weights(model.weights)
      trainer.accelerated_model_with_loss.replicate_state(model.state)
    (loss, stats) = trainer.one_step(
        batch, rng, step=step, learning_rate=learning_rate
    )

    for callback in self._callbacks:
      if callback.call_at(step):
        callback.on_step_end(step)

    return (loss, stats)

  def _log_training_progress(self, task, total_loss, n_steps, elapsed_time,
                             optimizer_metrics, summary_writer):
    """Logs training related metrics.

    Logs:
     * current learning rate,
     * steps per second,
     * average training loss,
     * average metrics returned from the optimizer
    to the provided summary writer. Training loss is also logged to stdout.

    Args:
      task (TrainTask): The current task.
      total_loss: Total training loss accumulated over n_steps training steps.
      n_steps: Number of steps over which the metrics were accumulated.
      elapsed_time: Time of execution of n_steps training steps.
      optimizer_metrics: Dict from optimizer metric name to metric values.
      summary_writer: Jaxboard summary writer for saving provided metrics.
    """
    loss_name = task.loss_layer.name
    # only here do avoid potential divide-by-0
    n_steps = max(1, n_steps)
    _log('')  # Separator for visibility on terminals.
    if self.step == 1:
      self._log_n_weights()
    self._log_step('Ran %d train steps in %0.2f secs' % (n_steps, elapsed_time))
    self.log_summary(
        {loss_name: total_loss / float(n_steps)},
        summary_writer, 'metrics/', 'train')
    if self.step == 1:
      self._save_gin(summary_writer)
    train_parameters = {
        'learning_rate': task.learning_rate(self.step),
        'steps per second': n_steps / elapsed_time,
    }
    # Average optimizer_metrics over n_steps.
    optimizer_metrics = {k: v / n_steps for k, v in optimizer_metrics.items()}
    train_parameters.update(optimizer_metrics)
    self.log_summary(
        train_parameters, summary_writer, 'training/', 'train', stdout=False)

  def _save_gin(self, summary_writer):
    """"Saves the operative gin config."""
    if not self.is_chief or self._output_dir is None:
      return
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    if summary_writer is not None:
      summary_writer.text(
          'gin_config', jaxboard.markdownify_operative_config_str(config_str)
      )

  def _log_n_weights(self):
    """"Logs the number of weights in the training model."""
    def _size(x):
      try:
        return x.size
      except Exception:  # pylint: disable=broad-except
        return 0
    sizes = fastmath.nested_map(_size, self._model.weights)
    total_size = sum(fastmath.tree_flatten(sizes))
    self._log_step('Total number of trainable weights: %d' % total_size)

  # TODO(afrozm): Fix multi-host evals, right now the reported numbers in the
  #   summary writer are only from the chief and not averaged across hosts.
  def run_evals(self, summary_writers=None):
    """Runs and records evals for this training session.

    Args:
      summary_writers: List of per-task Jaxboard summary writers to log metrics.
    """
    if summary_writers is None:
      summary_writers = (None,) * len(self._eval_tasks)

    self._eval_model.weights = self._model.weights
    self._eval_model.state = self._model.state

    def recursively_look_for_printable_states(state):
      if isinstance(state, (tuple, list)):
        for substate in state:
          for item in recursively_look_for_printable_states(substate):
            yield item
      if isinstance(state, dict):
        for key, value in state.items():
          if isinstance(key, str) and key.startswith('summary_'):
            for device_id, device_value in enumerate(value):
              yield ('device{}/{}'.format(device_id, key[len('summary_'):]),
                     device_value)

    # The most recently trained weights are in this trainer, use those for eval.
    cur_train_task_index = self._which_task(self._step)
    trainer = self._trainer_per_task[cur_train_task_index]

    for eval_task_index in range(len(self._eval_tasks)):
      eval_task = self._eval_tasks[eval_task_index]
      evaluator = self._evaluator_per_task[eval_task_index]
      if eval_task is None:
        continue

      # Extract the actual model weights and state, excluding the loss layer.
      if self._use_memory_efficient_trainer:
        model_weights, model_state = self._model.weights, self._model.state
      else:
        model_weights = trainer.accelerated_model_with_loss.weights[0]
        model_state = trainer.accelerated_model_with_loss.state[0]

      # evaluator.{weights,state} are already replicated.
      metrics_weights = (model_weights, evaluator.weights)
      metrics_state = (model_state, evaluator.state)

      n_batches = eval_task.n_eval_batches
      n_metrics = len(eval_task.metrics)
      sums = np.zeros((n_metrics,))
      for _ in range(n_batches):
        rng = self.new_rng()
        batch = eval_task.next_batch()
        metric_values, _ = evaluator.metrics_fn(
            batch, metrics_weights, metrics_state, rng)
        sums += metric_values
      averages = sums / n_batches
      all_metrics = dict(zip(eval_task.metric_names, averages))
      summary_writer = summary_writers[eval_task_index]
      self.log_summary(all_metrics, summary_writer, 'metrics/', 'eval')
      summary_metrics = dict(recursively_look_for_printable_states(
          model_state))
      self.log_summary(summary_metrics, summary_writer, 'summary_', 'eval')

  def log_summary(self, values, summary_writer, value_prefix, log_prefix,
                  stdout=True):
    """Logs and saves provided metrics.

    Args:
      values: Dict from metric name to metric value.
      summary_writer: Jaxboard summary writer.
      value_prefix: String appended in front of summary_writer entries.
      log_prefix: String appended in front of logs.
      stdout: Boolean saying if logs should be logged to stdout as well.
    """
    should_write_summaries = self.is_chief and summary_writer is not None
    for name, value in values.items():
      full_name = value_prefix + name
      s = jnp.shape(value)
      if not s:
        self._log_step(
            '%s %s | % .8f' %
            (log_prefix.ljust(5), name.rjust(self._rjust_len), value),
            stdout=stdout)
        if should_write_summaries:
          summary_writer.scalar(full_name, value, self.step)
      else:
        if should_write_summaries:
          summary_writer.image(full_name, value, self.step)
    if should_write_summaries:
      summary_writer.flush()

  def _log_step(self, msg, stdout=True):
    """Logs message, labeled with the current training step number."""
    _log('Step % 6d: %s' % (self.step, msg), stdout=stdout)

  def save_checkpoint(self, permanent=False):
    """Saves checkpoint to disk for the current training step."""
    if not self.is_chief:
      _log('Did not save checkpoint as we are not chief.')
      return
    if self._output_dir is None:
      _log('Did not save checkpoint as output_dir is None')
      return
    if permanent:
      filename = 'model_{}.pkl.gz'.format(self.step)
    else:
      filename = 'model.pkl.gz'
    ckpt_file = os.path.join(self._output_dir, filename)
    _log('Saving checkpoint to %s.' % ckpt_file, stdout=False)
    weights = self._model.weights
    state = self._model.state
    if self._use_memory_efficient_trainer:
      slots_per_task = [trainer.slots for trainer in self._trainer_per_task]
    else:
      slots_per_task = tuple(task.optimizer.slots for task in self._tasks)
    # We only need the input signature for the body, not for the loss layers.
    # That part is the same across tasks - take it from the first one.
    input_signature = self._batch_signature[:self._model.n_in]
    flat_weights, flat_state = tl.flatten_weights_and_state(weights, state)
    _, flat_eval_state = tl.flatten_weights_and_state(
        weights, self._eval_model.state)
    if self._use_memory_efficient_trainer:
      sharded_weights_len = self._save_weights_sharded(flat_weights, ckpt_file)
      # In the main dict we just save the number of shards in place of weights.
      weights_in_dict = sharded_weights_len
    else:
      weights_in_dict = self._to_bits(flat_weights)
    d = {
        'step': self.step,
        'flat_weights': weights_in_dict,
        'flat_state': flat_state,
        'flat_eval_state': flat_eval_state,
        'slots_per_task': slots_per_task,
        'input_signature': input_signature,
        'version_timestamp': 'Oct-28-2020'  # To update in the future if needed.
    }
    pickle_to_file(d, ckpt_file, gzip=True)
    # Move sharded files to non-tmp files after all is saved.
    if self._use_memory_efficient_trainer:
      for i in range(weights_in_dict):
        fname = ckpt_file + '.shard%d' % i
        tf.io.gfile.rename(fname + '.tmp', fname, overwrite=True)
    _log('Checkpoint saved in %s.' % ckpt_file, stdout=False)

  def _save_weights_sharded(self, flat_weights, ckpt_file):
    """Saves flat_weights in a sharded way to ckpt_file.shardN.tmp."""
    # In large models, we shard weights into multiple files.
    # Otherwise using pickle can lead to running out of RAM.
    # We shard weights into parts of over 4M floats to avoid tiny files.
    max_shard_size = 4 * 1024 * 1024
    sharded_weights, current_shard, current_shard_size = [], [], 0
    for w in flat_weights:
      current_shard.append(w)
      current_shard_size += int(np.prod(w.shape))
      if current_shard_size > max_shard_size:
        sharded_weights.append(current_shard)
        current_shard, current_shard_size = [], 0
    if current_shard:  # Append the last shard if it's not empty.
      sharded_weights.append(current_shard)
    # Save weight shards to files (tmp first to be resilient to failure).
    for i, w in enumerate(sharded_weights):
      path = ckpt_file + '.shard%d.tmp' % i
      _log('Saving sharded weights to %s.' % path, stdout=False)
      pickle_to_file(self._to_bits(w), path, gzip=False)
    return len(sharded_weights)

  def _to_bits(self, weights):
    """Converts a list of weights to bit-cast weights and their types."""
    # This is currently needed to pickle bfloat16 arrays from JAX.
    # TODO(lukaszkaiser): remove once it is not needed (the following unit test
    #   checks it: training_test/test_restores_step_bfloat16).
    if not fastmath.is_backend(fastmath.Backend.JAX):
      return weights
    bits = []
    for w in weights:
      if w.dtype == jnp.bfloat16:
        bits.append((jax.lax.bitcast_convert_type(w, np.uint16), 'bfloat16'))
      else:  # for non-bfloat16 weights, be compatible with earlier checkpoints
        bits.append(w)
    return bits

  def _from_bits(self, bits_and_types):
    """Converts a list of bit-cast weights and their types back to weights."""
    # This is the reverse of _to_bits, see above for explanation.
    if not fastmath.is_backend(fastmath.Backend.JAX):
      return bits_and_types
    weights = []
    for bits_and_dtype in bits_and_types:
      if isinstance(bits_and_dtype, tuple):
        bits, dtype = bits_and_dtype
        assert dtype == 'bfloat16'
        w = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
        weights.append(w)
      else:
        weights.append(bits_and_dtype)
    return weights

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
    _log('Loading checkpoint from %s.' % path, stdout=False)
    d = unpickle_from_file(path, gzip=True)
    # For large models, load weights from sharded files.
    if self._use_memory_efficient_trainer:
      weights = []
      n_shards = d['flat_weights']  # We store the number of shards in d here.
      for i in range(n_shards):
        w = unpickle_from_file(path + '.shard%d' % i, gzip=False)
        w = self._from_bits(w)  # bit-casting may put w on accelerator, go back
        weights.extend([tl.on_cpu(x) for x in w])
      d['flat_weights'] = weights
    else:
      d['flat_weights'] = self._from_bits(d['flat_weights'])
    self._step = d['step']
    if 'slots' in d:
      if len(self._tasks) != 1:
        raise ValueError(
            'Can\'t load a single-task checkpoint into a multitask Loop.'
        )
      d['slots_per_task'] = [d['slots']]
    if self._use_memory_efficient_trainer:
      for (trainer, slots) in zip(self._trainer_per_task, d['slots_per_task']):
        trainer.slots = slots
    else:
      for (task, slots) in zip(self._tasks, d['slots_per_task']):
        task.optimizer.slots = slots
    # This is self._model.init_from_file but optimized to not re-read.
    input_signature = d['input_signature']
    weights_and_state_sig = self._model.weights_and_state_signature(
        input_signature)
    weights, state = tl.unflatten_weights_and_state(
        d['flat_weights'], d['flat_state'], weights_and_state_sig)
    self._model.state = state
    self._model.weights = weights
    self._eval_model.weights = self._model.weights
    # Restore eval model state; note: it's not always the same as train state.
    if 'flat_eval_state' in d:
      flat_eval_state = d['flat_eval_state']
    else:  # It wasn't saved in old checkpoints; remove this branch once ported.
      flat_eval_state = d['flat_state']
    _, eval_state = tl.unflatten_weights_and_state(
        d['flat_weights'], flat_eval_state, weights_and_state_sig)
    self._eval_model.state = eval_state
    _log('Checkpoint loaded from %s.' % path, stdout=False)

  @contextlib.contextmanager
  def _open_summary_writers(self):
    """Opens the Jaxboard summary writers wrapped by context manager.

    Yields:
      A pair (train_summary_writers, eval_summary_writers) of lists of
      Jaxboard summary writers wrapped in a GeneratorContextManager object.
      Elements of the lists correspond to the training and evaluation task
      directories created during initialization. If there was no output_dir
      provided, yields lists of Nones with the appropriate length.
    """
    if self._output_dir is not None:
      _log(f'Metrics will be written in {self._output_dir}.', stdout=False)
      train_writers = [jaxboard.SummaryWriter(os.path.join(output_dir, 'train'))
                       for output_dir in self._output_dir_per_train_task]
      eval_writers = [jaxboard.SummaryWriter(os.path.join(output_dir, 'eval'))
                      for output_dir in self._output_dir_per_eval_task]
      try:
        yield (train_writers, eval_writers)
      finally:
        for writer in train_writers + eval_writers:
          writer.close()
        _log(f'Metrics were written in {self._output_dir}', stdout=False)
    else:
      yield ([None] * len(self._tasks), [None] * len(self._eval_tasks))


def _model_with_ends(model, end_layers, batch_signature):
  """Returns a model+ends layer built on an already initialized model.

  Ends can be loss or metric layers.

  Args:
    model: Layer with initialized weights and state.
    end_layers: List of end layers.
    batch_signature: Signature of the model input batch.

  Returns:
    An initialized, combined model+ends layer, preserving the initialization
    of `model`.
  """
  # TODO(jonni): Redo this function as part of an initialization refactor?
  metrics_layer = tl.Branch(*end_layers)
  metrics_input_signature = model.output_signature(batch_signature)
  _, _ = metrics_layer.init(metrics_input_signature)

  model_with_metrics = tl.Serial(model, metrics_layer)
  return model_with_metrics


def _model_with_metrics(model, eval_task):
  """Returns a model+metrics layer built on an already initialized model.

  Args:
    model: Layer with initialized weights and state.
    eval_task: EvalTask instance.

  Returns:
    An initialized, combined model+metrics layer, preserving the initialization
    of `model`.
  """
  return _model_with_ends(
      model, eval_task.metrics, shapes.signature(eval_task.sample_batch)
  )


class TrainTask:
  """A supervised task (labeled data + feedback mechanism) for training."""

  def __init__(self, labeled_data, loss_layer, optimizer,
               lr_schedule=None, n_steps_per_checkpoint=100,
               n_steps_per_permanent_checkpoint=None):
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
      n_steps_per_permanent_checkpoint: How many steps to run between permanent
          checkpoints.
    """
    self._labeled_data = labeled_data
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._lr_schedule = lr_schedule
    self._sample_batch = next(labeled_data)
    self._n_steps_per_checkpoint = n_steps_per_checkpoint
    self._n_steps_per_permanent_checkpoint = n_steps_per_permanent_checkpoint

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
  def n_steps_per_permanent_checkpoint(self):
    return self._n_steps_per_permanent_checkpoint

  @property
  def optimizer(self):
    return self._optimizer

  def learning_rate(self, step):
    """Return the learning rate for the given step."""
    if self._lr_schedule is not None:
      with fastmath.use_backend(fastmath.Backend.NUMPY):
        return self._lr_schedule(step)
    opt = self._optimizer
    if callable(opt):  # when optimizer is a function, like Adam, not Adam()
      opt = opt()
    params = opt._init_opt_params  # pylint: disable=protected-access
    return params['learning_rate']


@gin.configurable()
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
  if period is None:
    return lambda step_n: False

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
      pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      with gzip_lib.GzipFile(fileobj=f, compresslevel=2) as gzipf:
        pickle.dump(obj, gzipf, protocol=pickle.HIGHEST_PROTOCOL)
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
  if fastmath.is_backend(fastmath.Backend.JAX):
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
  if n_devices != device_count and fastmath.is_backend(fastmath.Backend.JAX):
    raise ValueError('JAX cannot work yet with n_devices != all devices: '
                     '%d != %d' % (n_devices, device_count))

  if random_seed is None and host_count > 1:
    random_seed = int(1e6 * (host_id + time.time())) % 2**32
  return (is_chief, host_count, n_devices,
          _init_random_number_generators(random_seed))


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
  n_devices_total = fastmath.psum(jnp.array(1.0), 'devices').astype(jnp.int32)

  # This sums up the weights and state across all devices.
  # NOTE: There will not be any leading axis remaining because we psum
  # over it.
  weights_and_state = fastmath.psum(weights_and_state, 'devices')

  # We finally take the average over all devices.
  # We also make sure we don't change the type of the weights and state.
  weights_and_state = jax.tree_util.tree_map(
      lambda ws: (ws / n_devices_total).astype(ws.dtype), weights_and_state)

  return weights_and_state


def _is_uninitialized(model):
  """Checks whether no weights in the model have been initialized."""
  def _is_empty(x):
    if isinstance(x, (list, tuple)):
      return all(_is_empty(y) for y in x)
    else:
      return x is None
  if not _is_empty(model.weights):
    return False
  return all(_is_uninitialized(l) for l in model.sublayers)
