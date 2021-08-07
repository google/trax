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

"""Original API for supervised learning/training in Trax.

Trax authors expect that the `supervised.training` module (under development)
will replace `trainer_lib`.
"""

import collections
import functools
import itertools
import os
import sys
import time

from absl import logging

import gin

import jax
import tensorflow.compat.v2 as tf
from trax import fastmath
from trax import jaxboard
from trax import layers as tl
from trax import optimizers as trax_opt
from trax.data import inputs as trax_inputs
from trax.fastmath import numpy as np
from trax.fastmath import random as jax_random
from trax.layers import base
from trax.shapes import ShapeDtype
from trax.supervised import history as trax_history
from trax.supervised import lr_schedules as lr
from trax.supervised import training


# TODO(afrozm): Maybe flatten everything from OptState into TrainerState.
TrainerState = collections.namedtuple('_TrainerState', [
    'step',         # Current training step number.
    'opt_state',    # OptState.
    'history',      # trax.history.History.
    'model_state',  # Auxilliary state of the model.
])


OptState = collections.namedtuple('_OptState', [
    'weights',     # Model weights.
    'slots',       # Per-parameter optimizer state, e.g. gradient moments.
    'opt_params',  # Optimizer (hyper)parameters, e.g. learning rate, momentum.
])


_DEFAULT_METRICS = {
    'loss': tl.WeightedCategoryCrossEntropy(),
    'accuracy': tl.WeightedCategoryAccuracy(),
    'sequence_accuracy': tl.MaskedSequenceAccuracy(),
    'neg_log_perplexity': tl.Serial(tl.WeightedCategoryCrossEntropy(),
                                    tl.Negate()),
    'weights_per_batch_per_core': tl.Serial(tl.Drop(), tl.Drop(), tl.Sum()),
}


NamedStream = collections.namedtuple(
    'NamedStream', ['name', 'stream']
)


@gin.configurable
def named_stream(name=gin.REQUIRED, stream=gin.REQUIRED):
  return NamedStream(name=name, stream=stream)


class Trainer:
  """Trax trainer.

  A trainer allows to make training steps, train for full epochs,
  save the training state and access evaluation data.
  """

  def __init__(self, model, loss_fn, optimizer, lr_schedule, inputs,
               output_dir=None, random_seed=None, n_devices=None,
               checkpoints_at=None, should_save_checkpoints=True,
               should_write_summaries=True,
               metrics=None, checkpoint_highest=None,
               checkpoint_lowest=None,
               init_checkpoint=None):

    self._is_chief, _, self._n_devices, rng = (
        training.init_host_and_devices(n_devices, random_seed))
    self._should_save_checkpoints = should_save_checkpoints and self._is_chief
    self._checkpoints_at = checkpoints_at if checkpoints_at is not None else []
    self._should_write_summaries = should_write_summaries
    if not output_dir:
      self._should_save_checkpoints = False
      self._should_write_summaries = False
    self._checkpoint_highest = checkpoint_highest
    self._checkpoint_lowest = checkpoint_lowest
    self._metrics_dict = metrics if metrics is not None else _DEFAULT_METRICS
    # Inputs is either an Inputs instance or a function that returns it.
    self._inputs = inputs
    if callable(inputs):  # If we pass a function, e.g., through gin, call it.
      self._inputs = inputs()
    # Initialize the learning rate to a dummy value. It will be set in reset().
    opt = optimizer(learning_rate=0.0)

    # Setup the model.
    model_train = model(mode='train')
    model_predict_eval = model(mode='eval')
    # Should work for fine-tuning of T5.
    if init_checkpoint:
      model_train.init_from_file(init_checkpoint, weights_only=True)
      model_predict_eval.init_from_file(init_checkpoint, weights_only=True)
    self._model_with_loss = tl.Serial(model_train, loss_fn)

    # Setup state.
    rng, init_rng = jax_random.split(rng)
    self._rngs = np.stack(jax_random.split(rng, self._n_devices))
    shapes, dtypes = self._inputs.example_shape_dtype
    input_signature = tuple(ShapeDtype(s, d) for (s, d) in zip(shapes, dtypes))

    def new_opt_state_and_model_state(rng):
      """Returns optimizer and model states suitable for training a model."""
      weights, state = self._model_with_loss.init(input_signature, rng=rng)
      (slots, opt_params) = opt.tree_init(weights)
      return (OptState(weights, slots, opt_params), state)

    if fastmath.is_backend(fastmath.Backend.JAX):
      # JIT parameter initialization to avoid memory fragmentation
      new_opt_state_and_model_state = (
          fastmath.jit(new_opt_state_and_model_state))
    self._new_opt_state_and_model_state = (
        lambda: new_opt_state_and_model_state(init_rng))

    # Arrange and initialize metrics layers.
    self._metrics = list(sorted(self._metrics_dict.keys()))
    metrics_layers = [self._metrics_dict[m] for m in self._metrics]
    metrics_in_parallel = tl.Branch(*metrics_layers)
    metrics_in_parallel.rng = init_rng
    example_signature = tuple(
        ShapeDtype(s, d) for (s, d) in zip(*self._inputs.example_shape_dtype)
    )
    model_predict_eval.init(example_signature)
    self._input_signature = example_signature
    output_signature = model_predict_eval.output_signature(example_signature)
    m_weights, m_state = metrics_in_parallel.init(output_signature)
    self._metrics_weights = self._for_n_devices(m_weights)
    self._metrics_state = self._for_n_devices(m_state)

    # Jit model_predict and update so they're fast.
    self._jit_eval = _jit_predict_fn(
        model_predict_eval, metrics_in_parallel, self._n_devices)
    self._jit_update_fn = _jit_update_fn(
        model_train, loss_fn, opt, self._n_devices)

    self._model_train = model_train
    self._model_predict_eval = model_predict_eval
    self._loss_fn = loss_fn
    self._lr_schedule = lr_schedule

    # Those fields will be set in reset().
    self._output_dir = None
    self._train_sw = None
    self._eval_sw = None
    self._history = None
    self._opt_state = None
    self._step = None
    self._model_state = None
    self.reset(output_dir)

  @property
  def n_devices(self):
    return self._n_devices

  @property
  def step(self):
    return self._step

  @property
  def model_weights(self):
    # Currently we need to pick [0] as we ignore loss weights (empty).
    weights = self._opt_state.weights[0]
    if self.n_devices > 1:
      unreplicate = lambda x: x[0]
      weights = fastmath.nested_map(unreplicate, weights)
    return weights

  @model_weights.setter
  def model_weights(self, weights):
    new_model_weights = self._for_n_devices(weights)
    if isinstance(self._opt_state.weights, list):
      self._opt_state.weights[0] = new_model_weights
    else:  # weights are a tuple, need to re-create
      new_weights = [new_model_weights] + list(self._opt_state.weights[1:])
      self._opt_state = self._opt_state._replace(weights=new_weights)

  @property
  def model_state(self):
    # Currently we need to pick [0] as we ignore loss state (empty).
    state = self._model_state[0]
    if self.n_devices > 1:
      unreplicate = lambda x: x[0]
      state = fastmath.nested_map(unreplicate, state)
    return state

  @model_state.setter
  def model_state(self, state):
    new_model_state = self._for_n_devices(state)
    if isinstance(self._model_state, list):
      self._model_state[0] = new_model_state
    else:  # weights are a tuple, need to re-create
      self._model_state = [new_model_state] + list(self._model_state[1:])

  @property
  def state(self):
    return TrainerState(
        opt_state=self._opt_state, step=self._step, history=self._history,
        model_state=self._model_state)

  @property
  def learning_rate(self):
    with fastmath.use_backend(fastmath.Backend.NUMPY):
      return self._lr_schedule(self._step)

  def reset(self, output_dir, init_checkpoint=None):
    """Reset the model parameters.

    Restores the parameters from the given output_dir if a checkpoint exists,
    otherwise randomly initializes them.

    Does not re-jit the model.

    Args:
      output_dir: Output directory.
      init_checkpoint: Initial checkpoint (default $output_dir/model.pkl.gz)
    """
    self.close()
    self._output_dir = output_dir
    if output_dir is not None:
      tf.io.gfile.makedirs(output_dir)
    else:
      assert not self._should_save_checkpoints
      assert not self._should_write_summaries

    # Create summary writers and history.
    if self._should_write_summaries:
      self._train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, 'train'),
                                              enable=self._is_chief)
      self._eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, 'eval'),
                                             enable=self._is_chief)

    # Reset the train and eval streams.
    self._train_stream = _repeat_stream(self._inputs.train_stream,
                                        self._n_devices)
    # TODO(lukaszkaiser): add an option to evaluate exactly on the full eval
    #   set by adding a padding and stopping the stream when too large.
    self._eval_stream = _repeat_stream(
        self._inputs.eval_stream, self._n_devices)
    self._train_eval_stream = _repeat_stream(
        self._inputs.train_eval_stream, self._n_devices)

    # Restore the training state.
    if output_dir is not None:
      state = load_trainer_state(output_dir, self._model_with_loss,
                                 init_checkpoint)
    else:
      state = TrainerState(step=None, opt_state=None,
                           history=trax_history.History(), model_state=None)
    self._step = state.step or 0
    history = state.history
    self._history = history
    if state.opt_state:
      opt_state = state.opt_state
      model_state = state.model_state
    else:
      opt_state, model_state = self._new_opt_state_and_model_state()
      model_state = self._for_n_devices(model_state)
    self._opt_state = OptState(*self._for_n_devices(opt_state))
    self._model_state = model_state
    if not state.opt_state and self._should_save_checkpoints:
      self.save_state(keep=False)

  def train_epoch(self, n_steps, n_eval_steps):
    """Runs `n_steps` of training, with periodic logging, saving, and evals."""
    # TODO(jonni): Clarify how this method relates to the stricter notion of
    # epoch (training for as many steps as needed for a full pass through the
    # training data).
    print()  # Add visual separator in logs for start of training epoch.
    start_time = time.time()

    for _ in range(n_steps):
      batch = next(self._train_stream)
      if self.n_devices > 1:  # TODO(lukaszkaiser): use everywhere if possible.
        batch = _reshape_by_device(batch, self.n_devices)
      self.train_step(batch)
      if self._should_save_now():
        self.save_state(keep=True)
      if self._should_log_now():
        self._train_sw.scalar('training/learning_rate', self.learning_rate)

    # At end of n_steps, do bookkeeping, run evals, and save state.
    elapsed_time = time.time() - start_time
    self.log_step('Ran %d train steps in %0.2f secs' % (n_steps, elapsed_time))
    if self._train_sw and n_steps > 1:
      self._train_sw.scalar('training/steps per second',
                            n_steps / elapsed_time, step=self._step)
      self._train_sw.flush()
    self.evaluate(n_eval_steps)
    if self._eval_sw:
      self._eval_sw.flush()
    if self._should_save_checkpoints:
      self.save_state(keep=False)
    if self._should_save_checkpoints and self._current_step_is_best(high=True):
      self.save_state(keep=False, prefix='highest_' + self._checkpoint_highest)
    if self._should_save_checkpoints and self._current_step_is_best(high=False):
      self.save_state(keep=False, prefix='lowest_' + self._checkpoint_lowest)

  def train_step(self, batch):
    """Run one training step and update self._opt_state."""
    # Calculate the current optimizer parameters.
    opt_param_updates = self._for_n_devices(
        {'learning_rate': np.array(self.learning_rate)})
    opt_state = self._opt_state
    opt_state.opt_params.update(opt_param_updates)

    # Run the update.
    weights, slots, opt_params = opt_state
    (weights, slots), stat, self._model_state, self._rngs = self._jit_update_fn(
        (weights, slots), self._step, opt_params, batch,
        self._model_state, self._rngs)
    self._opt_state = opt_state._replace(weights=weights, slots=slots)
    if self._should_log_now():
      for name, value in stat.items():
        # TODO(afrozm): value is a scalar, but sometimes JAX is crashing here
        # with a device put array error complaining that it should be an array.
        # On multiple devices, take the mean.
        scalar_value = np.mean(np.array(value))
        self._train_sw.scalar('training/' + name, scalar_value, step=self._step)
    self._step += 1

  def evaluate(self, n_eval_steps):
    """Evaluate the model and log metrics."""
    _, rng = jax_random.split(self._rngs[0])
    # TODO(lukaszkaiser): both model state and parameters by default include
    # the loss layer. Currently, we access the pure-model parameters by just
    # indexing, [0] here. But we should make it more explicit in a better API.
    weights = (self._opt_state.weights[0], self._metrics_weights)
    state = (self._model_state[0], self._metrics_state)
    self.log_step('Evaluation')
    train_eval_slice = itertools.islice(self._train_eval_stream, n_eval_steps)
    train_metrics, _ = self.evaluation_round(train_eval_slice, weights, state,
                                             rng)
    self.log_metrics(train_metrics, self._train_sw, 'train')
    eval_slice = itertools.islice(self._eval_stream, n_eval_steps)
    eval_metrics, _ = self.evaluation_round(eval_slice, weights, state, rng)
    self.log_metrics(eval_metrics, self._eval_sw, 'eval')
    self.log_step('Finished evaluation')

    # Save the learning rate in history.
    self._history.append('train', 'training/learning_rate',
                         self._step, self.learning_rate)

  def evaluation_round(self, inputs_stream, weights, state, rng):
    """Evaluate.

    Args:
      inputs_stream: Iterable of inputs to evaluate on.
      weights: Weights for each f in eval_fns.
      state: State for each f in eval_fns.
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      Tuple of `(metrics, state)`. `metrics` is a dict from metric name to
      metric value averaged over the number of inputs, and `state` is the end
      state returned by this trainer's `predict_fn`.
    """
    metrics = collections.defaultdict(float)
    count = 0
    for inp in inputs_stream:
      count += 1
      rng, subrng = jax_random.split(rng)
      metric_values, _ = self._jit_eval(inp, weights, state, subrng)
      try:
        metric_values = list(metric_values)
      except (TypeError, IndexError):
        metric_values = [float(metric_values)]
      for m, v in zip(self._metrics, metric_values):
        metrics[m] += v
    return {m: v / count for (m, v) in metrics.items()}, state

  def save_gin(self):
    """"Saves the operative gin config, only if it is the chief."""
    if not self._is_chief:
      return
    assert self._output_dir is not None
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    sw = self._train_sw
    if sw:
      sw.text('gin_config',
              jaxboard.markdownify_operative_config_str(config_str))

  def _save_state_dict(self, trainer_state_dict, weights_file):
    training.pickle_to_file(trainer_state_dict, weights_file, gzip=True)
    log('Model saved to %s' % weights_file, stdout=False)

  def save_state(self, keep, prefix='model'):
    """Save trainer state given a possibly replicated opt_state."""
    opt_state = self._opt_state
    if self.n_devices > 1:
      first_replica = lambda x: x[0]
      opt_state = OptState(*fastmath.nested_map(first_replica, opt_state))
    # This line, while optional, allows JAX to transfer arrays from the device
    # to the host in parallel, which is particularly important for cloud TPU.
    if fastmath.is_backend(fastmath.Backend.JAX):
      opt_state = jax.device_get(opt_state)
    step, history, model_state = self._step, self._history, self._model_state
    output_dir = self._output_dir

    weights_file = os.path.join(output_dir, prefix + '.pkl.gz')

    # This dict will be stored as the model.
    trainer_state_dict = make_trainer_state_dict(
        step, opt_state, history, model_state, self._input_signature)
    self._save_state_dict(trainer_state_dict, weights_file)

    if keep:
      weights_file = os.path.join(output_dir,
                                  '{}_{}.pkl.gz'.format(prefix, step))
      self._save_state_dict(trainer_state_dict, weights_file)

  def save_computation_graphs(self):
    """Dump computation graphs to files."""
    if self.n_devices != 1:
      return  # TODO(lukaszkaiser): make this work with more devices.
    batch = next(self._train_stream)
    output_dir = self._output_dir
    if self.n_devices > 1:
      batch = _reshape_by_device(batch, self.n_devices)
    weights = self._opt_state.weights[0]
    forward_computation = jax.xla_computation(self._model_predict_eval)(
        batch, weights=weights, state=self._model_state[0],
        rng=self._rngs[0])
    with tf.io.gfile.GFile(os.path.join(output_dir, 'forward.txt'), 'w') as f:
      f.write(forward_computation.as_hlo_text())
    with tf.io.gfile.GFile(os.path.join(output_dir, 'forward.dot'), 'w') as f:
      f.write(forward_computation.as_hlo_dot_graph())

  def log_step(self, step_message):
    log('Step % 6d: %s' % (self.step, step_message))

  def log_metrics(self, metrics, summ_writer, log_prefix):
    """Log metrics to summary writer and history."""
    history = self._history
    rjust_len = max([0] + [len(name) for name in metrics])
    for name, value in metrics.items():
      self.log_step('%s %s | % .8f' % (
          log_prefix.ljust(5), name.rjust(rjust_len), value))
      full_name = 'metrics/' + name
      if history:
        history.append(log_prefix, full_name, self.step, value)
      if summ_writer:
        summ_writer.scalar(full_name, value, self.step)

  def print_n_weights(self):
    """Prints the total count of trainable weights."""
    opt_state = self._opt_state
    sizes = _sizes(opt_state.weights)
    if self.n_devices > 1:
      unreplicate = lambda x: x[0]
      single_weights = fastmath.nested_map(unreplicate, opt_state.weights)
      sizes = _sizes(single_weights)
    total_size = _nested_reduce(sum, sizes)
    self.log_step('Total number of trainable weights: %d' % total_size)

  def _should_save_now(self):
    return self._should_save_checkpoints and self._step in self._checkpoints_at

  def _current_step_is_best(self, high):
    """Is the current step the best (highest if high, else lowest)."""
    metric = self._checkpoint_highest if high else self._checkpoint_lowest
    if metric is None:
      return False
    # History is a list of pairs (step, value).
    history = self._history.get('eval', 'metrics/' + metric)
    sequence = [float(i[1]) for i in history]  # Just the values.
    best = max(sequence) if high else min(sequence)  # Best value.
    last_is_best = float(history[-1][1]) == best  # Is last the best?
    cur_step = history[-1][0] == self._step  # Is last the current step?
    return cur_step and last_is_best

  def _should_log_now(self):
    return (self._train_sw is not None
            and (self._step == 1 or self._step % 10 == 0))

  def _for_n_devices(self, x):
    """Replicates/broadcasts `x` for n devices if `self.n_devices > 1`."""
    return tl.for_n_devices(x, self.n_devices)  # pylint: disable=protected-access

  def close(self):
    if self._train_sw is not None:
      self._train_sw.close()
      self._train_sw = None
    if self._eval_sw is not None:
      self._eval_sw.close()
      self._eval_sw = None


@gin.configurable(denylist=['output_dir'])
def train(output_dir,
          model=gin.REQUIRED,
          loss_fn=tl.WeightedCategoryCrossEntropy(),
          inputs=trax_inputs.batcher,
          optimizer=trax_opt.Adafactor,
          lr_schedule_fn=lr.multifactor,
          trainer_class=Trainer,
          steps=1000,
          checkpoints_at=None,
          permanent_checkpoints_at=None,
          eval_steps=10,
          eval_frequency=100,
          permanent_checkpoint_frequency=None,
          random_seed=None,
          save_graphs=True,
          metrics=None,
          checkpoint_highest=None,
          checkpoint_lowest=None,
          use_loop=True,
          loss_chunk_size=0,
          use_memory_efficient_trainer=False,
          adasum=False,
          init_checkpoint=None,
          callbacks=None,
          n_weights_shards=1,
          additional_train_tasks=None,
          additional_eval_tasks=None,
          additional_eval_streams=None):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fn
      and apply_fn.
    loss_fn: callable with signature: weights, trax.inputs.Inputs, model, state,
      rng -> loss.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer (see optimizers/base.py for signature).
    lr_schedule_fn: A learning rate schedule function, that when called returns
      a function from step to learning rate (a float).
    trainer_class: The trainer class to use.
    steps: int, total number of training steps.
    checkpoints_at: list of integers. Save a checkpoint for each training step
      in the list.
    permanent_checkpoints_at: list of integers. Save a permanent checkpoint for
      each training step in the list.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    permanent_checkpoint_frequency: int, how often to save permanent checkpoints
      (every permanent_checkpoint_frequency steps).
    random_seed: the random seed to use; time/os dependent if None (default).
    save_graphs: bool, if True, save computation graph to file.
    metrics: optionally override the default metrics dictionary.
    checkpoint_highest: save the checkpoint highest at this metric.
    checkpoint_lowest: save the checkpoint lowest at this metric.
    use_loop: whether to use training.Loop instead of Trainer.
    loss_chunk_size: int, if > 0 chunk loss into these sizes to save memory.
    use_memory_efficient_trainer: whether to use memory-efficient trainer.
    adasum: if True, use adaptive summation for multi-device gradients.
    init_checkpoint: a checkpoint for fine tuning.
    callbacks: a list of callbacks to call during training.
    n_weights_shards: shard weights into this many devices.
    additional_train_tasks: additional tasks which should be performed during
      training.
    additional_eval_tasks: additional tasks which should be performed during
      evaluation.
    additional_eval_streams: List[NamedStream], additional data streams that
      should be used during evaluation. Can be provided independently of
      additional_eval_tasks.

  Returns:
    trax.TrainerState or training.Loop if use_loop is True
  """
  base.N_WEIGHTS_SHARDS = n_weights_shards
  if (permanent_checkpoint_frequency is not None
      and permanent_checkpoints_at is not None):
    raise ValueError('Only one of ["permanent_checkpoint_frequency", '
                     '"permanent_checkpoints_at"] should be set.')
  if use_loop:
    n_devices = num_devices() or fastmath.local_device_count()

    # Prepare the training task.
    # Inputs is either an Inputs instance or a function that returns it.
    if callable(inputs):  # If we pass a function, e.g., through gin, call it.
      inputs = inputs()
    opt = optimizer if use_memory_efficient_trainer else optimizer()
    train_task = training.TrainTask(
        inputs.train_stream(n_devices),
        loss_layer=loss_fn,
        optimizer=opt,
        lr_schedule=lr_schedule_fn(),
        n_steps_per_checkpoint=eval_frequency,
        n_steps_per_permanent_checkpoint=permanent_checkpoint_frequency)

    if additional_train_tasks is None:
      additional_train_tasks = []

    # Prepare the evaluation.
    metrics_dict = metrics if metrics is not None else _DEFAULT_METRICS
    names, metrics = zip(*metrics_dict.items())
    eval_task = training.EvalTask(inputs.eval_stream(n_devices),
                                  metrics,
                                  metric_names=names,
                                  n_eval_batches=eval_steps)

    if additional_eval_tasks is None:
      additional_eval_tasks = []

    additional_eval_tasks_from_streams = []
    if additional_eval_streams is not None:
      for stream in additional_eval_streams:
        additional_eval_tasks_from_streams.append(
            training.EvalTask(stream.stream,
                              metrics,
                              metric_names=names,
                              n_eval_batches=eval_steps,
                              export_prefix=stream.name))

    # Prepare the training loop.
    checkpoint_at = None
    if checkpoints_at is not None:
      checkpoint_at = lambda step: step in checkpoints_at
    permanent_checkpoint_at = None
    if permanent_checkpoints_at is not None:
      permanent_checkpoint_at = (lambda step: step in permanent_checkpoints_at)

    # Setup the model.
    model_train = model(mode='train')
    model_predict_eval = model(mode='eval')
    if init_checkpoint:
      model_train.init_from_file(init_checkpoint, weights_only=True)
      model_predict_eval.init_from_file(init_checkpoint, weights_only=True)
    loop = training.Loop(
        model_train, [train_task] + additional_train_tasks,
        eval_model=model_predict_eval,
        eval_tasks=[eval_task] +
        additional_eval_tasks + additional_eval_tasks_from_streams,
        output_dir=output_dir,
        checkpoint_at=checkpoint_at,
        checkpoint_low_metric=checkpoint_lowest,
        checkpoint_high_metric=checkpoint_highest,
        permanent_checkpoint_at=permanent_checkpoint_at,
        n_devices=n_devices,
        loss_chunk_size=loss_chunk_size,
        use_memory_efficient_trainer=use_memory_efficient_trainer,
        adasum=adasum,
        random_seed=random_seed,
        callbacks=callbacks,
    )

    steps_to_go = steps - loop.step
    if steps_to_go <= 0:
      log('Stop training, already reached the total training steps %d' % steps)
      return loop

    # Train and return the loop.
    loop.run(steps_to_go)
    return loop

  n_devices = num_devices()
  trainer = trainer_class(model, loss_fn, optimizer, lr_schedule_fn(), inputs,
                          output_dir,
                          random_seed=random_seed,
                          n_devices=n_devices,
                          checkpoints_at=checkpoints_at,
                          metrics=metrics,
                          checkpoint_lowest=checkpoint_lowest,
                          checkpoint_highest=checkpoint_highest,
                          init_checkpoint=init_checkpoint)

  epoch_steps = [steps]  # Only training if eval_frequency is 0 or None
  if eval_frequency and eval_steps > 0:
    epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                   eval_frequency - 1],
                                  itertools.repeat(eval_frequency))
  trainer.log_step('Starting training using %d devices' % trainer.n_devices)
  trainer.print_n_weights()

  try:
    for epoch_steps in epochs(steps, trainer.step, epoch_steps):
      trainer.train_epoch(epoch_steps, eval_steps)

      # Bookkeeping we do at the first step
      if trainer.step == 1:
        # Save computation graph (single-device only for now)
        if (save_graphs and fastmath.is_backend(fastmath.Backend.JAX)):
          trainer.save_computation_graphs()

        # Save Gin config
        trainer.save_gin()

    trainer.log_step('Training done')
  except Exception as e:
    raise e
  finally:
    trainer.close()
  return trainer.state


@gin.configurable
def num_devices(value=None):
  """Returns how many devices to use (if None, default, use all available)."""
  return value


@gin.configurable
def _jit_update_fn(predict_fn, loss_fn, optimizer, n_devices, jit=True):
  """Returns a (JIT-compiled) function that computes updates for one step."""
  model_and_loss = tl.Serial(predict_fn, loss_fn)
  # Gradients are always wrt. the first argument, so putting weights first.
  def model_and_loss_call(weights, batch, state, rng):
    res = model_and_loss(batch, weights=weights, state=state, rng=rng)
    return res, model_and_loss.state
  if n_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_update(weights_and_slots, i, opt_params, batch, state, rng):
      weights, slots = weights_and_slots
      rng, subrng = jax_random.split(rng[0])
      grad_fn = fastmath.grad(model_and_loss_call, has_aux=True)
      grads, state = grad_fn(weights, batch, state, rng)
      new_weights, new_slots, stats = optimizer.tree_update(
          i, grads, weights, slots, opt_params)
      return (new_weights, new_slots), stats, state, [subrng]
    if jit:
      # TODO(lukaszkaiser): donate_argnums=(0,) when XLA supports it on GPU
      return fastmath.jit(single_update)
    else:
      return single_update

  # Else, for n_devices > 1:
  @functools.partial(fastmath.pmap, axis_name='batch')  # donate_argnums=(0,))
  def mapped_update(weights_and_slots, i, opt_params, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    weights, slots = weights_and_slots
    rng, subrng = jax_random.split(rng)
    grad_fn = fastmath.grad(model_and_loss_call, has_aux=True)
    grads, state = grad_fn(weights, batch, state, rng)
    # We do a psum(1.0) here instead of `n_devices` since `n_devices` is just
    # the number of devices on this host machine, however psum goes over all
    # devices of all hosts (ex: a TPU pod) and we need to be averaging over all
    # of them.
    #
    # Collect all gradients.
    grads = fastmath.psum(grads, 'batch')
    n_devices_total = fastmath.psum(np.array(1.0), 'batch')
    # Average across hosts.
    grads = jax.tree_util.tree_map(lambda g: g / n_devices_total, grads)

    new_weights, new_slots, stats = optimizer.tree_update(
        i, grads, weights, slots, opt_params)
    return (new_weights, new_slots), stats, state, subrng

  def update(weights_and_slots, i, opt_params, batch, state, rng):
    return mapped_update(weights_and_slots, np.repeat(i, n_devices),
                         opt_params, batch, state, rng)

  return update


@gin.configurable
def _jit_predict_fn(model_predict, metric_fn, n_devices, jit=True):
  """Returns a JIT-compiled predict function (unless jit=False)."""
  model = tl.Serial(model_predict, metric_fn)
  if not jit:
    return model.pure_fn

  return tl.jit_forward(model.pure_fn, n_devices)


@gin.configurable
def _jit_compute_loss_fn(predict_fn, loss_fn, n_devices, jit=True):
  """Returns a (JIT-compiled) function that computes the loss for one step."""
  if n_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_compute_loss(opt_state, batch, state, rng):
      rng, subrng = jax_random.split(rng[0])
      loss_val, state = loss_fn(opt_state[0], batch, predict_fn, state, rng)
      return loss_val, state, [subrng]
    return fastmath.jit(single_compute_loss) if jit else single_compute_loss

  # Else, for n_devices > 1:
  @functools.partial(fastmath.pmap, axis_name='batch')
  def mapped_compute_loss(opt_state, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    rng, subrng = jax_random.split(rng)
    loss_val, state = loss_fn(opt_state[0], batch, predict_fn, state, rng)
    return loss_val, state, subrng

  def compute_loss(opt_state, batch, state, rng):
    return mapped_compute_loss(
        opt_state, _reshape_by_device(batch, n_devices), state, rng)

  return compute_loss


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)
    sys.stdout.flush()


def epochs(total_steps, steps_to_skip, epoch_steps):
  """Generates the number of steps in each epoch before reaching total_steps.

  Args:
    total_steps: int, total number of steps.
    steps_to_skip: int, number of steps to skip because of a restart.
    epoch_steps: iterable of int, numbers of steps in each epoch.

  Yields:
    epoch_steps: int, number of steps in this epoch
  """
  steps_to_go = total_steps - steps_to_skip
  epoch_steps = iter(epoch_steps)

  # Remove the desired number of steps from the stream.
  for steps_this_epoch in epoch_steps:
    if steps_this_epoch > steps_to_skip:
      # Put back the number of steps left in the unfinished epoch.
      epoch_steps = itertools.chain(
          [steps_this_epoch - steps_to_skip], epoch_steps)
    if steps_this_epoch >= steps_to_skip:
      break
    steps_to_skip -= steps_this_epoch

  # Yield the remaining steps per epoch up to total_steps.
  for steps_this_epoch in epoch_steps:
    steps_this_epoch = min(steps_this_epoch, steps_to_go)
    yield steps_this_epoch
    steps_to_go -= steps_this_epoch
    if steps_to_go == 0:
      break


def make_trainer_state_dict(step,
                            opt_state,
                            history,
                            model_state,
                            input_signature):
  """Creates a trainer state dictionary to save to disk.

  Args:
    step: int, a step number
    opt_state: OptState namedtuple
    history: `trax.history.History`, the history object.
    model_state: A nested structure of the model state.
    input_signature: signature of model inputs.

  Returns:
    A dictionary with the fields of TrainerState and OptState flattened.
  """
  flat_weights, flat_state = tl.flatten_weights_and_state(
      opt_state.weights, model_state)
  return {
      'step': step,
      'flat_weights': flat_weights,
      'slots': opt_state.slots,
      'opt_params': opt_state.opt_params,
      'history': history,
      'flat_state': flat_state,
      'input_signature': input_signature,
      'version_timestamp': 'Jun-18-2020'  # To update in the future if needed.
  }


def trainer_state_from_dict(trainer_state_dict, model):
  """Given the trainer state dictionary, returns `TrainerState`."""
  # TODO(afrozm): This becomes simpler if OptState is flattened into
  # TrainerState.
  step = trainer_state_dict['step']
  history = trainer_state_dict['history']
  input_signature = trainer_state_dict['input_signature']
  weights_and_state_sig = model.weights_and_state_signature(input_signature)
  weights, model_state = tl.unflatten_weights_and_state(
      trainer_state_dict['flat_weights'], trainer_state_dict['flat_state'],
      weights_and_state_sig)
  opt_state = OptState(
      weights=weights,
      slots=trainer_state_dict['slots'],
      opt_params=trainer_state_dict['opt_params'])
  return TrainerState(step=step, opt_state=OptState(*opt_state),
                      history=history, model_state=model_state)


def load_trainer_state(output_dir, model, weights_file=None):
  """Returns a TrainerState instance loaded from the given `output_dir`."""
  if weights_file is None:
    weights_file = os.path.join(output_dir, 'model.pkl.gz')
    if not tf.io.gfile.exists(weights_file):
      return TrainerState(step=None, opt_state=None,
                          history=trax_history.History(), model_state=None)
  elif not tf.io.gfile.exists(weights_file):
    raise ValueError('File not found: %s' % weights_file)

  trainer_state_dict = training.unpickle_from_file(weights_file, gzip=True)
  trainer_state = trainer_state_from_dict(trainer_state_dict, model)
  log('Model loaded from %s at step %d' % (weights_file, trainer_state.step))
  logging.debug('From loaded model : history = %s', trainer_state.history)
  return trainer_state


def _reshape_by_device(x, n_devices):
  """Reshapes possibly nested x into a shape (n_devices, ...)."""
  return tl.reshape_by_device(x, n_devices)  # pylint: disable=protected-access


def _nested_reduce(f, x):
  """Fold the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return f([_nested_reduce(f, y) for y in x])
  if isinstance(x, tuple):
    return f([_nested_reduce(f, y) for y in x])
  if isinstance(x, dict):
    return f([_nested_reduce(f, v) for (_, v) in x.items()])
  return x


def _sizes(x):
  """Get a structure of sizes for a structure of nested arrays."""
  def size(x):
    try:
      return x.size
    except Exception:  # pylint: disable=broad-except
      return 0
  return fastmath.nested_map(size, x)


def _repeat_stream(stream, n_devices):
  """Repeat a stream indefinitely."""
  while True:
    for example in stream(n_devices):
      yield example
