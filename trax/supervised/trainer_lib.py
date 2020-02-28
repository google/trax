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

"""Trax main training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import os
import pickle
import random
import sys
import time

from absl import logging

import gin

import jax
import numpy
import six
import tensorflow.compat.v2 as tf
from trax import history as trax_history
from trax import jaxboard
from trax import layers as tl
from trax import lr_schedules as lr
from trax import math
from trax import optimizers as trax_opt
from trax.math import numpy as np
from trax.math import random as jax_random
from trax.shapes import ShapeDtype
from trax.supervised import inputs as trax_inputs


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
    'loss': tl.CrossEntropyLoss,
    'accuracy': tl.AccuracyScalar,
    'sequence_accuracy': tl.SequenceAccuracyScalar,
    'neg_log_perplexity': tl.CrossEntropyLoss,
    'weights_per_batch_per_core': tl.SumOfWeights,
}


class Trainer(object):
  """Trax trainer.

  A trainer allows to make training steps, train for full epochs,
  save the training state and access evaluation data.
  """

  def __init__(self, model, loss_fn, optimizer, lr_schedule, inputs,
               output_dir=None, random_seed=None, n_devices=None,
               checkpoints_at=None, should_save_checkpoints=True,
               should_write_summaries=True, has_weights=False,
               nontrainable_param_map=None, id_to_mask=None,
               metrics=None, checkpoint_highest=None, checkpoint_lowest=None):

    self._is_chief, self._n_devices, rng = (
        self._init_host_and_devices(n_devices, random_seed))
    self._should_save_checkpoints = should_save_checkpoints and self._is_chief
    self._checkpoints_at = checkpoints_at or []
    self._should_write_summaries = should_write_summaries
    self._checkpoint_highest = checkpoint_highest
    self._checkpoint_lowest = checkpoint_lowest
    self._has_weights = has_weights
    self._id_to_mask = id_to_mask
    self._metrics_dict = metrics if metrics is not None else _DEFAULT_METRICS
    loss_fn = loss_fn(has_weights=has_weights, id_to_mask=id_to_mask)
    # Inputs is either an Inputs instance or a function that returns it.
    self._inputs = inputs
    if callable(inputs):  # If we pass a function, e.g., through gin, call it.
      self._inputs = inputs()

    # Initialize the learning rate to a dummy value. It will be set in reset().
    opt = optimizer(learning_rate=0.0)

    # Setup the model.
    model_train = model(mode='train')
    model_predict_eval = model(mode='eval')

    # Setup state.
    rng, init_rng = jax_random.split(rng)
    self._rngs = np.stack(jax_random.split(rng, self._n_devices))
    # If the inputs are a tuple/list, add [None] (batch) to each element.
    if self._inputs.input_shape and isinstance(
        self._inputs.input_shape[0], (list, tuple)
    ):
      model_input_shape = tuple(
          tuple([None] + list(shape)) for shape in self._inputs.input_shape)
    else:  # Otherwise just add [None] to the input shape.
      model_input_shape = tuple([None] + list(self._inputs.input_shape))
    # Same for targets.
    if self._inputs.target_shape and isinstance(
        self._inputs.target_shape[0], (list, tuple)
    ):
      model_target_shape = tuple(
          tuple([None] + list(shape)) for shape in self._inputs.target_shape)
    else:
      model_target_shape = tuple([None] + list(self._inputs.target_shape))
    # Change all None to 1 in input and target shape.
    model_input_shape = math.nested_map(lambda x: x or 1, model_input_shape)
    model_target_shape = math.nested_map(lambda x: x or 1,
                                         model_target_shape)

    def new_opt_state_and_model_state(input_shape, input_dtype, target_shape,
                                      target_dtype, rng):
      """Returns optimizer and model states suitable for training a model."""
      # Combine inputs and targets on the stack.
      if not isinstance(input_dtype, (list, tuple)):
        input_dtype = [input_dtype]
        input_shape = [input_shape]
      if not isinstance(target_dtype, (list, tuple)):
        target_dtype = [target_dtype]
        target_shape = [target_shape]
      dtypes = list(input_dtype) + list(target_dtype)
      shapes = list(input_shape) + list(target_shape)
      if self._has_weights:
        shapes += list(target_shape)
        dtypes += [np.float32 for _ in target_dtype]
      input_signature = tuple(ShapeDtype(s, d)
                              for (s, d) in zip(shapes, dtypes))
      # We need to create a new model instance and not reuse `model_train` here,
      # because `m.initialize` puts cached parameter values in `m` and hence the
      # next call of `m.initialize` will give wrong results.
      m = tl.Serial(model(mode='train'), loss_fn)
      m._set_rng_recursive(rng)  # pylint: disable=protected-access
      weights, state = m.init(input_signature)
      (slots, opt_params) = opt.tree_init(weights)
      return (OptState(weights, slots, opt_params), state)

    if _is_jit_init():
      # JIT parameter initialization to avoid memory fragmentation
      new_opt_state_and_model_state = math.jit(new_opt_state_and_model_state,
                                               static_argnums=(0, 1, 2, 3))
    self._new_opt_state_and_model_state = (
        lambda: new_opt_state_and_model_state(  # pylint: disable=g-long-lambda
            model_input_shape, self._inputs.input_dtype,
            model_target_shape, self._inputs.target_dtype, init_rng))

    # Arrange and initialize metrics layers.
    self._metrics = list(sorted(self._metrics_dict.keys()))
    metrics_layers = [self._metrics_dict[m](has_weights=self._has_weights,
                                            id_to_mask=self._id_to_mask)
                      for m in self._metrics]
    metrics_in_parallel = tl.Branch(*metrics_layers)
    # TODO(lukaszkaiser): clean this up once layer API stabilizes.
    # For now, we need to initialize metric layers somehow, so here we go.
    # We assume that they do not have any parameters, so this is a dummy.
    dummy_shapes = ((1, 2), (1,), (1,)) if self._has_weights else ((1, 2), (1,))
    dummy_signature = tuple(ShapeDtype(s) for s in dummy_shapes)
    metrics_in_parallel._set_rng_recursive(init_rng)  # pylint: disable=protected-access
    m_weights, m_state = metrics_in_parallel.init(dummy_signature)
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
    # TODO(pkozakowski): "Learning rate schedules" are currently able to control
    # control all optimizer parameters and model state, so let's rename them
    # accordingly.
    self._lr_schedule = lr_schedule

    if nontrainable_param_map is None:
      nontrainable_param_map = {}
    self._nontrainable_param_map = nontrainable_param_map

    # Those fields will be set in reset().
    self._output_dir = None
    self._train_sw = None
    self._eval_sw = None
    self._history = None
    self._lr_fn = None
    self._opt_state = None
    self._step = None
    self._model_state = None

    if output_dir is not None:
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
      weights = math.nested_map(unreplicate, weights)
    return weights

  @property
  def state(self):
    return TrainerState(
        opt_state=self._opt_state, step=self._step, history=self._history,
        model_state=self._model_state)

  @property
  def nontrainable_params(self):
    # TODO(afrozm): Give further thought to this name.
    # TODO(lukaszkaiser): it makes no sense to use an accelerator (e.g. TPU)
    # in op-by-op mode just to compute the learning rate. However, there
    # should be a cleaner approach that forceably swapping out the backend.
    with math.use_backend('numpy'):
      return self._lr_fn(self._step)

  def reset(self, output_dir, init_checkpoint=None):
    """Reset the model parameters.

    Restores the parameters from the given output_dir if a checkpoint exists,
    otherwise randomly initializes them.

    Does not re-jit the model.

    Args:
      output_dir: Output directory.
      init_checkpoint: Initial checkpoint to use (default $output_dir/model.pkl)
    """
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
    state = load_trainer_state(output_dir, init_checkpoint)
    self._step = state.step or 0
    history = state.history
    self._lr_fn = self._lr_schedule(history)
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

    self.update_nontrainable_params()

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
        for (name, value) in self.nontrainable_params.items():
          self._train_sw.scalar('training/{}'.format(name), value)

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
    # TODO(pkozakowski): Optimizer parameters get polluted with model state,
    # which doesn't break anything but is weird. Filter it out.
    opt_param_updates = self._for_n_devices(
        math.nested_map(np.array, self.nontrainable_params))
    opt_state = self._opt_state
    opt_state.opt_params.update(opt_param_updates)

    # Run the update.
    (weights, slots), self._model_state, self._rngs = self._jit_update_fn(
        self._step, opt_state, batch, self._model_state, self._rngs)
    self._model_state = self._map_to_state_dicts(self._state_dicts_update)
    self._opt_state = opt_state._replace(weights=weights, slots=slots)
    self._step += 1

  def evaluate(self, n_eval_steps):
    """Evaluate the model and log metrics."""
    _, rng = jax_random.split(self._rngs[0])
    # TODO(lukaszkaiser): both model state and parameters by default include
    # the loss layer. Currently, we access the pure-model parameters by just
    # indexing, [0] here. But we should make it more explicit in a better API.
    weights = (self._opt_state[0][0], self._metrics_weights)
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

    # Save the optimizer weights in the history
    for (name, value) in self.nontrainable_params.items():
      self._history.append('train', 'training/{}'.format(name), self._step,
                           value)

  def evaluation_round(self, inputs_stream, weights, state, rng):
    """Evaluate.

    Args:
      inputs_stream: iterable of inputs to evaluate on.
      weights: weights for each f in eval_fns.
      state: state for each f in eval_fns.
      rng: random number generator.

    Returns:
      metrics: dict from metric name to metric value averaged over the number of
        inputs.
      state: end state for `predict_fn`.
    """
    metrics = collections.defaultdict(float)
    count = 0
    for inp in inputs_stream:
      count += 1
      rng, subrng = jax_random.split(rng)
      metric_values, _ = self._jit_eval(inp, weights, state, subrng)
      try:
        metric_values = list(metric_values)
      except TypeError:
        metric_values = [float(metric_values)]
      for m, v in zip(self._metrics, metric_values):
        metrics[m] += v
    return {m: v / count for (m, v) in six.iteritems(metrics)}, state

  def update_model_state(self, key, value):
    """Updates model state based on nontrainable_params."""
    # Translate model state keys to nontrainable param names.
    if key in self._nontrainable_param_map:
      p_name = self._nontrainable_param_map[key]
    else:
      # If a key is not in mapping, it stays the same.
      p_name = key
    if p_name in self.nontrainable_params:
      if self._step == 0:
        log('Mapping model state key {} to nontrainable param {}.'
            ''.format(key, p_name))
        return self._for_n_devices(np.array(self.nontrainable_params[p_name]))
    return value

  def update_nontrainable_params(self):
    self._lr_fn = self._lr_schedule(self._history)

  def save_gin(self):
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
    with tf.io.gfile.GFile(weights_file, 'wb') as f:
      pickle.dump(trainer_state_dict, f)
    log('Model saved to %s' % weights_file, stdout=False)

  def save_state(self, keep, prefix='model'):
    """Save trainer state given a possibly replicated opt_state."""
    opt_state = self._opt_state
    if self.n_devices > 1:
      first_replica = lambda x: x[0]
      opt_state = OptState(*math.nested_map(first_replica, opt_state))
    # This line, while optional, allows JAX to transfer arrays from the device
    # to the host in parallel, which is particularly important for cloud TPU.
    if math.backend_name() == 'jax':
      opt_state = jax.device_get(opt_state)
    step, history, model_state = self._step, self._history, self._model_state
    output_dir = self._output_dir

    weights_file = os.path.join(output_dir, prefix + '.pkl')

    # This dict will be stored as the model.
    trainer_state_dict = make_trainer_state_dict(step,
                                                 opt_state,
                                                 history,
                                                 model_state)
    self._save_state_dict(trainer_state_dict, weights_file)

    if keep:
      weights_file = os.path.join(output_dir, '{}_{}.pkl'.format(prefix, step))
      self._save_state_dict(trainer_state_dict, weights_file)

  def save_computation_graphs(self, save_backward_graph):
    """Dump computation graphs to files."""
    if self.n_devices != 1:
      return  # TODO(lukaszkaiser): make this work with more devices.
    batch = next(self._train_stream)
    output_dir = self._output_dir
    if self.n_devices > 1:
      batch = _reshape_by_device(batch, self.n_devices)
    weights = self._opt_state[0][0]
    forward_computation = jax.xla_computation(self._model_predict_eval)(
        batch, weights=weights, state=self._model_state[0],
        rng=self._rngs[0])
    with tf.io.gfile.GFile(os.path.join(output_dir, 'forward.txt'), 'w') as f:
      f.write(forward_computation.GetHloText())
    with tf.io.gfile.GFile(os.path.join(output_dir, 'forward.dot'), 'w') as f:
      f.write(forward_computation.GetHloDotGraph())
    backward_computation = jax.xla_computation(self._jit_update_fn)(
        self._step, self._opt_state, batch, self._model_state,
        self._rngs)
    with tf.io.gfile.GFile(os.path.join(output_dir, 'backward.txt'), 'w') as f:
      f.write(backward_computation.GetHloText())
    if save_backward_graph:  # Backward graphs can be large so we guard it.
      with tf.io.gfile.GFile(
          os.path.join(output_dir, 'backward.dot'), 'w') as f:
        f.write(backward_computation.GetHloDotGraph())

  def log_step(self, step_message):
    log('Step % 6d: %s' % (self.step, step_message))

  def log_metrics(self, metrics, summ_writer, log_prefix):
    """Log metrics to summary writer and history."""
    history = self._history
    rjust_len = max([0] + [len(name) for name in metrics])
    for name, value in six.iteritems(metrics):
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
      single_weights = math.nested_map(unreplicate, opt_state.weights)
      sizes = _sizes(single_weights)
    total_size = _nested_reduce(sum, sizes)
    self.log_step('Total number of trainable weights: %d' % total_size)

  def _init_host_and_devices(self, n_devices=None, random_seed=None):
    """Initializes host and device attributes for this trainer.

    Args:
      n_devices: Number of devices this trainer will use. If `None`, get the
          number from the backend.
      random_seed: Random seed as the starting point for all random numbers used
          by the trainer. If `None`, calculate one from system time and host id.

    Returns:
      is_chief: True if this trainer has special chief responsibilities.
      n_devices: The passed in value of n_devices or a computed default.
      random_seed: The passed in value of random_seed or a computed default.
    """
    if math.backend_name() == 'jax':
      host_id = jax.host_id()
      host_count = jax.host_count()
    else:
      host_id = 0
      host_count = 1
    is_chief = (host_id == 0)

    device_count = math.device_count()
    n_devices = n_devices or device_count
    # TODO(lukaszkaiser): remove this restriction when possible.
    if n_devices != device_count and math.backend_name() == 'jax':
      raise ValueError('JAX cannot work yet with n_devices != all devices: '
                       '%d != %d' % (n_devices, device_count))

    if random_seed is None and host_count > 1:
      random_seed = int(1e6 * (host_id + time.time())) % 2**32
    return is_chief, n_devices, init_random_number_generators(random_seed)

  def _map_to_state_dicts(self, f):
    """Map the function f to all dicts in model state."""
    # TODO(jonni): Can we replace _nested_map with math.nested_map?
    def _nested_map(f, x):
      if isinstance(x, list):
        return [_nested_map(f, y) for y in x]
      if isinstance(x, tuple):
        return tuple([_nested_map(f, y) for y in x])
      if isinstance(x, dict) and len(x) == 1:
        return f(x)
      return x
    return _nested_map(f, self._model_state)

  def _state_dicts_update(self, state_dict):
    assert len(state_dict.keys()) == 1
    key = list(state_dict.keys())[0]
    value = state_dict[key]
    return {key: self.update_model_state(key, value)}

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
    """Replicates/broadcasts `x` for n devices if `self.n_devicess > 1`."""
    return tl.for_n_devices(x, self.n_devices)  # pylint: disable=protected-access


@gin.configurable(blacklist=['output_dir'])
def train(output_dir,
          model=gin.REQUIRED,
          loss_fn=tl.CrossEntropyLoss,
          inputs=trax_inputs.inputs,
          optimizer=trax_opt.Adafactor,
          lr_schedule=lr.MultifactorSchedule,
          trainer_class=Trainer,
          steps=1000,
          checkpoints_at=None,
          eval_steps=10,
          eval_frequency=100,
          random_seed=None,
          save_graphs=True,
          save_backward_graph=False,
          has_weights=False,
          nontrainable_param_map=None,
          id_to_mask=None,
          metrics=None,
          checkpoint_highest=None,
          checkpoint_lowest=None):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fn
      and apply_fn.
    loss_fn: callable with signature: weights, trax.inputs.Inputs, model, state,
      rng -> loss.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer (see optimizers/base.py for signature).
    lr_schedule: A learning rate schedule as a function that takes history and
      returns a function from step to learning rate (a float).
    trainer_class: The trainer class to use.
    steps: int, total number of training steps.
    checkpoints_at: list of integers. Save a checkpoint for each training step
      in the list.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    random_seed: the random seed to use; time/os dependent if None (default).
    save_graphs: bool, if True, save computation graph to file.
    save_backward_graph: bool, if True, save backward graph to file too.
    has_weights: bool, whether weights are included in the inputs.
    nontrainable_param_map: dict, mapping from model nontrainable parameter
      names to control names in PolicySchedule.
    id_to_mask: id to mask out (None by default).
    metrics: optionally override the default metrics dictionary.
    checkpoint_highest: save the checkpoint highest at this metric.
    checkpoint_lowest: save the checkpoint lowest at this metric.

  Returns:
    trax.TrainerState
  """
  n_devices = num_devices()
  # TODO(lukaszkaiser): remove has_weights and id_to_mask (configure loss).
  trainer = trainer_class(model, loss_fn, optimizer, lr_schedule, inputs,
                          output_dir,
                          random_seed=random_seed, n_devices=n_devices,
                          checkpoints_at=checkpoints_at,
                          has_weights=has_weights,
                          nontrainable_param_map=nontrainable_param_map,
                          metrics=metrics, id_to_mask=id_to_mask,
                          checkpoint_lowest=checkpoint_lowest,
                          checkpoint_highest=checkpoint_highest)

  epoch_steps = [steps]  # Only training if eval_frequency is 0 or None
  if eval_frequency and eval_steps > 0:
    epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                   eval_frequency - 1],
                                  itertools.repeat(eval_frequency))
  trainer.log_step('Starting training using %d devices' % trainer.n_devices)
  trainer.print_n_weights()

  for epoch_steps in epochs(steps, trainer.step, epoch_steps):
    trainer.train_epoch(epoch_steps, eval_steps)

    # Update nontrainable parameters with new history
    trainer.update_nontrainable_params()

    # Bookkeeping we do at the first step
    if trainer.step == 1:
      # Save computation graph (single-device only for now)
      if (save_graphs and math.backend_name() == 'jax'):
        trainer.save_computation_graphs(save_backward_graph)

      # Save Gin config
      trainer.save_gin()

  trainer.log_step('Training done')
  return trainer.state


@gin.configurable
def num_devices(value=None):
  """Returns how many devices to use (if None, default, use all available)."""
  return value


@gin.configurable
def _is_jit_init(value=True):
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
    def single_update(i, opt_state, batch, state, rng):
      weights, slots, opt_params = opt_state
      rng, subrng = jax_random.split(rng[0])
      grad_fn = math.grad(model_and_loss_call, has_aux=True)
      grads, state = grad_fn(weights, batch, state, rng)
      return optimizer.tree_update(
          i, grads, weights, slots, opt_params), state, [subrng]
    return math.jit(single_update) if jit else single_update

  # Else, for n_devices > 1:
  @functools.partial(math.pmap, axis_name='batch')
  def mapped_update(i, opt_state, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    weights, slots, opt_params = opt_state
    rng, subrng = jax_random.split(rng)
    grad_fn = math.grad(model_and_loss_call, has_aux=True)
    grads, state = grad_fn(weights, batch, state, rng)
    # We do a psum(1.0) here instead of `n_devices` since `n_devices` is just
    # the number of devices on this host machine, however psum goes over all
    # devices of all hosts (ex: a TPU pod) and we need to be averaging over all
    # of them.
    grads = jax.tree_util.tree_map(
        lambda g: math.psum(g, 'batch') / math.psum(1.0, 'batch'), grads)
    return optimizer.tree_update(
        i, grads, weights, slots, opt_params), state, subrng

  def update(i, opt_state, batch, state, rng):
    return mapped_update(np.repeat(i, n_devices), opt_state, batch, state, rng)

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
    return math.jit(single_compute_loss) if jit else single_compute_loss

  # Else, for n_devices > 1:
  @functools.partial(math.pmap, axis_name='batch')
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
                            model_state):
  """Creates a trainer state dictionary to save to disk.

  Args:
    step: int, a step number
    opt_state: OptState namedtuple
    history: `trax.history.History`, the history object.
    model_state: A nested structure of the model state.

  Returns:
    A dictionary with the fields of TrainerState and OptState flattened.
  """

  return {
      'step': step,
      'weights': opt_state.weights[0],
      'loss_weights': opt_state.weights[1],
      'slots': opt_state.slots,
      'opt_params': opt_state.opt_params,
      'history': history,
      'state': model_state[0],
      'loss_state': model_state[1],
      'version_timestamp': 'Jan-13-2020'  # To update in the future if needed.
  }


def trainer_state_from_dict(trainer_state_dict):
  """Given the trainer state dictionary, returns `TrainerState`."""
  # TODO(afrozm): This becomes simpler if OptState is flattened into
  # TrainerState.
  step = trainer_state_dict['step']
  history = trainer_state_dict['history']
  # TODO(lukaszkaiser): remove the first branch after everyone ports to 'state'.
  if 'model_state' in trainer_state_dict:
    model_state = trainer_state_dict['model_state']
  else:
    model_state = (trainer_state_dict['state'],
                   trainer_state_dict['loss_state'])
  weights = trainer_state_dict['weights']
  # TODO(lukaszkaiser): remove the next 2 lines after 'loss_weights' is in use.
  if 'loss_weights' in trainer_state_dict:
    weights = (weights, trainer_state_dict['loss_weights'])
  opt_state = OptState(
      weights=weights,
      slots=trainer_state_dict['slots'],
      opt_params=trainer_state_dict['opt_params'])
  return TrainerState(step=step, opt_state=OptState(*opt_state),
                      history=history, model_state=model_state)


def load_trainer_state(output_dir, weights_file=None):
  """Returns a TrainerState instance loaded from the given `output_dir`."""
  if weights_file is None:
    weights_file = os.path.join(output_dir, 'model.pkl')
    if not tf.io.gfile.exists(weights_file):
      return TrainerState(step=None, opt_state=None,
                          history=trax_history.History(), model_state=None)
  elif not tf.io.gfile.exists(weights_file):
    raise ValueError('File not found: %s' % weights_file)

  with tf.io.gfile.GFile(weights_file, 'rb') as f:
    trainer_state_dict = pickle.load(f)
  trainer_state = trainer_state_from_dict(trainer_state_dict)
  log('Model loaded from %s at step %d' % (weights_file, trainer_state.step))
  logging.debug('From loaded model : history = %s', trainer_state.history)
  return trainer_state




def init_random_number_generators(seed=None):
  """Initializes random generators for Python, NumPy, TensorFlow, and JAX."""
  # Seed Python random (None as seed is okay), then use it to seed the others.
  random.seed(seed)
  if seed is None:
    seed = random.randint(0, 2**31 - 1)
  numpy.random.seed(seed)
  tf.random.set_seed(seed)
  return jax_random.get_prng(seed)


def _stack_inputs_targets_and_get_predictions(inputs_and_targets):
  """Helper to stack inputs and targets and retrieve predictions from output."""
  # Inputs and targets can be lists - we build a flat one to input to the model.
  model_inp = []
  for x in inputs_and_targets:
    if not isinstance(x, (list, tuple)):
      model_inp.append(x)
    else:
      model_inp.extend(x)
  # We retrieve as many predictions from model output as many there were inputs.
  inp = inputs_and_targets[0]
  inp_len = len(inp) if isinstance(inp, (list, tuple)) else 1
  get_pred = lambda x: x[0] if inp_len == 1 else x[:inp_len]
  return tuple(model_inp), get_pred


def _reshape_by_device(x, n_devices):
  """Reshapes possibly nested x into a shape (n_devices, ...)."""
  return tl.reshape_by_device(x, n_devices)  # pylint: disable=protected-access


def _nested_reduce(f, x):
  """Fold the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return f([_nested_reduce(f, y) for y in x])
  if isinstance(x, tuple):
    return f([_nested_reduce(f, y) for y in x])
  return x


def _sizes(x):
  """Get a structure of sizes for a structure of nested arrays."""
  def size(x):
    try:
      return x.size
    except Exception:  # pylint: disable=broad-except
      return 0
  return math.nested_map(size, x)


def _repeat_stream(stream, n_devices):
  """Repeat a stream indefinitely."""
  while True:
    for example in stream(n_devices):
      yield example
