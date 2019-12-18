# coding=utf-8
# Copyright 2019 The Trax Authors.
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
import random
import sys
import time

from absl import logging

import gin

import jax
import numpy
import six
import tensorflow.compat.v2 as tf
from trax import backend
from trax import history as trax_history
from trax import jaxboard
from trax import layers as tl
from trax import learning_rate as lr
from trax import optimizers as trax_opt
from trax import utils
from trax.backend import numpy as np
from trax.backend import random as jax_random
from trax.shapes import ShapeDtype
from trax.supervised import inputs as trax_inputs


TrainerState = collections.namedtuple('_TrainerState', [
    'step',       # Current training step number.
    'opt_state',  # OptState.
    'history',    # trax.history.History.
    'model_state',
])


OptState = collections.namedtuple('_OptState', [
    'weights',     # Model weights.
    'slots',       # Per-parameter optimizer state, e.g. gradient moments.
    'opt_params',  # Optimizer (hyper)parameters, e.g. learning rate, momentum.
])


class Trainer(object):
  """Trax trainer.

  A trainer allows to make training steps, train for full epochs,
  save the training state and access evaluation data.
  """

  def __init__(self, model, loss_fn, optimizer, lr_schedule, inputs,
               output_dir=None, random_seed=None, n_devices=None,
               save_steps=None, should_save_checkpoints=True,
               should_write_summaries=True, has_weights=False,
               nontrainable_param_map=None, mask_id=None, metrics=None):
    if backend.get_name() == 'jax':
      self._host_id = jax.host_id()
      self._host_count = jax.host_count()
    else:
      self._host_id = 0
      self._host_count = 1
    self._is_chief = (self._host_id == 0)

    if save_steps is None:
      save_steps = []
    self._save_steps = save_steps
    self._should_save_checkpoints = should_save_checkpoints
    self._should_write_summaries = should_write_summaries
    self._has_weights = has_weights
    self._mask_id = mask_id
    self._metrics_dict = _METRICS if metrics is None else metrics
    loss_fn = loss_fn(has_weights=has_weights, mask_id=mask_id)
    device_count = backend.device_count()
    n_devices = n_devices or device_count
    # TODO(lukaszkaiser): remove this restriction when possible.
    if n_devices != device_count and backend.get_name() == 'jax':
      raise ValueError('JAX cannot work yet with n_devices != all devices: '
                       '%d != %d' % (n_devices, device_count))
    self._n_devices = n_devices

    # Simple differential seeding of RNG across hosts by host_id and time.
    if random_seed is None and self._host_count > 1:
      _, random_seed = divmod(int(time.time() * 1e6) +
                              int(self._host_id * 1e6), 2**32)
    rng = get_random_number_generator_and_set_seed(random_seed)
    inputs = inputs(n_devices)
    self._inputs = inputs

    # Initialize the learning rate to a dummy value. It will be set in reset().
    opt = optimizer(learning_rate=0.0)

    # Setup the model.
    model_train = model(mode='train')
    model_predict_eval = model(mode='eval')

    # Setup state.
    rng, init_rng = jax_random.split(rng)
    self._rngs = np.stack(jax_random.split(rng, n_devices))
    first_shape = inputs.input_shape[0]
    # If the inputs are a tuple/list, add [None] (batch) to each element.
    if isinstance(first_shape, (list, tuple)):
      model_input_shape = tuple(
          tuple([None] + list(shape)) for shape in inputs.input_shape)
      model_target_shape = tuple(
          tuple([None] + list(shape)) for shape in inputs.target_shape)
    else:  # Otherwise just add [None] to the input shape.
      model_input_shape = tuple([None] + list(inputs.input_shape))
      model_target_shape = tuple([None] + list(inputs.target_shape))
    # Change all None to 1 in input and target shape.
    model_input_shape = backend.nested_map(lambda x: x or 1, model_input_shape)
    model_target_shape = backend.nested_map(lambda x: x or 1,
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
      new_opt_state_and_model_state = backend.jit(new_opt_state_and_model_state,
                                                  static_argnums=(0, 1, 2, 3))
    self._new_opt_state_and_model_state = (
        lambda: new_opt_state_and_model_state(  # pylint: disable=g-long-lambda
            model_input_shape, self._inputs.input_dtype,
            model_target_shape, self._inputs.target_dtype, init_rng))

    # jit model_predict and update so they're fast
    # TODO(lukaszkaiser): the code below creates a layer computing
    # multiple metrics from a single model output; re-factor for clarity.
    dup_layer = tl.Dup3() if self._has_weights else tl.Dup2()
    def lower(layer):
      """Apply layer below the current inputs, targets, and possibly weights."""
      if self._has_weights:
        # Apply layer below inputs, targets, and loss weights.
        return tl.Parallel([], [], [], layer)
      else:
        # Apply layer below inputs and targets.
        return tl.Parallel([], [], layer)
    metrics_layer = []
    self._metrics = list(sorted(self._metrics_dict.keys()))
    for i, m in enumerate(reversed(self._metrics)):
      metric = self._metrics_dict[m](has_weights=self._has_weights,
                                     mask_id=self._mask_id)
      if i != len(self._metrics) - 1:
        metrics_layer.append(dup_layer)
        metrics_layer.append(lower(metric))
      else:
        metrics_layer.append(metric)
    # TODO(lukaszkaiser): clean this up once layer API stabilizes.
    # For now, we need to initialize metric layers somehow, so here we go.
    # We assume that they do not have any parameters, so this is a dummy.
    dummy_shapes = ((1, 2), (1,), (1,)) if self._has_weights else ((1, 2), (1,))
    dummy_dtypes = [np.float32] * (3 if self._has_weights else 2)
    dummy_signature = tuple(ShapeDtype(s, d)
                            for s, d in zip(dummy_shapes, dummy_dtypes))
    metrics_layer = tl.Serial(metrics_layer)
    metrics_layer._set_rng_recursive(init_rng)  # pylint: disable=protected-access
    metrics_weights, metrics_state = (
        metrics_layer.init(dummy_signature))
    self._metrics_weights = self._for_n_devices(metrics_weights)
    self._metrics_state = self._for_n_devices(metrics_state)
    self._jit_eval = _jit_predict_fn(
        model_predict_eval, metrics_layer, n_devices)
    self._jit_update_fn = _jit_update_fn(model_train, loss_fn, opt, n_devices)

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
    # Currently we need ot pick [0] as we ignore loss weights (empty).
    return self._opt_state.weights[0]

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
    with backend.use_backend('numpy'):
      return self._lr_fn(self._step)

  def reset(self, output_dir):
    """Reset the model parameters.

    Restores the parameters from the given output_dir if a checkpoint exists,
    otherwise randomly initializes them.

    Does not re-jit the model.

    Args:
      output_dir: Output directory.
    """
    self._output_dir = output_dir
    tf.io.gfile.makedirs(output_dir)
    # Create summary writers and history.
    if self._should_write_summaries:
      self._train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, 'train'),
                                              enable=self._is_chief)
      self._eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, 'eval'),
                                             enable=self._is_chief)

    # Reset the train and eval streams.
    self._train_stream = self._inputs.train_stream()
    # TODO(lukaszkaiser): add an option to evaluate exactly on the full eval
    #   set by adding a padding and stopping the stream when too large.
    self._eval_stream = _repeat_stream(self._inputs.eval_stream)
    self._train_eval_stream = _repeat_stream(self._inputs.train_eval_stream)

    # Restore the training state.
    state = load_trainer_state(output_dir)
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
    if not state.opt_state and self._should_save():
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
    if self._should_save():
      self.save_state(keep=False)

  def train_step(self, batch):
    """Run one training step and update self._opt_state."""
    # Calculate the current optimizer parameters.
    # TODO(pkozakowski): Optimizer parameters get polluted with model state,
    # which doesn't break anything but is weird. Filter it out.
    opt_param_updates = self._for_n_devices(
        backend.nested_map(np.array, self.nontrainable_params))
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
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    sw = self._train_sw
    if sw:
      sw.text('gin_config',
              jaxboard.markdownify_operative_config_str(config_str))

  def save_state(self, keep):
    """Save trainer state given a possibly replicated opt_state."""
    opt_state = self._opt_state
    if self.n_devices > 1:
      first_replica = lambda x: x[0]
      opt_state = OptState(*backend.nested_map(first_replica, opt_state))
    # This line, while optional, allows JAX to transfer arrays from the device
    # to the host in parallel, which is particularly important for cloud TPU.
    if backend.get_name() == 'jax':
      opt_state = jax.device_get(opt_state)
    step, history, model_state = self._step, self._history, self._model_state
    output_dir = self._output_dir

    pkl_module = utils.get_pickle_module()
    weights_file = os.path.join(output_dir, 'model.pkl')
    with tf.io.gfile.GFile(weights_file, 'wb') as f:
      pkl_module.dump((tuple(opt_state), step, history, model_state), f)
    if keep:
      weights_file = os.path.join(output_dir, 'model_{}.pkl'.format(step))
      with tf.io.gfile.GFile(weights_file, 'wb') as f:
        pkl_module.dump((tuple(opt_state), step, history, model_state), f)
    log('Model saved to %s' % weights_file, stdout=False)

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
      single_weights = backend.nested_map(unreplicate, opt_state.weights)
      sizes = _sizes(single_weights)
    total_size = _nested_reduce(sum, sizes)
    self.log_step('Total number of trainable weights: %d' % total_size)

  def _map_to_state_dicts(self, f):
    """Map the function f to all dicts in model state."""
    # TODO(jonni): Can we replace _nested_map with backend.nested_map?
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

  def _should_save(self):
    return self._is_chief and self._should_save_checkpoints

  def _should_save_now(self):
    return self._should_save() and self._step in self._save_steps

  def _should_log_now(self):
    return (self._train_sw is not None
            and (self._step == 1 or self._step % 10 == 0))

  def _for_n_devices(self, x):
    """Replicates/broadcasts `x` for n devices if `self.n_devicess > 1`."""
    n = self.n_devices
    def f(x):
      if n > 1 and backend.get_name() == 'jax':
        return _multi_device_put(x)
      elif n > 1:
        return np.broadcast_to(x, (n,) + x.shape)
      else:
        return x
    return backend.nested_map(f, x)


@gin.configurable(blacklist=['output_dir'])
def train(output_dir,
          model=gin.REQUIRED,
          loss_fn=tl.CrossEntropyLossScalar,
          inputs=trax_inputs.inputs,
          optimizer=trax_opt.Adafactor,
          lr_schedule=lr.MultifactorSchedule,
          trainer_class=Trainer,
          train_steps=1000,
          save_steps=None,
          eval_steps=10,
          eval_frequency=100,
          random_seed=None,
          save_graphs=True,
          save_backward_graph=False,
          has_weights=False,
          nontrainable_param_map=None,
          mask_id=None,
          metrics=None):
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
    train_steps: int, total number of training steps.
    save_steps: list of integers. Keep a model file at each of the supplied save
      steps.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    random_seed: the random seed to use; time/os dependent if None (default).
    save_graphs: bool, if True, save computation graph to file.
    save_backward_graph: bool, if True, save backward graph to file too.
    has_weights: bool, whether weights are included in the inputs.
    nontrainable_param_map: dict, mapping from model nontrainable parameter
      names to control names in PolicySchedule.
    mask_id: id to mask out (None by default).
    metrics: optionally override the default metrics dictionary.

  Returns:
    trax.TrainerState
  """
  n_devices = num_devices()
  # TODO(lukaszkaiser): remove has_weights and mask_id later (configure loss).
  trainer = trainer_class(model, loss_fn, optimizer, lr_schedule, inputs,
                          output_dir,
                          random_seed=random_seed, n_devices=n_devices,
                          save_steps=save_steps, has_weights=has_weights,
                          nontrainable_param_map=nontrainable_param_map,
                          metrics=metrics, mask_id=mask_id)

  epoch_steps = [train_steps]  # Only training if eval_frequency is 0 or None
  if eval_frequency and eval_steps > 0:
    epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                   eval_frequency - 1],
                                  itertools.repeat(eval_frequency))
  trainer.log_step('Starting training using %d devices' % trainer.n_devices)
  trainer.print_n_weights()

  for epoch_steps in epochs(train_steps, trainer.step, epoch_steps):
    trainer.train_epoch(epoch_steps, eval_steps)

    # Update nontrainable parameters with new history
    trainer.update_nontrainable_params()

    # Bookkeeping we do at the first step
    if trainer.step == 1:
      # Save computation graph (single-device only for now)
      if (save_graphs and backend.get_name() == 'jax'):
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
      grad_fn = backend.grad(model_and_loss_call, has_aux=True)
      grads, state = grad_fn(weights, batch, state, rng)
      return optimizer.tree_update(
          i, grads, weights, slots, opt_params), state, [subrng]
    return backend.jit(single_update) if jit else single_update

  # Else, for n_devices > 1:
  @functools.partial(backend.pmap, axis_name='batch')
  def mapped_update(i, opt_state, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    weights, slots, opt_params = opt_state
    rng, subrng = jax_random.split(rng)
    grad_fn = backend.grad(model_and_loss_call, has_aux=True)
    grads, state = grad_fn(weights, batch, state, rng)
    # We do a psum(1.0) here instead of `n_devices` since `n_devices` is just
    # the number of devices on this host machine, however psum goes over all
    # devices of all hosts (ex: a TPU pod) and we need to be averaging over all
    # of them.
    grads = jax.tree_util.tree_map(
        lambda g: backend.psum(g, 'batch') / backend.psum(1.0, 'batch'), grads)
    return optimizer.tree_update(
        i, grads, weights, slots, opt_params), state, subrng

  def update(i, opt_state, batch, state, rng):
    return mapped_update(np.repeat(i, n_devices), opt_state, batch, state, rng)

  return update


@gin.configurable
def _jit_predict_fn(model_predict, metric_fn, n_devices, jit=True):
  """Returns a JIT-compiled predict function (unless jit=False)."""
  model = tl.Serial(model_predict, metric_fn)
  model_predict = model._forward_internal  # pylint: disable=protected-access
  if not jit:
    return model_predict

  model_predict = backend.accelerate(model_predict, n_devices)
  if n_devices == 1:
    return model_predict

  def predict(x, weights, state, rng):
    """Predict function jited and parallelized as requested."""
    res, state = _combine_devices(model_predict(
        _reshape_by_device(x, n_devices),
        weights,
        state,
        np.stack(jax_random.split(rng, n_devices))))
    return backend.nested_map(lambda y: np.mean(y, axis=0), res), state

  return predict


@gin.configurable
def _jit_compute_loss_fn(predict_fn, loss_fn, n_devices, jit=True):
  """Returns a (JIT-compiled) function that computes the loss for one step."""
  if n_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_compute_loss(opt_state, batch, state, rng):
      rng, subrng = jax_random.split(rng[0])
      loss_val, state = loss_fn(opt_state[0], batch, predict_fn, state, rng)
      return loss_val, state, [subrng]
    return backend.jit(single_compute_loss) if jit else single_compute_loss

  # Else, for n_devices > 1:
  @functools.partial(backend.pmap, axis_name='batch')
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


def load_trainer_state(output_dir):
  """Returns a TrainerState instance loaded from the given `output_dir`."""
  weights_file = os.path.join(output_dir, 'model.pkl')
  if not tf.io.gfile.exists(weights_file):
    return TrainerState(step=None, opt_state=None,
                        history=trax_history.History(), model_state=None)

  pkl_module = utils.get_pickle_module()
  with tf.io.gfile.GFile(weights_file, 'rb') as f:
    (opt_state, step, history, model_state) = pkl_module.load(f)
  log('Model loaded from %s at step %d' % (weights_file, step))
  logging.debug('From loaded model : history = %s', history)
  return TrainerState(step=step, opt_state=OptState(*opt_state),
                      history=history, model_state=model_state)


def get_random_number_generator_and_set_seed(seed=None):
  """Get a JAX random number generator and set random seed everywhere."""
  random.seed(seed)
  # While python random accepts None as seed and uses time/os seed then,
  # some other functions expect integers so we create one here.
  if seed is None:
    seed = random.randint(0, 2**31 - 1)
  tf.random.set_seed(seed)
  numpy.random.seed(seed)
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


def _multi_device_put(x, devices=None):
  """Memory efficient multi-device replication / broadcast in JAX.

  JAX uses a ShardedDeviceArray class that holds a list of device buffers
  on separate devices for use with pmap'd computations.  Sharded arrays
  are explicitly used to eliminate unneccessary inter-device transfer of
  memory buffers between use in pmap'd computations.  The JAX API currently
  does not have a multi-device 'put' function that copies a buffer onto
  N devices in a memory-efficient fashion, so we implement our own here.

  Args:
    x: jax DeviceArray or numpy ndarray to be replicated.
    devices: a jax.devices() list or subset thereof of devices to
      replicate onto.  Should match the list passed to any pmaps
      ingesting the replicated array.

  Returns:
    A ShardedDeviceArray with
    dtype = x.dtype and shape = (n_devices,) + x.shape
    that's backed by replicated device_buffers on each local device.
  """
  # Convert _FilledConstants that don't have device_buffer, etc.
  if type(x) != jax.xla.DeviceArray:  # pylint: disable=unidiomatic-typecheck
    x = np.array(x)
  # Calculate the abstract shape of the replicated array.
  if not devices:
    devices = jax.local_devices()
  n_devices = len(devices)
  x_aval = jax.xla.abstractify(x)
  broadcast_x_aval = jax.abstract_arrays.ShapedArray(
      (n_devices,) + x_aval.shape,
      x_aval.dtype)
  # Create copies of the underlying device buffer for each local device.
  broadcast_buffers = [
      jax.device_put(x, dv).device_buffer
      for dv in devices
  ]
  return jax.pxla.ShardedDeviceArray(broadcast_x_aval, broadcast_buffers)


def _reshape_by_device(x, n_devices):
  """Reshapes possibly nested x into a shape (n_devices, ...)."""
  def f(x):
    x_shape = list(x.shape)
    batch_size = x_shape[0]
    batch_size_per_device = batch_size // n_devices
    if batch_size_per_device * n_devices != batch_size:
      raise ValueError(
          'We require that n_devices[%d] divides batch_size[%d] evenly.' %
          (n_devices, batch_size))
    new_shape_prefix = [n_devices, batch_size_per_device]
    return backend.numpy.reshape(x, new_shape_prefix + x_shape[1:])
  return backend.nested_map(f, x)


def _combine_devices(x_tuple):
  """Combine multi-device tensors into a single batch."""
  def f(x):
    if len(x.shape) < 2:
      return x  # No extra batch dimension: use devices as batch, so return.
    batch_size = x.shape[0] * x.shape[1]
    return backend.numpy.reshape(x, [batch_size] + list(x.shape[2:]))
  return backend.nested_map(f, x_tuple)


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
  return backend.nested_map(size, x)


# Metrics to calculate and report.
_METRICS = {
    'accuracy': tl.AccuracyScalar,
    'neg_log_perplexity': tl.NegLogPerplexityScalar,
    'loss': tl.CrossEntropyLossScalar,
    'weights_per_batch_per_core': tl.CountWeights,
}


def _repeat_stream(stream):
  """Repeat a stream indefinitely."""
  while True:
    for example in stream():
      yield example
