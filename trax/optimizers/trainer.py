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
"""Multi-device accelerated optimization."""

from concurrent import futures
import functools
import os
import time

from absl import logging
import jax
import numpy as np
import psutil

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb


class Trainer:
  """Multi-device accelerated trainer.

  Given an optimizer and a composite layer containing model+loss, this class
  creates a multi-device accelerated function with which it can compute one step
  of updates to the model's weights/state and the optimizer slots. By default
  it uses all available accelerators, via JIT compilation and parallel mapping.

  The optimizer and model must be initialized prior to use by this class.

  The key `one_step` function runs one forward-backward pass through the model,
  and returns the resulting loss value and updated optimizer statistics. As a
  side effect, the function also modifies the model weights and optimizer slots.
  """

  def __init__(self, model_with_loss, optimizer, n_devices=None, adasum=False):
    self._model_with_loss = model_with_loss
    self._optimizer = optimizer
    self._n_devices = n_devices or fastmath.local_device_count()
    self._adasum = adasum

    # optimizer slots and opt_params may need to be replicated
    self._slots, self._opt_params = tl.on_cpu(tl.for_n_devices(
        (self._optimizer.slots, self._optimizer.opt_params), self._n_devices))

    # accelerated version of model+loss to replicate weights and state
    self._accelerated_model_with_loss = tl.Accelerate(
        model_with_loss, n_devices=n_devices)

    # Signature:
    # (batch, weights, state, rng) -> ((loss, state), gradients)
    self._forward_and_backward_fn = (
        fastmath.value_and_grad(
            model_with_loss.pure_fn,
            argnums=1,  # arg1 of pure_fn: weights
            has_aux=True))  # return (loss, state), gradients

    # Signature:
    # (weights, slots), step, opt_params, batch, state, rng ->
    # (weights, slots), state, stats
    self._accelerated_update_fn = (
        _accelerate_update_fn(
            self._forward_and_backward_fn,
            self._optimizer,
            n_devices=self._n_devices,
            accelerate=True,
            adasum=self._adasum
        )
    )

  @property
  def model_with_loss(self):
    """Returns the composite model+loss for this instance."""
    return self._model_with_loss

  @property
  def accelerated_model_with_loss(self):
    """Returns the accelerated composite model+loss for this instance."""
    return self._accelerated_model_with_loss

  @property
  def optimizer(self):
    """Returns the optimizer for this instance."""
    return self._optimizer

  @property
  def slots(self):
    """Returns the slots of the optimizers."""
    return self._optimizer.slots

  @slots.setter
  def slots(self, slots):
    """Sets the slots of the optimizers and this class (replicated)."""
    self._optimizer.slots = slots
    self._slots = tl.on_cpu(tl.for_n_devices(slots, self._n_devices))

  def one_step(self, batch, rng, step=0, learning_rate=None):
    """Runs one training step, to update model and optimizer parameters.

    Args:
      batch: Batch of labeled training data.
      rng: Single-use random number generator (JAX PRNG key).
      step: Training step number.
      learning_rate: Learning rate for the optimizer; if None, use optimizer's
          default learning rate.

    Returns:
      Tuple of (loss, optimizer_stats), with the newly computed loss and
      updated stats as reported by the optimizer.
    """
    if learning_rate is not None:
      self._opt_params['learning_rate'] = tl.for_n_devices(
          learning_rate, self._n_devices)

    # Split the batch across devices (batch_dim --> batch_dim // n_devices)
    # and create new rng's 1-1 with devices.
    if self._n_devices > 1:
      batch = tl.reshape_by_device(batch, self._n_devices)
      rng = jnp.stack(fastmath.random.split(rng, self._n_devices))

    weights = self._accelerated_model_with_loss.weights
    state = self._accelerated_model_with_loss.state
    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      # Prints every power of two, if debugging is enabled.
      logging.info('step[%d]', step)
      logging.info('opt_params[%s]', self._opt_params)
      logging.info('slots[%s]', self._slots)
      logging.info('weights[%s]', weights)
      logging.info('state[%s]', state)

    # NOTE: stats is a replicated dictionary of key to jnp arrays.
    (new_weights, new_slots), new_state, stats = self._accelerated_update_fn(
        (weights, self._slots), step, self._opt_params, batch, state, rng)

    if logging.vlog_is_on(1) and ((step & step - 1) == 0):
      logging.info('updated weights[%s]', new_weights)
      logging.info('stats[%s]', stats)

    self._accelerated_model_with_loss.weights = new_weights
    self._accelerated_model_with_loss.state = new_state
    self._slots = new_slots
    self._optimizer.slots = self._unreplicate(self._slots)
    return stats['loss'], stats

  def _unreplicate(self, x):
    if self._n_devices == 1:
      return x
    return fastmath.nested_map(lambda x: x[0], x)


def _adasum_merge(g1, g2):
  """Adasum gradient composition, see https://arxiv.org/pdf/2006.02924.pdf."""
  frac1 = jnp.vdot(g1, g2) / (2 * jnp.vdot(g1, g1) + 1e-30)
  frac2 = jnp.vdot(g1, g2) / (2 * jnp.vdot(g2, g2) + 1e-30)
  return  (1 - frac1) * g1 + (1 - frac2) * g2


def _average_multidevice_gradients(gradients, adasum=False):
  """Averages gradients over all the devices across different hosts."""
  n = fastmath.global_device_count() // base.N_WEIGHTS_SHARDS
  if adasum:
    # This implements a version of the Adasum algorithm from the following
    # paper: https://arxiv.org/pdf/2006.02924.pdf
    lg = max([i for i in range(20) if 2**i <= n])
    for lg_i in range(lg):
      shift = 2**lg_i
      perm = []
      for i in range(n):
        block_i = i % (2*shift)  # we do blocks of 2*shift size
        if block_i < shift:
          perm.append((i, i+shift))
        else:
          perm.append((i, i-shift))
      perm_grad = jax.lax.ppermute(gradients, perm=perm, axis_name='batch')
      gradients = fastmath.nested_map_multiarg(
          _adasum_merge, gradients, perm_grad)
  if base.N_WEIGHTS_SHARDS > 1:  # only sum gradients from matching shards
    groups = [[base.N_WEIGHTS_SHARDS * i + d for i in range(int(n))]
              for d in range(base.N_WEIGHTS_SHARDS)]
    gradients_psum = fastmath.psum(gradients, 'batch',
                                   axis_index_groups=groups)
  else:
    gradients_psum = fastmath.psum(gradients, 'batch')  # sum all gradients
  n = jnp.array(n, dtype=jnp.float32)
  return fastmath.nested_map(lambda g: g / n, gradients_psum)


# Returns a function with the following signature:
# (weights, slots), step, opt_params, batch, state, rng ->
# (weights, slots), state, stats
def _accelerate_update_fn(forward_and_backward_fn,
                          optimizer,
                          n_devices,
                          accelerate=True,
                          adasum=False):
  """Accelerates the given forward_and_backward_fn function."""
  if n_devices == 1:
    def single_device_update_fn(
        weights_and_slots, step, opt_params, batch, state, rng):
      step = jnp.array(step, dtype=jnp.int32)  # Needed in TFNP backend.
      weights, slots = weights_and_slots
      (loss, state), gradients = forward_and_backward_fn(
          batch, weights, state, rng)
      weights, slots, stats = optimizer.tree_update(
          step, gradients, weights, slots, opt_params, store_slots=False)
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
    # All tensors should have the first dimension = n_devices.
    weights, slots = weights_and_slots
    (loss, state), gradients = (
        forward_and_backward_fn(batch, weights, state, rng))
    gradients = _average_multidevice_gradients(gradients, adasum=adasum)
    weights, slots, stats = optimizer.tree_update(
        step, gradients, weights, slots, opt_params, store_slots=False)
    stats['loss'] = loss
    return (weights, slots), state, stats

  def multi_device_update_fn(
      weights_and_slots, step, opt_params, batch, state, rng):
    # Need to replicate step to n_devices leading dimension.
    return _multi_device_update_fn(weights_and_slots,
                                   jnp.repeat(step, n_devices), opt_params,
                                   batch, state, rng)

  return multi_device_update_fn


class ReversibleSerialTrainer:
  """Runs an optimizer on a series of layers, reversible and not.

  We provide layers to this trainer in blocks, each block consisting of
  a list of standard layers and a list of reversible layers. They all run
  in turn (like one huge Serial block) but in a more memory-efficient way.

  The main motivation for this class is to save memory: it allows to train
  models that have more weights than the memory available on accelerators.
  This happens by caching the weights in CPU memory and transferring only
  the weights of one layer at a time. The reversible layers are used to make
  the backward pass without using additional memory for storing activations.

  Note: we do not allow sharing weights between blocks for now.
  """

  def __init__(self, blocks, loss_layer, optimizer_fn, n_devices=None,
               memoize_jit=True, free_accelerators_on_step=False, adasum=False):
    """Creates a ReversibleSerialTrainer and the needed optimizers.

    This trainer performs updates equivalent to using the default Trainer on::

      tl.Serial(blocks + [loss_layer]).

    It is more memory-efficient though since weights are stored on CPU and only
    sent to accelerator layer-by-layer. Blocks are pairs consisting of a list
    of standard (arbitrary) layers and a list of reversible layers which help
    save memory thanks to being reversible.

    Args:
      blocks: A list of pairs of lists of standard and reversible layers.
      loss_layer: The final layer of the model; it can have trainable weights
        but should end with a loss: it is required to produce a scalar output.
      optimizer_fn: A function to create the optimizer, e.g., `optimizers.Adam`.
      n_devices: An optional integer, number of accelerator devices to use;
        by default, all available accelerators will be used.
      memoize_jit: Whether to memoize JITed functions; this significantly speeds
        up XLA compilation of larger models, but it uses `repr(layer)` as keys
        to memoize so it could fail if two layers with different functionality
        had the same string representaion. We have not encountered such case
        yet so this is turned on by default, but consider turning it off or
        reviewing your model if you use custom layers and encounter a problem.
      free_accelerators_on_step: If true, frees memory on accelerators when
        starting a step. All layers and arguments must be on host for that,
        otherwise it can lead to failures. Can prevent memory fragmentation.
      adasum: if True, use adaptive summation to gather multi-device gradients.
    """
    self._blocks = [(tl.Serial(std), rev) for (std, rev) in blocks]
    self._loss_layer = loss_layer
    self._optimizer_fn = optimizer_fn
    self._n_devices = n_devices or fastmath.local_device_count()
    self._adasum = adasum
    self._n_layers = 1 + sum([len(revs) + 1 for (_, revs) in self._blocks])
    self._n_steps_per_log = 100  # Log layers and stats every 100 steps.
    self._n_async_layers = 1  # How many layers to run asynchronously.
    self._jit_memory = {} if memoize_jit else None
    self._do_free = free_accelerators_on_step
    self._jit_per_device_rngs = fastmath.jit(
        self._per_device_rngs, backend='cpu')

    # Create accelerated versions of layers as pmaped/jited pure_fn.
    self._accelerated_layer_fns = fastmath.nested_map(
        lambda layer: self._pjit(layer.pure_fn, f'fwd {repr(layer)}'),
        self._blocks)

    # Create per-layer optimizers and replicate opt_params.
    def _make_optimizer(layer):
      opt = optimizer_fn()
      opt.tree_init(layer.weights)
      opt.slots = tl.on_cpu(opt.slots)
      return opt

    self._optimizers = fastmath.nested_map(_make_optimizer, self._blocks)
    self._replicated_opt_params = fastmath.nested_map(
        lambda opt: self._replicate_cpu(opt.opt_params), self._optimizers)

    self._loss_opt = _make_optimizer(loss_layer)
    self._replicated_loss_opt_params = self._replicate_cpu(
        self._loss_opt.opt_params)

    # Forward + backward + optimizer-update functions for all layers.
    # We call them in short FBO for "Forward + Backward + Optimizer update".
    # Reversible layers define a reverse_and_fbo function that also reverses.

    self._fbos = []
    for i, (std_layer, rev_layers) in enumerate(self._blocks):
      (std_opt, rev_opts) = self._optimizers[i]
      std_fbo = _fbo_with_layer_and_opt(
          std_layer, std_opt, self._n_devices, adasum=self._adasum)
      rev_and_fbos = []
      for layer, opt in zip(rev_layers, rev_opts):
        rev_and_fbo = _reverse_and_fbo_with_layer_and_opt(
            layer, opt, self._n_devices, self._adasum)
        # The donated args are (outputs, weights, grads) and we can donate
        # them because weights and grads are immediately replaced and in
        # case of reversible layers, the outputs are never used again.
        rev_and_fbos.append(self._pjit(
            rev_and_fbo, f'rev+bwd {repr(layer)}', donate_argnums=(0, 1, 2)))
      # In standard layers, the inputs cannot be donated as they may be used
      # as outputs for the reversible block below, but weights and grads can.
      jit_std_fbo = self._pjit(
          std_fbo, f'bwd {repr(std_layer)}', donate_argnums=(1, 2))
      self._fbos.append((jit_std_fbo, rev_and_fbos))

    loss_fbo = _fbo_with_layer_and_opt(
        self._loss_layer, self._loss_opt, self._n_devices, 'loss', self._adasum)
    self._loss_fbo = self._pjit(loss_fbo, donate_argnums=(1, 2))

  @property
  def loss_layer(self):
    """Returns the loss layer used to initialize this class."""
    return self._loss_layer

  @property
  def all_layers(self):
    """Returns all layers that compose the model and loss in this class."""
    layers = []
    for (std_layer, rev_layers) in self._blocks:
      layers.append(std_layer)
      layers.extend(rev_layers)
    layers.append(self._loss_layer)
    return layers

  @property
  def optimizer_fn(self):
    """Returns the optimizer function used to initialize this class."""
    return self._optimizer_fn

  @property
  def slots(self):
    """Returns the slots of all optimizers."""
    optimizers = list(self._optimizers) + [self._loss_opt]
    return fastmath.nested_map(lambda opt: opt.slots, optimizers)

  @slots.setter
  def slots(self, slots):
    """Sets the slots of all optimizers."""
    for ((s_opt, r_opts), (s_slots, r_slots)) in zip(
        self._optimizers, slots[:-1]):
      for (opt, slot) in zip([s_opt] + r_opts, [s_slots] + r_slots):
        opt.slots = slot
    self._loss_opt.slots = slots[-1]

  def _pjit(self, f, memory_key=None, donate_argnums=()):
    """JIT f if 1 device is available and pmap if more are available."""
    should_memoize = self._jit_memory is not None and memory_key is not None
    if (should_memoize and memory_key in self._jit_memory):
      logging.info('Found JITed function in memory for: %s', memory_key)
      return self._jit_memory[memory_key]
    if self._n_devices == 1:
      res = fastmath.jit(f, donate_argnums=donate_argnums)
    else:
      res = fastmath.pmap(f, axis_name='batch', donate_argnums=donate_argnums)
    if should_memoize:
      self._jit_memory[memory_key] = res
    return res

  def _replicate(self, x):
    if self._n_devices > 1:
      return tl.for_n_devices(x, self._n_devices)
    return tl.on_accelerator(x)

  def _replicate_cpu(self, x):
    # TODO(lukaszkaiser): move it to layers/acceleration to be together with
    #   tl.for_n_devices and other functions like that, possibly refactor them.
    def f(x):
      if self._n_devices > 1:
        return np.broadcast_to(x, (self._n_devices,) + np.asarray(x).shape)
      else:
        return x
    return tl.on_cpu(fastmath.nested_map(f, x))

  def _unreplicate(self, x):
    if self._n_devices == 1:
      return tl.on_cpu(x)
    return tl.on_cpu(fastmath.nested_map(lambda x: x[0], x))

  def _lazy_unreplicate(self, x):
    def unreplicate_and_start_async_copy(y):
      unreplicated = y if self._n_devices == 1 else y[0]
      unreplicated.copy_to_host_async()
      return unreplicated
    return fastmath.nested_map(unreplicate_and_start_async_copy, x)

  def _collect_weights(self, layer):
    layer.weights = fastmath.nested_map(np.asarray, layer.weights)

  def _free_accelerators(self, exceptions=(), keep_constants=True):
    """Deletes all live buffers from accelerator with no safety guarantees."""
    backend = jax.lib.xla_bridge.get_backend()
    live_buffers = backend.live_buffers()
    logging.info('Deleting %d live buffers.', len(live_buffers))
    exceptions_buffers = []
    for x in fastmath.tree_flatten(exceptions):
      if hasattr(x, 'device_buffer'):  # DeviceArray
        exceptions_buffers.append(x.device_buffer)
      if hasattr(x, 'device_buffers'):  # ShardedDeviceArray
        exceptions_buffers.extend(x.device_buffers)
    for b in live_buffers:
      should_delete = True
      for e in exceptions_buffers:
        if b is e:
          should_delete = False
      if keep_constants and not b.shape:
        should_delete = False
      if should_delete:
        b.delete()

  def _per_device_rngs(self, rng):
    """Create per-device RNGs from a given rng."""
    # Splitting by device first to be identical with default trainer.
    per_device_rng = fastmath.random.split(rng, self._n_devices)
    per_device_rngs = [
        fastmath.random.split(r, self._n_layers) for r in per_device_rng]
    rngs = [jnp.stack([r[i] for r in per_device_rngs])
            for i in range(self._n_layers)]
    return rngs

  def one_step(self, batch, rng, step=0, learning_rate=None):
    """Updates layers weights/state and optimizers slots by running one step.

    Args:
      batch: Batch of data to use for optimization.
      rng: Random number generator to use for running this step.
      step: Which step of the training are we running.
      learning_rate: Learning rate to use instead of the default one.

    Returns:
      Tuple (loss, stats) with new values from one step
      of training, where stats are all optimizer statistics.
    """
    # Update the learning rate if needed.
    if learning_rate is not None:
      self._replicated_loss_opt_params['learning_rate'] = self._replicate_cpu(
          learning_rate)
      for (std_op, rev_ops) in self._replicated_opt_params:
        std_op['learning_rate'] = self._replicate_cpu(learning_rate)
        for op in rev_ops:
          op['learning_rate'] = self._replicate_cpu(learning_rate)

    # Batch needs to be split across the local devices -- the difference
    # between _for_n_devices and _reshape_by_device is that the latter splits
    # the batch dim to batch // n_devices, vs _for_n_devices
    # broadcasts/replicates to n_devices dimension.
    step_int = step
    if self._n_devices > 1:
      batch = tl.reshape_by_device(batch, self._n_devices, pure_np=True)
      step = np.repeat(step, self._n_devices)

    # Create separate rng for each device and layer.
    if self._n_devices == 1:
      rngs = fastmath.random.split(rng, self._n_layers)
    else:
      # JIT the function and run it on CPU to avoid memory fragmentation.
      rngs = self._jit_per_device_rngs(tl.on_cpu(rng))
    # Group rngs by layer blocks.
    rng_blocks, rng_i = [], 0
    for _, rev_layers in self._blocks:
      l = len(rev_layers)
      rng_blocks.append((rngs[rng_i], rngs[rng_i + 1: rng_i + l + 1]))
      rng_i += l + 1

    # Run the layers forward upto the loss layer.
    if self._do_free:
      self._free_accelerators()
    process = psutil.Process(os.getpid())
    if isinstance(batch, (list, tuple)):
      batch_shapes = [x.shape for x in batch]
    else:
      batch_shapes = batch.shape
    logging.info('running step %d on shapes %s', step_int, str(batch_shapes))
    if step_int % self._n_steps_per_log == 1:
      logging.info('run fwd: cpu memory use (MB): %.2f',
                   process.memory_info().rss / float(1024 * 1024))

    stack = batch
    block_inputs_states = []
    for i, (std_layer, rev_layers) in enumerate(self._blocks):
      acc_std_layer_fn, acc_rev_layer_fns = self._accelerated_layer_fns[i]
      std_rng, rev_rngs = rng_blocks[i]
      # Run the standard layer.
      stack, std_inputs, std_state = self._run_forward_standard(
          stack, std_layer, acc_std_layer_fn, std_rng, step_int)

      # Run the reversible layers and collect old and new states.
      stack, rev_old_states, rev_new_states = self._run_forward_reversible(
          stack, rev_layers, acc_rev_layer_fns, rev_rngs, step_int)
      block_inputs_states.append(tl.on_cpu(
          ((std_inputs, std_state), (rev_old_states, rev_new_states))))

    # Run the loss layer forward and backward with optimizer update.
    if step_int % self._n_steps_per_log == 1:
      logging.info('run loss: cpu memory use (MB): %.2f',
                   process.memory_info().rss / float(1024 * 1024))
    loss_state = self._replicate(self._loss_layer.state)
    loss_inputs = cb.inputs_from_stack(stack, self._loss_layer.n_in)
    loss_stats, grad_stack = self._run_backward_standard(
        None, step, self._loss_layer, loss_inputs,
        loss_state, self._loss_fbo, rngs[-1], self._loss_opt,
        self._replicated_loss_opt_params)
    self._collect_weights(self._loss_layer)
    stats = [tl.on_cpu(loss_stats)]

    # De-fragment memory.
    if self._do_free:
      stack, grad_stack = tl.on_cpu(stack), tl.on_cpu(grad_stack)
      self._free_accelerators()

    # Run the layers backward and run optimizer updates.
    if step_int % self._n_steps_per_log == 1:
      logging.info('run bwd: cpu memory use (MB): %.2f',
                   process.memory_info().rss / float(1024 * 1024))
    for i in range(len(self._blocks) - 1, -1, -1):
      std_layer, rev_layers = self._blocks[i]
      (std_inputs, std_state), (rev_old_states,
                                rev_new_states) = block_inputs_states[i]
      std_fbo, rev_fbos = self._fbos[i]
      std_opt, rev_opts = self._optimizers[i]
      std_rng, rev_rngs = rng_blocks[i]
      repl_std_opt_params, repl_rev_opts_params = self._replicated_opt_params[i]

      # Run reversible layers backward with optimizer update.
      stack, grad_stack, new_stats = self._run_backward_reversible(
          stack, grad_stack, step, rev_layers, rev_fbos, rev_old_states,
          rev_new_states, rev_rngs, rev_opts, repl_rev_opts_params)
      stats.extend(tl.on_cpu(new_stats))

      # Run the standard layer forward-and-backward pass and optimizer update.
      std_layer_stats, grad_stack = self._run_backward_standard(
          grad_stack, step, std_layer, std_inputs, std_state, std_fbo, std_rng,
          std_opt, repl_std_opt_params)
      stack = cb.outputs_onto_stack(  # Put layer inputs on the stack.
          std_inputs, stack, std_layer.n_out)
      stats.append(tl.on_cpu(std_layer_stats))

      # Collect lazily unreplicated layer weights.
      for rev_layer_id in range(self._n_async_layers):
        self._collect_weights(rev_layers[rev_layer_id])
      self._collect_weights(std_layer)

    # Join stats from different optimizers into one.
    joint_stats = {}
    for i, stat in enumerate(reversed(stats)):
      for k, v in stat.items():
        joint_stats[f'layer{i}/' + k] = v
    return stats[0]['loss'], joint_stats

  def _run_forward_standard(self, stack, layer, accelerated_fn, rng, step):
    """Run standard layer forward."""
    if step % self._n_steps_per_log == 1:
      logging.info('running forward standard layer %s', str(layer))
    layer_inputs = cb.inputs_from_stack(stack, layer.n_in)
    layer_weights = self._replicate(layer.weights)
    layer_state = self._replicate(layer.state)
    outputs, layer_new_state = accelerated_fn(
        layer_inputs, layer_weights, layer_state, rng)
    stack = cb.outputs_onto_stack(outputs, stack, layer.n_in)
    return stack, layer_inputs, layer_new_state

  def _run_forward_reversible(self, stack, rev_layers, accelerated_fns,
                              rngs, step):
    """Run reversible layers forward, collect states for backwards pass."""
    old_states, new_states = [], []
    for i, layer in enumerate(rev_layers):
      if step % self._n_steps_per_log == 1:
        logging.info('running forward reversible layer %s', str(layer))
      weights = self._replicate(layer.weights)  # also copies cpu -> accelerator
      state = self._replicate(layer.state)
      old_states.append(state)
      inputs = cb.inputs_from_stack(stack, layer.n_in)
      outputs, new_state = accelerated_fns[i](
          inputs, weights, state, rngs[i])
      stack = cb.outputs_onto_stack(outputs, stack, layer.n_in)
      new_states.append(new_state)
    return stack, old_states, new_states

  def _run_backward_standard(self, grad_stack, step, layer, inp, state,
                             fbo_fn, rng, optimizer, replicated_opt_params):
    """Run reversible layers backwards."""
    step_int = int(step) if self._n_devices < 2 else int(step[0])
    if step_int % self._n_steps_per_log == 1:
      logging.info('running backward standard layer %s', str(layer))
    if grad_stack is not None:
      grads = cb.inputs_from_stack(grad_stack, layer.n_out)
    else:
      grads = None
    slots = self._replicate(optimizer.slots)
    weights = self._replicate(layer.weights)
    # Ensure all arguments are on accelerator.
    state = tl.on_accelerator(state)
    replicated_opt_params = tl.on_accelerator(replicated_opt_params)
    rng = tl.on_accelerator(rng)
    grads = tl.on_accelerator(grads)
    inp = tl.on_accelerator(inp)
    new_weights, new_state, new_slots, new_grads, stats = fbo_fn(
        inp, weights, grads, state, slots, replicated_opt_params, rng, step)
    layer.weights = self._lazy_unreplicate(new_weights)
    layer.state = self._unreplicate(new_state)
    optimizer.slots = self._unreplicate(new_slots)
    if grad_stack is not None:
      grad_stack = cb.outputs_onto_stack(new_grads, grad_stack, layer.n_out)
    else:
      grad_stack = new_grads
    return stats, grad_stack

  def _run_backward_reversible(self, stack, grad_stack, step,
                               rev_layers, rev_and_fbos,
                               old_states, new_states, rngs,
                               optimizers, replicated_opt_params):
    """Run reversible layers backwards."""
    counter = 0
    stats = []
    step_int = int(step) if self._n_devices < 2 else int(step[0])
    for layer, reverse_and_fbo, old_state, new_state, rng in reversed(list(zip(
        rev_layers, rev_and_fbos,
        old_states, new_states, rngs))):
      if step_int % self._n_steps_per_log == 1:
        logging.info('running backward reversible layer %s', str(layer))
      counter -= 1
      stack, grad_stack, layer_stats = self._run_backward_one_reversible(
          layer, stack, grad_stack, step, rng, optimizers[counter],
          replicated_opt_params[counter], reverse_and_fbo, old_state, new_state)
      stats.append(layer_stats)
      if counter + self._n_async_layers < 0:
        self._collect_weights(rev_layers[counter + self._n_async_layers])
    return stack, grad_stack, stats

  def _run_backward_one_reversible(self, layer, stack, grad_stack, step, rng,
                                   optimizer, opt_params, reverse_and_fbo,
                                   old_state, new_state):
    """Run one reversible layer backwards."""
    # We are running backwards and reversing, so we get *outputs* from stack.
    outputs = cb.inputs_from_stack(stack, layer.n_out)
    grads = cb.inputs_from_stack(grad_stack, layer.n_out)
    slots = self._replicate(optimizer.slots)
    weights = self._replicate(layer.weights)  # cpu -> accelerator
    # Ensure all arguments are on accelerator.
    outputs = tl.on_accelerator(outputs)
    grads = tl.on_accelerator(grads)
    old_state = tl.on_accelerator(old_state)
    new_state = tl.on_accelerator(new_state)
    opt_params = tl.on_accelerator(opt_params)
    rng = tl.on_accelerator(rng)
    new_weights, new_slots, inputs, grads, layer_stats = reverse_and_fbo(
        outputs, weights, grads, old_state, new_state,
        slots, opt_params, rng, step)
    layer.weights = self._lazy_unreplicate(new_weights)  # accelerator -> cpu
    layer.state = self._unreplicate(new_state)
    optimizer.slots = self._unreplicate(new_slots)
    stack = cb.outputs_onto_stack(inputs, stack, layer.n_out)
    grad_stack = cb.outputs_onto_stack(grads, grad_stack, layer.n_out)
    return stack, grad_stack, layer_stats


# Forward + backward + optimizer-update functions for all layers.
# We call them in short FBO for "Forward + Backward + Optimizer update".


def _fbo_with_layer_and_opt(layer, optimizer, n_devices,
                            stats_name=None, adasum=False):
  """Create the fbo function for a given layer and optimizer."""
  def fbo(inputs, weights, grads, state, slots, opt_params, rng, step):
    """FBO of the layer."""
    # We need a layer pure_fn but only for inputs and weights.
    def pure_fn_without_state_and_rng(x, w):
      return layer.pure_fn(x, w, state, rng)

    # Calculate the vector-Jacobian product of the reduced pure fn.
    activations, vjp_fn, new_state = fastmath.vjp(
        pure_fn_without_state_and_rng, inputs, weights, has_aux=True)

    # In the loss layer, set gradients to 1 with the dtype of activations=loss.
    if grads is None and stats_name is not None:
      grads = jnp.ones((), dtype=activations.dtype)

    # The vjp function returns gradients with respect to inputs and weights.
    grads_inputs, grads_weights = vjp_fn(grads)

    # For non-trainable layers, return the calculated arguments.
    if _is_empty_tuple(weights):
      stats = {}
      if stats_name is not None:
        stats[stats_name] = activations
      return weights, new_state, slots, grads_inputs, stats

    # In multi-device setting, average gradients from multiple devices.
    if n_devices > 1:
      grads_weights = _average_multidevice_gradients(
          grads_weights, adasum=adasum)

    # Run the optimizer.
    new_weights, new_slots, stats = optimizer.tree_update(
        step, grads_weights, weights, slots, opt_params, store_slots=False)
    if stats_name is not None:
      stats[stats_name] = activations
    return new_weights, new_state, new_slots, grads_inputs, stats

  return fbo


# Reversible layers define a reverse_and_fbo function that both reverses
# and runs the forward-backward pass and applied the optimizer.
# This function uses the `reverse_and_grad` method of reversible layers.


def _reverse_and_fbo_with_layer_and_opt(layer, optimizer, n_devices, adasum):
  """Create the reverse_and_fbo function for a given layer and optimizer."""
  def reverse_and_fbo(output, weights, grads, state, new_state,
                      slots, opt_params, rng, step):
    """Reverse and FBO of the layer."""
    # Call the reverse_and_grad method of the layer.
    inputs, (grads_inputs, grads_weights) = layer.reverse_and_grad(
        output, grads, weights, state, new_state, rng=rng)

    # For non-trainable layers, return the calculated arguments.
    if _is_empty_tuple(weights):
      return weights, slots, inputs, grads_inputs, {}

    # In multi-device setting, average gradients from multiple devices.
    if n_devices > 1:
      grads_weights = _average_multidevice_gradients(
          grads_weights, adasum=adasum)

    # Run the optimizer.
    new_weights, new_slots, stats = optimizer.tree_update(
        step, grads_weights, weights, slots, opt_params, store_slots=False)

    return new_weights, new_slots, inputs, grads_inputs, stats

  return reverse_and_fbo


def _is_empty_tuple(x):
  """Check if x is either empty or a tuple of (tuples of) empty things."""
  if not isinstance(x, (list, tuple)):
    return False
  for y in x:
    if not _is_empty_tuple(y):
      return False
  return True


def extract_reversible_blocks(layers, loss_chunk_size=0):
  """Extracts blocks and loss layer for use with ReversibleSerialTrainer.

  Args:
    layers: a list of layers of a single layer to extract blocks from;
      should end with a loss, e.g., [model, loss] or tl.Serial(model, loss).
    loss_chunk_size: int, if > 0 creates a chunked loss layer to save memory
      in models with larger vocabulary; requires the last sublayers of loss
      are [Dense, LogSoftmax, _CrossEntropy, _WeightedMean] in that order.

  Returns:
    a pair (blocks, loss_layer) to use with ReversibleSerialTrainer.
  """
  def _flatten(l):
    """Flatten all Serial layers and sub(sub-...) layers into a list."""
    if isinstance(l, (list, tuple)):
      return [x for layer in l for x in _flatten(layer)]  # pylint: disable=g-complex-comprehension
    elif isinstance(l, tl.Serial):
      return _flatten(l.sublayers)
    else:
      return [l]

  # Extract standard and reversible layer blocks.
  blocks, std_layers, rev_layers = [], [], []
  for layer in _flatten(layers):
    if isinstance(layer, tl.ReversibleLayer):
      rev_layers.append(layer)
    elif not rev_layers:
      std_layers.append(layer)
    else:
      blocks.append((std_layers, rev_layers))
      std_layers, rev_layers = [], []
      std_layers.append(layer)
  if rev_layers:
    raise ValueError('The final layer must be a standard loss, not reversible.')
  if loss_chunk_size > 0:
    # For now we only do chunking of [Dense, LogSoftmax, CrossEntopy, Mean]
    # Let's check that these are the last 4 layers.
    border_layers = ['StripFromConcatenateWithPadding', 'Select']

    loss_start = None
    for index, layer in enumerate(std_layers):
      if layer.name in border_layers:
        loss_start = index + 1
    if loss_start is None:
      raise ValueError('Loss layer should be preceeded by one of {}; got {}'
                       .format(border_layers, [l.name for l in std_layers]))
    if len(std_layers) - loss_start < 4:
      raise ValueError('Too short loss layer for chunking')
    last_3_names = ' '.join([l.name for l in std_layers[-3:]])
    if last_3_names != 'LogSoftmax _CrossEntropy _WeightedMean':
      raise ValueError('Loss chunking only works with last layers being "'
                       'LogSoftmax, _CrossEntropy, _WeightedMean" but got: ' +
                       last_3_names)

    # Create chunked dense+logsoftmax+cross-entropy-loss.
    chunked_xent = tl.Chunk(tl.Serial(std_layers[loss_start:-1]),
                            loss_chunk_size)
    # The chunked loss should operate on a merged batch dimension, e.g.,
    # including both length and batch size. Need to merge and un-merge later.
    def _reshape_to_batch_and_copy_targets(preds, targets):
      batched_preds = jnp.reshape(preds, [-1, preds.shape[-1]])
      batched_targets = jnp.reshape(targets, [-1])
      return batched_preds, batched_targets, targets
    def _reshape_xent_back(xent, targets):
      return jnp.reshape(xent, targets.shape)
    batched_xent = tl.Serial(
        tl.Fn('pre_xent_rebatch', _reshape_to_batch_and_copy_targets, n_out=3),
        chunked_xent,
        tl.Fn('after_xent_rebatch', _reshape_xent_back)
    )
    loss_layer = tl.Serial(std_layers[:loss_start] + [batched_xent],
                           std_layers[-1])
  else:
    loss_layer = tl.Serial(std_layers)
  return blocks, loss_layer


def init_reversible_blocks(blocks, loss_layer, input_signature, rng):
  """Initialize reversible blocks and the loss layer and place weights on CPU.

  Args:
    blocks: List of reversible blocks (pairs of layer lists).
    loss_layer: The final loss layer to initialize.
    input_signature: The signature of the input to the blocks.
    rng: Random key used to initialize the layers.
  """
  sig_stack = input_signature
  process = psutil.Process(os.getpid())
  mem_use = process.memory_info().rss
  for (std_layers, rev_layers) in blocks:
    rngs = fastmath.random.split(rng, len(std_layers) + len(rev_layers) + 1)
    rng = rngs[0]
    for layer, layer_rng in zip(std_layers + rev_layers, rngs[1:]):
      sig = cb.inputs_from_stack(sig_stack, layer.n_in)
      layer.init(sig, rng=layer_rng)
      layer.weights = tl.on_cpu(layer.weights)  # store weights in cpu memory
      layer.state = tl.on_cpu(layer.state)  # store weights in cpu memory
      logging.info('init: layer %s\nadded cpu memory (MB): %.2f', str(layer),
                   (process.memory_info().rss - mem_use) / float(1024 * 1024))
      mem_use = process.memory_info().rss
      logging.info('init: cpu memory use (MB): %.2f',
                   mem_use / float(1024 * 1024))
      out_sig = layer.output_signature(sig)
      sig_stack = cb.outputs_onto_stack(out_sig, sig_stack, layer.n_in)
  loss_layer.init(cb.inputs_from_stack(sig_stack, loss_layer.n_in), rng=rng)
  loss_layer.weights = tl.on_cpu(loss_layer.weights)
  loss_layer.state = tl.on_cpu(loss_layer.state)


