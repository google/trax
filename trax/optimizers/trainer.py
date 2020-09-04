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
"""Trainer class that accelerates running optimizers on layers."""
import functools
from absl import logging
import jax
from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers import combinators as cb

# pylint: disable=protected-access
_inputs_from_stack = cb._inputs_from_stack
_outputs_onto_stack = cb._outputs_onto_stack
# pylint: enable=protected-access


class Trainer(object):
  """Accelerates running an optimizer on a Trax layer returning a scalar loss.

  By default it uses all available accelerators, runs an accelerated version
  of the loss layer forward and then updates its weights using the optimizer.
  If only one accelerator is available, this JIT-compiles the underlying
  computation and in this way makes it run faster.

  We assume that both the loss layer (usually model with loss) and the optimizer
  have already been initialized.

  The output after running the `one_step` function is just the loss from the
  loss layer and optimizer statistics but, as a side effect, it also updates
  the weights of the loss layer and the slots of the optimizer.
  """

  def __init__(self, loss_layer, optimizer, n_devices=None):
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._n_devices = n_devices or fastmath.device_count()

    # optimizer slots and opt_params may need to be replicated
    self._slots, self._opt_params = tl.for_n_devices(
        (self._optimizer.slots, self._optimizer.opt_params), self._n_devices)

    # accelerated version of loss layer to replicate weights and state
    self._accelerated_loss_layer = tl.Accelerate(
        loss_layer, n_devices=n_devices)

    # Signature:
    # (batch, weights, state, rng) -> ((loss, state), gradients)
    self._forward_and_backward_fn = (
        fastmath.value_and_grad(
            loss_layer.pure_fn,
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
        )
    )

  @property
  def loss_layer(self):
    """Returns the loss layer used to initialize this class."""
    return self._loss_layer

  @property
  def accelerated_loss_layer(self):
    """Returns the accelerated loss layer managed by this class."""
    return self._accelerated_loss_layer

  @property
  def optimizer(self):
    """Returns the optimizer used to initialize this class."""
    return self._optimizer

  def one_step(self, batch, rng, step=0, learning_rate=None):
    """Updates loss layer weights/state and optimizer slots by running one step.

    Args:
      batch: Batch of data to use for optimization.
      rng: Random number generator to use for running this step.
      step: Which step of the training are we running.
      learning_rate: Learning rate to use instead of the default one.

    Returns:
      Tuple (loss, stats) with new values from one step
      of training, where stats are current optimizer statistics.
    """
    # Update the learning rate if needed.
    if learning_rate is not None:
      self._opt_params['learning_rate'] = tl.for_n_devices(
          learning_rate, self._n_devices)

    # batch needs to be split across the local devices -- the difference
    # between _for_n_devices and _reshape_by_device is that the latter splits
    # the batch dim to batch // n_devices, vs _for_n_devices
    # broadcasts/replicates to n_devices dimension.
    if self._n_devices > 1:
      batch = tl.reshape_by_device(batch, self._n_devices)

    # separate rng needs to be created for each device
    if self._n_devices > 1:
      rng = jnp.stack(fastmath.random.split(rng, self._n_devices))

    weights = self._accelerated_loss_layer.weights
    state = self._accelerated_loss_layer.state
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

    self._accelerated_loss_layer.weights = new_weights
    self._accelerated_loss_layer.state = new_state
    self._slots = new_slots
    self._optimizer.slots = self._unreplicate(self._slots)
    return stats['loss'], stats

  def _unreplicate(self, x):
    if self._n_devices == 1:
      return x
    return fastmath.nested_map(lambda x: x[0], x)


def _average_multidevice_gradients(gradients):
  """Averages gradients over all the devices across different hosts."""
  # Sum gradients over all devices across all hosts.
  gradients = fastmath.psum(gradients, 'batch')
  # Calculate the total number of devices.
  # Note: the usual n_devices is only the number of devices at this host,
  # here we are calculating the number of all devices across all hosts.
  n_devices_total = fastmath.psum(jnp.array(1.0), 'batch')
  # Average across hosts.
  return fastmath.nested_map(lambda g: g / n_devices_total, gradients)


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
      step = jnp.array(step, dtype=jnp.int32)  # Needed in TFNP backend.
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

    # Average gradients from multiple devices.
    gradients = _average_multidevice_gradients(gradients)

    # Run the optimizer.
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


class ReversibleSerialTrainer(object):
  """Runs an optimizer on a series of layers, all but 2 of them reversible.

  The main motivation for this class is to save memory: it allows to train
  models that have more weights than the memory available on accelerators.
  This happens by caching the weights in CPU memory and transferring only
  the weights of one layer at a time. To make the backward pass without
  using additional memory, we require that all layers except for the first
  and last one are reversible.
  """

  def __init__(self, first_layer, reversible_layers, loss_layer,
               optimizer_fn, n_devices=None):
    """Creates a ReversibleSerialTrainer and the needed optimizers.

    This trainer performs updates equivalent to using the default Trainer on::

      tl.Serial([first_layer] + reversible_layer + [loss_layer]).

    It is more memory-efficient though since weights are stored on CPU and only
    sent to accelerator layer-by-layer. Note that the first layer and loss layer
    can be arbitrary layers, so they can be a `tl.Serial` combination of layers
    too. For now, we only support one block of reversible layers though.

    Args:
      first_layer: The first layer of the model, it can be arbitraty.
      reversible_layers: A list of reversible layers that are executed after
        the first layer. We do not keep their activations in memory and weights
        are moved to CPU RAM after each layer to free accelerator memory.
      loss_layer: The final layer of the model; it can have trainable weights
        but should end with a loss: it is required to produce a scalar output.
      optimizer_fn: A function to create the optimizer, e.g., `optimizers.Adam`.
      n_devices: An optional integer, number of accelerator devices to use;
        by default, all available accelerators will be used.
    """
    # TODO(lukaszkaiser): remove these 2 lines once PR #4039 lands for JAX.
    if fastmath.is_backend(fastmath.Backend.JAX):
      jax.api._check_inexact_input_vjp = lambda x: None  # pylint: disable=protected-access
    self._first_layer = first_layer
    self._reversible_layers = reversible_layers
    self._loss_layer = loss_layer
    self._optimizer_fn = optimizer_fn
    self._n_devices = n_devices or fastmath.device_count()

    # Create accelerated versions of layers as pmaped/jited pure_fn.
    self._accelerated_first_layer_fn = self._pjit(first_layer.pure_fn)

    self._accelerated_reversible_layers_fns = []
    for layer in reversible_layers:
      self._accelerated_reversible_layers_fns.append(
          self._pjit(layer.pure_fn))

    # Create per-layer optimizers and replicate opt_params.
    self._optimizers, self._replicated_opt_params = [], []
    for layer in [first_layer] + reversible_layers + [loss_layer]:
      optimizer = optimizer_fn()
      optimizer.tree_init(layer.weights)
      self._optimizers.append(optimizer)
      opt_params = self._replicate(optimizer.opt_params)
      self._replicated_opt_params.append(opt_params)

    # Forward + backward + optimizer-update functions for all layers.
    # We call them in short FBO for "Forward + Backward + Optimizer update".

    def first_fbo(inputs, weights, state, slots, opt_params, rng, step, grads):
      """FBO of the first layer."""
      # We need the first layer's pure_fn but only for inputs and weights.
      def first_layer_pure_fn_without_state_and_rng(x, w):
        return first_layer.pure_fn(x, w, state, rng)

      # Calculate vector-Jacobian product of the reduced first layer pure fn.
      activations_after_first_layer, vjp_fn, new_state = fastmath.vjp(
          first_layer_pure_fn_without_state_and_rng,
          inputs, weights, has_aux=True)
      del activations_after_first_layer  # unused

      # The vjp function returns gradients with respect to inputs and weights.
      _, grads_weights = vjp_fn(grads)

      # In multi-device setting, average gradients from multiple devices.
      if self._n_devices > 1:
        grads_weights = _average_multidevice_gradients(grads_weights)

      # Run the first layer optimizer, which is the first one.
      new_weights, new_slots, stats = self._optimizers[0].tree_update(
          step, grads_weights, weights, slots, opt_params)
      return new_weights, new_state, new_slots, stats

    # Accelerate the first layer FBO function and store it.
    self._first_fbo = self._pjit(first_fbo)

    # Loss layer FBO is like the first layer, but has no gradients argument
    # as it is the last layer and we always use 1.0 for that. On the other
    # hand, it adds the final activation (loss) into the returned stats.

    def loss_fbo(inputs, weights, state, slots, opt_params, rng, step):
      """FBO of the final loss layer."""
      # We need a loss layer pure_fn but only for inputs and weights.
      def loss_pure_fn_without_state_and_rng(x, w):
        return loss_layer.pure_fn(x, w, state, rng)

      # Calculate the vector-Jacobian product of the reduced loss pure fn.
      loss, vjp_fn, new_state = fastmath.vjp(
          loss_pure_fn_without_state_and_rng, inputs, weights, has_aux=True)

      # The vjp function returns gradients with respect to inputs and weights.
      # Since loss is scalar and there are no other layers, run it at 1.0.
      grads_inputs, grads_weights = vjp_fn(jnp.ones((), dtype=loss.dtype))

      # In multi-device setting, average gradients from multiple devices.
      if self._n_devices > 1:
        grads_weights = _average_multidevice_gradients(grads_weights)

      # Run the loss optimizer, which is the last one since it's the last layer.
      new_weights, new_slots, stats = self._optimizers[-1].tree_update(
          step, grads_weights, weights, slots, opt_params)
      stats['loss'] = loss
      return new_weights, new_state, new_slots, grads_inputs, stats

    # Accelerate the loss layer FBO function and store it.
    self._loss_fbo = self._pjit(loss_fbo)

    # Reversible layers define a reverse_and_fbo function that both reverses
    # and runs the forward-backward pass and applied the optimizer.
    # This function uses the `reverse_and_grad` method of reversible layers.

    def reverse_and_fbo_with_layer_and_opt(layer, optimizer):
      """Create the reverse_and_fbo function for a given layer and optimizer."""
      def reverse_and_fbo(output, weights, state, new_state,
                          slots, opt_params, rng, step, grads):
        """Reverse and FBO of the layer."""
        # Call the reverse_and_grad method of the layer.
        inputs, (grads_inputs, grads_weights) = layer.reverse_and_grad(
            output, grads, weights, state, new_state, rng=rng)

        # For non-trainable layers, return the calculated arguments.
        if not weights:
          return weights, slots, inputs, grads_inputs, {}

        # In multi-device setting, average gradients from multiple devices.
        if self._n_devices > 1:
          grads_weights = _average_multidevice_gradients(grads_weights)

        # Run the optimizer.
        new_weights, new_slots, stats = optimizer.tree_update(
            step, grads_weights, weights, slots, opt_params)

        return new_weights, new_slots, inputs, grads_inputs, stats
      return reverse_and_fbo

    # Accelerate the reverse_and_fbo functions and store them.
    self._reverse_and_fbos = []
    for layer, opt in zip(reversible_layers, self._optimizers[1:-1]):
      reverse_and_fbo = reverse_and_fbo_with_layer_and_opt(layer, opt)
      self._reverse_and_fbos.append(self._pjit(reverse_and_fbo))

  @property
  def loss_layer(self):
    """Returns the loss layer used to initialize this class."""
    return self._loss_layer

  @property
  def optimizer_fn(self):
    """Returns the optimizer function used to initialize this class."""
    return self._optimizer_fn

  def _pjit(self, f):
    """JIT f if 1 device is available and pmap if more are available."""
    if self._n_devices == 1:
      return fastmath.jit(f)
    else:
      return fastmath.pmap(f, axis_name='batch')

  def _replicate(self, x):
    if self._n_devices > 1:
      return tl.for_n_devices(x, self._n_devices)
    return tl.on_accelerator(x)

  def _unreplicate(self, x):
    if self._n_devices == 1:
      return tl.on_cpu(x)
    return tl.on_cpu(fastmath.nested_map(lambda x: x[0], x))

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
      for op in self._replicated_opt_params:
        op['learning_rate'] = tl.for_n_devices(
            learning_rate, self._n_devices)

    # Batch needs to be split across the local devices -- the difference
    # between _for_n_devices and _reshape_by_device is that the latter splits
    # the batch dim to batch // n_devices, vs _for_n_devices
    # broadcasts/replicates to n_devices dimension.
    if self._n_devices > 1:
      batch = tl.reshape_by_device(batch, self._n_devices)
      step = jnp.repeat(step, self._n_devices)

    # Separate rng needs to be created for each device.
    if self._n_devices == 1:
      rngs = fastmath.random.split(rng, len(self._reversible_layers) + 2)
    else:
      # Splitting by device first to be identical with default trainer.
      per_device_rng = fastmath.random.split(rng, self._n_devices)
      per_device_rngs = [
          fastmath.random.split(r, len(self._reversible_layers) + 2)
          for r in per_device_rng]
      rngs = [jnp.stack([r[i] for r in per_device_rngs])
              for i in range(len(self._reversible_layers) + 2)]

    # Run the layers forward upto the loss layer.
    stack = batch

    # Run the first layer.
    first_layer_inputs = _inputs_from_stack(self._first_layer, stack)
    first_layer_weights = self._replicate(self._first_layer.weights)
    first_layer_state = self._replicate(self._first_layer.state)
    outputs, first_layer_new_state = self._accelerated_first_layer_fn(
        first_layer_inputs, first_layer_weights, first_layer_state, rngs[0])
    stack = _outputs_onto_stack(self._first_layer, outputs, stack)

    # Run the reversible layers and collect old and new states.
    old_states, new_states = [], []
    for i, layer in enumerate(self._reversible_layers):
      weights = self._replicate(layer.weights)  # also copies cpu -> accelerator
      state = self._replicate(layer.state)
      old_states.append(state)
      inputs = _inputs_from_stack(layer, stack)
      outputs, new_state = self._accelerated_reversible_layers_fns[i](
          inputs, weights, state, rngs[i+1])
      stack = _outputs_onto_stack(layer, outputs, stack)
      new_states.append(new_state)

    # Run the loss layer forward and backward with optimizer update.
    loss_weights = self._replicate(self._loss_layer.weights)
    loss_state = self._replicate(self._loss_layer.state)
    loss_inputs = _inputs_from_stack(self._loss_layer, stack)
    loss_slots = self._replicate(self._optimizers[-1].slots)
    new_weights, new_state, new_slots, grad_stack, loss_stats = self._loss_fbo(
        loss_inputs, loss_weights, loss_state,
        loss_slots, self._replicated_opt_params[-1], rngs[-1], step)
    stats = [loss_stats]
    self._loss_layer.weights = self._unreplicate(new_weights)  # acceler. -> cpu
    self._loss_layer.state = self._unreplicate(new_state)
    self._optimizers[-1].slots = self._unreplicate(new_slots)

    # Run reversible layers backward with optimizer update.
    counter = -1
    for layer, reverse_and_fbo, old_state, new_state, rng in reversed(list(zip(
        self._reversible_layers, self._reverse_and_fbos,
        old_states, new_states, rngs[1:-1]))):
      counter -= 1
      # We are running backwards and reversing, so we get *outputs* from stack.
      outputs = _inputs_from_stack(layer, stack, layer.n_out)
      grads = _inputs_from_stack(layer, grad_stack, layer.n_out)
      slots = self._replicate(self._optimizers[counter].slots)
      opt_params = self._replicated_opt_params[counter]
      weights = self._replicate(layer.weights)  # cpu -> accelerator
      new_weights, new_slots, inputs, grads, layer_stats = reverse_and_fbo(
          outputs, weights, old_state, new_state,
          slots, opt_params, rng, step, grads)
      layer.weights = self._unreplicate(new_weights)  # accelerator -> cpu
      layer.state = self._unreplicate(new_state)
      self._optimizers[counter].slots = self._unreplicate(new_slots)
      stats.append(layer_stats)
      stack = _outputs_onto_stack(
          layer, inputs, stack, layer.n_out, layer.n_in)
      grad_stack = _outputs_onto_stack(
          layer, grads, grad_stack, layer.n_out, layer.n_in)

    # Run the first layer forward-and-backward pass and optimizer update.
    grads = _inputs_from_stack(
        self._first_layer, grad_stack, self._first_layer.n_out)
    slots = self._replicate(self._optimizers[0].slots)
    new_weights, new_state, new_slots, first_layer_stats = self._first_fbo(
        first_layer_inputs, first_layer_weights, first_layer_new_state,
        slots, self._replicated_opt_params[0], rngs[0], step, grads)
    stats.append(first_layer_stats)
    self._first_layer.weights = self._unreplicate(new_weights)
    self._first_layer.state = self._unreplicate(new_state)
    self._optimizers[0].slots = self._unreplicate(new_slots)

    return stats[0]['loss'], stats
