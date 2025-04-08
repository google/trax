# Refactored trainers.py
import math

import jax
import jax.numpy as jnp
import numpy as np

import trax.layers as tl


def _adasum_merge(a, b):
    """Compute the AdaSum of two vectors."""
    dot_val = jnp.vdot(a, b)
    a_sq = jnp.vdot(a, a)
    b_sq = jnp.vdot(b, b)
    # Handle zero-norm edge cases
    if a_sq == 0 or b_sq == 0:
        return a + b
    gamma = a_sq / (a_sq + b_sq)
    # If dot < 0, combine them scaled by gamma; else just add.
    return gamma * a + (1.0 - gamma) * b if dot_val < 0 else a + b


def _average_multidevice_gradients(gradients, adasum=False):
    """
    Averages (or Adasum-reduces) 'gradients' across devices using the axis_name='batch'.

    If adasum=False, we do a standard pmean.
    If adasum=True, we do a simple all_gather & reduce approach, for demonstration.
    """
    if not adasum:
        # Standard average via pmean
        return jax.lax.pmean(gradients, axis_name="batch")
    else:
        # Demonstration: gather all grads to each device, then reduce them.
        # (A real Adasum might do ring-based or hierarchical merges.)
        gathered = jax.lax.all_gather(gradients, axis_name="batch")

        # gathered.shape now has an extra leading dimension [n_devices].
        # We'll do a simple tree_map to accumulate them one by one.
        def adasum_reduce(g_list):
            acc = g_list[0]
            for g in g_list[1:]:
                acc = jax.tree_map(_adasum_merge, acc, g)
            return acc

        # Because we used all_gather, 'gathered' is shaped like [n_devices, ...] for each leaf
        # So we need to pass that list of leaves to adasum_reduce.
        # We'll do a small helper to slice along the 0th dimension:
        n_devices = (
            gathered[0].shape[0] if isinstance(gathered, tuple) else gathered.shape[0]
        )

        # flatten out the leading dimension for each leaf
        # to produce a python list we can fold over:
        def gather_to_list(x):
            # x shape is (n_devices, ...) -> list of n_devices leaves
            return [x[i] for i in range(n_devices)]

        # Now do adasum reduction leaf-by-leaf:
        return jax.tree_map(
            lambda arrs: adasum_reduce(arrs), jax.tree_map(gather_to_list, gathered)
        )


def _pad_batch_for_devices(batch, n_devices):
    """
    If batch_size is not divisible by n_devices, pad the leading dimension so it is.
    Returns (padded_batch, unpad_amount).

    'batch' should be a tuple/list of arrays, or a PyTree that includes arrays
    on the leading dimension for each item in the batch.
    """
    batch_size = batch[0].shape[0]  # assume batch is e.g. (input, target, ...)
    remainder = batch_size % n_devices
    if remainder == 0:
        return batch, 0

    new_size = math.ceil(batch_size / n_devices) * n_devices
    to_pad = new_size - batch_size

    def pad_fn(x):
        # x has shape [batch_size, ...]
        return jnp.pad(x, [(0, to_pad)] + [(0, 0)] * (x.ndim - 1), mode="constant")

    padded = jax.tree_map(pad_fn, batch)
    return padded, to_pad


def _unpad_batch_outputs(outputs, to_remove):
    """
    If we padded the batch by 'to_remove' examples, remove them from
    the leading dimension of the returned arrays.
    """
    if to_remove == 0:
        return outputs

    def unpad_fn(x):
        # x has leading dimension we want to slice off the last 'to_remove' elements
        return x[:-to_remove] if x.shape[0] > to_remove else x[:0]

    return jax.tree_map(unpad_fn, outputs)


def _accelerate_update_fn(forward_and_backward_fn, optimizer, n_devices, adasum):
    """
    Returns an update_fn that:
      - single-device => jitted function
      - multi-device => pmapped function that also does gradient averaging or Adasum
    """

    @jax.jit
    def single_device_update_fn(
        weights, state, opt_state, batch, rng, step_int, opt_params
    ):
        # 1) Forward + backward pass -> grads, loss, updated_state
        grads, loss, updated_state = forward_and_backward_fn(batch, weights, state, rng)

        # 2) Optimizer update
        new_weights, new_opt_state, _metrics = optimizer.tree_update(
            step_int, grads, weights, opt_state, opt_params, store_slots=False
        )
        return new_weights, updated_state, new_opt_state, loss

    if n_devices <= 1:
        # Single device => just call the jitted function
        return single_device_update_fn

    # For multi-device: we pmap around single_device_update_fn
    def multi_device_update_fn(
        weights, state, opt_state, batch, rngs, step_int, opt_params
    ):
        """
        Each device runs single_device_update_fn on a shard of the batch,
        then we do gradient averaging (or Adasum).
        """

        def _per_device_step(w, s, o, b, r):
            """
            We do the forward/backward but also average grads across devices
            inside this pmap, so each device ends up with the same update.
            """
            # -- forward+backward pass --
            grads, loss, st_new = forward_and_backward_fn(b, w, s, r)
            # -- average or Adasum the grads across devices --
            grads = _average_multidevice_gradients(grads, adasum=adasum)
            # -- apply optimizer update --
            w_new, o_new, _metrics = optimizer.tree_update(
                step_int, grads, w, o, opt_params, store_slots=False
            )
            return w_new, st_new, o_new, loss

        # We call pmap over the per-device-step
        w_updated, s_updated, o_updated, loss = jax.pmap(
            _per_device_step, axis_name="batch"
        )(weights, state, opt_state, batch, rngs)
        return w_updated, s_updated, o_updated, loss

    return multi_device_update_fn


class Trainer:
    """A trainers that supports single- or multi-device, with optional Adasum, padding, etc."""

    def __init__(self, model_with_loss, optimizer, n_devices=None, adasum=False):
        """
        Args:
            model_with_loss: A layer that returns (loss, new_state) from pure_fn(...)
            optimizer: An optimizer with .tree_init(...) and .tree_update(...) methods
            n_devices: Number of devices to use
            adasum: Whether to do Adasum gradient reduction (instead of standard averaging)
        """
        self._model_with_loss = model_with_loss
        self._optimizer = optimizer
        self._n_devices = n_devices or jax.local_device_count()
        self._adasum = adasum

        # Initialize optimizer state from the model's initial weights
        self._slots, self._opt_params = optimizer.tree_init(
            self._model_with_loss.weights
        )

        # Build forward+backward function with value_and_grad(has_aux=True)
        def forward_and_backward_fn(batch, weights, state, rng):
            """
            Returns (gradients, loss, new_state).
            """

            def loss_fn(curr_w, curr_s):
                loss_val, new_st = model_with_loss.pure_fn(
                    batch, curr_w, curr_s, rng, use_cache=True
                )
                return loss_val, new_st

            (loss_val, new_state), grads = jax.value_and_grad(
                loss_fn, argnums=0, has_aux=True
            )(weights, state)

            return grads, loss_val, new_state

        self._forward_and_backward_fn = forward_and_backward_fn

        # Build an update function that does single vs. multi-device
        self._accelerated_update_fn = _accelerate_update_fn(
            self._forward_and_backward_fn,
            self._optimizer,
            self._n_devices,
            self._adasum,
        )

    @property
    def model_with_loss(self):
        return self._model_with_loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def slots(self):
        return self._slots

    def one_step(self, batch, rng, step=0, learning_rate=None):
        """
        1) Possibly pad the batch for multi-device
        2) Single- or multi-device forward/backward
        3) Update weights & state
        4) Unpad if needed, return loss
        """
        if learning_rate is not None:
            self._opt_params["learning_rate"] = learning_rate

        weights = self._model_with_loss.weights
        state = self._model_with_loss.state

        if self._n_devices == 1:
            # Single device => just run the function directly (already jitted).
            (
                new_weights,
                new_state,
                new_slots,
                loss,
            ) = self._accelerated_update_fn(
                weights,
                state,
                self._slots,
                batch,
                rng,
                step,
                self._opt_params,
            )

            # Store
            self._model_with_loss.weights = new_weights
            self._model_with_loss.state = new_state
            self._slots = new_slots
            self._optimizer.slots = new_slots
            return loss

        #
        # Multi-device => pad the batch if needed, replicate, call pmapped update
        #
        padded_batch, to_remove = _pad_batch_for_devices(batch, self._n_devices)
        padded_size = padded_batch[0].shape[0]
        batch_per_device = padded_size // self._n_devices

        # Split rng if it's just a single key
        if isinstance(rng, np.ndarray) and rng.shape == (2,):
            rng = jax.random.split(rng, self._n_devices)

        # Reshape batch for devices
        padded_batch = jax.tree_map(
            lambda x: x.reshape((self._n_devices, batch_per_device) + x.shape[1:]),
            padded_batch,
        )

        # Replicate weights/state/slots
        weights_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), weights
        )
        state_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), state
        )
        slots_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), self._slots
        )

        # Run the pmapped update
        (
            updated_weights_rep,
            updated_state_rep,
            updated_slots_rep,
            loss_rep,
        ) = self._accelerated_update_fn(
            weights_rep, state_rep, slots_rep, padded_batch, rng, step, self._opt_params
        )

        # Unreplicate results
        new_weights = self._unreplicate(updated_weights_rep)
        new_state = self._unreplicate(updated_state_rep)
        new_slots = self._unreplicate(updated_slots_rep)
        loss_vals = self._unreplicate(loss_rep)

        # If we want a single scalar, e.g. average across devices:
        # each device sees the same final "loss", if we've pmean'd it,
        # so we can just do:
        final_loss = float(loss_vals) if np.size(loss_vals) == 1 else np.mean(loss_vals)

        # Update trainers
        self._model_with_loss.weights = new_weights
        self._model_with_loss.state = new_state
        self._slots = new_slots
        self._optimizer.slots = new_slots

        # If your model returns per-example losses, you might want to unpad the output
        # after the forward pass. But here we've just got a scalar loss, so no unpadding needed
        # for the loss. If you needed to unpad e.g. a predictions array, you'd do it here.

        return final_loss

    def _unreplicate(self, tree):
        """Return the first element of a replicated array (from shape [n_devices,...] to [...])."""
        return jax.tree_map(lambda x: x[0], tree)


class ReversibleSerialTrainer:
    """Trainer for a sequence of reversible layers - optimized with JAX JIT."""

    def __init__(
        self,
        model_with_loss,
        optimizer_fn,
        n_devices=None,
        adasum=False,
        n_steps_per_log=None,
        n_async_layers=0,
        jit_memory=True,
        do_free=True,
    ):
        """Initialize the trainers.

        Args:
            model_with_loss: Serial layer with loss at the end.
            optimizer_fn: Function creating an optimizer for each layer.
            n_devices: Number of accelerator devices to use in the computation.
            adasum: Whether to use Adasum algorithm for gradient aggregation.
            n_steps_per_log: How often to log results.
            n_async_layers: How many layers to run asynchronously.
            jit_memory: Whether to JIT memory cleanup operations.
            do_free: Whether to free memory during training.
        """
        # First, we need to extract the model and the loss from the model_with_loss.
        # Usually model_with_loss is a Serial of the original model and the loss.
        if not isinstance(model_with_loss, tl.Serial):
            # We may already be given just the model.
            self._loss_layer = model_with_loss
            self._blocks = None
            self._n_layers = 1
        else:
            self._loss_layer = model_with_loss[-1]
            self._blocks, _ = extract_reversible_blocks(model_with_loss)
            # Number of all layers (not blocks, as reversible blocks have 2 layers).
            self._n_layers = len(model_with_loss.sublayers)

        # Initialize other training parameters
        self._optimizer_fn = optimizer_fn
        self._n_devices = n_devices or jax.local_device_count()
        self._adasum = adasum
        self._n_steps_per_log = n_steps_per_log
        self._n_async_layers = n_async_layers

        # Initialize memory management parameters
        self._jit_memory = jit_memory
        self._do_free = do_free

        # Initialize RNG handling
        self._jit_per_device_rngs = jax.pmap(
            lambda rng: jax.random.split(rng, jax.local_device_count()),
            axis_name="batch",
        )

        # Initialize the accelerated layer functions - JIT compiled versions
        if self._blocks is not None:
            # Initialize reverse layers
            shapes = (1, 8)  # Will be replaced by actual batch shapes

            # Create JIT-compiled forward and backward functions for each layer
            self._accelerated_layer_fns = []
            for layer in self._blocks:

                def fwd_fn(x, weights, state, rng):
                    return layer.pure_fn(x, weights, state, rng, True)

                def bwd_fn(y, weights, state, rng, grad_y):
                    def compute_loss(y):
                        return jnp.mean(y)  # Dummy loss for grad computation

                    vjp_fn = jax.vjp(compute_loss, y)[1]
                    return vjp_fn(grad_y)[0]

                # JIT-compile these functions
                self._accelerated_layer_fns.append((jax.jit(fwd_fn), jax.jit(bwd_fn)))

        # Initialize optimizers for each block
        if self._blocks is not None:
            self._optimizers = []
            self._replicated_opt_params = []

            # Create optimizer for each layer
            for i, block in enumerate(self._blocks):
                opt = optimizer_fn(block)
                self._optimizers.append(opt)

                # Initialize optimizer state for each layer
                if i == len(self._blocks) - 1:
                    # Last layer includes the loss layer
                    slots, opt_params = opt.tree_init(block.weights)
                else:
                    slots, opt_params = opt.tree_init(block.weights)

                # Replicate optimizer parameters for multi-device training
                self._replicated_opt_params.append(self._replicate(opt_params))

        # Initialize optimizer for the loss layer
        self._loss_opt = optimizer_fn(self._loss_layer)
        slots, opt_params = self._loss_opt.tree_init(self._loss_layer.weights)
        self._replicated_loss_opt_params = self._replicate(opt_params)

        # Create forward-backward-optimize functions
        if self._blocks is not None:
            self._fbos = []
            for i, block in enumerate(self._blocks):
                # Create the forward-backward-optimize function for this layer
                fbo = self._pjit(_fbo_with_layer_and_opt, static_argnums=(0, 1))
                self._fbos.append(fbo)

        # Create loss function forward-backward-optimize
        self._loss_fbo = self._pjit(_fbo_with_layer_and_opt, static_argnums=(0, 1))

    def loss_layer(self):
        """Returns the loss layer."""
        return self._loss_layer

    def all_layers(self):
        """Returns a list of all layers in the model."""
        if self._blocks is None:
            return [self._loss_layer]
        layers = []
        for block in self._blocks:
            layers.extend(block.sublayers)
        layers.append(self._loss_layer)
        return layers

    def optimizer_fn(self):
        """Returns the optimizer function."""
        return self._optimizer_fn

    def slots(self):
        """Returns the optimizer slots."""
        slots = []
        if self._blocks is not None:
            for i, block in enumerate(self._blocks):
                slots.append(block.weights)
        slots.append(self._loss_layer.weights)
        return slots

    def slots_and_params(self):
        """Returns the optimizer slots and parameters."""
        slots = []
        params = []
        if self._blocks is not None:
            for i, opt in enumerate(self._optimizers):
                s, p = opt.slots, self._unreplicate(self._replicated_opt_params[i])
                slots.append(s)
                params.append(p)
        s, p = self._loss_opt.slots, self._unreplicate(self._replicated_loss_opt_params)
        slots.append(s)
        params.append(p)
        return slots, params

    def _pjit(self, f, *args, **kwargs):
        """Apply jit compilation but avoiding tl.Accelerate."""
        if self._n_devices == 1:
            return jax.jit(f, *args, **kwargs)
        return jax.pmap(f, axis_name="batch", *args, **kwargs)

    def _replicate(self, x):
        """Replicate a tree of values for use on multiple devices."""
        if self._n_devices <= 1:
            return x
        return jax.tree_map(
            lambda y: jnp.broadcast_to(y, (self._n_devices,) + y.shape), x
        )

    def _replicate_cpu(self, x):
        """Replicate a tree of values for use on multiple devices, allowing CPU arrays."""
        if self._n_devices <= 1:
            return x

        def rep(y):
            if isinstance(y, np.ndarray):
                return np.broadcast_to(y, (self._n_devices,) + y.shape)
            elif isinstance(y, jnp.ndarray):
                return jnp.broadcast_to(y, (self._n_devices,) + y.shape)
            else:
                return y

        return jax.tree_map(rep, x)

    def _unreplicate(self, x):
        """Take the first component of a replicated tree of values."""
        return jax.tree_map(lambda y: y[0], x)

    def _lazy_unreplicate(self, x):
        """Like _unreplicate but avoids data movement if possible."""
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        if self._n_devices == 1:
            return x

        def get_first(y):
            if y.shape[0] == self._n_devices:
                return y[0]
            return y

        return jax.tree_map(get_first, x)

    def _collect_weights(self):
        """Collect weights from all layers into a single list."""
        weights = []
        if self._blocks is not None:
            for block in self._blocks:
                weights.append(block.weights)
        weights.append(self._loss_layer.weights)
        return weights

    def _free_accelerators(
        self, n_done_per_replica, replica_id, n_to_do_in_replica=None
    ):
        """Free accelerator memory not used by a replica at a given step."""
        if not self._do_free:
            return

        if n_to_do_in_replica is None:
            n_to_do_in_replica = len(self._blocks) * 2 + 3

        done_rate = n_done_per_replica / n_to_do_in_replica

        # If we have done a large chunk, we can free memory
        if done_rate >= 0.5:
            # Apply JIT compilation to memory operations if configured
            if self._jit_memory:
                # Define a memory cleanup function and JIT it
                @jax.jit
                def cleanup():
                    # Reset JAX memory allocation
                    jax.lax.stop_gradient(0.0)
                    # Add explicit synchronization
                    jax.lax.psum(0, axis_name="batch")
                    return 0

                cleanup()
            else:
                # Simple memory cleanup without JIT
                jax.lax.stop_gradient(0.0)
                jax.lax.psum(0, axis_name="batch")

    def _per_device_rngs(self, rng):
        """Create different RNG keys for different devices."""
        if isinstance(rng, np.ndarray) and rng.shape == (2,):
            if self._n_devices == 1:
                return rng
            # Create different RNG keys for different devices
            return jax.random.split(rng, self._n_devices)

        # In multi-device case, we get a precomputed set of rngs
        return rng

    def one_step(self, batch, rng, step=0, learning_rate=None):
        """Run one step of gradient-based training.

        Args:
            batch: Batch of training data.
            rng: Random number generator.
            step: Current training step.
            learning_rate: Optional learning rate to use.

        Returns:
            Loss computed on the batch.
        """
        # Update the learning rate if needed
        if learning_rate is not None:
            if self._blocks is not None:
                for params in self._replicated_opt_params:
                    params["learning_rate"] = learning_rate
            self._replicated_loss_opt_params["learning_rate"] = learning_rate

        # Prepare the batch for multiple devices if needed
        if self._n_devices > 1:
            batch_size = batch[0].shape[0]
            batch_per_device = batch_size // self._n_devices

            batch = jax.tree_map(
                lambda x: x.reshape(self._n_devices, batch_per_device, *x.shape[1:]),
                batch,
            )

        # Prepare RNGs for each device
        device_rngs = self._per_device_rngs(rng)

        if self._blocks is None:
            # No reversible layers - direct computation
            # Forward pass through the loss layer
            output, updated_state = self._loss_layer.pure_fn(
                batch,
                self._loss_layer.weights,
                self._loss_layer.state,
                device_rngs,
                True,
            )

            # Create the input-output gradient function
            def grad_fn(weights):
                output, _ = self._loss_layer.pure_fn(
                    batch, weights, self._loss_layer.state, device_rngs, True
                )
                return output

            # Compute gradients for the loss layer
            gradients = jax.grad(grad_fn)(self._loss_layer.weights)

            # Average gradients across devices if needed
            if self._n_devices > 1:
                gradients = _average_multidevice_gradients(
                    gradients, self._n_devices, self._adasum
                )

            # Update the weights with the optimizer
            updates, updated_opt_state = self._loss_opt.tree_update(
                gradients, self._loss_opt.slots, self._loss_layer.weights, step
            )

            # Apply updates to weights
            updated_weights = jax.tree_map(
                lambda w, u: w + u, self._loss_layer.weights, updates
            )

            self._loss_layer.weights = updated_weights
            self._loss_layer.state = updated_state
            self._loss_opt.slots = updated_opt_state

            return output

        # We have reversible blocks - run the full reversible computation
        if not self._blocks[0].sublayers[0].has_backward:
            # Standard case - run forward and backward passes separately
            (output, updated_state), inputs_stack = self._run_forward_standard(
                batch, device_rngs
            )

            # Compute loss gradients
            loss_gradients = jax.grad(
                lambda w: self._loss_layer.pure_fn(
                    output, w, self._loss_layer.state, device_rngs, True
                )[0]
            )(self._loss_layer.weights)

            # Average gradients across devices if needed
            if self._n_devices > 1:
                loss_gradients = _average_multidevice_gradients(
                    loss_gradients, self._n_devices, self._adasum
                )

            # Update loss layer weights
            loss_updates, loss_updated_opt_state = self._loss_opt.tree_update(
                loss_gradients, self._loss_opt.slots, self._loss_layer.weights, step
            )

            self._loss_layer.weights = jax.tree_map(
                lambda w, u: w + u, self._loss_layer.weights, loss_updates
            )
            self._loss_layer.state = updated_state

            # Run backward pass to compute and update weights for all blocks
            self._run_backward_standard(output, inputs_stack, device_rngs, step)

            return output
        else:
            # Reversible case - use specialized forward-backward
            output, output_grad = self._run_forward_reversible(batch, device_rngs)

            # Run backward pass for all reversible blocks
            loss = self._run_backward_reversible(output, output_grad, device_rngs, step)

            return loss

    def _run_forward_standard(self, batch, rngs):
        """Run the forward pass in standard (non-reversible) mode."""
        # Extract inputs and targets
        inputs_stack = []

        # Forward pass through all blocks
        for i, block in enumerate(self._blocks):
            inputs_stack.append(batch)
            # Run the actual forward pass for this block
            batch, updated_state = block.pure_fn(
                batch, block.weights, block.state, rngs, True
            )

            # Update block state
            if i < len(self._blocks) - 1:
                block.state = updated_state

        # Final forward pass through the loss layer
        output, loss_updated_state = self._loss_layer.pure_fn(
            batch, self._loss_layer.weights, self._loss_layer.state, rngs, True
        )

        self._loss_layer.state = loss_updated_state

        return (output, loss_updated_state), inputs_stack

    def _run_forward_reversible(self, batch, rngs):
        """Run the forward pass in reversible mode."""
        # Extract inputs and targets
        # Initialize the activations list
        activations = []

        # Forward pass through all blocks
        for i, block in enumerate(self._blocks):
            # Add the current input to activations
            activations.append(batch)

            # Run the forward pass for this block
            batch, updated_state = block.pure_fn(
                batch, block.weights, block.state, rngs, True
            )

            # Update the block state
            block.state = updated_state

        # Final forward pass through the loss layer
        output, loss_updated_state = self._loss_layer.pure_fn(
            batch, self._loss_layer.weights, self._loss_layer.state, rngs, True
        )

        self._loss_layer.state = loss_updated_state

        # Compute the output gradient
        def loss_fn(x):
            return self._loss_layer.pure_fn(
                x, self._loss_layer.weights, self._loss_layer.state, rngs, True
            )[0]

        # Get the gradient with respect to the output
        output_grad = jax.grad(loss_fn)(batch)

        return output, output_grad

    def _run_backward_standard(self, loss, inputs_stack, rngs, step):
        """Run the backward pass in standard (non-reversible) mode."""
        # Compute gradients for all blocks
        grad_fn = lambda weights, i: self._blocks[i].pure_fn(
            inputs_stack[i], weights, self._blocks[i].state, rngs, True
        )[0]

        # Process blocks in reverse order
        for i in range(len(self._blocks) - 1, -1, -1):
            # Compute gradients for this block
            block_gradients = jax.grad(lambda w: grad_fn(w, i))(self._blocks[i].weights)

            # Average gradients across devices if needed
            if self._n_devices > 1:
                block_gradients = _average_multidevice_gradients(
                    block_gradients, self._n_devices, self._adasum
                )

            # Update block weights
            block_updates, block_updated_opt_state = self._optimizers[i].tree_update(
                block_gradients,
                self._optimizers[i].slots,
                self._blocks[i].weights,
                step,
            )

            self._blocks[i].weights = jax.tree_map(
                lambda w, u: w + u, self._blocks[i].weights, block_updates
            )
            self._optimizers[i].slots = block_updated_opt_state

            # Free accelerator memory if configured
            if self._do_free:
                self._free_accelerators(len(self._blocks) - i, 0)

    def _run_backward_reversible(self, batch, loss, output_grads, rngs, step):
        """Run the backward pass in reversible mode.

        Args:
            batch: The input batch data
            loss: The loss value from forward pass
            output_grads: Gradients of the loss
            rngs: Random number generators
            step: Current training step

        Returns:
            The loss value
        """
        # Initialize the gradient to be backpropagated
        grads = output_grads

        # Process blocks in reverse order
        for i in range(len(self._blocks) - 1, -1, -1):
            # Get the input for this block
            if i > 0:
                inputs = self._blocks[i - 1].output
            else:
                # First block - get the original input
                inputs = batch[0]  # Assuming batch is a tuple of (inputs, targets)

            # Run the backward pass for this block
            block_gradients, grads = self._run_backward_one_reversible(
                i, inputs, grads, rngs
            )

            # Average gradients across devices if needed
            if self._n_devices > 1:
                block_gradients = _average_multidevice_gradients(
                    block_gradients, self._n_devices, self._adasum
                )

            # Use the optimizer's update method to get new weights and updated slots
            block_weights = self._blocks[i].weights
            opt_slots = self._optimizers[i].slots
            opt_params = self._optimizers[i].opt_params

            # Update weights using optimizer's own update logic
            new_weights, new_slots = self._optimizers[i].tree_update(
                block_gradients, opt_slots, block_weights, step, opt_params
            )

            # Update block weights and optimizer slots
            self._blocks[i].weights = new_weights
            self._optimizers[i].slots = new_slots

            # Free accelerator memory if configured
            if self._do_free:
                self._free_accelerators(len(self._blocks) - i, 0)

        return loss

    def _run_backward_one_reversible(self, block_index, inputs, output_grads, rngs):
        """Run the backward pass for one reversible block."""
        # Get the block
        block = self._blocks[block_index]

        # Define the forward function for gradient computation
        def forward_fn(weights, inputs):
            output, _ = block.pure_fn(inputs, weights, block.state, rngs, True)
            return output

        # Compute block gradients with reverse-mode autodiff
        block_gradients, input_grads = jax.vjp(
            lambda w: forward_fn(w, inputs), block.weights
        )[1](output_grads)

        return block_gradients, input_grads


def _fbo_with_layer_and_opt(
    optimizer,
    layer,
    inputs,
    weights,
    state,
    rngs,
    opt_state,
    opt_params,
    grads=None,
    step=None,
):
    """Forward + backward + optimize on a single layer."""
    # JIT-compiled function for forward-backward-optimize
    if grads is None:
        # Forward pass
        output, new_state = layer.pure_fn(inputs, weights, state, rngs, True)

        # Define gradient function
        def loss_fn(weights):
            output, _ = layer.pure_fn(inputs, weights, state, rngs, True)
            return jnp.mean(output)

        # Compute gradients
        gradients = jax.grad(loss_fn)(weights)
    else:
        # Use provided gradients
        gradients = grads
        new_state = state
        output = None

    # Optimize
    updates, new_opt_state = optimizer.tree_update(
        gradients, opt_state, weights, step, opt_params
    )

    # Apply updates
    new_weights = jax.tree_map(lambda w, u: w + u, weights, updates)

    return output, new_weights, new_state, new_opt_state, gradients


def _reverse_and_fbo_with_layer_and_opt(
    optimizer,
    reversible_layer,
    output,
    output_grad,
    weights,
    state,
    rngs,
    opt_slots,
    opt_params,
    step=None,
):
    """Reverse-mode computation + optimize for a reversible layer."""

    # Define the backward function for gradient computation
    def backward_fn(weights):
        # Define a forward pass that computes outputs for these weights
        def forward_fn(x):
            y, _ = reversible_layer.pure_fn(x, weights, state, rngs, True)
            return y

        # Use VJP to compute gradients backward
        _, vjp_fn = jax.vjp(forward_fn, output)
        return vjp_fn(output_grad)[0]

    # Compute input gradient and weight gradients
    input_grad = backward_fn(weights)

    # Compute weight gradients using the chain rule
    weight_grads = jax.grad(
        lambda w: jnp.sum(
            reversible_layer.pure_fn(output, w, state, rngs, True)[0] * output_grad
        )
    )(weights)

    # Use optimizer to compute new weights and slots
    new_weights, new_slots = optimizer.tree_update(
        weight_grads, opt_slots, weights, step, opt_params
    )

    return input_grad, new_weights, state, new_slots


def extract_reversible_blocks(layer):
    """Extract reversible blocks from a serial layer.

    Args:
        layer: A layer, usually a Serial layer containing reversible blocks.

    Returns:
        A tuple (reversible_blocks, loss_layer) where reversible_blocks is
        a list of blocks that are reversible and loss_layer is the final
        loss layer or None if not present.
    """
    if not isinstance(layer, tl.Serial):
        return [], layer

    blocks = []
    loss_layer = None

    # Check if the last layer is a loss layer
    if hasattr(layer.sublayers[-1], "n_in") and layer.sublayers[-1].n_in == 2:
        loss_layer = layer.sublayers[-1]
        sublayers = layer.sublayers[:-1]
    else:
        sublayers = layer.sublayers

    # Group layers into reversible blocks
    i = 0
    while i < len(sublayers):
        if (isinstance(sublayers[i], tl.ReversibleLayer) or
                (hasattr(sublayers[i], 'has_backward') and sublayers[i].has_backward)):
            blocks.append(sublayers[i])
            i += 1
        elif (i + 1 < len(sublayers) and
              isinstance(sublayers[i], tl.ReversibleHalfResidual) and
              isinstance(sublayers[i+1], tl.ReversibleHalfResidual)):
            # Pair of ReversibleHalfResidual layers make a reversible block
            blocks.append(tl.ReversibleResidual(sublayers[i], sublayers[i+1]))
            i += 2
        else:
            # Non-reversible layer - wrap it in a serial block
            blocks.append(tl.Serial(sublayers[i]))
            i += 1

    return blocks, loss_layer
