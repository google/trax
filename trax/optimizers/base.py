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
"""Trax base optimizer class."""

from trax import fastmath
from trax.fastmath import numpy as jnp


class Optimizer:
  """Base class for optimizers that work hand in hand with Trax layers.

  To define an optimizer subclass, specify its behavior with respect to a
  single node in the network (e.g., a single dense layer):

    - `init`: how to create/initialize optimizer-internal parameters ("slots"),
        as a function of the node's weights.
    - `update`: how to use gradient information to update node weights and
        optimizer slots.

  The Trax runtime combines these node-local computations into layer weight
  updates and optimizer slot updates for the whole tree of layers in the model.
  """

  def __init__(self, learning_rate=0.01, clip_grad_norm=None,
               **init_opt_params):
    """Sets initial hyperparameter values for this optimizer.

    Takes optimizer hyperparameters as keyword arguments. These values can
    change over time (training steps), e.g., for learning rate schedules.

    To expose subclass hyperparameters for gin configuration, override this
    constructor and use explicitly named keyword arguments. See
    `momentum.Momentum.__init__` for one such example.

    Args:
      learning_rate: Learning rate for the optimizer. This can change during
          training by means of a training rate schedule.
      clip_grad_norm: If specified, this scalar value is used to limit gradient
          size -- all gradient elements in a training step are treated as if
          they belonged to a single vector and then scaled back if needed so
          that such a vector's L2 norm does not exceed `clip_grad_norm`. If
          None, no clipping happens.
      **init_opt_params: Initial values of any additional optimizer parameters.
    """
    init_opt_params['learning_rate'] = learning_rate
    self._init_opt_params = {
        name: jnp.array(value) for (name, value) in init_opt_params.items()
    }
    self._slots = None
    # Gradient clipping happens with respect to the norm of the whole gradient
    # tree, so it is not passed to single-slot updates, but done in this class
    # for the whole gradient tree.
    self._clip_grad_norm = clip_grad_norm

  def init(self, weights):
    """Creates optimizer slots that fit the given weights.

    Args:
      weights: Trainable weights for one layer. Optimizer slots typically match
          the data shape and type of the given layer weights.
    """
    raise NotImplementedError

  def update(self, step, grads, weights, slots, opt_params):
    """Computes updated layer weights and optimizer slots for one training step.

    Args:
      step: Training step number.
      grads: Gradient values for this node (from back-propagation during a
          training step).
      weights: Current weight values for this node (i.e., layer weights).
      slots: Current slot values for this node.
      opt_params: Optimizer hyperparameters (e.g. learning rate, momentum),
          same across all nodes in the model.

    Returns:
      Tuple of (new_weights, new_slots), which the Trax runtime will use to
      update the model and optimizer within each training step.
    """
    raise NotImplementedError

  @property
  def slots(self):
    return self._slots

  @slots.setter
  def slots(self, slots):
    self._slots = slots

  @property
  def opt_params(self):
    return self._init_opt_params

  @opt_params.setter
  def opt_params(self, opt_params):
    self._init_opt_params = opt_params

  def tree_init(self, weight_tree):
    """Assembles node-local initializations into full-tree initialization.

    Args:
      weight_tree: Weights for an entire model, in a tree that matches the
          model's layer structure.

    Returns:
      Tuple `(slots, opt_params)`, where `slots` are the initialized optimizer
      slot values and `opt_params` are optimizer hyperparameters (e.g.,
      learning rate, momentum).
    """
    self._slots = tuple(self.init(weight)
                        for weight in fastmath.tree_flatten(weight_tree))
    return (self._slots, self._init_opt_params)

  def tree_update(self, step, grad_tree, weight_tree, slots, opt_params,
                  store_slots=True):
    """Assembles node-local weight and slot updates for the full layer tree.

    Args:
      step: Current step number in the training process.
      grad_tree: Gradients for the entire model, in a tree that matches the
          model's layer structure.
      weight_tree: Current weights for the entire model, in a tree that matches
          the model's layer structure.
      slots: Optimizer slots.
      opt_params: Optimizer hyperparameters (e.g. learning rate, momentum).
      store_slots: Boolean; if True, stores resulting slots in this object;
        when set to False, this becomes a pure function.

    Returns:
      Tuple `(weights, slots)`, where `weights` are the optimizer-updated
      weights for the whole model (in a tree matching the model's layer
      structure) and `slots` are the updated optimizer slot values.
    """
    grads_flat = fastmath.tree_flatten(grad_tree)
    grads_norm = self._l2_norm(grads_flat)
    if self._clip_grad_norm is not None:
      max_norm = self._clip_grad_norm
      grads_flat = [jnp.where(grads_norm < max_norm,  # pylint: disable=g-complex-comprehension
                              g,
                              g * (max_norm / grads_norm))
                    for g in grads_flat]
    weights_flat = fastmath.tree_flatten(weight_tree)
    weights_norm = self._l2_norm(weights_flat)
    updated_pairs = [
        self._update_and_check(step, grad, weight, slot, opt_params)
        for (grad, weight, slot) in zip(grads_flat, weights_flat, slots)
    ]
    new_weights_flat, slots = zip(*updated_pairs)
    new_weights, _ = fastmath.tree_unflatten(new_weights_flat, weight_tree)
    metrics = {'gradients_l2': grads_norm, 'weights_l2': weights_norm}
    slots = tuple(slots)
    if store_slots:
      self.slots = slots
    return new_weights, slots, metrics

  def _l2_norm(self, flat_list):
    """Returns an L2-like norm of all elements of all tensors in `flat_list`.

    Args:
      flat_list: Collection of tensors as a flat list (rather than, e.g., a
          tree).

    Returns:
      A scalar value computed as if all the tensors in `flat_list` were joined
      and flattened into a single vector, and then the L2 norm of that vector
      was calculated.
    """
    if fastmath.is_backend(fastmath.Backend.JAX):
      norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in flat_list))
    else:  # TODO(lukaszkaiser): add vdot to TF-numpy
      norm = jnp.sqrt(sum(jnp.sum(x*x) for x in flat_list))
    return norm

  def _update_and_check(self, step, grads, weights, slots, opt_params):
    """Updates a single weight array and checks types."""
    new_weights, new_slots = self.update(
        step, grads, weights, slots, opt_params)
    if isinstance(weights, jnp.ndarray):
      if not isinstance(new_weights, jnp.ndarray):
        raise ValueError(
            f'New weight values should be of type jnp.ndarray or a subclass; '
            f'instead got {type(new_weights)}.')
      if new_weights.dtype != weights.dtype:
        raise ValueError(
            f'New weight values dtype ({new_weights.dtype}) does not match '
            f'the old one ({weights.dtype}).')
    return new_weights, new_slots


class SGD(Optimizer):
  """Stochastic gradient descent (SGD) optimizer."""

  def init(self, weights):
    return None

  def update(self, step, grads, weights, slots, opt_params):
    del step, slots
    lr = opt_params['learning_rate']
    new_weights = weights - (lr * grads).astype(weights.dtype)
    return new_weights, None


# Utilities.


def l2_norm(tree):
  """Returns an L2 norm computed over all elements of all tensors in `tree`.

  Args:
    tree: Tree-structured collection of tensors, e.g., model weights matching
        the model's layer structure.

  Returns:
    A scalar value computed as if all the tensors in `tree` were combined
    and flattened into a single vector, and then the L2 norm of that vector
    was calculated.
  """
  leaves = fastmath.tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
  """Proportionally reduces each gradient value to respect an aggregate limit.

  Args:
    grad_tree: Gradient values structured as a tree of tensors matching the
        model's layer structure.
    max_norm: The aggregate limit on gradient values. All gradient elements in
        `grad_tree` are treated as if they belonged to a single vector and
        that vector is shortened if needed so that its L2 norm does not exceed
        `clip_grad_norm`.

  Returns:
    A new tree of tensors matching the structure of `grad_tree`, but with
    element values proportionally rescaled as needed to respect the `max_norm`
    limit.
  """
  norm = l2_norm(grad_tree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return fastmath.nested_map(grad_tree, normalize)
