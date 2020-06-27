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
"""Trax base optimizer class."""

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base as layers


class Optimizer(object):
  """Base class for optimizers that work hand in hand with Trax layers.

  To define an optimizer subclass, specify its behavior with respect to a
  single level/node in the network (e.g., a single dense layer):

    - `init`: how to create/initialize optimizer-internal weights ("slots")
        whose shape matches the node's weight shape.
    - `update`: how to use gradient information to update node weights and
        optimizer slots.

  The Trax runtime combines these node-local computations into weight updates
  and slot updates for the whole tree of layers in the model.
  """

  def __init__(self, learning_rate, clip_grad_norm=None, **init_opt_params):
    """Sets initial hyperparameter values for this optimizer.

    Takes initial optimizer parameters as keyword arguments. These values can
    be changed between training steps, e.g., for learning rate schedules.

    If you want your subclass to expose hyperparameters for gin configuration,
    override this constructor and use explicitly named keyword arguments. See
    `momentum.Momentum.__init__` for one such example.

    Args:
      learning_rate: The initial learning rate.
      clip_grad_norm: float; the value to which gradients will be clipped.
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
    """Creates optimizer slots for the given parameters.

    Args:
      weights: Trainable weights for one layer. Optimizer slots typically match
          the data shape and type of the given layer weights.
    """
    raise NotImplementedError

  def update(self, step, grads, weights, slots, opt_params):
    """Computes one step's worth of updates.

    The update computes both new weights for the layer/node and new slot values
    for the optimizer.

    Args:
      step: Current step number in the training process.
      grads: Gradients for the weights of the sublayer.
      weights: Current weights for the sublayer.
      slots: Optimizer slots.
      opt_params: Optimizer hyperparameters (e.g. learning rate, momentum).

    Returns:
      Tuple of (new_weights, new_slots).
    """
    raise NotImplementedError

  @property
  def slots(self):
    return self._slots

  @slots.setter
  def slots(self, slots):
    self._slots = slots

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
    self._slots = [self.init(weight)
                   for weight in fastmath.tree_flatten(weight_tree)]
    return (
        self._slots,
        self._init_opt_params,
    )

  def tree_update(self, step, grad_tree, weight_tree, slots, opt_params):
    """Assembles node-local weight and slot updates for the full layer tree.

    Args:
      step: Current step number in the training process.
      grad_tree: Gradients for the entire model, in a tree that matches the
          model's layer structure.
      weight_tree: Current weights for the entire model, in a tree that matches
          the model's layer structure.
      slots: Optimizer slots.
      opt_params: Optimizer hyperparameters (e.g. learning rate, momentum).

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
    new_weights_flat, self.slots = zip(*updated_pairs)
    new_weights, _ = fastmath.tree_unflatten(new_weights_flat, weight_tree)
    metrics = {'gradients_l2': grads_norm, 'weights_l2': weights_norm}
    return new_weights, self.slots, metrics

  def _l2_norm(self, flat_list):
    """Returns the aggregate L2 norm of a list of tensors."""
    if fastmath.backend_name() == 'jax':
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
  """Stochastic gradient descent (SGD) optimizer.

  A simple optimizer with no weights ("slots") of its own.
  """

  def init(self, weights):
    return None

  def update(self, step, grads, weights, slots, opt_params):
    del step, slots
    lr = opt_params['learning_rate']
    new_weights = weights - (lr * grads).astype(weights.dtype)
    return new_weights, None


# Utilities.


def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves = fastmath.tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return layers.nested_map(grad_tree, normalize)
