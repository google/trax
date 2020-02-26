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

from trax.layers import base as layers
from trax.math import numpy as np


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

  def __init__(self, learning_rate, **init_opt_params):
    """Sets initial hyperparameter values for this optimizer.

    Takes initial optimizer parameters as keyword arguments. These values can
    be changed between training steps, e.g., for learning rate schedules.

    If you want your subclass to expose hyperparameters for gin configuration,
    override this constructor and use explicitly named keyword arguments. See
    `momentum.Momentum.__init__` for one such example.

    Args:
      learning_rate: The initial learning rate.
      **init_opt_params: Initial values of any additional optimizer parameters.
    """
    init_opt_params['learning_rate'] = learning_rate
    self._init_opt_params = {
        name: np.array(value) for (name, value) in init_opt_params.items()
    }
    self._slots = None

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
    """Assembles node-local initializations into full-tree initialization."""
    self._slots = [self.init(weight) for weight in _tree_flatten(weight_tree)]
    return (
        self._slots,
        self._init_opt_params,
    )

  def _update_and_check(self, step, grads, weights, slots, opt_params):
    """Update a single weight array and check types."""
    new_weights, new_slots = self.update(
        step, grads, weights, slots, opt_params)
    if isinstance(weights, np.ndarray):
      if not isinstance(new_weights, np.ndarray):
        raise ValueError(
            f'New weight values should be of type np.ndarray or a subclass; '
            f'instead got {type(new_weights)}.')
      if new_weights.dtype != weights.dtype:
        raise ValueError(
            f'New weight values dtype ({new_weights.dtype}) does not match '
            f'the old one ({weights.dtype}).')
    return new_weights, new_slots

  def tree_update(self, step, grad_tree, weight_tree, slots, opt_params):
    """Assembles node-local weight and slot updates for the full layer tree."""
    grads_flat = _tree_flatten(grad_tree)
    weights_flat = _tree_flatten(weight_tree)
    updated_pairs = [
        self._update_and_check(step, grad, weight, slot, opt_params)
        for (grad, weight, slot) in zip(grads_flat, weights_flat, slots)
    ]
    new_weights_flat, self.slots = zip(*updated_pairs)
    new_weights, _ = _tree_unflatten(new_weights_flat, weight_tree)
    return new_weights, self.slots


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


def _tree_flatten(tree):
  """Flatten a tree into a list."""
  if isinstance(tree, (list, tuple)):
    # In python, sum of lists starting from [] is the concatenation.
    return sum([_tree_flatten(t) for t in tree], [])
  if isinstance(tree, dict):
    # Only use the values in case of a dictionary node.
    return sum([_tree_flatten(v) for v in tree.values()], [])
  return [tree]


def _tree_unflatten(flat, tree):
  """Unflatten a list into a tree given the tree shape as second argument.

  Args:
    flat: a flat list of elements to be assembled into a tree.
    tree: a tree with the structure we want to have in the new tree.

  Returns:
    A pair (new_tree, rest_of_flat) where the new tree that has the structure
    of tree but with leaves from flat, and the remaining elements of flat if
    more were provided than the number of leaves of tree (useful for recursion).
  """
  if isinstance(tree, (list, tuple)):
    new_tree, rest = [], flat
    for t in tree:
      new_t, rest = _tree_unflatten(rest, t)
      new_tree.append(new_t)
    new_tree = tuple(new_tree) if isinstance(tree, tuple) else new_tree
    return new_tree, rest
  if isinstance(tree, dict):
    new_tree, rest = {}, flat
    for k in tree:
      new_v, rest = _tree_unflatten(rest, tree[k])
      new_tree[k] = new_v
    return new_tree, rest
  return flat[0], flat[1:]

# Utilities.


def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves = _tree_flatten(tree)
  return np.sqrt(sum(np.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: np.where(norm < max_norm, g, g * (max_norm / norm))
  return layers.nested_map(grad_tree, normalize)
