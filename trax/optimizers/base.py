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

"""Trax base optimizer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax.layers import base as layers
from trax.math import numpy as np


class Optimizer(object):
  """Optimizer object, base class. Maps per-parameter functions to trees."""

  def __init__(self, learning_rate, **init_opt_params):
    """Initialize the optimizer.

    Takes the initial optimizer parameters as positional arguments. They are fed
    back to the optimizer in tree_update, in the same order. They can be changed
    between updates, e.g. for learning rate schedules.

    The constructor should be overridden in derived classes to give names to the
    optimizer parameters, so the gin configuration can set them.

    Args:
      learning_rate: The initial learning rate.
      **init_opt_params: Initial values of any additional optimizer parameters.
    """
    init_opt_params['learning_rate'] = learning_rate
    self._init_opt_params = {
        name: np.array(value) for (name, value) in init_opt_params.items()
    }

  def init(self, params):
    """Create optimizer slots for the given parameters."""
    raise NotImplementedError

  def update(self, step, grads, weights, slots, opt_params):
    """Update a single parameter array.

    Args:
      step: Current step.
      grads: Gradients.
      weights: Trainable model weights.
      slots: Optimizer slots (e.g. gradient moments).
      opt_params: Optimizer (hyper)parameters (e.g. learning rate, momentum).

    Returns:
      (new_weights, new_slots)
    """
    raise NotImplementedError

  # End subclass interface.

  def tree_init(self, weight_tree):
    return (
        [self.init(weight) for weight in _tree_flatten(weight_tree)],
        self._init_opt_params,
    )

  def _update_and_check(self, step, grads, weights, slots, opt_params):
    """Update a single weight array and check types."""
    new_weights, new_slots = self.update(
        step, grads, weights, slots, opt_params)
    if isinstance(weights, np.ndarray):
      assert isinstance(new_weights, np.ndarray), (
          'The type of the new weight values should be np.ndarray; got %s' %
          type(new_weights))
      assert new_weights.dtype == weights.dtype, (
          'The dtype of the new weight values (%s) is not the same as the '
          'old one (%s)' % (new_weights.dtype, weights.dtype))
    return new_weights, new_slots

  def tree_update(self, step, grad_tree, weight_tree, slots, opt_params):
    grads_flat = _tree_flatten(grad_tree)
    weights_flat = _tree_flatten(weight_tree)
    updated_pairs = [
        self._update_and_check(step, grad, weight, slot, opt_params)
        for (grad, weight, slot) in zip(grads_flat, weights_flat, slots)
    ]
    new_weights_flat, new_slots = zip(*updated_pairs)
    new_weights, _ = _tree_unflatten(new_weights_flat, weight_tree)
    return new_weights, new_slots


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
