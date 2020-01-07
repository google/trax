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

"""Implementations of reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from trax import math
from trax.layers import base
from trax.layers import combinators as cb


class ReversibleLayer(base.Layer):
  """Reversible Layer."""

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    """Reverse this layer: compute input given output."""
    raise NotImplementedError

  def reverse_and_grad(self, output, grad, weights=(), state=(), new_state=(),
                       **kwargs):
    """Backward pass: computes the inverse of a layer and propagates gradients.

    While you may choose to only implement reverse, some layers implement this
    function directly as computation may be shared between reversing and
    computing gradients.

    Args:
      output: Output activations; can be a (possibly nested) tuple.
      grad: gradient signal (cotangent) computed based on subsequent layers.
        The structure and shape must match the output.
      weights: layer weights
      state: start state
      new_state: updated state computed by the forward pass
      **kwargs: kwargs for the layer

    Returns:
      A tuple (x, (x_grad, weights_grad)), where x is the reconstructed input,
      x_grad is the gradient signal for the input, and weights_grad is the
      gradient signal for the weights.
    """
    # Note: jax.vjp does not allow us to use **kwargs in the signature here.
    def _do_forward(x, weights):
      return super(ReversibleLayer, self).forward_with_state(
          x, weights=weights, state=state, **kwargs)[0]

    reconstructed_x = self.reverse(output, weights, state, new_state, **kwargs)
    _, vjpfun = jax.vjp(_do_forward, reconstructed_x, weights)
    x_weights_grad = vjpfun(grad)
    return reconstructed_x, x_weights_grad

  @property
  def has_backward(self):
    return True

  def backward(self, inputs, output, ct, weights, state, new_state, **kwargs):
    del inputs
    _, inputs_weights_ct = self.reverse_and_grad(output, ct, weights,
                                                 state, new_state, **kwargs)
    return inputs_weights_ct


class ReversibleSwap(ReversibleLayer, cb.Swap):
  """Swap the first two element on the stack."""

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    # Swap is its own inverse, except that reverse doesn't return the state.
    return self.forward_with_state(output, weights=weights, state=state,
                                   **kwargs)[0]


class ReversibleSerial(ReversibleLayer, cb.Serial):
  """A reversible version of tl.Serial (requires reversible sub-layers)."""

  def __init__(self, *layers):
    super(ReversibleSerial, self).__init__(*layers)

    # Note that sublayers has already been flattened to remove nested lists.
    for i, layer in enumerate(self.sublayers):
      if not isinstance(layer, ReversibleLayer):
        raise ValueError(
            'Sub-layer {} of ReversibleSerial is not reversible: {}'.format(
                i, layer))

  def reverse(self, output, weights=(), state=(), new_state=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = math.random.split(rng, self._n_layers)

    layer_val = output
    for layer, p, s, ns, rng in reversed(list(zip(
        self.sublayers, weights, state, new_state, rngs))):
      layer_val = layer.reverse(layer_val, p, s, ns, rng=rng, **kwargs)

    return layer_val

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = math.random.split(rng, self._n_layers)

    layer_val = output
    layer_ct = ct
    weights_ct = []
    for layer, p, s, ns, rng in reversed(list(zip(
        self.sublayers, weights, state, new_state, rngs))):
      layer_val, layer_ct = layer.reverse_and_grad(
          layer_val, layer_ct, p, s, ns, rng=rng, **kwargs)
      layer_ct, p_ct = layer_ct
      weights_ct.insert(0, p_ct)

    return layer_val, (layer_ct, weights_ct)
