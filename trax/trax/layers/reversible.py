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
"""Implementations of reversible layers."""

from trax import fastmath
from trax.layers import base
from trax.layers import combinators as cb

# pylint: disable=protected-access
_inputs_from_stack = cb._inputs_from_stack
_outputs_onto_stack = cb._outputs_onto_stack
# pylint: enable=protected-access


class ReversibleLayer(base.Layer):
  """Reversible Layer."""

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    """Reverse this layer: compute input given output."""
    raise NotImplementedError

  def reverse_and_grad(self, output, grad, weights=(), state=(), new_state=(),
                       rng=None):
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
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      A tuple (x, (x_grad, weights_grad)), where x is the reconstructed input,
      x_grad is the gradient signal for the input, and weights_grad is the
      gradient signal for the weights.
    """
    def _do_forward(x, weights):
      old_weights, old_state, old_rng = self._weights, self._state, self._rng
      self._state, self._rng = state, rng
      self._weights = weights
      res = self.forward(x)
      self._weights, self._state, self._rng = old_weights, old_state, old_rng
      return res

    reconstructed_x = self.reverse(output, weights, state, new_state, rng)
    _, vjpfun = fastmath.vjp(_do_forward, reconstructed_x, weights)
    x_weights_grad = vjpfun(grad)
    return reconstructed_x, x_weights_grad

  @property
  def has_backward(self):
    return True

  def backward(self, inputs, output, grad, weights, state, new_state, rng):
    del inputs
    _, inputs_weights_grad = (
        self.reverse_and_grad(output, grad, weights, state, new_state, rng))
    return inputs_weights_grad


class ReversibleSwap(ReversibleLayer):
  """Swap the first two element on the stack."""

  def __init__(self):
    super().__init__(n_in=2, n_out=2)

  def forward(self, inputs):
    x0, x1 = inputs
    return x1, x0

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng, weights
    # Swap is its own inverse, except that reverse doesn't return the state.
    return self.forward(output)


class ReversibleSerial(ReversibleLayer, cb.Serial):
  """A reversible version of tl.Serial (requires reversible sub-layers)."""

  def __init__(self, *layers):
    super().__init__(*layers)
  # def __init__(self, *layers):  # pylint: disable=super-init-not-called
  #   cb.Serial.__init__(self, layers)

    # Note that sublayers has already been flattened to remove nested lists.
    for i, layer in enumerate(self.sublayers):
      if not isinstance(layer, ReversibleLayer):
        raise ValueError(
            'Sub-layer {} of ReversibleSerial is not reversible: {}'.format(
                i, layer))

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = fastmath.random.split(rng, self._n_layers)

    stack = output
    for layer, p, s, ns, rng in reversed(list(zip(
        self.sublayers, weights, state, new_state, rngs))):
      layer_val = _inputs_from_stack(layer, stack, layer.n_out)
      layer_val = layer.reverse(layer_val, p, s, ns, rng=rng)
      stack = _outputs_onto_stack(
          layer, layer_val, stack, layer.n_out, layer.n_in)

    return stack

  def reverse_and_grad(self, output, grad, weights=(), state=(), new_state=(),
                       rng=None):
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = fastmath.random.split(rng, self._n_layers)

    stack = output
    stack_grad = grad
    weights_grad = []
    for layer, p, s, ns, rng in reversed(list(zip(
        self.sublayers, weights, state, new_state, rngs))):
      layer_val = _inputs_from_stack(layer, stack, layer.n_out)
      layer_ct = _inputs_from_stack(layer, stack_grad, layer.n_out)
      layer_val, layer_ct = layer.reverse_and_grad(
          layer_val, layer_ct, p, s, ns, rng=rng)
      layer_ct, p_ct = layer_ct
      weights_grad.insert(0, p_ct)
      stack = _outputs_onto_stack(
          layer, layer_val, stack, layer.n_out, layer.n_in)
      stack_grad = _outputs_onto_stack(
          layer, layer_ct, stack_grad, layer.n_out, layer.n_in)

    return stack, (stack_grad, weights_grad)
