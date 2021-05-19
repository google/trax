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
"""Layers that can run in reverse to compute inputs from outputs.

Reversible layers reduce the memory required for backpropagation-based
training, especially for *deep* networks. In a series of reversible layers,
input activations from a forward pass don't need to be stored: they can be
reconstructed on the backward pass, layer by layer, from outputs to inputs.

See, e.g., [The Reversible Residual Network: Backpropagation Without Storing
Activations](https://arxiv.org/abs/1707.04585) and [Reformer: The Efficient
Transformer](https://arxiv.org/abs/2001.04451).
"""

from absl import logging
import jax
from trax import fastmath
from trax.layers import base
from trax.layers import combinators as cb


_split_rngs = cb._split_rngs  # pylint: disable=protected-access


class ReversibleLayer(base.Layer):
  """Reversible Layer."""

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    """Reverse this layer: compute input given output."""
    raise NotImplementedError

  def _pure_forward(self, x, weights, state, rng):
    """Call self.forward in a pure way."""
    old_weights, old_state, old_rng = self.weights, self.state, self._rng
    self.weights, self.state, self._rng = weights, state, rng
    res = self.forward(x)
    self.weights, self.state, self._rng = old_weights, old_state, old_rng
    return res

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
    reconstructed_x = self.reverse(output, weights, state, new_state, rng)
    _, vjpfun = fastmath.vjp(
        self._pure_forward, reconstructed_x, weights, state, rng)
    x_grad, weights_grad, _, _ = vjpfun(grad)
    return reconstructed_x, (x_grad, weights_grad)

  @property
  def has_backward(self):
    return True

  def backward(self, inputs, output, grad, weights, state, new_state, rng):
    del inputs
    _, inputs_weights_grad = (
        self.reverse_and_grad(output, grad, weights, state, new_state, rng))
    return inputs_weights_grad


class ReversibleConcatenatePair(ReversibleLayer):
  """Maps (x, y) -> ([x, y], [x, y]);  [x, y] is concatenation on last axis."""

  def __init__(self):
    super().__init__(n_in=2, n_out=2)

  def forward(self, inputs):
    x, y = inputs
    r = fastmath.numpy.concatenate((x, y), axis=-1)
    return r, r

  def reverse(self, outputs, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng, weights
    pair, _ = outputs
    x, y = fastmath.numpy.split(pair, 2, axis=-1)
    return x, y


class ReversibleSelect(ReversibleLayer):
  """Reversible version of the Select combinator."""

  def __init__(self, indices, n_in=None, name=None):
    if n_in is None:
      n_in = max(indices) + 1
    if name is None:
      name = f'ReversibleSelect{indices}'.replace(' ', '')
    super().__init__(n_in=n_in, n_out=len(indices), name=name)
    self._indices = indices

    # Calculate reverse indices.
    self._reverse_indices = []
    for i in range(n_in):
      if i not in indices:
        raise ValueError('To be reversible, all inputs to Select must be in '
                         'indices. Did not find %d in indices.' % i)
      else:
        self._reverse_indices.append(indices.index(i))

  def forward(self, inputs):
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    selected = tuple(inputs[i] for i in self._indices)
    return selected[0] if len(selected) == 1 else selected

  def reverse(self, outputs, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng, weights
    if not isinstance(outputs, (tuple, list)):
      outputs = (outputs,)
    selected = tuple(outputs[i] for i in self._reverse_indices)
    return selected[0] if len(selected) == 1 else selected


def ReversibleSwap():  # pylint: disable=invalid-name
  return ReversibleSelect([1, 0], name='ReversibleSwap')


class ReversibleReshape(ReversibleLayer):
  """Reversible reshaping layer."""

  def __init__(self, shape1, shape2, n_in=1):
    self._shape1 = list(shape1)
    self._shape2 = list(shape2)
    name = 'ReversibleReshape_%s_%s' % (str(shape1), str(shape2))
    super().__init__(n_in=n_in, n_out=n_in, name=name)

  def forward(self, inputs):
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    res = []
    for x in inputs:
      new_shape = self._shape2 + list(x.shape)[len(self._shape1):]
      res.append(fastmath.numpy.reshape(x, new_shape))
    if len(res) == 1:
      return res[0]
    return tuple(res)

  def reverse(self, outputs, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng, weights
    if not isinstance(outputs, (tuple, list)):
      outputs = (outputs,)
    res = []
    for x in outputs:
      new_shape = self._shape1 + list(x.shape)[len(self._shape2):]
      res.append(fastmath.numpy.reshape(x, new_shape))
    if len(res) == 1:
      return res[0]
    return tuple(res)


class ReversiblePrintShape(ReversibleLayer):
  """Reversible PrintShape for debugging reversible serial layers."""

  def __init__(self, n_in=1, msg=''):
    super().__init__(n_in=n_in, n_out=n_in)
    self._msg = msg

  def forward(self, xs):
    shapes_and_dtypes = ', '.join([str(x.shape) + f'[{x.dtype}]' for x in xs])
    info = f'PrintShape: {self._msg}: [{shapes_and_dtypes}]'
    print(info)
    logging.info(info)
    return xs

  def reverse(self, outputs, weights=(), state=(), new_state=(), rng=None):
    del state, new_state, rng, weights
    return outputs


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
      layer_val = cb.inputs_from_stack(stack, layer.n_out)
      layer_val = layer.reverse(layer_val, p, s, ns, rng=rng)
      stack = cb.outputs_onto_stack(layer_val, stack, layer.n_out)

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
      layer_val = cb.inputs_from_stack(stack, layer.n_out)
      layer_ct = cb.inputs_from_stack(stack_grad, layer.n_out)
      layer_val, layer_ct = layer.reverse_and_grad(
          layer_val, layer_ct, p, s, ns, rng=rng)
      layer_ct, p_ct = layer_ct
      weights_grad.insert(0, p_ct)
      stack = cb.outputs_onto_stack(layer_val, stack, layer.n_out)
      stack_grad = cb.outputs_onto_stack(layer_ct, stack_grad, layer.n_out)

    return stack, (stack_grad, tuple(weights_grad))


class ReversibleHalfResidual(ReversibleLayer):
  """Half of a RevNet-style residual that optionally performs attention.

  When attention_layer is None, this layer has the signature ::

      [accumulator, *context] -> [accumulator + f(context), *context]

  The attention_layer must be an instance of EfficientAttentionBase or one of
  its subclasses (see efficient_attention.py), or None.

  Attention is special-cased for the following two reasons:

  - LSH attention needs to save bucket assignments from the forward pass to the
    backward pass, for training stability. This requires special-casing it.
  - We can call attention_layer.forward_and_or_backward to compute its output
    (needed for inverting a reversible residual layer) while simultaneously
    performing the backward pass. Sharing computation between these two
    operations improves training speed.
  """

  def __init__(self, *residual_layers, attention_layer=None, name=None):
    super().__init__(name=name)

    self._compute_residual = cb.Serial(*residual_layers)
    self._attention_layer = attention_layer

    if self._attention_layer is None:
      self._sublayers = (self._compute_residual,)
    else:
      if hasattr(attention_layer, 'forward_and_or_backward'):
        self._forward_and_or_backward = attention_layer.forward_and_or_backward
      else:
        self._forward_and_or_backward = _forward_and_or_backward(
            attention_layer)
      self._sublayers = (self._compute_residual, self._attention_layer)

    running_max = 0
    running_total = 0
    for layer in self._sublayers:
      running_total += layer.n_in
      running_max = max(running_max, running_total)
      running_total -= layer.n_out
    self._n_in = self._n_out = running_max + 1

  def forward(self, xs):
    rngs = _split_rngs(self.rng, len(self.sublayers))
    accumulator, *context = xs
    stack = context = tuple(context)
    new_state = []
    for layer, w, s, rng in zip(self.sublayers, self.weights, self.state, rngs):
      inputs = cb.inputs_from_stack(stack, layer.n_in)
      if base.N_WEIGHTS_SHARDS > 1:
        # With sharded weights, make sure we don't keep them concatenated
        # in memory on each device by using remat.
        outputs, s = jax.remat(layer.pure_fn)(inputs, w, s, rng)
      else:
        outputs, s = layer.pure_fn(inputs, w, s, rng)
      stack = cb.outputs_onto_stack(outputs, stack, layer.n_in)
      new_state.append(s)
    residual = stack[0] if isinstance(stack, (tuple, list)) else stack

    output = accumulator + residual
    stack = (output,) + context
    self.state = tuple(new_state)
    return stack

  def reverse(self, output, weights=(), state=(), new_state=(), rng=None):
    raise NotImplementedError('Only reverse_and_grad is actually used.')

  def reverse_and_grad(self, output, ct, weights=(), state=(), new_state=(),
                       rng=None):
    rngs = _split_rngs(rng, len(self.sublayers))

    accumulator_output, *context = output
    context = tuple(context)
    accumulator_output_ct, *context_ct = ct
    context_ct = tuple(context_ct)

    # Forward pass through self._compute_residual. Outputs that will not receive
    # a gradient signal from subsequent layers are moved to aux.
    def call_compute_residual(x, weights):
      state_to_pass = state[0]  # old_state

      # _replace_second_time is currently used exclusively in _RememberInReverse
      # layer to combat numerical instability in Terraformer when quantizing
      # the mask in SparseFF.
      def _replace_second_time(stt, nstt):
        if (isinstance(stt, tuple) and len(stt) == 2 and
            isinstance(stt[1], dict) and 'running_second_time' in stt[1]):
          return (nstt[0], {'running_second_time_yes': ()})
        elif isinstance(stt, (tuple, list)):
          assert isinstance(nstt, (tuple, list)) and len(nstt) == len(stt)
          return type(stt)([
              _replace_second_time(s, ns) for s, ns in zip(stt, nstt)])
        else:
          return stt

      state_to_pass = _replace_second_time(state_to_pass, new_state[0])
      res, _ = self._compute_residual.pure_fn(
          x, weights=weights, state=state_to_pass, rng=rngs[0])
      if not isinstance(res, (tuple, list)):
        return res, None
      else:
        n_differentiable = 1
        if self._attention_layer is not None:
          n_differentiable = min(len(res), self._attention_layer.n_in)
        return res[:n_differentiable], res[n_differentiable:]

    stack = context
    inputs = cb.inputs_from_stack(stack, self._compute_residual.n_in)
    outputs, compute_residual_vjpfun, outputs_aux = fastmath.vjp(
        call_compute_residual, inputs, weights[0], has_aux=True)
    if outputs_aux is not None:
      n_differentiable_outputs = len(outputs)
      outputs = outputs + outputs_aux
    stack = cb.outputs_onto_stack(outputs, stack, self._compute_residual.n_in)

    stack_ct = accumulator_output_ct
    if self._attention_layer is None:
      residual = stack[0] if isinstance(stack, (tuple, list)) else stack
    else:
      inputs = cb.inputs_from_stack(stack, self._attention_layer.n_in)
      (residual, _, attn_inputs_ct, attn_weights_ct
      ) = self._forward_and_or_backward(
          inputs, weights[1], new_state[1], rngs[1],
          output_grad=accumulator_output_ct,
          compute_output=True, update_state=False)
      stack_ct = cb.outputs_onto_stack(
          attn_inputs_ct, stack_ct, self._attention_layer.n_out)

    compute_residual_ct = cb.inputs_from_stack(
        stack_ct, self._compute_residual.n_out)
    if outputs_aux is not None:
      if not isinstance(compute_residual_ct, (tuple, list)):
        compute_residual_ct = (compute_residual_ct,)
      compute_residual_ct = compute_residual_ct[:n_differentiable_outputs]
      assert len(compute_residual_ct) == n_differentiable_outputs
    (compute_residual_inputs_ct, compute_residual_weights_ct
    ) = compute_residual_vjpfun(compute_residual_ct)
    stack_ct = cb.outputs_onto_stack(
        compute_residual_inputs_ct, stack_ct, self._compute_residual.n_out)
    if not isinstance(stack_ct, (tuple, list)):
      stack_ct = (stack_ct,)
    def _add(x, y):
      # `None` is for TFNP backend, which uses `None` as the gradient of
      # int/bool instead of an array of dtype `float0`.
      if x is None or x.dtype == jax.float0:
        return y
      if y is None or y.dtype == jax.float0:
        return x
      return x + y
    stack_ct = (accumulator_output_ct,) + fastmath.nested_map_multiarg(
        _add, context_ct[:len(stack_ct)], stack_ct) + context_ct[len(stack_ct):]

    reconstructed_x = accumulator_output - residual
    stack = (reconstructed_x,) + context
    if self._attention_layer is None:
      weights_ct = (compute_residual_weights_ct,)
    else:
      weights_ct = (compute_residual_weights_ct, attn_weights_ct)
    return stack, (stack_ct, weights_ct)

  # pylint: disable=protected-access
  def init_weights_and_state(self, input_signature):
    stack = input_signature[1:]
    if len(stack) == 1:
      stack = stack[0]

    inputs = cb.inputs_from_stack(stack, self._compute_residual.n_in)
    weights, state = self._compute_residual.init(inputs)
    outputs, _ = self._compute_residual._forward_abstract(inputs)
    stack = cb.outputs_onto_stack(outputs, stack, self._compute_residual.n_in)

    if self._attention_layer is None:
      self.state = (state,)
      self.weights = (weights,)
    else:
      inputs = cb.inputs_from_stack(stack, self._attention_layer.n_in)
      attn_weights, attn_state = self._attention_layer.init(inputs)
      self.state = (state, attn_state)
      self.weights = (weights, attn_weights)
  # pylint: enable=protected-access


def _forward_and_or_backward(layer):
  """Create forward_and_or_backward for layers that don't define it."""

  def forward_and_or_backward(inputs, weights, state, rng, output_grad=None,
                              compute_output=True, update_state=True):
    """Performs batched forward and/or backward passes.

    Args:
      inputs: inputs to the attention layer
      weights: weights for the attention layer
      state: state of the attention layer
      rng: PRNG key for the layer (shared across all examples and heads)
      output_grad: gradient of the loss wrt the output of the layer, or None.
          This function performs the backward pass iff `output_grad` is not
          None.
      compute_output: bool: whether to return the output of the forward pass
          (for example, a pure backwards pass does not need to return the
          output).
      update_state: bool: whether to return an updated layer state.

    Returns:
      A tuple (output, new_state, inputs_grad, weights_grad).
      - output is not None iff compute_output is True
      - new_state is not None iff update_state is True
      - inputs_grad & weights_grad are not None iff output_grad is not None
    """
    # Calculate the vector-Jacobian product of the layer pure_fn.
    output, vjp_fn, new_state = fastmath.vjp(
        layer.pure_fn, inputs, weights, state, rng, has_aux=True)
    output = output if compute_output else None
    new_state = new_state if update_state else None

    # The vjp function returns gradients with respect to inputs and weights.
    if output_grad is not None:
      grads_inputs, grads_weights, _, _ = vjp_fn(output_grad)
    else:
      grads_inputs, grads_weights = None, None

    return (output, new_state, grads_inputs, grads_weights)
  return forward_and_or_backward
