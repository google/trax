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
"""Combinators for composing layers."""

import copy

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers.base import Fn
from trax.shapes import ShapeDtype


class Serial(base.Layer):
  """Combinator that applies layers serially (by function composition).

  This combinator is commonly used to construct deep networks, e.g., like this::

      mlp = tl.Serial(
        tl.Dense(128),
        tl.Relu(),
        tl.Dense(10),
      )

  A Serial combinator uses stack semantics to manage data for its sublayers.
  Each sublayer sees only the inputs it needs and returns only the outputs it
  has generated. The sublayers interact via the data stack. For instance, a
  sublayer k, following sublayer j, gets called with the data stack in the
  state left after layer j has applied. The Serial combinator then:

    - takes n_in items off the top of the stack (n_in = k.n_in) and calls
      layer k, passing those items as arguments; and

    - takes layer k's n_out return values (n_out = k.n_out) and pushes
      them onto the data stack.

  A Serial instance with no sublayers acts as a special-case (but useful)
  1-input 1-output no-op.
  """

  def __init__(self, *sublayers, name=None, sublayers_to_print=None):
    super().__init__(
        name=name, sublayers_to_print=sublayers_to_print)

    sublayers = _ensure_flat(sublayers)
    self._sublayers = sublayers
    self._n_layers = len(sublayers)

    if sublayers:
      self._n_in, self._n_out = self._n_inputs_n_outputs(sublayers)
      self._weights = tuple(None for l in sublayers)
      self._state = tuple(None for l in sublayers)

  def forward(self, xs):
    """Executes this layer as part of a forward pass through the model."""
    self._validate_forward_inputs(xs)
    if not self.sublayers:  # No-op: outputs = inputs
      return xs

    state, weights = self.state, self.weights
    rngs = _split_rngs(self.rng, self._n_layers)
    stack = xs
    new_state = []
    n_layers = self._n_layers
    if len(weights) != n_layers:
      raise ValueError(
          f'Number of weight elements ({len(weights)}) does not equal '
          f'number of sublayers ({n_layers}).')
    if len(state) != n_layers:
      raise ValueError(
          f'Number of state elements ({len(state)}) does not equal '
          f'number of sublayers ({n_layers}).')

    for layer, w, s, rng in zip(self.sublayers, weights, state, rngs):
      inputs = inputs_from_stack(stack, layer.n_in)
      outputs, s = layer.pure_fn(inputs, w, s, rng, use_cache=True)
      stack = outputs_onto_stack(outputs, stack, layer.n_in)
      new_state.append(s)
    self.state = tuple(new_state)
    return stack

  # pylint: disable=protected-access
  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    weights = []
    states = []
    # In the code below, stack, inputs, and outputs are abstract (shapes and
    # dtypes), but weights and states are non-abstract actual values.
    stack = input_signature
    for sublayer in self.sublayers:
      inputs = inputs_from_stack(stack, sublayer.n_in)
      weights_or_cache_marker, state_or_cache_marker = (
          sublayer.init(inputs, use_cache=True))
      outputs, _ = sublayer._forward_abstract(inputs)
      stack = outputs_onto_stack(outputs, stack, sublayer.n_in)

      weights.append(weights_or_cache_marker)
      states.append(state_or_cache_marker)
    self.state = tuple(states)
    self.weights = tuple(weights)
  # pylint: enable=protected-access

  def _n_inputs_n_outputs(self, layers):
    del self
    running_max = 0
    running_total = 0
    for layer in layers:
      running_total += layer.n_in
      running_max = max(running_max, running_total)
      running_total -= layer.n_out
    return running_max, (running_max - running_total)

  def _validate_forward_inputs(self, xs):
    if not isinstance(xs, (tuple, list)) and self._n_in != 1:
      raise TypeError(f'Serial.forward input must be a tuple or list; '
                      f'instead got {type(xs)}.')
      # TODO(jonni): Include full xs (or shape) in error message?
    len_xs = 1 if isinstance(xs, jnp.ndarray) else len(xs)
    if len_xs < self.n_in:
      raise ValueError(
          f'Number of inputs ({len(xs)}) to Serial.forward less than n_in '
          f'({self.n_in}).')


class Parallel(base.Layer):
  """Combinator that applies a list of layers in parallel to its inputs.

  Layers in the list apply to successive spans of inputs, where the spans are
  determined how many inputs each layer takes. The resulting output is the
  (flattened) concatenation of the respective layer outputs.

  For example, suppose one has three layers:

    - F: 1 input, 1 output
    - G: 3 inputs, 1 output
    - H: 2 inputs, 2 outputs (h1, h2)

  Then Parallel(F, G, H) will take 6 inputs and give 4 outputs:

    - inputs: a, b, c, d, e, f
    - outputs: F(a), G(b, c, d), h1, h2     where h1, h2 = H(e, f)

  As an important special case, a None argument to Parallel acts as if it takes
  one argument, which it leaves unchanged. (It acts as a one-arg no-op.) For
  example:

    Parallel(None, F)

  creates a layer that passes its first input unchanged and applies F to the
  following input(s).
  """

  def __init__(self, *sublayers, name=None):
    """The constructor.

    Args:
      *sublayers: A list of sublayers.
      name: Descriptive name for this layer.

    Returns:
      A new layer in which each of the given sublayers applies to its
      corresponding span of elements in the dataflow stack.
    """
    super().__init__(name=name)
    sublayers = self._validate(sublayers)
    self._n_layers = len(sublayers)
    self._sublayers = sublayers
    self._n_in = sum(l.n_in for l in sublayers)
    self._n_out = sum(l.n_out for l in sublayers)
    self._weights = tuple(None for l in sublayers)
    self._state = tuple(None for l in sublayers)

  def forward(self, inputs):
    """Executes this layer as part of a forward pass through the model."""
    n_layers, layers = self._n_layers, self.sublayers
    sublayer_inputs = self._allot_to_sublayers(inputs)
    state, weights = self.state, self.weights
    rngs = _split_rngs(self.rng, n_layers)
    if len(sublayer_inputs) != n_layers:
      raise ValueError(
          f'Number of inputs for sublayers ({len(sublayer_inputs)}) does not equal '
          f'number of sublayers ({n_layers}).')
    if len(weights) != n_layers:
      raise ValueError(
          f'Number of weight elements ({len(weights)}) does not equal '
          f'number of sublayers ({n_layers}).')
    if len(state) != n_layers:
      raise ValueError(
          f'Number of state elements ({len(state)}) does not equal '
          f'number of sublayers ({n_layers}).')
    if len(rngs) != n_layers:
      raise ValueError(
          f'Number of rngs ({len(rngs)}) does not equal '
          f'number of sublayers ({n_layers}).')
    outputs = []
    new_state = []
    for layer, x, w, s, r in zip(layers, sublayer_inputs, weights, state, rngs):
      # Note that zip silently truncates its result if lengths don't match.
      sub_outputs, sub_state = layer.pure_fn(x, w, s, r, use_cache=True)
      if layer.n_out == 1:
        outputs.append(sub_outputs)
      else:
        outputs.extend(sub_outputs)
      new_state.append(sub_state)
    output = outputs[0] if self.n_out == 1 else tuple(outputs)
    self.state = tuple(new_state)
    return output

  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    sublayer_signatures = self._allot_to_sublayers(input_signature)
    inits = [layer.init(signature, use_cache=True)
             for layer, signature
             in zip(self.sublayers, sublayer_signatures)]
    if inits:
      weights, state = tuple(zip(*inits))
      self.state = state
      self.weights = weights

  def _validate(self, layers):
    if not layers or len(layers) < 2:
      raise ValueError(
          f'layers ({layers}) must be a list with at least two elements')
    layers = list(layers)  # Ensure we can modify layers.
    for i, obj in enumerate(layers):
      if obj is None or obj == []:  # pylint: disable=g-explicit-bool-comparison
        layers[i] = Serial(None)
      elif isinstance(obj, (list, tuple)):
        layers[i] = Serial(obj)
      else:
        if not isinstance(obj, base.Layer):
          raise ValueError(
              f'Found nonlayer object ({obj}) in layers list: [{layers}]')
      if layers[i].n_in == 0:
        raise ValueError(
            f'Sublayer with n_in = 0 not allowed in Parallel: {layers[i]}')
    return layers

  def _allot_to_sublayers(self, inputs):
    """Divides Parallel's inputs for use by the sublayers.

    Args:
      inputs: Tuple of ndarrays or ShapeDtype instances.

    Returns:
      A tuple that partitions this layer's inputs among its sublayers.
      Sublayers that take one argument get that argument directly. All other
      sublayers get a tuple of items.
    """
    start, end = 0, 0
    sub_inputs = []
    for layer in self.sublayers:
      n_in = layer.n_in
      end = start + n_in
      if n_in == 1:
        sub_inputs.append(inputs[start])
      else:
        sub_inputs.append(inputs[start:end])
      start = end
    return tuple(sub_inputs)


class Concatenate(base.Layer):
  """Concatenates a number of tensors into a single tensor.

  For example::

      x = np.array([1, 2])
      y = np.array([3, 4])
      z = np.array([5, 6])
      concat3 = tl.Concatenate(n_items=3)
      z = concat3((x, y, z))  # z = [1, 2, 3, 4, 5, 6]

  Use the `axis` argument to specify on which axis to concatenate the tensors.
  By default it's the last axis, `axis=-1`, and `n_items=2`.
  """

  def __init__(self, n_items=2, axis=-1):
    name = 'Concatenate' if axis == -1 else f'Concatenate_axis{axis}'
    super().__init__(n_in=n_items, name=name)
    self._n_items = n_items
    self._axis = axis

  def forward(self, xs):
    """Executes this layer as part of a forward pass through the model."""
    return jnp.concatenate(xs, self._axis)


class Split(base.Layer):
  """Splits the input into n items along an axis."""

  def __init__(self, n_items=2, axis=-1):
    super().__init__(n_out=n_items)
    self._n_items = n_items
    self._axis = axis

  def forward(self, inputs):
    """Executes this layer as part of a forward pass through the model."""
    return tuple(jnp.split(inputs, self._n_items, self._axis))


def _scan(f, xs, init_value, axis=0, remat=False):
  """Scans the f over the given axis of xs.

  In pseudo-python, the scan function would look as follows:

  def scan(f, xs, init_value, axis):
    xs  = [xs[..., i, ...] for i in range(xs.shape[axis])]
    cur_value = init_value
    ys = []
    for x in xs:
      y, cur_value = f(x, cur_value)
      ys.append(y)
    return np.stack(ys, axis), cur_value

  Args:
    f: function (x, carry) -> (y, new_carry)
    xs: tensor, x will be xs slices on axis
    init_value: tensor, initial value of the carry-over
    axis: int, the axis on which to slice xs
    remat: whether to re-materialize f

  Returns:
    A pair (ys, last_value) as described above.
  """
  def swapaxes(x):
    transposed_axes = list(range(len(x.shape)))
    transposed_axes[axis] = 0
    transposed_axes[0] = axis
    return jnp.transpose(x, axes=transposed_axes)
  if axis != 0:
    xs = fastmath.nested_map(swapaxes, xs)
  def transposed_f(c, x):
    y, d = f(x, c)
    return d, y
  if remat:
    transposed_f = fastmath.remat(transposed_f)
  last_value, ys = fastmath.scan(transposed_f, init_value, xs)
  if axis != 0:
    ys = fastmath.nested_map(swapaxes, ys)
  return ys, last_value


class Scan(base.Layer):
  """Applies a layer progressively/cumulatively to an axis-derived sequence.

  Conceptually, this is a function from a list to a same-length list of partial
  (cumulative) results. For instance, a list of values (`[1, 2, 3, 4, 5]`) can
  transform to a list of cumulative sums (`[1, 3, 6, 10, 15]`). Functions for
  the same concept are called `scan` in Scala, `scanl` in Haskell, and
  `accumulate*` in Factor.

  In more detail, we assume the layer takes a tuple of inputs of the following
  form:

    (input1, ..., inputN, carry1, ..., carryM)

  and returns:

    (output1, ..., outputK, new_carry1, ..., new_carryM)

  The scanned version applies the layer iteratively to a tensor treating values
  at the given axis as if they were a list. For example, to calculate all
  sums of prefixes of a tensor, we can do this::

    def add(x, carry):
      def f(input, carry):
        res = input + carry
        return res, res  # output and carry are the same
      return tl.Fn('add', f, n_out=2)

    Scan(add)([1, 2, 3], 0) = [1, 3, 6], 6
  """

  def __init__(self, layer, axis=0, n_carry=1, remat=False, mode='train'):
    super().__init__(n_in=layer.n_in, n_out=layer.n_out)
    self._sublayers = [layer]
    self._n_carry = n_carry
    self._axis = axis
    self._remat = remat
    self._weights = (None,)
    self._state = (None, ())
    self._mode = mode

  @property
  def sublayer(self):
    """Returns the unique sublayer managed by this layer."""
    return self._sublayers[0]

  @property
  def state(self):
    """Returns a tuple containing this layer's state."""
    return (self.sublayer.state, self._state[1])

  @state.setter
  def state(self, state):
    """Recursively sets state on this layer the sublayer."""
    if isinstance(state, dict) and state == base.GET_STATE_FROM_CACHE:
      return
    self._state = (None, state[1])
    self.sublayer.state = state[0]

  def forward(self, inputs):
    """Executes this layer as part of a forward pass through the model."""
    weights = self.weights[0]
    if isinstance(inputs, list):
      inputs = tuple(inputs)  # so that inputs structure matches outputs
    n_carry = self._n_carry
    def scannable_fn(x, carry_and_state):  # pylint: disable=invalid-name
      carry, state, i = carry_and_state
      x_and_carry = x + carry if n_carry > 0 else x
      rng = fastmath.random.fold_in(self.rng, i)
      res, new_state = self.sublayer.pure_fn(
          x_and_carry, weights, state, rng, use_cache=True)
      if n_carry > 0:
        return (res[:-n_carry], (res[-n_carry:], new_state, i+1))
      else:
        return (res, ([], new_state, i+1))

    if n_carry > 0:
      xs = inputs[:-n_carry]  # Split input stack into inputs and carry.
      xs_carry = inputs[-n_carry:]
      if self._mode == 'predict' and self._state[1] is not ():  # pylint: disable=literal-comparison
        xs_carry = self._state[1]
      init = (xs_carry, self.state[0], jnp.array(0, dtype=jnp.int32))
    else:
      xs_carry = ()
      xs, init = inputs, ([], self.state[0], jnp.array(0, dtype=jnp.int32))
    ys, (carry, new_state, _) = _scan(scannable_fn, xs, init,
                                      axis=self._axis, remat=self._remat)
    res = ys + carry if n_carry > 0 else ys
    state_carry = carry if self._mode == 'predict' and n_carry > 0 else ()
    self.state = (new_state, state_carry)
    return res  # Put outputs and carry back on stack.

  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    n_carry = self._n_carry
    if n_carry == 0:
      if isinstance(input_signature, (list, tuple)):
        layer_sig = [ShapeDtype(_shape_without_axis(x, self._axis), x.dtype)
                     for x in input_signature]
        layer_sig = tuple(layer_sig)
      else:
        layer_sig = ShapeDtype(_shape_without_axis(input_signature, self._axis),
                               input_signature.dtype)
      weights, state = self.sublayer.init(layer_sig)
      self.state = (state, ())
      self.weights = (weights,)
    else:
      xs = input_signature[:-n_carry]
      init = input_signature[-n_carry:]
      xs_slices = [ShapeDtype(_shape_without_axis(x, self._axis), x.dtype)
                   for x in xs]
      layer_signature = tuple(xs_slices + list(init))
      weights, state = self.sublayer.init(layer_signature, use_cache=True)
      self.state = (state, ())
      self.weights = (weights,)


class Cond(base.Layer):
  """Applies layers conditionally.

  For parameters `cond`, `true`, and `false` runs the equivalent of `true(y)
  if cond(x) else false(y)`, where `x` is `cond.n_in` elements from front of the
  stack and `y` is the rest of the stack.
  Exactly one of `true` and `false` functions is executed, so it can be used to
  conditionally run long computations. The state of non-executed function is not
  updated. Note that different branches may be executed on different devices
  if `cond` returns different values on them.
  By default 'false' function is an identity.

  `cond` must return exactly one element: a Boolean value.
  `true` and `false` must have the same n_in, and the same n_out.
  """

  def __init__(self, cond, true, false=None, name=None):
    super(Cond, self).__init__(name=name)

    if false is None:
      self._identity_false_fun = True
      # We don't need this function, but it will be useful for checking if
      # 'true' has proper n_in/n_out.
      false = Serial()
      self._false = false
    else:
      self._identity_false_fun = False
      self._false = false

    sublayers = [cond, true, false]
    self._sublayers = sublayers
    self._n_layers = len(sublayers)
    self._cond = cond
    self._true = true

    if cond.n_out != 1:
      raise ValueError(
          'cond.n_out must be 1: cond:{}->{}'.format(cond.n_in, cond.n_out))
    if true.n_in != false.n_in:
      raise ValueError(
          'true.n_in and false.n_in must be equal: true:{}->{} ; false:{}->{}'
          .format(true.n_in, true.n_out, false.n_in, false.n_out))
    if true.n_out != false.n_out:
      raise ValueError(
          'true.n_out and false.n_out must be equal: true:{}->{} ; false:{}->{}'
          .format(true.n_in, true.n_out, false.n_in, false.n_out))

    self._n_in = cond.n_in + true.n_in
    self._n_out = true.n_out
    self._weights = tuple(None for l in sublayers)
    self._state = tuple(None for l in sublayers)

  # pylint: disable=protected-access
  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    weights = []
    states = []
    # In the code below, stack, inputs, and outputs are abstract (shapes and
    # dtypes), but weights and states are non-abstract actual values.
    stack = _make_tuple(input_signature)

    # Inputs/outputs of `cond`.
    inputs = inputs_from_stack(stack, self._cond.n_in)
    weights_or_cache_marker, state_or_cache_marker = (
        self._cond.init(inputs, use_cache=True))
    weights.append(weights_or_cache_marker)
    states.append(state_or_cache_marker)
    self._cond._forward_abstract(inputs)
    stack = _make_tuple(outputs_onto_stack([], stack, self._cond.n_in))

    # Inputs/outputs of `true` and `false`.
    for sublayer in [self._true, self._false]:
      inputs = inputs_from_stack(stack, sublayer.n_in)
      weights_or_cache_marker, state_or_cache_marker = (
          sublayer.init(inputs, use_cache=True))
      weights.append(weights_or_cache_marker)
      states.append(state_or_cache_marker)

    self.state = states
    self.weights = weights
    # pylint: enable=protected-access

  def _validate_forward_inputs(self, xs):
    xs = _make_tuple(xs)
    if len(xs) < self.n_in:
      raise ValueError(
          f'Number of inputs ({len(xs)}) to Cond.forward less than n_in '
          f'({self.n_in}).')

  def forward(self, xs):
    """Executes this layer as part of a forward pass through the model.

    Args:
      xs: Tensors of as required by the branches of this conditional.

    Returns:
      Tensors resulting from running the chosen branch.
    """
    # TODO(jaszczur): modify; it's a copy from SkippingSerial
    self._validate_forward_inputs(xs)
    layers_state = self.state
    # Get 3 rngs, one for each layer.
    rngs = _split_rngs(self.rng, 3)

    # Prepare the stack and do some safety checks as in the parent class.
    stack = _make_tuple(xs)
    weights = self.weights
    if len(weights) != 3:
      raise ValueError('number of weights ({}) not equal to 3'
                       .format(len(weights)))
    if len(layers_state) != 3:
      raise ValueError('length of state ({}) not equal to 3'
                       .format(len(layers_state)))

    def true_func(t):
      outputs, new_true_state = self._true.pure_fn(
          t[0][0], t[1][0], t[2][0], t[3][0])
      # t[2][1] is old_false_state which is not changing if true is executed.
      return outputs, (new_true_state, t[2][1])

    def false_func(t):
      if self._identity_false_fun:
        # Memory optimization: we don't need pure_fn call.
        return t[0][1], t[2]
      outputs, new_false_state = self._false.pure_fn(
          t[0][1], t[1][1], t[2][1], t[3][1])
      # t[2][1] is old_true_state, which is not changing if false is executed.
      return outputs, (t[2][0], new_false_state)

    cond_inputs = inputs_from_stack(xs, self._cond.n_in)
    cond_output, s = self._cond.pure_fn(cond_inputs, self.weights[0],
                                        self.state[0], rngs[0], use_cache=True)
    stack = outputs_onto_stack([], stack, self._cond.n_in)
    self._cond.state = s

    outputs, both_states = fastmath.cond(
        cond_output,
        true_func,
        false_func,
        [(stack, stack),
         (self.weights[1], self.weights[2]),
         (self.state[1], self.state[2]),
         (rngs[1], rngs[2])]
    )
    stack = outputs_onto_stack([], stack, self._cond.n_in)

    # We don't know which (`true` or `false`) branch was run, but both of them
    # are adding (n_out) and removing (n_in) the same number of elements of the
    # stack (this was checked in __init__). outputs_onto_stack just uses the
    # layer's n_in, so we can pass either `true` or `false` to it.
    # Note that `outputs` is the actual output of `true` or `false` branch,
    # whichever was run, and we add it to the stack in any case.
    stack = outputs_onto_stack(outputs, stack, self._true.n_in)
    self._true.state = both_states[0]
    self._false.state = both_states[1]
    return _make_singleitem_or_original(stack)


# pylint: disable=invalid-name
def Chunk(layer, chunk_size, pass_unchunkable=True):
  """Executes `layer` using batch chunks of size `chunk_size` to save memory."""
  if chunk_size < 1:
    return layer
  def reshape_to_chunks(x):
    chunk_batch = x.shape[0]
    size = chunk_size
    n_chunks = chunk_batch // size
    if chunk_batch % size != 0:
      if pass_unchunkable:
        n_chunks = 1
        size = chunk_batch
      else:
        raise ValueError(f'Chunk size {size} must divide batch '
                         f'size {chunk_batch}')
    return jnp.reshape(x, [n_chunks, size] + list(x.shape[1:]))
  reshape_to_chunks_layer = base.PureLayer(
      lambda xs: fastmath.nested_map(reshape_to_chunks, xs),
      n_in=layer.n_in, n_out=layer.n_in, name='ReshapeToChunks')
  def reshape_from_chunks(x):
    batch_size = x.shape[0] * x.shape[1]
    return jnp.reshape(x, [batch_size] + list(x.shape[2:]))
  reshape_from_chunks_layer = base.PureLayer(
      lambda xs: fastmath.nested_map(reshape_from_chunks, xs),
      n_in=layer.n_out, n_out=layer.n_out, name='ReshapeFromChunks')
  return Serial(
      reshape_to_chunks_layer,
      Scan(layer, axis=0, n_carry=0, remat=True),
      reshape_from_chunks_layer,
  )


def Branch(*layers, name='Branch'):
  """Combinator that applies a list of layers in parallel to copies of inputs.

  Each layer in the input list is applied to as many inputs from the stack
  as it needs, and their outputs are successively combined on stack.

  For example, suppose one has three layers:

    - F: 1 input, 1 output
    - G: 3 inputs, 1 output
    - H: 2 inputs, 2 outputs (h1, h2)

  Then Branch(F, G, H) will take 3 inputs and give 4 outputs:

    - inputs: a, b, c
    - outputs: F(a), G(a, b, c), h1, h2    where h1, h2 = H(a, b)

  As an important special case, a None argument to Branch acts as if it takes
  one argument, which it leaves unchanged. (It acts as a one-arg no-op.)

  Args:
    *layers: List of layers.
    name: Descriptive name for this layer.

  Returns:
    A branch layer built from the given sublayers.
  """
  if len(layers) == 1:
    return layers[0]
  parallel_layer = Parallel(*layers)
  indices = [list(range(layer.n_in)) for layer in parallel_layer.sublayers]
  return Serial(Select(_deep_flatten(indices)), parallel_layer,
                name=name, sublayers_to_print=layers)


def Residual(*layers, shortcut=None):
  """Wraps a series of layers with a residual connection.

  Args:
    *layers: One or more layers, to be applied in series.
    shortcut: If None (the usual case), the Residual layer computes the
        element-wise sum of the stack-top input with the output of the layer
        series. If specified, the `shortcut` layer applies to a copy of the
        inputs and (elementwise) adds its output to the output from the main
        layer series.

  Returns:
      A layer representing a residual connection paired with a layer series.
  """
  layers = _ensure_flat(layers)
  layer = layers[0] if len(layers) == 1 else Serial(layers)
  # TODO(jonni): Should we require layer.n_out = 1 and shortcut.n_out = 1?
  return Serial(
      Branch(shortcut, layer),
      Add(),  # pylint: disable=no-value-for-parameter
  )


def Select(indices, n_in=None, name=None):
  """Copies, reorders, or deletes stack elements according to `indices`.

  Args:
    indices: A list or tuple of 0-based indices to select elements relative to
        the top of the stack.
    n_in: Number of input elements to pop from the stack, and replace with
        those specified by `indices`. If not specified, its value will be
        calculated as `max(indices) + 1`.
    name: Descriptive name for this layer.

  Returns:
    Tensors, matching the number selected (`n_out = len(indices)`).
    Specifically:

      - n_out = 0: an empty tuple
      - n_out = 1: one tensor (NOT wrapped in a tuple)
      - n_out > 1: a tuple of tensors, with n_out items
  """
  if n_in is None:
    n_in = max(indices) + 1
  if name is None:
    name = f'Select{indices}'.replace(' ', '')

  def select(xs):  # pylint: disable=invalid-name
    if not isinstance(xs, (tuple, list)):
      xs = (xs,)
    selected = tuple(xs[i] for i in indices)
    return selected[0] if len(selected) == 1 else selected

  return base.PureLayer(select, n_in=n_in, n_out=len(indices), name=name)


def Drop():
  """Drops the top stack element."""
  return Fn('Drop', lambda x: (), n_out=0)


def Dup():
  """Duplicates (copies) the top element on the data stack."""
  return Fn('Dup', lambda x: (x, x), n_out=2)


def Swap():
  """Swaps the top two stack elements."""
  return Fn('Swap', lambda x0, x1: (x1, x0), n_out=2)


def SerialWithSideOutputs(layers, n_side_outputs=1):
  """Serial layer with side outputs.

  This layer makes it easier to manage the stack when layers have side outputs.

  In the simplest case of layers with n_in=1, n_out=2 and with
  n_side_outputs=1, this layer runs the following computation on x::

    side_outputs = []
    for i in range(len(layers)):
      x, side_output = layers[i](x)
      side_outputs.append(side_output)
    return [x] + side_outputs

  In the general case of layers with variable n_in and n_out and
  n_side_outputs being a list of N integers, it does the following::

    side_outputs = []
    for i in range(N):
      res = layer[i](cur_stack)  # remove n_in from stack
      cur_stack.append(res[:n_side_outputs[i]])  # put back some on stack
      side_outputs.extend(res[n_side_outputs:])
    return cur_stack + side_outputs

  Args:
    layers: a list of layers to execute
    n_side_outputs: an int or a list of ints, how many outputs of each layer
        to put aside

  Returns:
    A layer that performs the above computation.
  """
  if isinstance(n_side_outputs, int):
    n_side_outputs = [n_side_outputs] * len(layers)

  # Calculate the n_in for this layer.
  running_max = 0
  running_total = 0
  for layer, n_side_output in zip(layers, n_side_outputs):
    running_total += layer.n_in
    running_max = max(running_max, running_total)
    running_total -= layer.n_out - n_side_output
  n_in = running_max

  # Create the list of layers to run serially.
  cur_stack_size = n_in
  serial_layers = []
  for layer, n_side_output in zip(layers, n_side_outputs):
    serial_layers.append(layer)
    cur_stack_size += layer.n_out - layer.n_in
    # Indices to move n_side_outputs to the back of the stack.
    # Don't touch first n_out - n_side_outputs.
    move_back_indices = list(range(layer.n_out - n_side_output))
    # Then comes the rest of the stack that we're not moving.
    move_back_indices += [i + layer.n_out
                          for i in range(cur_stack_size - layer.n_out)]
    # Finally the indices we move.
    move_back_indices += [i + layer.n_out - n_side_output
                          for i in range(n_side_output)]
    # Swap them on stack.
    serial_layers.append(Select(move_back_indices))

  return Serial(serial_layers)


def FlattenList():
  """Flatten lists."""
  # TODO(jonni): Consider renaming layer to DeepFlatten.
  return Fn('FlattenList', lambda x: tuple(_deep_flatten(x)))


def Add():
  """Adds two tensors."""
  return Fn('Add', lambda x0, x1: x0 + x1)


def SubtractTop():
  """Subtracts the first tensor from the second."""
  return Fn('SubtractTop', lambda x0, x1: x1 - x0)


def Multiply():
  """Multiplies two tensors."""
  return Fn('Multiply', lambda x0, x1: x0 * x1)


def Gate():
  """Returns a gating layer on a (memory, gate, candidate) tuple.

  Final update is memory * gate + (1 - gate) * candidate

  This gating equation may also be referred to as Highway Network.
  Highway Networks: https://arxiv.org/abs/1505.00387
  """
  return Fn('Gate', lambda m, g, c: g * m + (1.0 - g) * c)


class Cache(base.Layer):
  """Applies a layer on the first run and returns the outputs on next calls."""

  def __init__(self, layer):
    super().__init__(n_in=layer.n_in, n_out=layer.n_out)
    self._sublayers = [layer]

  @property
  def sublayer(self):
    """Returns the unique sublayer managed by this layer."""
    return self._sublayers[0]

  @property
  def state(self):
    """Returns a tuple containing this layer's state; may be empty."""
    return self._state

  @state.setter
  def state(self, state):
    """Recursively sets state on this layer and all sublayers."""
    if isinstance(state, dict) and state == base.GET_STATE_FROM_CACHE:
      return
    self._state = state
    self.sublayer.state = state[1]

  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    weights, layer_state = self.sublayer.init(input_signature, use_cache=True)
    self.state = ((), layer_state)
    self._weights = (weights,)

  def forward(self, inputs):
    """Executes this layer as part of a forward pass through the model.

    Args:
      inputs: Tensors required by the sublayer.

    Returns:
      Tensors resulting from running the sublayer the first time.
    """
    state, weights = self.state, self.weights[0]
    if state[0] is ():  # pylint: disable=literal-comparison
      res, layer_state = self.sublayer.pure_fn(
          inputs, weights, state[1], self.rng)
      self.state = (res, layer_state)
      return res
    else:
      return state[0]


class BatchLeadingAxes(base.Layer):
  """Applies a layer after flattening all but n_last_axes_to_keep to batch.

  This can be used to make layers accept an arbitrary number of leading
  axes (dimensions) as batch. For example, a Convolution layer may normally
  only operate on tensors of shape [B, W, H, C]. In this case, the layer

      BatchLeadingAxes(Convolution(), n_last_axes_to_keep=3)

  will operate on any tensor [..., W, H, C] and treat the leading axes as batch.
  """

  def __init__(self, layer, n_last_axes_to_keep=1):
    if layer.n_out != 1:
      raise ValueError('BatchLeadingAxes currently only works for layers with '
                       f'n_out = 1, got {layer.n_out}.')
    super().__init__(n_in=layer.n_in, n_out=layer.n_out)
    self._sublayers = [layer]
    self._n_last_axes_to_keep = n_last_axes_to_keep
    self._weights = (None,)
    self._state = (None,)

  @property
  def sublayer(self):
    """Returns the unique sublayer managed by this layer."""
    return self._sublayers[0]

  def forward(self, inputs):
    """Executes this layer as part of a forward pass through the model."""
    if self._n_in == 1:
      inputs = [inputs]
    new_inputs = []
    for old_input in inputs:
      batched_axes_shape = list(old_input.shape[:-self._n_last_axes_to_keep])
      batched_shape = [-1] + list(old_input.shape[-self._n_last_axes_to_keep:])
      new_inputs.append(jnp.reshape(old_input, batched_shape))
    new_inputs = tuple(new_inputs)
    if self._n_in == 1:
      new_inputs = new_inputs[0]
    res, layer_state = self.sublayer.pure_fn(
        new_inputs, self.weights[0], self.state[0], self.rng)
    self.state = (layer_state,)
    return jnp.reshape(res, batched_axes_shape + list(res.shape[1:]))

  def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
    if self._n_in == 1 and not isinstance(input_signature, (list, tuple)):
      input_signature = (input_signature,)
    batched_signature = []
    for sub_input_signature in input_signature:
      batched_size = 1
      for d in sub_input_signature.shape[:-self._n_last_axes_to_keep]:
        batched_size *= d
      batched_shape = [batched_size] + list(
          sub_input_signature.shape[-self._n_last_axes_to_keep:])
      batched_signature.append(ShapeDtype(batched_shape,
                                          sub_input_signature.dtype))
    if self._n_in == 1:
      batched_signature = batched_signature[0]
    weights, layer_state = self.sublayer.init(batched_signature, use_cache=True)
    self.state = (layer_state,)
    self.weights = (weights,)


def Bidirectional(forward_layer, axis=1, merge_layer=Concatenate()):
  """Bidirectional combinator for RNNs.

  Args:
    forward_layer: A layer, such as `trax.layers.LSTM` or `trax.layers.GRU`.
    axis: a time axis of the inputs. Default value is `1`.
    merge_layer: A combinator used to combine outputs of the forward
      and backward RNNs. Default value is 'trax.layers.Concatenate'.

  Example:
      Bidirectional(RNN(n_units=8))

  Returns:
    The Bidirectional combinator for RNNs.
  """
  backward_layer = copy.deepcopy(forward_layer)
  flip = base.Fn('_FlipAlongTimeAxis', lambda x: jnp.flip(x, axis=axis))
  backward = Serial(
      flip,
      backward_layer,
      flip,
  )

  return Serial(
      Branch(forward_layer, backward),
      merge_layer,
  )


# All module-private helper functions are below.
# pylint: disable=invalid-name


def _deep_flatten(items):
  """Returns a list of objects, flattening sublists/subtuples along the way.

  Example: _deep_flatten([1, (2, 3, (4, 5), [6, 7]), [[[8]]]]) would return
  the list [1, 2, 3, 4, 5, 6, 7, 8].

  Args:
    items: An iterable. If elements of this iterable are lists or tuples, they
        will be (recursively) flattened until non-list non-tuple objects are
        reached.

  Returns:
    A list of non-list, non-tuple objects.
  """
  def _flat_gen(xs):
    for x in xs:
      if isinstance(x, (list, tuple)):
        for y in _flat_gen(x):
          yield y
      else:
        yield x
  return list(_flat_gen(items))


def _ensure_sublayers(layers):
  """Ensures that elements in a layer list are layers.

  Args:
    layers: A tuple or list whose elements can each be a layer, tuple, or list,
        and so on recursively.

  Returns:
    An analogous collection of layers in which embedded layer lists are
    wrapped in Serial layer instances.
  """
  if not layers:  # None or an empty list can signal a no-op.
    return Serial(None)  # no-op, but still handles shapes and initialization
  elif isinstance(layers, (list, tuple)):
    sublayers_not_lists = []
    for layer in layers:
      sublayers_not_lists.append(
          Serial(layer) if isinstance(layer, (list, tuple)) else layer)
    return sublayers_not_lists
  else:
    raise TypeError(type(layers))


def _split_rngs(rng, n_copies):
  if rng is None:
    return (None,) * n_copies
  return fastmath.random.split(rng, n_copies)


def inputs_from_stack(stack, n):
  """Returns n inputs from stack."""
  stack = _make_tuple(stack)
  return _make_singleitem_or_original(stack[:n])


def outputs_onto_stack(outputs, stack, n):
  """"Returns the new stack after removing n items and pushing outputs there."""
  outputs = _make_tuple(outputs)
  stack = _make_tuple(stack)
  return _make_singleitem_or_original(outputs + stack[n:])


def _make_tuple(xs):
  """Returns a tuple from a list, a tuple, or a single element."""
  if isinstance(xs, (list, tuple)):
    return tuple(xs)
  else:
    return (xs,)


def _make_singleitem_or_original(xs):
  """Returns a single element if possible, or the original list/tuple if not."""
  if isinstance(xs, (list, tuple)) and len(xs) == 1:
    return xs[0]
  else:
    return xs


def _shape_without_axis(x, axis):
  return x.shape[:axis] + x.shape[axis + 1:]


def _ensure_flat(layers):
  """Ensures that layers is a single flat list of Layer instances."""
  if len(layers) == 1 and layers[0] is None:
    layers = ()
  else:
    layers = _deep_flatten(layers)
  for obj in layers:
    if not isinstance(obj, base.Layer):
      raise ValueError(
          f'Found nonlayer object ({obj}) in layers: {layers}')
  return layers
