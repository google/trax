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
"""Base layer class."""

import copy
import inspect
import pickle
import traceback

import jax
import numpy as onp
import tensorflow as tf

from trax import math
from trax.math import nested_map
from trax.math import numpy as np
from trax.shapes import ShapeDtype
from trax.shapes import signature


EMPTY_WEIGHTS = ()
EMPTY_STATE = ()


class Layer(object):
  """Base class for composable layers in a deep learning network.

  Layers are the basic building blocks for deep learning models. A Trax layer
  computes a function from zero or more inputs to zero or more outputs,
  optionally using trainable weights (common) and non-parameter state (not
  common). Authors of new layer subclasses typically override at most two
  methods of the base `Layer` class:

    forward(inputs, weights):
      Computes this layer's output as part of a forward pass through the model.

    new_weights(self, input_signature):
      Returns new weights suitable for inputs with the given signature.

  A small subset of layer types are combinators -- they organize the computation
  of their sublayers, e.g., applying their sublayers in series or in parallel.

  All layers have the following properties, with default values implemented
  in the base `Layer` class:

    - n_in: int (default 1)
    - n_out: int (default 1)
    - weights: tuple (default empty -- the layer has no weights)
    - state: tuple (default empty -- the layer has no non-parameter state)
    - sublayers: tuple (default empty -- the layer has no sublayers)

  The inputs to a layer are tensors, packaged according to how many there are:

    - n_in = 0: an empty tuple ()
    - n_in = 1: one tensor (NOT wrapped in a tuple)
    - n_in > 1: a tuple of tensors

  (The special treatment of the single-input case is meant to simplify the
  work of layer writers; this design choice may be revisited in the future.)

  The outputs from a layer are also tensors, packaged the same as layer inputs:

    - n_out = 0: an empty tuple ()
    - n_out = 1: the tensor (NOT wrapped in a tuple)
    - n_out > 1: a tuple of tensors

  The Trax runtime maintains a data stack with which layer calls are composed.
  For more complex data network architectures, possibly involving multiple data
  flows, one can view each layer as a function from stack state to stack state,
  where the function's inputs are a slice from the stack, and the function's
  outputs are spliced back into the stack.
  """

  def __init__(self, n_in=1, n_out=1, name=None):
    """Creates a partially initialized, unconnected layer instance.

    Args:
      n_in: Number of inputs expected by this layer.
      n_out: Number of outputs promised by this layer.
      name: Descriptive name for this layer.
    """
    self._n_in = n_in
    self._n_out = n_out
    self._name = name or self.__class__.__name__
    self._sublayers = ()  # Default is no sublayers.
    self._input_signature = None
    self._rng = None
    self._weights = EMPTY_WEIGHTS  # cached weights
    self._state = EMPTY_STATE
    # record root call site for custom error messages:
    frame = _find_frame(inspect.currentframe())
    # Turns out that frame can mutate in time, so we just copy what we need.
    self._caller = {'filename': copy.copy(frame.f_code.co_filename),
                    'lineno': int(frame.f_lineno)}
    del frame  # Just in case.
    self._init_finished = False
    self._jit_cache = {}

  def __repr__(self):
    name_str = self._name
    n_in, n_out = self.n_in, self.n_out
    if n_in != 1: name_str += f'_in{n_in}'
    if n_out != 1: name_str += f'_out{n_out}'
    objs = self.sublayers
    if objs:
      objs_str = ' '.join(str(x) for x in objs)
      return f'{name_str}[ {objs_str} ]'
    else:
      return name_str

  def __call__(self, x, weights=None, state=None, rng=None, n_accelerators=0):
    """Makes Layer instances callable; for use in tests or interactive settings.

    This convenience method helps library users play with, test, or otherwise
    probe the behavior of layers outside of a full training environment. It
    presents the layer as callable function from inputs to outputs, with the
    option of manually specifying weights and non-parameter state per individual
    call. For convenience, weights and non-parameter state are cached per layer
    instance, starting from default values of `EMPTY_WEIGHTS` and `EMPTY_STATE`,
    and acquiring non-empty values either by initialization or from values
    explicitly provided via the weights and state keyword arguments.

    Args:
      x: 0 or more input tensors, formatted the same as the inputs to
          Layer.forward.
      weights: Weights or None; if None, use self's cached weights value.
      state: State or None; if None, use self's cached state value.
      rng: rng object or None; if None, use a default computed from an
          integer 0 seed.
      n_accelerators: Number of accelerators to target.

    Returns:
      0 or more output tensors, formatted the same as the outputs from
          Layer.forward.
    """
    weights = self.weights if weights is None else weights
    state = self.state if state is None else state
    rng = self._rng if rng is None else rng
    rng = math.random.get_prng(0) if rng is None else rng

    forward_w_s_r = self.pure_fn
    # TODO(lukaszkaiser): n_accelerators is experimental, to decide on API
    if n_accelerators:
      if n_accelerators not in self._jit_cache:
        self._jit_cache[n_accelerators] = (
            jit_forward(forward_w_s_r, n_accelerators))
      forward_w_s_r = self._jit_cache[n_accelerators]
    outputs, new_state = forward_w_s_r(x, weights, state, rng)
    self.state = new_state
    self.weights = weights
    return outputs

  def forward(self, inputs, weights):
    """Computes this layer's output as part of a forward pass through the model.

    Authors of new Layer subclasses should override this method to define the
    forward computation that their layer performs, unless they need to use
    local non-trainable state or randomness, in which case they should
    override `forward_with_state` instead.

    Args:
      inputs: Input tensors, matching the number (n_in) expected by this
          layer. Specifically:
            - n_in = 0: an empty tuple ()
            - n_in = 1: a tensor (NOT wrapped in a tuple)
            - n_in > 1: a tuple of tensors, with n_in items
      weights: A tuple of trainable weights, with one element for this layer
          if this layer has no sublayers, or one for each sublayer if this
          layer has sublayers. If a layer (or sublayer) has no trainable
          weights, the corresponding weights element is an empty tuple.

    Returns:
      Tensors, matching the number (n_out) promised by this layer.
      Specifically:
        - n_out = 0: an empty tuple
        - n_out = 1: one tensor (NOT wrapped in a tuple)
        - n_out > 1: a tuple of tensors, with n_out items
    """
    raise NotImplementedError

  def forward_with_state(self, inputs, weights=EMPTY_WEIGHTS, state=EMPTY_STATE,
                         rng=None):
    """Computes this layer's output as part of a forward pass through the model.

    Authors of new Layer subclasses should override this method to define the
    forward computation that their layer performs only if their layer uses
    local state or randomness. Otherwise override `forward` instead.

    Args:
      inputs: Input tensors, matching the number (n_in) expected by this
          layer. Specifically:
            - n_in = 0: an empty tuple ()
            - n_in = 1: a tensor (NOT wrapped in a tuple)
            - n_in > 1: a tuple of tensors, with n_in items
      weights: A tuple of trainable weights, with one element for this layer
          if this layer has no sublayers, or one for each sublayer if this
          layer has sublayers. If a layer (or sublayer) has no trainable
          weights, the corresponding weights element is an empty tuple.
      state: Layer-specific non-parameter state that can update between batches.
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      A tuple of (tensors, state). The tensors match the number (n_out) promised
      by this layer, and are formatted according to that number, specifically:
        - n_out = 0: an empty tuple
        - n_out = 1: one tensor (NOT wrapped in a tuple)
        - n_out > 1: a tuple of tensors, with n_out items
    """
    # Default implementation only computes with inputs and weights.
    del rng
    return self.forward(inputs, weights), state

  def new_weights(self, input_signature):
    """Returns new weights suitable for inputs with the given signature.

    Authors of new Layer subclasses should override this method if their layer
    uses trainable weights. The default implementation works for layers that
    have no weights. Layers that have trainable state should override the
    `new_weights_and_state` method instead.

    Args:
      input_signature: A ShapeDtype instance (if this layer takes one input)
          or a list/tuple of ShapeDtype instances; signatures of inputs.
    """
    del input_signature
    return EMPTY_WEIGHTS

  def new_weights_and_state(self, input_signature):
    """Returns a (weights, state) pair suitable for initializing this layer.

    Authors of new Layer subclasses should override this method if their layer
    uses trainable weights or has non-parameter state that gets updated
    between batches. The default implementation works for layers that have
    no weights or state.

    Args:
      input_signature: A ShapeDtype instance (if this layer takes one input)
          or a list/tuple of ShapeDtype instances.
    """
    return self.new_weights(input_signature), EMPTY_STATE

  @property
  def has_backward(self):
    """Returns True if this layer provides its own (custom) backward pass code.

    A layer subclass that provides custom backward pass code (for custom
    gradients) must override this method to return True.
    """
    return False

  def backward(self, inputs, output, grad, weights, state, new_state, rng):
    """Custom backward pass to propagate gradients in a custom way.

    Args:
      inputs: Input tensors; can be a (possibly nested) tuple.
      output: The result of running this layer on inputs.
      grad: gradient signal (called cotangent in jax) computed based on
        subsequent layers. The structure and shape must match output.
      weights: layer weights
      state: start state.
      new_state: end state computed by running the layer
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      The custom gradient signal for the input. Note that we need to return
      a gradient for each argument of forward, so it will usually be a tuple
      of signals: the gradient for inputs and weights.
    """
    raise NotImplementedError

  # End of public subclassing interface.
  # Begin public callable interface.

  def init(self, input_signature, rng=None):
    """Initializes this layer and its sublayers recursively.

    This method is designed to initialize each layer instance once, even if the
    same layer instance occurs in multiple places in the network. This enables
    weight sharing to be implemented as layer sharing.

    Args:
      input_signature: `ShapeDtype` instance (if this layer takes one input)
          or list/tuple of `ShapeDtype` instances.
      rng: Single-use random number generator (JAX PRNG key). If none is
          provided, a default rng based on the integer seed 0 will be used.

    Returns:
      A (weights, state) tuple, in which weights contains newly created weights
          on the first call and `EMPTY_WEIGHTS` on all subsequent calls.
    """
    try:
      if self._rng is None:
        rng = math.random.get_prng(0) if rng is None else rng
        self._set_rng_recursive(rng)
      # Initialize weights once; store them for use when this layer is called.
      # Needs to call new_weights_and_state regardless of _init_finished because
      # state also needs to be initialized. After jitting, graph pruning should
      # be able to remove unnecessary computation.
      # TODO(lukaszkaiser): Revisit this decision and see whether layers sharing
      #   weights should also share states.
      weights, state = self.new_weights_and_state(input_signature)
      if not self._init_finished:
        self._init_finished = True
        self._weights = weights
        self._state = state
        return (weights, state)
      else:
        return (EMPTY_WEIGHTS, state)
    except Exception as e:
      name, trace = self._name, _short_traceback(skip=3)
      raise LayerError(name, 'init', self._caller,
                       input_signature, trace) from e

  def init_from_file(self, file_name, weights_only=False):
    """Initializes this layer and its sublayers from a file.

    We assume that the file is a pickled dictionary that contains the fields
    'weights' and 'state' with structures corresponding to this layers weights
    and state. Note that the pickled dictionary is allowed to contain other
    fields too, but these two are required to init.

    Args:
      file_name: the name of the file to initialize from.
      weights_only: if True, initialize only the weights, not state.
    """
    with tf.io.gfile.GFile(file_name, 'rb') as f:
      dictionary = pickle.load(f)
    self.weights = dictionary['weights']
    if not weights_only:
      self.state = dictionary['state']

  def new_rng(self):
    """Returns a new single-use random number generator (JAX PRNG key)."""
    self._rng, rng = math.random.split(self._rng)
    return rng

  def new_rngs(self, n):
    """Returns `n` single-use random number generators (JAX PRNG keys).

    Args:
      n: The number of rngs to return; must be an integer > 0.

    Returns:
      A tuple of `n` rngs. Successive calls will yield continually new values.
    """
    if n < 1:
      raise ValueError(f"Requested number of new rng's ({n}) less than 1.")
    rngs = math.random.split(self._rng, n + 1)
    self._rng = rngs[0]
    return tuple(rngs[1:])

  # End of public callable methods.
  # Methods and properties below are reserved for internal use.

  @property
  def n_in(self):
    """Returns how many tensors this layer expects as input."""
    return self._n_in

  @property
  def n_out(self):
    """Returns how many tensors this layer promises as output."""
    return self._n_out

  @property
  def sublayers(self):
    """Returns a tuple containing this layer's sublayers; may be empty."""
    return self._sublayers

  @property
  def input_signature(self):
    """Returns this layer's input signature.

    An input signature is a ShapeDtype instance (if the layer takes one input)
    or a tuple of ShapeDtype instances.
    """
    return self._input_signature

  @property
  def weights(self):
    """Returns this layer's weights.

    Depending on the layer, the weights can be in the form of:
      - an empty tuple
      - a tensor (ndarray)
      - a nested structure of tuples and tensors
    TODO(jonni): Simplify this picture (and underlying implementation).
    """
    return self._weights

  @weights.setter
  def weights(self, weights):
    self._weights = weights

  @property
  def state(self):
    """Returns a tuple containing this layer's state; may be empty."""
    return self._state

  @state.setter
  def state(self, state):
    self._state = state

  def pure_fn(self, x, weights, state, rng):
    """Applies this layer as a pure function with no optional args.

    This method exposes the layer's computation as a pure function. This is
    esp. useful for JIT compilation. Do not override, use `forward` instead.

    Args:
      x: See Layer.forward_with_state inputs.
      weights: See Layer.forward_with_state.
      state: See Layer.forward_with_state.
      rng: See Layer.forward_with_state.

    Returns:
      See Layer.forward_with_state.
    """
    try:
      # If weights are nothing, we may be reusing this layer.
      # Use the cached weights to calculate the value.
      # Note: to make sure jit tracers can decide this branch in python we use
      # `weights is EMPTY_WEIGHTS` instead of, e.g., `not weights` or
      # `weights == EMPTY_WEIGHTS`.
      if weights is EMPTY_WEIGHTS:  # pylint: disable=literal-comparison
        weights = self._weights
      else:
        # In this case, we're called for the first time: cache weights.
        self._weights = weights

      if not self.has_backward:
        outputs, s = (
            self.forward_with_state(x, weights=weights, state=state, rng=rng))
      else:
        outputs, s = self._do_custom_gradients(x, weights, state, rng=rng)
      self._state = s
      return outputs, s

    except Exception as e:
      name, trace = self._name, _short_traceback()
      raise LayerError(name, 'pure_fn',
                       self._caller, signature(x), trace) from e

  def output_signature(self, input_signature):
    """Returns output signature this layer would give for `input_signature`."""
    return self._forward_abstract(input_signature)[0]  # output only, not state

  def _forward_abstract(self, input_signature):
    """Computes shapes and dtypes this layer would produce in a forward pass.

    Args:
      input_signature: ShapeDtype instance (if this layer takes one input)
          or list/tuple of ShapeDtype instances.

    Returns:
      Tuple of (output, state).

      The output part of the tuple is a ShapeDtype instance representing the
      shape and type of the output (if this layer has one output) or a tuple
      of ShapeDtype instances (if this layer has more than one output).
    """
    try:
      # Note: By using rng_signature in place of an rng, we avoid computing and
      # permanently storing in global memory a large number of dropout masks.
      # TODO(jonni): Check if using an rng still carries this cost.
      rng_signature = ShapeDtype((2,), onp.uint32)
      weight_signature = nested_map(signature, self.weights)

      # Wrap forward_with_state so as to use only positional args.
      def _forward_with_state(x, weights, state, rng):
        return self.forward_with_state(x, weights=weights, state=state, rng=rng)

      forward_infer_shapes = math.abstract_eval(_forward_with_state)
      return forward_infer_shapes(
          input_signature, weight_signature, self.state, rng_signature)
    except Exception as e:
      name, trace = self._name, _short_traceback(skip=3)
      raise LayerError(name, '_forward_abstract', self._caller, input_signature,
                       trace) from e

  # pylint: disable=protected-access
  def _set_rng_recursive(self, rng):
    """Sets the rng (JAX PRNG key) for this layer and sublayers, recursively."""
    self._rng = rng
    sublayers = self.sublayers
    if sublayers:
      rngs = math.random.split(rng, len(sublayers))
      for sublayer, rng in zip(sublayers, rngs):
        sublayer._set_rng_recursive(rng)

  def _set_input_signature_recursive(self, input_signature):
    """Sets input_signatures for this layer and sublayers, recursively.

    General combinators (those that can take multiple sublayers) must override
    this method to calculate and set input signatures for the sublayers. (See
    the `Serial` class in combinators.py for an example.)

    Args:
      input_signature: A `ShapeDtype` instance (if this layer takes one input)
          or a list/tuple of `ShapeDtype` instances
    """
    self._input_signature = input_signature

    # Handle the special case of a single immediate sublayer (which may in turn
    # have its own sublayers).
    sublayers = self.sublayers
    if sublayers and len(sublayers) == 1:
      sublayers[0]._set_input_signature_recursive(input_signature)
    if sublayers and len(sublayers) > 1:
      raise ValueError('A layer class whose instances can have more than one '
                       'sublayer must override the input_signature property '
                       'setter.')
  # pylint: enable=protected-access

  def replicate(self, n_accelerators):
    """Replicate weights and state for use on n accelerators. Experimental."""
    if n_accelerators > 1:
      self.weights = for_n_devices(self.weights, n_accelerators)
      self.state = for_n_devices(self.state, n_accelerators)

  def unreplicate(self, unreplicate_state=False):
    """Unreplicate weights and optionally state. Experimental."""
    self.weights = math.nested_map(self.weights, lambda x: x[0])
    if unreplicate_state:
      self.state = math.nested_map(self.state, lambda x: x[0])

  def _do_custom_gradients(self, x, weights, state, rng):
    """Calls this layer for a forward pass, but with custom gradients."""
    assert math.backend_name() == 'jax', (
        'Custom gradients are only supported in JAX for now.')

    # See this link for how custom transformations are defined in JAX:
    # https://jax.readthedocs.io/en/latest/jax.html#jax.custom_transforms
    @jax.custom_transforms
    def _do_forward(y, weights):
      res = self.forward_with_state(y, weights=weights, state=state, rng=rng)
      return res

    # This is the custom gradient (vector-jacobian product in JAX) function.
    # For the exact specification of this custom transformation see this link:
    # https://jax.readthedocs.io/en/latest/jax.html#jax.defjvp_all
    def do_forward_vjp(y, weights):
      """Custom gradient (vjp) function."""
      output, new_state = (
          self.forward_with_state(y, weights=weights, state=state, rng=rng))
      def vjpfun(grad):
        grad = grad[0]  # Ignore dummy gradient wrt state.
        res = self.backward(y, output, grad, weights, state, new_state, rng)
        return res
      return (output, new_state), vjpfun

    jax.defvjp_all(_do_forward, do_forward_vjp)
    output, state = _do_forward(x, weights)
    state = jax.lax.stop_gradient(state)
    return output, state


def layer(n_in=1, n_out=1, name=None):
  """Returns a decorator that converts a function into a Layer class builder."""

  def _build_layer_class(raw_fn):
    """Returns a Layer class whose callable instances execute the function."""

    def _init(self, **kwargs):
      self._kwargs = kwargs  # pylint: disable=protected-access
      Layer.__init__(self, n_in=n_in, n_out=n_out, name=name)

    def _forward(self, inputs, weights):
      """Uses this layer as part of a forward pass through the model."""
      del weights
      _validate_forward_input(inputs, n_in)
      raw_output = raw_fn(inputs, **self._kwargs)  # pylint: disable=protected-access
      output = () if _is_empty(raw_output) else raw_output
      return output

    def _is_empty(raw_output):
      return raw_output is None or (isinstance(raw_output, (list, tuple))
                                    and len(raw_output) == 0)  # pylint: disable=g-explicit-length-test

    # Set docstrings and create the class.
    _forward.__doc__ = raw_fn.__doc__
    # Note: None.__doc__ is None
    cls = type(raw_fn.__name__, (Layer,),
               {'__init__': _init,
                'forward': _forward})
    return cls

  return _build_layer_class


def Fn(f, n_in=None, n_out=None):  # pylint: disable=invalid-name
  """Returns a layer with no weights that applies the function f.

  The function f can take and return any number of arguments, but it cannot
  have default arguments or keywords arguments. It can use numpy though, e.g:

  A layer that takes 2 arguments and returns sum and concatenation on stack:

    Fn(lambda x, y: (x + y, np.concatenate([x, y], axis=0)))

  Sometimes determining the number of outputs automatically fails,
  in such cases specify n_in and n_out.

  Args:
    f: the function to execute
    n_in: optional, number of inputs
    n_out: optional, number of outputs

  Returns:
    A layer executing the function f.
  """
  # Inspect the function f to restrict to no-defaults and no-kwargs functions.
  argspec = inspect.getfullargspec(f)
  varkwargs = argspec.varkw
  # This layer cannot handle functions with kwargs or defaults.
  if argspec.defaults is not None:
    raise ValueError('Function has default arguments (not allowed).')
  if varkwargs:
    raise ValueError('Function has keyword arguments (not allowed).')

  # Determine n_in from function signature if not set.
  if n_in is None:
    if argspec.varargs is not None:
      raise ValueError('Argument n_in is not set and f has variable args.')
    n_in = len(argspec.args)
  # Try to determine n_out from function signature.
  if n_out is None:
    try:
      dummy_args = [np.array([[0.0]]) for _ in range(n_in)]
      res = f(*dummy_args)
      n_out = len(res) if isinstance(res, (list, tuple)) else 1
    except Exception as e:
      raise ValueError(
          'Argument n_out is not set and could not be determined.') from e

  # Create the layer.
  @layer(n_in=n_in, n_out=n_out)
  def F(xs, **unused_kwargs):  # pylint: disable=invalid-name
    if not isinstance(xs, (tuple, list)):
      xs = (xs,)
    return f(*xs)
  return F()  # pylint: disable=no-value-for-parameter


class LayerError(Exception):
  """Exception raised in the layer stack.

  Attributes:
    message: the message corresponding to this exception.
  """

  def __init__(self, layer_name, function_name, caller,
               input_signature, traceback_string):
    self._layer_name = layer_name
    self._function_name = function_name
    self._caller = caller  # Python inspect object with init caller info.
    self._traceback = traceback_string
    self._input_signature = input_signature
    super(LayerError, self).__init__(self.message)

  @property
  def message(self):
    """Create error message."""
    prefix = 'Exception passing through layer '
    prefix += '%s (in %s):\n' % (self._layer_name, self._function_name)
    short_path = '[...]/' + '/'.join(
        self._caller['filename'].split('/')[-3:])
    caller = '  layer created in file %s, line %d\n' % (short_path,
                                                        self._caller['lineno'])
    shapes_str = '  layer input shapes: %s\n\n' % str(self._input_signature)
    return prefix + caller + shapes_str + self._traceback


def check_shape_agreement(layer_obj, input_signature):
  """Compares the layer's __call__ output to its _foward_abstract shape output.

  This function helps test layer mechanics and inter-layer connections that
  aren't dependent on specific data values.

  Args:
    layer_obj: A layer object.
    input_signature: A `ShapeDtype` instance (if `layer_obj` takes one input)
        or a list/tuple of ShapeDtype instances.

  Returns:
    A tuple representing either a single shape (if the layer has one output) or
    a tuple of shape tuples (if the layer has more than one output).
  """
  weights, state = layer_obj.init(input_signature)
  output_signature, _ = layer_obj._forward_abstract(input_signature)  # pylint: disable=protected-access
  if isinstance(output_signature, tuple):
    shape_output = tuple(x.shape for x in output_signature)
  else:
    shape_output = output_signature.shape

  rng1, rng2 = layer_obj.new_rngs(2)
  random_input = _random_values(input_signature, rng1)
  call_output = layer_obj(random_input, weights=weights, state=state, rng=rng2)
  call_output_shape = _shapes(call_output)

  msg = '_foward_abstract shape output %s != __call__ output shape %s' % (
      shape_output, call_output_shape)
  assert shape_output == call_output_shape, msg
  # TODO(jonni): Remove this assert? It makes test logs harder to read.
  return shape_output


def _validate_forward_input(x, n_in):
  if n_in != 1:
    if not isinstance(x, tuple):
      raise TypeError(f'Expected input to be a tuple; instead got {type(x)}.')
    if len(x) != n_in:
      raise ValueError(f'Input tuple length ({len(x)}) does not equal required '
                       f'number of inputs ({n_in}).')


def _find_frame(frame):
  """Find the frame with the caller on the stack."""
  # TODO(lukaszkaiser): rewrite this function in a systematic way.
  # We want to find the first place where the layer was called
  # that is *not* an __init__ function of an inheriting layer.
  # We also need to exclude a few decorator functions.
  while frame.f_code.co_name in ['__init__', 'gin_wrapper', '_validate',
                                 '_validate_forward_inputs', '_init']:
    # We only skip __init__ in internal layers, return otherwise.
    try:
      dirname = frame.f_code.co_filename.split('/')[-2]
    except IndexError:
      # Notebook cells have dummy filenames that do not contain any slashes
      dirname = frame.f_code.co_filename
    if dirname != 'layers' and frame.f_code.co_name == '__init__':
      return frame
    # If we are in an init, move up.
    frame = frame.f_back
  return frame


def _shorten_file_path(line):
  """Shorten file path in error lines for more readable tracebacks."""
  start = line.lower().find('file')
  if start < 0:
    return line
  first_quote = line.find('"', start)
  if first_quote < 0:
    return line
  second_quote = line.find('"', first_quote + 1)
  if second_quote < 0:
    return line
  path = line[first_quote + 1:second_quote]
  new_path = '/'.join(path.split('/')[-3:])
  return line[:first_quote] + '[...]/' + new_path + line[second_quote + 1:]


def _short_traceback(skip=3):
  """Cleaned-up form of traceback."""
  counter, res = 0, []
  # Skipping 3 lines by default: the top (useless) and self-call.
  # In python 3, we need to set chain to False (it doesn't exist in python 2).
  lines = traceback.format_exc(chain=False).splitlines()[skip:]  # pylint: disable=unexpected-keyword-arg
  for l in lines:
    if l.startswith('trax.layers.base.LayerError'):
      l = l[len('trax.layers.base.'):]  # Remove the trax.layers.base prefix.
    res.append(_shorten_file_path(l))
    if counter % 2 == 1:
      res.append('')
    counter += 1
    # If we see a LayerError, the traceback has already been processed.
    if l.startswith('LayerError'):
      # Skip 4 back except last as these are internal base-layer calls.
      res = res[:-4] + [res[-1]]
      res += lines[counter:]
      break
  return '\n'.join(res)


def _random_values(input_signature, rng):
  """Creates random floats or ints of the given shape.

  Args:
    input_signature: A `ShapeDtype` instance (if `layer_obj` takes one input)
        or a list/tuple of ShapeDtype instances.
    rng: Single-use random number generator (JAX PRNG key).

  Returns:
    Random values with the shape and type specified.
  """
  if isinstance(input_signature, ShapeDtype):
    shape, dtype = input_signature.shape, input_signature.dtype
    if onp.issubdtype(dtype, onp.integer):
      return math.random.bernoulli(rng, 0.5, shape).astype(onp.int32)
    else:
      return math.random.uniform(rng, shape, minval=-1.0, maxval=1.0)
  elif isinstance(input_signature, (list, tuple)):
    return tuple(_random_values(x, rng) for x in input_signature)
  else:
    raise TypeError(type(input_signature))


def _shapes(x):
  """Get a structure of shapes for a structure of nested arrays."""
  def shape(x):
    try:
      return tuple([int(i) for i in x.shape])
    except Exception:  # pylint: disable=broad-except
      return ()
  return tuple(nested_map(shape, x))


def jit_forward(forward, n_devices):
  """Returns a JIT-compiled forward function running on n_devices."""
  model_predict = _accelerate(forward, n_devices)
  if n_devices == 1:
    return model_predict

  def predict(x, weights, state, rng):
    """Predict function jited and parallelized as requested."""
    res, state = _combine_devices(model_predict(
        reshape_by_device(x, n_devices),
        weights,
        state,
        np.stack(math.random.split(rng, n_devices))))
    return math.nested_map(lambda y: np.mean(y, axis=0), res), state

  return predict


def _combine_devices(x_tuple):
  """Combine multi-device tensors into a single batch."""
  def f(x):
    if len(x.shape) < 2:
      return x  # No extra batch dimension: use devices as batch, so return.
    batch_size = x.shape[0] * x.shape[1]
    return math.numpy.reshape(x, [batch_size] + list(x.shape[2:]))
  return math.nested_map(f, x_tuple)


def _accelerate(f, n_devices):
  """JITed version of f running on n_devices."""
  if n_devices == 1:
    return math.jit(f)

  return math.pmap(f, axis_name='batch')


def reshape_by_device(x, n_devices):
  """Reshapes possibly nested x into a shape (n_devices, ...)."""
  def f(x):
    x_shape = list(x.shape)
    batch_size = x_shape[0]
    batch_size_per_device = batch_size // n_devices
    if batch_size_per_device * n_devices != batch_size:
      raise ValueError(f'Number of devices ({n_devices}) does not evenly '
                       f'divide batch size ({batch_size}).')
    new_shape_prefix = [n_devices, batch_size_per_device]
    return math.numpy.reshape(x, new_shape_prefix + x_shape[1:])
  return math.nested_map(f, x)


def for_n_devices(x, n_devices):
  """Replicates/broadcasts `x` for n_devices."""
  def f(x):
    if n_devices > 1 and math.backend_name() == 'jax':
      return _multi_device_put(x)
    elif n_devices > 1:
      return np.broadcast_to(x, (n_devices,) + x.shape)
    else:
      return x
  return math.nested_map(f, x)


def _multi_device_put(x, devices=None):
  """Memory efficient multi-device replication / broadcast in JAX.

  JAX uses a ShardedDeviceArray class that holds a list of device buffers
  on separate devices for use with pmap'd computations.  Sharded arrays
  are explicitly used to eliminate unnecessary inter-device transfer of
  memory buffers between use in pmap'd computations.  The JAX API currently
  does not have a multi-device 'put' function that copies a buffer onto
  N devices in a memory-efficient fashion, so we implement our own here.

  Args:
    x: jax DeviceArray or numpy ndarray to be replicated.
    devices: a jax.devices() list or subset thereof of devices to
      replicate onto.  Should match the list passed to any pmaps
      ingesting the replicated array.

  Returns:
    A ShardedDeviceArray with
    dtype = x.dtype and shape = (n_devices,) + x.shape
    that's backed by replicated device_buffers on each local device.
  """
  # Convert _FilledConstants that don't have device_buffer, etc.
  if type(x) != jax.xla.DeviceArray:  # pylint: disable=unidiomatic-typecheck
    x = np.array(x)
  # Calculate the abstract shape of the replicated array.
  if not devices:
    devices = jax.local_devices()
  n_devices = len(devices)
  x_aval = jax.xla.abstractify(x)
  broadcast_x_aval = jax.abstract_arrays.ShapedArray(
      (n_devices,) + x_aval.shape,
      x_aval.dtype)
  # Create copies of the underlying device buffer for each local device.
  broadcast_buffers = [
      jax.interpreters.xla.device_put(x, dv)
      for dv in devices
  ]
  return jax.pxla.ShardedDeviceArray(broadcast_x_aval, broadcast_buffers)
