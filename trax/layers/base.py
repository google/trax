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
"""The key layer abstraction (Layer class) and supporting machinery."""

import copy
import functools
import gzip
import inspect
import pickle
import random
import traceback

import jax
import numpy as np
import tensorflow as tf

from trax import fastmath
from trax.fastmath import nested_map
from trax.fastmath import numpy as jnp
from trax.shapes import ShapeDtype
from trax.shapes import signature


# TODO(lukaszkaiser): should we use special objects for these for clarity?
EMPTY_WEIGHTS = ()    # Used for layers that have no trainable weights.
EMPTY_STATE = ()      # Used for layers that have no non-trainable state.
GET_WEIGHTS_FROM_CACHE = {'__marker_for_cached_weights_': ()}
GET_STATE_FROM_CACHE = {'__marker_for_cached_state_': ()}
N_WEIGHTS_SHARDS = 1  # TODO(lukaszkaiser): make weight-sharding non-global


class Layer:
  """Base class for composable layers in a deep learning network.

  Layers are the basic building blocks for deep learning models. A layer
  computes a function from zero or more inputs to zero or more outputs,
  optionally using trainable weights (common) and non-parameter state (not
  common).

  Layer subclasses typically override at most two methods of the base `Layer`
  class:

    `forward(inputs)`:
      Computes the layer's output as part of a forward pass through the model.

    `init_weights_and_state(self, input_signature)`:
      Initializes the layer's weights and state to handle input with the given
      signature (number, shapes and dtypes of input arguments).

  A small number of layer types are combinators -- they organize the computation
  of their sublayers, e.g., applying their sublayers in series or in parallel.

  All layers have the following properties, with default values implemented
  in the base `Layer` class:

    - `n_in`: int (default 1)
    - `n_out`: int (default 1)
    - `weights`: tuple (default empty -- the layer has no weights)
    - `state`: tuple (default empty -- the layer has no non-parameter state)
    - `sublayers`: tuple (default empty -- the layer has no sublayers)

  The inputs to a layer are tensors, packaged according to how many there are:

    - `n_in = 0`: an empty tuple
    - `n_in = 1`: one tensor (NOT wrapped in a tuple)
    - `n_in > 1`: a tuple of tensors

  (The special treatment of the single-input case is meant to simplify the
  work of layer writers; this design choice may be revisited in the future.)

  The outputs from a layer are also tensors, packaged the same as layer inputs:

    - `n_out = 0`: an empty tuple
    - `n_out = 1`: the tensor (NOT wrapped in a tuple)
    - `n_out > 1`: a tuple of tensors

  The Trax runtime maintains a data stack with which layer calls are composed.
  For more complex data network architectures, possibly involving multiple data
  flows, one can view each layer as a function from stack state to stack state,
  where the function's inputs are a slice from the stack, and the function's
  outputs are spliced back into the stack.
  """

  def __init__(self, n_in=1, n_out=1, name=None, sublayers_to_print=None):
    """Creates a partially initialized, unconnected layer instance.

    Args:
      n_in: Number of inputs expected by this layer.
      n_out: Number of outputs promised by this layer.
      name: Class-like name for this layer; for use when printing this layer.
      sublayers_to_print: Sublayers to display when printing out this layer;
        if None (the default), display all sublayers.
    """
    self._n_in = n_in
    self._n_out = n_out
    self._name = self.__class__.__name__ if name is None else name
    self._sublayers_to_print = sublayers_to_print
    self._sublayers = ()  # Default is no sublayers.

    # The actual rng value/shape depends on the backend, which may not yet be
    # initialized at the point this method is run. Hence, at first initialize
    # only a seed random integer, in a backend-neutral way.
    self._rng = None
    self._rng_seed_int = random.randint(0, 2**31 - 1)

    # The private fields _weights and _state store the private part of
    # layer weights and state. When a layer has no sublayers, these are
    # the same as layer.weights and layer.state. For layers with sublayers
    # (i.e., combinators), these just mark which weights are cached -- see
    # the getter and setter for weights and state for details.
    # There is no need to use these fields in most user-implemented classes.
    self._weights = EMPTY_WEIGHTS  # By default no trainable weights.
    self._state = EMPTY_STATE  # By default no non-trainable state.

    # Record layer creation site for use in LayerError messages.
    # The frame can mutate, so copy relevant values out of it.
    frame = _find_frame(inspect.currentframe())
    self._caller = {'filename': copy.copy(frame.f_code.co_filename),
                    'lineno': int(frame.f_lineno)}
    del frame  # Just in case.

    self._init_cached = False
    self._jit_cache = {}

  def __repr__(self):
    """Renders this layer as a medium-detailed string, to help in debugging.

    Subclasses should aim for high-signal/low-noise when overriding this
    method.

    Returns:
      A high signal-to-noise string representing this layer.
    """
    def indent_string(x):
      return '  ' + x.replace('\n', '\n  ')

    name_str = self._name
    n_in, n_out = self.n_in, self.n_out
    if n_in != 1: name_str += f'_in{n_in}'
    if n_out != 1: name_str += f'_out{n_out}'

    if self._sublayers_to_print is not None:
      substructure = self._sublayers_to_print
    else:
      substructure = self.sublayers
    if substructure:
      substructure_strs = [str(x) for x in substructure if str(x)]
      substructure_str = '\n'.join(indent_string(s) for s in substructure_strs)
      return f'{name_str}[\n{substructure_str}\n]'
    else:
      return name_str

  def __call__(self, x, weights=None, state=None, rng=None):
    """Makes layers callable; for use in tests or interactive settings.

    This convenience method helps library users play with, test, or otherwise
    probe the behavior of layers outside of a full training environment. It
    presents the layer as callable function from inputs to outputs, with the
    option of manually specifying weights and non-parameter state per individual
    call. For convenience, weights and non-parameter state are cached per layer
    instance, starting from default values of `EMPTY_WEIGHTS` and `EMPTY_STATE`,
    and acquiring non-empty values either by initialization or from values
    explicitly provided via the weights and state keyword arguments, in which
    case the old weights will be preserved, and the state will be updated.

    Args:
      x: Zero or more input tensors, packaged as described in the `Layer` class
          docstring.
      weights: Weights or `None`; if `None`, use self's cached weights value.
      state: State or `None`; if `None`, use self's cached state value.
      rng: Single-use random number generator (JAX PRNG key), or `None`;
          if `None`, use a default computed from an integer 0 seed.

    Returns:
      Zero or more output tensors, packaged as described in the `Layer` class
      docstring.
    """
    weights = self.weights if weights is None else weights
    rng = self.rng if rng is None else rng
    if state is not None:
      self.state = state  # Needed if the model wasn't fully initialized.
    state = self.state
    outputs, new_state = self.pure_fn(x, weights, state, rng)
    self.state = new_state
    return outputs

  def forward(self, inputs):
    """Computes this layer's output as part of a forward pass through the model.

    A layer subclass overrides this method to define how the layer computes
    outputs from inputs. If the layer depends on weights, state, or randomness
    as part of the computation, the needed information can be accessed as
    properties of the layer object: `self.weights`, `self.state`, and
    `self.rng`. (See numerous examples in `trax.layers.core`.)

    Args:
      inputs: Zero or more input tensors, packaged as described in the `Layer`
          class docstring.

    Returns:
      Zero or more output tensors, packaged as described in the `Layer` class
      docstring.
    """
    raise NotImplementedError

  def init_weights_and_state(self, input_signature):
    """Initializes weights and state, to handle input with the given signature.

    A layer subclass must override this method if the layer uses weights or
    state. To initialize weights, set `self.weights` to desired (typically
    random) values. To initialize state (uncommon), set `self.state` to desired
    starting values.

    Args:
      input_signature: A `ShapeDtype` instance (if this layer takes one input)
          or a list/tuple of `ShapeDtype` instances.
    """
    del input_signature

  @property
  def has_backward(self):
    """Returns `True` if this layer provides its own custom backward pass code.

    A layer subclass that provides custom backward pass code (for custom
    gradients) must override this method to return `True`.
    """
    return False

  def backward(self, inputs, output, grad, weights, state, new_state, rng):
    """Custom backward pass to propagate gradients in a custom way.

    Args:
      inputs: Input tensors; can be a (possibly nested) tuple.
      output: The result of running this layer on inputs.
      grad: Gradient signal computed based on subsequent layers; its structure
          and shape must match output.
      weights: This layer's weights.
      state: This layer's state prior to the current forward pass.
      new_state: This layer's state after the current forward pass.
      rng: Single-use random number generator (JAX PRNG key).

    Returns:
      The custom gradient signal for the input. Note that we need to return
      a gradient for each argument of forward, so it will usually be a tuple
      of signals: the gradient for inputs and weights.
    """
    raise NotImplementedError

  # End of public subclassing interface.
  # Begin public callable interface.

  def init(self, input_signature, rng=None, use_cache=False):
    """Initializes weights/state of this layer and its sublayers recursively.

    Initialization creates layer weights and state, for layers that use them.
    It derives the necessary array shapes and data types from the layer's input
    signature, which is itself just shape and data type information.

    For layers without weights or state, this method safely does nothing.

    This method is designed to create weights/state only once for each layer
    instance, even if the same layer instance occurs in multiple places in the
    network. This enables weight sharing to be implemented as layer sharing.

    Args:
      input_signature: `ShapeDtype` instance (if this layer takes one input)
          or list/tuple of `ShapeDtype` instances.
      rng: Single-use random number generator (JAX PRNG key), or `None`;
          if `None`, use a default computed from an integer 0 seed.
      use_cache: If `True`, and if this layer instance has already been
          initialized elsewhere in the network, then return special marker
          values -- tuple `(GET_WEIGHTS_FROM_CACHE, GET_STATE_FROM_CACHE)`.
          Else return this layer's newly initialized weights and state.

    Returns:
      A `(weights, state)` tuple.
    """
    try:
      if self._init_cached and use_cache:
        return (GET_WEIGHTS_FROM_CACHE, GET_STATE_FROM_CACHE)

      if rng is not None:
        self.rng = rng
      self.init_weights_and_state(input_signature)

      if use_cache:
        self._init_cached = True
      else:
        self._clear_init_cache()

      return (self.weights, self.state)

    except Exception:
      # Skipping 3 lines as it's always the uninteresting internal call.
      name, trace = self._name, _short_traceback(skip=3)
      raise LayerError(name, 'init', self._caller,
                       input_signature, trace) from None

  def init_from_file(self, file_name, weights_only=False, input_signature=None):
    """Initializes this layer and its sublayers from a pickled checkpoint.

    In the common case (`weights_only=False`), the file must be a gziped pickled
    dictionary containing items with keys `'flat_weights', `'flat_state'` and
    `'input_signature'`, which are used to initialize this layer.
    If `input_signature` is specified, it's used instead of the one in the file.
    If `weights_only` is `True`, the dictionary does not need to have the
    `'flat_state'` item and the state it not restored either.

    Args:
      file_name: Name/path of the pickled weights/state file.
      weights_only: If `True`, initialize only the layer's weights. Else
          initialize both weights and state.
      input_signature: Input signature to be used instead of the one from file.

    Returns:
      A `(weights, state)` tuple.
    """
    with tf.io.gfile.GFile(file_name, 'rb') as f:
      with gzip.GzipFile(fileobj=f, compresslevel=2) as gzipf:
        dictionary = pickle.load(gzipf)
    # In the current checkpoint format, we store weights in a separate
    # non-pickled file with the same name but added ".npy".
    if isinstance(dictionary['flat_weights'], int):
      if file_name.endswith('.pkl.gz'):
        weights_path = file_name[:-6] + 'weights.npy.gz'
      else:
        weights_path = file_name + '.npy'
      if not tf.io.gfile.exists(weights_path):  # old format compatibility
        weights_path = file_name + '.npy'
      dictionary['flat_weights'] = np_from_file(
          weights_path, compresslevel=dictionary['flat_weights'])
    if input_signature is None:
      input_signature = dictionary['input_signature']
    if weights_only and input_signature is not None:
      self.init(input_signature)
    weights_and_state_sig = self.weights_and_state_signature(input_signature)
    weights, state = unflatten_weights_and_state(
        dictionary['flat_weights'], dictionary['flat_state'],
        weights_and_state_sig, weights_only=weights_only)
    if not weights_only:
      self.state = state
    self.weights = weights
    return (self.weights, self.state)

  def save_to_file(self, file_name, weights_only=False, input_signature=None):
    """Saves this layer and its sublayers to a pickled checkpoint.

    Args:
      file_name: Name/path of the pickled weights/state file.
      weights_only: If `True`, save only the layer's weights. Else
          save both weights and state.
      input_signature: Input signature to be used.
    """
    flat_weights, flat_state = flatten_weights_and_state(
        self.weights, self.state)
    dictionary = {
        'flat_weights': flat_weights,
    }
    if not weights_only:
      dictionary['flat_state'] = flat_state
    if input_signature is not None:
      dictionary['input_signature'] = input_signature

    tmp_file_path = file_name + '._tmp_'
    with tf.io.gfile.GFile(tmp_file_path, 'wb') as f:
      with gzip.GzipFile(fileobj=f, compresslevel=2) as gzipf:
        pickle.dump(dictionary, gzipf, protocol=pickle.HIGHEST_PROTOCOL)
    # Moving a file is much less error-prone than pickling large files.
    tf.io.gfile.rename(tmp_file_path, file_name, overwrite=True)

  # End of public callable methods.
  # Methods and properties below are reserved for internal use.

  @property
  def name(self):
    """Returns the name of this layer."""
    return self._name

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
  def weights(self):
    """Returns this layer's weights.

    Depending on the layer, the weights can be in the form of:

      - an empty tuple
      - a tensor (ndarray)
      - a nested structure of tuples and tensors

    If the layer has sublayers, the weights by convention will be
    a tuple of length `len(sublayers)` containing the weights of sublayers.
    Note that in this case self._weights only marks which ones are shared.
    """
    if not self.sublayers:
      return self._weights
    else:
      return tuple(layer.weights if w is None else w
                   for (layer, w) in zip(self.sublayers, self._weights))

  @weights.setter
  def weights(self, weights):
    """Sets the weights of this layer and its sublayers.

    Args:
      weights: the weights to set; if layer has sublayers, weights should be
        either a list or a tuple of the same length as `len(self.sublayers)`
        and it will be used to set the weights of all sublayers.
    """
    if isinstance(weights, dict) and weights == GET_WEIGHTS_FROM_CACHE:
      return
    if not self.sublayers:
      self._weights = weights
    else:
      # When having sublayers, self._weights just marks which are cached,
      # the actual weights are stored by sublayers.
      self._weights = []
      for w in weights:
        if isinstance(w, dict) and w == GET_WEIGHTS_FROM_CACHE:
          self._weights.append(w)
        else:
          self._weights.append(None)
      # Set sublayer weights.
      n_layers = len(self.sublayers)
      if len(weights) != n_layers:
        raise ValueError(
            f'Number of weight elements ({len(weights)}) does not equal the '
            f'number of sublayers ({n_layers}) in: {str(self)}.')
      for sublayer, sublayer_weights in zip(self.sublayers, weights):
        sublayer.weights = sublayer_weights

  @property
  def state(self):
    """Returns a tuple containing this layer's state; may be empty.

    If the layer has sublayers, the state by convention will be
    a tuple of length `len(sublayers)` containing sublayer states.
    Note that in this case self._state only marks which ones are shared.
    """
    if not self.sublayers:
      return self._state
    else:
      return tuple(layer.state if s is None else s
                   for (layer, s) in zip(self.sublayers, self._state))

  @state.setter
  def state(self, state):
    """Sets the state of this layer and its sublayers.

    Args:
      state: the state to set; if layer has sublayers, state should be
        either a list or a tuple of the same length as `len(self.sublayers)`
        and it will be used to set the state of all sublayers.
    """
    if isinstance(state, dict) and state == GET_STATE_FROM_CACHE:
      return
    if not self._sublayers:
      self._state = state
    else:
      # When having sublayers, self._state just marks which are cached,
      # the actual weights are stored by sublayers.
      self._state = []
      for s in state:
        if isinstance(s, dict) and s == GET_STATE_FROM_CACHE:
          self._state.append(s)
        else:
          self._state.append(None)
      # Set sublayer states.
      n_layers = len(self.sublayers)
      if len(state) != n_layers:
        raise ValueError(
            f'Number of state elements ({len(state)}) does not equal the '
            f'number of sublayers ({n_layers}) in: {str(self)}.')
      for sublayer, sublayer_state in zip(self.sublayers, state):
        sublayer.state = sublayer_state

  def weights_and_state_signature(self, input_signature, unsafe=False):
    """Return a pair containing the signatures of weights and state."""
    rng, state, weights = self.rng, self.state, self.weights
    abstract_init = fastmath.abstract_eval(self.init)
    sig = abstract_init(input_signature)
    self.rng = rng
    if not unsafe:
      self.state, self.weights = state, weights
    return sig

  @property
  def rng(self):
    """Returns this layer's current single-use random number generator.

    Code that wants to base random samples on this generator must explicitly
    split off new generators from it. (See, for example, the `rng` setter code
    below.)
    """
    if self._rng is None:
      # One-time initialization from backend-neutral seed int.
      self._rng = fastmath.random.get_prng(self._rng_seed_int)
    return self._rng

  @rng.setter
  def rng(self, rng):
    """Sets the rng (JAX PRNG key) for this layer and sublayers, recursively."""
    self._rng = rng
    sublayers = self.sublayers
    if sublayers:
      rngs = fastmath.random.split(rng, len(sublayers))
      for sublayer, rng in zip(sublayers, rngs):
        sublayer.rng = rng

  def _clear_init_cache(self):
    self._init_cached = False
    for sublayer in self.sublayers:
      sublayer._clear_init_cache()  # pylint: disable=protected-access

  def pure_fn(self, x, weights, state, rng, use_cache=False):
    """Applies this layer as a pure function with no optional args.

    This method exposes the layer's computation as a pure function. This is
    especially useful for JIT compilation. Do not override, use `forward`
    instead.

    Args:
      x: Zero or more input tensors, packaged as described in the `Layer` class
          docstring.
      weights: A tuple or list of trainable weights, with one element for this
          layer if this layer has no sublayers, or one for each sublayer if
          this layer has sublayers. If a layer (or sublayer) has no trainable
          weights, the corresponding weights element is an empty tuple.
      state: Layer-specific non-parameter state that can update between batches.
      rng: Single-use random number generator (JAX PRNG key).
      use_cache: if `True`, cache weights and state in the layer object; used
        to implement layer sharing in combinators.

    Returns:
      A tuple of `(tensors, state)`. The tensors match the number (`n_out`)
      promised by this layer, and are packaged as described in the `Layer`
      class docstring.
    """
    try:
      old_weights, old_state, old_rng = self.weights, self.state, self.rng
      self._rng = rng
      # The isinstance check is only needed when == is overloaded, as in TF.
      if (isinstance(weights, dict) and isinstance(state, dict) and
          weights == GET_WEIGHTS_FROM_CACHE and state == GET_STATE_FROM_CACHE):
        was_cached = True
        weights = self.weights
        state = self.state
      else:
        # In this case, we're called for the first time: cache weights.
        was_cached = False
        self.weights, self.state = weights, state

      # If weights are sharded across multiple devices, unshard before forward.
      sharded_weights, weights_were_unsharded = weights, False
      if N_WEIGHTS_SHARDS > 1 and not self.sublayers:
        self.weights, weights_were_unsharded = unshard_in_pmap(
            weights, N_WEIGHTS_SHARDS)

      if not self.has_backward:
        outputs = self.forward(x)
        s = self.state
      else:
        outputs, s = self._do_custom_gradients(x)
        self.state = s
      self._rng = old_rng
      if weights_were_unsharded:  # only store a shard of weights if sharded
        self.weights = sharded_weights

      if not use_cache:
        self.weights, self.state = old_weights, old_state
      if was_cached:  # If the layer was shared, return a state marking this.
        s = GET_STATE_FROM_CACHE
      return outputs, s

    except Exception:
      # Skipping 3 lines as it's always the uninteresting internal call.
      name, trace = self._name, _short_traceback(skip=3)
      raise LayerError(name, 'pure_fn',
                       self._caller, signature(x), trace) from None

  def output_signature(self, input_signature):
    """Returns output signature this layer would give for `input_signature`."""
    return self._forward_abstract(input_signature)[0]  # output only, not state

  def _forward_abstract(self, input_signature):
    """Computes shapes and dtypes this layer would produce in a forward pass.

    Args:
      input_signature: `ShapeDtype` instance (if this layer takes one input)
          or list/tuple of `ShapeDtype` instances.

    Returns:
      Tuple of (output, state).

      The output part of the tuple is a `ShapeDtype` instance representing the
      shape and type of the output (if this layer has one output) or a tuple
      of `ShapeDtype` instances (if this layer has more than one output).
    """
    try:
      # Note: By using rng_signature in place of an rng, we avoid computing and
      # permanently storing in global memory a large number of dropout masks.
      # TODO(jonni): Check if using an rng still carries this cost.
      dummy_rng = fastmath.random.get_prng(0)
      rng_signature = ShapeDtype(dummy_rng.shape, dummy_rng.dtype)
      weights_signature = nested_map(signature, self.weights)
      state_signature = nested_map(signature, self.state)
      forward_infer_shapes = fastmath.abstract_eval(self.pure_fn)
      return forward_infer_shapes(
          input_signature, weights_signature, state_signature, rng_signature)
    except Exception:
      # TODO(lukaszkaiser): the choice of 7 is a heuristic, can we automate it?
      # Skipping 7 lines which are all JAX abstract'ifying wrappers.
      name, trace = self._name, _short_traceback(skip=7)
      raise LayerError(name, '_forward_abstract', self._caller, input_signature,
                       trace) from None

  # pylint: disable=protected-access
  def _do_custom_gradients(self, x):
    """Calls this layer for a forward pass, but with custom gradients."""

    def _f(state, rng, y, weights):
      old_weights, old_state, old_rng = self.weights, self.state, self._rng
      self.weights, self.state, self._rng = weights, state, rng
      res = self.forward(y)
      s = self.state
      self.weights, self.state, self._rng = old_weights, old_state, old_rng
      return res, s

    def _f_fwd(state, rng, y, weights):
      old_weights, old_state, old_rng = self.weights, self.state, self._rng
      self.weights, self.state, self._rng = weights, state, rng
      res = self.forward(y)
      s = self.state
      self.weights, self.state, self._rng = old_weights, old_state, old_rng
      return (res, s), (state, rng, y, res, weights, s)

    def _f_bwd(residual, grad):
      """Custom gradient function."""
      state, rng, y, output, weights, new_state = residual
      grad = grad[0]  # Ignore dummy gradient wrt state.
      out = self.backward(y, output, grad, weights, state, new_state, rng)
      return (None, None, *out)

    do_forward = fastmath.custom_vjp(_f, _f_fwd, _f_bwd, nondiff_argnums=(0, 1))

    output, state = do_forward(self.state, self._rng, x, self.weights)
    return output, state

  def _settable_attrs(self):
    """We only allow to set these attributes in Trax layers to prevent typos."""
    return ('weights', 'state', 'rng')

  def __setattr__(self, attr, value):
    """Sets class attributes and protects from typos.

    In Trax layers, we only allow to set the following public attributes::

      - weights
      - state
      - rng

    This function prevents from setting other public attributes to avoid typos,
    for example, this is not possible and would be without this function::

      [typo]   layer.weighs = some_tensor

    If you need to set other public attributes in a derived class (which we
    do not recommend as in almost all cases it suffices to use a private
    attribute), override self._settable_attrs to include the attribute name.

    Args:
      attr: Name of the attribute to be set.
      value: Value to be assigned to the attribute.
    """
    if attr[0] != '_' and attr not in self._settable_attrs():
      raise ValueError(
          f'Trax layers only allow to set {self._settable_attrs()} as public '
          f'attribues, not {attr}.')
    else:
      super().__setattr__(attr, value)


class PureLayer(Layer):
  """Pure function from inputs to outputs, packaged as neural network layer.

  The `PureLayer` class represents the simplest kinds of layers: layers with
  no trainable weights and no randomness, hence pure functions from inputs to
  outputs.
  """

  def __init__(self, forward_fn, n_in=1, n_out=1, name='PureLayer'):
    """Creates an unconnected `PureLayer` instance.

    Args:
      forward_fn: Pure function from input tensors to output tensors, where
          inputs and outputs are packaged as specified for `forward`.
      n_in: Number of inputs expected by this layer.
      n_out: Number of outputs promised by this layer.
      name: Class-like name for this layer; for use only in debugging.
    """
    super().__init__(n_in, n_out, name)
    self._forward_fn = forward_fn

  def forward(self, inputs):
    """Overrides `Layer.forward`.

    Args:
      inputs: Zero or more input tensors, packaged as described in the `Layer`
          class docstring.

    Returns:
      Zero or more output tensors, packaged as described in the `Layer` class
      docstring.
    """
    _validate_forward_input(inputs, self.n_in)
    raw_output = self._forward_fn(inputs)
    output = () if _is_empty(raw_output) else raw_output
    return output


def Fn(name, f, n_out=1):  # pylint: disable=invalid-name
  """Returns a layer with no weights that applies the function `f`.

  `f` can take and return any number of arguments, and takes only positional
  arguments -- no default or keyword arguments. It often uses JAX-numpy (`jnp`).
  The following, for example, would create a layer that takes two inputs and
  returns two outputs -- element-wise sums and maxima:

      `Fn('SumAndMax', lambda x0, x1: (x0 + x1, jnp.maximum(x0, x1)), n_out=2)`

  The layer's number of inputs (`n_in`) is automatically set to number of
  positional arguments in `f`, but you must explicitly set the number of
  outputs (`n_out`) whenever it's not the default value 1.

  Args:
    name: Class-like name for the resulting layer; for use in debugging.
    f: Pure function from input tensors to output tensors, where each input
        tensor is a separate positional arg, e.g., `f(x0, x1) --> x0 + x1`.
        Output tensors must be packaged as specified in the `Layer` class
        docstring.
    n_out: Number of outputs promised by the layer; default value 1.

  Returns:
    Layer executing the function `f`.
  """
  argspec = inspect.getfullargspec(f)
  if argspec.defaults is not None:
    raise ValueError('Function has default arguments (not allowed).')
  if argspec.varkw is not None:
    raise ValueError('Function has keyword arguments (not allowed).')
  if argspec.varargs is not None:
    raise ValueError('Function has variable args (not allowed).')

  def _forward(xs):  # pylint: disable=invalid-name
    if not isinstance(xs, (tuple, list)):
      xs = (xs,)
    return f(*xs)

  n_in = len(argspec.args)
  name = name or 'Fn'
  return PureLayer(_forward, n_in=n_in, n_out=n_out, name=name)


class LayerError(Exception):
  """Exception raised in the layer stack."""

  def __init__(self, layer_name, function_name, caller,
               input_signature, traceback_string):
    self._layer_name = layer_name
    self._function_name = function_name
    self._caller = caller  # Python inspect object with init caller info.
    self._traceback = traceback_string
    self._input_signature = input_signature
    super().__init__(self.message)

  @property
  def message(self):
    """Assembles current layer context into an error message."""
    prefix = 'Exception passing through layer '
    prefix += '%s (in %s):\n' % (self._layer_name, self._function_name)
    short_path = '[...]/' + '/'.join(
        self._caller['filename'].split('/')[-3:])
    caller = '  layer created in file %s, line %d\n' % (short_path,
                                                        self._caller['lineno'])
    shapes_str = '  layer input shapes: %s\n\n' % str(self._input_signature)
    return prefix + caller + shapes_str + self._traceback


def flatten_weights_and_state(weights, state):
  """Flatten weights and state into lists, excluding empty and cached ones."""
  def _is_empty_weight(x):
    return (x is EMPTY_WEIGHTS or
            (isinstance(x, dict) and x == GET_WEIGHTS_FROM_CACHE))
  flat_weights = [w for w in fastmath.tree_flatten(weights)
                  if not _is_empty_weight(w)]
  def _is_empty_state(x):
    return (x is EMPTY_STATE or
            (isinstance(x, dict) and x == GET_STATE_FROM_CACHE))
  flat_state = [s for s in fastmath.tree_flatten(state)
                if not _is_empty_state(s)]
  return flat_weights, flat_state


def unflatten_weights_and_state(
    flat_weights, flat_state, weights_and_state_signature, weights_only=False):
  """Unflatten weights and state given their signatures."""
  weights_tree, state_tree = weights_and_state_signature
  weights_to_copy = [EMPTY_WEIGHTS, GET_WEIGHTS_FROM_CACHE]
  weights, _ = fastmath.tree_unflatten(flat_weights, weights_tree,
                                       copy_from_tree=weights_to_copy)
  state = None
  if not weights_only:
    states_to_copy = [EMPTY_STATE, GET_STATE_FROM_CACHE]
    state, _ = fastmath.tree_unflatten(flat_state, state_tree,
                                       copy_from_tree=states_to_copy)
  return weights, state


def np_to_file(list_of_nparrays, file_path, compresslevel):
  """Save numpy arrays to file_path with gzipping and failure protection."""
  # Pickle to tmp file and overwrite to prevent writing partial files.
  tmp_file_path = file_path + '._tmp_'
  with tf.io.gfile.GFile(tmp_file_path, 'wb') as f:
    with gzip.GzipFile(fileobj=f, compresslevel=compresslevel) as gzipf:
      for x in list_of_nparrays:
        np.save(gzipf, x, allow_pickle=False)
  # Moving a file is much less error-prone than pickling large files.
  tf.io.gfile.rename(tmp_file_path, file_path, overwrite=True)


def np_from_file(file_path, compresslevel):
  """Load numpy arrays from file_path with gzipping."""
  if not tf.io.gfile.exists(file_path):
    raise FileNotFoundError(file_path)
  res = []
  with tf.io.gfile.GFile(file_path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, compresslevel=compresslevel) as gzipf:
      while True:
        try:
          res.append(np.load(gzipf, allow_pickle=False))
        except Exception:  # pylint: disable=broad-except
          break
  return res


def to_list(outputs):
  """Converts layer outputs to a nested list, for easier equality testing.

  Args:
    outputs: A tensor or tuple/list of tensors coming from the forward
        application of a layer. Each tensor is NumPy ndarray-like, which
        complicates simple equality testing (e.g., via `assertEquals`):
        such tensors require equality testing to use either `all` (all
        elements match) or `any` (at least one element matches), which is not
        directly supported in `absltest`.

  Returns:
    A nested list structure containing all the output values, but now directly
    testable using `assertEquals`.
  """
  if isinstance(outputs, (list, tuple)):
    return [y.tolist() for y in outputs]
  else:
    return outputs.tolist()


def _validate_forward_input(x, n_in):
  if n_in != 1:
    if not isinstance(x, (tuple, list)):
      raise TypeError(
          f'Expected input to be a tuple or list; instead got {type(x)}.')
    if len(x) != n_in:
      raise ValueError(f'Input tuple length ({len(x)}) does not equal required '
                       f'number of inputs ({n_in}).')


def _is_empty(container):
  if container is None:
    raise ValueError('Argument "container" is None.')
  return isinstance(container, (list, tuple)) and len(container) == 0  # pylint: disable=g-explicit-length-test


def _find_frame(frame):
  """Find the frame with the caller on the stack."""
  def _dirname_is_trax_layers_or_gin(frame):
    """Skip frames coming from trax/layers or .../gin."""
    try:
      dirname1 = frame.f_code.co_filename.split('/')[-3]
      dirname2 = frame.f_code.co_filename.split('/')[-2]
      return (dirname1 == 'trax' and dirname2 == 'layers') or dirname2 == 'gin'
    except IndexError:
      return False

  while _dirname_is_trax_layers_or_gin(frame):
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
    if np.issubdtype(dtype, np.integer):
      return fastmath.random.bernoulli(rng, 0.5, shape).astype(np.int32)
    else:
      return fastmath.random.uniform(rng, shape, minval=-1.0, maxval=1.0)
  elif isinstance(input_signature, (list, tuple)):
    return tuple(_random_values(x, rng) for x in input_signature)
  else:
    raise TypeError(type(input_signature))


def _shapes(x):
  """Gets a structure of shapes for a structure of nested arrays."""
  def shape(x):
    try:
      return tuple([int(i) for i in x.shape])
    except Exception:  # pylint: disable=broad-except
      return ()
  return tuple(nested_map(shape, x))


@functools.partial(fastmath.pmap, axis_name='batch')
def _axis_index(unused_x):
  """Return the axis indices."""
  return jax.lax.axis_index('batch')


def _axis_to_shard_heuristic(shape):
  """Chooses an axis to shard on - a simple heuristic to be revisited."""
  axis = 0 if len(shape) < 3 else -1
  return axis


def shard(tensors, n_shards=None):
  """Shard tensors across n_shards."""
  n_shards = N_WEIGHTS_SHARDS if n_shards is None else n_shards
  indices = _axis_index(np.zeros(fastmath.local_device_count()))
  def _shard_fn(x):
    axis = _axis_to_shard_heuristic(x.shape)
    if int(x.shape[axis]) % n_shards != 0:
      raise ValueError(f'Cannot split x with shape {x.shape} into {n_shards}.')
    split_x = jnp.split(x, n_shards, axis=axis)
    split_x = [split_x[i % n_shards] for i in indices]
    return np.stack(split_x, axis=0)
  return fastmath.nested_map(_shard_fn, tensors)


def unshard_in_pmap(tensors, n_shards):
  """Unshard tensors that were sharded into n_shards (call inside pmap)."""
  groups = [[n_shards * i + d for d in range(n_shards)]
            for i in range(fastmath.global_device_count() // n_shards)]
  def _unshard_fn(x):
    y = jax.lax.all_gather(x, 'batch', axis_index_groups=groups)
    split_y = jnp.split(y, n_shards, axis=0)
    split_y = [jnp.squeeze(sy, axis=0) for sy in split_y]
    axis = _axis_to_shard_heuristic(split_y[0].shape)
    return jnp.concatenate(split_y, axis=axis)
  try:
    jax.lax.axis_index('batch')  # will throw if not in pmap, e.g., on init
    res = fastmath.nested_map(_unshard_fn, tensors)
    return res, True
  except NameError:  # thrown from axis_index above
    return tensors, False


@functools.partial(fastmath.pmap, axis_name='batch')
def _all_gather(x, groups):
  return jax.lax.all_gather(x, 'batch', axis_index_groups=groups)


def unshard(tensors, n_shards=None):
  """Unshard tensors that were sharded into n_shards (outside of pmap)."""
  n_shards = N_WEIGHTS_SHARDS if n_shards is None else n_shards
  def _unshard_fn(x):
    # We use numpy here to put the large un-sharded arrays in CPU memory.
    # For unsharding on accelerators use ushard_in_pmap above and pmap it.
    split_y = np.split(np.asarray(x), n_shards, axis=0)
    split_y = [np.squeeze(sy, axis=0) for sy in split_y]
    axis = _axis_to_shard_heuristic(split_y[0].shape)
    return np.concatenate(split_y, axis=axis)
  return fastmath.nested_map(_unshard_fn, tensors)
