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
"""Trax decorators and layers for asserts on tensor shapes."""

import functools
import inspect
import string
from absl import logging
from trax.layers import base
from trax.layers import combinators


def assert_shape(specification):
  """Decorator for checking the input and output shapes of Layer.

  Decorator can be applied on trax.base.Layer class, or a function returning
  a trax.base.Layer class. It uses notation similar to einsum (Einstein
  summation convention), achieving concise and simple representation of tensors.
  For example 'ij,jh->ih' is a valid representation of a function taking two
  2D matrices as input, and returning a single output, also a 2D matrix.

  It improves readability and puts puts three levels of asserts on the function:
  first level is the number of input tensors and output tensors; second level is
  the rank of each tensor; third level is the size of each dimension of each
  tensor. The decorator inserts those asserts right before and right after
  'forward' call.

  First level, assert on number of inputs and outputs. In the representation
  input tensors are separated from output tensors by an arrow '->'. For layers
  taking multiple input tensors or returning multiple output tensors, those
  tensors will be separated by a comma ','.
  For example, specification 'bsd,df->bsf' asserts that there will be two
  input tensors, with shapes represented by 'bsd' and 'df' respectively; and
  a single output tensor with shape represented by 'bsf'.

  Second level, asserts on possible rank of each tensor. Most commonly,
  each letter represents a single dimension. For example,the tensor with shapes
  represented by 'bsd' has rank three; with 'df' it has rank two. The special
  case is an ellipsis ('...'), which expand to arbitrary number of dimensions,
  including zero. For example, the tensor with specification '...sf' has at
  least two dimensions. Each tensor may have in its representation one ellipsis.

  Third level, asserts the size of each dimension. If two dimensions in any
  of input or output tensors have the same letter in the representation then
  they must have the same size. For example, with a tensor A represented by 'df'
  and a tensor B represented by 'bsf', the size of the second dimension of A
  must equal the size of the third dimension of B. Another example: with a
  tensor C represented by '...dv' and a tensor D represented by 'd', the size of
  the first and only dimension of D must be equal to the size of the second to
  last dimension of tensor C.

  If two distinct tensors have an ellipsis in their representation then all of
  dimensions covered by those ellipses must match. For example, with a tensor E
  represented by '...d' and tensor F represented by '...x' then E and F must
  have the same rank, and the sizes of all but the last dimensions must match.

  Examples:
  # In Dense layer there is a single input and single output; the last dimension
  # may change in size, while the sizes of all previous dimensions, marked by
  # an ellipsis, will stay the same.
  @assert_shape('...a->...b')
  class Dense(base.Layer):
    (...)

  # DotProductCausalAttention takes three tensors as input: Queries, Keys, and
  # Values, and outputs a single tensor. Sizes of the first two dimensions in
  # all those tensors must match, while the last dimension must match only
  # between Queries and Keys, and separately between Values and output tensor.
  @assert_shape('blk,blk,bld->bld')
  class DotProductCausalAttention(base.Layer):
    (...)

  # assert_shape can also be placed before the function returning base.Layer.
  @assert_shape('...d->...')
  def ReduceSum():
    return Fn('ReduceSum', lambda x: jnp.sum(x, axis=-1, keepdims=False))

  Args:
    specification: A text specification for the input/output tensors.

  Returns:
    The decorator changing the class or function.
  """
  caller = inspect.getframeinfo(inspect.stack()[1][0])
  message = f'Defined at {caller.filename}:{caller.lineno}'

  def wrap_cls(cls):
    forward = getattr(cls, 'forward')
    init = getattr(cls, '__init__')

    before_spec, after_spec = specification.split('->')

    @functools.wraps(init)
    def init_wrapper(self, *args, **kwargs):
      before_assert = AssertShape(before_spec,
                                  message=message + ' function input')
      after_assert = AssertShape(after_spec,
                                 message=message + ' function output')
      after_assert._create_link(before_assert)  # pylint: disable=protected-access
      out = init(self, *args, **kwargs)
      self._before_assert_fun = before_assert  # pylint: disable=protected-access
      self._after_assert_fun = after_assert  # pylint: disable=protected-access
      return out

    @functools.wraps(forward)
    def forward_wrapper(self, x, *args, **kwargs):
      x = self._before_assert_fun.forward(x)  # pylint: disable=protected-access
      y = forward(self, x, *args, **kwargs)
      y = self._after_assert_fun.forward(y)  # pylint: disable=protected-access
      return y

    setattr(cls, 'forward', forward_wrapper)
    setattr(cls, '__init__', init_wrapper)
    return cls

  # TODO(jaszczur): replace this with forward/init override.
  def wrap_fun(fun):
    @functools.wraps(fun)
    def fun_wrapper(*args, **kwargs):
      layer = fun(*args, **kwargs)
      return AssertFunction(specification, layer, message)
    return fun_wrapper

  def wrap_fun_or_cls(fun_or_cls):
    return (wrap_cls(fun_or_cls) if inspect.isclass(fun_or_cls) else
            wrap_fun(fun_or_cls))

  return wrap_fun_or_cls


def AssertFunction(specification, layer, message=None):  # pylint: disable=invalid-name
  """AssertFunction asserts shapes on the input/output tensors of a layer.

  It passes all inputs to the layer, and returns all outputs of the layer
  unchanged.

  Args:
    specification: A specification. See assert_shape decorator for a full
        documentation.
    layer: A base.Layer to wrap around.
    message: An optional message to print if an assert fails. By default it will
        print the filename and the line number where AssertFunction was called.

  Returns:
    The given layer wrapped in asserts on its inputs and outputs.
  """
  if message is None:
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    message = f'Defined at {caller.filename}:{caller.lineno}'
  before_spec, after_spec = specification.split('->')
  before_assert = AssertShape(before_spec, message=message + ' function input')
  after_assert = AssertShape(after_spec, message=message + ' function output')
  after_assert._create_link(before_assert)  # pylint: disable=protected-access
  return combinators.Serial(
      before_assert, layer, after_assert)


class AssertShape(base.Layer):
  """Layer which put asserts on shapes of tensors, and returns them unchanged.

  It borrows the notation from assert_shape decorator, except it doesn't have
  the arrow '->' special character, as the input tensors are the same as output.
  """

  def __init__(self, spec, message=None, visible_layer=False):
    """Creates AssertShape layer.

    Args:
      spec: Specification for input tensors. See assert_shape decorator for the
          full documentation.
      message: An optional message to include when assert fails. By default it
          includes the filename and line number where this function was called.
      visible_layer: If true, print this layer inside the model (default: False)
    """
    name = 'AssertShape' if visible_layer else ''
    super().__init__(name=name)
    spec = spec.replace('...', '.')
    for letter in spec:
      assert letter in string.ascii_letters + string.digits + '.' + ','
    self._specs = spec.split(',')
    self._n_in = self._n_out = len(self._specs)

    self._defined_shapes = {str(i): i for i in range(10)}
    self._linked = False

    if message is None:
      caller = inspect.getframeinfo(inspect.stack()[1][0])
      self._message = f'Defined at {caller.filename}:{caller.lineno}'
    else:
      self._message = message

  def forward(self, xs):
    if not self._linked:
      for k in list(self._defined_shapes.keys()):
        if not k.isdigit():
          del self._defined_shapes[k]

    if not isinstance(xs, (list, tuple)):
      xs = [xs]

    # Try-except below checks if something is wrong with shapes. It can happen
    # e.g. when using trax2keras. If this is the case we cannot check if shapes
    # are correct or not
    try:
      for x in xs:
        for i in range(len(x.shape)):
          if x.shape[i] != x.shape[i]:
            raise TypeError()
    except TypeError:
      message = ('AssertShape cannot check shapes. This often happens when'
                 ' using trax2keras. Shape asserts are skipped.')
      print(message)
      logging.warning(message)
      if len(xs) == 1:
        return xs[0]
      else:
        return xs

    # helper functions
    def assert_true(cond):
      if not cond:
        shapes = [x.shape for x in xs]
        defined_shapes_dict_without_digits = {
            k: v for k, v in self._defined_shapes.items() if not k.isdigit()}
        raise ValueError(
            f'AssertShape Error. Expected {self._specs}, got {shapes} with dict'
            f' {defined_shapes_dict_without_digits}. {self._message}')

    def assert_equal(a, b):
      assert_true(a == b)
      return a

    def check_shape(shape, spec):
      assert_equal(len(shape), len(spec))
      for shape_dim, letter in zip(shape, spec):
        if letter in self._defined_shapes:
          self._defined_shapes[letter] = assert_equal(
              self._defined_shapes[letter], shape_dim)
        else:
          self._defined_shapes[letter] = shape_dim

    def check_ellipsys(shape):
      if '.' not in self._defined_shapes:
        self._defined_shapes['.'] = shape
      else:
        assert_equal(len(shape), len(self._defined_shapes['.']))
        for s1, s2 in zip(shape, self._defined_shapes['.']):
          assert_equal(s1, s2)

    # actual asserts
    assert_equal(len(xs), len(self._specs))

    for x, spec in zip(xs, self._specs):
      if '.' in spec:
        assert_true(len(x.shape) >= (len(spec) - 1))

        before, after = spec.split('.')
        check_shape(x.shape[:len(before)], before)
        if after:
          check_shape(x.shape[-len(after):], after)
          check_ellipsys(x.shape[len(before):-len(after)])
        else:
          # if len(after) == 0 then -len(after) in indices evaluates badly.
          check_ellipsys(x.shape[len(before):])
      else:
        check_shape(x.shape, spec)

    if len(xs) == 1:
      return xs[0]
    else:
      return xs

  def _create_link(self, other):
    """Internal. Used to create a shared dictionary."""
    # This works well for assert_shape and AssertFunction; but it can break
    # easily if the order of calls to forward() is not known in advance.
    self._linked = True
    self._defined_shapes = other._defined_shapes  # pylint: disable=protected-access
