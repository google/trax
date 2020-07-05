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

"""Extensions such as `jit`, `grad`, `logsumexp`, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import contextlib
import threading
import numpy as np
import six

import tensorflow.compat.v2 as tf

import trax.tf_numpy.numpy as tf_np

_int_dtype_lower_bounds = [
    -2**63, -2**31, -2**15, -2**7, 0, 2**7, 2**15, 2**31, 2**64
]
_int_dtypes = [
    tf.int64, tf.int32, tf.int16, tf.int8, tf.uint8, tf.uint16, tf.uint32,
    tf.uint64
]


def most_precise_int_dtype(x):
  if not isinstance(x, six.integer_types) or isinstance(x, bool):
    return None
  i = bisect.bisect_right(_int_dtype_lower_bounds, x)
  if i in (0, len(_int_dtype_lower_bounds)):
    raise ValueError("Integer %s is out of bounds" % x)
  assert len(_int_dtype_lower_bounds) == len(_int_dtypes) + 1
  return _int_dtypes[i - 1]


def _canonicalize_jit_arg(x):
  if isinstance(x, tf_np.ndarray):
    return x.data
  else:
    try:
      # We need to convert `int` to the most precise dtype, otherwise the dtype
      # of the result may be different from numpy's. For example, when a binary
      # op takes in a Python integer 5 and an array of uint32, numpy will pick
      # uint32 as 5's dtype, while tf.convert_to_tensor will choose int32 which
      # will cause the two arguments to be promoted to int64. We pick uint8
      # here, which will be promoted to uint32 by the binary op.
      # Note that we prefer unsigned int to signed int when both are equally
      # precise. For example, for 5, we pick uint8 instead of int8. There is no
      # reason to prefer one to the other, because for each there is a case
      # where the behavior diverges from numpy. If we prefer signed int,
      # consider the case where the first operand is 5 and the second is
      # 2**64-1. Numpy picks uint64 as the result dtype, but because we choose a
      # signed type for 5 such as int8, the result type will be float64. On the
      # other hand, if we prefer unsigned int, consider the case where the first
      # operand is 2**31-1 and the second is -1. Numpy will pick int32, but
      # because we choose uint32 for 2*32-1, the result will be int64. The root
      # of the problem is that `jit` converts `int` to tensors (hence committing
      # to a dtype) too early, when we don't have enough information about the
      # jitted function (e.g. which subset of the arguments should be promoted
      # together using np.result_type). tf.function doesn't have this problem
      # because it doesn't convert `int` to tensors. jax.jit doesn't have this
      # problem because it converts `int` to "int tracer" which doesn't commit
      # to a dtype.
      # TODO(wangpeng): Revisit this design and see whether we can improve `jit`
      #   and tf.function.
      dtype = most_precise_int_dtype(x)
      if dtype is None and isinstance(x, float):
        dtype = tf_np.default_float_type()
      return tf.convert_to_tensor(value=x, dtype=dtype)
    except (TypeError, ValueError):
      return x


def _canonicalize_jit_arguments(inp):
  """Canonicalize arguments to be used for jit.

  Args:
    inp: a nested structure of arguments to be canonicalized (i.e. to be
      converted to Tensors). Only tf_np.ndarray and things accepted by
      `tf.convert_to_tensor` will be converted.

  Returns:
    The canonicalized version.
  """
  return tf.nest.map_structure(_canonicalize_jit_arg, inp)


def _np_to_tf(inp):

  def f(x):
    if isinstance(x, tf_np.ndarray):
      return x.data
    else:
      return x

  return tf.nest.map_structure(f, inp)


def _tf_to_np(inp):

  def f(x):
    if isinstance(x, (tf.Tensor, tf.IndexedSlices)):
      return tf_np.array(x, copy=False)
    else:
      return x

  return tf.nest.map_structure(f, inp)


def stop_gradient(x):
  return _tf_to_np(tf.nest.map_structure(tf.stop_gradient, _np_to_tf(x)))


def custom_grad(f_vjp, f_original=None):
  """Decorator to define a function with a custom gradient.

  This function is very similar to `tf.custom_gradient`. See the documentation
  of `tf.custom_gradient` for detailed usage.

  The differences with `tf.custom_gradient` are:

  - All arguments and results are tf_np.ndarrays instead of tensors.

  - The `grad_fn` returned by `f_vjp` accepts and returns nested structures,
    unlike that in `tf.custom_gradient` which only accepts and returns lists.

  Args:
    f_vjp: the same as the `f` argument of `tf.custom_gradient`. Note that all
      inputs and outputs of `f_vjp` and of the `grad_fn` function it returns can
      be nested structures.
    f_original: (optional) not used.

  Returns:
    The same as `tf.custom_gradient`.
  """
  del f_original

  @tf.custom_gradient
  def tf_f(*tf_args, **tf_kwargs):
    np_args = _tf_to_np(tf_args)
    np_kwargs = _tf_to_np(tf_kwargs)
    np_y, np_vjp = f_vjp(*np_args, **np_kwargs)
    tf_y = _np_to_tf(np_y)

    def tf_vjp(*flat_tf_dy):
      tf_dy = tf.nest.pack_sequence_as(tf_y, flat_tf_dy)
      np_dy = _tf_to_np(tf_dy)
      np_dx = np_vjp(np_dy)
      return tf.nest.flatten(_np_to_tf(np_dx))

    return tf_y, tf_vjp

  def np_f(*args, **kwargs):
    return _tf_to_np(tf_f(*_np_to_tf(args), **_np_to_tf(kwargs)))

  return np_f


def vjp(f, *primals, has_aux=False):
  """Returns the result and the VJP function of `f`.

  This function returns the result and the vector-Jacobian-product (VJP)
  function of `f`.

  Args:
    f: a function from (nested structures of) tf_np.ndarrays to a (nested
      structure of) tf_np.ndarray. If `has_aux` is True, it should return an
      extra output.
    *primals: the inputs to be fed to `f`.
    has_aux: if True, the second output of `f` will be regarded as an auxiliary,
      non-differentiable output that will be ignored by the VJP function.

  Returns:
    A pair `(y, vjpfun)` if `has_aux` is False; a tuple `(y, vjpfun, aux)`
    otherwise. `y` and `aux` are the outputs of `f`, i.e. `y, aux =
    f(*primals)`. `vjpfun` is a function `dx = vjpfun(dy)`, where `dy` is the
    cotengents of `y`, having the same structures, shapes and dtypes as
    `y`. `dx` is the cotengents of `x`, having the same structures, shapes and
    dtypes as `x`.
  """
  tf_primals = _np_to_tf(primals)
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(tf.nest.flatten(tf_primals))
    outputs = f(*primals)
    if has_aux:
      np_out, aux = outputs
    else:
      np_out = outputs
    tf_out = _np_to_tf(np_out)

    def _vjp(dy):
      tf_dy = _np_to_tf(dy)
      tf_dx = tape.gradient(tf_out, tf_primals, output_gradients=tf_dy)
      return _tf_to_np(tf_dx)

  if has_aux:
    ret = (np_out, _vjp, aux)
  else:
    ret = (np_out, _vjp)
  return ret


# TODO(wangpeng): match JAX's handling of kwargs and non-ndarray args
def grad(f, has_aux=False):
  """Returns a function that computes gradient of f.

  Gradients can only be computed through numpy and tensorflow operations and not
  through python float operations and values.

  Args:
    f: a function of type (params, *args) -> scalar. 'params' can be a nested
      structure (made of lists and tuples) of ndarrays and the gradient is
      evaluated against it. `scalar` is a scalar ndarray.
    has_aux: bool, indicates whether fun returns a pair where the first element
      is considered the output of the mathematical function to be differentiated
      and the second element is auxiliary data.

  Returns:
    A gradient function of type (params, *args) -> gradients, where the result
    'gradients' has the same structure and shapes as 'params'.
  """

  def check_loss_shape(np_loss):
    if not isinstance(np_loss, tf_np.ndarray):
      raise ValueError(
          "The result of the function to take gradient must be an ndarray.")
    if not np_loss.data.shape.is_compatible_with([]):
      raise ValueError(
          "The result of the function to take gradient must be a scalar.")

  def _f(params, *args):
    """The gradient function to be returned."""
    tf_params = _np_to_tf(params)
    with tf.GradientTape() as g:
      g.watch(tf.nest.flatten(tf_params))
      outputs = f(params, *args)
      if has_aux:
        np_loss, aux = outputs
      else:
        np_loss = outputs
      check_loss_shape(np_loss)
      tf_grads = g.gradient(np_loss.data, tf_params)
      if has_aux:
        res = (tf_grads, aux)
      else:
        res = tf_grads
      return _tf_to_np(res)

  return _f


# A workaround for b/121383831
_orig_result_is_list = threading.local()


def _record_result_type(f):
  # A wrapper just for setting _orig_result_is_list, as a workaround for
  # b/121383831
  def wrapper(*args, **kwargs):
    res = f(*args, **kwargs)
    _orig_result_is_list.val = isinstance(res, list)
    return res

  return wrapper


def jit(f,
        static_argnums=(),
        xla_forced_compile=False,
        input_signature=None,
        autograph=False):
  """Returns a function that runs a trace-compiled version of `f`.

  A trace-compiled version of a function `f` has the same behavior as `f` (when
  called with the same "static arguments", see below), but runs faster because
  the whole computation is compiled into a computation graph once which is
  reused for subsequent executions.

  The trace compilation happens lazily, when the returned function is called for
  the first time. The compiled function may not be cached implicitly and
  multiple calls to `jit` may not share the compiled function (see below for
  "static" vs "dynamic" arguments).

  Args:
    f: a function that takes any positional arguments `args` and any keyword
      arguments `kwargs`. `ndarray`s and things accepted by
      `tf.convert_to_tensor` in `args` and `kwargs` will be treated as 'dynamic
      arguments' in the sense that calling the function with different values
      for these arguments will not cause retracing. In contrast, arguments of
      other types in `args` and `kwargs` are treated as 'static arguments' and
      calling the function with different values of them will cause
      re-compiling. Positional arguments whose positions are in `static_argnums`
      are always treated as static arguments.
    static_argnums: a tuple of positions of arguments that will be treated as
      static arguments. Note that as aforementioned, any arguments that were not
      convertible to tensor will also be static.
    xla_forced_compile: if true, it will use XLA to force-compile the graph.
      This requires that the function only contain ops that are XLA compatible.
    input_signature: a list of `tf.TensorSpec`, as the input signature to
      control tracing behavior. See the
      [doc](https://www.tensorflow.org/api_docs/python/tf/function]) of
        `tf.function` for details.
    autograph: whether to use autograph to convert Python constructs such as
      `if` and `while` to their TensorFlow counterparts. See the
      [doc](https://www.tensorflow.org/api_docs/python/tf/function]) of
        `tf.function` for details.

  Returns:
    A trace-compiled version of f.
  """

  @tf.function(input_signature=input_signature, autograph=autograph)
  def _tf_f(*args, **kwargs):
    """Accelerated function with tensor inputs/outputs."""
    np_args = _tf_to_np(args)
    kwargs = {k: _tf_to_np(v) for k, v in kwargs.items()}
    if xla_forced_compile:
      # Workaround b/121383831
      f_ = _record_result_type(f)
      np_out = tf.xla.experimental.compile(lambda: f_(*np_args, **kwargs))
      # Workaround b/121383831
      if (isinstance(np_out, list) and len(np_out) == 1 and
          not _orig_result_is_list.val):
        np_out = np_out[0]
    else:
      np_out = f(*np_args, **kwargs)
    return _np_to_tf(np_out)

  def _f(*args, **kwargs):
    args = [
        _canonicalize_jit_arguments(arg) if i not in static_argnums else arg
        for i, arg in enumerate(args)
    ]
    kwargs = {k: _canonicalize_jit_arguments(v) for k, v in kwargs.items()}
    tf_out = _tf_f(*args, **kwargs)
    return _tf_to_np(tf_out)

  _f.tf_function = _tf_f

  return _f


def eval_on_shapes(f, static_argnums=()):
  """Returns a function that evaluates `f` given input shapes and dtypes.

  It transforms function `f` to a function that performs the same computation as
  `f` but only on shapes and dtypes (a.k.a. shape inference).

  Args:
    f: the function to be transformed.
    static_argnums: See documentation of `jit`.

  Returns:
    A function whose input arguments can be either the same as `f`'s or only
    their shapes/dtypes represented by `TensorSpec`, and whose return values are
    `TensorSpec`s with the same nested structure as `f`'s return values.
  """
  # TODO(wangpeng): tf.function could add a knob to turn off materializing the
  #   graph, so that we don't waste computation and memory when we just want
  #   shape inference.
  tf_f = jit(f, static_argnums=static_argnums).tf_function

  # pylint: disable=missing-docstring
  def f_return(*args):

    def abstractify(x):
      x = _canonicalize_jit_arg(x)
      if isinstance(x, (tf.Tensor, tf_np.ndarray)):
        return tf.TensorSpec(x.shape, x.dtype)
      else:
        return x

    def to_tensor_spec(x):
      if isinstance(x, tf.Tensor):
        return tf.TensorSpec(x.shape, x.dtype)
      else:
        return x

    new_args = []
    for i, arg in enumerate(args):
      if i in static_argnums:
        new_args.append(arg)
      else:
        new_args.append(tf.nest.map_structure(abstractify, arg))
    res = tf_f.get_concrete_function(*new_args).structured_outputs

    return tf.nest.map_structure(to_tensor_spec, res)

  # Provides access to `tf_f` for testing purpose.
  f_return._tf_function = tf_f  # pylint: disable=protected-access
  return f_return


def logsumexp(x, axis=None, keepdims=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `x` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.
  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  Args:
    x: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(x), rank(x))`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    The reduced tensor.
  """
  return tf_np.asarray(
      tf.math.reduce_logsumexp(
          input_tensor=x.data, axis=axis, keepdims=keepdims))


def expit(x):
  """Compute 1 / (1 + exp(-x))."""
  return tf_np.asarray(tf.math.sigmoid(x.data))


def erf(x):
  """Computes the Gauss error function of x element-wise."""
  return tf_np.asarray(tf.math.erf(x.data))


def conv(inp,
         fltr,
         window_strides,
         padding,
         dimension_numbers,
         filter_dilation=None):
  """Convolution over an N-D array.

  See https://www.tensorflow.org/api_docs/python/tf/nn/convolution and
  https://www.tensorflow.org/xla/operation_semantics#conv_convolution for
  reference.

  Args:
    inp: an (N+2)-D array. The input of the convolution.
    fltr: an (N+2)-D array. The filter (i.e. kernel) of the convolution.
    window_strides: a sequence of N ints, the strides for moving the convolution
      window.
    padding: a string, either "VALID" or "SAME". The padding algorithm.
    dimension_numbers: a tuple of three strings encoding the data format of
      input, filter and output. "I" means input; "O" means output; "C" means
      channel; other characters such as "W", "H" and "D" means spatial
      dimensions.
    filter_dilation: the dilation rates for the filter. Dilating the filter
      means adding "holes" to the filter.

  Returns:
    An (N+2)-D array. The convolution result.
  """
  input_spec, filter_spec, output_spec = dimension_numbers
  if input_spec != output_spec:
    raise ValueError("Input and output data formats must be the same; got %s "
                     "and %s" % (input_spec, output_spec))
  supported_filter_spec = ["WIO", "HWIO", "DHWIO"]
  if filter_spec not in supported_filter_spec:
    raise ValueError("The supported data format for the filter are %s; got %s" %
                     (supported_filter_spec, filter_spec))
  if input_spec[1:-1] != filter_spec[:-2]:
    raise ValueError("Input data format (%s) is not compatible with filter "
                     "data format (%s)" % (input_spec, filter_spec))
  # No type promotion in order to prevent accidentally doing more expensive
  # computation.
  inp = tf_np.asarray(inp)
  fltr = tf_np.asarray(fltr)
  return tf_np.asarray(
      tf.nn.convolution(
          input=inp.data,
          filters=fltr.data,
          padding=padding,
          strides=window_strides,
          dilations=filter_dilation,
          data_format=input_spec))


def avg_pool(x, pool_size, strides, padding):
  """Performs an N-D average pooling.

  Args:
    x: ndarray of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]`. Pooling happens over the spatial dimensions only.
    pool_size: sequence of N ints.
    strides: sequence of N ints.
    padding: a string, the padding algorithm. Must be "SAME" or "VALID".

  Returns:
    An (N+2)-D array,  of shape
      [batch_size] + output_spatial_shape + [num_channels],
    where `output_spatial_shape` depends on the value of padding:
    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (pool_size[i] - 1)) / strides[i]).
  """
  x = tf_np.asarray(x)
  return tf_np.asarray(
      tf.nn.pool(
          input=x,
          window_shape=pool_size,
          pooling_type="AVG",
          strides=strides,
          padding=padding))


def max_pool(x, pool_size, strides, padding):
  """Performs an N-D max pooling.

  Args:
    x: ndarray of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]`. Pooling happens over the spatial dimensions only.
    pool_size: sequence of N ints.
    strides: sequence of N ints.
    padding: a string, the padding algorithm. Must be "SAME" or "VALID".

  Returns:
    An (N+2)-D array,  of shape
      [batch_size] + output_spatial_shape + [num_channels],
    where `output_spatial_shape` depends on the value of padding:
    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (pool_size[i] - 1)) / strides[i]).
  """
  x = tf_np.asarray(x)
  return tf_np.asarray(
      tf.nn.pool(
          input=x,
          window_shape=pool_size,
          pooling_type="MAX",
          strides=strides,
          padding=padding))


def sort_key_val(keys, values, dimension=-1):
  """Sorts keys along a dimension and applies same permutation to values.

  Args:
    keys: an array. The dtype must be comparable numbers (integers and reals).
    values: an array, with the same shape of `keys`.
    dimension: an `int`. The dimension along which to sort.

  Returns:
    Permuted keys and values.
  """
  keys = tf_np.asarray(keys)
  values = tf_np.asarray(values)
  rank = keys.data.shape.ndims
  if rank is None:
    rank = values.data.shape.ndims
  if rank is None:
    # We need to know the rank because tf.gather requires batch_dims to be `int`
    raise ValueError("The rank of either keys or values must be known, but "
                     "both are unknown (i.e. their shapes are both None).")
  if dimension in (-1, rank - 1):

    def maybe_swapaxes(a):
      return a
  else:

    def maybe_swapaxes(a):
      return tf_np.swapaxes(a, dimension, -1)

  # We need to swap axes because tf.gather (and tf.gather_nd) supports
  # batch_dims on the left but not on the right.
  # TODO(wangpeng): Investigate whether we should do swapaxes or moveaxis.
  keys = maybe_swapaxes(keys)
  values = maybe_swapaxes(values)
  idxs = tf_np.argsort(keys)
  idxs = idxs.data

  # Using tf.gather rather than np.take because the former supports batch_dims
  def gather(a):
    return tf_np.asarray(tf.gather(a.data, idxs, batch_dims=rank - 1))

  keys = gather(keys)
  values = gather(values)
  keys = maybe_swapaxes(keys)
  values = maybe_swapaxes(values)
  return keys, values


# Use int64 instead of int32 to avoid TF's "int32 problem"
_RNG_KEY_DTYPE = np.int64


def _key2seed(a):
  """Converts an RNG key to an RNG seed.

  Args:
    a: an RNG key, an ndarray of shape [] and dtype `np.int64`.

  Returns:
    an RNG seed, a tensor of shape [2] and dtype `tf.int32`.
  """

  def int64_to_int32s(a):
    """Converts an int64 tensor of shape [] to an int32 tensor of shape [2]."""
    a = tf.cast(a, tf.uint64)
    fst = tf.cast(a, tf.uint32)
    snd = tf.cast(
        tf.bitwise.right_shift(a, tf.constant(32, tf.uint64)), tf.uint32)
    a = [fst, snd]
    a = tf.nest.map_structure(lambda x: tf.cast(x, tf.int32), a)
    a = tf.stack(a)
    return a

  return int64_to_int32s(a.data)


def _seed2key(a):
  """Converts an RNG seed to an RNG key.

  Args:
    a: an RNG seed, a tensor of shape [2] and dtype `tf.int32`.

  Returns:
    an RNG key, an ndarray of shape [] and dtype `np.int64`.
  """

  def int32s_to_int64(a):
    """Converts an int32 tensor of shape [2] to an int64 tensor of shape []."""
    a = tf.bitwise.bitwise_or(
        tf.cast(a[0], tf.uint64),
        tf.bitwise.left_shift(
            tf.cast(a[1], tf.uint64), tf.constant(32, tf.uint64)))
    a = tf.cast(a, tf.int64)
    return a

  return tf_np.asarray(int32s_to_int64(a))


def prng(s):
  """Creates RNG state from seed.

  Args:
    s: the seed, an integer.

  Returns:
    An RNG state, as a scalar array of dtype `np.int64`.
  """
  # TODO(wangpeng): Become bitwise-identical to JAX when TF stateless RNGs get
  #   improved.
  return tf_np.asarray(s, dtype=_RNG_KEY_DTYPE)


def stateless_split(seed, num=2):
  """Splits an RNG seed into `num` new seeds by adding a leading axis.

  Example:

  >>> seed = [1, 2]
  >>> new_seeds = tf.random.experimental.stateless_split(seed, num=3)
  >>> print(new_seeds)
  tf.Tensor(
  [[1105988140 1738052849]
   [-335576002  370444179]
   [  10670227 -246211131]], shape=(3, 2), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.59835213, -0.9578608 ,
  0.9002807 ], dtype=float32)>

  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or
      `int64`). (When using XLA, only `int32` is allowed.)
    num: optional, a positive integer or scalar tensor indicating the number of
      seeds to produce (default 2).

  Returns:
    A tensor with shape [num, 2] representing `num` new seeds. It will have the
    same dtype as `seed` (if `seed` doesn't have an explict dtype, the dtype
    will be determined by `tf.convert_to_tensor`).
  """
  seed = tf.convert_to_tensor(seed)
  return tf.random.stateless_uniform(
      shape=[num, 2], seed=seed, dtype=seed.dtype, minval=None, maxval=None)


def split(state, num):
  """Creates new independent RNG states from an existing state.

  Args:
    state: the existing state.
    num: the number of the new states.

  Returns:
    A tuple of new states.
  """
  state = tf_np.asarray(state, dtype=_RNG_KEY_DTYPE)
  state = _key2seed(state)
  try:
    states = tf.random.experimental.stateless_split(state, num)
  except AttributeError as e:  # pylint: disable=unused-variable
    # TODO(afrozm): For TF < 2.3 we need to do this. Delete once 2.3 launches.
    states = stateless_split(state, num)
  states = tf.unstack(states, num)
  states = tf.nest.map_structure(_seed2key, states)
  return states


def uniform(key,
            shape,
            dtype=tf_np.random.DEFAULT_RANDN_DTYPE,
            minval=0.,
            maxval=1.):
  """Sample uniform random values in range [`minval`, `maxval`).

  Args:
    key: the RNG key.
    shape: the shape of the result.
    dtype: the dtype of the result.
    minval: the minimal value (inclusive).
    maxval: the maximal value (exclusive).

  Returns:
    An ndarray with shape `shape` and dtype `dtype`. Each value in the ndarray
    is sampled uniformly randomly in range [`minval`, `maxval`).
  """
  key = tf_np.asarray(key, dtype=_RNG_KEY_DTYPE)
  return tf_np.asarray(
      tf.random.stateless_uniform(
          shape, seed=_key2seed(key), dtype=dtype, minval=minval,
          maxval=maxval))


def normal(key, shape, dtype=tf.float32):
  """Sample standard-normal random values.

  Args:
    key: the RNG key.
    shape: the shape of the result.
    dtype: the dtype of the result.

  Returns:
    Random values in standard-normal distribution.
  """
  key = tf_np.asarray(key, dtype=_RNG_KEY_DTYPE)
  return tf_np.asarray(
      tf.random.stateless_normal(shape, seed=_key2seed(key), dtype=dtype))


def bernoulli(key, mean=np.float32(0.5), shape=None):
  """Sample Bernoulli random values with given shape and mean.

  Args:
    key: the RNG key.
    mean: optional, an array_like broadcastable to `shape` for the mean of the
      random variables (default 0.5).
    shape: optional, a tuple of nonnegative integers representing the shape
      (default to `mean`'s shape).

  Returns:
    A random array with the specified shape and boolean dtype.
  """
  mean = tf_np.asarray(mean)
  if shape is None:
    shape = mean.shape
  return uniform(key, shape) < mean


def _eager_dataset_iterator(dataset):
  for item in dataset:
    yield tf.nest.map_structure(tf_np.asarray, item)


def dataset_as_numpy(dataset):
  """Converts a `tf.data.Dataset` to an iterable of ndarrays.

  `dataset_as_numpy` converts a possibly nested structure of `tf.data.Dataset`s
  and `tf.Tensor`s to iterables of ndarrays and ndarrays, respectively. This
  function must be run in eager mode outside tf.function.

  Args:
    dataset: a possibly nested structure of `tf.data.Dataset`s and/or
      `tf.Tensor`s.

  Returns:
    A structure matching `dataset` where `tf.data.Dataset`s are converted to
    generators of ndarrays and `tf.Tensor`s are converted to ndarrays.
  """
  if not tf.executing_eagerly():
    raise ValueError(
        "dataset_as_numpy must be run in eager mode outside tf.function")
  nested_ds = dataset
  del dataset

  # Flatten
  flat_ds = tf.nest.flatten(nested_ds)
  flat_np = []

  # Type check for Tensors and Datasets
  for ds_el in flat_ds:
    if not isinstance(ds_el, (tf.Tensor, tf.data.Dataset)):
      types = tf.nest.map_structure(type, nested_ds)
      raise ValueError("Arguments to dataset_as_numpy must be (possibly nested "
                       "structure of) tf.Tensors or tf.data.Datasets. Got: %s" %
                       types)

  for ds_el in flat_ds:
    if isinstance(ds_el, tf.Tensor):
      np_el = tf_np.asarray(ds_el)
    elif isinstance(ds_el, tf.data.Dataset):
      np_el = _eager_dataset_iterator(ds_el)
    else:
      assert False
    flat_np.append(np_el)

  return tf.nest.pack_sequence_as(nested_ds, flat_np)


# TODO(nareshmodi): Group key should change based on the set of devices that we
# are mapping over. Make it so that we assign a unique group_key for every
# unique set of devices. We don't change it every time to avoid the overhead of
# discovering the full group (though may not be problematic in the local case).
_GROUP_KEY = 1
_INSTANCE_KEY = 0
_INSTANCE_LOCK = threading.Lock()


# TODO(b/142565636): Ensure that multiple concurrent calls to a tf.function
# containing a collective op run reasonably.
def _get_instance_key():
  global _INSTANCE_KEY
  global _INSTANCE_LOCK
  with _INSTANCE_LOCK:
    _INSTANCE_KEY = _INSTANCE_KEY + 1
    return _INSTANCE_KEY


# Don't use a namedtuple since nest considers that a tuple and unflattens and
# flattens it.
class ShardedNdArray(object):
  """Wrapper over ndarray that can contain tensors on multiple devices.

    This is returned by extensions.pmap, and contains the individual tensors on
    different devices.
  """

  def __init__(self, tensors):
    """Initializes the ShardedNdArray.

    Note that the tensors should be ordered in the way the pmap producing these
    tensors is run.

    Args:
      tensors: list or tuple of eager tensors, one for each device.
    """

    if not isinstance(tensors, (list, tuple)) or not tensors:
      raise ValueError(
          "Unable to create a ShardedNdArray without a list of tensors.")
    self.tensors = tensors
    self.n_devices = len(tensors)

  def __getitem__(self, i):
    return self.tensors[i]

  @property
  def shape(self):
    return (self.n_devices,) + self.tensors[0]._shape_tuple()  # pylint: disable=protected-access

  @property
  def dtype(self):
    return self.tensors[0].dtype


def convert_sharded_tensor_to_eager_tensor(value, *args, **kwargs):
  del args, kwargs
  # TODO(nareshmodi): Consider a collective op to gather the tensors from the
  # various devices for performance reasons.
  return tf.stack(value.tensors)


tf.register_tensor_conversion_function(ShardedNdArray,
                                       convert_sharded_tensor_to_eager_tensor)


class _PmapConfig(threading.local):
  """Simple config used to maintain state related to a current pmap call."""

  def __init__(self):
    super(_PmapConfig, self).__init__()
    self._axis_name = None
    self._devices = None

  def axis_name(self):
    return self._axis_name

  def set_axis_name(self, axis_name):
    self._axis_name = axis_name

  def devices(self):
    return self._devices

  def set_devices(self, devices):
    self._devices = devices


_pmap_config = _PmapConfig()


@contextlib.contextmanager
def pmap_config(axis_name, devices):
  """Records axis_name and devices for this context."""
  old_axis_name = _pmap_config.axis_name()
  old_devices = _pmap_config.devices()
  _pmap_config.set_axis_name(axis_name)
  _pmap_config.set_devices(devices)
  try:
    yield
  finally:
    _pmap_config.set_axis_name(old_axis_name)
    _pmap_config.set_devices(old_devices)


def psum(tensor, axis_name=None):
  """Sum all-reduction.

  Args:
    tensor: A tensor.
    axis_name: The axis name to reduce. Must equal to that of the surrounding
      pmap.

  Returns:
    The sum of the `tensor` replicas on each participating devices.
  """
  if axis_name != _pmap_config.axis_name():
    raise ValueError("axis_name (%s) is not equal to that of the surrounding "
                     "pmap (%s)" % (axis_name, _pmap_config.axis_name()))
  devices = _pmap_config.devices()
  if devices is None:
    raise ValueError("Can't retrieve the device list from the surrounding pmap")
  if tpu_devices(devices):
    # TODO(wangpeng): Supply the `group_assignment` argument to
    # tpu.cross_replica_sum, calculated from `devices`.
    return tf.compat.v1.tpu.cross_replica_sum(tensor)
  else:
    return tf.raw_ops.CollectiveReduce(
        input=tensor.data,
        group_size=len(devices),
        group_key=_GROUP_KEY,
        instance_key=_get_instance_key(),
        merge_op="Add",
        final_op="Id",
        subdiv_offsets=(0,))


# Note this is not available in the jax api, but seemed like a reasonable API
# to have.
def pmean(tensor, axis_name=None):
  """Mean all-reduction.

  Args:
    tensor: A tensor.
    axis_name: The axis name to reduce. Must equal to that of the surrounding
      pmap.

  Returns:
    The mean of the `tensor` replicas on each participating devices.
  """
  if axis_name != _pmap_config.axis_name():
    raise ValueError("axis_name (%s) is not equal to that of the surrounding "
                     "pmap (%s)" % (axis_name, _pmap_config.axis_name()))
  devices = _pmap_config.devices()
  if devices is None:
    raise ValueError("Can't retrieve the device list from the surrounding pmap")
  if tpu_devices(devices):
    # TODO(wangpeng): Implement this.
    raise ValueError("pmean for TPU is not supported yet.")
  else:
    return tf.raw_ops.CollectiveReduce(
        input=tensor.data,
        group_size=len(devices),
        group_key=_GROUP_KEY,
        instance_key=_get_instance_key(),
        merge_op="Add",
        final_op="Div",
        subdiv_offsets=(0,))


def _get_pmap_impl(f, devices, has_tpu):
  """This is a helper function to return the pmap impl.

  Args:
    f: a function that takes ndarrays and returns ndarrays.
    devices: a list of strings; the device list.
    has_tpu: boolean; whether `devices` contains TPU devices.

  Returns:
    A function that takes tensors and returns tensors.
  """
  if has_tpu:
    # Workaround b/121383831
    f = _record_result_type(f)

  def tf_f(*tf_args):
    """A wrapper for `f` that takes/returns tensors."""
    np_args = _tf_to_np(tf_args)
    np_out = f(*np_args)
    return _np_to_tf(np_out)

  if has_tpu:

    @tf.function(autograph=False)
    def fn(inputs):
      # TODO(wangpeng): Supply the `device_assignment` argument to
      # tpu.replicate, calculated from `devices`.
      return tf.compat.v1.tpu.replicate(tf_f, inputs)

    return fn
  else:
    # This is run in a tf.function so that the various underlying functions can
    # be run in parallel.
    # The trace happens on the client, so any devices should not depend on any
    # side effects.

    jit_tf_f = tf.function(tf_f, autograph=False)

    @tf.function(autograph=False)
    def fn(all_per_device_args):
      """Multi-device function with calls placed on the correct device."""

      results = []
      for per_device_args, device in zip(all_per_device_args, devices):
        with tf.device(device):
          results.append(jit_tf_f(*per_device_args))
      return results

    return fn


def pmap(f, axis_name=None, devices=None):
  """Transforms a function into a multi-device function.

  The semantics are similar to JAX's pmap.

  Args:
    f: The function to be converted.
    axis_name: Used for nested pmap, which is not supported yet.
    devices: The devices over which the returned function will run.

  Returns:
    A function that runs the underlying function `f` on `devices`. Its arguments
    can be `ShardedNdArray`s, tensors or other Python objects, and its return
    values are all `ShardedNdArray`s. If an input is a tensor, the length of its
    first dimension must equal the number of devices, and the tensor will be
    splitted along its first dimension among the devices. If an input is an
    unknown Python object, it will be replicated among the devices.
  """
  if devices is None:
    devices = accelerators()
  if not isinstance(devices, (list, tuple)):
    raise ValueError("Must pass a list or tuple of devices")
  num_devices = len(devices)
  if not num_devices:
    raise ValueError("There must be at least 1 device")
  has_tpu = bool(tpu_devices(devices))

  pmap_fn = _get_pmap_impl(f, devices, has_tpu)

  def wrapper(*args):
    """Wrapper that wraps/unwraps args, retvals, and runs the function."""
    if _pmap_config.devices() is not None:
      raise ValueError("Found a surrounding pmap. Nested pmap is not supported "
                       "yet.")
    # TODO(wangpeng): Maybe we should use `asarray` to convert everything
    # to ndarray first.
    args = _np_to_tf(args)

    flattened_input_args = tf.nest.flatten(args)
    flattened_per_device_args = [[] for _ in devices]
    for arg in flattened_input_args:
      if isinstance(arg, tf.Tensor):
        # TODO(nareshmodi): Try and use the dynamic shape instead.
        if (not arg.shape.rank) or arg.shape[0] != len(devices):
          # TODO(nareshmodi): Fix this restriction
          raise ValueError(
              "Input tensors need to have a first dimension equal to "
              "the number of devices; got tensor of shape %s and %s devices" %
              (arg.shape, len(devices)))
        # NOTE: Alternatively use tf.split, and place the split tensors on the
        # appropriate device. The best solution for this is to have an API that
        # splits a tensor across devices.
        for j, device in enumerate(devices):
          updated_arg = tf.gather(arg, j)
          # TODO(wangpeng): Investigate whether we need a tf.identity for TPU.
          if not has_tpu:
            with tf.device(device):
              updated_arg = tf.identity(updated_arg)
          flattened_per_device_args[j].append(updated_arg)
      elif isinstance(arg, ShardedNdArray):
        for device_args, tensor in zip(flattened_per_device_args, arg.tensors):
          device_args.append(tensor)
      else:
        for device_args in flattened_per_device_args:
          device_args.append(arg)

    all_per_device_args = [
        tf.nest.pack_sequence_as(args, device_args)
        for device_args in flattened_per_device_args
    ]

    with pmap_config(axis_name, devices):
      results = pmap_fn(all_per_device_args)

    # Rewrap things. This can probably be written better.
    flattened_results = [tf.nest.flatten(result) for result in results]
    final_tree = []

    # TODO(nareshmodi): assert all items in flattened_results have the same
    # structures

    for i in range(len(flattened_results[0])):
      tensors = []
      for j, device in enumerate(devices):
        assert isinstance(
            flattened_results[j][i],
            tf.Tensor), ("currently only tensor return items are supported")
        tensors.append(flattened_results[j][i])
      final_tree.append(ShardedNdArray(tensors))

    final_actual_result = tf.nest.pack_sequence_as(results[0], final_tree)

    # Workaround b/121383831
    if (has_tpu and isinstance(final_actual_result, list) and
        len(final_actual_result) == 1) and not _orig_result_is_list.val:
      return final_actual_result[0]
    else:
      return final_actual_result

  return wrapper


def find_devices(device_type, devices=None):
  if not devices:
    devices = [d.name for d in tf.config.experimental.list_logical_devices()]
  devices = [(d, tf.DeviceSpec.from_string(d)) for d in devices]
  results = [name for name, d in devices if d.device_type == device_type]
  return results


def tpu_devices(devices=None):
  """Gets TPU devices out of `devices`.

  Args:
    devices: A device list (as a list of strings). If None, the list of all
      available devices will be used for it.

  Returns:
    Those in `devices` that are TPUs.
  """
  return find_devices("TPU", devices)


def gpu_devices(devices=None):
  """Gets GPU devices out of `devices`.

  Args:
    devices: A device list (as a list of strings). If None, the list of all
      available devices will be used for it.

  Returns:
    Those in `devices` that are GPUs.
  """
  return find_devices("GPU", devices)


def accelerators(devices=None):
  return tpu_devices(devices) or gpu_devices(devices)
