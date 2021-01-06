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

"""Trax-to-Keras converter."""

import functools

import tensorflow.compat.v2 as tf

from trax import fastmath as math_lib
from trax import shapes as shapes_lib
from trax.fastmath import numpy as jnp
from trax.layers import base


def _replace_none_batch(x, batch_size=None):
  if batch_size is None:
    return x
  if isinstance(x, tf.Tensor) and x.shape[0] is None:
    x.set_shape([batch_size] + x.shape[1:])
    return x
  elif isinstance(x, tf.TensorShape) and x[0] is None:
    return [batch_size] + x[1:]
  return x


def tensor_shapes_to_shape_dtypes(shapes, dtype):
  return math_lib.nested_map(
      lambda s: shapes_lib.ShapeDtype(s.as_list(), dtype), shapes)


def read_values(variables):
  return math_lib.nested_map(lambda v: v.read_value(), variables)


def to_tensors(args):
  return math_lib.nested_map(tf.convert_to_tensor, args)


def to_arrays(args):
  return math_lib.nested_map(jnp.asarray, args)


class AsKeras(tf.keras.layers.Layer):
  """A Keras layer built from a Trax layer.

  This subclass of `tf.keras.layers.Layer` takes in a Trax layer as a
  constructor argument and wraps it to be a Keras layer. It uses
  `tf.Variable` to store weights and state (initialized according to the Trax
  layer), and uses the Trax layer's forward function as its forward function.

  Consider this code snippet::

      keras_layer = AsKeras(trax_layer, initializer_rng=initializer_rng,
                                   rng=rng, rng_updater=rng_updater)
      keras_layer.build(...)  # optional
      outputs = keras_layer(inputs)

  (Note that in Keras calling `Layer.build` is optional. If omitted, it will be
  called automatically by `Layer.__call__`.)

  If `trax_layer` already has weights at `build` time, the snippet is roughly
  equivalent to::

      weights = trax_layer.weights
      state = trax_layer.state
      keras_layer = tf.keras.layers.Layer()
      keras_layer._weights = tf.Variable(weights)
      keras_layer._state = tf.Variable(state)
      keras_layer._rng = tf.Variable(rng)
      outputs, new_state = trax_layer(inputs, keras_layer._weights,
                                      keras_layer._state, keras_layer._rng)
      keras_layer._state.assign(new_state)
      keras_layer._rng.assign(rng_updater(rng))

  If `trax_layer` doesn't have weights at `build` time, the snippet is roughly
  equivalent to::

      weights, state = trax_layer.init(..., rng=initializer_rng)
      keras_layer = ...
      ...

  `AsKeras` uses `tf.Variable` to store weights, not shared with the
  original Trax layer (which uses tensors to store weights), so using
  `AsKeras` may double the memory footprint. This problem can be solved
  by making sure that the Trax layer's weights/state are cleared whenever
  `tf.Variable.assign` (and `tf.Variable.assign_add` etc.) is called, because
  `tf.Variable` is copy-on-write by default.

  Mutations in those `tf.Variable`s won't affect the Trax layer's weights, but
  `AsKeras`'s forward function calls the Trax layer's forward function,
  which caches the weights in the Trax layer object, so a forward pass may
  change the weights cached in the original Trax layer.

  Note that this class is not thread-safe. If the same `AsKeras` object
  is used in multiple threads, the `tf.Variable` updates may happen in a
  non-deterministic order.
  """

  def __init__(self, trax_layer, batch_size=None, initializer_rng=None,
               rng=None, rng_updater=None, dtype=None):
    """Creates a Keras layer wrapping around a Trax layer.

    Args:
      trax_layer: an object of class `trax.layers.Layer`, the trax layer to
        wrap.
      batch_size: (optional) an integer, the batch size that this Keras layer
        will be used on. Keras sometimes needs to generate a TF graph for a
        layer (e.g. for acceleration or checkpointing). The inputs used to trace
        the graph will have `None` as the length of their batch dimensions, so
        as to generate a graph that can handle any batch size. Some Trax layers
        can't handle tensors whose shapes contain `None`. If `batch_size` is set
        to an integer, the graph will be traced with `batch_size` as the batch
        size instead of `None`. Note that in this case the graph (and the Keras
        layer) can only be used on a specific batch size. If you want to use a
        different batch size, you need to create another `AsKeras` object
        with a different `batch_size`.
      initializer_rng: (optional) an RNG key used to create the weights and
        state if `trax_layer` doesn't have them. If `None`,
        `trax.fastmath.random.get_prng(0)` will be used.
      rng: (optional) an RNG key for the forward function (aka the "forward
        key"). If `None`, `trax.fastmath.random.get_prng(0)` will be used.
      rng_updater: (optional) a function of type rng_key -> rng_key, used to
        update the forward key after each forward pass. If `None`, the function
        `lambda x: trax.fastmath.random.split(x, 1)[0]` will be used, which
        advances the RNG key.
      dtype: (optional) the dtype of the inputs. See the `dtype` argument of
        `tf.keras.layers.Layer.__init__` for details.
    """
    super().__init__(dtype=dtype)
    with math_lib.use_backend(math_lib.Backend.TFNP):
      if initializer_rng is None:
        initializer_rng = math_lib.random.get_prng(0)
      if rng is None:
        rng = math_lib.random.get_prng(0)
      if rng_updater is None:
        rng_updater = lambda x: math_lib.random.split(x, 1)[0]
      self._trax_layer = trax_layer
      self._batch_size = batch_size
      self._initializer_rng = initializer_rng
      self._forward_rng_init = rng
      self._rng_updater = rng_updater

  def build(self, input_shape):
    with math_lib.use_backend(math_lib.Backend.TFNP):
      # Using `is` instead of `==` following Trax's practice
      if self._trax_layer.weights is base.EMPTY_WEIGHTS:
        sanitized_input_shape = math_lib.nested_map(
            functools.partial(_replace_none_batch, batch_size=self._batch_size),
            input_shape)
        weights, state = self._trax_layer.init(
            tensor_shapes_to_shape_dtypes(sanitized_input_shape, self.dtype),
            rng=self._initializer_rng)
      else:
        weights = self._trax_layer.weights
        state = self._trax_layer.state
      # Note: `weights` may contain `EMPTY_WEIGHTS`
      self._weights = math_lib.nested_map(
          functools.partial(tf.Variable, trainable=True), weights)
      self._state = math_lib.nested_map(
          functools.partial(tf.Variable, trainable=False), state)
      self._rng = tf.Variable(self._forward_rng_init, trainable=False)
    super().build(input_shape)

  def call(self, inputs):
    with math_lib.use_backend(math_lib.Backend.TFNP):
      inputs = math_lib.nested_map(
          functools.partial(_replace_none_batch, batch_size=self._batch_size),
          inputs)
      weights, state, rng = read_values([self._weights, self._state, self._rng])
      inputs, weights, state, rng = to_arrays([inputs, weights, state, rng])
      outputs, new_state = self._trax_layer.pure_fn(inputs, weights=weights,
                                                    state=state, rng=rng)
      tf.nest.map_structure(lambda v, t: v.assign(t), self._state, new_state)
      self._rng.assign(self._rng_updater(rng))
      outputs = to_tensors(outputs)
      return outputs
