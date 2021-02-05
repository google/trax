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

"""Tests for trax2keras."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

import tensorflow.compat.v2 as tf

import trax
from trax import fastmath as math_lib
from trax import layers
from trax import trax2keras
from trax.fastmath import numpy as jnp
from trax.models import mlp
from trax.models import transformer
from trax.trax2keras import read_values
from trax.trax2keras import to_arrays
from trax.trax2keras import to_tensors


tf.enable_v2_behavior()


def has_gpu():
  return bool(tf.config.list_physical_devices("GPU"))


def dummy_inputs(rng, input_sig):
  def f(sig):
    shape = sig.shape
    if shape and shape[0] is None:
      shape = (2,) + tuple(shape[1:])
    if onp.issubdtype(sig.dtype, onp.integer):
      minval = 1
      # Must specify maxval for integer dtype.
      # TODO(afrozm): Revisit after TF 2.3
      maxval = 10000
    else:
      minval = 0
      maxval = 1
    return rng.uniform(
        shape=shape, dtype=sig.dtype, minval=minval, maxval=maxval)
  return math_lib.nested_map(f, input_sig)


def Mod(n):  # pylint: disable=invalid-name
  return layers.Fn("Mod", lambda x: x % n)


# Format:
#   (trax-layer maker, input shapes, input dtype, can handle None batch size?)
_LAYERS = [
    (lambda: layers.Dense(3), tf.TensorShape([4]), onp.float32, True),
    (mlp.MLP, tf.TensorShape([4]), onp.float32, False),
    (lambda: layers.Serial(Mod(8), transformer.TransformerLM(8)),
     tf.TensorShape([4]), onp.int32, False),
]


_RNG_UPDATERS = [
    lambda x: x,
    lambda rng: math_lib.random.split(rng, 1)[0],
]


# Needs tf.test.TestCase for `assertAllClose` and `get_temp_dir`
class Trax2KerasTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      [{"testcase_name": "_%s_%s_%s_%s_%s_%s" % (  # pylint: disable=g-complex-comprehension
          layer_id, rng_updater_id, batch_size, trax_has_weights,
          explicit_build, use_model),
        "layer_id": layer_id,
        "rng_updater_id": rng_updater_id,
        "batch_size": batch_size,
        "trax_has_weights": trax_has_weights,
        "explicit_build": explicit_build,
        "use_model": use_model,}
       for use_model in [True, False]
       for explicit_build in [True, False]
       for trax_has_weights in [True, False]
       for batch_size in [2, None]
       for rng_updater_id in [1]
       for layer_id in range(len(_LAYERS))
      ])
  def testTrain(self, layer_id, rng_updater_id, batch_size, trax_has_weights,
                explicit_build, use_model):
    """Tests training (forward and backward pass) for AsKeras.

    Args:
      layer_id: an integer, the index into `_LAYERS`.
      rng_updater_id: an integer, the index into `_RNG_UPDATERS`.
      batch_size: an integer or `None`, the value for the `batch_size` argument
        in `AsKeras.__init__`.
      trax_has_weights: bool, whether to make the trax layer contain weights at
        the time when `AsKeras.build` is called.
      explicit_build: bool, whether to explicitly call `AsKeras.build`.
      use_model: bool, whether to build a `tf.keras.Model` out of the
        `AsKeras` layer and use the model to do the training instead of
        the bare layer. If `True`, we will also test checkpointing and restoring
        using the model.
    """
    with trax.fastmath.use_backend("tensorflow-numpy"):
      make_trax_layer, input_shapes_no_batch, dtype, allow_none_batch = (
          _LAYERS[layer_id])
      # We make a fresh trax layer for each test case, so that different test
      # cases won't interfere with each other.
      trax_layer = make_trax_layer()
      if not allow_none_batch and batch_size is None:
        self.skipTest("This Trax layer can't handle None batch size.")
      rng_updater = _RNG_UPDATERS[rng_updater_id]
      input_shapes = math_lib.nested_map(
          lambda s: [batch_size] + s, input_shapes_no_batch)
      input_sig = trax2keras.tensor_shapes_to_shape_dtypes(input_shapes, dtype)
      initializer_rng = math_lib.random.get_prng(765)
      weights, state = trax_layer.init(input_sig, rng=initializer_rng)
      generator = tf.random.Generator.from_seed(567)
      def get_inputs():
        return dummy_inputs(generator, input_sig)
      if trax_has_weights:
        trax_layer(to_arrays(get_inputs()), weights=weights, state=state)
      rng = math_lib.random.get_prng(1234)
      keras_layer = trax2keras.AsKeras(
          trax_layer, batch_size=batch_size, initializer_rng=initializer_rng,
          rng=rng, rng_updater=rng_updater)
      if explicit_build:
        keras_layer.build(input_shapes)
      if use_model:
        x = tf.keras.Input(shape=input_shapes_no_batch, dtype=dtype)
        y = keras_layer(x)
        keras_model = tf.keras.Model(inputs=x, outputs=y)
      lr = 0.1  # learning rate
      for _ in range(3):
        inputs = get_inputs()
        with tf.GradientTape() as trax_tape:
          trax_tape.watch(tf.nest.flatten(weights))
          trax_outputs, state = trax_layer.pure_fn(
              to_arrays(inputs), weights=weights, state=state, rng=rng)
        trax_grads = trax_tape.gradient(*to_tensors([trax_outputs, weights]))
        # `g` may be `tf.IndexedSlices`, so we need to `convert_to_tensor`
        # before multiplication.
        weights = tf.nest.map_structure(
            lambda w, g: w + jnp.asarray(lr * tf.convert_to_tensor(g), w.dtype),
            weights, trax_grads)
        rng = rng_updater(rng)
        with tf.GradientTape() as keras_tape:
          if use_model:
            keras_outputs = keras_model(inputs)
          else:
            keras_outputs = keras_layer(inputs)
        if isinstance(keras_outputs, tuple) and len(keras_outputs) == 1:
          keras_outputs = keras_outputs[0]
        self.assertAllClose(to_tensors(trax_outputs), keras_outputs, atol=1e-5)
        keras_grads = keras_tape.gradient(keras_outputs,
                                          keras_layer.trainable_variables)
        tf.nest.map_structure(
            lambda v, g: v.assign_add(  # pylint: disable=g-long-lambda
                tf.cast(lr * tf.convert_to_tensor(g), v.dtype)),
            keras_layer.trainable_variables, keras_grads)
        self.assertAllClose(
            to_tensors(weights), read_values(keras_layer._weights),
            rtol=2e-6, atol=4.5e-4 if has_gpu() else 1e-6)
        self.assertAllClose(to_tensors(state), read_values(keras_layer._state))
        self.assertAllClose(to_tensors(rng), read_values(keras_layer._rng))
      if use_model:
        fname = os.path.join(self.get_temp_dir(), "checkpoint")
        keras_model.save(fname)
        loaded_model = tf.keras.models.load_model(fname)
        for _ in range(2):
          inputs = get_inputs()
          self.assertAllClose(keras_model(inputs), loaded_model(inputs))


if __name__ == "__main__":
  absltest.main()
