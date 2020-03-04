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

"""Tests for tf numpy mathematical methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy import extensions
from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import array_manipulation
from trax.tf_numpy.numpy import array_methods
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import math
from trax.tf_numpy.numpy import random
from trax.tf_numpy.numpy.array_creation import asarray

FLAGS = flags.FLAGS

flags.DEFINE_bool("requires_tpu", False, "Requires TPU.")


def generate_params_inputs_targets(num_examples=1000):
  params = (arrays.tensor_to_ndarray(tf.constant(5.)),
            arrays.tensor_to_ndarray(tf.constant(0.)))

  params_true = (arrays.tensor_to_ndarray(tf.constant(3.)),
                 arrays.tensor_to_ndarray(tf.constant(2.)))

  inputs = arrays.tensor_to_ndarray(tf.random.normal(shape=[num_examples]))
  noise = arrays.tensor_to_ndarray(tf.random.normal(shape=[num_examples]))
  targets = inputs * params_true[0] + params_true[1] + noise

  return params, params_true, inputs, targets


def loss_fn(params, inputs, targets):
  predicted = params[0] * inputs + params[1]
  loss = tf.reduce_mean(input_tensor=tf.square(predicted - targets))
  return arrays.tensor_to_ndarray(loss)


def train_step(params, inputs, targets, learning_rate=0.1):
  grad_fn = extensions.grad(loss_fn)
  grads = grad_fn(params, inputs, targets)
  new_w = params[0] - (grads[0] * learning_rate)
  new_b = params[1] - (grads[1] * learning_rate)

  return new_w, new_b


class ExtensionsTest(tf.test.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(ExtensionsTest, self).__init__(methodName)
    physical_devices = tf.config.experimental.list_physical_devices("CPU")
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0], [
            tf.config.experimental.VirtualDeviceConfiguration(),
            tf.config.experimental.VirtualDeviceConfiguration()
        ])
    if extensions.tpu_devices():
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
      tf.tpu.experimental.initialize_tpu_system(resolver)

  def _hasGPU(self):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    return physical_devices

  def testGrad(self):
    def f(a, b):
      return array_methods.sum(math.sqrt(math.exp(a)) + b)
    g = extensions.grad(f)
    def compare(a, b):
      with tf.GradientTape() as tape:
        tape.watch(a.data)
        r = f(a, b)
      expected = tape.gradient(r.data, a.data)
      self.assertAllEqual(expected, g(a, b))
    shape = [10]
    a = random.randn(*shape)
    b = random.randn(*shape)
    compare(a, b)

  def testGradNonArrayOutput(self):
    def f(_):
      return 1.0
    g = extensions.grad(f)
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"result .* must be an ndarray"):
      g(asarray(1.0))

  def testGradNonScalarOutput(self):
    def f(a):
      return a
    g = extensions.grad(f)
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"result .* must be a scalar"):
      g(asarray([1.0, 2.0]))
    @extensions.jit
    def g_jitted(a):
      return extensions.grad(f)(a)
    g_jitted(asarray(1.0))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"result .* must be a scalar"):
      g_jitted(asarray([1.0, 2.0]))

  def testJit(self):
    def f(a, b):
      return array_methods.sum(math.sqrt(math.exp(a)) + b)
    f_jitted = extensions.jit(f)
    shape = [10]
    a = random.randn(*shape)
    b = random.randn(*shape)
    self.assertAllClose(f(a, b), f_jitted(a, b))
    # Call again since the code path is different on second call
    self.assertAllClose(f(a, b), f_jitted(a, b))

  def testJitNoUnnecessaryTracing(self):
    def num_traces(f):
      return len(f.tf_function.
                 _list_all_concrete_functions_for_serialization())
    def check_trace_only_once(arg1, arg2):
      @extensions.jit
      def f(a):
        return a + 1
      self.assertAllEqual(0, num_traces(f))
      f(arg1)
      self.assertAllEqual(1, num_traces(f))
      f(arg2)
      self.assertAllEqual(1, num_traces(f))
    check_trace_only_once(1, 2)
    check_trace_only_once(1.1, 2.1)
    check_trace_only_once(asarray(1), asarray(2))
    check_trace_only_once(tf.convert_to_tensor(value=1),
                          tf.convert_to_tensor(value=2))

  def _testEvalOnShapes(self, transformer):
    def f(a, b):
      return array_methods.sum(math.sqrt(math.exp(a)) + b)
    f_prime = transformer(f)
    shape = [10]
    dtype = np.float16
    a = array_creation.zeros(shape=shape, dtype=dtype)
    b = array_creation.zeros(shape=shape, dtype=dtype)
    expected = f(a, b)
    got = f_prime(a, b)
    self.assertAllEqual(expected.shape, got.shape)
    self.assertAllEqual(expected.dtype, got.dtype)
    # Call again since the code path is different on second call
    got = f_prime(a, b)
    self.assertAllEqual(expected.shape, got.shape)
    self.assertAllEqual(expected.dtype, got.dtype)

  def testEvalOnShapes(self):
    def transformer(f):
      return extensions.eval_on_shapes(f)
    self._testEvalOnShapes(transformer)

  def testJitOfEvalOnShapes(self):
    """Tests that eval_on_shapes can be called within jit."""
    def transformer(f):
      @extensions.jit
      def f_prime(a, b):
        shape_dtype = extensions.eval_on_shapes(f)(a, b)
        return array_creation.zeros(shape=shape_dtype.shape,
                                    dtype=shape_dtype.dtype)
      return f_prime
    self._testEvalOnShapes(transformer)

  def testEvalOnShapesNoUnnecessaryTracing(self):
    def num_traces(f):
      return len(f._tf_function.
                 _list_all_concrete_functions_for_serialization())
    def check_trace_only_once(arg1, arg2):
      @extensions.eval_on_shapes
      def f(a):
        return a + 1
      self.assertAllEqual(0, num_traces(f))
      f(arg1)
      self.assertAllEqual(1, num_traces(f))
      f(arg2)
      self.assertAllEqual(1, num_traces(f))
    check_trace_only_once(1, 2)
    check_trace_only_once(1.1, 2.1)
    check_trace_only_once(asarray(1), asarray(2))
    check_trace_only_once(tf.convert_to_tensor(value=1),
                          tf.convert_to_tensor(value=2))

  def testConv(self):
    y = extensions.conv(np.ones([5, 320, 480, 3], dtype=np.float32),
                        np.ones([3, 4, 3, 11], dtype=np.float32), [1, 1],
                        "SAME", ("NHWC", "HWIO", "NHWC"))
    self.assertAllClose(y.shape, [5, 320, 480, 11])
    self.assertAllClose(
        y, tf.nn.conv2d(input=tf.ones([5, 320, 480, 3], dtype=tf.float32),
                        filters=tf.ones([3, 4, 3, 11], dtype=tf.float32),
                        strides=1,
                        padding="SAME"))

  def testAvgPool(self):
    y = extensions.avg_pool(np.ones([5, 320, 480, 3]), [3, 5], [2, 3], "VALID")
    self.assertAllEqual(
        y, tf.nn.pool(input=tf.ones([5, 320, 480, 3]), window_shape=[3, 5],
                      pooling_type="AVG", padding="VALID", strides=[2, 3],))

  def testMaxPool(self):
    y = extensions.max_pool(np.ones([5, 320, 480, 3]), [3, 5], [2, 3], "VALID")
    self.assertAllEqual(
        y, tf.nn.pool(input=tf.ones([5, 320, 480, 3]), window_shape=[3, 5],
                      pooling_type="MAX", padding="VALID", strides=[2, 3],))

  def testPrng(self):
    self.assertAllEqual(0, extensions.prng(123))

  def testUniform(self):
    minval = 0.43
    maxval = 3.10
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.uniform(
        None, shape, minval=minval, maxval=maxval)
    self.assertAllClose((minval + maxval) / 2.0, np.mean(outputs), atol=atol)

  def testNormal(self):
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.normal(None, shape)
    self.assertAllClose(0, np.mean(outputs), atol=atol)
    self.assertAllClose(1, np.std(outputs), atol=atol)

  def testBernoulli(self):
    mean = 0.23
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.bernoulli(None, mean, shape)
    self.assertAllClose(mean, np.mean(outputs), atol=atol)

  def testBernoulliWrongShape(self):
    mean = [0.1, 0.2]
    shape = [3]
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError, r"Incompatible shapes"):
      extensions.bernoulli(None, mean, shape)

  def testDatasetAsNumpy(self):
    arrs = extensions.dataset_as_numpy(
        [tf.constant([1, 2]), tf.constant([3, 4])])
    for a in arrs:
      self.assertIsInstance(a, arrays.ndarray)
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        r"dataset_as_numpy must be run in eager mode outside tf.function"):
      @tf.function
      def f():
        return extensions.dataset_as_numpy([tf.constant([1, 2])])
      f()

  def _get_two_devices(self, require_same_type=False):
    tpus = extensions.tpu_devices()
    if FLAGS.requires_tpu:
      if len(tpus) == 2:
        res = tpus
      else:
        raise ValueError("This test requires 2 TPU cores but %s are found"
                         % len(tpus))
    else:
      if len(tpus) == 2:
        res = tpus
      elif self._hasGPU() and not require_same_type:
        res = ("CPU:0", "GPU:0")
      else:
        res = ("CPU:0", "CPU:1")
    return res

  def testPmap(self):
    devices = self._get_two_devices()

    @functools.partial(extensions.pmap, devices=devices)
    def return_three(f):
      return f, f + 1.0, f + 2.0

    result = return_three(tf.ones((2, 20)))
    # The function returned 3 items, so we got 3 items back.
    self.assertLen(result, 3)

    # Each of the items should be a ShardedNdarray that when converted to tensor
    # should produce a tensor of shape (2, 20)
    converted = tf.nest.map_structure(tf.convert_to_tensor, result)

    self.assertLen(result, 3)

    self.assertAllEqual(converted[0].shape, converted[1].shape)
    self.assertAllEqual(converted[0].shape, converted[2].shape)

    self.assertAllEqual(converted[0], tf.ones((2, 20)))
    self.assertAllEqual(converted[1], 1 + tf.ones((2, 20)))
    self.assertAllEqual(converted[2], 2 + tf.ones((2, 20)))

    @functools.partial(extensions.pmap, devices=devices)
    def return_one(f):
      return f + 2.0

    result = return_one(tf.ones((2, 20)))

    # Only a single item is returned, so we can convert it directly.
    converted = tf.convert_to_tensor(value=result)
    self.assertAllEqual(converted, 2 + tf.ones((2, 20)))

    @functools.partial(extensions.pmap, devices=devices)
    def return_list(f):
      return [f + 2.0]

    result = return_list(tf.ones((2, 20)))

    # A singleton list is returned.
    self.assertLen(result, 1)
    converted = tf.convert_to_tensor(value=result[0])
    self.assertAllEqual(converted, 2 + tf.ones((2, 20)))

  def testGradSimpleModel(self):
    params, params_true, inputs, targets = generate_params_inputs_targets()

    for _ in range(50):
      params = train_step(params, inputs, targets)

    # This is not trained super well, but it usually gets "close".
    self.assertAllClose(params[0], params_true[0], atol=1e-1)
    self.assertAllClose(params[1], params_true[1], atol=1e-1)

  # NOTE: Compare to testGradSimpleModel to see the differences when pmapping.
  def testPmapSimpleModel(self):
    devices = self._get_two_devices(require_same_type=True)
    n_devices = len(devices)

    params, params_true, inputs, targets = generate_params_inputs_targets()

    def _train_and_reduce(params, inputs, targets, learning_rate=0.1):
      new_w, new_b = train_step(params, inputs, targets, learning_rate)

      return (extensions.psum(new_w) / n_devices,
              extensions.psum(new_b) / n_devices)

    train_step_pmapped = extensions.pmap(
        _train_and_reduce, devices=devices)

    def replicate(x, num_devices=2):
      return array_manipulation.broadcast_to(x, (num_devices,) + x.shape)

    params = tf.nest.map_structure(replicate, params)

    def reshape(x, num_devices=2):
      x_shape = list(x.shape)
      batch_size = x_shape[0]
      batch_size_per_device = batch_size // num_devices

      # New shape.
      new_shape_prefix = [num_devices, batch_size_per_device]
      return array_methods.reshape(x, new_shape_prefix + x_shape[1:])

    inputs = tf.nest.map_structure(reshape, inputs)
    targets = tf.nest.map_structure(reshape, targets)

    for _ in range(50):
      params = train_step_pmapped(params, inputs, targets)

    # PMAP returns sharded tensors.

    # Since the inputs are identical, the returned tensors should be identical
    self.assertAllClose(params[0][0], params[0][1])
    self.assertAllClose(params[1][0], params[1][1])

    # This is not trained super well, but it usually gets "close".
    self.assertAllClose(params[0][0], params_true[0], atol=1e-1)
    self.assertAllClose(params[1][0], params_true[1], atol=1e-1)

  def testPsum(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(f):
      return extensions.psum(f)

    data = array_creation.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_sum, devices=devices)
    result = pmapped(data)

    self.assertAllClose(result[0], 4)
    self.assertAllClose(result[1], 4)

  def testPmean(self):
    if extensions.tpu_devices():
      self.skipTest("pmean for TPU is not supported yet")
    devices = self._get_two_devices(require_same_type=True)

    def reduce_mean(f):
      return extensions.pmean(f)

    data = array_creation.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_mean, devices=devices)
    result = pmapped(data)

    self.assertAllClose(result[0], 2)
    self.assertAllClose(result[1], 2)

  def testAxisName(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(f):
      return extensions.psum(f, axis_name="foo")

    data = array_creation.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_sum, axis_name="foo", devices=devices)
    pmapped(data)

  def testWrongAxisName(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(f):
      return extensions.psum(f, axis_name="bar")

    data = array_creation.asarray(tf.convert_to_tensor(value=[1, 3]))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"axis_name (.*) is not equal to that of the surrounding"):
      pmapped = extensions.pmap(reduce_sum, axis_name="foo", devices=devices)
      pmapped(data)

  def testNoNestedPmap(self):
    devices = self._get_two_devices(require_same_type=True)

    def f(x):
      return x + 1.0

    data = array_creation.asarray(tf.convert_to_tensor(value=[1, 3]))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"Nested pmap is not supported"):
      f = extensions.pmap(f, devices=devices)
      f = extensions.pmap(f, devices=devices)
      f(data)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
