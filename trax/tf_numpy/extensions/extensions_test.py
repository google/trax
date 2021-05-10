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

"""Tests for tf numpy mathematical methods."""
import functools
import itertools

from absl import flags
from absl.testing import parameterized

from jax import lax
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy import extensions
import trax.tf_numpy.numpy as tf_np


FLAGS = flags.FLAGS

flags.DEFINE_bool("requires_tpu", False, "Requires TPU.")


def generate_params_inputs_targets(num_examples=1000):
  params = (tf_np.asarray(tf.constant(5.)), tf_np.asarray(tf.constant(0.)))

  params_true = (tf_np.asarray(tf.constant(3.)), tf_np.asarray(tf.constant(2.)))

  inputs = tf_np.asarray(tf.random.normal(shape=[num_examples]))
  noise = tf_np.asarray(tf.random.normal(shape=[num_examples]))
  targets = inputs * params_true[0] + params_true[1] + noise

  return params, params_true, inputs, targets


def loss_fn(params, inputs, targets):
  predicted = params[0] * inputs + params[1]
  loss = tf.reduce_mean(input_tensor=tf.square(predicted - targets))
  return tf_np.asarray(loss)


def train_step(params, inputs, targets, learning_rate=0.1):
  grad_fn = extensions.grad(loss_fn)
  grads = grad_fn(params, inputs, targets)
  new_w = params[0] - (grads[0] * learning_rate)
  new_b = params[1] - (grads[1] * learning_rate)

  return new_w, new_b


def uniform(rng, shape, dtype):
  if np.issubdtype(dtype, np.integer):
    minval = None
  else:
    minval = 0
  return tf_np.asarray(rng.uniform(shape=shape, dtype=dtype, minval=minval))


def to_np(a):
  return tf.nest.map_structure(tf_np.asarray, a)


def to_tf_fn(f):
  return lambda *args: f(*to_np(args))


def scan_reference(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    (carry, y) = f(carry, x)
    ys.append(tf_np.reshape(y, (1,) + y.shape))
  ys = tf_np.concatenate(ys, 0)
  return carry, ys


def spec(*args):
  return tf.TensorSpec(args, tf.float32)


class ExtensionsTest(tf.test.TestCase, parameterized.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super().__init__(methodName)
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

  def testCustomGrad(self):
    """Test for custom_grad."""
    x_shape = (tf.TensorShape([10]), tf.TensorShape([1, 10]))
    y_shape = (tf.TensorShape([]))
    dtype = np.float32
    scale1 = 5.0
    scale2 = 6.0

    def fwd(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    @extensions.custom_grad
    def f(a, b):
      y = fwd(a, b)

      def vjp(dy):
        return dy * scale1 * a, dy * scale2 * b

      return y, vjp

    rng = tf.random.Generator.from_seed(1234)
    x, dy = tf.nest.map_structure(lambda shape: uniform(rng, shape, dtype),
                                  [x_shape, y_shape])
    expected_y = fwd(*x)
    expected_dx = (dy * scale1 * x[0], dy * scale2 * x[1])
    y, vjp = extensions.vjp(f, *x)
    dx = vjp(dy)
    self.assertAllClose(expected_y, y)
    self.assertAllClose(expected_dx, dx)

  @parameterized.named_parameters([
      (  # pylint: disable=g-complex-comprehension
          ("_%s_%s_%s" % (decorator_id, x_struct, y_struct)).replace(
              " ", "").replace("None", ""), decorator, x_struct, y_struct)
      for y_struct in [[None, ()], (None, (), [], (None, ((), None)))]
      for x_struct in [(None, ()), (((), ()), [None, None], [], (None, ()))]
      for decorator_id, decorator in enumerate([lambda f: f, extensions.jit])
  ])
  def testCustomGradStructure(self, decorator, x_struct, y_struct):
    """Tests that custom_grad can handle structured inputs/outputs."""

    def zeros(x):
      return tf.nest.map_structure(lambda _: tf_np.zeros([], np.float32), x)

    def get_struct(x):
      return tf.nest.map_structure(lambda _: None, x)

    @extensions.custom_grad
    def f(*x):
      del x

      def vjp(dy):
        self.assertEqual(y_struct, get_struct(dy))
        return zeros(x_struct)

      return zeros(y_struct), vjp

    x, dy = zeros([x_struct, y_struct])

    @decorator
    def run(x, dy):
      y, vjp = extensions.vjp(f, *x)
      dx = vjp(dy)
      return dx, y

    dx, y = run(x, dy)
    self.assertEqual(x_struct, get_struct(dx))
    self.assertEqual(y_struct, get_struct(y))

  @parameterized.named_parameters([
      ("_%s" % has_aux, has_aux) for has_aux in [True, False]
  ])
  def testVjp(self, has_aux):
    x_shape = (tf.TensorShape([10]), tf.TensorShape([1, 10]))
    y_shape = (tf.TensorShape([]))
    dtype = np.float32

    def f(a, b):
      y = tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)
      if has_aux:
        return y, tf_np.asarray(1)
      else:
        return y

    rng = tf.random.Generator.from_seed(1234)
    x, dy_list = tf.nest.map_structure(lambda shape: uniform(rng, shape, dtype),
                                       [x_shape, [y_shape] * 2])
    tf_x = x
    outputs = extensions.vjp(f, *x, has_aux=has_aux)
    if has_aux:
      y, vjp, aux = outputs
    else:
      y, vjp = outputs
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(tf_x)
      outputs = f(*x)
      if has_aux:
        expected_y, expected_aux = outputs
        self.assertAllClose(expected_aux, aux)
      else:
        expected_y = outputs
    self.assertAllClose(expected_y, y)
    for dy in dy_list:
      expected_dx = tape.gradient(
          expected_y, tf_x, output_gradients=dy)
      self.assertAllClose(expected_dx, vjp(dy))

  def testGrad(self):

    def f(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    g = extensions.grad(f)

    def compare(a, b):
      with tf.GradientTape() as tape:
        tape.watch(a)
        r = f(a, b)
      expected = tape.gradient(r, a)
      self.assertAllEqual(expected, g(a, b))

    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)
    compare(a, b)

  def testGradNonArrayOutput(self):

    def f(_):
      return 1.0

    g = extensions.grad(f)
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r"result .* must be an ndarray"):
      g(tf_np.asarray(1.0))

  def testGradNonScalarOutput(self):

    def f(a):
      return a

    g = extensions.grad(f)
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r"result .* must be a scalar"):
      g(tf_np.asarray([1.0, 2.0]))

    @extensions.jit
    def g_jitted(a):
      return extensions.grad(f)(a)

    g_jitted(tf_np.asarray(1.0))
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r"result .* must be a scalar"):
      g_jitted(tf_np.asarray([1.0, 2.0]))

  def testJit(self):

    def f(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    f_jitted = extensions.jit(f)
    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)
    self.assertAllClose(f(a, b), f_jitted(a, b))
    # Call again since the code path is different on second call
    self.assertAllClose(f(a, b), f_jitted(a, b))

  def testJitNoUnnecessaryTracing(self):

    def num_traces(f):
      return len(f.tf_function._list_all_concrete_functions_for_serialization())

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
    check_trace_only_once(tf_np.asarray(1), tf_np.asarray(2))
    check_trace_only_once(
        tf.convert_to_tensor(value=1), tf.convert_to_tensor(value=2))

  def _testEvalOnShapes(self, transformer, allow_static_outputs):

    # A class that's not convertable to tensor
    class Thing:

      def __init__(self, value):
        self.value = value

    def f(a, b, reverse=False):
      res = tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)
      res = (res, 10)
      if allow_static_outputs:
        res = res + (Thing(20),)
      if reverse:
        res = tuple(reversed(res))
      return res

    f_prime = transformer(
        f, static_argnums=(2,), allow_static_outputs=allow_static_outputs)
    shape = [10]
    dtype = np.float16
    a = tf_np.zeros(shape=shape, dtype=dtype)
    b = tf_np.zeros(shape=shape, dtype=dtype)
    expected, *_ = f(a, b)
    got = f_prime(a, b)
    def check(got):
      self.assertIsInstance(got[0], (tf.TensorSpec, tf_np.ndarray))
      self.assertAllEqual(expected.shape, got[0].shape)
      self.assertAllEqual(expected.dtype, got[0].dtype)
      if allow_static_outputs:
        self.assertIsInstance(got[1], int)
        self.assertEqual(10, got[1])
        self.assertIsInstance(got[2], Thing)
        self.assertEqual(20, got[2].value)
      else:
        self.assertIsInstance(got[1], (tf.TensorSpec, tf_np.ndarray))
        self.assertAllEqual((), got[1].shape)
    check(got)
    # Call again since the code path is different on second call
    got = f_prime(a, b)
    check(got)
    # Retrace and check again
    got = f_prime(a, b, True)
    check(tuple(reversed(got)))
    got = f_prime(a, b, True)
    check(tuple(reversed(got)))

  @parameterized.named_parameters(("_%s" % b, b) for b in [False, True])
  def testEvalOnShapes(self, allow_static_outputs):
    self._testEvalOnShapes(extensions.eval_on_shapes, allow_static_outputs)

  def testEvalOnShapesNested(self):
    transformer = functools.partial(extensions.eval_on_shapes,
                                    allow_static_outputs=True)
    @transformer
    def outer():
      @transformer
      def inner():
        return 1
      return inner() + 2
    r = outer()
    self.assertIsInstance(r, int)
    self.assertEqual(3, r)

  def testJitOfEvalOnShapes(self):
    """Tests that eval_on_shapes can be called within jit."""

    def transformer(f, **kwargs):
      def f_prime(*args):
        res = extensions.eval_on_shapes(f, **kwargs)(*args)
        return tf.nest.map_structure(
            lambda x: tf_np.zeros(x.shape, x.dtype), res)
      return extensions.jit(f_prime, kwargs.get("static_argnums", ()))

    self._testEvalOnShapes(transformer, False)

  def testEvalOnShapesNoUnnecessaryTracing(self):

    def num_traces(f):
      return len(
          f._tf_function._list_all_concrete_functions_for_serialization())

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
    check_trace_only_once(tf_np.asarray(1), tf_np.asarray(2))
    check_trace_only_once(
        tf.convert_to_tensor(value=1), tf.convert_to_tensor(value=2))

  @parameterized.parameters(
      {
          "lhs_np": np.ones((5, 3)),
          "rhs_np": np.ones((3, 2)),
          "dims": (((1,), (0,)), ((), ()))
      },
      {
          "lhs_np": np.ones((5, 3)),
          "rhs_np": np.ones((5, 3)),
          "dims": (((0, 1), (0, 1)), ((), ()))
      },
      {
          "lhs_np": np.ones((5, 3, 2)),
          "rhs_np": np.ones((2, 3, 2)),
          "dims": (((1, 2), (1, 0)), ((), ()))
      },
      {
          "lhs_np": np.ones((6, 5, 3)),
          "rhs_np": np.ones((6, 3, 2)),
          "dims": (((2,), (1,)), ((0,), (0,)))
      },
      {
          "lhs_np": np.ones((6, 3, 5)),
          "rhs_np": np.ones((6, 3, 2)),
          "dims": (((1,), (1,)), ((0,), (0,)))
      },
      {
          "lhs_np": np.ones((5, 3, 2, 2)),
          "rhs_np": np.ones((5, 2, 2, 6)),
          "dims": (((2, 3), (1, 2)), ((0,), (0,)))
      },
      {
          "lhs_np": np.ones((2, 2, 5, 3)),
          "rhs_np": np.ones((2, 2, 3, 2)),
          "dims": (((3,), (2,)), ((0, 1), (0, 1)))
      },
      {
          "lhs_np": np.ones((2, 2, 5, 2)),
          "rhs_np": np.ones((2, 2, 3, 2)),
          "dims": (((3,), (1,)), ((0,), (0,)))
      },
      {
          "lhs_np": np.ones((2, 2, 5, 3, 3)),
          "rhs_np": np.ones((2, 3, 2, 3, 2)),
          "dims": (((4,), (1,)), ((0,), (0,)))
      },
  )
  def test_tf_dot_general(self, lhs_np, rhs_np, dims):
    ans = lax.dot_general(lhs_np, rhs_np, dims)
    result = extensions.tf_dot_general(lhs_np, rhs_np, dims)
    self.assertAllClose(result, np.array(ans))

  @parameterized.named_parameters([
      ("_lhs_shape={}_rhs_shape={}_strides={}_padding={}"  # pylint: disable=g-complex-comprehension
       "_lhs_dilation={}_rhs_dilation={}"
       "_feature_group_count={}_batch_group_count={}_dims={}"
       "_perms={}".format(lhs_shape, rhs_shape,
                          strides, padding, lhs_dilation, rhs_dilation,
                          feature_group_count, batch_group_count, ",".join(
                              dimension_numbers), perms),
       lhs_shape, rhs_shape, strides, padding, lhs_dilation, rhs_dilation,
       feature_group_count, batch_group_count, dimension_numbers, perms)
      for batch_group_count, feature_group_count in [(1, 1)]
      for lhs_shape, rhs_shape in [
          ((b * batch_group_count, i * feature_group_count, 9, w),
           (j * feature_group_count * batch_group_count, i, 4, 5))
          for w in [0, 10]
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for strides in [(1, 1), (2, 1)]
      for padding in ["SAME"]
      for lhs_dilation, rhs_dilation in [
          (None, (1, 1))
      ]
      for dimension_numbers, perms in [
          (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0]))
      ]])
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, strides,
                             padding, lhs_dilation, rhs_dilation,
                             feature_group_count, batch_group_count,
                             dimension_numbers, perms):
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    lhs = np.transpose(np.ones(lhs_shape), lhs_perm)
    rhs = np.transpose(np.ones(rhs_shape), rhs_perm)

    jax_conv = lax.conv_general_dilated(lhs, rhs, strides, padding,
                                        lhs_dilation, rhs_dilation,
                                        dimension_numbers,
                                        feature_group_count,
                                        batch_group_count)

    tf_conv = extensions.tf_conv_general_dilated(lhs, rhs, strides,
                                                 padding, None,
                                                 lhs_dilation, rhs_dilation,
                                                 dimension_numbers,
                                                 feature_group_count,
                                                 batch_group_count)

    self.assertAllClose(tf_conv, tf_np.asarray(jax_conv))

  def testConv(self):
    y = extensions.conv(
        np.ones([5, 320, 480, 3], dtype=np.float32),
        np.ones([3, 4, 3, 11], dtype=np.float32), [1, 1], "SAME",
        ("NHWC", "HWIO", "NHWC"))
    self.assertAllClose(y.shape, [5, 320, 480, 11])
    self.assertAllClose(
        y,
        tf.nn.conv2d(
            input=tf.ones([5, 320, 480, 3], dtype=tf.float32),
            filters=tf.ones([3, 4, 3, 11], dtype=tf.float32),
            strides=1,
            padding="SAME"))

  def testAvgPool(self):
    y = extensions.avg_pool(np.ones([5, 320, 480, 3]), [3, 5], [2, 3], "VALID")
    self.assertAllEqual(
        y,
        tf.nn.pool(
            input=tf.ones([5, 320, 480, 3]),
            window_shape=[3, 5],
            pooling_type="AVG",
            padding="VALID",
            strides=[2, 3],
        ))

  def testMaxPool(self):
    y = extensions.max_pool(np.ones([5, 320, 480, 3]), [3, 5], [2, 3], "VALID")
    self.assertAllEqual(
        y,
        tf.nn.pool(
            input=tf.ones([5, 320, 480, 3]),
            window_shape=[3, 5],
            pooling_type="MAX",
            padding="VALID",
            strides=[2, 3],
        ))

  def assertDTypesEqual(self, a, b):
    get_dtype = lambda t: t.dtype
    self.assertEqual(tf.nest.map_structure(get_dtype, a),
                     tf.nest.map_structure(get_dtype, b))

  @parameterized.named_parameters(
      (f"_{jit_scan}_{jit_f}", jit_scan, jit_f)  # pylint: disable=g-complex-comprehension
      for jit_f in [False, True]
      for jit_scan in ["no", "no_xla", "xla_forced_compile"])
  def testScanImpl(self, jit_scan, jit_f):
    rng = np.random.RandomState(0)

    d = rng.randn(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = tf_np.cos(tf_np.sum(tf_np.sin(a)) + tf_np.sum(tf_np.cos(c)) +
                    tf_np.sum(tf_np.tan(d)))
      c = tf_np.sin(c * b)
      assert b.shape == ()  # pylint: disable=g-explicit-bool-comparison
      return c, b

    if jit_f:
      f = extensions.jit(f)

    if jit_scan == "no_xla":
      scan = extensions.jit(extensions.scan, static_argnums=(0,))
    elif jit_scan == "xla_forced_compile":
      scan = extensions.jit(extensions.scan, static_argnums=(0,),
                            xla_forced_compile=True)
    else:
      scan = extensions.scan

    xs = rng.randn(5, 3)
    c = rng.randn(4)

    ans = scan(f, c, xs)
    expected = scan_reference(f, c, xs)
    if jit_scan == "xla_forced_compile":
      # xla.compile doesn't preserve list-vs-tuple properly for the outputs, so
      # we canonicalize them to lists here.
      expected = list(expected)
      ans = list(ans)
    self.assertDTypesEqual(expected, ans)
    self.assertAllClose(expected, ans)

  def testScanStruct(self):
    rng = np.random.RandomState(0)

    d = rng.randn(2)
    def f(c_g_i, a_e_h):
      c_g, i = c_g_i
      c, g = c_g
      a, e_h = a_e_h
      e, h = e_h
      assert a.shape == (3,)
      assert e.shape == ()  # pylint: disable=g-explicit-bool-comparison
      assert c.shape == (4,)
      assert g.shape == (2,)
      assert i is None
      assert h is None
      b = tf_np.cos(tf_np.sum(tf_np.sin(a)) + tf_np.sum(tf_np.cos(c)) +
                    tf_np.sum(tf_np.tan(d)))
      f = tf_np.cos(a)
      c = tf_np.sin(c * b)
      g = tf_np.sin(g * b)
      assert b.shape == ()  # pylint: disable=g-explicit-bool-comparison
      assert f.shape == (3,)
      return [(c, g), i], (b, [f, h])

    xs = (rng.randn(5, 3), [rng.randn(5), None])
    init = [(rng.randn(4), rng.randn(2)), None]

    c_g_i, b_f_h = extensions.scan(f, init, xs)
    self.assertIsInstance(c_g_i, list)
    self.assertIsInstance(b_f_h, tuple)
    c_g, i = c_g_i
    c, g = c_g
    self.assertIsInstance(c_g, tuple)
    self.assertEqual((4,), c.shape)
    self.assertEqual((2,), g.shape)
    self.assertIsNone(i)
    b, f_h = b_f_h
    f, h = f_h
    self.assertIsInstance(f_h, list)
    self.assertEqual((5,), b.shape)
    self.assertEqual((5, 3), f.shape)
    self.assertIsNone(h)

  @parameterized.named_parameters(
      (f"_{jit_grad}_{jit_scan}_{jit_f}", jit_grad, jit_scan, jit_f)  # pylint: disable=g-complex-comprehension
      for jit_f in [False, True]
      for jit_scan in ["no", "no_xla", "xla_forced_compile"]
      for jit_grad in ["no", "no_xla", "xla_forced_compile"])
  def testScanGrad(self, jit_grad, jit_scan, jit_f):
    rng = np.random.RandomState(0)

    d = rng.randn(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = (tf_np.sum(tf_np.sin(a)) + tf_np.sum(tf_np.sin(c)) +
           tf_np.sum(tf_np.sin(d)))
      c = tf_np.sin(c * b)
      assert b.shape == ()  # pylint: disable=g-explicit-bool-comparison
      return c, b

    if jit_f:
      f = extensions.jit(f)

    if jit_scan == "no_xla":
      scan = extensions.jit(extensions.scan, static_argnums=(0,))
    elif jit_scan == "xla_forced_compile":
      # TODO(b/187107596): Remove `skipTest`
      self.skipTest(
          "Taking gradients of `jit(scan, experimental_compile=True)` triggers "
          "'Support for TensorList crossing the XLA/TF boundary is not "
          "implemented' error")
      # `xla_forced_compile=True` doesn't support gradients, so we use
      # `experimental_compile=True`.
      scan = extensions.jit(extensions.scan, static_argnums=(0,),
                            experimental_compile=True)
    else:
      scan = extensions.scan

    xs = tf_np.asarray(rng.randn(5, 3))
    c = tf_np.asarray(rng.randn(4))

    def losses(scan, c, xs):
      c, ys = scan(f, c, xs)
      return tf_np.concatenate(tf.nest.flatten(tf.nest.map_structure(
          lambda a: tf_np.reshape(a, [-1]), (c, ys))))
    def loss(scan, c, xs):
      return tf_np.sum(losses(scan, c, xs))

    def grad_origin(c, xs):
      return extensions.grad(functools.partial(loss, scan))(c, xs)

    if jit_grad == "no_xla":
      grad_jit = extensions.jit(grad_origin)
    elif jit_grad == "xla_forced_compile":
      grad_jit = extensions.jit(grad_origin, xla_forced_compile=True)
    else:
      grad_jit = grad_origin

    ans = grad_jit(c, xs)
    expected = extensions.grad(functools.partial(loss, scan_reference))(c, xs)
    self.assertDTypesEqual(expected, ans)
    self.assertAllClose(expected, ans)

    theoretical, numerical = tf.test.compute_gradient(
        to_tf_fn(functools.partial(losses, scan)), (c, xs))
    self.assertAllClose(theoretical, numerical, atol=1e-3, rtol=3e-4)

  @parameterized.named_parameters(
      (f"_{i}", *args)  # pylint: disable=g-complex-comprehension
      for i, args in enumerate([
          (lambda c, x: (c + 1, tf_np.sum(c + x, 0)),
           [spec(2), spec(4, 3, 2)], [spec(2), spec(4, 2)]),
          (lambda c, x: (c + 1, tf_np.sum(c + x, 0)),
           [spec(2), spec(0, 3, 2), 0], [spec(2), spec(0, 2)]),
      ]))
  def testScanShape(self, f, inputs, expected_outputs):
    outputs = extensions.eval_on_shapes(
        functools.partial(extensions.scan, f), static_argnums=(2,))(*inputs)
    self.assertAllEqual(expected_outputs, outputs)

  def testMap(self):
    shape = [2, 3]
    dtype = tf_np.int32
    xs1 = tf_np.zeros(shape, dtype)
    xs2 = tf_np.ones(shape, dtype)
    ys_expected = [xs2 + 10, xs1 + 20]
    def f(x):
      self.assertIsInstance(x, tuple)
      for a in x:
        self.assertEqual(a.shape, shape[1:])
      x1, x2 = x
      return [x2 + 10, x1 + 20]
    ys = extensions.tf_map(f, (xs1, xs2))
    self.assertIsInstance(ys, list)
    self.assertAllClose(ys, ys_expected)

  def testPrng(self):
    self.assertAllEqual(tf_np.asarray(123, np.int64), extensions.prng(123))

  def testUniform(self):
    minval = 0.43
    maxval = 3.10
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.uniform(123, shape, minval=minval, maxval=maxval)
    self.assertAllClose((minval + maxval) / 2.0, np.mean(outputs), atol=atol)

  def testNormal(self):
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.normal(123, shape)
    self.assertAllClose(0, np.mean(outputs), atol=atol)
    self.assertAllClose(1, np.std(outputs), atol=atol)

  def testBernoulli(self):
    mean = 0.23
    shape = [13, 34, 29]
    atol = 0.1
    outputs = extensions.bernoulli(123, mean, shape)
    self.assertAllClose(mean, np.mean(outputs), atol=atol)

  def testBernoulliWrongShape(self):
    mean = [0.1, 0.2]
    shape = [3]
    with self.assertRaisesIncompatibleShapesError():
      extensions.bernoulli(123, mean, shape)

  def testDatasetAsNumpy(self):
    arrs = extensions.dataset_as_numpy(
        [tf.constant([1, 2]), tf.constant([3, 4])])
    for a in arrs:
      self.assertIsInstance(a, tf_np.ndarray)
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
        raise ValueError("This test requires 2 TPU cores but %s are found" %
                         len(tpus))
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

    train_step_pmapped = extensions.pmap(_train_and_reduce, devices=devices)

    def replicate(x, num_devices=2):
      return tf_np.broadcast_to(x, (num_devices,) + x.shape)

    params = tf.nest.map_structure(replicate, params)

    def reshape(x, num_devices=2):
      x_shape = list(x.shape)
      batch_size = x_shape[0]
      batch_size_per_device = batch_size // num_devices

      # New shape.
      new_shape_prefix = [num_devices, batch_size_per_device]
      return tf_np.reshape(x, new_shape_prefix + x_shape[1:])

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

    data = tf_np.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_sum, devices=devices)
    result = pmapped(data)

    self.assertAllClose(result[0], 4)
    self.assertAllClose(result[1], 4)

  def testPsumStruct(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(a):
      a = extensions.psum(a)
      tf.nest.map_structure(
          lambda x: self.assertIsInstance(x, tf_np.ndarray), a)
      return a

    data = [tf_np.asarray([1, 3]), tf_np.asarray([2, 4], np.int64)]
    pmapped = extensions.pmap(reduce_sum, devices=devices)
    result = pmapped(data)

    self.assertIsInstance(result[0][0], tf_np.ndarray)
    self.assertIsInstance(result[0][1], tf_np.ndarray)
    self.assertIsInstance(result[1][0], tf_np.ndarray)
    self.assertIsInstance(result[1][1], tf_np.ndarray)
    self.assertAllClose(result[0][0], 4)
    self.assertAllClose(result[0][1], 4)
    self.assertAllClose(result[1][0], 6)
    self.assertAllClose(result[1][1], 6)

  def testPmean(self):
    if extensions.tpu_devices():
      self.skipTest("pmean for TPU is not supported yet")
    devices = self._get_two_devices(require_same_type=True)

    def reduce_mean(f):
      return extensions.pmean(f)

    data = tf_np.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_mean, devices=devices)
    result = pmapped(data)

    self.assertAllClose(result[0], 2)
    self.assertAllClose(result[1], 2)

  def testAxisName(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(f):
      return extensions.psum(f, axis_name="foo")

    data = tf_np.asarray(tf.convert_to_tensor(value=[1, 3]))
    pmapped = extensions.pmap(reduce_sum, axis_name="foo", devices=devices)
    pmapped(data)

  def testWrongAxisName(self):
    devices = self._get_two_devices(require_same_type=True)

    def reduce_sum(f):
      return extensions.psum(f, axis_name="bar")

    data = tf_np.asarray(tf.convert_to_tensor(value=[1, 3]))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"axis_name (.*) is not equal to that of the surrounding"):
      pmapped = extensions.pmap(reduce_sum, axis_name="foo", devices=devices)
      pmapped(data)

  def testNoNestedPmap(self):
    devices = self._get_two_devices(require_same_type=True)

    def f(x):
      return x + 1.0

    data = tf_np.asarray(tf.convert_to_tensor(value=[1, 3]))
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r"Nested pmap is not supported"):
      f = extensions.pmap(f, devices=devices)
      f = extensions.pmap(f, devices=devices)
      f(data)

  def testVmap(self):
    fn1 = extensions.vmap(lambda z: z * z)

    x = tf_np.arange(10)
    self.assertAllClose(x * x, fn1(x))

    y = tf.range(10)
    np_y = tf_np.asarray(y)
    output = fn1(y)
    self.assertIsInstance(output, tf_np.ndarray)
    self.assertAllClose(np_y * np_y, output)

    fn2 = extensions.vmap(lambda x, y: x + y)
    x = tf_np.random.randn(10, 3)
    y = tf_np.random.randn(10, 2, 3)
    self.assertAllClose(tf_np.expand_dims(x, 1) + y, fn2(x, y))

  def testRemat(self):
    def f(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    f_remat = extensions.remat(f)

    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)

    actual = extensions.grad(f_remat)(a, b)
    expected = extensions.grad(f)(a, b)
    self.assertAllClose(actual, expected)

  def testRematLambdaFunction(self):
    f = lambda a, b: tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)
    f_remat = extensions.remat(f)

    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)

    actual = extensions.grad(f_remat)(a, b)
    expected = extensions.grad(f)(a, b)
    self.assertAllClose(actual, expected)

  def testRematJit(self):
    def f(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    f_remat = extensions.remat(f)

    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)

    actual = extensions.jit(extensions.grad(f_remat))(a, b)
    expected = extensions.jit(extensions.grad(f))(a, b)
    self.assertAllClose(actual, expected)

  def testRematJitXla(self):
    def f(a, b):
      return tf_np.sum(tf_np.sqrt(tf_np.exp(a)) + b)

    f_remat = extensions.remat(f)

    shape = [10]
    a = tf_np.random.randn(*shape)
    b = tf_np.random.randn(*shape)

    actual = extensions.jit(
        extensions.grad(f_remat), xla_forced_compile=True)(a, b)
    expected = extensions.jit(extensions.grad(f), xla_forced_compile=True)(a, b)
    self.assertAllClose(actual, expected)

    actual = extensions.jit(
        extensions.grad(f_remat), experimental_compile=True)(a, b)
    expected = extensions.jit(
        extensions.grad(f), experimental_compile=True)(a, b)
    self.assertAllClose(actual, expected)

  def testStaticStopGradient(self):
    self.assertEqual(extensions.stop_gradient(5.), 5.)
    self.assertEqual(type(extensions.stop_gradient(5.)), type(5.))

    self.assertEqual(extensions.stop_gradient(tf_np.asarray(5.)), 5.)
    self.assertNotEqual(
        type(extensions.stop_gradient(tf_np.asarray(5.))), type(5.))


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
