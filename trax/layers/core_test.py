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
"""Tests for core layers."""

from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl


class DenseTest(absltest.TestCase):
  """Test Dense layer per se and as a key example of trainable layers."""

  def test_call_before_init_raises_error(self):
    layer = tl.Dense(5)
    x = np.array([1, 2, 3])

    # Without init, layer lacks the weights it needs for forward computation.
    with self.assertRaises(tl.LayerError):
      _ = layer(x)

  def test_call_uses_and_caches_supplied_weights(self):
    layer = tl.Dense(4)
    x = np.array([2, 3])

    # Weights from random initialization are cached in the layer.
    _, _ = layer.init(shapes.signature(x))
    w_init, b_init = layer.weights

    # Call the layer with externally specified weights.
    w = np.array([[10000, 20000, 30000, 40000], [100, 200, 100, 200]])
    b = np.array([9, 8, 7, 6])
    y = layer(x, weights=(w, b))

    # Using weights keyword arg overrides any previous cached weights ...
    self.assertEqual(y.tolist(), [20309, 40608, 60307, 80606])
    self.assertNotEqual(w.tolist(), w_init.tolist())
    self.assertNotEqual(b.tolist(), b_init.tolist())

    # ... and the provided values become the new cached weights.
    w_cached, b_cached = layer.weights
    self.assertEqual(w.tolist(), w_cached.tolist())
    self.assertEqual(b.tolist(), b_cached.tolist())

  def test_separate_instances_have_separate_weights(self):
    # Two dense layer instances: each will get its own initial weights (w, b).
    model = tl.Serial(tl.Dense(5), tl.Dense(5))

    sample_input = np.array([1, 2, 3, 4, 5])
    _, _ = model.init(shapes.signature(sample_input))
    weights_0 = model.sublayers[0].weights
    weights_1 = model.sublayers[1].weights

    w0, b0 = weights_0
    w1, b1 = weights_1
    self.assertNotEqual(w0.tolist(), w1.tolist())
    self.assertNotEqual(b0.tolist(), b1.tolist())

  def test_shared_instance_means_shared_weights(self):
    # Same dense layer instance in two places --> shared weights.
    layer = tl.Dense(5)
    model = tl.Serial(layer, layer)
    sample_input = np.array([1, 2, 3, 4, 5])
    weights, _ = model.init(shapes.signature(sample_input))
    self.assertIs(weights[1], tl.GET_WEIGHTS_FROM_CACHE)

  def test_call_no_bias(self):
    layer = tl.Dense(4, use_bias=False)
    x = np.array([2, 5, 3])
    _, _ = layer.init(shapes.signature(x))

    w = np.array([[100, 200, 300, 400], [10, 10, 10, 10], [1, 2, 1, 2]])
    y = layer(x, weights=w)
    self.assertEqual(y.tolist(), [253, 456, 653, 856])

  def test_new_weights_use_bias(self):
    layer = tl.Dense(4)
    x = np.array([1, 2])
    _, _ = layer.init(shapes.signature(x))
    self.assertLen(layer.weights, 2)
    self.assertEqual(layer.weights[0].shape, (2, 4))
    self.assertEqual(layer.weights[1].shape, (4,))

  def test_new_weights_no_bias(self):
    layer = tl.Dense(4, use_bias=False)
    x = np.array([1, 2])
    _, _ = layer.init(shapes.signature(x))
    self.assertEqual(layer.weights.shape, (2, 4))

  def test_init_twice_weights_same_shape(self):
    layer = tl.Dense(4, use_bias=False)
    x = np.array([1, 2])
    w1, _ = layer.init(shapes.signature(x))
    w2, _ = layer.init(shapes.signature(x))
    self.assertEqual(w1.shape, (2, 4))
    self.assertEqual(w2.shape, (2, 4))


class EmbeddingTest(absltest.TestCase):

  def test_forward(self):
    layer = tl.Embedding(10, 3)  # vocab_size=10, d_feature=3
    _, _ = layer.init(None)  # Embedding init doesn't use input signature.
    x = np.array([2, 3, 5, 3, 2])
    y = layer(x)
    self.assertEqual(y.shape, (5, 3))

    # For distinct in-domain token ids, resulting vectors should be distinct.
    self.assertNotEqual(y[0].tolist(), y[1].tolist())
    self.assertNotEqual(y[0].tolist(), y[2].tolist())
    self.assertNotEqual(y[1].tolist(), y[2].tolist())

    # For repeats of a token id, resulting vectors should match.
    self.assertEqual(y[0].tolist(), y[4].tolist())
    self.assertEqual(y[1].tolist(), y[3].tolist())

  def test_negative_inputs_clip_to_zero(self):
    layer = tl.Embedding(10, 3)
    _, _ = layer.init(None)
    x = np.array([0, 2, 3, -2, -3])
    y = layer(x)
    self.assertNotEqual(y[0].tolist(), y[1].tolist())
    self.assertNotEqual(y[0].tolist(), y[2].tolist())
    self.assertEqual(y[0].tolist(), y[3].tolist())
    self.assertEqual(y[0].tolist(), y[4].tolist())

  def test_large_inputs_clip_to_upper_bound(self):
    layer = tl.Embedding(10, 3)
    _, _ = layer.init(None)
    x = np.array([2, 3, 9, 10, 20])
    y = layer(x)

    # vocab_size of 10 means max valid token id is 9.
    self.assertNotEqual(y[2].tolist(), y[0].tolist())
    self.assertNotEqual(y[2].tolist(), y[1].tolist())
    self.assertEqual(y[2].tolist(), y[3].tolist())
    self.assertEqual(y[2].tolist(), y[4].tolist())

  def test_new_weights(self):
    layer = tl.Embedding(20, 5)
    _, _ = layer.init(None)

    # Default weights sampled from Gaussian, mu = 0, sigma = 1.
    w = layer.weights
    self.assertEqual(w.shape, (20, 5))
    self.assertLess(np.abs(np.mean(w)), .4)  # .4 is 4 sigma deviation

  def test_explicit_kernel_initializer(self):

    def f(shape, rng):
      del rng
      n_elements = np.prod(shape)
      return np.arange(n_elements).reshape(shape)

    layer = tl.Embedding(5, 2, kernel_initializer=f)
    _, _ = layer.init(None)
    x = np.array([0, 1, 2, 3, 4])
    y = layer(x)
    self.assertEqual(y.tolist(), [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])


class DropoutTest(absltest.TestCase):

  def test_call_in_train_mode(self):
    layer = tl.Dropout(rate=0.1, mode='train')
    x = np.ones((2, 5, 1000))  # 10,000 values
    y = layer(x)
    self.assertEqual(y.shape, (2, 5, 1000))

    # Dropout is stochastic; test it nonflakily at 4 sigmas (.99994).
    n_remaining = np.count_nonzero(y)
    mu_of_remaining = 9000  # N * q:  10000 * .9
    sigma_of_remaining = 30  # sqrt(N * p * q):  sqrt(10000 * .1 * .9)
    self.assertLess(
        np.abs(n_remaining - mu_of_remaining), 4 * sigma_of_remaining)

  def test_call_in_eval_mode_does_no_dropout(self):
    layer = tl.Dropout(rate=0.1, mode='eval')
    x = np.ones((2, 5, 1000))
    y = layer(x)
    self.assertEqual(np.count_nonzero(y), 10_000)

  def test_new_weights(self):
    layer = tl.Dropout(rate=0.1, mode='train')
    layer.init(None)
    self.assertEmpty(layer.weights)


class FlattenTest(absltest.TestCase):

  def test_keep_default(self):
    layer = tl.Flatten()
    x = np.ones((1, 2, 3, 4, 5))
    y = layer(x)
    # Default is leave first axis untouched, flatten the rest.
    self.assertEqual(y.shape, (1, 2 * 3 * 4 * 5))

  def test_keep_3(self):
    layer = tl.Flatten(n_axes_to_keep=3)
    x = np.ones((1, 2, 3, 4, 5))
    y = layer(x)
    self.assertEqual(y.shape, (1, 2, 3, 4 * 5))

  def test_keep_max_number(self):
    layer = tl.Flatten(n_axes_to_keep=4)
    x = np.ones((1, 2, 3, 4, 5))
    y = layer(x)
    self.assertEqual(y.shape, (1, 2, 3, 4, 5))

  def test_keep_too_many_raises_error(self):
    layer = tl.Flatten(n_axes_to_keep=5)
    with self.assertRaises(tl.LayerError):
      x = np.ones((1, 2, 3, 4, 5))
      _ = layer(x)


class LogGaussianTest(absltest.TestCase):
  # TODO(jonni): Find a more fitting home for this test.

  def test_log_gaussian_pdf(self):
    x = np.zeros((2, 5), dtype=np.float32)
    mu = x
    dsigma = np.eye(5)[None, :, :]
    sigma = np.concatenate([dsigma, 2 * dsigma], axis=0)
    prob = tl.log_gaussian_pdf(x, mu, sigma)
    self.assertEqual(prob.shape, (2,))
    self.assertEqual(int(prob[0]), -4)
    self.assertEqual(int(prob[1]), -6)

  def test_log_gaussian_diag_pdf(self):
    x = np.zeros((2, 5), dtype=np.float32)
    mu = x
    sigma = np.ones((5,))[None, :]
    sigma = np.concatenate([sigma, 2 * sigma], axis=0)
    prob = tl.log_gaussian_diag_pdf(x, mu, sigma)
    self.assertEqual(prob.shape, (2,))
    self.assertEqual(int(prob[0]), -4)
    self.assertEqual(int(prob[1]), -6)


if __name__ == '__main__':
  absltest.main()
