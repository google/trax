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
"""Tests for normalization layers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import fastmath
from trax import shapes
import trax.layers as tl


class BatchNormTest(parameterized.TestCase):

  def test_forward_shape(self):
    layer = tl.BatchNorm()
    x = np.ones((30, 20, 70)).astype(np.float32)
    _, _ = layer.init(shapes.signature(x))
    y = layer(x)
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters(
      ('jax32', fastmath.Backend.JAX, np.float32),
      ('tf32', fastmath.Backend.TFNP, np.float32),
      ('tf64', fastmath.Backend.TFNP, np.float64),
  )
  def test_forward_dtype(self, backend, dtype):
    with fastmath.use_backend(backend):
      layer = tl.BatchNorm()
      x = np.ones((3, 2, 7)).astype(dtype)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.dtype, dtype)

  @parameterized.named_parameters(
      ('momentum_999', .999),
      ('momentum_900', .900),
      ('momentum_800', .800),
  )
  def test_forward(self, momentum):
    layer = tl.BatchNorm(momentum=momentum)
    x = np.array([[[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]],
                  [[12, 13, 14, 15],
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]]).astype(np.float32)
    _, _ = layer.init(shapes.signature(x))
    y = layer(x)
    running_mean, running_var, n_batches = layer.state

    fraction_old = momentum
    fraction_new = 1.0 - momentum
    mean_of_x = 11.5  # mean of range(24)
    var_of_x = 47.9167  # variance of range(24)
    np.testing.assert_allclose(
        running_mean, 0.0 * fraction_old + mean_of_x * fraction_new)
    np.testing.assert_allclose(
        running_var, 1.0 * fraction_old + var_of_x * fraction_new, rtol=1e-6)
    self.assertEqual(n_batches, 1)
    eps = 1e-5
    np.testing.assert_allclose(
        y, (x - mean_of_x) / np.sqrt(var_of_x + eps), rtol=1e-6)

  def test_new_weights_and_state(self):
    layer = tl.BatchNorm()
    x = np.ones((3, 2, 7)).astype(np.float32)
    _, _ = layer.init(shapes.signature(x))

    running_mean, running_var, n_batches = layer.state
    np.testing.assert_allclose(running_mean, 0.0)
    np.testing.assert_allclose(running_var, 1.0)
    self.assertEqual(n_batches, 0)


class LayerNormTest(parameterized.TestCase):

  def test_forward_shape(self):
    layer = tl.LayerNorm()
    x = np.ones((3, 2, 7)).astype(np.float32)
    _, _ = layer.init(shapes.signature(x))
    y = layer(x)
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters(
      ('jax32', fastmath.Backend.JAX, np.float32),
      ('tf32', fastmath.Backend.TFNP, np.float32),
      ('tf64', fastmath.Backend.TFNP, np.float64),
  )
  def test_forward_dtype(self, backend, dtype):
    with fastmath.use_backend(backend):
      layer = tl.LayerNorm()
      x = np.ones((3, 2, 7)).astype(dtype)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.dtype, dtype)


class FilterResponseNormTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('learn_epsilon_false', False),
      ('learn_epsilon_true', True),
  )
  def test_forward_shape(self, learn_epsilon):
    layer = tl.FilterResponseNorm(learn_epsilon=learn_epsilon)

    B, H, W, C = 64, 5, 7, 3  # pylint: disable=invalid-name
    x = np.ones((B, H, W, C)).astype(np.float32)
    _, _ = layer.init(shapes.signature(x))
    y = layer(x)
    self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
  absltest.main()
