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
"""Tests for acceleration."""

from absl.testing import absltest

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes


class AccelerationTest(absltest.TestCase):

  def test_accelerated_same_result(self):
    layer = tl.Dense(2)
    x = np.random.uniform(size=(8, 7))
    layer.init(shapes.signature(x))
    y = layer(x)
    z = tl.Accelerate(layer)(x)
    for i in range(8):
      self.assertAlmostEqual(float(y[i, 0]), float(z[i, 0]), places=4)
      self.assertAlmostEqual(float(y[i, 1]), float(z[i, 1]), places=4)

  def test_accelerated_pad(self):
    layer = tl.Dense(2)
    x = np.random.uniform(size=(3, 7))
    layer.init(shapes.signature(x))
    y = layer(x)
    z = tl.Accelerate(layer)(x)
    self.assertEqual(z.shape, y.shape)
    for i in range(3):
      self.assertAlmostEqual(float(y[i, 0]), float(z[i, 0]), places=4)
      self.assertAlmostEqual(float(y[i, 1]), float(z[i, 1]), places=4)

  def test_accelerated_weighted_category_accuracy(self):
    """Test multi-device aggregation of weights."""
    layer = tl.Accelerate(tl.WeightedCategoryAccuracy())
    weights = np.array([1., 1., 1., 0.])
    targets = np.array([0, 1, 2, 3])

    model_outputs = np.array([[.2, .1, .7, 0.],
                              [.2, .1, .7, 0.],
                              [.2, .1, .7, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(np.mean(accuracy), 1 / 3)

  def test_chunk_memory(self):
    """Test chunking here to exercise accelerator memory usage."""
    layer = tl.Serial(tl.Dense(1024*1024), tl.Dense(128))
    chunked = tl.Chunk(layer, 256)
    x = np.random.uniform(size=(16*1024, 16))
    chunked.init(shapes.signature(x))
    y = chunked(x)
    z = tl.Accelerate(chunked)(x)
    self.assertEqual(y.shape, (16*1024, 128))
    self.assertEqual(z.shape, (16*1024, 128))

  def test_chunk_grad_memory(self):
    """Test chunking gradient here to exercise accelerator memory usage."""
    layer = tl.Serial(tl.Dense(1024*1024), tl.Dense(24))
    chunked = tl.Chunk(layer, 256)

    @fastmath.jit
    def mock_training_step(x, weights, state, rng):
      def compute_mock_loss(weights):
        logits, new_state = chunked.pure_fn(x, weights, state, rng)
        loss = fastmath.numpy.mean(logits)
        return loss, (new_state, logits)
      gradients, (new_state, logits) = fastmath.grad(
          compute_mock_loss, has_aux=True)(weights)
      new_weights = fastmath.nested_map_multiarg(
          lambda w, g: w - 1e-4 * g, weights, gradients)
      return new_weights, new_state, logits

    x = np.random.uniform(size=(32*1024, 16))
    chunked.init(shapes.signature(x))
    weights, _, logits = mock_training_step(
        x, chunked.weights, chunked.state, fastmath.random.get_prng(0))
    self.assertEqual(logits.shape, (32*1024, 24))
    self.assertEqual(weights[1][0][0][0].shape, (16, 1024*1024))


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
