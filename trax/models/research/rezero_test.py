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
"""Tests for ReZero models."""

from absl.testing import absltest
import numpy as np

from trax import layers as tl
from trax import shapes
from trax.models.research import rezero


class ResidualZeroTest(absltest.TestCase):

  def test_residual_layer_forward(self):
    """Tests that the forward pass runs and returns the expected shape."""
    model = rezero.ResidualZero(tl.Dense(5))
    x = [np.arange(5).astype(np.float32)]
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.tolist(), [0., 1., 2., 3., 4.])


class ReZeroTransformerLMTest(absltest.TestCase):

  def test_rezero_lm_forward_shape(self):
    """Tests that the forward pass runs and returns the expected shape."""
    vocab_size = 16
    model = rezero.ReZeroTransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2, max_len=16)
    xs = [np.ones((1, 8)).astype(np.int32),
          np.ones((1, 8)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    ys = model(xs)
    self.assertEqual([y.shape for y in ys], [(1, 8, 16), (1, 8)])


class ReZeroTransformerTest(absltest.TestCase):

  def test_rezero_forward_shape(self):
    """Tests that the forward pass runs and returns the expected shape."""
    vocab_size = 16
    model = rezero.ReZeroTransformer(
        vocab_size, d_model=32, d_ff=64, n_encoder_layers=2, n_decoder_layers=2,
        n_heads=2, max_len=16)
    xs = [np.ones((1, 8)).astype(np.int32),
          np.ones((1, 8)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    ys = model(xs)
    self.assertEqual([y.shape for y in ys], [(1, 8, 16), (1, 8)])


if __name__ == '__main__':
  absltest.main()
