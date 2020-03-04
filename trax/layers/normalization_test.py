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
"""Tests for normalization layers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

from trax.layers import base
from trax.layers import normalization
from trax.math import numpy as np
from trax.math import use_backend
from trax.shapes import ShapeDtype


class NormalizationLayerTest(parameterized.TestCase):

  def test_batch_norm_shape(self):
    input_signature = ShapeDtype((29, 5, 7, 20))
    result_shape = base.check_shape_agreement(normalization.BatchNorm(),
                                              input_signature)
    self.assertEqual(result_shape, input_signature.shape)

  def test_batch_norm(self):
    input_shape = (2, 3, 4)
    input_dtype = np.float32
    input_signature = ShapeDtype(input_shape, input_dtype)
    eps = 1e-5
    inp1 = np.reshape(np.arange(np.prod(input_shape), dtype=input_dtype),
                      input_shape)
    m1 = 11.5  # Mean of this random input.
    v1 = 47.9167  # Variance of this random input.
    layer = normalization.BatchNorm(axis=(0, 1, 2))
    _, _ = layer.init(input_signature)
    state = layer.state
    onp.testing.assert_allclose(state[0], 0)
    onp.testing.assert_allclose(state[1], 1)
    self.assertEqual(state[2], 0)
    out = layer(inp1)
    state = layer.state
    onp.testing.assert_allclose(state[0], m1 * 0.001)
    onp.testing.assert_allclose(state[1], 0.999 + v1 * 0.001, rtol=1e-6)
    self.assertEqual(state[2], 1)
    onp.testing.assert_allclose(out, (inp1 - m1) / np.sqrt(v1 + eps),
                                rtol=1e-6)

  def test_layer_norm_shape(self):
    input_signature = ShapeDtype((29, 5, 7, 20))
    result_shape = base.check_shape_agreement(
        normalization.LayerNorm(), input_signature)
    self.assertEqual(result_shape, input_signature.shape)

  @parameterized.named_parameters(
      ('jax32', 'jax', np.float32),
      ('tf32', 'tf', np.float32),
      ('tf64', 'tf', np.float64),
      # NOTE: float64 requires more fix in jax
      # ('jax64', 'jax', np.float64),
  )
  def test_layer_norm_dtype(self, backend, dtype):
    with use_backend(backend):
      input_shape = (2, 3, 4)
      input_signature = ShapeDtype(input_shape, dtype)
      layer = normalization.LayerNorm()
      layer.init(input_signature)
      out = layer(onp.empty(input_shape, dtype=dtype))
      self.assertEqual(out.dtype, dtype)

  def test_frn_shape(self):
    B, H, W, C = 64, 5, 7, 3  # pylint: disable=invalid-name
    input_signature = ShapeDtype((B, H, W, C))
    result_shape = base.check_shape_agreement(
        normalization.FilterResponseNorm(), input_signature)
    self.assertEqual(result_shape, input_signature.shape)

    result_shape = base.check_shape_agreement(
        normalization.FilterResponseNorm(learn_epsilon=False),
        input_signature)
    self.assertEqual(result_shape, input_signature.shape)


if __name__ == '__main__':
  absltest.main()
