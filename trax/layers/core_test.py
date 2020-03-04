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
import numpy as onp
from trax.layers import base
from trax.layers import combinators
from trax.layers import core
from trax.shapes import ShapeDtype


def divide_by(val):
  """Returns a simple division layer with n_in == 1 and n_out == 1."""
  return base.Fn(lambda x: x / val)


class CoreLayerTest(absltest.TestCase):

  def test_flatten_n(self):
    input_signature = ShapeDtype((29, 87, 10, 20, 30))

    layer = core.Flatten()
    expected_shape = (29, 87 * 10 * 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=2)
    expected_shape = (29, 87, 10 * 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=3)
    expected_shape = (29, 87, 10, 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=4)
    expected_shape = (29, 87, 10, 20, 30)
    actual_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(actual_shape, expected_shape)

    # Not enough dimensions.
    with self.assertRaises(base.LayerError):
      base.check_shape_agreement(core.Flatten(n_axes_to_keep=5),
                                 input_signature)

    with self.assertRaises(base.LayerError):
      base.check_shape_agreement(core.Flatten(n_axes_to_keep=6),
                                 input_signature)

  def test_div(self):
    layer = divide_by(2.0)
    input_np = onp.array([[1, 2, 3], [4, 5, 6]], dtype=onp.float32)
    output_np = layer(input_np)
    # absltest doesn't have ndarray equalities.
    expected_output_np = input_np / 2.0
    self.assertAlmostEqual(
        0.0,
        onp.sum((output_np - expected_output_np) ** 2),
        delta=1e-6)

  def test_div_shapes(self):
    layer = divide_by(2.0)
    input_signature = ShapeDtype((3, 2))
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_signature)
    self.assertEqual(output_shape, expected_shape)

  def test_dense_weight_sharing(self):
    model1 = combinators.Serial(core.Dense(32), core.Dense(32))
    layer = core.Dense(32)
    model2 = combinators.Serial(layer, layer)

    input_signature = ShapeDtype((1, 32))
    weights1, _ = model1.init(input_signature)
    weights2, _ = model2.init(input_signature)
    # The first weights have 2 kernels of size (32, 32).
    self.assertEqual((32, 32), weights1[0][0].shape)
    self.assertEqual((32, 32), weights1[1][0].shape)
    # The second weights have 1 kernel of size (32, 32) and an empty dict.
    self.assertEqual((32, 32), weights2[0][0].shape)
    self.assertEqual((), weights2[1])

  def test_dropout(self):
    input_signature = ShapeDtype((8, 7, 9))
    output_shape = (8, 7, 9)
    final_shape = base.check_shape_agreement(
        core.Dropout(rate=0.1, mode='train'), input_signature)
    self.assertEqual(final_shape, output_shape)
    final_shape = base.check_shape_agreement(
        core.Dropout(rate=0.1, mode='eval'), input_signature)
    self.assertEqual(final_shape, output_shape)

  def test_log_gaussian_pdf(self):
    x = onp.zeros((2, 5), dtype=onp.float32)
    mu = x
    dsigma = onp.eye(5)[None, :, :]
    sigma = onp.concatenate([dsigma, 2*dsigma], axis=0)
    prob = core.log_gaussian_pdf(x, mu, sigma)
    self.assertEqual(prob.shape, (2,))
    self.assertEqual(int(prob[0]), -4)
    self.assertEqual(int(prob[1]), -6)

  def test_log_gaussian_diag_pdf(self):
    x = onp.zeros((2, 5), dtype=onp.float32)
    mu = x
    sigma = onp.ones((5,))[None, :]
    sigma = onp.concatenate([sigma, 2*sigma], axis=0)
    prob = core.log_gaussian_diag_pdf(x, mu, sigma)
    self.assertEqual(prob.shape, (2,))
    self.assertEqual(int(prob[0]), -4)
    self.assertEqual(int(prob[1]), -6)

if __name__ == '__main__':
  absltest.main()
