# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Tests for metrics layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp
from trax.layers import base
from trax.layers import metrics
from trax.shapes import ShapeDtype
from trax.shapes import signature


class MetricsLayerTest(absltest.TestCase):

  def test_cross_entropy(self):
    input_signature = (ShapeDtype((29, 4, 4, 20)), ShapeDtype((29, 4, 4)))
    result_shape = base.check_shape_agreement(
        metrics.CrossEntropy(), input_signature)
    self.assertEqual(result_shape, (29, 4, 4))

  def test_accuracy(self):
    input_signature = (ShapeDtype((29, 4, 4, 20)), ShapeDtype((29, 4, 4)))
    result_shape = base.check_shape_agreement(
        metrics.Accuracy(), input_signature)
    self.assertEqual(result_shape, (29, 4, 4))

  def test_element_mask(self):
    input_signature = ShapeDtype((29, 4, 4, 20))
    result_shape = base.check_shape_agreement(
        metrics.ElementMask(), input_signature)
    self.assertEqual(result_shape, (29, 4, 4, 20))

  def test_weighted_mean_shape(self):
    input_signature = (ShapeDtype((29, 4, 4, 20)), ShapeDtype((29, 4, 4, 20)))
    result_shape = base.check_shape_agreement(
        metrics.WeightedMean(), input_signature)
    self.assertEqual(result_shape, ())

  def test_weighted_mean_semantics(self):
    inputs = onp.array([1, 2, 3], dtype=onp.float32)
    weights1 = onp.array([1, 1, 1], dtype=onp.float32)
    layer = metrics.WeightedMean()
    full_signature = (signature(inputs), signature(weights1))
    layer.init(full_signature)
    mean1 = layer((inputs, weights1))
    onp.testing.assert_allclose(mean1, 2.0)
    weights2 = onp.array([0, 0, 1], dtype=onp.float32)
    mean2 = layer((inputs, weights2))
    onp.testing.assert_allclose(mean2, 3.0)
    weights3 = onp.array([1, 0, 0], dtype=onp.float32)
    mean3 = layer((inputs, weights3))
    onp.testing.assert_allclose(mean3, 1.0)

  def test_cross_entropy_loss(self):
    input_signature = (ShapeDtype((29, 4, 4, 20)), ShapeDtype((29, 4, 4)))
    result_shape = base.check_shape_agreement(
        metrics.CrossEntropyLoss(), input_signature)
    self.assertEqual(result_shape, ())

  def test_accuracy_scalar(self):
    input_signature = (ShapeDtype((29, 4, 4, 20)), ShapeDtype((29, 4, 4)))
    result_shape = base.check_shape_agreement(
        metrics.AccuracyScalar(), input_signature)
    self.assertEqual(result_shape, ())


if __name__ == '__main__':
  absltest.main()
