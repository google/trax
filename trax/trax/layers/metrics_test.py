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
"""Tests for metrics layers."""

from absl.testing import absltest
import numpy as np

from trax import shapes
import trax.layers as tl
from trax.layers import metrics


class MetricsTest(absltest.TestCase):

  def test_cross_entropy(self):
    layer = metrics._CrossEntropy()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 4, 4))

  def test_accuracy(self):
    layer = metrics._Accuracy()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, (9, 4, 4))

  def test_weighted_mean_shape(self):
    layer = metrics._WeightedMean()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4, 20))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_weighted_mean_semantics(self):
    layer = metrics._WeightedMean()
    sample_input = np.ones((3,))
    sample_weights = np.ones((3,))
    layer.init(shapes.signature([sample_input, sample_weights]))

    x = np.array([1., 2., 3.])
    weights = np.array([1., 1., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 2.)

    weights = np.array([0., 0., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 3.)

    weights = np.array([1., 0., 0.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 1.)

  def test_weighted_sequence_mean_semantics(self):
    layer = metrics._WeightedSequenceMean()
    sample_input = np.ones((2, 3))
    sample_weights = np.ones((3,))
    full_signature = shapes.signature([sample_input, sample_weights])
    layer.init(full_signature)

    x = np.array([[1., 1., 1.], [1., 1., 0.]])
    weights = np.array([1., 1., 1.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 0.5)

    weights = np.array([1., 1., 0.])
    mean = layer((x, weights))
    np.testing.assert_allclose(mean, 1.)

  def test_cross_entropy_loss(self):
    layer = tl.CrossEntropyLoss()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_accuracy_scalar(self):
    layer = tl.Accuracy()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_l2_loss(self):
    layer = tl.L2Loss()
    sample_input = np.ones((2, 2))
    sample_target = np.ones((2, 2))
    sample_weights = np.ones((2, 2))
    full_signature = shapes.signature([sample_input,
                                       sample_target,
                                       sample_weights])
    layer.init(full_signature)

    x = np.array([[1., 1.], [1., 1.]])
    target = np.array([[1., 1.], [1., 0.]])
    weights = np.array([[1., 1.], [1., 0.]])
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.0)

    weights = np.array([[1., 0.], [0., 1.]])
    loss = layer((x, target, weights))
    np.testing.assert_allclose(loss, 0.5)

  def test_names(self):
    layer = tl.L2Loss()
    self.assertEqual('L2Loss_in3', str(layer))
    layer = tl.Accuracy()
    self.assertEqual('Accuracy_in3', str(layer))
    layer = tl.SequenceAccuracy()
    self.assertEqual('SequenceAccuracy_in3', str(layer))
    layer = tl.CrossEntropyLoss()
    self.assertEqual('CrossEntropyLoss_in3', str(layer))
    layer = tl.CrossEntropySum()
    self.assertEqual('CrossEntropySum_in3', str(layer))


if __name__ == '__main__':
  absltest.main()
