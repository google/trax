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
"""Tests for metrics layers."""

from absl.testing import absltest
import numpy as np
import trax.layers as tl


class MetricsTest(absltest.TestCase):

  def test_category_accuracy(self):
    layer = tl.CategoryAccuracy()
    targets = np.array([0, 1, 2])

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.2, .1, .7, 0.],
                              [.2, .1, .7, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets])
    self.assertEqual(accuracy, 1 / 3)

  def test_weighted_category_accuracy_even_weights(self):
    layer = tl.WeightedCategoryAccuracy()
    weights = np.array([1., 1., 1.])
    targets = np.array([0, 1, 2])

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.2, .1, .7, 0.],
                              [.2, .1, .7, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1 / 3)

  def test_weighted_category_accuracy_uneven_weights(self):
    layer = tl.WeightedCategoryAccuracy()
    weights = np.array([1., 5., 2.])
    targets = np.array([0, 1, 2])

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.2, .7, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .7, .1, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .625)

  def test_category_cross_entropy(self):
    layer = tl.CategoryCrossEntropy()
    targets = np.array([0, 1])

    # Near-perfect prediction (for both items in batch).
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 9., 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .001, places=3)

    # More right than wrong (for both items in batch).
    model_outputs = np.array([[2.2, 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .665, places=3)

    # First item near perfect, second item more right than wrong.
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .333, places=3)

  def test_category_cross_entropy_with_label_smoothing(self):
    epsilon = 0.01
    layer = tl.CategoryCrossEntropy(label_smoothing=epsilon)
    targets = np.array([0, 1])

    # Near-perfect prediction (for both items in batch).
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 9., 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .069, places=3)

    # More right than wrong (for both items in batch).
    model_outputs = np.array([[2.2, 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .682, places=3)

    # First item near perfect, second item more right than wrong.
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets])
    self.assertAlmostEqual(loss, .375, places=3)

  def test_weighted_category_cross_entropy(self):
    layer = tl.WeightedCategoryCrossEntropy()
    targets = np.array([0, 1])
    weights = np.array([30, 10])

    # Near-perfect prediction (for both items in batch).
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 9., 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .001, places=3)

    # More right than wrong (for both items in batch).
    model_outputs = np.array([[2.2, 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .665, places=3)

    # First item (with 75% weight) near perfect, second more right than wrong.
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .167, places=3)

  def test_weighted_category_cross_entropy_with_label_smoothing(self):
    epsilon = 0.01
    layer = tl.WeightedCategoryCrossEntropy(label_smoothing=epsilon)
    targets = np.array([0, 1])
    weights = np.array([30, 10])

    # Near-perfect prediction (for both items in batch).
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 9., 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .069, places=3)

    # More right than wrong (for both items in batch).
    model_outputs = np.array([[2.2, 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .682, places=3)

    # First item (with 75% weight) near perfect, second more right than wrong.
    model_outputs = np.array([[9., 2., 0., -2.],
                              [2., 2.2, 0., -2.]])
    loss = layer([model_outputs, targets, weights])
    self.assertAlmostEqual(loss, .222, places=3)

  def test_masked_sequence_accuracy(self):
    layer = tl.MaskedSequenceAccuracy()
    targets = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0]])
    weights = np.array([[1., 1., 1., 0.],
                        [1., 1., 1., 0.]])

    # Model gets both sequences right; output in final position would give
    # wrong category but is ignored.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.7, .3], [.35, .65]],
                              [[.3, .7], [.8, .2], [.1, .9], [.35, .65]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.)

    # Model gets the first element of the first sequence barely wrong.
    model_outputs = np.array([[[.45, .55], [.2, .8], [.7, .3], [.6, .4]],
                              [[.3, .7], [.8, .2], [.1, .9], [.6, .4]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .5)

    # Model gets second-to-last element of each sequence barely wrong.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.48, .52], [.6, .4]],
                              [[.3, .7], [.8, .2], [.51, .49], [.6, .4]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 0.)

  def test_binary_cross_entropy(self):
    layer = tl.BinaryCrossEntropy()
    targets = np.array([1, 1, 0, 0, 0])

    # Near-perfect prediction for all five items in batch.
    model_outputs = np.array([9., 9., -9., -9., -9.])
    metric_output = layer([model_outputs, targets])
    self.assertAlmostEqual(metric_output, 0.000123, places=6)

    # More right than wrong for all five items in batch.
    model_outputs = np.array([1., 1., -1., -1., -1.])
    metric_output = layer([model_outputs, targets])
    self.assertAlmostEqual(metric_output, 0.313, places=3)

    # Near-perfect for 2, more right than wrong for 3.
    model_outputs = np.array([9., 1., -1., -1., -9.])
    metric_output = layer([model_outputs, targets])
    self.assertAlmostEqual(metric_output, 0.188, places=3)

    # More wrong than right for all five.
    model_outputs = np.array([-1., -1., 1., 1., 1.])
    metric_output = layer([model_outputs, targets])
    self.assertAlmostEqual(metric_output, 1.313, places=3)

  def test_accuracy_even_weights(self):
    layer = tl.Accuracy()
    weights = np.array([1., 1., 1.])
    targets = np.array([0, 1, 2])

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.2, .1, .7, 0.],
                              [.2, .1, .7, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1 / 3)

  def test_accuracy_uneven_weights(self):
    layer = tl.Accuracy()
    weights = np.array([1., 5., 2.])
    targets = np.array([0, 1, 2])

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .1, .7, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.2, .7, .1, 0.],
                              [.2, .7, .1, 0.],
                              [.2, .7, .1, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .625)

    model_outputs = np.array([[.7, .2, .1, 0.],
                              [.7, .2, .1, 0.],
                              [.7, .2, .1, 0.]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .125)

  def test_accuracy_binary_classifier(self):
    layer = tl.Accuracy(classifier=tl.ThresholdToBinary())
    targets = np.array([[0, 0, 1, 1],
                        [1, 1, 1, 0]])
    weights = np.ones_like(targets)

    model_outputs = np.array([[.499, .500, .501, .502],
                              [.503, .502, .501, .500]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.0)

    model_outputs = np.array([[.498, .499, .500, .501],
                              [.502, .501, .500, .499]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .75)

  def test_sequence_accuracy_weights_all_ones(self):
    layer = tl.SequenceAccuracy()
    targets = np.array([[0, 1, 0, 1],
                        [1, 0, 1, 1]])
    weights = np.ones_like(targets)

    # Model gets both sequences right; for each position in each sequence, the
    # category (integer ID) selected by argmax matches the target category.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.7, .3], [.4, .6]],
                              [[.3, .7], [.8, .2], [.1, .9], [.4, .6]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.)

    # Model gets the first element of the first sequence barely wrong.
    model_outputs = np.array([[[.45, .55], [.2, .8], [.7, .3], [.4, .6]],
                              [[.3, .7], [.8, .2], [.1, .9], [.4, .6]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .5)

    # Model gets the last element of each sequence barely wrong.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.7, .3], [.55, .45]],
                              [[.3, .7], [.8, .2], [.1, .9], [.52, .48]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 0.)

  def test_sequence_accuracy_last_position_zero_weight(self):
    layer = tl.SequenceAccuracy()
    targets = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0]])
    weights = np.array([[1., 1., 1., 0.],
                        [1., 1., 1., 0.]])

    # Model gets both sequences right; output in final position would give
    # wrong category but is ignored.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.7, .3], [.35, .65]],
                              [[.3, .7], [.8, .2], [.1, .9], [.35, .65]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 1.)

    # Model gets the first element of the first sequence barely wrong.
    model_outputs = np.array([[[.45, .55], [.2, .8], [.7, .3], [.6, .4]],
                              [[.3, .7], [.8, .2], [.1, .9], [.6, .4]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, .5)

    # Model gets second-to-last element of each sequence barely wrong.
    model_outputs = np.array([[[.9, .1], [.2, .8], [.48, .52], [.6, .4]],
                              [[.3, .7], [.8, .2], [.51, .49], [.6, .4]]])
    accuracy = layer([model_outputs, targets, weights])
    self.assertEqual(accuracy, 0.)

  def test_binary_cross_entropy_loss(self):
    # TODO(jonni): Clarify desired semantics/naming, then test it.
    layer = tl.BinaryCrossEntropyLoss()
    xs = [np.ones((9, 1)),
          np.ones((9, 1)),
          np.ones((9, 1))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_cross_entropy_loss(self):
    # TODO(jonni): Clarify desired semantics/naming, then test it.
    layer = tl.CrossEntropyLoss()
    xs = [np.ones((9, 4, 4, 20)),
          np.ones((9, 4, 4)),
          np.ones((9, 4, 4))]
    y = layer(xs)
    self.assertEqual(y.shape, ())

  def test_l2_loss(self):
    layer = tl.L2Loss()

    model_outputs = np.array([[1., 1.], [1., 1.]])
    targets = np.array([[1., 1.], [1., 0.]])
    weights = np.array([[1., 1.], [1., 0.]])
    loss = layer([model_outputs, targets, weights])
    np.testing.assert_allclose(loss, 0.0)

    weights = np.array([[1., 0.], [0., 1.]])
    loss = layer([model_outputs, targets, weights])
    np.testing.assert_allclose(loss, 0.5)

  def test_smooth_l1_loss(self):
    layer = tl.SmoothL1Loss()

    model_outputs = np.array([[1., 1.], [1., 2.]])
    targets = np.array([[1., 1.], [1., 0.]])
    l1_dist = 2

    weights = np.array([[1., 1.], [1., 0.]])
    loss = layer([model_outputs, targets, weights])
    np.testing.assert_allclose(loss, 0.0)

    weights = np.array([[1., 0.], [0., 1.]])
    sum_weights = 2

    loss = layer([model_outputs, targets, weights])
    np.testing.assert_allclose(loss, (l1_dist-0.5) / sum_weights)

    model_outputs = np.array([[1., 1.], [1., 1.5]])
    targets = np.array([[1., 1.], [1., 1.]])
    l1_dist = 0.5
    loss = layer([model_outputs, targets, weights])
    np.testing.assert_allclose(loss, 0.5 * l1_dist**2 / sum_weights)

  def test_names(self):
    layer = tl.L2Loss()
    self.assertEqual('L2Loss_in3', str(layer))
    layer = tl.Accuracy()
    self.assertEqual('Accuracy_in3', str(layer))
    layer = tl.SequenceAccuracy()
    self.assertEqual('SequenceAccuracy_in3', str(layer))
    layer = tl.BinaryCrossEntropyLoss()
    self.assertEqual('BinaryCrossEntropyLoss_in3', str(layer))
    layer = tl.CrossEntropyLoss()
    self.assertEqual('CrossEntropyLoss_in3', str(layer))
    layer = tl.BinaryCrossEntropySum()
    self.assertEqual('BinaryCrossEntropySum_in3', str(layer))
    layer = tl.CrossEntropySum()
    self.assertEqual('CrossEntropySum_in3', str(layer))


if __name__ == '__main__':
  absltest.main()
