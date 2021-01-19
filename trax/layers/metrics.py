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
"""Layers for computing loss functions and evaluation metrics.

A metric layer computes a scalar value from two or three ndarray inputs:

  - model outputs: Batch of predicted values (typically vectors).

  - targets: Batch of target values (e.g., categories or vectors).

  - weights: Float values that allow for uneven weighting of batch items,
    sequence positions, or vector components when computing an overall scalar
    value for the batch.

Most metric computations take into account the items that make up a batch. For
each item in a batch, a raw metric value is computed by comparing (item-wise)
the model output to the target value. These item-wise values are then combined
into a single scalar for the batch by a function such as sum, average, or
weighted-average. For example:

  - `CategoryAccuracy`: Treat model output as vectors whose components
    correspond to the possible categories; measure a vector as correct (value
    1) if its largest component is the target category, else as incorrect
    (value 0). The accuracy for the batch is then the average across vectors of
    these 1's and 0's.

  - `CategoryCrossEntropy`: Treat model output and target values as the source
    of two probability distributions; measure the cross-entropy of the model's
    predicted distribution relative to the (assumed true) target distribution.
    The scalar value for the batch is then the average of the item-wise
    cross-entropy values.
"""

from trax import fastmath
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core


def CategoryAccuracy():
  r"""Returns a layer that computes category prediction accuracy.

  The layer takes two inputs:

    - A batch of activation vectors. The components in a given vector should
      be mappable to a probability distribution in the following loose sense:
      within a vector, a higher component value corresponds to a higher
      probability, such that argmax within a vector (``axis=-1``) picks the
      index (category) having the highest probablity.

    - A batch of target categories; each target is an integer in
      :math:`\{0, ..., N-1\}`.

  The predicted category from each vector is the index of the highest-valued
  vector component. The layer returns the accuracy of these predictions
  averaged over the batch.
  """
  def f(model_output, targets):  # pylint: disable=invalid-name
    predictions = jnp.argmax(model_output, axis=-1)
    shapes.assert_same_shape(predictions, targets)
    n_total = predictions.size
    n_correct = jnp.sum(jnp.equal(predictions, targets))
    return n_correct / n_total

  return base.Fn('CategoryAccuracy', f)


def WeightedCategoryAccuracy():
  r"""Returns a layer that computes a weighted category prediction accuracy.

  The layer takes three inputs:

    - A batch of activation vectors. The components in a given vector should
      be mappable to a probability distribution in the following loose sense:
      within a vector, a higher component value corresponds to a higher
      probability, such that argmax within a vector (``axis=-1``) picks the
      index (category) having the highest probablity.

    - A batch of target categories; each target is an integer in
      :math:`\{0, ..., N-1\}`, where :math:`N` is the activation vector
      depth/dimensionality.

    - A batch of weights, which matches or can be broadcast to match the shape
      of the target ndarray. This arg can give uneven weighting to different
      items in the batch (depending, for instance, on the item's target
      category).

  The predicted category from each vector is the index of the highest-valued
  vector component. The layer returns a weighted average accuracy of these
  predictions.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    predictions = jnp.argmax(model_output, axis=-1)
    shapes.assert_same_shape(predictions, targets)
    ones_and_zeros = jnp.equal(predictions, targets)
    return jnp.sum(ones_and_zeros * weights) / jnp.sum(weights)

  return base.Fn('WeightedCategoryAccuracy', f)


def CategoryCrossEntropy(label_smoothing=None):
  r"""Returns a layer that computes cross-entropy from activations and integers.

  The layer takes two inputs:

    - A batch of activation vectors. The components in a given vector should
      be pre-softmax activations (mappable to a probability distribution via
      softmax). For performance reasons, the softmax and cross-entropy
      computations are combined inside the layer.

    - A batch of target categories; each target is an integer in
      :math:`\{0, ..., N-1\}`, where :math:`N` is the activation vector
      depth/dimensionality.

  To compute cross-entropy per batch item, the layer derives probability
  distributions:

    - from model output (vectors): :math:`\ q = \text{softmax}(v)`

    - from target categories (integers): :math:`\ p = \text{one_hot}(n)` or
      :math:`p = (1-\varepsilon)\cdot\text{one_hot}(n) + \frac{\varepsilon}{N}`,
      where :math:`\varepsilon` is the label smoothing factor.

  (The conversion of integer category targets to one-hot vectors amounts to
  assigning all the probability mass to the target category.) Cross-entropy
  per batch item is computed between the resulting distributions:

  .. math::
      \text{cross_entropy} = - \sum_{i=0}^{N-1} p_i \log q_i

  The layer returns the average of these cross-entropy values over all items in
  the batch.

  Args:
    label_smoothing: Creates soft targets if provided. Must be between 0 and 1.
  """
  def f(model_output, targets):  # pylint: disable=invalid-name
    cross_entropies = _category_cross_entropy(
        model_output, targets, label_smoothing)
    return jnp.average(cross_entropies)

  return base.Fn('CategoryCrossEntropy', f)


def WeightedCategoryCrossEntropy(label_smoothing=None):
  r"""Returns a layer like ``CategoryCrossEntropy``, with weights as third input.

  The layer takes three inputs:

    - A batch of activation vectors. The components in a given vector should
      be pre-softmax activations (mappable to a probability distribution via
      softmax). For performance reasons, the softmax and cross-entropy
      computations are combined inside the layer.

    - A batch of target categories; each target is an integer in
      :math:`\{0, ..., N-1\}`, where :math:`N` is the activation vector
      depth/dimensionality.

    - A batch of weights, which matches or can be broadcast to match the shape
      of the target ndarray. This arg can give uneven weighting to different
      items in the batch (depending, for instance, on the item's target
      category).

  The layer returns the weighted average of these cross-entropy values over all
  items in the batch.

  Args:
    label_smoothing: Creates soft targets if provided. Must be between 0 and 1.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    cross_entropies = _category_cross_entropy(
        model_output, targets, label_smoothing)
    return jnp.sum(cross_entropies * weights) / jnp.sum(weights)

  return base.Fn('WeightedCategoryCrossEntropy', f)


def BinaryCrossEntropy():
  r"""Returns a layer that computes cross-entropy for binary classification.

  The layer takes two inputs:

    - A batch of activation values; each batch item :math:`x` is a float in
      :math:`(-\infty, \infty)`.

    - A batch of binary targets; each target :math:`t` is an integer in
      :math:`\{0, 1\}`.

  The layer maps each activation value into the range :math:`(0, 1)`,
  interpreted as the model-predicted probability that item's category is 1:

  .. math::
      q = \frac 1 {1 + e^{-x}} \ \ \text{[model-predicted probability]}

  and computes cross-entropy (per batch item) by treating the target category
  as having probability 1:

  .. math::
      \text{cross_entropy} = \left\{ \begin{array}{cl}
          - \log q       & \text{if}\ t = 1, \\
          - \log (1 - q) & \text{if}\ t = 0.
      \end{array} \right.

  The layer returns the average of these cross-entropy values over all items in
  the batch.
  """
  def f(model_output, targets):  # pylint: disable=invalid-name
    probabilities = fastmath.expit(model_output)
    binary_entropies = - (targets * jnp.log(probabilities) +
                          (1 - targets) * (jnp.log(1 - probabilities)))
    return jnp.average(binary_entropies)

  return base.Fn('BinaryCrossEntropy', f)


def MaskedSequenceAccuracy():
  r"""Returns a layer that computes sequence prediction accuracy with masking.

  This layer type is intended for variable length sequences, especially text,
  represented as a batch of fixed-length sequences via padding for unused
  positions.

  The layer takes three inputs:

    - A batch of sequences of activation vectors. The components in a given
      vector should be mappable to a probability distribution in the following
      loose sense: within a vector, a higher component value corresponds to a
      higher probability, such that argmax within a vector (``axis=-1``) picks
      the index having the highest probablity. In text modeling, the index
      represents a token id from a predetermined token vocabulary (or padding).

    - A batch of target integer sequences, with values in
      :math:`\{0, ..., N-1\}`, where :math:`N` is the activation vector
      depth/dimensionality. In text modeling, these sequences typically
      represent token ids from a predetermined token vocabulary (or padding).

    - A batch of weights/masks, which matches or can be broadcast to match the
      shape of the target ndarray. This arg is used to give weight 0 to padding
      positions, which masks those positions out of the calculation. Only the
      zero/non-zero distinction matters; all non-zero values are treated alike
      as signaling non-masked (i.e., valid/in-use) positions.

  The predicted integer value for each sequence position is the index of the
  highest-valued component of the position's vector. A predicted integer
  sequence is judged correct if it matches the target integer sequence in all
  non-zero-weighted positions. The layer returns the accuracy of predicted
  sequences averaged over the batch.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    predictions = jnp.argmax(model_output, axis=-1)
    shapes.assert_same_shape(predictions, targets)
    position_is_padding = jnp.equal(weights, 0)
    position_is_accurate = jnp.logical_or(jnp.equal(predictions, targets),
                                          position_is_padding)
    sequence_is_accurate = jnp.all(position_is_accurate, axis=-1)
    return jnp.average(sequence_is_accurate)

  return base.Fn('MaskedSequenceAccuracy', f)


def Accuracy(classifier=core.ArgMax()):
  """Returns a layer that computes mean category prediction accuracy.

  DEPRECATED; use ``WeightedCategoryAccuracy`` instead.

  Args:
    classifier: Layer that transforms activation vectors into category
        predictions.
  """
  return cb.Serial(classifier,
                   _Accuracy(),
                   _WeightedMean(),
                   name='Accuracy',
                   sublayers_to_print=[])


def SequenceAccuracy(classifier=core.ArgMax()):
  """Returns a layer that computes mean sequence prediction accuracy.

  DEPRECATED; use ``MaskedSequenceAccuracy`` instead.

  Args:
    classifier: Layer that transforms activation vectors into category
        predictions.
  """
  return cb.Serial(classifier,
                   _Accuracy(),
                   _WeightedSequenceMean(),
                   name='SequenceAccuracy',
                   sublayers_to_print=[])


def CrossEntropyLoss():
  """Returns a layer that outputs multiclass prediction-target cross-entropy.

  DEPRECATED; refactor to use ``WeightedCategoryCrossEntropy`` or
  ``CategoryCrossEntropy`` instead.

  (``CrossEntropyLoss`` by itself does not compute cross-entropy. In older
  code, this layer had to be preceded by ``LogSoftmax``, and the two layers
  together did the work of converting category information to probability
  distributions and computing the cross-entropy between those distributions.
  All this is now done by ``WeightedCategoryCrossEntropy``.)
  """
  return cb.Serial(_CrossEntropy(),
                   _WeightedMean(),
                   name='CrossEntropyLoss',
                   sublayers_to_print=[])


def CrossEntropyLossWithLogSoftmax():
  """Mean prediction-target cross-entropy for multiclass classification."""
  return cb.Serial(core.LogSoftmax(), _CrossEntropy(), _WeightedMean(),
                   name='CrossEntropyLossWithLogSoftmax',
                   sublayers_to_print=[])


def BinaryCrossEntropyLoss():
  """Returns a layer that outputs binary prediction-target cross-entropy.

  DEPRECATED; refactor to use ``BinaryCrossEntropy`` instead. (The newer
  ``BinaryCrossEntropy`` does not use weights, so refactor accordingly. Unless
  and until clear motivating use cases arise, the library will not include a
  binary cross-entropy function with weights.)
  """
  return cb.Serial(_BinaryCrossEntropy(),
                   _WeightedMean(),
                   name='BinaryCrossEntropyLoss',
                   sublayers_to_print=[])


def L2Loss():
  r"""Returns a layer that computes an L2-like loss for one batch.

  The layer takes three inputs:

    - Model output from one batch, an ndarray of float-valued elements.

    - A batch of element-wise target values, which matches the shape of the
      model output.

    - A batch of weights, which matches the shape of the model output.

  The layer returns a weighted average of element-wise squared error terms
  :math:`(y_i - t_i)^2`.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    shapes.assert_same_shape(model_output, targets)
    shapes.assert_same_shape(model_output, weights)
    weighted_sse = weights * (model_output - targets)**2
    return jnp.sum(weighted_sse) / jnp.sum(weights)
  return base.Fn('L2Loss', f)


def SmoothL1Loss():
  r"""Returns a layer that computes a weighted, smoothed L1 loss for one batch.

  The layer takes three inputs:

    - Model output from one batch, an ndarray of float-valued elements.

    - A batch of element-wise target values, which matches the shape of the
      model output.

    - A batch of weights, which matches the shape of the model output.

  The layer computes a "smooth" L1 loss (a.k.a. Huber loss), for model output
  float :math:`y_i` and target float :math:`t_i`:

  .. math::
      \text{output} = \left\{ \begin{array}{cl}
          \frac 1 2 (y_i - t_i)^2, & \text{if}\ |y_i - t_i| < 1, \\
          |y_i - t_i| - \frac 1 2, & \text{otherwise}.
      \end{array} \right.

  The layer returns a weighted average of these element-wise values.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    shapes.assert_same_shape(model_output, targets)
    shapes.assert_same_shape(model_output, weights)
    l1_dist = jnp.abs(model_output - targets)
    smooth_dist = jnp.where(l1_dist < 1, 0.5 * l1_dist**2, l1_dist - 0.5)
    weighted_smooth_dist = weights * smooth_dist
    return jnp.sum(weighted_smooth_dist) / jnp.sum(weights)
  return base.Fn('SmoothL1Loss', f)


def WeightedSum():
  """Returns a layer that computes a weighted sum of the given values."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights)
  return base.Fn('WeightedSum', f)


def _Accuracy():
  """Returns a layer that scores predicted versus target category."""
  def f(predicted_category, target_category):  # pylint: disable=invalid-name
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_same_shape(predicted_category, target_category)
    return jnp.equal(predicted_category, target_category).astype(jnp.float32)
  return base.Fn('_Accuracy', f)


def _CrossEntropy():
  """Returns a layer that computes prediction-target cross entropies."""
  def f(model_output, target_category):  # pylint: disable=invalid-name
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_shape_equals(target_category, model_output.shape[:-1])
    target_distribution = core.one_hot(target_category, model_output.shape[-1])
    return -1.0 * jnp.sum(model_output * target_distribution, axis=-1)
  return base.Fn('_CrossEntropy', f)


def _BinaryCrossEntropy():
  """Returns a layer that computes prediction-target cross entropies."""
  def f(model_output, target_category):  # pylint: disable=invalid-name
    shapes.assert_same_shape(model_output, target_category)
    batch_size = model_output.shape[0]
    j = jnp.dot(jnp.transpose(target_category), jnp.log(model_output))
    j += jnp.dot(jnp.transpose(1 - target_category), jnp.log(1 - model_output))
    j = -1.0/batch_size * jnp.squeeze(j)
    return j
  return base.Fn('_BinaryCrossEntropy', f)


def CrossEntropySum():
  """Sum of prediction-target cross entropies for multiclass classification."""
  return cb.Serial(_CrossEntropy(),
                   WeightedSum(),
                   name='CrossEntropySum',
                   sublayers_to_print=[])


def BinaryCrossEntropySum():
  """Sum of prediction-target cross entropies for binary classification."""
  return cb.Serial(_BinaryCrossEntropy(),
                   WeightedSum(),
                   name='BinaryCrossEntropySum',
                   sublayers_to_print=[])
# pylint: enable=no-value-for-parameter


def _WeightedMean():
  """Returns a layer that computes a weighted mean of the given values."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights) / jnp.sum(weights)
  return base.Fn('_WeightedMean', f)


def _WeightedSequenceMean():
  """Returns a layer that computes a weighted sequence accuracy mean."""
  def f(values, weights):  # pylint: disable=invalid-name
    # This function assumes weights are 0 or 1.
    # Then compute 1: not-correct, 0: correct or masked
    not_correct = (1.0 - values) * weights
    axis_to_sum = list(range(1, len(not_correct.shape)))
    # Summing not-correct on all axes but batch. We're summing 0s and 1s,
    # so the sum is 0 if it's all 0 and >=1 in all other cases.
    not_correct_seq = jnp.sum(not_correct, axis=axis_to_sum)
    # Sequence is correct if not_correct_seq is 0, reverting here.
    correct_seq = 1.0 - jnp.minimum(1.0, not_correct_seq)
    return jnp.mean(correct_seq)  # Mean over batch.
  return base.Fn('_WeightedSequenceMean', f)


def _category_cross_entropy(  # pylint: disable=invalid-name
    model_output, targets, label_smoothing):
  """Computes category cross entropy with label smoothing."""
  n_categories = model_output.shape[-1]
  target_distributions = core.one_hot(targets, n_categories)
  if label_smoothing:
    if label_smoothing < 0. or label_smoothing > 1.:
      raise ValueError(
          f'Arg label_smoothing ({label_smoothing}) must be between 0 and 1.')
    target_distributions *= (1. - label_smoothing)
    target_distributions += label_smoothing / n_categories
  model_log_distributions = core.log_softmax(model_output)
  return - jnp.sum(target_distributions * model_log_distributions, axis=-1)
