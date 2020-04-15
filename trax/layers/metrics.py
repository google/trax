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
"""Trax metrics layers.

Trax computes metrics (loss functions and evaluation metrics) using layers.
A metrics layer takes 2 or 3 batch inputs:

  - output values (vectors)
  - target values (vectors or scalars)
  - weights [optional]

and gives a single scalar as output. Trax reduces batch values to a scalar by
taking the weighted (and often also masked) mean of those values:

  - `L2Loss`: weighted masked mean of L2 of (prediction_vector - target_vector)

  - `AccuracyScalar`: weighted masked mean of category predictions
    (argmax(prediction_vector) vs. target_category)

  - `CrossEntropyLoss`: weighted masked mean of pairwise cross entropy of
    (prediction_vector, target_vector)


TODO(jonni): Explain masks and weighting.
"""

import jax

from trax import math
from trax import shapes
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.math import numpy as np


# pylint: disable=no-value-for-parameter
@base.layer(n_in=3, n_out=1)
def L2Loss(inputs, **unused_kwargs):
  y_hat, y, mask = inputs
  shapes.assert_same_shape(y_hat, y)
  shapes.assert_same_shape(y, mask)
  l2 = mask * (y_hat - y)**2
  return np.sum(l2) / np.sum(mask)


def AccuracyScalar():
  """Computes weighted masked mean of category prediction accuracy."""
  return _WeightedMaskedMean(_Accuracy())


def SequenceAccuracyScalar():
  """Computes weighted masked mean of sequence prediction accuracy."""
  return _WeightedMaskedMean(_Accuracy(),
                             final_layer_override=_WeightedSequenceMean())


def CrossEntropyLoss():
  """Computes weighted masked mean of prediction-target cross entropies."""
  return _WeightedMaskedMean(_CrossEntropy())


def CrossEntropySum():
  """Computes weighted masked sum of prediction-target cross entropies."""
  return _WeightedMaskedMean(_CrossEntropy(),
                             final_layer_override=WeightedSum())


def SumOfWeights():
  """Returns a layer to compute sum of weights of all non-masked elements."""
  return cb.Serial(
      cb.Drop(),  # Drop inputs.
      cb.Drop(),  # Drop targets.
      core.Sum(axis=None)  # Sum weights.
  )
# pylint: enable=no-value-for-parameter


@base.layer(n_in=2, n_out=1)
def _Accuracy(inputs, axis=-1, **unused_kwargs):
  """Returns a layer to score matches of predicted versus target categories."""
  y_hat, target_category = inputs
  predicted_category = np.argmax(y_hat, axis=axis)
  # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
  # shapes.assert_same_shape(predicted_category, target_category)
  return np.equal(predicted_category, target_category).astype(np.float32)


@base.layer(n_in=2, n_out=1)
def _CrossEntropy(inputs, **unused_kwargs):
  """Returns a layer to compute prediction-target cross entropies."""
  y_hat, target_category = inputs
  # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
  # shapes.assert_shape_equals(target_category, y_hat.shape[:-1])
  return -1.0 * np.sum(y_hat * one_hot(target_category, y_hat.shape[-1]),
                       axis=-1)


@base.layer(n_in=2, n_out=1)
def _WeightedMean(inputs, **unused_kwargs):
  """Returns a layer to compute weighted mean over all values in the input."""
  values, weights = inputs
  return np.sum(values * weights) / np.sum(weights)


@base.layer(n_in=2, n_out=1)
def WeightedSum(inputs, **unused_kwargs):
  """Returns a layer to compute weighted sum over all values in the input."""
  values, weights = inputs
  return np.sum(values * weights)


@base.layer(n_in=2, n_out=1)
def _WeightedSequenceMean(inputs, **unused_kwargs):
  """Returns a layer to compute weighted seqeunce accuracy mean."""
  values, weights = inputs  # This function assumes weights are 0 or 1.
  not_correct = (1.0 - values) * weights  # 1: not-correct, 0: correct or masked
  axis_to_sum = list(range(1, len(not_correct.shape)))
  # Summing not-correct on all axes but batch. We're summing 0s and 1s,
  # so the sum is 0 if it's all 0 and >=1 in all other cases.
  not_correct_seq = np.sum(not_correct, axis=axis_to_sum)
  # Sequence is correct if not_correct_seq is 0, reverting here.
  correct_seq = 1.0 - np.minimum(1.0, not_correct_seq)
  return np.mean(correct_seq)  # Mean over batch.


# pylint: disable=no-value-for-parameter
def _WeightedMaskedMean(metric_layer, final_layer_override=None):
  """Computes weighted masked mean of metric_layer(predictions, targets)."""
  final_layer = final_layer_override or _WeightedMean()  # For sequence acc.
  return cb.Serial(
      metric_layer,
      final_layer
  )
# pylint: enable=no-value-for-parameter


# TODO(jonni): Figure out the right name and home for this function.
def one_hot(x, n_categories, dtype=np.float32):  # pylint: disable=invalid-name
  """Makes a one-hot array (n+1 dims) from an int-categorical array (n dims)."""
  indices_less_than_n = np.arange(n_categories)
  if math.backend_name() == 'jax':
    # Work around a jax broadcasting issue.
    indices_less_than_n = jax.lax.tie_in(x, indices_less_than_n)
  return np.array(x[..., np.newaxis] == indices_less_than_n, dtype)
