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

from trax import fastmath
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.base import Fn


def L2Loss():
  def f(y_hat, y, mask):  # pylint: disable=invalid-name
    shapes.assert_same_shape(y_hat, y)
    shapes.assert_same_shape(y, mask)
    l2 = mask * (y_hat - y)**2
    return jnp.sum(l2) / jnp.sum(mask)
  return Fn('L2Loss', f)


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


def _Accuracy(axis=-1):
  """Returns a layer to score matches of predicted versus target categories."""
  def f(y_hat, target_category):  # pylint: disable=invalid-name
    predicted_category = jnp.argmax(y_hat, axis=axis)
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_same_shape(predicted_category, target_category)
    return jnp.equal(predicted_category, target_category).astype(jnp.float32)
  return Fn('_Accuracy', f)


def _CrossEntropy():
  """Returns a layer to compute prediction-target cross entropies."""
  def f(y_hat, target_category):  # pylint: disable=invalid-name
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_shape_equals(target_category, y_hat.shape[:-1])
    return -1.0 * jnp.sum(y_hat * one_hot(target_category, y_hat.shape[-1]),
                          axis=-1)
  return Fn('_CrossEntropy', f)


def _WeightedMean():
  """Returns a layer to compute weighted mean over all values in the input."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights) / jnp.sum(weights)
  return Fn('_WeightedMean', f)


def WeightedSum():
  """Returns a layer to compute weighted sum over all values in the input."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights)
  return Fn('WeightedSum', f)


def _WeightedSequenceMean():
  """Returns a layer to compute weighted seqeunce accuracy mean."""
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
  return Fn('_WeightedSequenceMean', f)


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
def one_hot(x, n_categories, dtype=jnp.float32):  # pylint: disable=invalid-name
  """Makes a one-hot array (n+1 dims) from an int-categorical array (n dims)."""
  indices_less_than_n = jnp.arange(n_categories)
  if fastmath.backend_name() == 'jax':
    # Work around a jax broadcasting issue.
    indices_less_than_n = jax.lax.tie_in(x, indices_less_than_n)
  return jnp.array(x[..., jnp.newaxis] == indices_less_than_n, dtype)
