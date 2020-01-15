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

"""Trax metrics layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax

from trax import math
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.math import numpy as np


@base.layer(n_in=2, n_out=1)
def L2(inputs, **unused_kwargs):
  """Returns a layer to compute L2 norms of predicted minus target vectors."""
  y_hat, y = inputs
  return np.sum((y_hat - y)**2, axis=-1)


@base.layer(n_in=2, n_out=1)
def Accuracy(inputs, axis=-1, **unused_kwargs):
  """Returns a layer to score matches of predicted versus target categories."""
  y_hat, target_category = inputs
  predicted_category = np.argmax(y_hat, axis=axis)
  return np.equal(predicted_category, target_category)


@base.layer(n_in=2, n_out=1)
def CrossEntropy(inputs, **unused_kwargs):
  """Returns a layer to compute prediction-target cross entropies."""
  y_hat, target_category = inputs
  return np.sum(y_hat * one_hot(target_category, y_hat.shape[-1]), axis=-1)


def L2Loss(id_to_mask=None, has_weights=False):
  """Returns a layer to computen L2 loss."""
  return MaskedScalar(L2(), id_to_mask=id_to_mask, has_weights=has_weights)  # pylint: disable=no-value-for-parameter


def CrossEntropyLoss(id_to_mask=None, has_weights=False):
  """Returns a layer to compute cross-entropy loss."""
  return cb.Serial(
      MaskedScalar(
          CrossEntropy(), id_to_mask=id_to_mask, has_weights=has_weights),  # pylint: disable=no-value-for-parameter
      base.Fn(lambda x: x * -1.0),
  )


NegLogPerplexityScalar = CrossEntropyLoss


def SumOfWeights(id_to_mask=None, has_weights=False):
  """Returns a layer to compute sum of weights of all non-masked elements."""
  multiply_by_weights = cb.Multiply() if has_weights else []
  return cb.Serial(
      cb.Drop(),  # Drop inputs.
      ElementMask(id_to_mask=id_to_mask),  # pylint: disable=no-value-for-parameter
      multiply_by_weights,
      core.Sum(axis=None)  # Sum all.
  )


def AccuracyScalar(id_to_mask=None, has_weights=False):
  """Returns an accuracy scalar metric layer (with masking and weights)."""
  return MaskedScalar(
      Accuracy(), id_to_mask=id_to_mask, has_weights=has_weights)  # pylint: disable=no-value-for-parameter


@base.layer()
def ElementMask(target, id_to_mask=0, **unused_kwargs):
  """Returns a mask with zeros for elements that don't belong in metrics."""
  if id_to_mask is None:
    return np.ones_like(target)
  return 1.0 - np.equal(target, id_to_mask).astype(np.float32)


@base.layer(n_in=2, n_out=1)
def WeightedMean(inputs, **unused_kwargs):
  metric, weights = inputs
  weights_sum = np.sum(weights)
  return np.sum(metric * weights) / weights_sum


def MaskedScalar(metric_layer, id_to_mask=None, has_weights=False):
  """Metric as scalar compatible with Trax masking."""
  # Stack of (inputs, targets) --> (metric, weight-mask).
  metric_and_mask = [
      cb.Parallel(
          [],
          cb.Dup()  # Duplicate targets
      ),
      cb.Parallel(
          metric_layer,  # Metric: (inputs, targets) --> metric
          ElementMask(id_to_mask=id_to_mask)  # pylint: disable=no-value-for-parameter
      )
  ]
  if not has_weights:
    # Take (metric, weight-mask) and return the weighted mean.
    return cb.Serial(metric_and_mask, WeightedMean())  # pylint: disable=no-value-for-parameter
  return cb.Serial(
      metric_and_mask,
      cb.Parallel(
          [],
          cb.Multiply()  # Multiply weights by masks
      ),
      WeightedMean()  # pylint: disable=no-value-for-parameter
  )


def one_hot(x, n_categories, dtype=np.float32):  # pylint: disable=invalid-name
  """Makes a one-hot array (n+1 dims) from an int-categorical array (n dims)."""
  indices_less_than_n = np.arange(n_categories)
  if math.backend_name() == 'jax':
    # Work around a jax broadcasting issue.
    indices_less_than_n = jax.lax.tie_in(x, indices_less_than_n)
  return np.array(x[..., np.newaxis] == indices_less_than_n, dtype)
