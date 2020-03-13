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
"""Probability distributions for RL training in Trax."""

import gym

from trax import layers as tl
from trax.math import numpy as np


class Distribution:
  """Abstract class for parametrized probability distributions."""

  @property
  def n_inputs(self):
    """Returns the number of inputs to the distribution (i.e. parameters)."""
    raise NotImplementedError

  def sample(self, inputs):
    """Samples a point from the distribution.

    Args:
      inputs (np.ndarray): Distribution inputs. Shape is subclass-specific.
        Broadcasts along the first dimensions. For example, in the categorical
        distribution parameter shape is (C,), where C is the number of
        categories. If (B, C) is passed, the object will represent a batch of B
        categorical distributions with different parameters.

    Returns:
      Sampled point of shape dependent on the subclass and on the shape of
      inputs.
    """
    raise NotImplementedError

  def log_prob(self, inputs, point):
    """Retrieves log probability (or log probability density) of a point.

    Args:
      inputs (np.ndarray): Distribution parameters.
      point (np.ndarray): Point from the distribution. Shape should be
        consistent with inputs.

    Returns:
      Array of log probabilities of points in the distribution.
    """
    raise NotImplementedError

  def LogProb(self):  # pylint: disable=invalid-name
    @tl.layer(n_in=2, n_out=1)
    def Layer(x, **unused_kwargs):  # pylint: disable=invalid-name
      """Builds a log probability layer for this distribution."""
      (inputs, point) = x
      return self.log_prob(inputs, point)
    return Layer()  # pylint: disable=no-value-for-parameter


class Categorical(Distribution):
  """Categorical distribution parametrized by logits."""

  def __init__(self, n_categories):
    self._n_categories = n_categories

  @property
  def n_inputs(self):
    return self._n_inputs

  def sample(self, inputs):
    return tl.gumbel_sample(inputs)

  def log_prob(self, inputs, point):
    # Flatten the prefix dimensions for easy indexing.
    flat_point = np.reshape(point, -1)
    flat_inputs = np.reshape(inputs, (point.size, -1))
    flat_log_probs = flat_inputs[np.arange(point.size), flat_point.astype(int)]
    return np.reshape(flat_log_probs, point.shape)


# TODO(pkozakowski): Implement GaussianDistribution,
# GaussianMixtureDistribution.


def create_distribution(space):
  """Creates a Distribution for the given Gym space."""
  if isinstance(space, gym.spaces.Discrete):
    return Categorical(space.n)
  else:
    raise TypeError('Space {} unavailable as a distribution support.')


def LogLoss(distribution, has_weights, **unused_kwargs):  # pylint: disable=invalid-name
  """Builds a log loss layer for a Distribution."""
  return tl.Serial([
      distribution.LogProb(),
      tl.Negate(),
      tl.WeightedSum() if has_weights else tl.Sum(),
  ])
