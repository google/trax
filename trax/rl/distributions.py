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

import gin
import gym
import numpy as onp

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

  def __init__(self, shape, n_categories):
    """Initializes Categorical distribution.

    Args:
      shape (tuple): Shape of the sample.
      n_categories (int): Number of categories.
    """
    self._shape = shape
    self._n_categories = n_categories

  @property
  def n_inputs(self):
    return np.prod(self._shape, dtype=np.int32) * self._n_categories

  def _unflatten_inputs(self, inputs):
    return np.reshape(
        inputs, inputs.shape[:-1] + self._shape + (self._n_categories,)
    )

  def sample(self, inputs):
    return tl.gumbel_sample(self._unflatten_inputs(inputs))

  def log_prob(self, inputs, point):
    # TODO(pkozakowski): Put log softmax here. For now we assume that the
    # network output activation is log softmax, preventing the use of the same
    # architecture across tasks with different action spaces.
    inputs = self._unflatten_inputs(inputs)
    return np.sum(
        # Select the logits specified by point.
        inputs * tl.one_hot(point, self._n_categories),
        # Sum over the parameter dimensions.
        axis=[-a for a in range(1, len(self._shape) + 2)],
    )


@gin.configurable(blacklist=['shape'])
class Gaussian(Distribution):
  """Independent multivariate Gaussian distribution parametrized by mean."""

  def __init__(self, shape, std=1.0):
    """Initializes Gaussian distribution.

    Args:
      shape (tuple): Shape of the sample.
      std (float): Standard deviation, shared across the whole sample.
    """
    self._shape = shape
    self._std = std

  @property
  def n_inputs(self):
    return np.prod(self._shape)

  def sample(self, inputs):
    return onp.random.normal(
        loc=np.reshape(inputs, inputs.shape[:-1] + self._shape),
        scale=self._std,
    )

  def log_prob(self, inputs, point):
    point = point.reshape(inputs.shape[:-1] + (-1,))
    return (
        # L2 term.
        -np.sum((point - inputs) ** 2, axis=-1) / (2 * self._std ** 2) -
        # Normalizing constant.
        (np.log(self._std) + np.log(np.sqrt(2 * np.pi))) * np.prod(self._shape)
    )


# TODO(pkozakowski): Implement GaussianMixture.


def create_distribution(space):
  """Creates a Distribution for the given Gym space."""
  if isinstance(space, gym.spaces.Discrete):
    return Categorical(shape=(), n_categories=space.n)
  elif isinstance(space, gym.spaces.MultiDiscrete):
    assert space.nvec.size
    assert min(space.nvec) == max(space.nvec), (
        'Every dimension must have the same number of categories, got '
        '{}.'.format(space.nvec)
    )
    return Categorical(shape=(len(space.nvec),), n_categories=space.nvec[0])
  elif isinstance(space, gym.spaces.Box):
    return Gaussian(shape=space.shape)
  else:
    raise TypeError('Space {} unavailable as a distribution support.')


def LogLoss(distribution, has_weights, **unused_kwargs):  # pylint: disable=invalid-name
  """Builds a log loss layer for a Distribution."""
  return tl.Serial([
      distribution.LogProb(),
      tl.Negate(),
      tl.WeightedSum() if has_weights else tl.Sum(),
  ])
