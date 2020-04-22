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
"""Normalization helpers."""

import numpy as np


class RunningMean:
  """Numerically stable online mean calculation."""

  def __init__(self):
    self._mean = 0
    self._n = 0

  @property
  def mean(self):
    return self._mean

  @property
  def count(self):
    return self._n

  def update(self, x):
    self._mean = float(self._n) / (self._n + 1) * self._mean + x / (self._n + 1)
    self._n += 1


class RunningMeanAndVariance:
  """Numerically stable online variance calculation."""

  def __init__(self):
    self._running_mean = RunningMean()
    self._running_variance = RunningMean()

  @property
  def mean(self):
    return self._running_mean.mean

  @property
  def variance(self):
    return self._running_variance.mean

  @property
  def count(self):
    return self._running_mean.count
  
  def update(self, x):
    old_mean = self._running_mean.mean
    self._running_mean.update(x)
    new_mean = self._running_mean.mean

    self._running_variance.update((x - new_mean) * (x - old_mean))


class Normalizer:
  """Numerically stable normalization."""

  def __init__(self, sample_limit=float('+inf'), epsilon=1e-5):
    self._stats = RunningMeanAndVariance()
    self._sample_limit = sample_limit
    self._epsilon = epsilon

  def update(self, x):
    if self._stats.count < self._sample_limit:
      self._stats.update(x)

  def normalize(self, x):
    return (
        (x - self._stats.mean) / (np.sqrt(self._stats.variance) + self._epsilon)
    )
