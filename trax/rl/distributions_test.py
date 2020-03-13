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
"""Tests for initializers."""

from absl.testing import absltest

from trax.math import numpy as np
from trax.rl import distributions


class DistributionsTest(absltest.TestCase):

  def test_categorical(self):
    n_categories = 3
    distribution = distributions.Categorical(n_categories)
    self.assertEqual(distribution.n_inputs, n_categories)
    inputs = np.random.random(distribution.n_inputs)
    point = distribution.sample(inputs)
    self.assertEqual(point.shape, ())
    log_prob = distribution.log_prob(point)
    self.assertEqual(log_prob.shape, ())
