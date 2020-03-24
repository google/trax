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
"""Policy networks."""

from trax import layers as tl


def Policy(policy_distribution, body=None, mode='train'):
  """Attaches a policy head to a model body."""
  if body is None:
    body = lambda mode: []
  return tl.Serial(
      body(mode=mode),
      tl.Dense(policy_distribution.n_inputs),
  )


def Value(body=None, mode='train'):
  """Attaches a value head to a model body."""
  if body is None:
    body = lambda mode: []
  return tl.Serial(
      body(mode=mode),
      tl.Dense(1),
  )


def PolicyAndValue(
    policy_distribution,
    body=None,
    policy_top=Policy,
    value_top=Value,
    mode='train',
):
  """Attaches policy and value heads to a model body."""
  if body is None:
    body = lambda mode: []
  return tl.Serial(
      body(mode=mode),
      tl.Branch(
          policy_top(policy_distribution=policy_distribution, mode=mode),
          value_top(mode=mode),
      ),
  )
