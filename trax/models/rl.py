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
from trax import models


def _Batch(x, batch_axes):
  """Combine all but batch_axes last axes into batch to avoid shape problems."""
  if batch_axes is None:
    return x
  if isinstance(x, list) and not x:
    return []
  return tl.BatchLeadingAxes(x, n_last_axes_to_keep=batch_axes)


def Policy(
    policy_distribution,
    body=None,
    normalizer=None,
    head_init_range=None,
    batch_axes=None,
    mode='train',
):
  """Attaches a policy head to a model body."""
  if body is None:
    body = lambda mode: []
  if normalizer is None:
    normalizer = lambda mode: []

  head_kwargs = {}
  if head_init_range is not None:
    head_kwargs['kernel_initializer'] = tl.RandomUniformInitializer(
        lim=head_init_range
    )

  return tl.Serial(
      _Batch(normalizer(mode=mode), batch_axes),
      _Batch(body(mode=mode), batch_axes),
      tl.Dense(policy_distribution.n_inputs, **head_kwargs),
  )


def Value(
    body=None,
    normalizer=None,
    inject_actions=False,
    inject_actions_n_layers=1,
    inject_actions_dim=64,
    batch_axes=None,
    mode='train',
    is_discrete=False,
    vocab_size=2
):
  """Attaches a value head to a model body."""
  if body is None:
    body = lambda mode: []
  if normalizer is None:
    normalizer = lambda mode: []

  def ActionInjector(mode):
    if inject_actions:
      if is_discrete:
        encode_layer = tl.Parallel(
            tl.Dense(inject_actions_dim),
            tl.Embedding(vocab_size, inject_actions_dim)
        )
      else:
        encode_layer = tl.Parallel(
            tl.Dense(inject_actions_dim),
            tl.Dense(inject_actions_dim),
        )
      return tl.Serial(
          # Input: (body output, actions).
          encode_layer,
          tl.Add(),
          models.PureMLP(
              layer_widths=(inject_actions_dim,) * inject_actions_n_layers,
              out_activation=True,
              flatten=False,
              mode=mode,
          )
      )
    else:
      return []

  return tl.Serial(
      _Batch(normalizer(mode=mode), batch_axes),
      _Batch(body(mode=mode), batch_axes),
      ActionInjector(mode=mode),
      tl.Dense(1),
  )


def PolicyAndValue(
    policy_distribution,
    body=None,
    policy_top=Policy,
    value_top=Value,
    normalizer=None,
    head_init_range=None,
    mode='train',
):
  """Attaches policy and value heads to a model body."""

  head_kwargs = {}
  if head_init_range is not None:
    head_kwargs['kernel_initializer'] = tl.RandomUniformInitializer(
        lim=head_init_range
    )

  if normalizer is None:
    normalizer = lambda mode: []
  if body is None:
    body = lambda mode: []
  return tl.Serial(
      normalizer(mode=mode),
      body(mode=mode),
      tl.Branch(
          policy_top(policy_distribution=policy_distribution, mode=mode),
          value_top(mode=mode),
      ),
  )
