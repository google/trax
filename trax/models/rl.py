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
"""Policy networks."""

from trax import layers as tl
from trax import models
from trax.fastmath import numpy as jnp


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
    vocab_size=2,
    multiplicative_action_injection=False,
    head_init_range=None,
):
  """Attaches a value head to a model body."""
  if body is None:
    body = lambda mode: []
  if normalizer is None:
    normalizer = lambda mode: []

  def ActionInjector(mode):
    if inject_actions:
      if is_discrete:
        action_encoder = tl.Embedding(vocab_size, inject_actions_dim)
      else:
        action_encoder = tl.Dense(inject_actions_dim)
      encoders = tl.Parallel(
          tl.Dense(inject_actions_dim),
          action_encoder,
      )
      if multiplicative_action_injection:
        action_injector = tl.Serial(
            tl.Fn('TanhMulGate', lambda x, a: x * jnp.tanh(a)),
            tl.LayerNorm()  # compensate for reduced variance
        )
      else:
        action_injector = tl.Add()
      return tl.Serial(
          # Input: (body output, actions).
          encoders,
          action_injector,
          models.MLP(
              layer_widths=(inject_actions_dim,) * inject_actions_n_layers,
              out_activation=True,
              flatten=False,
              mode=mode,
          )
      )
    else:
      return []

  head_kwargs = {}
  if head_init_range is not None:
    head_kwargs['kernel_initializer'] = tl.RandomUniformInitializer(
        lim=head_init_range
    )

  return tl.Serial(
      _Batch(normalizer(mode=mode), batch_axes),
      _Batch(body(mode=mode), batch_axes),
      ActionInjector(mode=mode),
      tl.Dense(1, **head_kwargs),
  )


def PolicyAndValue(
    policy_distribution,
    body=None,
    policy_top=Policy,
    value_top=Value,
    normalizer=None,
    joint=True,
    mode='train',
):
  """Attaches policy and value heads to a model body."""
  if normalizer is None:
    normalizer = lambda mode: []
  if body is None:
    body = lambda mode: []

  common_kwargs = {'body': None, 'normalizer': None, 'mode': mode}
  policy_top = policy_top(
      policy_distribution=policy_distribution, **common_kwargs
  )
  value_top = value_top(**common_kwargs)

  layers = [normalizer(mode=mode)]
  if joint:
    layers.extend([
        body(mode=mode),
        tl.Branch(policy_top, value_top),
    ])
  else:
    layers.append(tl.Branch(
        tl.Serial(body(mode=mode), policy_top),
        tl.Serial(body(mode=mode), value_top),
    ))
  return tl.Serial(layers)


def Quality(
    body=None,
    normalizer=None,
    batch_axes=None,
    mode='train',
    n_actions=2,
    head_init_range=None,
):
  """The network takes as input an observation and outputs values of actions."""

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
      tl.Dense(n_actions, **head_kwargs),
  )
