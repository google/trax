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
"""Utilities for AWR."""

from typing import List, Optional, Union

import numpy as onp
from trax.rl.trajectory import replay_buffer


def pad_array_to_length(list_of_ndarrays: List[onp.ndarray],
                        t_final: int,
                        back: bool = True):
  """Pad list of (t_variable, *SHAPE) elements to (t_final, *SHAPE)."""
  # Each element of `list_of_ndarrays` is of shape: (t, *underlying_shape)
  # where `t` changes but `underlying_shape` doesn't.
  underlying_shape = list_of_ndarrays[0].shape[1:]
  batch = len(list_of_ndarrays)
  padded_objs = onp.zeros(shape=(batch, t_final, *underlying_shape))
  mask = onp.ones(shape=(batch, t_final))
  padding_config = [(0, 0)] * len(list_of_ndarrays[0].shape)
  for i, obs in enumerate(list_of_ndarrays):
    cfg = (0, t_final - obs.shape[0]) if back else (t_final - obs.shape[0], 0)
    padding_config[0] = cfg
    padded_objs[i] = onp.pad(obs, padding_config, 'constant')
    if back:
      mask[i, obs.shape[0]:] = 0
    else:
      mask[i, :t_final - obs.shape[0]] = 0

  return padded_objs, mask


def replay_buffer_to_padded_observations(rb: replay_buffer.ReplayBuffer,
                                         idx: Union[List[int],
                                                    onp.ndarray] = None,
                                         boundary: Optional[int] = None):
  """Gets the trajectories in the buffer at indices pads them."""
  if idx is None:
    idx = rb.get_unrolled_indices()

  idx = onp.asarray(idx)
  assert len(idx.shape) == 1

  observations = [
      rb.get(replay_buffer.ReplayBuffer.OBSERVATIONS_KEY,
             idx[start_idx:end_plus_1_idx])
      for (start_idx, end_plus_1_idx) in rb.iterate_over_paths(idx)
  ]

  # Every element in observations is [t, *OBS] where the `t` can vary.
  t_max = max(len(o) for o in observations)  # ex: t_max: 1500
  if boundary is None:
    # e such that 10^e <= t_max < 10^(e+1)
    e = onp.floor(onp.log10(t_max))  # ex: e: 3
    # we will deal in integer multiples of boundary
    boundary = 10**e  # ex: boundary: 1000

  # m (int) such that (m-1) * boundary < t_max <= m * boundary
  m = onp.ceil(t_max / boundary).astype(onp.int32)  # ex: m: 2
  t_final = int(m * boundary)  # t_final: 2000

  # observations[0]'s shape is (t,) + OBS, where OBS is the core observation's
  # shape.
  return pad_array_to_length(observations, t_final)


def padding_length(observations, boundary=None):
  """Returns the padding length optionally given a boundary."""
  # Every element in observations is [t, *OBS] where the `t` can vary.
  t_max = max(len(o) for o in observations)  # ex: t_max: 1500
  if boundary is None:
    # e such that 10^e <= t_max < 10^(e+1)
    e = onp.floor(onp.log10(t_max))  # ex: e: 3
    # we will deal in integer multiples of boundary
    boundary = 10**e  # ex: boundary: 1000

  # m (int) such that (m-1) * boundary < t_max <= m * boundary
  m = onp.ceil(t_max / boundary).astype(onp.int32)  # ex: m: 2
  t_final = int(m * boundary)  # t_final: 2000
  return t_final


def replay_buffer_to_padded_rewards(rb, idx, t_final):
  """Pad replay buffer's rewards at indices to specified size."""
  rewards = [
      rb.get('rewards', idx[start_idx:end_plus_1_idx][:-1])
      for (start_idx, end_plus_1_idx) in rb.iterate_over_paths(idx)
  ]
  return pad_array_to_length(rewards, t_final)


def compute_td_lambda_return(rewards: onp.ndarray, value_preds: onp.ndarray,
                             gamma: float, td_lambda: float) -> onp.ndarray:
  """Computes td-lambda returns."""
  (t_final,) = rewards.shape
  if value_preds.shape != (t_final + 1,):
    raise ValueError(
        f'Shapes are not as expected: rewards.shape {rewards.shape}'
        f'value_preds.shape = {value_preds.shape}'
    )

  td_lambda_return = onp.zeros_like(rewards)
  td_lambda_return[-1] = rewards[-1] + (gamma * value_preds[-1])

  for i in reversed(range(0, t_final - 1)):
    td_lambda_return[i] = rewards[i] + (
        gamma * ((1 - td_lambda) * value_preds[i + 1] +
                 td_lambda * td_lambda_return[i + 1]))

  return td_lambda_return


def batched_compute_td_lambda_return(padded_rewards: onp.ndarray,
                                     padded_rewards_mask: onp.ndarray,
                                     value_preds: onp.ndarray,
                                     value_preds_mask: onp.ndarray,
                                     gamma: float, td_lambda: float):
  """Computes td-lambda returns, in a batched manner."""
  batch, t = padded_rewards.shape
  if ((batch, t) != padded_rewards_mask.shape or
      ((batch, t + 1) != value_preds.shape) or
      ((batch, t + 1) != value_preds_mask.shape)):
    raise ValueError(
        f'Shapes are not as expected: batch {batch}, t {t}'
        f'padded_rewards_mask.shape = {padded_rewards_mask.shape}'
        f'value_preds.shape = {value_preds.shape}'
        f'value_preds_mask.shape = {value_preds_mask.shape}'
    )

  bool_padded_rewards_mask = padded_rewards_mask == 1
  bool_value_preds_mask = value_preds_mask == 1

  td_lambda_returns = []
  for b in range(batch):
    td_lambda_returns.append(
        compute_td_lambda_return(
            padded_rewards[b][bool_padded_rewards_mask[b]],
            value_preds[b][bool_value_preds_mask[b]],
            gamma,
            td_lambda,
        ))
  return td_lambda_returns
