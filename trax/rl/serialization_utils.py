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

"""Utilities for serializing trajectories into discrete sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from trax import layers as tl
from trax import shapes
from trax.math import numpy as np


# TODO(pkozakowski): Start using those layers directly instead of through
# serialize_observations_and_actions, then move them to trax.layers and remove
# this module.
@tl.layer()
def Serialize(x, serializer, **unused_kwargs):
  """Serializes a given array."""
  (batch_size, length) = x.shape[:2]
  shape_suffix = x.shape[2:]
  x = np.reshape(x, (batch_size * length,) + shape_suffix)
  x = serializer.serialize(x)
  return np.reshape(x, (batch_size, -1, serializer.representation_length,))


@tl.layer(n_in=2, n_out=1)
def Interleave(inputs, **unused_kwargs):
  """Interleaves and flattens two serialized sequences.

  The first sequence can be longer by 1 than the second one. This is so we can
  interleave sequences of observations and actions, when there's 1 extra
  observation at the end.

  For serialized sequences [[x_1_1, ..., x_1_R1], ..., [x_L1_1, ..., x_L1_R1]]
  and [[y_1_1, ..., y_1_R2], ..., [y_L2_1, ..., y_L2_R2]], where L1 = L2 + 1,
  the result is [x_1_1, ..., x_1_R1, y_1_1, ..., y_1_R2, ..., x_L2_1, ...,
  x_L2_R1, y_L2_1, ..., y_L2_R2, x_L1_1, ..., x_L1_R1] (batch dimension omitted
  for clarity).

  Args:
    inputs: Pair of sequences of shapes (B, L1, R1) and (B, L2, R2), where B
      is batch size, L* is the length of the sequence and R* is the
      representation length of each element in the sequence.

  Returns:
    Interleaved sequence of shape (B, L1 * R1 + L2 * R2).
  """
  (x, y) = inputs
  (batch_size, _, _) = x.shape
  (_, length, _) = y.shape
  assert x.shape[1] in (length, length + 1)

  reprs = np.concatenate((x[:, :length], y), axis=2)
  reprs = np.reshape(reprs, (batch_size, -1))
  remainder = np.reshape(x[:, length:], (batch_size, -1))
  return np.concatenate((reprs, remainder), axis=1)


@tl.layer(n_in=1, n_out=2)
def Deinterleave(inputs, x_size, y_size, **unused_kwargs):
  """Inverse of Interleave."""
  reprs = inputs
  (batch_size, length) = reprs.shape[:2]
  shape_suffix = reprs.shape[2:]
  remainder_length = length % (x_size + y_size)
  remainder = reprs[:, None, -remainder_length:]
  reprs = reprs[:, :-remainder_length]
  reprs = np.reshape(reprs, (batch_size, -1, x_size + y_size) + shape_suffix)
  x_reprs = reprs[:, :, :x_size]
  y_reprs = reprs[:, :, x_size:]
  x_reprs = np.concatenate((x_reprs, remainder), axis=1)
  return (x_reprs, y_reprs)


@tl.layer()
def RepresentationMask(mask, serializer, **unused_kwargs):
  """Upsamples a mask to cover the serialized representation."""
  # Trax enforces the mask to be of the same size as the target. Get rid of the
  # extra dimensions.
  mask = np.amax(mask, axis=tuple(range(2, mask.ndim)))
  return np.broadcast_to(
      mask[:, :, None], mask.shape + (serializer.representation_length,)
  )


@tl.layer()
def SignificanceWeights(mask, serializer, decay, **unused_kwargs):
  """Multiplies a binary mask with a symbol significance mask."""
  # (repr,) -> (batch, length, repr)
  significance = serializer.significance_map[None, None]
  return mask * decay ** np.broadcast_to(significance, mask.shape)


def SerializedModel(
    seq_model,
    observation_serializer,
    action_serializer,
    significance_decay,
):
  """Wraps a world model in serialization machinery for training.

  The resulting model takes as input the observation and action sequences,
  serializes them and interleaves into one sequence, which is fed into a given
  autoregressive model. The resulting logit sequence is deinterleaved into
  observations and actions, and the observation logits are returned together
  with computed symbol significance weights.

  Args:
    seq_model: Trax autoregressive model taking as input a sequence of symbols
      and outputting a sequence of symbol logits.
    observation_serializer: Serializer to use for observations.
    action_serializer: Serializer to use for actions.
    significance_decay: Float from (0, 1) for exponential weighting of symbols
      in the representation.

  Returns:
    A model of signature
    (obs, act, obs, mask) -> (obs_logits, obs_repr, weights), where obs are
    observations (the second occurrence is the target), act are actions, mask is
    the observation mask, obs_logits are logits of the output observation
    representation, obs_repr is the target observation representation and
    weights are the target weights.
  """
  weigh_by_significance = [     # (mask,)
      RepresentationMask(  # pylint: disable=no-value-for-parameter
          serializer=observation_serializer,
      ),                        # (repr_mask)
      SignificanceWeights(  # pylint: disable=no-value-for-parameter
          serializer=observation_serializer,
          decay=significance_decay,
      ),                        # (mask, sig_weights)
  ]
  return tl.Serial([            # (obs, act, obs, mask)
      tl.Parallel(
          Serialize(serializer=observation_serializer),  # pylint: disable=no-value-for-parameter
          Serialize(serializer=action_serializer),  # pylint: disable=no-value-for-parameter
          Serialize(serializer=observation_serializer),  # pylint: disable=no-value-for-parameter
      ),                        # (obs_repr, act_repr, obs_repr, mask)
      Interleave(  # pylint: disable=no-value-for-parameter
      ),                        # (obs_act_repr, obs_repr, mask)
      seq_model,                # (obs_act_logits, obs_repr, mask)
      Deinterleave(  # pylint: disable=no-value-for-parameter
          x_size=observation_serializer.representation_length,
          y_size=action_serializer.representation_length,
      ),                        # (obs_logits, act_logits, obs_repr, mask)
      tl.Parallel(
          None, tl.Drop(), None, weigh_by_significance
      ),                        # (obs_logits, obs_repr, weights)
  ])


# TODO(pkozakowski): Figure out a more generic way to do this (submodel tags
# inside the model?).
def extract_inner_model(serialized_model):  # pylint: disable=invalid-name
  """Extracts the weights/state of the inner model from a SerializedModel."""
  return serialized_model[2]


def serialize_observations_and_actions(  # pylint: disable=invalid-name
    observations,
    actions,
    observation_serializer,
    action_serializer,
    representation_length,
):
  """Serializes observations and actions into a discrete sequence.

  Args:
    observations: Array (B, T + 1, ...), of observations, where B is the batch
      size and T is the number of timesteps excluding the last observation.
    actions: Array (B, T, ...) of actions.
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      sequence is padded up to this number.
  Returns:
    Serialized sequence of shape (B, R) where R = representation_length.
  """
  (batch_size, n_timesteps) = actions.shape[:2]
  assert observations.shape[:2] == (batch_size, n_timesteps + 1)

  serialization = tl.Serial([
      tl.Parallel(
          Serialize(serializer=observation_serializer),  # pylint: disable=no-value-for-parameter
          Serialize(serializer=action_serializer),  # pylint: disable=no-value-for-parameter
      ),
      Interleave(),  # pylint: disable=no-value-for-parameter
  ])
  serialization.init(shapes.signature((observations, actions)))
  reprs = serialization((observations, actions))

  assert reprs.shape[1] <= representation_length
  return np.pad(
      reprs,
      pad_width=((0, 0), (0, representation_length - reprs.shape[1])),
      mode='constant',
  )


def observation_mask(  # pylint: disable=invalid-name
    observation_serializer, action_serializer, representation_length
):
  """Calculates an observation mask for a serialized sequence.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      mask is padded up to this number.

  Returns:
    Binary mask indicating which symbols in the representation correspond to
    observations.
  """
  mask = onp.zeros(representation_length, dtype=np.int32)
  obs_repr_length = observation_serializer.representation_length
  step_repr_length = obs_repr_length + action_serializer.representation_length
  for step_start_index in range(0, representation_length, step_repr_length):
    mask[step_start_index:(step_start_index + obs_repr_length)] = 1
  return mask


def action_mask(  # pylint: disable=invalid-name
    observation_serializer, action_serializer, representation_length
):
  """Calculates an action mask for a serialized sequence.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      mask is padded up to this number.

  Returns:
    Binary mask indicating which symbols in the representation correspond to
    actions.
  """
  return 1 - observation_mask(
      observation_serializer, action_serializer, representation_length
  )


def rewards_to_actions_map(  # pylint: disable=invalid-name
    observation_serializer,
    action_serializer,
    n_timesteps,
    representation_length,
):
  """Calculates a mapping between the rewards and the serialized sequence.

  Used to broadcast advantages over the log-probabilities of corresponding
  actions.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    n_timesteps: Number of timesteps (length of the reward sequence).
    representation_length: Number of symbols in the serialized sequence.

  Returns:
    Array (T, R) translating from the reward sequence to actions in the
    representation.
  """
  r2a_map = onp.zeros((n_timesteps, representation_length))
  obs_repr_length = observation_serializer.representation_length
  act_repr_length = action_serializer.representation_length
  step_repr_length = obs_repr_length + act_repr_length
  for t in range(n_timesteps):
    act_start_index = t * step_repr_length + obs_repr_length
    r2a_map[t, act_start_index:(act_start_index + act_repr_length)] = 1
  return r2a_map
