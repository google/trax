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
"""Utilities for serializing trajectories into discrete sequences."""

import functools

import gym
import numpy as onp

from trax import layers as tl
from trax.math import numpy as np
from trax.rl import space_serializer


# TODO(pkozakowski): Move the layers to trax.layers and remove this module.
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
  if remainder_length > 0:
    remainder = reprs[:, None, -remainder_length:]
    reprs = reprs[:, :-remainder_length]
  reprs = np.reshape(reprs, (batch_size, -1, x_size + y_size) + shape_suffix)
  x_reprs = reprs[:, :, :x_size]
  y_reprs = reprs[:, :, x_size:]
  if remainder_length > 0:
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
  # pylint: disable=no-value-for-parameter
  weigh_by_significance = [
      # (mask,)
      RepresentationMask(serializer=observation_serializer),
      # (repr_mask)
      SignificanceWeights(serializer=observation_serializer,
                          decay=significance_decay),
      # (mask, sig_weights)
  ]
  return tl.Serial(
      # (obs, act, obs, mask)
      tl.Parallel(Serialize(serializer=observation_serializer),
                  Serialize(serializer=action_serializer),
                  Serialize(serializer=observation_serializer)),
      # (obs_repr, act_repr, obs_repr, mask)
      Interleave(),
      # (obs_act_repr, obs_repr, mask)
      seq_model,
      # (obs_act_logits, obs_repr, mask)
      Deinterleave(x_size=observation_serializer.representation_length,
                   y_size=action_serializer.representation_length),
      # (obs_logits, act_logits, obs_repr, mask)
      tl.Parallel(None, tl.Drop(), None, weigh_by_significance),
      # (obs_logits, obs_repr, weights)
  )
  # pylint: enable=no-value-for-parameter


# TODO(pkozakowski): Figure out a more generic way to do this (submodel tags
# inside the model?).
def extract_inner_model(serialized_model):  # pylint: disable=invalid-name
  """Extracts the weights/state of the inner model from a SerializedModel."""
  return serialized_model[2]


def RawPolicy(seq_model, n_controls, n_actions):
  """Wraps a sequence model in a policy interface.

  The resulting model takes as input observation anc action sequences, but only
  uses the observations. Adds output heads for action logits and value
  predictions.

  Args:
    seq_model: Trax sequence model taking as input and outputting a sequence of
      continuous vectors.
    n_controls: Number of controls.
    n_actions: Number of action categories in each control.

  Returns:
    A model of signature (obs, act) -> (act_logits, values), with shapes:
      obs: (batch_size, length + 1, obs_depth)
      act: (batch_size, length, n_controls)
      act_logits: (batch_size, length, n_controls, n_actions)
      values: (batch_size, length)
  """
  @tl.layer()
  def SplitControls(x, **unused_kwargs):  # pylint: disable=invalid-name
    """Splits logits for actions in different controls."""
    return np.reshape(x, x.shape[:2] + (n_controls, n_actions))

  action_head = [
      # Predict all action logits at the same time.
      tl.Dense(n_controls * n_actions),
      # Then group them into separate controls, adding a new dimension.
      SplitControls(),  # pylint: disable=no-value-for-parameter
      tl.LogSoftmax(),
  ]
  return tl.Serial(                             # (obs, act)
      tl.Select([0], n_in=2),                   # (obs,)
      seq_model,                                # (obs_hidden,)
      tl.Dup(),                                 # (obs_hidden, obs_hidden)
      tl.Parallel(action_head, [tl.Dense(1),
                                tl.Flatten()])  # (act_logits, values)
  )


def substitute_inner_policy_raw(raw_policy, inner_policy):  # pylint: disable=invalid-name
  """Substitutes the weights/state of the inner model in a RawPolicy."""
  return raw_policy[:1] + [inner_policy] + raw_policy[2:]


def SerializedPolicy(
    seq_model, n_controls, n_actions, observation_serializer, action_serializer
):
  """Wraps a policy in serialization machinery for training.

  The resulting model takes as input observation and action sequences, and
  serializes them into one sequence similar to SerializedModel, before passing
  to the given sequence model. Adds output heads for action logits and value
  predictions.

  Args:
    seq_model: Trax sequence model taking as input a sequence of symbols and
      outputting a sequence of continuous vectors.
    n_controls: Number of controls.
    n_actions: Number of action categories in each control.
    observation_serializer: Serializer to use for observations.
    action_serializer: Serializer to use for actions.

  Returns:
    A model of signature (obs, act) -> (act_logits, values), same as in
    RawPolicy.
  """
  if action_serializer.representation_length != n_controls:
    raise ValueError(
        'Action symbols should correspond 1-1 to controls, but got {} '
        'controls and {} symbols.'.format(
            n_controls, action_serializer.representation_length
        )
    )

  @tl.layer()
  def FirstSymbol(x, **unused_kwargs):
    return x[:, :, 0]

  @tl.layer()
  def PadRight(x, n_to_pad, **unused_kwargs):
    pad_widths = [(0, 0), (0, n_to_pad)] + [(0, 0)] * (x.ndim - 2)
    return np.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
    )

  action_head = [
      tl.Dense(n_actions),
      tl.LogSoftmax(),
  ]
  value_head = [
      # Take just the vectors corresponding to the first action symbol.
      FirstSymbol(),  # pylint: disable=no-value-for-parameter
      # Predict values.
      tl.Dense(1),
      # Get rid of the singleton dimension.
      tl.Flatten(),
  ]
  # pylint: disable=no-value-for-parameter
  return tl.Serial(
      # (obs, act)
      tl.Parallel(Serialize(serializer=observation_serializer),
                  Serialize(serializer=action_serializer)),
      # (obs_repr, act_repr)
      Interleave(),
      # (obs_act_repr,)

      # Add one dummy action to the right - we'll use the output at its first
      # symbol to predict the value for the last observation.
      PadRight(n_to_pad=action_serializer.representation_length),

      # Shift one symbol to the right, so we predict the n-th action symbol
      # based on action symbols 1..n-1 instead of 1..n.
      tl.ShiftRight(),
      seq_model,
      # (obs_act_hidden,)
      Deinterleave(x_size=observation_serializer.representation_length,
                   y_size=action_serializer.representation_length),
      # (obs_hidden, act_hidden)
      tl.Select([1, 1]),
      # (act_hidden, act_hidden)
      tl.Parallel(action_head, value_head)
      # (act_logits, values)
  )
  # pylint: enable=no-value-for-parameter


def substitute_inner_policy_serialized(serialized_policy, inner_policy):  # pylint: disable=invalid-name
  """Substitutes the weights/state of the inner model in a SerializedPolicy."""
  return serialized_policy[:4] + [inner_policy] + serialized_policy[5:]


def analyze_action_space(action_space):  # pylint: disable=invalid-name
  """Returns the number of controls and actions for an action space."""
  assert isinstance(
      action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)
  ), 'Action space expected to be Discrete of MultiDiscrete, got {}.'.format(
      type(action_space)
  )
  if isinstance(action_space, gym.spaces.Discrete):
    n_actions = action_space.n
    n_controls = 1
  else:
    (n_controls,) = action_space.nvec.shape
    assert n_controls > 0
    assert onp.min(action_space.nvec) == onp.max(action_space.nvec), (
        'Every control must have the same number of actions.'
    )
    n_actions = action_space.nvec[0]
  return (n_controls, n_actions)


def wrap_policy(seq_model, observation_space, action_space, vocab_size):  # pylint: disable=invalid-name
  """Wraps a sequence model in either RawPolicy or SerializedPolicy.

  Args:
    seq_model: Trax sequence model.
    observation_space: Gym observation space.
    action_space: Gym action space.
    vocab_size: Either the number of symbols for a serialized policy, or None.

  Returns:
    RawPolicy if vocab_size is None, else SerializedPolicy.
  """
  (n_controls, n_actions) = analyze_action_space(action_space)
  if vocab_size is None:
    policy_wrapper = RawPolicy
  else:
    obs_serializer = space_serializer.create(observation_space, vocab_size)
    act_serializer = space_serializer.create(action_space, vocab_size)
    policy_wrapper = functools.partial(SerializedPolicy,
                                       observation_serializer=obs_serializer,
                                       action_serializer=act_serializer)
  return policy_wrapper(seq_model, n_controls, n_actions)


def substitute_inner_policy(wrapped_policy, inner_policy, vocab_size):  # pylint: disable=invalid-name
  """Substitutes the inner weights/state in a {Raw,Serialized}Policy.

  Args:
    wrapped_policy (pytree): Weights or state of a wrapped policy.
    inner_policy (pytree): Weights or state of an inner policy.
    vocab_size (int or None): Vocabulary size of a serialized policy, or None
      in case of a raw policy.

  Returns:
    New weights or state of wrapped_policy, with the inner weights/state
      copied from inner_policy.
  """
  if vocab_size is None:
    substitute_fn = substitute_inner_policy_raw
  else:
    substitute_fn = substitute_inner_policy_serialized
  return substitute_fn(wrapped_policy, inner_policy)
