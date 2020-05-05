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
"""A number of RL functions intended to be later wrapped as Trax layers.

  Wrapping happens with help of the function tl.Fn.
"""

from trax.math import numpy as jnp


def ValueLoss(values, returns, value_loss_coeff):
  """Definition of the loss of the value function."""
  advantages = returns - values
  l2_value_loss = jnp.mean(advantages**2) * value_loss_coeff
  return l2_value_loss


def ExplainedVariance(values, returns):
  """Definition of explained variance - an approach from OpenAI baselines."""
  values = values.squeeze()
  returns = returns.squeeze()
  assert values.shape[0] == returns.shape[0]
  # TODO(henrykm): it would be good to explain the relation with the time dim.
  returns_variance = jnp.var(returns)
  explained_variance = 1 - jnp.var(returns-values)/returns_variance
  return explained_variance


def PreferredMove(dist_inputs, sample):
  """Definition of the preferred move."""
  preferred_moves = sample(dist_inputs, temperature=0.0)
  return jnp.mean(preferred_moves)


def NewLogProbs(dist_inputs, actions, log_prob_fun):
  """Given distribution and actions calculate log probs."""
  new_log_probs = log_prob_fun(dist_inputs,
                               actions)
  # we want to reformat from e.g. [[-2.8527899], [-2.8768425]]
  # to [-2.8527899, -2.8768425]
  # new_log_probs = new_log_probs.flatten()
  return new_log_probs


# TODO(henrykm): Clarify how jnp.mean is applied.
def EntropyLoss(dist_inputs, actions, log_prob_fun,
                entropy_coeff, entropy_fun):
  """Definition of the Entropy Layer."""
  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  entropy_loss = entropy_fun(new_log_probs) * entropy_coeff
  return jnp.mean(entropy_loss)


def ProbsRatio(dist_inputs, actions, old_log_probs, log_prob_fun):
  """Probability Ratio from the PPO algorithm."""
  # Old log probs have an undesirable extra dimension which we remove here
  old_log_probs = jnp.array(old_log_probs.squeeze(axis=-1),
                            dtype=jnp.float32)
  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  # The ratio between new_probs and old_probs expressed
  # using log_probs and exponentaion
  probs_ratio = jnp.exp(new_log_probs - old_log_probs)
  return probs_ratio


def ApproximateKLDivergence(dist_inputs, actions, old_log_probs, log_prob_fun):
  """Probability Ratio from the PPO algorithm."""
  # TODO(henrykm): Clarify the old_log_probs and squeezing
  # Old log probs have an undesirable extra dimension which we remove here
  old_log_probs = jnp.array(old_log_probs.squeeze(axis=-1),
                            dtype=jnp.float32)
  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  # The ratio between new_probs and old_probs expressed
  # using log_probs and exponentaion
  approximate_kl_divergence = 0.5 * \
      jnp.mean(new_log_probs - old_log_probs) ** 2
  return approximate_kl_divergence


def UnclippedObjective(probs_ratio, advantages):
  """Unclipped Objective from the PPO algorithm."""
  unclipped_objective = probs_ratio * advantages
  return unclipped_objective


def ClippedObjective(probs_ratio, advantages, epsilon):
  """Clipped Objective from the PPO algorithm."""
  clipped_objective = jnp.clip(probs_ratio, 1 - epsilon,
                               1 + epsilon) * advantages
  return clipped_objective


def PPOObjective(dist_inputs, values, returns, actions, old_log_probs,
                 log_prob_fun, epsilon, normalize_advantages):
  """PPO Objective."""
  # Returns and values are arriving with two extra dimensions
  # TODO(henrykm): remove these dimensions at an earlier stage?
  returns = returns.squeeze()
  values = values.squeeze()
  probs_ratio = ProbsRatio(dist_inputs, actions, old_log_probs, log_prob_fun)
  advantages = returns - values
  if normalize_advantages:
    advantages = advantages - jnp.mean(advantages)
    advantages /= jnp.std(advantages) + 1e-8
  unclipped_objective = UnclippedObjective(probs_ratio, advantages)
  clipped_objective = ClippedObjective(probs_ratio, advantages, epsilon)
  ppo_objective = jnp.minimum(unclipped_objective, clipped_objective)
  return ppo_objective


def A2CObjective(dist_inputs, values, returns,
                 actions, mask, log_prob_fun, normalize_advantages):
  """Definition of the Advantage Actor Critic (A2C) loss."""
  # values of the shape float32[128,1,1]
  # returns of the shape float32[128,1,1]
  # actions of the shape int32[128,1]
  # and mask of the shape float32[128,1]
  # We have to squeeze values and returns, because we
  # are planning to compute (return - values) * new_log_probs * mask
  # and all of them should be of the same dimension
  values = values.squeeze(axis=2)
  returns = returns.squeeze(axis=2)

  assert returns.shape == values.shape, (
      f'returns.shape was {returns.shape} and values.shape was {values.shape}')
  assert values.shape == mask.shape, (
      f'values.shape was {values.shape} and mask.shape was {mask.shape}')
  assert returns.shape[0] == dist_inputs.shape[0], (
      f'returns.shape[0] was {returns.shape[0]} and dist_inputs.shape[0] was '
      f'{dist_inputs.shape[0]}')

  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  assert new_log_probs.shape == mask.shape, (
      f'new_log_probs.shape was {new_log_probs.shape} and mask.shape was '
      f'{mask.shape}')

  advantages = returns - values
  if normalize_advantages:
    advantages = advantages - jnp.mean(advantages)
    advantages /= jnp.std(advantages) + 1e-8
  assert new_log_probs.shape == advantages.shape, (
      f'new_log_probs.shape was {new_log_probs.shape} and advantages.shape was '
      f'{advantages.shape}')

  # One of the motivation to the squeezes and assertions is to
  # avoid [128,1] * [128,1,1] * [128] multiplications in the definition
  # of the a2c objective - we insist on the same shapes
  a2c_objective = -jnp.sum(new_log_probs * advantages * mask) / jnp.sum(mask)
  return a2c_objective
