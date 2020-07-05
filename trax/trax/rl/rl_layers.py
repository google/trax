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

from trax.fastmath import numpy as jnp


def ValueLoss(values, returns, value_loss_coeff):
  """Definition of the loss of the value function."""
  advantages = returns - values
  l2_value_loss = jnp.mean(advantages**2) * value_loss_coeff
  return l2_value_loss


def ExplainedVariance(values, returns):
  """Definition of explained variance - an approach from OpenAI baselines."""
  assert returns.shape == values.shape, (
      f'returns.shape was {returns.shape} and values.shape was {values.shape}')
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
  # dist_inputs of the shape float32[128,1,18]
  # actions of the shape int32[128,1]
  # and old_log_probs of the shape float32[128,1]
  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  assert new_log_probs.shape == old_log_probs.shape, (
      f'new_log_probs.shape was {new_log_probs.shape} and'
      f'old_log_probs.shape was {old_log_probs.shape}')
  # The ratio between new_probs and old_probs expressed
  # using log_probs and exponentaion
  probs_ratio = jnp.exp(new_log_probs - old_log_probs)
  return probs_ratio


def ApproximateKLDivergence(dist_inputs, actions, old_log_probs, log_prob_fun):
  """Probability Ratio from the PPO algorithm."""
  new_log_probs = NewLogProbs(dist_inputs, actions, log_prob_fun)
  assert new_log_probs.shape == old_log_probs.shape, (
      f'new_log_probs.shape was {new_log_probs.shape} and'
      f'old_log_probs.shape was {old_log_probs.shape}')
  approximate_kl_divergence = 0.5 * \
      jnp.mean(new_log_probs - old_log_probs) ** 2
  return approximate_kl_divergence


def UnclippedObjective(probs_ratio, advantages):
  """Unclipped Objective from the PPO algorithm."""
  assert probs_ratio.shape == advantages.shape, (
      f'probs_ratio.shape was {probs_ratio.shape} and'
      f'advantages.shape was {advantages.shape}')
  unclipped_objective = probs_ratio * advantages
  return unclipped_objective


def ClippedObjective(probs_ratio, advantages, epsilon):
  """Clipped Objective from the PPO algorithm."""
  assert probs_ratio.shape == advantages.shape, (
      f'probs_ratio.shape was {probs_ratio.shape} and'
      f'advantages.shape was {advantages.shape}')
  clipped_objective = jnp.clip(probs_ratio, 1 - epsilon,
                               1 + epsilon) * advantages
  assert probs_ratio.shape == clipped_objective.shape, (
      f'probs_ratio.shape was {probs_ratio.shape} and'
      f'clipped_objective.shape was {clipped_objective.shape}')
  return clipped_objective


def PPOObjective(dist_inputs, values, returns, dones, rewards,
                 actions, old_log_probs, log_prob_fun, epsilon,
                 normalize_advantages):
  """PPO Objective."""
  # dist_inputs of the shape float32[128,1,18]
  # values of the shape float32[128,1,1]
  # returns of the shape float32[128,1,1]
  # dones of the shape float32[128,1,1]
  # rewards of the shape int32[128,1,1]
  # actions of the shape int32[128,1]
  # and old_log_probs of the shape float32[128,1]
  returns = returns.squeeze(axis=2)
  values = values.squeeze(axis=2)
  dones = dones.squeeze(axis=2)
  rewards = rewards.squeeze(axis=2)
  assert rewards.shape == dones.shape, (
      f'rewards.shape was {rewards.shape} and dones.shape was {dones.shape}')
  assert dones.shape == values.shape, (
      f'dones.shape was {dones.shape} and values.shape was {values.shape}')
  assert returns.shape == values.shape, (
      f'returns.shape was {returns.shape} and values.shape was {values.shape}')
  assert returns.shape == old_log_probs.shape, (
      f'returns.shape was {returns.shape} and'
      f'old_log_probs.shape was {old_log_probs.shape}')

  probs_ratio = ProbsRatio(dist_inputs, actions, old_log_probs, log_prob_fun)
  assert probs_ratio.shape == old_log_probs.shape, (
      f'probs_ratio.shape was {probs_ratio.shape} and'
      f'old_log_probs.shape was {old_log_probs.shape}')

  # jaxified versions of
  # returns[dones] = rewards[dones]
  # values[dones] = 0
  returns = jnp.where(dones, rewards, returns)
  values = jnp.where(dones, jnp.zeros_like(values), values)
  advantages = returns - values
  if normalize_advantages:
    advantages = advantages - jnp.mean(advantages)
    advantages /= jnp.std(advantages) + 1e-8
  assert old_log_probs.shape == advantages.shape, (
      f'old_log_probs.shape was {old_log_probs.shape} and advantages.shape was '
      f'{advantages.shape}')

  unclipped_objective = UnclippedObjective(probs_ratio, advantages)
  assert unclipped_objective.shape == advantages.shape, (
      f'old_log_probs.shape was {old_log_probs.shape} and'
      f'unclipped_objective.shape was {unclipped_objective.shape}')

  clipped_objective = ClippedObjective(probs_ratio, advantages, epsilon)
  assert clipped_objective.shape == advantages.shape, (
      f'clipped_objective.shape was {clipped_objective.shape} and'
      f'advantages.shape was {advantages.shape}')

  ppo_objective = jnp.minimum(unclipped_objective, clipped_objective)
  assert ppo_objective.shape == advantages.shape, (
      f'ppo_objective.shape was {ppo_objective.shape} and'
      f'advantages.shape was {advantages.shape}')

  return ppo_objective


def A2CObjective(dist_inputs, values, returns, dones, rewards,
                 actions, mask, log_prob_fun, normalize_advantages):
  """Definition of the Advantage Actor Critic (A2C) loss."""
  # dist_inputs of the shape float32[128,1,18]
  # values of the shape float32[128,1,1]
  # returns of the shape float32[128,1,1]
  # dones of the shape int32[128,1,1]
  # actions of the shape int32[128,1]
  # and mask of the shape float32[128,1]
  # We have to squeeze values and returns, because we
  # are planning to compute (return - values) * new_log_probs * mask
  # and all of them should be of the same dimension
  values = values.squeeze(axis=2)
  returns = returns.squeeze(axis=2)
  dones = dones.squeeze(axis=2)
  rewards = rewards.squeeze(axis=2)
  assert rewards.shape == dones.shape, (
      f'rewards.shape was {rewards.shape} and dones.shape was {dones.shape}')
  assert dones.shape == values.shape, (
      f'dones.shape was {dones.shape} and values.shape was {values.shape}')
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

  # jaxified versions of
  # returns[dones] = rewards[dones]
  # values[dones] = 0
  returns = jnp.where(dones, rewards, returns)
  values = jnp.where(dones, jnp.zeros_like(values), values)
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
