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

"""PPO in JAX.

Notation:

B, scalar   - batch size
T, scalar   - number of actions in a trajectory
C, scalar   - number of controls (dimensions) in an action
OBS, tuple  - shape of a singular observation from the environment.
             Ex: For CartPole-v0 this is (4,) and Pong-v0 it's (210, 160, 3)
A, scalar   - Number of actions, assuming a discrete space.

Policy and Value function signatures:

Policy            :: ([B, T + 1] + OBS, [B, T + 1, C]) ->  [B, T + 1, C, A]
Value             :: ([B, T + 1] + OBS, [B, T + 1, C]) ->  [B, T + 1]
Policy and Value  :: ([B, T + 1] + OBS, [B, T + 1, C]) -> ([B, T + 1, C, A],
                                                           [B, T + 1])

i.e. the policy net should take a batch of *trajectories* and at each time-step
in each batch deliver a probability distribution over actions.

NOTE: It doesn't return logits, rather the expectation is that it returns
log-probabilities instead.

NOTE: The policy and value functions need to take care to not take into account
future time-steps while deciding the actions (or value) for the current
time-step.

Policy and Value Function produces a tuple of the expected output of a policy
function and a value function.

"""

import functools
import itertools

import jax
from jax import numpy as np
import numpy as onp


def rewards_to_go(rewards, mask, gamma):
  r"""Computes rewards to go.

  Reward to go is defined as follows, the discounted reward that we have to
  yet collect, going forward from this point, i.e.:

  r2g_t = \sum_{l=0}^{\infty} (\gamma^{l} * reward_{t+l})

  Args:
    rewards: np.ndarray of shape (B, T) of rewards.
    mask: np.ndarray of shape (B, T) of mask for the rewards.
    gamma: float, discount factor.

  Returns:
    rewards to go, np.ndarray of shape (B, T).
  """
  B, T = rewards.shape  # pylint: disable=invalid-name,unused-variable

  masked_rewards = rewards * mask  # (B, T)

  # The lax.scan version of this is slow, but we still show it here for
  # completeness.
  #   rewards_rev = np.flip(masked_rewards, axis=1)  # (B, T) flipped on time.
  #   rrt = np.transpose(rewards_rev)  # (T, B) transpose to scan over time.
  #
  #   def discounting_add(carry, reward):
  #     x = reward + (gamma * carry)
  #     return x, x
  #
  #   _, ys = lax.scan(discounting_add,
  #                    np.zeros_like(rrt[0], dtype=np.float32),
  #                    rrt.astype(np.float32))
  #
  #   # ys is (T, B) and T is in reverse order.
  #   return np.flip(np.transpose(ys), axis=1)

  # We use the following recurrence relation, derived from the equation above:
  #
  # r2g[t+1] = (r2g[t] - r[t]) / gamma
  #
  # This means we'll need to calculate r2g[0] first and then r2g[1] and so on ..
  #
  # **However** this leads to overflows for long sequences: r2g[t] - r[t] > 0
  # and gamma < 1.0, so the division keeps increasing.
  #
  # So we just run the recurrence in reverse, i.e.
  #
  # r2g[t] = r[t] + (gamma*r2g[t+1])
  #
  # This is much better, but might have lost updates since the (small) rewards
  # at earlier time-steps may get added to a (very?) large sum.

  # Compute r2g_{T-1} at the start and then compute backwards in time.
  r2gs = [masked_rewards[:, -1]]

  # Go from T-2 down to 0.
  for t in reversed(range(T - 1)):
    r2gs.append(masked_rewards[:, t] + (gamma * r2gs[-1]))

  # The list should have length T.
  assert T == len(r2gs)

  # First we stack them in the correct way to make it (B, T), but these are
  # still from newest (T-1) to oldest (0), so then we flip it on time axis.
  return np.flip(np.stack(r2gs, axis=1), axis=1)


@jax.jit
def value_loss_given_predictions(value_prediction,
                                 rewards,
                                 reward_mask,
                                 gamma,
                                 epsilon,
                                 value_prediction_old=None):
  """Computes the value loss given the prediction of the value function.

  Args:
    value_prediction: np.ndarray of shape (B, T+1, 1)
    rewards: np.ndarray of shape (B, T) of rewards.
    reward_mask: np.ndarray of shape (B, T), the mask over rewards.
    gamma: float, discount factor.
    epsilon: float, clip-fraction, used if value_value_prediction_old isn't None
    value_prediction_old: np.ndarray of shape (B, T+1, 1) of value predictions
      using the old parameters. If provided, we incorporate this in the loss as
      well. This is from the OpenAI baselines implementation.

  Returns:
    Pair (value_loss, summaries), where value_loss is the average L2 value loss,
      averaged over instances where reward_mask is 1. Summaries is a dict of
      summaries collected during value loss computation.
  """

  B, T = rewards.shape  # pylint: disable=invalid-name
  assert (B, T) == reward_mask.shape
  assert (B, T + 1) == value_prediction.shape

  value_prediction = value_prediction[:, :-1] * reward_mask  # (B, T)
  r2g = rewards_to_go(rewards, reward_mask, gamma=gamma)  # (B, T)
  loss = (value_prediction - r2g)**2

  # From the baselines implementation.
  if value_prediction_old is not None:
    value_prediction_old = value_prediction_old[:, :-1] * reward_mask  # (B, T)

    v_clipped = value_prediction_old + np.clip(
        value_prediction - value_prediction_old, -epsilon, epsilon)
    v_clipped_loss = (v_clipped - r2g)**2
    loss = np.maximum(v_clipped_loss, loss)

  # Take an average on only the points where mask != 0.
  value_loss = np.sum(loss) / np.sum(reward_mask)

  summaries = {
      'value_loss': value_loss,
  }

  return (value_loss, summaries)


def deltas(predicted_values, rewards, mask, gamma):
  r"""Computes TD-residuals from V(s) and rewards.

  Where a `delta`, i.e. a td-residual is defined as:

  delta_{b,t} = r_{b,t} + \gamma * v_{b,t+1} - v_{b,t}.

  Args:
    predicted_values: ndarray of shape (B, T+1). NOTE: Expects axis 2 was
      squeezed. These represent V(s_bt) for b < B and t < T+1
    rewards: ndarray of shape (B, T) of rewards.
    mask: ndarray of shape (B, T) of mask for rewards.
    gamma: float, discount factor.

  Returns:
    ndarray of shape (B, T) of one-step TD-residuals.
  """

  # Predicted values at time t, cutting off the last to have shape (B, T).
  predicted_values_bt = predicted_values[:, :-1]
  # Predicted values at time t+1, by cutting off the first to have shape (B, T)
  predicted_values_btplus1 = predicted_values[:, 1:]
  # Return the deltas as defined above.
  return (rewards +
          (gamma * predicted_values_btplus1) - predicted_values_bt) * mask


def gae_advantages(td_deltas, mask, lambda_, gamma):
  r"""Computes the GAE advantages given the one step TD-residuals.

  The formula for a GAE advantage estimator is as follows:

  A_{bt} = \sum_{l=0}^{\infty}(\gamma * \lambda)^{l}(\delta_{b,t+l}).

  Internally we just call rewards_to_go, since it is the same computation.

  Args:
    td_deltas: np.ndarray of shape (B, T) of one step TD-residuals.
    mask: np.ndarray of shape (B, T) of mask for the residuals. It maybe the
      case that the `td_deltas` are already masked correctly since they are
      produced by `deltas(...)`
    lambda_: float, lambda parameter for GAE estimators.
    gamma: float, lambda parameter for GAE estimators.

  Returns:
    GAE advantage estimates.
  """

  return rewards_to_go(td_deltas, mask, lambda_ * gamma)


def chosen_probabs(probab_actions, actions):
  """Picks out the probabilities of the actions along batch and time-steps.

  Args:
    probab_actions: ndarray of shape `[B, T, C, A]`, where
      probab_actions[b, t, i] contains the log-probability of action = i at
      the t^th time-step in the b^th trajectory.
    actions: ndarray of shape `[B, T]`, with each entry in [0, A) denoting
      which action was chosen in the b^th trajectory's t^th time-step.

  Returns:
    `[B, T, C, A]` ndarray with the log-probabilities of the chosen actions.
  """
  B, T, C = actions.shape  # pylint: disable=invalid-name
  assert (B, T, C) == probab_actions.shape[:3]
  return probab_actions[
      np.arange(B)[:, None, None], np.arange(T)[:, None], np.arange(C), actions
  ]


def compute_probab_ratios(p_new, p_old, actions, action_mask):
  """Computes the probability ratios for each time-step in a trajectory.

  Args:
    p_new: ndarray of shape [B, T, A] of the log-probabilities that the
      policy network assigns to all the actions at each time-step in each batch
      using the old parameters.
    p_old: ndarray of shape [B, T, A], same as above, but using old policy
      network parameters.
    actions: ndarray of shape [B, T] where each element is from [0, A).
    action_mask: ndarray of shape [B, T] masking over probabilities.

  Returns:
    probab_ratios: ndarray of shape [B, T], where
    probab_ratios_{b,t,} = p_new_{b,t,action_{b,t}} /
                           p_old_{b,t,action_{b,t}}
  """

  B, T, C = actions.shape  # pylint: disable=invalid-name
  assert (B, T, C) == p_old.shape[:3]
  assert (B, T, C) == p_new.shape[:3]

  logp_old = chosen_probabs(p_old, actions)
  logp_new = chosen_probabs(p_new, actions)

  assert (B, T, C) == logp_old.shape
  assert (B, T, C) == logp_new.shape

  # Since these are log-probabilities, we just subtract them.
  probab_ratios = np.exp(logp_new - logp_old) * action_mask
  assert (B, T, C) == probab_ratios.shape
  return probab_ratios


def clipped_probab_ratios(probab_ratios, epsilon):
  return np.clip(probab_ratios, 1 - epsilon, 1 + epsilon)


def clipped_objective(probab_ratios, advantages, action_mask, epsilon):
  advantages = advantages
  return np.minimum(
      probab_ratios * advantages,
      clipped_probab_ratios(probab_ratios, epsilon=epsilon) *
      advantages) * action_mask


@jax.jit
def ppo_loss_given_predictions(log_probab_actions_new,
                               log_probab_actions_old,
                               value_predictions_old,
                               padded_actions,
                               padded_rewards,
                               reward_mask,
                               gamma,
                               lambda_,
                               epsilon):
  """PPO objective, with an eventual minus sign, given predictions."""
  # The last timestep has been cut here.
  B, T, C, A = log_probab_actions_old.shape  # pylint: disable=invalid-name

  assert (B, T) == padded_rewards.shape  # pylint: disable=invalid-name
  assert (B, T) == padded_rewards.shape
  assert (B, T, C) == padded_actions.shape
  assert (B, T) == reward_mask.shape

  assert (B, T + 1) == value_predictions_old.shape
  assert (B, T, C, A) == log_probab_actions_old.shape
  assert (B, T, C, A) == log_probab_actions_new.shape

  # (B, T)
  td_deltas = deltas(
      value_predictions_old,  # (B, T+1)
      padded_rewards,
      reward_mask,
      gamma=gamma)

  # (B, T)
  advantages = gae_advantages(
      td_deltas, reward_mask, lambda_=lambda_, gamma=gamma)

  # Normalize the advantages.
  advantage_mean = np.mean(advantages)
  advantage_std = np.std(advantages)
  advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

  # Broadcast advantages and reward action over controls.
  advantages = advantages[:, :, None]
  action_mask = reward_mask[:, :, None]

  # (B, T, C)
  ratios = compute_probab_ratios(log_probab_actions_new, log_probab_actions_old,
                                 padded_actions, action_mask)
  assert (B, T, C) == ratios.shape

  # (B, T, C)
  objective = clipped_objective(
      ratios, advantages, action_mask, epsilon=epsilon)
  assert (B, T, C) == objective.shape

  # ()
  average_objective = np.sum(objective) / np.sum(reward_mask)

  # Loss is negative objective.
  ppo_loss = -average_objective

  summaries = {
      'ppo_loss': ppo_loss,
      'advantage_mean': advantage_mean,
      'advantage_std': advantage_std,
  }

  return (ppo_loss, summaries)


@jax.jit
def combined_loss_given_predictions(log_probab_actions_new,
                                    log_probab_actions_old,
                                    value_prediction_new,
                                    value_prediction_old,
                                    padded_actions,
                                    padded_rewards,
                                    reward_mask,
                                    gamma,
                                    lambda_,
                                    value_weight,
                                    entropy_weight,
                                    epsilon):
  """Computes the combined (clipped loss + value loss) given predictions."""
  (value_loss, value_summaries) = value_loss_given_predictions(
      value_prediction_new,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      value_prediction_old=value_prediction_old,
      epsilon=epsilon)
  (ppo_loss, ppo_summaries) = ppo_loss_given_predictions(
      log_probab_actions_new,
      log_probab_actions_old,
      value_prediction_old,
      padded_actions,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      lambda_=lambda_,
      epsilon=epsilon)
  entropy_bonus = masked_entropy(log_probab_actions_new, reward_mask)
  combined_loss_ = ppo_loss + (value_weight * value_loss) - (
      entropy_weight * entropy_bonus)

  summaries = {
      'combined_loss': combined_loss_,
      'entropy_bonus': entropy_bonus,
  }
  for loss_summaries in (value_summaries, ppo_summaries):
    summaries.update(loss_summaries)

  return (combined_loss_, (ppo_loss, value_loss, entropy_bonus), summaries)


@functools.partial(jax.jit, static_argnums=(3,))
def combined_loss(new_weights,
                  log_probab_actions_old,
                  value_predictions_old,
                  policy_and_value_net_apply,
                  padded_observations,
                  padded_actions,
                  padded_rewards,
                  reward_mask,
                  nontrainable_params,
                  state=None,
                  rng=None):
  """Computes the combined (clipped loss + value loss) given observations."""
  gamma = nontrainable_params['gamma']
  lambda_ = nontrainable_params['lambda']
  value_weight = nontrainable_params['value_weight']
  entropy_weight = nontrainable_params['entropy_weight']
  epsilon = nontrainable_params['epsilon']

  # TODO(pkozakowski): Pass the actual actions here, to enable autoregressive
  # action sampling.
  dummy_actions = np.zeros_like(padded_actions)
  (log_probab_actions_new, value_predictions_new) = (
      policy_and_value_net_apply(
          (padded_observations, dummy_actions),
          weights=new_weights,
          state=state,
          rng=rng,
      )
  )
  # Cut off the last extra action to obtain shape (B, T, C, A).
  log_probab_actions_new = log_probab_actions_new[:, :-1]

  (loss, component_losses, summaries) = combined_loss_given_predictions(
      log_probab_actions_new,
      log_probab_actions_old,
      value_predictions_new,
      value_predictions_old,
      padded_actions,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      lambda_=lambda_,
      value_weight=value_weight,
      entropy_weight=entropy_weight,
      epsilon=epsilon,
  )
  return (loss, component_losses, summaries, state)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def policy_and_value_opt_step(i,
                              opt_state,
                              opt_update,
                              get_params,
                              policy_and_value_net_apply,
                              log_probab_actions_old,
                              value_predictions_old,
                              padded_observations,
                              padded_actions,
                              padded_rewards,
                              reward_mask,
                              nontrainable_params,
                              state=None,
                              rng=None):
  """Policy and Value optimizer step."""

  # Combined loss function given the new params.
  def policy_and_value_loss(params, state):
    """Returns the combined loss given just parameters."""
    (loss, _, _, state) = combined_loss(
        params,
        log_probab_actions_old,
        value_predictions_old,
        policy_and_value_net_apply,
        padded_observations,
        padded_actions,
        padded_rewards,
        reward_mask,
        nontrainable_params,
        state=state,
        rng=rng)
    return loss, state

  new_weights = get_params(opt_state)
  g, state = jax.grad(policy_and_value_loss, has_aux=True)(new_weights, state)
  # TODO(afrozm): Maybe clip gradients?
  return opt_update(i, g, opt_state), state


def approximate_kl(log_prob_new, log_prob_old, mask):
  """Computes the approximate KL divergence between the old and new log-probs.

  Args:
    log_prob_new: (B, T, C, A) log probs new
    log_prob_old: (B, T, C, A) log probs old
    mask: (B, T)

  Returns:
    Approximate KL.
  """
  diff = log_prob_old - log_prob_new
  # Mask out the irrelevant part.
  diff *= mask[:, :, None, None]  # make mask (B, T, 1, 1)
  # Average on non-masked part.
  return np.sum(diff) / np.sum(mask)


def masked_entropy(log_probs, mask):
  """Computes the entropy for the given log-probs.

  Args:
    log_probs: (B, T, C, A) log probs
    mask: (B, T) mask.

  Returns:
    Entropy.
  """
  # Mask out the irrelevant part.
  lp = log_probs * mask[:, :, None, None]  # make mask (B, T, 1, 1)
  p = np.exp(lp) * mask[:, :, None, None]  # (B, T, 1, 1)
  # Average on non-masked part and take negative.
  return -(np.sum(lp * p) / np.sum(mask))


def shuffled_index_batches(dataset_size, batch_size):
  """Generates batches of shuffled indices over a dataset."""
  def shuffled_indices():
    while True:
      perm = onp.random.permutation(dataset_size)
      for x in perm:
        yield x

  indices = shuffled_indices()
  while True:
    yield onp.array(list(itertools.islice(indices, int(batch_size))))
