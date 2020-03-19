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

"""Utilities for policy-based trainers."""

import collections
import functools
import os
import re
import time

from absl import logging
import gym
from jax import lax
import numpy as onp
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
import tensorflow as tf

from trax import history as trax_history
from trax import layers as tl
from trax import utils
from trax.math import numpy as np
from trax.math import random as trax_random
from trax.rl import serialization_utils


def policy_and_value_net(
    bottom_layers_fn, observation_space, action_space, vocab_size, two_towers
):
  """A policy and value net function.

  Runs bottom_layers_fn either as a single network or as two separate towers.
  Attaches action and value heads and wraps the network in a policy wrapper.

  Args:
    bottom_layers_fn: Trax model to use as a policy network.
    observation_space (gym.Space): Observation space.
    action_space (gym.Space): Action space.
    vocab_size (int or None): Vocabulary size to use with a SerializedPolicy
      wrapper. If None, RawPolicy will be used.
    two_towers (bool): Whether to run bottom_layers_fn as two separate towers
      for action and value prediction.

  Returns:
    Pair (network, substitute_fn), where network is the final network and
      substitute_fn is a function (wrapped_tree, inner_tree) -> wrapped_tree
      for substituting weights or state of the constructed model based on the
      weights or state of a model returned from bottom_layers_fn. substitute_fn
      is used for initializing the policy from parameters of a world model.
  """
  kwargs = {}
  if vocab_size is not None:
    kwargs['vocab_size'] = vocab_size

  def wrapped_policy_fn():
    return serialization_utils.wrap_policy(
        bottom_layers_fn(**kwargs),
        observation_space,
        action_space,
        vocab_size,
    )

  # Now, with the current logits, one head computes action probabilities and the
  # other computes the value function.
  # NOTE: The LogSoftmax instead of the Softmax because of numerical stability.
  if two_towers:
    # Two towers: run two two-head networks in parallel and drop one head from
    # each.
    net = tl.Serial([             # (obs, act)
        tl.Select([0, 1, 0, 1]),  # (obs, act, obs, act)
        tl.Parallel(
            wrapped_policy_fn(),
            wrapped_policy_fn(),
        ),                        # (act_logits_1, vals_1, act_logits_2, vals_2)
        tl.Select([0, 3]),        # (act_logits_1, vals_2)
    ])
    def substitute_fn(wrapped_policy, inner_policy):
      return (
          wrapped_policy[:1] + [tuple(
              # Substitute in both towers.
              serialization_utils.substitute_inner_policy(  # pylint: disable=g-complex-comprehension
                  tower, inner_policy, vocab_size
              )
              for tower in wrapped_policy[1]
          )] +
          [wrapped_policy[2:]]
      )
  else:
    # One tower: run one two-headed network.
    net = wrapped_policy_fn()
    substitute_fn = functools.partial(
        serialization_utils.substitute_inner_policy,
        vocab_size=vocab_size,
    )
  return (net, substitute_fn)


# Should this be collect 'n' trajectories, or
# Run the env for 'n' steps and take completed trajectories, or
# Any other option?
def collect_trajectories(env,
                         policy_fn,
                         n_trajectories=1,
                         n_observations=None,
                         max_timestep=None,
                         reset=True,
                         len_history_for_policy=32,
                         boundary=32,
                         state=None,
                         temperature=1.0,
                         rng=None,
                         abort_fn=None,
                         raw_trajectory=False,):
  """Collect trajectories with the given policy net and behaviour.

  Args:
    env: A gym env interface, for now this is not-batched.
    policy_fn: Callable
      (observations(B,T+1), actions(B, T+1, C)) -> log-probabs(B, T+1, C, A).
    n_trajectories: int, number of trajectories.
    n_observations: int, number of non-terminal observations. NOTE: Exactly one
      of `n_trajectories` and `n_observations` should be None.
    max_timestep: int or None, the index of the maximum time-step at which we
      return the trajectory, None for ending a trajectory only when env returns
      done.
    reset: bool, true if we want to reset the envs. The envs are also reset if
      max_max_timestep is None or < 0
    len_history_for_policy: int or None, the maximum history to keep for
      applying the policy on. If None, use the full history.
    boundary: int, pad the sequences to the multiples of this number.
    state: state for `policy_fn`.
    temperature: (float) temperature to sample action from policy_fn.
    rng: jax rng, splittable.
    abort_fn: callable, If not None, then at every env step call and abort the
      trajectory collection if it returns True, if so reset the env and return
      None.
    raw_trajectory: bool, if True a list of trajectory.Trajectory objects is
      returned, otherwise a list of numpy representations of
      `trajectory.Trajectory` is returned.

  Returns:
    A tuple (trajectory, number of trajectories that are done)
    trajectory: list of (observation, action, reward) tuples, where each element
    `i` is a tuple of numpy arrays with shapes as follows:
    observation[i] = (B, T_i + 1)
    action[i] = (B, T_i)
    reward[i] = (B, T_i)
  """

  assert isinstance(env, env_problem.EnvProblem)

  # We need to reset all environments, if we're coming here the first time.
  if reset or max_timestep is None or max_timestep <= 0:
    env.reset()
  else:
    # Clear completed trajectories held internally.
    env.trajectories.clear_completed_trajectories()

  num_done_trajectories = 0

  # The stopping criterion, returns True if we should stop.
  def should_stop():
    if n_trajectories is not None:
      assert n_observations is None
      return env.trajectories.num_completed_trajectories >= n_trajectories
    assert n_observations is not None
    # The number of non-terminal observations is what we want.
    return (env.trajectories.num_completed_time_steps -
            env.trajectories.num_completed_trajectories) >= n_observations

  policy_application_total_time = 0
  env_actions_total_time = 0
  bare_env_run_time = 0
  while not should_stop():
    # Check if we should abort and return nothing.
    if abort_fn and abort_fn():
      # We should also reset the environment, since it will have some
      # trajectories (complete and incomplete) that we want to discard.
      env.reset()
      return None, 0, {}, state

    # Get all the observations for all the active trajectories.
    # Shape is (B, T+1) + OBS
    # Bucket on whatever length is needed.
    padded_observations, lengths = env.trajectories.observations_np(
        boundary=boundary,
        len_history_for_policy=len_history_for_policy)

    B = padded_observations.shape[0]  # pylint: disable=invalid-name

    assert B == env.batch_size
    assert (B,) == lengths.shape

    t1 = time.time()
    log_probs, value_preds, state, rng = policy_fn(
        padded_observations, lengths, state=state, rng=rng)
    policy_application_total_time += (time.time() - t1)

    assert B == log_probs.shape[0]

    actions = tl.gumbel_sample(log_probs, temperature)
    if (isinstance(env.action_space, gym.spaces.Discrete) and
        (actions.shape[1] == 1)):
      actions = onp.squeeze(actions, axis=1)

    # Step through the env.
    t1 = time.time()
    _, _, dones, env_infos = env.step(
        actions,
        infos={
            'log_prob_actions': log_probs.copy(),
            'value_predictions': value_preds.copy(),
        })
    env_actions_total_time += (time.time() - t1)
    bare_env_run_time += sum(
        info['__bare_env_run_time__'] for info in env_infos)

    # Count the number of done trajectories, the others could just have been
    # truncated.
    num_done_trajectories += onp.sum(dones)

    # Get the indices where we are done ...
    done_idxs = env_problem_utils.done_indices(dones)

    # ... and reset those.
    t1 = time.time()
    if done_idxs.size:
      env.reset(indices=done_idxs)
    env_actions_total_time += (time.time() - t1)

    if max_timestep is None or max_timestep < 1:
      continue

    # Are there any trajectories that have exceeded the time-limit we want.
    lengths = env.trajectories.trajectory_lengths
    exceeded_time_limit_idxs = env_problem_utils.done_indices(
        lengths > max_timestep
    )

    # If so, reset these as well.
    t1 = time.time()
    if exceeded_time_limit_idxs.size:
      # This just cuts the trajectory, doesn't reset the env, so it continues
      # from where it left off.
      env.truncate(indices=exceeded_time_limit_idxs, num_to_keep=1)
    env_actions_total_time += (time.time() - t1)

  # We have the trajectories we need, return a list of triples:
  # (observations, actions, rewards)
  completed_trajectories = (
      env_problem_utils.get_completed_trajectories_from_env(
          env, env.trajectories.num_completed_trajectories,
          raw_trajectory=raw_trajectory))

  timing_info = {
      'trajectory_collection/policy_application': policy_application_total_time,
      'trajectory_collection/env_actions': env_actions_total_time,
      'trajectory_collection/env_actions/bare_env': bare_env_run_time,
  }
  timing_info = {k: round(1000 * v, 2) for k, v in timing_info.items()}

  return completed_trajectories, num_done_trajectories, timing_info, state


# This function can probably be simplified, ask how?
# Can we do something much simpler than lax.pad, maybe np.pad?
# Others?


def get_padding_value(dtype):
  """Returns the padding value given a dtype."""
  padding_value = None
  if dtype == np.uint8:
    padding_value = np.uint8(0)
  elif dtype == np.uint16:
    padding_value = np.uint16(0)
  elif dtype == np.float32 or dtype == np.float64:
    padding_value = 0.0
  else:
    padding_value = 0
  assert padding_value is not None
  return padding_value


# TODO(afrozm): Use np.pad instead and make jittable?
def pad_trajectories(trajectories, boundary=20):
  """Pad trajectories to a bucket length that is a multiple of boundary.

  Args:
    trajectories: list[(observation, actions, rewards)], where each observation
      is shaped (t+1,) + OBS and actions & rewards are shaped (t,), with the
      length of the list being B (batch size).
    boundary: int, bucket length, the actions and rewards are padded to integer
      multiples of boundary.

  Returns:
    tuple: (padding lengths, reward_mask, padded_observations, padded_actions,
        padded_rewards) where padded_observations is shaped (B, T+1) + OBS and
        padded_actions, padded_rewards & reward_mask are shaped (B, T).
        Where T is max(t) rounded up to an integer multiple of boundary.
        padded_length is how much padding we've added and
        reward_mask is 1s for actual rewards and 0s for the padding.
  """

  # Let's compute max(t) over all trajectories.
  t_max = max(r.shape[0] for (_, _, r, _) in trajectories)

  # t_max is rounded to the next multiple of `boundary`
  boundary = int(boundary)
  bucket_length = boundary * int(np.ceil(float(t_max) / boundary))

  # So all obs will be padded to t_max + 1 and actions and rewards to t_max.
  padded_observations = []
  padded_actions = []
  padded_rewards = []
  padded_infos = collections.defaultdict(list)
  padded_lengths = []
  reward_masks = []

  for (o, a, r, i) in trajectories:
    # Determine the amount to pad, this holds true for obs, actions and rewards.
    num_to_pad = bucket_length + 1 - o.shape[0]
    padded_lengths.append(num_to_pad)
    if num_to_pad == 0:
      padded_observations.append(o)
      padded_actions.append(a)
      padded_rewards.append(r)
      reward_masks.append(onp.ones_like(r, dtype=np.int32))
      if i:
        for k, v in i.items():
          padded_infos[k].append(v)
      continue

    # First pad observations.
    padding_config = tuple([(0, num_to_pad, 0)] + [(0, 0, 0)] * (o.ndim - 1))

    padding_value = get_padding_value(o.dtype)
    action_padding_value = get_padding_value(a.dtype)
    reward_padding_value = get_padding_value(r.dtype)

    padded_obs = lax.pad(o, padding_value, padding_config)
    padded_observations.append(padded_obs)

    # Now pad actions and rewards.
    padding_config = tuple([(0, num_to_pad, 0)] + [(0, 0, 0)] * (a.ndim - 1))
    padded_action = lax.pad(a, action_padding_value, padding_config)
    padded_actions.append(padded_action)

    assert r.ndim == 1
    padding_config = ((0, num_to_pad, 0),)
    padded_reward = lax.pad(r, reward_padding_value, padding_config)
    padded_rewards.append(padded_reward)

    # Also create the mask to use later.
    reward_mask = onp.ones_like(r, dtype=np.int64)
    reward_masks.append(lax.pad(reward_mask, 0, padding_config))

    if i:
      for k, v in i.items():
        # Create a padding configuration for this value.
        padding_config = [(0, num_to_pad, 0)] + [(0, 0, 0)] * (v.ndim - 1)
        padded_infos[k].append(lax.pad(v, 0.0, tuple(padding_config)))

  # Now stack these padded_infos if they exist.
  stacked_padded_infos = None
  if padded_infos:
    stacked_padded_infos = {k: np.stack(v) for k, v in padded_infos.items()}

  return padded_lengths, np.stack(reward_masks), np.stack(
      padded_observations), np.stack(padded_actions), np.stack(
          padded_rewards), stacked_padded_infos


def get_time(t1, t2=None):
  if t2 is None:
    t2 = time.time()
  return round((t2 - t1) * 1000, 2)


def get_policy_model_files(output_dir):
  return list(
      reversed(
          sorted(
              tf.io.gfile.glob(os.path.join(output_dir, 'model-??????.pkl')))))


def get_epoch_from_policy_model_file(policy_model_file):
  base_name = os.path.basename(policy_model_file)
  return int(re.match(r'model-(\d+).pkl', base_name).groups()[0])


def get_policy_model_file_from_epoch(output_dir, epoch):
  return os.path.join(output_dir, 'model-%06d.pkl' % epoch)


def maybe_restore_opt_state(output_dir,
                            policy_and_value_opt_state=None,
                            policy_and_value_state=None):
  """Maybe restore the optimization state from the checkpoint dir.

  Optimization state includes parameters and optimizer slots.

  Args:
    output_dir: Directory where saved model checkpoints are stored.
    policy_and_value_opt_state: Default optimization state, returned if model
      isn't found.
    policy_and_value_state: state of the policy and value network.

  Returns:
    tuple (opt_state, state, epoch (int), opt_step (int)) where epoch is the
    epoch from which we restored the optimization state, 0 if no checkpoint was
    found, and opt_step is the total optimization step (sum of all optimization
    steps made up to the current epoch).
  """
  pkl_module = utils.get_pickle_module()
  epoch = 0
  total_opt_step = 0
  history = trax_history.History()
  for model_file in get_policy_model_files(output_dir):
    logging.info('Trying to restore model from %s', model_file)
    try:
      with tf.io.gfile.GFile(model_file, 'rb') as f:
        (policy_and_value_opt_state, policy_and_value_state, total_opt_step,
         history) = pkl_module.load(f)
      epoch = get_epoch_from_policy_model_file(model_file)
      break
    except EOFError as e:
      logging.error('Unable to load model from: %s with %s', model_file, e)
      # Try an older version.
      continue
  return (
      policy_and_value_opt_state,
      policy_and_value_state,
      epoch,
      total_opt_step,
      history,
  )


LAST_N_POLICY_MODELS_TO_KEEP = 5


def save_opt_state(output_dir,
                   policy_and_value_opt_state,
                   policy_and_value_state,
                   epoch,
                   total_opt_step,
                   history):
  """Saves the policy and value network optimization state etc."""
  pkl_module = utils.get_pickle_module()
  old_model_files = get_policy_model_files(output_dir)
  params_file = os.path.join(output_dir, 'model-%06d.pkl' % epoch)
  with tf.io.gfile.GFile(params_file, 'wb') as f:
    pkl_module.dump((policy_and_value_opt_state, policy_and_value_state,
                     total_opt_step, history), f)
  # Keep the last k model files lying around (note k > 1 because the latest
  # model file might be in the process of getting read async).
  for path in old_model_files[LAST_N_POLICY_MODELS_TO_KEEP:]:
    if path != params_file:
      tf.io.gfile.remove(path)


def init_policy_from_world_model_checkpoint(
    policy_weights, model_output_dir, substitute_fn
):
  """Initializes policy parameters from world model parameters."""
  pkl_module = utils.get_pickle_module()
  weights_file = os.path.join(model_output_dir, 'model.pkl')
  # Don't use trax.load_trainer_state to avoid a circular import.
  with tf.io.gfile.GFile(weights_file, 'rb') as f:
    model_weights = pkl_module.load(f)['weights']
  model_weights = serialization_utils.extract_inner_model(model_weights)
  # TODO(pkozakowski): The following, brittle line of code is hardcoded for
  # transplanting parameters from TransformerLM to TransformerDecoder-based
  # policy network of the same configuration. Figure out a more general method.
  return substitute_fn(policy_weights, model_weights[1:-2])


def write_eval_reward_summaries(reward_stats_by_mode, log_fn, epoch):
  """Writes evaluation reward statistics to summary and logs them.

  Args:
    reward_stats_by_mode: Nested dict of structure: {
          'raw': {
              <temperature 1>: {
                  'mean': <reward mean>,
                  'std': <reward std>, },
              <temperature 2>: ... },
          'processed': ... }
    log_fn: Function mode, metric_name, value -> None for logging the summaries.
    epoch: Current epoch number.
  """
  for (reward_mode, reward_stats_by_temp) in reward_stats_by_mode.items():
    for (temperature, reward_stats) in reward_stats_by_temp.items():
      for (stat_name, stat) in reward_stats.items():
        metric_name = 'eval/{}_reward_{}/temperature_{}'.format(
            reward_mode, stat_name, temperature
        )
        log_fn('eval', metric_name, stat)
      logging.info(
          'Epoch [% 6d] Policy Evaluation (%s reward) '
          '[temperature %.2f] = %10.2f (+/- %.2f)', epoch, reward_mode,
          temperature, reward_stats['mean'], reward_stats['std'])


def run_policy_all_timesteps(
    policy_and_value_net_apply,
    observations,
    weights,
    state,
    rng,
    action_space,
):
  """Runs the policy network."""
  # TODO(pkozakowski): Pass the actual actions here, to enable autoregressive
  # action sampling.
  (B, T_plus_1) = observations.shape[:2]  # pylint: disable=invalid-name
  dummy_actions = onp.zeros(
      (B, T_plus_1 - 1) + action_space.shape, dtype=action_space.dtype
  )
  policy_input = (observations, dummy_actions)
  (rng, subrng) = trax_random.split(rng)
  (log_probs, value_preds) = policy_and_value_net_apply(
      policy_input, weights=weights, state=state, rng=subrng
  )
  return log_probs, value_preds, state, rng


def run_policy(
    policy_and_value_net_apply,
    observations,
    lengths,
    weights,
    state,
    rng,
    action_space,
):
  """Runs the policy network and returns lps, vps for the last timestep."""
  log_probs, value_preds, state, rng = run_policy_all_timesteps(
      policy_and_value_net_apply,
      observations,
      weights,
      state,
      rng,
      action_space,
  )

  # We need the log_probs of those actions that correspond to the last actual
  # time-step.
  (B, unused_T_plus_1) = observations.shape[:2]  # pylint: disable=invalid-name
  index = lengths - 1  # Since we want to index using lengths.
  log_probs = log_probs[np.arange(B), index]
  value_preds = value_preds[np.arange(B), index]
  return (log_probs, value_preds, state, rng)
