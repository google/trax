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
"""Trainer for Advantage Weighted Regression (AWR)."""

import functools
import os
import time
from absl import logging
from jax.api import grad
from jax.api import jit
import numpy as onp
from tensor2tensor.envs import env_problem
from trax import jaxboard
from trax.math import numpy as np
from trax.rl import awr_utils
from trax.rl import policy_based_trainer
from trax.rl import policy_based_utils
from trax.rl.trajectory import replay_buffer


class AwrTrainer(policy_based_trainer.PolicyBasedTrainer):
  """Trainer for AWR."""

  ADV_EPS = 1e-5

  def __init__(self,
               train_env: env_problem.EnvProblem,
               eval_env: env_problem.EnvProblem,
               td_lambda=0.95,
               gamma=0.99,
               replay_buffer_sample_size=50000,
               num_samples_to_collect=2048,
               temperature=0.05,
               weight_clip=20,
               actor_batch_size=256,
               critic_batch_size=256,
               actor_optimization_steps=1000,
               critic_optimization_steps=500,
               actor_momentum=0.9,
               critic_momentum=0.9,
               actor_learning_rate=5e-5,
               critic_learning_rate=1e-4,
               actor_loss_weight=1.0,
               entropy_bonus=0.01,
               **kwargs):
    super(AwrTrainer, self).__init__(train_env, eval_env, **kwargs)

    self._td_lambda = td_lambda
    self._gamma = gamma
    self._replay_buffer_sample_size = replay_buffer_sample_size
    self._num_samples_to_collect = num_samples_to_collect
    self._temperature = temperature
    self._weight_clip = weight_clip
    self._actor_batch_size = actor_batch_size
    self._critic_batch_size = critic_batch_size
    self._actor_optimization_steps = actor_optimization_steps
    self._critic_optimization_steps = critic_optimization_steps
    self._actor_momentum = actor_momentum
    self._critic_momentum = critic_momentum
    self._actor_learning_rate = actor_learning_rate
    self._critic_learning_rate = critic_learning_rate
    self._actor_loss_weight = actor_loss_weight
    self._entropy_bonus = entropy_bonus

    # Unified loss.
    self._optimization_batch_size = critic_batch_size
    self._optimization_steps = critic_optimization_steps
    self._momentum = critic_momentum
    self._learning_rate = critic_learning_rate

    self._replay_buffer = replay_buffer.ReplayBuffer(
        buffer_size=replay_buffer_sample_size)

    # self._action_space and _observation_space were set in the base class.
    self._action_shape = self._action_space.shape
    self._action_dtype = self._action_space.dtype
    self._observation_shape = self._observation_space.shape
    self._observation_dtype = self._observation_space.dtype

    # TODO(afrozm): Offload all these to `trainer_lib.Trainer`.
    self._total_opt_step = 0
    # TODO(afrozm): Ensure that this is updated.
    self._n_observations_seen = 0
    self._opt_sw = None

  def reset(self, output_dir=None):
    super(AwrTrainer, self).reset(output_dir=output_dir)

    if self._should_write_summaries:
      self._opt_sw = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'opt'))

    # Reset the replay buffer.
    self._replay_buffer.clear()
    self._replay_buffer.buffers = None

    # TODO(afrozm): Ensure that this is updated.
    self._n_observations_seen = 0

  def collect_trajectories(self, train=True, **kwargs):
    env = self.train_env if train else self.eval_env
    # Get a specific number of `samples` if train, else complete trajectories.
    n_trajectories = None if train else env.batch_size
    n_observations = self._num_samples_to_collect if train else None

    return super(AwrTrainer, self).collect_trajectories(
        train=train,
        n_trajectories=n_trajectories,
        n_observations=n_observations,
        **kwargs)

  def train_epoch(self, evaluate=True):
    epoch_start_time = time.time()

    # Evaluate the policy.
    policy_eval_start_time = time.time()
    if evaluate and (self.epoch + 1) % self._eval_every_n == 0:
      self.evaluate()
    policy_eval_time = policy_based_utils.get_time(policy_eval_start_time)

    def write_metric(key, value):
      self._train_sw.scalar(key, value, step=self.epoch)
      self._history.append('train', key, self.epoch, value)

    # Get fresh trajectories every time.
    self._should_reset_train_env = True

    trajectory_collection_start_time = time.time()
    logging.vlog(1, 'AWR epoch [% 6d]: collecting trajectories.', self._epoch)
    trajs, _, timing_info, self._model_state = self.collect_trajectories(
        train=True, temperature=1.0, raw_trajectory=True)
    del timing_info
    trajectory_collection_time = policy_based_utils.get_time(
        trajectory_collection_start_time
    )

    logging.vlog(1, 'AWR epoch [% 6d]: n_trajectories [%s].',
                 self._epoch, len(trajs))

    # Convert these into numpy now.
    def extract_obs_act_rew_dones(traj_np):
      return traj_np[0], traj_np[1], traj_np[2], traj_np[4]

    trajs_np = [extract_obs_act_rew_dones(traj.as_numpy) for traj in trajs]

    # number of new actions.
    new_sample_count = sum(traj[1].shape[0] for traj in trajs_np)
    self._n_observations_seen += new_sample_count
    logging.vlog(1, 'AWR epoch [% 6d]: new_sample_count [%d].',
                 self._epoch, new_sample_count)

    if self._should_write_summaries:
      write_metric('trajs/batch', len(trajs))
      write_metric('trajs/new_sample_count', new_sample_count)

    # The number of trajectories, i.e. `B`can keep changing from iteration to
    # iteration, since we are capped on the number of observations requested.
    # So let's operate on each trajectory on this own?

    # TODO(afrozm): So should our batches look like (B, T+1, *OBS) or B
    # different examples of (T+1, *OBS) each. Since B can keep changing?

    # Add these to the replay buffer.
    for traj in trajs:
      _ = self._replay_buffer.store(traj)

    rewards = np.array([np.sum(traj[2]) for traj in trajs_np])
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    self._log('train', 'train/reward_mean_truncated', avg_reward)
    if evaluate and not self._separate_eval and self._should_write_summaries:
      metrics = {'raw': {1.0: {'mean': avg_reward, 'std': std_reward}}}
      policy_based_utils.write_eval_reward_summaries(metrics, self._log,
                                                     self.epoch)

    logging.vlog(
        1, 'AWR epoch [% 6d]: Rewards avg=[%0.2f], max=[%0.2f], '
        'min=[%0.2f].', self.epoch, avg_reward, max_reward, min_reward)

    if self._should_write_summaries:
      write_metric('reward/avg', avg_reward)
      write_metric('reward/std', std_reward)
      write_metric('reward/max', max_reward)
      write_metric('reward/min', min_reward)

    # Wrap these observations/rewards inside ReplayBuffer.
    idx, valid_mask, valid_idx = self._replay_buffer.get_valid_indices()

    # pylint: disable=g-complex-comprehension
    observations = [
        self._replay_buffer.get(replay_buffer.ReplayBuffer.OBSERVATIONS_KEY,
                                idx[start_idx:end_plus_1_idx])
        for (start_idx,
             end_plus_1_idx) in self._replay_buffer.iterate_over_paths(idx)
    ]

    rewards = [
        self._replay_buffer.get(replay_buffer.ReplayBuffer.REWARDS_KEY,
                                idx[start_idx:end_plus_1_idx][:-1])
        for (start_idx,
             end_plus_1_idx) in self._replay_buffer.iterate_over_paths(idx)
    ]
    # pylint: enable=g-complex-comprehension

    t_final = awr_utils.padding_length(rewards, boundary=self._boundary)
    logging.vlog(1, 'AWR epoch [% 6d]: t_final [%s].', self._epoch, t_final)

    if self._should_write_summaries:
      write_metric('trajs/t_final', t_final)

    # These padded observations are over *all* the non-final observations in
    # the entire replay buffer.
    # Shapes:
    # padded_observations      = (B, T + 1, *OBS)
    # padded_observations_mask = (B, T + 1)
    padded_observations, padded_observations_mask = (
        awr_utils.pad_array_to_length(observations, t_final + 1)
    )

    batch = len(observations)
    self._check_shapes('padded_observations', '(batch, t_final + 1)',
                       padded_observations, (batch, t_final + 1),
                       array_prefix=2)
    self._check_shapes('padded_observations_mask', '(batch, t_final + 1)',
                       padded_observations_mask, (batch, t_final + 1))

    # Shapes:
    # padded_rewards      = (B, T)
    # padded_rewards_mask = (B, T)
    padded_rewards, padded_rewards_mask = awr_utils.pad_array_to_length(
        rewards, t_final)
    self._check_shapes('padded_rewards', '(batch, t_final)',
                       padded_rewards, (batch, t_final))
    self._check_shapes('padded_rewards_mask', '(batch, t_final)',
                       padded_rewards_mask, (batch, t_final))

    # Shapes:
    # lengths = (B,)
    lengths = np.sum(padded_rewards_mask, axis=1, dtype=np.int32)
    self._check_shapes('lengths', '(batch,)', lengths, (batch,))

    # TODO(pkozakowski): Pass the actual actions here, to enable autoregressive
    # action sampling.
    dummy_actions = np.zeros(
        (batch, t_final + 1) + self._action_shape,
        self._action_dtype,
    )

    # Shapes:
    # log_probabs_traj       = (B, T + 1, #controls, #actions)
    # value_predictions_traj = (B, T + 1)
    log_probabs_traj, value_predictions_traj, self._model_state, unused_rng = (
        self._policy_fun_all_timesteps(
            padded_observations, lengths, self._model_state, self._get_rng())
    )
    self._check_shapes('log_probabs_traj',
                       '(batch, t_final + 1, n_controls, n_actions)',
                       log_probabs_traj,
                       (batch, t_final + 1, self._n_controls, self._n_actions))
    self._check_shapes('value_predictions_traj',
                       '(batch, t_final + 1)',
                       value_predictions_traj,
                       (batch, t_final + 1))

    # Zero out the padding's value predictions, since the net may give some
    # prediction to the padding observations.
    value_predictions_traj *= padded_observations_mask

    # Compute td-lambda returns, and reshape to match value_predictions_traj.
    list_td_lambda_returns = awr_utils.batched_compute_td_lambda_return(
        padded_rewards, padded_rewards_mask, value_predictions_traj,
        padded_observations_mask, self._gamma, self._td_lambda)

    if logging.vlog_is_on(1) and list_td_lambda_returns:
      l = len(list_td_lambda_returns)
      logging.vlog(1, f'Len of list_td_lambda_returns: {l}.')
      self._log_shape('td_lambda_returns[0]', list_td_lambda_returns[0])

    # pad an extra 0 for each to match lengths of value predictions.
    list_target_values = [
        onp.pad(l, (0, 1), 'constant') for l in list_td_lambda_returns
    ]

    if batch != len(list_target_values):
      raise ValueError(f'batch != len(list_target_values) : '
                       f'{batch} vs {len(list_target_values)}')

    # Shape: (len(idx),)
    target_values = onp.concatenate(list_target_values)
    self._check_shapes('target_values', '(len(idx),)',
                       target_values, (len(idx),))

    # Shape: (len(idx),)
    vals = self.flatten_vals(value_predictions_traj, padded_observations_mask)
    self._check_shapes('vals', '(len(idx),)', vals, (len(idx),))

    # Calculate advantages.
    adv, norm_adv, adv_mean, adv_std = self._calc_adv(
        target_values, vals, valid_mask)
    self._check_shapes('norm_adv', '(len(idx),)', norm_adv, (len(idx),))

    adv_weights, adv_weights_mean, adv_weights_min, adv_weights_max = (
        self._calc_adv_weights(norm_adv, valid_mask)
    )
    self._check_shapes('adv_weights', '(len(idx),)', adv_weights, (len(idx),))

    del adv, adv_mean, adv_std
    del adv_weights_min, adv_weights_max, adv_weights_mean

    combined_steps = int(
        np.ceil(self._optimization_steps * new_sample_count /
                self._num_samples_to_collect))
    optimization_start_time = time.time()
    combined_losses = self._update_combined(combined_steps, valid_idx,
                                            target_values, adv_weights)
    optimization_time = policy_based_utils.get_time(optimization_start_time)

    self._epoch += 1

    if self._should_write_summaries:
      write_metric('combined/optimization_steps', combined_steps)
      epoch_time = policy_based_utils.get_time(epoch_start_time)
      timing_dict = {
          'epoch': epoch_time,
          'trajectory_collection': trajectory_collection_time,
          'optimization': optimization_time,
          'policy_eval': policy_eval_time,
      }

      if self._should_write_summaries:
        for k, v in timing_dict.items():
          write_metric('timing/{}'.format(k), v)

      # Only dump the average post losses.
      if combined_losses:
        for k, v in combined_losses.items():
          if 'post_entropy' in k:
            write_metric(k.replace('post_entropy', 'entropy'), v)
          if 'post_loss' in k:
            write_metric(k.replace('post_loss', 'loss'), v)

    self.flush_summaries()

  def maybe_save(self):
    # MuJoCo envs don't seem to be returning `done` and we just cut them to
    # `max_timestep` anyways.
    # TODO(afrozm): Refactor to trax.save_trainer_state.
    if (self.epoch % self._save_every_n == 0) or self._async_mode:
      self.save()

  def flatten_vals(self, value_predictions_traj, padded_observations_mask):
    batch = len(padded_observations_mask)
    lens = np.sum(padded_observations_mask, axis=1)
    return np.concatenate(
        [value_predictions_traj[b][:int(lens[b])] for b in range(batch)])

  def _step_combined(self, observations, actions, critic_target,
                     advantage_weights):
    key = self._get_rng()

    pre_c_loss, pre_a_loss, pre_ent_val, self._model_state = combined_loss(
        self._policy_and_value_net_weights,
        observations,
        actions,
        critic_target,
        advantage_weights,
        self._policy_and_value_net_apply,
        self.train_env.action_space,
        state=self._model_state,
        rng=key,
    )

    key = self._get_rng()

    self._policy_and_value_opt_state, self._model_state = (
        combined_opt_step(
            self._total_opt_step,
            self._policy_and_value_opt_state,
            self._policy_and_value_opt_update,
            self._policy_and_value_get_params,
            self._policy_and_value_net_apply,
            observations,
            actions,
            critic_target,
            advantage_weights,
            self._actor_loss_weight,
            self._entropy_bonus,
            self.train_env.action_space,
            state=self._model_state,
            rng=key,
        ))

    key = self._get_rng()

    post_c_loss, post_a_loss, post_ent_val, self._model_state = combined_loss(
        self._policy_and_value_net_weights,
        observations,
        actions,
        critic_target,
        advantage_weights,
        self._policy_and_value_net_apply,
        self.train_env.action_space,
        state=self._model_state,
        rng=key,
    )

    combined_pre_loss = combine_loss_components(
        pre_c_loss, pre_a_loss, pre_ent_val, self._actor_loss_weight,
        self._entropy_bonus)
    combined_post_loss = combine_loss_components(
        post_c_loss, post_a_loss, post_ent_val, self._actor_loss_weight,
        self._entropy_bonus)

    loss_dict = {
        'critic/loss_pre': pre_c_loss,
        'critic/loss_post': post_c_loss,
        'critic/loss_reduction': 1 - (post_c_loss/pre_c_loss),
        'actor/loss_pre': pre_a_loss,
        'actor/loss_post': post_a_loss,
        'actor/loss_reduction': 1 - (post_a_loss/pre_a_loss),
        'combined/loss_pre': combined_pre_loss,
        'combined/loss_post': combined_post_loss,
        'combined/loss_reduction': 1 - (combined_post_loss/combined_pre_loss),
        'combined/entropy_pre': pre_ent_val,
        'combined/entropy_post': post_ent_val,
        'combined/entropy_reduction': 1 - (post_ent_val/pre_ent_val),
    }

    for k, v in loss_dict.items():
      self._opt_sw.scalar(k, v, step=self._total_opt_step)

    self._total_opt_step += 1

    return loss_dict

  def _calc_adv(self, new_vals, vals, valid_mask):
    adv = new_vals - vals

    valid_adv = adv[valid_mask]
    adv_mean = np.mean(valid_adv)
    adv_std = np.std(valid_adv)

    norm_adv = (adv - adv_mean) / (adv_std + self.ADV_EPS)
    return adv, norm_adv, adv_mean, adv_std

  def _calc_adv_weights(self, adv, valid_mask):
    weights = np.exp(adv / self._temperature)

    valid_weights = weights[valid_mask]
    weights_mean = np.mean(valid_weights)
    weights_min = np.min(valid_weights)
    weights_max = np.max(valid_weights)

    weights = np.minimum(weights, self._weight_clip)
    return weights, weights_mean, weights_min, weights_max

  def _update_combined(self, steps, valid_idx, target_val_preds, adv_weights):
    num_idx = valid_idx.shape[0]
    steps_per_shuffle = int(onp.ceil(num_idx / self._optimization_batch_size))
    losses = None

    for b in range(steps):
      if b % steps_per_shuffle == 0:
        onp.random.shuffle(valid_idx)

      batch_idx_beg = b * self._optimization_batch_size
      batch_idx_end = batch_idx_beg + self._optimization_batch_size
      batch_idx = onp.array(
          range(batch_idx_beg, batch_idx_end), dtype=onp.int32)
      batch_idx = onp.mod(batch_idx, num_idx)

      batch = valid_idx[batch_idx]
      critic_batch_vals = target_val_preds[batch[:, 1]]

      actor_batch_adv = adv_weights[batch[:, 1]]

      # Shape: (_critic_batch_size, *OBS)
      critic_s = self._replay_buffer.get(
          replay_buffer.ReplayBuffer.OBSERVATIONS_KEY, batch[:, 0])
      # Shape: (_critic_batch_size, #C)
      actor_a = self._replay_buffer.get(
          replay_buffer.ReplayBuffer.ACTIONS_KEY, batch[:, 0])

      curr_losses = self._step_combined(critic_s, actor_a, critic_batch_vals,
                                        actor_batch_adv)

      if losses is None:
        losses = curr_losses
      else:
        for key, val in curr_losses.items():
          losses[key] += val  # pylint: disable=unsupported-assignment-operation

    if losses:
      for key in losses.keys():
        losses[key] /= steps

    return losses


@functools.partial(jit, static_argnums=(5, 6))
def combined_loss(new_weights,
                  observations,
                  actions,
                  target_values,
                  advantage_weights,
                  policy_and_value_net_apply,
                  action_space,
                  state=None,
                  rng=None):
  """Returns the loss components."""

  # TODO(afrozm): This is where we need to eventually feed the earlier
  #  observations than this observation, currently the replay buffer just gives
  #  the observation as is, for transformer like policies, we should also get
  #  all the earlier observations as well, and then the extra dimension will
  #  just be time. For now we reshape as (batch, 1, *obs_shape).
  observations = np.expand_dims(observations, axis=1)

  (log_probab_actions_new, value_predictions_new, state_new, unused_rng_new) = (
      policy_based_utils.run_policy_all_timesteps(
          policy_and_value_net_apply,
          observations,
          new_weights,
          state,
          rng,
          action_space,
      )
  )

  critic_loss_val, intermediate_state = critic_loss(
      observations,
      target_values,
      value_predictions_new,
      state=state_new)
  actor_loss_val, final_state = actor_loss(
      actions,
      advantage_weights,
      log_probab_actions_new,
      state=intermediate_state)
  entropy_val = entropy(log_probab_actions_new)

  return critic_loss_val, actor_loss_val, entropy_val, final_state


@jit
def entropy(log_probab_actions_new):
  """Entropy."""
  # log_probab_actions_new's shape is (B, 1, A)
  lp = log_probab_actions_new
  p = np.exp(lp)
  return -np.mean(lp * p)


@jit
def actor_loss(actions,
               advantage_weights,
               log_probab_actions_new,
               state=None):
  """Actor loss."""

  # log_probab_actions_new's shape is (AB, 1, #C, #A), AB is actor batch.
  lp = np.squeeze(log_probab_actions_new, axis=1)
  AB, NC = actions.shape  # pylint: disable=invalid-name
  log_probs = lp[np.arange(AB)[:, None], np.arange(NC)[None, :], actions]

  # TODO(afrozm): Clarify this.
  #   log_probs are shaped (AB, #C), however advantage_weights are (AB,)
  return -1.0 * np.mean(log_probs * advantage_weights[:, None]), state


@jit
def critic_loss(observations,
                target_values,
                value_predictions_new,
                state=None):
  """Critic loss."""
  # There is no padding involved here, these are all observations.
  (batch, *obs_shape) = observations.shape
  del obs_shape
  if (batch,) != target_values.shape:
    raise ValueError(f'batch dimension is not the same: obs batch {batch}'
                     f' vs target values batch {target_values.shape[0]}')

  # TODO(afrozm): In the reference implementation, they pass the target through
  #  a trained normalizer before subtracting.

  loss = 0.5 * np.mean(np.square(target_values - value_predictions_new))
  return loss, state


def combine_loss_components(critic_loss_val, actor_loss_val, entropy_val,
                            actor_loss_weight, entropy_bonus):
  """Combine the components in the combined AWR loss."""
  return critic_loss_val + (actor_loss_val * actor_loss_weight) - (
      entropy_val * entropy_bonus)


@functools.partial(jit, static_argnums=(2, 3, 4, 11))
def combined_opt_step(i,
                      opt_state,
                      opt_update,
                      get_params,
                      policy_and_value_net_apply,
                      observations,
                      actions,
                      target_values,
                      advantage_weights,
                      actor_loss_weight,
                      entropy_bonus,
                      action_space,
                      state=None,
                      rng=None):
  """Optimization step for combined loss."""

  def _combined_loss(params, in_state):  # pylint: disable=missing-docstring
    critic_loss_val, actor_loss_val, entropy_val, final_state = combined_loss(
        params,
        observations,
        actions,
        target_values,
        advantage_weights,
        policy_and_value_net_apply,
        action_space,
        state=in_state,
        rng=rng,
    )
    return combine_loss_components(critic_loss_val, actor_loss_val, entropy_val,
                                   actor_loss_weight,
                                   entropy_bonus), final_state

  new_weights = get_params(opt_state)
  g, state = grad(_combined_loss, has_aux=True)(new_weights, state)
  return opt_update(i, g, opt_state), state
