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

"""PPO trainer."""

import functools
import time

from absl import logging
from jax import numpy as np
from jax import random as jax_random
from trax import models as trax_models
from trax import optimizers as trax_opt
from trax.rl import policy_based_trainer
from trax.rl import policy_based_utils
from trax.rl import ppo

DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.1
EPOCHS = 50  # 100
N_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32


class PPO(policy_based_trainer.PolicyBasedTrainer):
  """PPO trainer."""

  def __init__(self,
               train_env,
               eval_env,
               policy_and_value_model=trax_models.FrameStackMLP,
               policy_and_value_optimizer=functools.partial(
                   trax_opt.Adam, learning_rate=1e-3),
               print_every_optimizer_steps=PRINT_EVERY_OPTIMIZER_STEP,
               target_kl=0.01,
               gamma=GAMMA,
               lambda_=LAMBDA,
               value_weight=1.0,
               entropy_weight=0.01,
               epsilon=0.1,
               **kwargs):
    """Creates the PPO trainer.

    Args:
      train_env: gym.Env to use for training.
      eval_env: gym.Env to use for evaluation.
      policy_and_value_model: Function defining the policy and value network,
        without the policy and value heads.
      policy_and_value_optimizer: Function defining the optimizer.
      print_every_optimizer_steps: How often to log during the policy
        optimization process.
      target_kl: Policy iteration early stopping. Set to infinity to disable
        early stopping.
      gamma: Reward discount factor.
      lambda_: N-step TD-error discount factor in GAE.
      value_weight: Value loss coefficient.
      entropy_weight: Entropy loss coefficient.
      epsilon: Clipping coefficient.
      **kwargs: Additional keyword arguments passed to the base class.
    """
    super(PPO, self).__init__(
        train_env,
        eval_env,
        policy_and_value_model=policy_and_value_model,
        policy_and_value_optimizer=policy_and_value_optimizer,
        **kwargs)

    self._print_every_optimizer_steps = print_every_optimizer_steps
    self._target_kl = target_kl
    self._nontrainable_params = {
        'gamma': np.array(gamma),
        'lambda': np.array(lambda_),
        'value_weight': np.array(value_weight),
        'entropy_weight': np.array(entropy_weight),
        'epsilon': np.array(epsilon),
    }

  def collect_trajectories(self, train=True, **kwargs):
    # We specialize this function to get a specific number of trajectories based
    # on the environment's batch size.

    env = self.train_env if train else self.eval_env
    n_trajectories = env.batch_size

    return super(PPO, self).collect_trajectories(
        train=train, n_trajectories=n_trajectories, **kwargs)

  def train_epoch(self, evaluate=True):
    """Train one PPO epoch."""
    epoch_start_time = time.time()

    # Evaluate the policy.
    policy_eval_start_time = time.time()
    if evaluate and (self.epoch + 1) % self._eval_every_n == 0:
      self.evaluate()

    policy_eval_time = policy_based_utils.get_time(policy_eval_start_time)

    trajectory_collection_start_time = time.time()
    logging.vlog(1, 'PPO epoch [% 6d]: collecting trajectories.', self.epoch)
    key = self._get_rng()
    trajs, _, timing_info, self._model_state = self.collect_trajectories(
        train=True, temperature=1.0)
    trajs = [(t[0], t[1], t[2], t[4]) for t in trajs]
    self._should_reset_train_env = False
    trajectory_collection_time = policy_based_utils.get_time(
        trajectory_collection_start_time
    )

    logging.vlog(1, 'Collecting trajectories took %0.2f msec.',
                 trajectory_collection_time)

    rewards = np.array([np.sum(traj[2]) for traj in trajs])
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    self._log('train', 'train/reward_mean_truncated', avg_reward)
    if evaluate and not self._separate_eval:
      metrics = {'raw': {1.0: {'mean': avg_reward, 'std': std_reward}}}
      policy_based_utils.write_eval_reward_summaries(
          metrics, self._log, self.epoch
      )

    logging.vlog(1, 'Rewards avg=[%0.2f], max=[%0.2f], min=[%0.2f], all=%s',
                 avg_reward, max_reward, min_reward,
                 [float(np.sum(traj[2])) for traj in trajs])

    logging.vlog(1,
                 'Trajectory Length average=[%0.2f], max=[%0.2f], min=[%0.2f]',
                 float(sum(len(traj[0]) for traj in trajs)) / len(trajs),
                 max(len(traj[0]) for traj in trajs),
                 min(len(traj[0]) for traj in trajs))
    logging.vlog(2, 'Trajectory Lengths: %s', [len(traj[0]) for traj in trajs])

    preprocessing_start_time = time.time()
    (padded_observations, padded_actions, padded_rewards, reward_mask,
     padded_infos) = self._preprocess_trajectories(trajs)
    preprocessing_time = policy_based_utils.get_time(preprocessing_start_time)

    logging.vlog(1, 'Preprocessing trajectories took %0.2f msec.',
                 policy_based_utils.get_time(preprocessing_start_time))
    logging.vlog(1, 'Padded Observations\' shape [%s]',
                 str(padded_observations.shape))
    logging.vlog(1, 'Padded Actions\' shape [%s]', str(padded_actions.shape))
    logging.vlog(1, 'Padded Rewards\' shape [%s]', str(padded_rewards.shape))

    # Some assertions.
    (B, T) = reward_mask.shape  # pylint: disable=invalid-name
    assert (B, T) == padded_rewards.shape
    assert B == padded_observations.shape[0]

    log_prob_recompute_start_time = time.time()
    # TODO(pkozakowski): The following commented out code collects the network
    # predictions made while stepping the environment and uses them in PPO
    # training, so that we can use non-deterministic networks (e.g. with
    # dropout). This does not work well with serialization, so instead we
    # recompute all network predictions. Let's figure out a solution that will
    # work with both serialized sequences and non-deterministic networks.

    # assert ('log_prob_actions' in padded_infos and
    #         'value_predictions' in padded_infos)
    # These are the actual log-probabs and value predictions seen while picking
    # the actions.
    # actual_log_probabs_traj = padded_infos['log_prob_actions']
    # actual_value_predictions_traj = padded_infos['value_predictions']

    # assert (B, T, C) == actual_log_probabs_traj.shape[:3]
    # A = actual_log_probabs_traj.shape[3]  # pylint: disable=invalid-name
    # assert (B, T, 1) == actual_value_predictions_traj.shape

    del padded_infos

    # NOTE: We don't have the log-probabs and value-predictions for the last
    # observation, so we re-calculate for everything, but use the original ones
    # for all but the last time-step.
    key = self._get_rng()

    # TODO(pkozakowski): Pass the actual actions here, to enable autoregressive
    # action sampling.
    dummy_actions = np.zeros_like(padded_actions)
    (log_probabs_traj, value_predictions_traj) = (
        self._policy_and_value_net_apply(
            (padded_observations, dummy_actions),
            weights=self._policy_and_value_net_weights,
            state=self._model_state,
            rng=key,
        ))
    # Cut off the last extra action to obtain shape (B, T, C, A).
    log_probabs_traj_cut = log_probabs_traj[:, :-1]

    assert (B, T) == log_probabs_traj_cut.shape[:2]
    assert (B, T + 1) == value_predictions_traj.shape

    # TODO(pkozakowski): Commented out for the same reason as before.

    # Concatenate the last time-step's log-probabs and value predictions to the
    # actual log-probabs and value predictions and use those going forward.
    # log_probabs_traj = np.concatenate(
    #     (actual_log_probabs_traj, log_probabs_traj[:, -1:, :]), axis=1)
    # value_predictions_traj = np.concatenate(
    #     (actual_value_predictions_traj, value_predictions_traj[:, -1:, :]),
    #     axis=1)

    log_prob_recompute_time = policy_based_utils.get_time(
        log_prob_recompute_start_time
    )

    # Compute value and ppo losses.
    key1 = self._get_rng()
    logging.vlog(2, 'Starting to compute P&V loss.')
    loss_compute_start_time = time.time()
    (cur_combined_loss, component_losses, summaries, self._model_state) = (
        ppo.combined_loss(
            self._policy_and_value_net_weights,
            log_probabs_traj_cut,
            value_predictions_traj,
            self._policy_and_value_net_apply,
            padded_observations,
            padded_actions,
            padded_rewards,
            reward_mask,
            nontrainable_params=self._nontrainable_params,
            state=self._model_state,
            rng=key1))
    loss_compute_time = policy_based_utils.get_time(loss_compute_start_time)
    (cur_ppo_loss, cur_value_loss, cur_entropy_bonus) = component_losses
    logging.vlog(
        1,
        'Calculating P&V loss [%10.2f(%10.2f, %10.2f, %10.2f)] took %0.2f msec.',
        cur_combined_loss, cur_ppo_loss, cur_value_loss, cur_entropy_bonus,
        policy_based_utils.get_time(loss_compute_start_time))

    key1 = self._get_rng()
    logging.vlog(1, 'Policy and Value Optimization')
    optimization_start_time = time.time()
    keys = jax_random.split(key1, num=self._n_optimizer_steps)
    opt_step = 0
    opt_batch_size = min(self._optimizer_batch_size, B)
    index_batches = ppo.shuffled_index_batches(
        dataset_size=B, batch_size=opt_batch_size)
    for (index_batch, key) in zip(index_batches, keys):
      k1, k2, k3 = jax_random.split(key, num=3)
      t = time.time()
      # Update the optimizer state on the sampled minibatch.
      self._policy_and_value_opt_state, self._model_state = (
          ppo.policy_and_value_opt_step(
              # We pass the optimizer slots between PPO epochs, so we need to
              # pass the optimization step as well, so for example the
              # bias-correction in Adam is calculated properly. Alternatively we
              # could reset the slots and the step in every PPO epoch, but then
              # the moment estimates in adaptive optimizers would never have
              # enough time to warm up. So it makes sense to reuse the slots,
              # even though we're optimizing a different loss in every new
              # epoch.
              self._total_opt_step,
              self._policy_and_value_opt_state,
              self._policy_and_value_opt_update,
              self._policy_and_value_get_params,
              self._policy_and_value_net_apply,
              log_probabs_traj_cut[index_batch],
              value_predictions_traj[index_batch],
              padded_observations[index_batch],
              padded_actions[index_batch],
              padded_rewards[index_batch],
              reward_mask[index_batch],
              nontrainable_params=self._nontrainable_params,
              state=self._model_state,
              rng=k1))
      opt_step += 1
      self._total_opt_step += 1

      # Compute the approx KL for early stopping. Use the whole dataset - as we
      # only do inference, it should fit in the memory.
      # TODO(pkozakowski): Pass the actual actions here, to enable
      # autoregressive action sampling.
      dummy_actions = np.zeros_like(padded_actions)
      (log_probab_actions_new, _) = (
          self._policy_and_value_net_apply(
              (padded_observations, dummy_actions),
              weights=self._policy_and_value_net_weights,
              state=self._model_state,
              rng=k2))
      # Cut off the last extra action to obtain shape (B, T, C, A).
      log_probab_actions_new_cut = log_probab_actions_new[:, :-1]

      approx_kl = ppo.approximate_kl(log_probab_actions_new_cut,
                                     log_probabs_traj_cut, reward_mask)

      early_stopping = approx_kl > 1.5 * self._target_kl
      if early_stopping:
        logging.vlog(
            1, 'Early stopping policy and value optimization after %d steps, '
            'with approx_kl: %0.2f', opt_step, approx_kl)
        # We don't return right-away, we want the below to execute on the last
        # iteration.

      t2 = time.time()
      if (opt_step % self._print_every_optimizer_steps == 0 or
          opt_step == self._n_optimizer_steps or early_stopping):
        # Compute and log the loss.
        (combined_loss, component_losses, _, self._model_state) = (
            ppo.combined_loss(
                self._policy_and_value_net_weights,
                log_probabs_traj_cut,
                value_predictions_traj,
                self._policy_and_value_net_apply,
                padded_observations,
                padded_actions,
                padded_rewards,
                reward_mask,
                nontrainable_params=self._nontrainable_params,
                state=self._model_state,
                rng=k3))
        logging.vlog(1, 'One Policy and Value grad desc took: %0.2f msec',
                     policy_based_utils.get_time(t, t2))
        (ppo_loss, value_loss, entropy_bonus) = component_losses
        logging.vlog(
            1, 'Combined Loss(value, ppo, entropy_bonus) [%10.2f] ->'
            ' [%10.2f(%10.2f,%10.2f,%10.2f)]', cur_combined_loss, combined_loss,
            ppo_loss, value_loss, entropy_bonus)

      if early_stopping:
        break

    optimization_time = policy_based_utils.get_time(optimization_start_time)

    logging.vlog(
        1, 'Total Combined Loss reduction [%0.2f]%%',
        (100 * (cur_combined_loss - combined_loss) / np.abs(cur_combined_loss)))

    summaries.update({
        'n_optimizer_steps': opt_step,
        'approx_kl': approx_kl,
    })
    for (name, value) in summaries.items():
      self._log('train', 'train/{}'.format(name), value)

    logging.info(
        'PPO epoch [% 6d], Reward[min, max, avg] [%5.2f,%5.2f,%5.2f], Combined'
        ' Loss(ppo, value, entropy) [%2.5f(%2.5f,%2.5f,%2.5f)]', self.epoch,
        min_reward, max_reward, avg_reward, combined_loss, ppo_loss, value_loss,
        entropy_bonus)

    # Bump the epoch counter before saving a checkpoint, so that a call to
    # save() after the training loop is a no-op if a checkpoint was saved last
    # epoch - otherwise it would bump the epoch counter on the checkpoint.
    last_epoch = self.epoch
    self._epoch += 1

    epoch_time = policy_based_utils.get_time(epoch_start_time)

    timing_dict = {
        'epoch': epoch_time,
        'policy_eval': policy_eval_time,
        'trajectory_collection': trajectory_collection_time,
        'preprocessing': preprocessing_time,
        'log_prob_recompute': log_prob_recompute_time,
        'loss_compute': loss_compute_time,
        'optimization': optimization_time,
    }

    timing_dict.update(timing_info)

    if self._should_write_summaries:
      for k, v in timing_dict.items():
        self._timing_sw.scalar('timing/%s' % k, v, step=last_epoch)

    max_key_len = max(len(k) for k in timing_dict)
    timing_info_list = [
        '%s : % 10.2f' % (k.rjust(max_key_len + 1), v)
        for k, v in sorted(timing_dict.items())
    ]
    logging.info('PPO epoch [% 6d], Timings: \n%s', last_epoch,
                 '\n'.join(timing_info_list))

    # Flush summary writers once in a while.
    if self.epoch % 1000 == 0:
      self.flush_summaries()

  def evaluate(self):
    """Evaluate the agent."""
    if not self._separate_eval:
      return

    logging.vlog(1, 'PPO epoch [% 6d]: evaluating policy.', self.epoch)

    if self._controller is not None:
      ntp_updates = self._controller(self._history)(self.epoch)
      self._nontrainable_params.update(ntp_updates)
      (_, _, opt_params) = self._policy_and_value_opt_state
      opt_params.update(ntp_updates)
      for (name, value) in self._nontrainable_params.items():
        self._log('train', 'training/{}'.format(name), value)

    super(PPO, self).evaluate()
