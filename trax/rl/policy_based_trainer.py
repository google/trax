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
"""Base class for policy based algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os

from absl import logging
import jax
from jax import numpy as np
from jax import random as jax_random
import numpy as onp
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import trajectory
from trax import jaxboard
from trax.rl import base_trainer
from trax.rl import policy_based_utils
from trax.rl import serialization_utils
from trax.shapes import ShapeDtype
from trax.supervised import trainer_lib

DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.1
EPOCHS = 50  # 100
N_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32


class PolicyBasedTrainer(base_trainer.BaseTrainer):
  """Base trainer for policy based algorithms."""

  def __init__(
      self,
      train_env,
      eval_env,
      output_dir=None,
      random_seed=None,
      controller=None,
      # Policy and Value model arguments.
      policy_and_value_model=None,
      policy_and_value_optimizer=None,
      policy_and_value_two_towers=False,
      policy_and_value_vocab_size=None,
      init_policy_from_world_model_output_dir=None,
      # Trajectory collection arguments.
      boundary=20,
      max_timestep=100,
      max_timestep_eval=20000,
      len_history_for_policy=4,
      # Save / Restore arguments.
      should_save_checkpoints=True,
      should_write_summaries=True,
      eval_every_n=1000,
      save_every_n=1000,
      done_frac_for_policy_save=0.5,
      # Eval arguments.
      n_evals=1,
      eval_temperatures=(1.0, 0.5),
      separate_eval=True,
      # Optimization arguments.
      n_optimizer_steps=N_OPTIMIZER_STEPS,
      optimizer_batch_size=64,
      **kwargs):
    """Creates the PolicyBasedTrainer.

    Args:
      train_env: gym.Env to use for training.
      eval_env: gym.Env to use for evaluation.
      output_dir: Output dir.
      random_seed: Random seed.
      controller: Function history -> (step -> {'name': value}) controlling
        nontrainable parameters.
      policy_and_value_model: Function defining the policy and value network,
        without the policy and value heads.
      policy_and_value_optimizer: Function defining the optimizer.
      policy_and_value_two_towers: Whether to use two separate models as the
        policy and value networks. If False, share their parameters.
      policy_and_value_vocab_size: Vocabulary size of a policy and value network
        operating on serialized representation. If None, use raw continuous
        representation.
      init_policy_from_world_model_output_dir: Model output dir for initializing
        the policy. If None, initialize randomly.
      boundary: We pad trajectories at integer multiples of this number.
      max_timestep: If set to an integer, maximum number of time-steps in a
        trajectory. Used in the collect procedure.
      max_timestep_eval: If set to an integer, maximum number of time-steps in
        an evaluation trajectory. Used in the collect procedure.
      len_history_for_policy: How much of history to give to the policy.
      should_save_checkpoints: Whether to save policy checkpoints.
      should_write_summaries: Whether to save summaries.
      eval_every_n: How frequently to eval the policy.
      save_every_n: How frequently to save the policy.
      done_frac_for_policy_save: Fraction of the trajectories that should be
        done to checkpoint the policy.
      n_evals: Number of times to evaluate.
      eval_temperatures: Sequence of temperatures to try for categorical
        sampling during evaluation.
      separate_eval: Whether to run separate evaluation using a set of
        temperatures. If False, the training reward is reported as evaluation
        reward with temperature 1.0.
      n_optimizer_steps: Number of optimizer steps.
      optimizer_batch_size: Batch size of an optimizer step.
      **kwargs: Additional keyword arguments passed to the base class.
    """
    super(PolicyBasedTrainer, self).__init__(train_env, eval_env, output_dir,
                                             **kwargs)

    self._rng = trainer_lib.init_random_number_generators(random_seed)
    self._controller = controller
    self._history = None
    self._epoch = 0

    # Trajectory collection arguments.
    self._boundary = boundary
    self._max_timestep = max_timestep
    self._max_timestep_eval = max_timestep_eval
    self._len_history_for_policy = len_history_for_policy

    # Save / Restore arguments.
    self._should_save_checkpoints = should_save_checkpoints
    self._should_write_summaries = should_write_summaries
    self._train_sw, self._eval_sw, self._timing_sw = None, None, None
    self._eval_every_n = eval_every_n
    self._save_every_n = save_every_n
    self._done_frac_for_policy_save = done_frac_for_policy_save
    self._n_trajectories_done_since_last_save = 0
    self._last_saved_at_epoch = self._epoch

    # Eval arguments.
    self._n_evals = n_evals
    self._eval_temperatures = eval_temperatures
    self._separate_eval = separate_eval

    # Optimization arguments.
    self._n_optimizer_steps = n_optimizer_steps
    self._optimizer_batch_size = optimizer_batch_size
    self._total_opt_step = 0

    # Policy and Value model arguments.
    self._policy_and_value_vocab_size = policy_and_value_vocab_size

    self.init_policy_from_world_model_output_dir = (
        init_policy_from_world_model_output_dir
    )

    (self._n_controls, self._n_actions) = (
        serialization_utils.analyze_action_space(train_env.action_space)
    )

    self._policy_and_value_net_fn = functools.partial(
        policy_based_utils.policy_and_value_net,
        bottom_layers_fn=policy_and_value_model,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        vocab_size=self._policy_and_value_vocab_size,
        two_towers=policy_and_value_two_towers,
    )
    (policy_and_value_net, self._substitute_fn) = (
        self._policy_and_value_net_fn()
    )
    self._policy_and_value_net_apply = jax.jit(policy_and_value_net)
    self._policy_and_value_optimizer = policy_and_value_optimizer()
    self._model_state = None
    self._policy_and_value_opt_state = None

  def _get_rng(self):
    self._rng, key = jax_random.split(self._rng)
    return key

  def reset(self, output_dir=None):
    super(PolicyBasedTrainer, self).reset(output_dir)

    # Create summary writers and history.
    if self._should_write_summaries:
      self._train_sw = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'train'))
      self._timing_sw = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'timing'))
      self._eval_sw = jaxboard.SummaryWriter(
          os.path.join(self._output_dir, 'eval'))

    # Try to initialize from a saved checkpoint, or initialize from scratch if
    # there is no saved checkpoint.
    self.update_optimization_state(output_dir)

    # If uninitialized, i.e. _policy_and_value_opt_state is None, then
    # initialize.
    if self._policy_and_value_opt_state is None:
      (policy_and_value_net, _) = self._policy_and_value_net_fn()
      obs_space = self.train_env.observation_space
      act_space = self.train_env.action_space
      input_signature = (
          ShapeDtype(
              (1, self._max_timestep + 1) + obs_space.shape, obs_space.dtype
          ),
          ShapeDtype(
              (1, self._max_timestep) + act_space.shape, act_space.dtype
          ),
      )
      weights, self._model_state = policy_and_value_net.init(
          input_signature, rng=self._get_rng()
      )

      # Initialize the optimizer.
      self._init_state_from_weights(weights)

    # If we need to initialize from the world model, do that here.
    if self.init_policy_from_world_model_output_dir is not None:
      weights = policy_based_utils.init_policy_from_world_model_checkpoint(
          self._policy_and_value_net_weights,
          self.init_policy_from_world_model_output_dir,
          self._substitute_fn,
      )
      # Initialize the optimizer.
      self._init_state_from_weights(weights)

    self._n_trajectories_done_since_last_save = 0
    self._last_saved_at_epoch = self.epoch

    if self._async_mode:
      logging.info('Saving model on startup to have a model policy file.')
      self.save()

  def _init_state_from_weights(self, weights):
    # Initialize the optimizer.
    (init_slots, init_opt_params) = (
        self._policy_and_value_optimizer.tree_init(weights)
    )
    self._policy_and_value_opt_state = (weights, init_slots, init_opt_params)

  def _policy_and_value_opt_update(self, step, grads, opt_state):
    (params, slots, opt_params) = opt_state
    (params, slots) = self._policy_and_value_optimizer.tree_update(
        step, grads, params, slots, opt_params)
    return (params, slots, opt_params)

  # Maybe restore the optimization state. If there is nothing to restore, then
  # epoch = 0 and policy_and_value_opt_state is returned as is.
  def update_optimization_state(self, output_dir=None):
    if output_dir is None:
      output_dir = self._output_dir
    (self._policy_and_value_opt_state, self._model_state, self._epoch,
     self._total_opt_step, self._history) = (
         policy_based_utils.maybe_restore_opt_state(
             output_dir, self._policy_and_value_opt_state, self._model_state
         )
     )

    if self.epoch > 0:
      logging.info('Restored parameters from epoch [%d]', self.epoch)

  @property
  def epoch(self):
    return self._epoch

  @property
  def history(self):
    return self._history

  def collect_trajectories_async(self,
                                 env,
                                 train=True,
                                 n_trajectories=1,
                                 n_observations=None,
                                 temperature=1.0):
    """Collects trajectories in an async manner."""

    assert self._async_mode

    # TODO(afrozm): Make this work, should be easy.
    # NOTE: We still collect whole trajectories, however the async trajectory
    # collectors now will poll not on the amount of trajectories collected but
    # on the amount of observations in the completed trajectories and bail out.
    assert n_observations is None

    # trajectories/train and trajectories/eval are the two subdirectories.
    trajectory_dir = os.path.join(self._output_dir, 'trajectories',
                                  'train' if train else 'eval')
    epoch = self.epoch

    logging.info(
        'Loading [%s] trajectories from dir [%s] for epoch [%s] and temperature'
        ' [%s]', n_trajectories, trajectory_dir, epoch, temperature)

    bt = trajectory.BatchTrajectory.load_from_directory(
        trajectory_dir,
        epoch=epoch,
        temperature=temperature,
        wait_forever=True,
        n_trajectories=n_trajectories)

    if bt is None:
      logging.error(
          'Couldn\'t load [%s] trajectories from dir [%s] for epoch [%s] and '
          'temperature [%s]', n_trajectories, trajectory_dir, epoch,
          temperature)
      assert bt

    # Doing this is important, since we want to modify `env` so that it looks
    # like `env` was actually played and the trajectories came from it.
    env.trajectories = bt

    trajs = env_problem_utils.get_completed_trajectories_from_env(
        env, n_trajectories)
    n_done = len(trajs)
    timing_info = {}
    return trajs, n_done, timing_info, self._model_state

  def collect_trajectories(self,
                           train=True,
                           n_trajectories=1,
                           n_observations=None,
                           temperature=1.0,
                           abort_fn=None,
                           raw_trajectory=False):
    key = self._get_rng()

    env = self.train_env
    max_timestep = self._max_timestep
    should_reset = self._should_reset_train_env
    if not train:  # eval
      env = self.eval_env
      max_timestep = self._max_timestep_eval
      should_reset = True

    # If async, read the required trajectories for the epoch.
    if self._async_mode:
      trajs, n_done, timing_info, self._model_state = self.collect_trajectories_async(
          env,
          train=train,
          n_trajectories=n_trajectories,
          n_observations=n_observations,
          temperature=temperature)
    else:
      trajs, n_done, timing_info, self._model_state = (
          policy_based_utils.collect_trajectories(
              env,
              policy_fn=self._policy_fun,
              n_trajectories=n_trajectories,
              n_observations=n_observations,
              max_timestep=max_timestep,
              state=self._model_state,
              rng=key,
              len_history_for_policy=self._len_history_for_policy,
              boundary=self._boundary,
              reset=should_reset,
              temperature=temperature,
              abort_fn=abort_fn,
              raw_trajectory=raw_trajectory,
          )
      )

    if train:
      self._n_trajectories_done_since_last_save += n_done

    return trajs, n_done, timing_info, self._model_state

  def train_epoch(self, evaluate=True):
    raise NotImplementedError

  def evaluate(self):
    """Evaluate the agent."""
    if not self._separate_eval:
      return

    logging.vlog(1, 'PolicyBasedTrainer epoch [% 6d]: evaluating policy.',
                 self.epoch)

    processed_reward_sums = collections.defaultdict(list)
    raw_reward_sums = collections.defaultdict(list)
    for _ in range(self._n_evals):
      for temperature in self._eval_temperatures:
        trajs, _, _, self._model_state = self.collect_trajectories(
            train=False, temperature=temperature)

        processed_reward_sums[temperature].extend(
            sum(traj[2]) for traj in trajs)
        raw_reward_sums[temperature].extend(sum(traj[3]) for traj in trajs)

    # Return the mean and standard deviation for each temperature.
    def compute_stats(reward_dict):
      # pylint: disable=g-complex-comprehension
      return {
          temperature: {
              'mean': onp.mean(rewards),
              'std': onp.std(rewards)
          } for (temperature, rewards) in reward_dict.items()
      }
      # pylint: enable=g-complex-comprehension

    reward_stats = {
        'processed': compute_stats(processed_reward_sums),
        'raw': compute_stats(raw_reward_sums),
    }

    policy_based_utils.write_eval_reward_summaries(
        reward_stats, self._log, epoch=self.epoch
    )

  def maybe_save(self):
    # Save parameters every time we see the end of at least a fraction of batch
    # number of trajectories that are done (not completed -- completed includes
    # truncated and done).
    # Also don't save too frequently, enforce a minimum gap.
    min_done_trajs_to_save = (
        self._done_frac_for_policy_save *
        getattr(self.train_env, 'batch_size', 10))
    # TODO(afrozm): Refactor to trax.save_trainer_state.
    if (self._n_trajectories_done_since_last_save >= min_done_trajs_to_save and
        self.epoch % self._save_every_n == 0) or self._async_mode:
      self.save()

  def save(self):
    """Save the agent parameters."""
    if not self._should_save_checkpoints:
      return

    logging.vlog(1, 'PolicyBasedTrainer epoch [% 6d]: saving model.',
                 self.epoch)
    policy_based_utils.save_opt_state(
        self._output_dir,
        self._policy_and_value_opt_state,
        self._model_state,
        self.epoch,
        self._total_opt_step,
        self._history,
    )
    # Reset this number.
    self._n_trajectories_done_since_last_save = 0
    self._last_saved_at_epoch = self.epoch

  def flush_summaries(self):
    if self._should_write_summaries:
      self._train_sw.flush()
      self._timing_sw.flush()
      self._eval_sw.flush()

  def _log(self, mode, metric, value):
    if self._should_write_summaries:
      summary_writer = {
          'train': self._train_sw,
          'eval': self._eval_sw,
      }[mode]
      summary_writer.scalar(metric, value, step=self.epoch)
    self._history.append(mode, metric, self.epoch, value)

  def _policy_and_value_get_params(self, opt_state):
    # (params, slots, opt_params)
    (params, _, _) = opt_state
    return params

  @property
  def _policy_and_value_net_weights(self):
    return self._policy_and_value_get_params(self._policy_and_value_opt_state)

  # Prepares the trajectories for policy training.
  def _preprocess_trajectories(self, trajectories):
    (_, reward_mask, observations, actions, rewards, infos) = (
        policy_based_utils.pad_trajectories(
            trajectories, boundary=self._max_timestep)
    )
    if actions.ndim == 2:
      # Add the control dimension.
      actions = actions[:, :, None]
    (low, high) = self.train_env.reward_range
    outside = np.logical_or(rewards < low, rewards > high)
    rewards = jax.ops.index_update(rewards, jax.ops.index[outside], 0)
    assert self.train_env.observation_space.shape == observations.shape[2:]
    return (observations, actions, rewards, reward_mask, infos)

  def _policy_fun(self, observations, lengths, state, rng):
    return policy_based_utils.run_policy(
        self._policy_and_value_net_apply,
        observations,
        lengths,
        self._policy_and_value_net_weights,
        state,
        rng,
        self.train_env.action_space,
    )

  def _policy_fun_all_timesteps(self, observations, lengths, state, rng):
    return policy_based_utils.run_policy_all_timesteps(
        self._policy_and_value_net_apply,
        observations,
        self._policy_and_value_net_weights,
        state,
        rng,
        self.train_env.action_space,
    )

  @staticmethod
  def _log_shape(array_name, array):
    logging.vlog(1, f'Shape of {array_name} is {array.shape}.')

  @staticmethod
  def _check_shapes(array_name,
                    expected_shape_string,
                    array,
                    expected_shape,
                    array_prefix=None):
    actual_shape = array.shape[:array_prefix]
    prefix = '' if not array_prefix else f'[:{array_prefix}]'
    logging.vlog(1, f'Shape of {array_name}{prefix} is {actual_shape}.')
    if array_prefix:
      logging.vlog(1, f'Shape of {array_name} is {array.shape}.')
    if actual_shape != expected_shape:
      raise ValueError(
          f'Shape of {array_name}{prefix} is expected to be '
          f'{expected_shape_string} which is {expected_shape}, but is '
          f'{actual_shape} instead.')
