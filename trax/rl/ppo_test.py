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

"""Tests for trax.rl.ppo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import gym
import jax
import numpy as np
from tensorflow import test
from tensorflow.compat.v1.io import gfile
from trax import layers
from trax import shapes
from trax.rl import ppo
from trax.supervised import trainer_lib


class PpoTest(test.TestCase):

  def setUp(self):
    super(PpoTest, self).setUp()
    self.rng_key = trainer_lib.init_random_number_generators(0)

  def test_get_policy_model_files(self):
    output_dir = self.get_temp_dir()

    def write_policy_model_file(epoch):
      with gfile.GFile(
          ppo.get_policy_model_file_from_epoch(output_dir, epoch), 'w') as f:
        f.write('some data')

    epochs = [200, 100, 300]

    # 300, 200, 100
    expected_policy_model_files = [
        output_dir + '/model-000300.pkl',
        output_dir + '/model-000200.pkl',
        output_dir + '/model-000100.pkl',
    ]

    for epoch in epochs:
      write_policy_model_file(epoch)

    policy_model_files = ppo.get_policy_model_files(output_dir)

    self.assertEqual(expected_policy_model_files, policy_model_files)

    gfile.rmtree(output_dir)

  def test_get_epoch_from_policy_model_file(self):
    self.assertEqual(0,
                     ppo.get_epoch_from_policy_model_file('model-000000.pkl'))
    self.assertEqual(123456,
                     ppo.get_epoch_from_policy_model_file('model-123456.pkl'))

  def test_get_policy_model_file_from_epoch(self):
    self.assertEqual('/tmp/model-000000.pkl',
                     ppo.get_policy_model_file_from_epoch('/tmp', 0))
    self.assertEqual('/tmp/model-123456.pkl',
                     ppo.get_policy_model_file_from_epoch('/tmp', 123456))

  def test_policy_and_value_net(self):
    observation_shape = (3, 4, 5)
    n_actions = 2
    n_controls = 3
    batch = 2
    time_steps = 10
    observations = np.random.uniform(
        size=(batch, time_steps) + observation_shape)
    actions = np.random.randint(
        n_actions, size=(batch, time_steps - 1, n_controls))
    (pnv_model, _) = ppo.policy_and_value_net(
        bottom_layers_fn=lambda: [layers.Flatten(n_axes_to_keep=2)],
        observation_space=gym.spaces.Box(
            shape=observation_shape, low=0, high=1
        ),
        action_space=gym.spaces.MultiDiscrete((n_actions,) * n_controls),
        vocab_size=None,
        two_towers=True,
    )
    input_signature = shapes.signature((observations, actions))
    _, _ = pnv_model.init(input_signature)

    (action_logits, values) = pnv_model((observations, actions))

    # Output is a list, first is probab of actions and the next is value output.
    self.assertEqual(
        (batch, time_steps, n_controls, n_actions), action_logits.shape)
    self.assertEqual((batch, time_steps), values.shape)

  def test_pad_trajectories(self):
    observation_shape = (2, 3, 4)
    trajectories = []
    n_trajectories = 7
    n_actions = 10

    # Time-steps are between [min_allowable_time_step, max_allowable_time_step]
    max_allowable_time_step = 19
    min_allowable_time_step = 5

    # The actual max we see in the data.
    max_time_step = -1

    # Bucket length.
    bucket_length = 15

    # Make `n_trajectories` random trajectories.
    for i in range(n_trajectories):
      time_steps = np.random.randint(min_allowable_time_step,
                                     max_allowable_time_step + 1)
      if time_steps > max_time_step:
        max_time_step = time_steps
      observations = np.random.randint(
          0, 255, size=(time_steps + 1,) + observation_shape).astype(np.uint8)
      rewards = np.random.uniform(size=(time_steps,)).astype(np.float32)
      actions = np.random.randint(
          0, n_actions, size=(time_steps,)).astype(np.int32)
      infos = {
          'a': np.random.uniform(size=(time_steps,)).astype(np.float32),
          'b': np.random.uniform(size=(time_steps,)).astype(np.float32)
      }
      trajectories.append((observations, rewards, actions, infos))

    # Now pad these trajectories.
    padded_trajectories = ppo.pad_trajectories(
        trajectories, boundary=bucket_length)

    # Expected padding.
    i = 1
    while i * bucket_length < max_time_step:
      i += 1
    expected_padding = i * bucket_length

    # Get the padded objects.
    (pad_lengths, reward_mask, padded_observations, padded_actions,
     padded_rewards, padded_infos) = padded_trajectories

    # Expectations on the padded shapes.
    self.assertEqual(padded_observations.shape, (
        n_trajectories,
        expected_padding + 1,
    ) + observation_shape)
    self.assertEqual(padded_actions.shape, (n_trajectories, expected_padding))
    self.assertEqual(padded_rewards.shape, (n_trajectories, expected_padding))
    self.assertEqual(reward_mask.shape, (n_trajectories, expected_padding))

    self.assertEqual(padded_infos['a'].shape,
                     (n_trajectories, expected_padding))
    self.assertEqual(padded_infos['b'].shape,
                     (n_trajectories, expected_padding))

    # Assert that the padding lengths and reward mask are consistent.
    self.assertAllEqual(
        np.full((n_trajectories,), expected_padding),
        np.array(np.sum(reward_mask, axis=1)) + pad_lengths)

  def test_rewards_to_go(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5

    rewards_to_go = ppo.rewards_to_go(rewards, rewards_mask, gamma)

    self.assertAllEqual(
        np.array([
            [5, 8, 12, 16, 16, 0, 0, 0],
            [1.984375, 1.96875, 1.9375, 1.875, 1.75, 1.5, 1.0, 0],
        ]), rewards_to_go)

  def test_rewards_to_go_really_long_sequences(self):
    T = 1200  # pylint: disable=invalid-name

    rewards = np.random.uniform(1e-3, 1e-2, (1, T))

    # Make a mask, clear out a fixed number `L` of 1s from the end.
    L = 36  # pylint: disable=invalid-name
    assert L < T
    rewards_mask = np.ones_like(rewards)
    rewards_mask[0, L:] = 0

    gamma = 0.94

    actual_r2g = ppo.rewards_to_go(rewards, rewards_mask, gamma).reshape(-1)

    # Let's compute r2g the slow way.
    masked_rewards = (rewards_mask * rewards).reshape(-1)
    expected_r2g = np.zeros_like(masked_rewards)
    for t in range(T):
      for j in range(t, T):
        expected_r2g[t] += (gamma**(j - t)) * masked_rewards[j]

    self.assertAllClose(expected_r2g, actual_r2g)

  def test_value_loss(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5
    epsilon = 0.1

    # Random observations and a value function that returns a constant value.
    # NOTE: Observations have an extra time-step.
    B, T = rewards.shape  # pylint: disable=invalid-name
    observation_shape = (210, 160, 3)  # atari pong
    random_observations = np.random.uniform(size=(B, T + 1) + observation_shape)

    def value_net_apply(observations, params, rng=None):
      del params, rng
      # pylint: disable=invalid-name
      B, T_p_1, OBS = (observations.shape[0], observations.shape[1],
                       observations.shape[2:])
      del OBS
      return np.ones((B, T_p_1))
      # pylint: enable=invalid-name

    value_prediction = value_net_apply(random_observations, [])

    with jax.disable_jit():
      (value_loss, _) = ppo.value_loss_given_predictions(
          value_prediction,
          rewards,
          rewards_mask,
          gamma,
          epsilon)

    self.assertNear(53.3637084961, value_loss, 1e-6)

  def test_deltas(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    B, T = rewards.shape  # pylint: disable=invalid-name

    # Say, all predicted values are 1.
    predicted_values = np.ones((B, T + 1))

    gamma = 1.0

    td_residuals = ppo.deltas(predicted_values, rewards, rewards_mask, gamma)

    # With V(s) being the same for all s, td_residuals should be
    # equal to the rewards + (\gamma - 1)*v(s), masked in the right places.
    truncated_pv = predicted_values[:, :-1]
    masked_rewards = rewards * rewards_mask
    expected_residuals = (masked_rewards +
                          (gamma - 1) * truncated_pv) * rewards_mask
    self.assertAllEqual(expected_residuals, td_residuals)

    gamma = 0.5
    td_residuals = ppo.deltas(predicted_values, rewards, rewards_mask, gamma)
    expected_residuals = (masked_rewards +
                          (gamma - 1) * truncated_pv) * rewards_mask
    self.assertAllEqual(expected_residuals, td_residuals)

  def test_gae_advantages(self):
    td_deltas = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5
    lambda_ = 1.0

    expected_gae_advantages = np.array([
        [5, 8, 12, 16, 16, 0, 0, 0],
        [1.984375, 1.96875, 1.9375, 1.875, 1.75, 1.5, 1.0, 0],
    ])

    gae_advantages = ppo.gae_advantages(td_deltas * rewards_mask, rewards_mask,
                                        lambda_, gamma)
    self.assertAllEqual(expected_gae_advantages, gae_advantages)

    gamma = 1.0
    lambda_ = 0.5

    gae_advantages = ppo.gae_advantages(td_deltas * rewards_mask, rewards_mask,
                                        lambda_, gamma)
    self.assertAllEqual(expected_gae_advantages, gae_advantages)

  def test_chosen_probabs(self):
    # Shape (2, 2, 1, 3)
    probab_observations = np.array(
        [[[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]],
         [[0.3, 0.1, 0.6], [0.1, 0.1, 0.8]]]
    )[:, :, None, :]

    # Shape (2, 2, 1)
    actions = np.array([[1, 2], [0, 1]])[:, :, None]

    chosen_probabs = ppo.chosen_probabs(probab_observations, actions)

    self.assertAllEqual(
        np.array([[0.2, 0.5], [0.3, 0.1]])[:, :, None], chosen_probabs)

  def test_compute_probab_ratios(self):
    p_old = np.array([[
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.4), np.log(0.1), np.log(0.4), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
    ], [
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.1), np.log(0.4), np.log(0.4)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
    ]])[:, :, None, :]

    p_new = np.array([[
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.4), np.log(0.1), np.log(0.1), np.log(0.3)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
    ], [
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
        [np.log(0.1), np.log(0.1), np.log(0.2), np.log(0.6)],
        [np.log(0.3), np.log(0.1), np.log(0.3), np.log(0.3)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
    ]])[:, :, None, :]

    actions = np.array([[1, 2, 0, 1], [0, 3, 3, 0]])[:, :, None]

    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])[:, :, None]

    probab_ratios = ppo.compute_probab_ratios(p_new, p_old, actions, mask)

    self.assertAllClose(
        np.array([
            [0.1 / 0.2, 0.1 / 0.4, 0.0, 0.0],
            [0.1 / 0.3, 0.6 / 0.4, 0.3 / 0.1, 0.0],
        ])[:, :, None], probab_ratios)

  def test_clipped_probab_ratios(self):
    probab_ratios = np.array([
        [1.5, 1.0, 0.5, 0.7],
        [2.5, 2.0, 0.1, 1.0],
    ])

    clipped_probab_ratios = ppo.clipped_probab_ratios(probab_ratios, 0.1)

    self.assertAllClose(
        np.array([
            [1.1, 1.0, 0.9, 0.9],
            [1.1, 1.1, 0.9, 1.0],
        ]), clipped_probab_ratios)

  def test_clipped_objective(self):
    probab_ratios = np.array([
        [1.5, 2.0, 0.5, 0.7],
        [2.5, 2.0, 0.1, 1.0],
    ])

    advantages = np.array([
        [0.1, -0.1, 0.5, 0.7],
        [2.0, -2.0, 2.0, 2.0],
    ])

    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])

    epsilon = 0.1

    clipped_probab_ratios = np.array([
        [1.1, 1.1, 0.9, 0.9],
        [1.1, 1.1, 0.9, 1.0],
    ])

    unused_advantages_x_probab_ratios = np.array([
        [0.15, -0.2, 0.25, 0.49],
        [5.00, -4.0, 0.20, 2.00]
    ])

    unused_advantages_x_clipped_probab_ratios = np.array([
        [0.11, -0.11, 0.45, 0.63],
        [2.20, -2.20, .80, 2.00]
    ])

    unused_minimums = np.array([
        [0.11, -0.2, 0.25, 0.49],
        [2.20, -4.0, 0.20, 2.00]
    ])

    # minimums * mask
    objective = np.array([
        [0.11, -0.2, 0.0, 0.],
        [2.20, -4.0, 0.2, 0.]
    ])

    # Assert that we computed things correctly in this test.
    self.assertAllClose(
        np.minimum(probab_ratios * advantages,
                   clipped_probab_ratios * advantages) * mask,
        objective)

    self.assertAllClose(
        objective,
        ppo.clipped_objective(probab_ratios, advantages, mask, epsilon))

  def test_combined_loss(self):
    B, T, C, A, OBS = 2, 10, 1, 2, (28, 28, 3)  # pylint: disable=invalid-name

    make_net = lambda: ppo.policy_and_value_net(  # pylint: disable=g-long-lambda
        bottom_layers_fn=lambda: [layers.Flatten(n_axes_to_keep=2)],
        observation_space=gym.spaces.Box(shape=OBS, low=0, high=1),
        action_space=gym.spaces.Discrete(A),
        vocab_size=None,
        two_towers=True,
    )[0]
    net = make_net()

    observations = np.random.uniform(size=(B, T + 1) + OBS)
    actions = np.random.randint(0, A, size=(B, T, C))
    input_signature = shapes.signature((observations, actions))
    old_params, _ = net.init(input_signature)
    new_params, state = make_net().init(input_signature)

    # Generate a batch of observations.

    rewards = np.random.uniform(0, 1, size=(B, T))
    mask = np.ones_like(rewards)

    # Just test that this computes at all.
    (new_log_probabs, value_predictions_new) = (
        net((observations, actions), weights=new_params, state=state))
    (old_log_probabs, value_predictions_old) = (
        net((observations, actions), weights=old_params, state=state))

    gamma = 0.99
    lambda_ = 0.95
    epsilon = 0.2
    value_weight = 1.0
    entropy_weight = 0.01

    nontrainable_params = {
        'gamma': gamma,
        'lambda': lambda_,
        'epsilon': epsilon,
        'value_weight': value_weight,
        'entropy_weight': entropy_weight,
    }

    (value_loss_1, _) = ppo.value_loss_given_predictions(
        value_predictions_new, rewards, mask, gamma=gamma,
        value_prediction_old=value_predictions_old, epsilon=epsilon)
    (ppo_loss_1, _) = ppo.ppo_loss_given_predictions(
        new_log_probabs[:, :-1],
        old_log_probabs[:, :-1],
        value_predictions_old,
        actions,
        rewards,
        mask,
        gamma=gamma,
        lambda_=lambda_,
        epsilon=epsilon)

    (combined_loss, (ppo_loss_2, value_loss_2, entropy_bonus), _, state) = (
        ppo.combined_loss(new_params,
                          old_log_probabs[:, :-1],
                          value_predictions_old,
                          net,
                          observations,
                          actions,
                          rewards,
                          mask,
                          nontrainable_params=nontrainable_params,
                          state=state)
    )

    # Test that these compute at all and are self consistent.
    self.assertGreater(entropy_bonus, 0.0)
    self.assertNear(value_loss_1, value_loss_2, 1e-6)
    self.assertNear(ppo_loss_1, ppo_loss_2, 1e-6)
    self.assertNear(
        combined_loss,
        ppo_loss_2 + (value_weight * value_loss_2) -
        (entropy_weight * entropy_bonus),
        1e-6
    )

  def test_masked_entropy(self):
    # (2, 4+1, 4)
    log_probs = np.array([[
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.4), np.log(0.1), np.log(0.4), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
    ], [
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.1), np.log(0.4), np.log(0.4)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
    ]])[:, :, None, :]

    # (2, 4)
    mask = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0]
    ])

    def plp(p):
      return p * np.log(p)

    # Removing the last time-step and the masked stuff, gets us this.
    filtered_log_probs = np.array([[
        [plp(0.1), plp(0.2), plp(0.6), plp(0.1)],
        [plp(0.4), plp(0.1), plp(0.4), plp(0.1)],
        [plp(0.3), plp(0.1), plp(0.5), plp(0.1)],
        [plp(0.1), plp(0.1), plp(0.4), plp(0.4)],
        [plp(0.3), plp(0.1), plp(0.5), plp(0.1)],
    ]])

    self.assertNear(ppo.masked_entropy(log_probs, mask),
                    -np.sum(filtered_log_probs) / 5.0,
                    1e-6)

  def test_saves_and_restores_opt_state(self):
    opt_state = 123
    state = 456
    epoch = 7
    opt_step = 89
    history = 0
    output_dir = self.get_temp_dir()
    ppo.save_opt_state(output_dir, opt_state, state, epoch, opt_step, history)
    restored_data = ppo.maybe_restore_opt_state(output_dir)
    self.assertEqual(
        restored_data, (opt_state, state, epoch, opt_step, history)
    )

  def test_shuffled_index_batches_generates_valid_batch(self):
    dataset_size = 16
    batch_size = 4
    stream = ppo.shuffled_index_batches(dataset_size, batch_size)
    batch = next(stream)
    self.assertEqual(batch.shape, (batch_size,))
    # Assert that all indices are different.
    self.assertEqual(len(set(batch)), batch_size)

  def test_shuffled_index_batches_generates_all_indices(self):
    dataset_size = 16
    batch_size = 4
    stream = ppo.shuffled_index_batches(dataset_size, batch_size)
    indices = np.reshape(
        list(itertools.islice(stream, dataset_size // batch_size)), -1
    )
    self.assertEqual(set(indices), set(range(dataset_size)))

  def test_shuffled_index_batches_gives_different_permutations(self):
    dataset_size = 256
    batch_size = 8
    stream1 = ppo.shuffled_index_batches(dataset_size, batch_size)
    stream2 = ppo.shuffled_index_batches(dataset_size, batch_size)
    self.assertFalse(np.array_equal(next(stream1), next(stream2)))

  def _make_run_policy_kwargs(self, action_space):
    return {
        'weights': None,
        'state': None,
        'rng': self.rng_key,
        'action_space': action_space,
    }

  def _make_log_prob_and_value_seqs(
      self, log_probs, values, start_indices, n_symbols
  ):
    (batch_size, n_controls, n_actions) = log_probs.shape
    log_prob_seq = np.zeros((batch_size, n_symbols * n_controls, n_actions))
    value_seq = np.zeros((batch_size, n_symbols * n_controls))
    for (x, x_seq) in ((log_probs, log_prob_seq), (values, value_seq)):
      for (i, start_index) in enumerate(start_indices):
        x_seq[i, start_index:(start_index + n_controls)] = x[i]
    return (log_prob_seq, value_seq)


if __name__ == '__main__':
  test.main()
