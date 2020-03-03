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

"""Tests for trax.rl.policy_based_utils."""

import gym
import numpy as np
from tensorflow import test
from tensorflow.compat.v1.io import gfile
from trax import layers
from trax import shapes
from trax.rl import policy_based_utils
from trax.supervised import trainer_lib


class PolicyBasedUtilsTest(test.TestCase):

  def setUp(self):
    super(PolicyBasedUtilsTest, self).setUp()
    self.rng_key = trainer_lib.init_random_number_generators(0)

  def test_get_policy_model_files(self):
    output_dir = self.get_temp_dir()

    def write_policy_model_file(epoch):
      with gfile.GFile(policy_based_utils.get_policy_model_file_from_epoch(
          output_dir, epoch
      ), 'w') as f:
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

    policy_model_files = policy_based_utils.get_policy_model_files(output_dir)

    self.assertEqual(expected_policy_model_files, policy_model_files)

    gfile.rmtree(output_dir)

  def test_get_epoch_from_policy_model_file(self):
    self.assertEqual(
        policy_based_utils.get_epoch_from_policy_model_file('model-000000.pkl'),
        0,
    )
    self.assertEqual(
        policy_based_utils.get_epoch_from_policy_model_file('model-123456.pkl'),
        123456,
    )

  def test_get_policy_model_file_from_epoch(self):
    self.assertEqual(
        policy_based_utils.get_policy_model_file_from_epoch('/tmp', 0),
        '/tmp/model-000000.pkl',
    )
    self.assertEqual(
        policy_based_utils.get_policy_model_file_from_epoch('/tmp', 123456),
        '/tmp/model-123456.pkl',
    )

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
    (pnv_model, _) = policy_based_utils.policy_and_value_net(
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
    padded_trajectories = policy_based_utils.pad_trajectories(
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

  def test_saves_and_restores_opt_state(self):
    opt_state = 123
    state = 456
    epoch = 7
    opt_step = 89
    history = 0
    output_dir = self.get_temp_dir()
    policy_based_utils.save_opt_state(
        output_dir, opt_state, state, epoch, opt_step, history
    )
    restored_data = policy_based_utils.maybe_restore_opt_state(output_dir)
    self.assertEqual(
        restored_data, (opt_state, state, epoch, opt_step, history)
    )

  def _make_run_policy_kwargs(self, action_space):
    return {
        'weights': None,
        'state': None,
        'rng': self.rng_key,
        'action_space': action_space,
    }


if __name__ == '__main__':
  test.main()
