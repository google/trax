# coding=utf-8
# Copyright 2021 The Trax Authors.
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
"""Tests for RL environments created from supervised data-sets."""

from absl.testing import absltest
import numpy as np
from trax.rl.envs import data_envs


class SequenceDataEnvTest(absltest.TestCase):

  def _assert_masks(self, info, control, discount):
    self.assertEqual(info, {'control_mask': control, 'discount_mask': discount})

  def test_copy_task_short_sequence_correct_actions(self):
    """Test sequence data env on the copying task, correct replies.

    With input (x1, x2) this tests for the following sequence of
    (observations, rewards, dones, actions):
    x1                 = env.reset()
    x2,   0.0,  F, _   = env.step(ignored_action)
    eos,  0.0,  F, _   = env.step(ignored_action)
    x1,   0.0,  F, _   = env.step(x1)
    x2,   0.0,  F, _   = env.step(x2)
    eos,  1.0,  T, _   = env.step(eos)
    """
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(2, n=1), 16)
    x1 = env.reset()
    x2, r0, d0, i0 = env.step(0)
    self.assertEqual(r0, 0.0)
    self.assertEqual(d0, False)
    self._assert_masks(i0, control=0, discount=0)
    eos, r1, d1, i1 = env.step(0)
    self.assertEqual(eos, 1)
    self.assertEqual(r1, 0.0)
    self.assertEqual(d1, False)
    self._assert_masks(i1, control=0, discount=0)
    y1, r2, d2, i2 = env.step(x1)
    self.assertEqual(y1, x1)
    self.assertEqual(r2, 0.0)
    self.assertEqual(d2, False)
    self._assert_masks(i2, control=1, discount=0)
    y2, r3, d3, i3 = env.step(x2)
    self.assertEqual(y2, x2)
    self.assertEqual(r3, 0.0)
    self.assertEqual(d3, False)
    self._assert_masks(i3, control=1, discount=0)
    eos2, r4, d4, i4 = env.step(1)
    self.assertEqual(eos2, 1)
    self.assertEqual(r4, 1.0)
    self.assertEqual(d4, True)
    self._assert_masks(i4, control=1, discount=1)

  def test_copy_task_longer_sequnece_mixed_actions(self):
    """Test sequence data env on the copying task, mixed replies.

    With input (x1, x2) and (y1, y2) this tests for the following sequence of
    (observations, rewards, dones, actions):
    x1                 = env.reset()
    x2,   0.0,  F, _   = env.step(ignored_action)
    eos,  0.0,  F, _   = env.step(ignored_action)
    x1,   0.0,  F, _   = env.step(x1)
    x2+1, 0.0,  F, _   = env.step(x2+1)
    y1,   0,5,  F, _   = env.step(eos)
    y2,   0.0,  F, _   = env.step(ignored_action)
    eos,  0.0,  F, _   = env.step(ignored_action)
    y1+1  0.0,  F, _   = env.step(y1+1)
    y2+1, 0.0,  F, _   = env.step(y2+1)
    eos,  0.0,  T, _   = env.step(eos)
    """
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(2, n=2), 16)
    x1 = env.reset()
    x2, _, _, _ = env.step(0)
    eos, _, _, _ = env.step(0)
    _, _, _, _ = env.step(x1)
    _, _, _, _ = env.step(x2 + 1)  # incorrect
    y1, r1, d1, _ = env.step(1)
    self.assertEqual(r1, 0.5)
    self.assertEqual(d1, False)
    y2, _, _, _ = env.step(0)
    eos, _, _, _ = env.step(0)
    _, _, _, _ = env.step(y1 + 1)  # incorrect
    _, _, _, _ = env.step(y2 + 1)  # incorrect
    eos, r2, d2, _ = env.step(1)
    self.assertEqual(eos, 1)
    self.assertEqual(r2, 0.0)
    self.assertEqual(d2, True)

  def test_copy_task_action_observation_space(self):
    """Test that sequence data env returns correct action/observation space."""
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(2, n=1), 16)
    self.assertEqual(env.action_space.n, 16)
    self.assertEqual(env.observation_space.n, 16)

  def test_copy_task_max_length(self):
    """Test that sequence data env respects max_length."""
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(10, n=1), 16,
                                    max_length=2)
    obs = env.reset()
    for _ in range(10):
      obs, reward, done, _ = env.step(0)
      self.assertEqual(reward, 0.0)
      self.assertEqual(done, False)
    self.assertEqual(obs, 1)  # produces EOS
    obs, reward, done, _ = env.step(7)
    self.assertEqual(obs, 7)  # repeats action
    self.assertEqual(reward, 0.0)
    self.assertEqual(done, False)
    obs, reward, done, _ = env.step(8)
    self.assertEqual(obs, 8)  # repeats action
    self.assertEqual(reward, 0.0)
    self.assertEqual(done, False)
    obs, reward, done, _ = env.step(9)
    self.assertEqual(done, True)  # exceeded max_length, stop
    self.assertEqual(obs, 1)      # produce EOS on done
    obs, reward, done, _ = env.step(10)
    self.assertEqual(done, True)  # continue producing done = True
    self.assertEqual(obs, 1)      # continue producing EOS

  def test_number_of_active_masks(self):
    """Test that we have the correct number of control and discount masks."""
    n_input_seqs = 3
    n_output_seqs = 2
    input_len = 4
    output_len = 5

    def data_stream():
      i = 2 * np.ones(input_len)
      o = np.zeros(output_len)
      while True:
        yield (i, o, i, o, i)  # 3 input, 2 output sequences.

    env = data_envs.SequenceDataEnv(data_stream(), 16, max_length=output_len)
    env.reset()

    n_discount = 0
    n_control = 0
    n_steps = 0
    done = False
    while not done:
      (_, _, done, info) = env.step(action=0)
      n_discount += info['discount_mask']
      n_control += info['control_mask']
      n_steps += 1

    # One discount_mask=1 per output sequence.
    self.assertEqual(n_discount, n_output_seqs)
    # One control_mask=1 per output token, including EOS, because it's also
    # controlled by the agent.
    self.assertEqual(n_control, (output_len + 1) * n_output_seqs)
    # One control_mask=0 per input token, excluding EOS, because when the env
    # emits it, control transfers to the agent immediately.
    self.assertEqual(n_steps - n_control, input_len * n_input_seqs)


if __name__ == '__main__':
  absltest.main()
