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
from trax.rl.envs import data_envs


class SequenceDataEnvTest(absltest.TestCase):

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
    x2, r0, d0, _ = env.step(0)
    self.assertEqual(r0, 0.0)
    self.assertEqual(d0, False)
    eos, r1, d1, _ = env.step(0)
    self.assertEqual(eos, 1)
    self.assertEqual(r1, 0.0)
    self.assertEqual(d1, False)
    y1, r2, d2, _ = env.step(x1)
    self.assertEqual(y1, x1)
    self.assertEqual(r2, 0.0)
    self.assertEqual(d2, False)
    y2, r3, d3, _ = env.step(x2)
    self.assertEqual(y2, x2)
    self.assertEqual(r3, 0.0)
    self.assertEqual(d3, False)
    eos2, r4, d4, _ = env.step(1)
    self.assertEqual(eos2, 1)
    self.assertEqual(r4, 1.0)
    self.assertEqual(d4, True)

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


if __name__ == '__main__':
  absltest.main()
