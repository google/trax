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
"""RL environments created from supervised data-sets."""

import gym
import numpy as np


class SequenceDataEnv(object):
  """RL environment created from a generator of sequential data.

  This class allows to create RL environments from supervised sequential data,
  such as tokenized natural languague processing tasks. The data comes as:
    (input1, output1, input2, output2, ...)
  where inputs and outputs are all sequences of integers.

  For example, with input (2, 3) and output (4, 5), so data = [(2, 3), (4, 5)],
  the sequence of (observations, rewards, actions) will look like:
    2                   = env.reset()                # first observation
    3,    0.0,   _, _   = env.step(ignored_action)
    eos,  0.0,   _, _   = env.step(ignored_action)
    act1, 0.0,   _, _   = env.step(act1)             # observation = action
    act2, 0.0,   _, _   = env.step(act2)             # observation = action
    eos,  score, _, _   = env.step(eos)

  where score = metric((4, 5), (act1, act2)) is the reward gotten from
  comparing the two actions to the actual output from the data.

  The environment first presents the input as observations, doing this
  sequentially, token-by-token, and ignoring all actions taken by the policy.
  Then, the policy is asked to generate the response, again, token-by-token,
  until it generates EOS. Generated tokens are repeated as observations.
  When EOS is encountered, a metric is computed between the generated
  output and the output from data, and this metric is returned as reward.
  """

  def __init__(self, data_stream, vocab_size, metric=None,
               eos_id=1, max_length=1000):
    """The constructor.

    Args:
      data_stream: A python generator creating lists or tuples of
        sequences (list, tuples or numpy arrays) of integers.
      vocab_size: Integer, the size of the vocabulary. All integers in the
        data stream must be positive and smaller than this value.
      metric: A function taking two lists of integers and returning a float.
        If None, we use per-token accuracy as the default metric.
      eos_id: Integer, the id of the EOS symbol.
      max_length: Integer, maximum length of the policy reply to avoid
        infinite episodes if policy never produces EOS.

    Returns:
      A new environment which presents the data and compares the policy
      response with the expected data, returning metric as reward.
    """
    self._data = data_stream
    self._vocab_size = vocab_size
    self._eos = eos_id
    self._max_length = max_length
    self._metric = _accuracy if metric is None else metric
    self.reset()

  @property
  def _on_input(self):
    """Return True if we're currently processing input, False if output."""
    cur_sequence_id, _ = self._cur_position
    return cur_sequence_id % 2 == 0

  @property
  def observation(self):
    cur_sequence_id, cur_token_id = self._cur_position
    if cur_sequence_id >= len(self._cur_sequence):
      obs = self._eos
    elif self._on_input:
      obs = self._cur_sequence[cur_sequence_id][cur_token_id]
    else:
      obs = self._response[-1] if self._response else self._eos
    return np.array(int(obs), dtype=np.int32)

  @property
  def action_space(self):
    return gym.spaces.Discrete(self._vocab_size)

  @property
  def observation_space(self):
    return gym.spaces.Discrete(self._vocab_size)

  def reset(self):
    """Reset this environment."""
    self._cur_sequence = next(self._data)
    # Position contains 2 indices: which sequnece are we in? (input1, output1,
    # input2, output2 and so on) and which token in the sequence are we in?
    self._cur_position = (0, 0)
    self._response = []
    return self.observation

  def step(self, action):
    """Single step of the environment when policy took `action`."""
    cur_sequence_id, cur_token_id = self._cur_position
    if cur_sequence_id >= len(self._cur_sequence):
      return np.array(self._eos, dtype=np.int32), 0.0, True, None

    # Emit the control mask on the output.
    control_mask = int(not self._on_input)

    if self._on_input:
      self._response = []
      if cur_token_id + 1 < len(self._cur_sequence[cur_sequence_id]):
        self._cur_position = (cur_sequence_id, cur_token_id + 1)
        done = False
      else:
        self._cur_position = (cur_sequence_id + 1, 0)
        done = cur_sequence_id + 1 >= len(self._cur_sequence)
      reward = 0.0
      discount_mask = 0

    else:
      self._response.append(action)
      if action == self._eos or len(self._response) > self._max_length:
        self._cur_position = (cur_sequence_id + 1, 0)
        reward = self._metric(
            self._response[:-1], self._cur_sequence[cur_sequence_id])
        done = cur_sequence_id + 1 >= len(self._cur_sequence)
        # Emit the discount mask on the last token of each action.
        discount_mask = 1
      else:
        reward = 0.0
        done = False
        discount_mask = 0

    info = {'control_mask': control_mask, 'discount_mask': discount_mask}
    return self.observation, reward, done, info


def copy_stream(length, low=2, high=15, n=1):
  """Generate `n` random sequences of length `length` and yield with copies."""
  while True:
    res = []
    for _ in range(n):
      seq = np.random.randint(low, high, size=(length,), dtype=np.int32)
      res.extend([seq, seq])
    yield res


def _accuracy(seq1, seq2):
  """Token-level accuracy."""
  seq1, seq2 = np.array(seq1), np.array(seq2)
  max_length = max(seq1.shape[-1], seq2.shape[-1])
  min_length = min(seq1.shape[-1], seq2.shape[-1])
  seq1s, seq2s = seq1[..., :min_length], seq2[..., :min_length]
  return np.sum(np.equal(seq1s, seq2s)) / max_length

