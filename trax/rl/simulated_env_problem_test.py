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

"""Tests for trax.rl.simulated_env_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

import gym
import mock
import numpy as np

from tensor2tensor.envs import trajectory
from tensorflow import test
from trax import math
from trax.layers import base
from trax.models import transformer
from trax.rl import simulated_env_problem
from trax.supervised import trainer_lib


class RawSimulatedEnvProblemTest(test.TestCase):

  @staticmethod
  @mock.patch.object(trainer_lib, 'load_trainer_state', autospec=True)
  def _create_env(mock_restore_state, model, histories,
                  trajectory_length):
    # (model_params, opt_state)
    mock_restore_state.return_value.params = (None, None)
    space = gym.spaces.Discrete(100)
    return simulated_env_problem.RawSimulatedEnvProblem(
        model=model,
        history_length=histories.shape[2],
        trajectory_length=trajectory_length,
        batch_size=1,
        observation_space=space,
        action_space=space,
        reward_range=(-1, 1),
        discrete_rewards=True,
        history_stream=iter(histories),
        output_dir=None,
    )

  def test_takes_new_history(self):
    histories = np.array([[[0, 1, 2]], [[3, 4, 5]]])

    mock_model_fn = mock.MagicMock()
    mock_model_fn.return_value.init.return_value = (None, None)

    with math.use_backend('numpy'):
      env = self._create_env(  # pylint: disable=no-value-for-parameter
          model=mock_model_fn,
          histories=histories,
          trajectory_length=2,
      )
      env.reset()
      observation = env.reset()
      np.testing.assert_array_equal(observation, [5])


class SerializedSequenceSimulatedEnvProblemTest(test.TestCase):

  def _make_env(
      self, observation_space, action_space, vocab_size,
      predict_fn=None, reward_fn=None, done_fn=None,
      batch_size=None, max_trajectory_length=None,
  ):
    mock_model_fn = mock.MagicMock()
    mock_model_fn.return_value.init.return_value = (None, None)
    if predict_fn is not None:
      mock_model_fn.return_value = predict_fn
      mock_model_fn.return_value.init.return_value = (
          base.EMPTY_WEIGHTS, base.EMPTY_STATE)
    return simulated_env_problem.SerializedSequenceSimulatedEnvProblem(
        model=mock_model_fn,
        reward_fn=reward_fn,
        done_fn=done_fn,
        vocab_size=vocab_size,
        max_trajectory_length=max_trajectory_length,
        batch_size=batch_size,
        observation_space=observation_space,
        action_space=action_space,
        reward_range=(-1, 1),
        discrete_rewards=False,
        history_stream=itertools.repeat(None),
        output_dir=None,
    )

  def _make_trajectory(self, observations, actions):
    assert len(observations) == len(actions) + 1
    t = trajectory.Trajectory()
    for (obs, act) in zip(observations, actions):
      t.add_time_step(observation=obs, action=act, done=False)
    t.add_time_step(observation=observations[-1], done=True)
    return t

  def test_runs_with_transformer(self):
    env = simulated_env_problem.SerializedSequenceSimulatedEnvProblem(
        model=functools.partial(
            transformer.TransformerLM, d_model=2, d_ff=2, n_heads=1, n_layers=1
        ),
        reward_fn=(lambda _1, _2: np.array([0.5])),
        done_fn=(lambda _1, _2: np.array([False])),
        vocab_size=4,
        max_trajectory_length=3,
        batch_size=1,
        observation_space=gym.spaces.Box(low=0, high=5, shape=(4,)),
        action_space=gym.spaces.Discrete(n=2),
        reward_range=(-1, 1),
        discrete_rewards=False,
        history_stream=itertools.repeat(None),
        output_dir=None,
    )

    env.reset()
    for expected_done in [False, True]:
      (_, _, dones, _) = env.step(np.array([0]))
      np.testing.assert_array_equal(dones, [expected_done])

  def test_makes_training_example(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=3,
    )
    t = self._make_trajectory(observations=[0, 1, 0], actions=[1, 0])
    examples = env.trajectory_to_training_examples(t)

    # There should be 1 example with the whole trajectory.
    self.assertEqual(len(examples), 1)
    [(input_obs, input_act, target_obs, weights)] = examples
    np.testing.assert_array_equal(input_obs, [0, 1, 0])
    np.testing.assert_array_equal(input_act, [1, 0])
    # inputs == targets for autoregressive sequence prediction.
    np.testing.assert_array_equal(input_obs, target_obs)
    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0])

  def test_makes_training_example_padded(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=4,
    )
    t = self._make_trajectory(observations=[0, 1, 0], actions=[1, 0])
    examples = env.trajectory_to_training_examples(t)
    [(input_obs, input_act, target_obs, weights)] = examples
    # Should pad by 1 on the right.
    np.testing.assert_array_equal(input_obs, [0, 1, 0, 0])
    np.testing.assert_array_equal(input_act, [1, 0, 0])
    # inputs == targets for autoregressive sequence prediction.
    np.testing.assert_array_equal(input_obs, target_obs)
    # The last timestep should be masked out.
    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0, 0.0])


if __name__ == '__main__':
  test.main()
