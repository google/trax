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

"""Tests for trax.rl.serialization_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import gym
from jax import numpy as np
import numpy as onp
from tensorflow import test

from trax import shapes
from trax.layers import base as layers_base
from trax.models import transformer
from trax.rl import serialization_utils
from trax.rl import space_serializer


class SerializationTest(test.TestCase):

  def setUp(self):
    super(SerializationTest, self).setUp()
    self._serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=2
    )
    self._repr_length = 100
    self._serialization_utils_kwargs = {
        'observation_serializer': self._serializer,
        'action_serializer': self._serializer,
        'representation_length': self._repr_length,
    }

  def test_serialized_model_discrete(self):
    vocab_size = 3
    obs = onp.array([[[0, 1], [1, 1], [1, 0], [0, 0]]])
    act = onp.array([[1, 0, 0]])
    mask = onp.array([[1, 1, 1, 0]])

    test_model_inputs = []

    @layers_base.layer()
    def TestModel(inputs, **unused_kwargs):
      # Save the inputs for a later check.
      test_model_inputs.append(inputs)
      # Change type to onp.float32 and add the logit dimension.
      return np.broadcast_to(
          inputs.astype(onp.float32)[:, :, None], inputs.shape + (vocab_size,)
      )

    obs_serializer = space_serializer.create(
        gym.spaces.MultiDiscrete([2, 2]), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        TestModel(),  # pylint: disable=no-value-for-parameter
        observation_serializer=obs_serializer,
        action_serializer=act_serializer,
        significance_decay=0.9,
    )

    example = (obs, act, obs, mask)
    serialized_model.init(shapes.signature(example))
    (obs_logits, obs_repr, weights) = serialized_model(example)
    # Check that the model has been called with the correct input.
    onp.testing.assert_array_equal(
        # The model is called multiple times for determining shapes etc.
        # Check the last saved input - that should be the actual concrete array
        # calculated during the forward pass.
        test_model_inputs[-1],
        # Should be serialized observations and actions interleaved.
        [[0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]],
    )
    # Check the output shape.
    self.assertEqual(obs_logits.shape, obs_repr.shape + (vocab_size,))
    # Check that obs_logits are the same as obs_repr, just broadcasted over the
    # logit dimension.
    onp.testing.assert_array_equal(onp.min(obs_logits, axis=-1), obs_repr)
    onp.testing.assert_array_equal(onp.max(obs_logits, axis=-1), obs_repr)
    # Check that the observations are correct.
    onp.testing.assert_array_equal(obs_repr, obs)
    # Check weights.
    onp.testing.assert_array_equal(weights, [[[1, 1], [1, 1], [1, 1], [0, 0]]])

  def test_serialized_model_continuous(self):
    precision = 3
    gin.bind_parameter('BoxSpaceSerializer.precision', precision)

    vocab_size = 32
    obs = onp.array([[[1.5, 2], [-0.3, 1.23], [0.84, 0.07], [0, 0]]])
    act = onp.array([[0, 1, 0]])
    mask = onp.array([[1, 1, 1, 0]])

    @layers_base.layer()
    def TestModel(inputs, **unused_kwargs):
      # Change type to onp.float32 and add the logit dimension.
      return np.broadcast_to(
          inputs.astype(onp.float32)[:, :, None], inputs.shape + (vocab_size,)
      )

    obs_serializer = space_serializer.create(
        gym.spaces.Box(shape=(2,), low=-2, high=2), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        TestModel(),  # pylint: disable=no-value-for-parameter
        observation_serializer=obs_serializer,
        action_serializer=act_serializer,
        significance_decay=0.9,
    )

    example = (obs, act, obs, mask)
    serialized_model.init(shapes.signature(example))
    (obs_logits, obs_repr, weights) = serialized_model(example)
    self.assertEqual(obs_logits.shape, obs_repr.shape + (vocab_size,))
    self.assertEqual(
        obs_repr.shape, (1, obs.shape[1], obs.shape[2] * precision)
    )
    self.assertEqual(obs_repr.shape, weights.shape)

  def test_extract_inner_model(self):
    vocab_size = 3

    inner_model = transformer.TransformerLM(
        vocab_size=vocab_size, d_model=2, d_ff=2, n_layers=0
    )
    obs_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        inner_model,
        observation_serializer=obs_serializer,
        action_serializer=act_serializer,
        significance_decay=0.9,
    )

    obs_sig = shapes.ShapeDtype((1, 2))
    act_sig = shapes.ShapeDtype((1, 1))
    (weights, state) = serialized_model.init(
        input_signature=(obs_sig, act_sig, obs_sig, obs_sig),
    )
    (inner_weights, inner_state) = map(
        serialization_utils.extract_inner_model, (weights, state)
    )
    inner_model(np.array([[0]]), weights=inner_weights, state=inner_state)

  def test_serializes_observations_and_actions(self):
    reprs = serialization_utils.serialize_observations_and_actions(
        observations=onp.array([[0, 1]]),
        actions=onp.array([[0]]),
        **self._serialization_utils_kwargs
    )
    self.assertEqual(reprs.shape, (1, self._repr_length))

  def test_observation_and_action_masks_are_valid_and_complementary(self):
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    self.assertEqual(obs_mask.shape, (self._repr_length,))
    self.assertEqual(onp.min(obs_mask), 0)
    self.assertEqual(onp.max(obs_mask), 1)

    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )
    self.assertEqual(act_mask.shape, (self._repr_length,))
    self.assertEqual(onp.min(act_mask), 0)
    self.assertEqual(onp.max(act_mask), 1)

    onp.testing.assert_array_equal(
        obs_mask + act_mask, onp.ones(self._repr_length)
    )

  def test_masks_observations(self):
    reprs = serialization_utils.serialize_observations_and_actions(
        # Observations are different, actions are the same.
        observations=onp.array([[0, 1], [1, 1]]),
        actions=onp.array([[0], [0]]),
        **self._serialization_utils_kwargs
    )
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )

    self.assertFalse(onp.array_equal(reprs[0] * obs_mask, reprs[1] * obs_mask))
    onp.testing.assert_array_equal(reprs[0] * act_mask, reprs[1] * act_mask)

  def test_masks_actions(self):
    reprs = serialization_utils.serialize_observations_and_actions(
        # Observations are the same, actions are different.
        observations=onp.array([[0, 1], [0, 1]]),
        actions=onp.array([[0], [1]]),
        **self._serialization_utils_kwargs
    )
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )

    onp.testing.assert_array_equal(reprs[0] * obs_mask, reprs[1] * obs_mask)
    self.assertFalse(onp.array_equal(reprs[0] * act_mask, reprs[1] * act_mask))

  def test_rewards_to_actions_map(self):
    rewards = onp.array([1, 2, 3])
    r2a_map = serialization_utils.rewards_to_actions_map(
        observation_serializer=space_serializer.create(
            gym.spaces.MultiDiscrete(nvec=[2, 2, 2]), vocab_size=2
        ),
        action_serializer=space_serializer.create(
            gym.spaces.MultiDiscrete(nvec=[2, 2]), vocab_size=2
        ),
        n_timesteps=len(rewards),
        representation_length=16,
    )
    broadcast_rewards = onp.dot(rewards, r2a_map)
    onp.testing.assert_array_equal(
        broadcast_rewards,
        # obs1, act1, obs2, act2, obs3 cut after 1st symbol.
        [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 0],
    )


if __name__ == '__main__':
  test.main()
