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

"""Tests for trax.rl.serialization_utils."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import gin
import gym
from jax import numpy as jnp
import numpy as np
from trax import models as trax_models
from trax import shapes
from trax import test_utils
from trax.data import inputs as trax_input
from trax.layers import base as layers_base
from trax.models import transformer
from trax.rl import serialization_utils
from trax.rl import space_serializer
from trax.supervised import trainer_lib


# pylint: disable=invalid-name
def TestModel(extra_dim, mode='train'):
  """Dummy sequence model for testing."""
  del mode
  def f(inputs):
    # Cast the input to float32 - this is for simulating discrete-input models.
    inputs = inputs.astype(np.float32)
    # Add an extra dimension if requested, e.g. the logit dimension for output
    # symbols.
    if extra_dim is not None:
      return jnp.broadcast_to(inputs[:, :, None], inputs.shape + (extra_dim,))
    else:
      return inputs
  return layers_base.Fn('TestModel', f)
  # pylint: enable=invalid-name


def signal_inputs(seq_len, batch_size, depth=1):
  def stream_fn(num_devices):
    del num_devices
    while True:
      x = np.random.uniform(size=(batch_size, seq_len, depth))
      y = np.random.uniform(size=(batch_size, seq_len, depth))
      mask = np.ones_like(x).astype(np.float32)
      yield (x, y, x, mask)

  return trax_input.Inputs(
      train_stream=stream_fn,
      eval_stream=stream_fn,
  )


class SerializationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=2
    )
    self._repr_length = 100
    self._serialization_utils_kwargs = {
        'observation_serializer': self._serializer,
        'action_serializer': self._serializer,
        'representation_length': self._repr_length,
    }
    test_utils.ensure_flag('test_tmpdir')

  def test_serialized_model_discrete(self):
    vocab_size = 3
    obs = np.array([[[0, 1], [1, 1], [1, 0], [0, 0]]])
    act = np.array([[1, 0, 0]])
    mask = np.array([[1, 1, 1, 0]])

    test_model_inputs = []

    # pylint: disable=invalid-name
    def TestModelSavingInputs(mode):
      del mode
      def f(inputs):
        # Save the inputs for a later check.
        test_model_inputs.append(inputs)
        # Change type to np.float32 and add the logit dimension.
        return jnp.broadcast_to(
            inputs.astype(np.float32)[:, :, None], inputs.shape + (vocab_size,)
        )
      return layers_base.Fn('TestModelSavingInputs', f)
      # pylint: enable=invalid-name

    obs_serializer = space_serializer.create(
        gym.spaces.MultiDiscrete([2, 2]), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        TestModelSavingInputs,  # pylint: disable=no-value-for-parameter
        observation_serializer=obs_serializer,
        action_serializer=act_serializer,
        significance_decay=0.9,
    )

    example = (obs, act, obs, mask)
    serialized_model.init(shapes.signature(example))

    (obs_logits, obs_repr, weights) = serialized_model(example)
    # Check that the model has been called with the correct input.
    np.testing.assert_array_equal(
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
    np.testing.assert_array_equal(np.min(obs_logits, axis=-1), obs_repr)
    np.testing.assert_array_equal(np.max(obs_logits, axis=-1), obs_repr)
    # Check that the observations are correct.
    np.testing.assert_array_equal(obs_repr, obs)
    # Check weights.
    np.testing.assert_array_equal(
        weights,
        [[[1., 1.], [1., 1.], [1., 1.], [0., 0.]]],
    )

  def test_train_model_with_serialization(self):
    # Serializer handles discretization of the data.
    precision = 2
    number_of_time_series = 2
    vocab_size = 16
    srl = space_serializer.BoxSpaceSerializer(
        space=gym.spaces.Box(shape=(number_of_time_series,),
                             low=0.0, high=16.0),
        vocab_size=vocab_size,
        precision=precision,
    )

    def model(mode):
      del mode
      return serialization_utils.SerializedModel(
          functools.partial(
              trax_models.TransformerLM,
              vocab_size=vocab_size,
              d_model=16,
              d_ff=8,
              n_layers=1,
              n_heads=1,
          ),
          observation_serializer=srl,
          action_serializer=srl,
          significance_decay=0.9,
      )

    output_dir = self.create_tempdir().full_path
    state = trainer_lib.train(
        output_dir=output_dir,
        model=model,
        inputs=functools.partial(signal_inputs, seq_len=5,
                                 batch_size=64, depth=number_of_time_series),
        steps=2)
    self.assertEqual(2, state.step)

  def test_serialized_model_continuous(self):
    precision = 3
    gin.bind_parameter('BoxSpaceSerializer.precision', precision)

    vocab_size = 32
    obs = np.array([[[1.5, 2], [-0.3, 1.23], [0.84, 0.07], [0, 0]]])
    act = np.array([[0, 1, 0]])
    mask = np.array([[1, 1, 1, 0]])

    obs_serializer = space_serializer.create(
        gym.spaces.Box(shape=(2,), low=-2, high=2), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        functools.partial(TestModel, extra_dim=vocab_size),
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

  def test_serialized_model_extracts_seq_model_weights_and_state(self):
    vocab_size = 3

    seq_model_fn = functools.partial(
        transformer.TransformerLM,
        vocab_size=vocab_size,
        d_model=2,
        d_ff=2,
        n_layers=0,
    )
    seq_model = seq_model_fn(mode='eval')
    obs_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=vocab_size
    )
    serialized_model = serialization_utils.SerializedModel(
        seq_model_fn,
        observation_serializer=obs_serializer,
        action_serializer=act_serializer,
        significance_decay=0.9,
    )

    obs_sig = shapes.ShapeDtype((1, 2))
    act_sig = shapes.ShapeDtype((1, 1))
    serialized_model.init(input_signature=(obs_sig, act_sig, obs_sig, obs_sig))
    seq_model.weights = serialized_model.seq_model_weights
    seq_model.state = serialized_model.seq_model_state
    # Run the model to check if the weights and state have correct structure.
    seq_model(jnp.array([[0]]))

  @parameterized.named_parameters(('raw', None), ('serialized', 32))
  def test_wrapped_policy_continuous(self, vocab_size):
    precision = 3
    n_controls = 2
    n_actions = 4
    gin.bind_parameter('BoxSpaceSerializer.precision', precision)

    obs = np.array([[[1.5, 2], [-0.3, 1.23], [0.84, 0.07], [0.01, 0.66]]])
    act = np.array([[[0, 1], [2, 0], [1, 3]]])

    wrapped_policy = serialization_utils.wrap_policy(
        TestModel(extra_dim=vocab_size),  # pylint: disable=no-value-for-parameter
        observation_space=gym.spaces.Box(shape=(2,), low=-2, high=2),
        action_space=gym.spaces.MultiDiscrete([n_actions] * n_controls),
        vocab_size=vocab_size,
    )

    example = (obs, act)
    wrapped_policy.init(shapes.signature(example))
    (act_logits, values) = wrapped_policy(example)
    self.assertEqual(act_logits.shape, obs.shape[:2] + (n_controls, n_actions))
    self.assertEqual(values.shape, obs.shape[:2])

  def test_analyzes_discrete_action_space(self):
    space = gym.spaces.Discrete(n=5)
    (n_controls, n_actions) = serialization_utils.analyze_action_space(space)
    self.assertEqual(n_controls, 1)
    self.assertEqual(n_actions, 5)

  def test_analyzes_multi_discrete_action_space_with_equal_categories(self):
    space = gym.spaces.MultiDiscrete(nvec=(3, 3))
    (n_controls, n_actions) = serialization_utils.analyze_action_space(space)
    self.assertEqual(n_controls, 2)
    self.assertEqual(n_actions, 3)

  def test_doesnt_analyze_multi_disccrete_action_space_with_inequal_categories(
      self
  ):
    space = gym.spaces.MultiDiscrete(nvec=(2, 3))
    with self.assertRaises(AssertionError):
      serialization_utils.analyze_action_space(space)

  def test_doesnt_analyze_box_action_space(self):
    space = gym.spaces.Box(shape=(2, 3), low=0, high=1)
    with self.assertRaises(AssertionError):
      serialization_utils.analyze_action_space(space)


if __name__ == '__main__':
  absltest.main()
