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

"""EnvProblem for environments simulated by a Trax model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

import jax
import numpy as np

from tensor2tensor.envs import env_problem
from trax import math
from trax import utils
from trax.math import random as jax_random
from trax.rl import serialization_utils
from trax.rl import space_serializer
from trax.shapes import ShapeDtype
from trax.supervised import trainer_lib


class SimulatedEnvProblem(env_problem.EnvProblem):
  """EnvProblem base class for environments simulated by Trax models.

  The initial observations to start the model are taken from
  initial_observation_stream. This iterator in incremented in every reset().

  A checkpoint saved by the Trax trainer should be available in output_dir.
  """

  def __init__(self, model, batch_size, observation_space, action_space,
               reward_range, discrete_rewards, history_stream, output_dir,
               model_predict_kwargs=None):
    """Initializes the env.

    Args:
      model: Trax model.
      batch_size: (int) Number of simulated environments run in parallel.
      observation_space: (gym.Space) Observation space.
      action_space: (gym.Space) Action space.
      reward_range: (tuple) Pair (min_reward, max_reward).
      discrete_rewards: (bool) Whether to discretize the rewards.
      history_stream: Iterator yielding batches of initial input data for the
        model. The format is implementation-specific.
      output_dir: (str) Output dir.
      model_predict_kwargs: (dict) Additional model keyword arguments for
        inference. Useful when different config is needed for training and
        inference, e.g. train with memory efficient attention and predict with
        the regular one.
    """
    self._model = model
    if model_predict_kwargs is None:
      model_predict_kwargs = {}
    model_predict = self._model(mode='predict', **model_predict_kwargs)
    # NOTE: can set non-default PRNG key: model_predict._set_rng_recursive(...)
    def predict_with_state(*args, **kwargs):
      output = model_predict(*args, **kwargs)
      return (output, model_predict.state)
    self._model_predict = math.jit(predict_with_state)
    self._model_initialize = model_predict.init
    self._init_model_weights = None
    self._init_model_state = None

    self._observation_space = observation_space
    self._action_space = action_space
    self._reward_range = reward_range
    self._output_dir = output_dir

    self._predict_fn = None
    self._rng = None
    self._model_state = None
    self._history_stream = None

    # Call the super's ctor. It will use some of the member fields, so we call
    # it in the end.
    super(SimulatedEnvProblem, self).__init__(
        batch_size=batch_size,
        discrete_rewards=discrete_rewards,
        history_stream=history_stream,
    )

    self.seed()

  def initialize_environments(self,
                              history_stream,
                              batch_size=1,
                              parallelism=1):
    """Initializes the environments.

    Args:
      history_stream: Iterator yielding batches of initial input data for the
        model. The format is implementation-specific.
      batch_size: (int) Number of environments in a batch.
      parallelism: (int) Unused.
    """
    del parallelism

    if self._output_dir is None:
      model_weights = self._init_model_weights
      self._model_state = None
    else:
      trax_state = trainer_lib.load_trainer_state(self._output_dir)
      # TODO(lukaszkaiser): both model state and parameters by default include
      # the loss layer. Currently, we access the pure-model parameters by just
      # indexing, [0] here. But we should make it more explicit in a better API.
      model_weights = self._extract_weights(trax_state.opt_state.weights[0])
      self._model_state = trax_state.model_state[0]

    def predict_fn(inputs, rng):
      (output, self._model_state) = self._model_predict(
          inputs, weights=model_weights, state=self._model_state, rng=rng
      )
      return output

    self._predict_fn = predict_fn
    self._history_stream = history_stream
    self._steps = np.zeros(batch_size, dtype=np.int32)

  def _extract_weights(self, weights):
    return weights

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def action_space(self):
    return self._action_space

  @property
  def reward_range(self):
    return self._reward_range

  def seed(self, seed=None):
    if seed is None:
      seed = random.randint(0, 2**31 - 1)
    self._rng = jax_random.get_prng(seed)
    return super(SimulatedEnvProblem, self).seed(seed=seed)

  def _reset_model(self, predict_fn, indices, history, rng):
    """Resets the environments at the given indices.

    Should be implemented in subclasses.

    Args:
      predict_fn: Function running prediction with the model.
      indices: List of indices of underlying envs to call reset on.
      history: Initial input data for the model.
      rng: Jax RNG.

    Returns:
      np.ndarray of batched observations from the reset envs.
    """
    raise NotImplementedError

  def _step_model(self, predict_fn, actions, rng):
    """Takes a step in all environments.

    Should be implemented in subclasses.

    Args:
      predict_fn: Function running prediction with the model.
      actions: (np.ndarray) with first dimension equal to the batch size.
      rng: Jax RNG.

    Returns:
      a tuple of batched raw observations, rewards and dones.
    """
    raise NotImplementedError

  def trajectory_to_training_examples(self, trajectory):
    raise NotImplementedError

  def _reset(self, indices):
    """Resets environments at the given indices.

    Args:
      indices: list of indices of underlying envs to call reset on.

    Returns:
      np.ndarray of batched observations from the reset envs.
    """
    history = next(self._history_stream)
    (subrng, self._rng) = jax_random.split(self._rng)
    return self._reset_model(self._predict_fn, indices, history, subrng)

  def _step(self, actions):
    """Takes a step in all environments.

    Args:
      actions: (np.ndarray) with first dimension equal to the batch size.

    Returns:
      a tuple of batched raw observations, raw rewards, dones and infos.
    """
    # Predict the next observation.
    (subrng, self._rng) = jax_random.split(self._rng)
    (observation, reward, done) = self._step_model(
        self._predict_fn, actions, subrng)
    return (observation, reward, done, {})

  @property
  def model(self):
    return lambda mode: serialization_utils.SerializedModel(  # pylint: disable=g-long-lambda
        seq_model=self._model(mode=mode),
        observation_serializer=self._obs_serializer,
        action_serializer=self._action_serializer,
        significance_decay=self._significance_decay,
    )


class RawSimulatedEnvProblem(SimulatedEnvProblem):
  """SimulatedEnvProblem running a model operating on raw tensors.

  Wraps an autoregressive trax model of signature
  (observation_history, action) -> (observation, reward) in an EnvProblem.
  The model is assumed to take a fixed number of last observations as input
  and produce a single observation, which is fed back into the model in the
  next environment step.

  Shape requirements (without the batch dimension):
    observation: Consistent with observation_space.
    observation_history: (history_length,) + observation.shape.
    action: Consistent with action_space.
    reward: (1,). The singleton dimension is removed in step().
  """

  def __init__(self, history_length, trajectory_length, *args, **kwargs):
    """Initializes the env.

    Args:
      history_length: (int) Number of last observations fed into the model.
      trajectory_length: (int) Length of each trajectory unrolled from the
        model.
      *args: (tuple) Positional arguments passed to the base class.
      **kwargs: (dict) Keyword arguments passed to the base class.
    """
    self._history_length = history_length
    self._trajectory_length = trajectory_length
    self._history = None
    self._steps = None

    super(RawSimulatedEnvProblem, self).__init__(*args, **kwargs)

  def initialize_environments(self, batch_size=1, **kwargs):
    """Initializes the environments."""
    self._history = None
    self._steps = np.zeros(batch_size)
    return super(RawSimulatedEnvProblem, self).initialize_environments(
        batch_size=batch_size, **kwargs)

  def _reset_model(self, predict_fn, indices, history, rng):
    del predict_fn
    del rng
    assert history.shape == ((self._batch_size, self._history_length) +
                             self.observation_space.shape)

    if self._history is None:
      # At the first reset, all indices should be triggered.
      assert set(indices) == set(range(self._batch_size))
      self._history = np.array(history)
    else:
      history = history[indices, ...]
      self._history[indices, ...] = history

    # Reset the step counters.
    self._steps[indices] = 0

    # Return just the last timestep at the given indices.
    return history[:, -1, ...]

  def _step_model(self, predict_fn, actions, rng):
    (observation, reward) = predict_fn((self._history, actions), rng=rng)

    # Roll the history one timestep back and append the new observation.
    self._history = np.roll(self._history, shift=-1, axis=1)
    self._history[:, -1, ...] = observation

    # Increment the step counters and determine which envs are done.
    self._steps += 1
    done = self._steps == self._trajectory_length

    # Call copy() to get the data as numpy arrays.
    observation = observation.copy()
    # Reshape the rewards to get rid of the extra dimension.
    reward = np.squeeze(reward.copy(), axis=1)
    return (observation, reward, done)


class SerializedSequenceSimulatedEnvProblem(SimulatedEnvProblem):
  """SimulatedEnvProblem running a model operating on sequences of symbols.

  Wraps an autoregressive trax model of signature past_symbols -> symbol_probs
  in an EnvProblem. The model is assumed to take a sequence of symbols as input
  and produce distributions over all symbols in the sequence. The next symbol
  is sampled and fed back to the model in the next decoding step.

  Shape requirements (without the batch dimension):
    past_symbols: (max_trajectory_length * L,)
    symbol_probs: (max_trajectory_length * L, vocab_size)
  where L is the representation length of one environment step.

  Observations, actions, rewards and done flags are (de)serialized from/to
  sequences of symbols using an EnvSerializer passed to the constructor.
  """

  def __init__(self, model, reward_fn, done_fn, vocab_size,
               max_trajectory_length, observation_space, action_space,
               significance_decay=1.0, **kwargs):
    """Initializes the env.

    Args:
      model: trax model to use for simulation. It's assumed to take keyword
        arguments vocab_size and mode, where vocab_size is the number of symbols
        in the vocabulary and mode is either 'train' or 'eval'.

      reward_fn: Function (previous_observation, current_observation) -> reward.
      done_fn: Function (previous_observation, current_observation) -> done.
      vocab_size: (int) Number of symbols in the vocabulary.
      max_trajectory_length: (int) Maximum length of a trajectory unrolled from
        the model.
      observation_space: (gym.Space) Observation space.
      action_space: (gym.Space) Action space.
      significance_decay: (float) Decay for training weights of progressively
        less significant symbols in the representation.
      **kwargs: (dict) Keyword arguments passed to the base class.
    """
    self._reward_fn = reward_fn
    self._done_fn = done_fn
    self._vocab_size = vocab_size
    self._max_trajectory_length = max_trajectory_length
    self._significance_decay = significance_decay
    self._steps = None
    self._observation_space = None
    self._action_space = None
    self._last_observations = None

    self._obs_serializer = space_serializer.create(
        observation_space, self._vocab_size)
    self._action_serializer = space_serializer.create(
        action_space, self._vocab_size)
    self._obs_repr_length = self._obs_serializer.representation_length
    self._act_repr_length = self._action_serializer.representation_length
    self._step_repr_length = self._obs_repr_length + self._act_repr_length

    # We assume that the model takes vocab_size as an argument (e.g.
    # TransformerLM).
    model = functools.partial(model, vocab_size=vocab_size)
    super(SerializedSequenceSimulatedEnvProblem, self).__init__(
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        **kwargs
    )

  def initialize_environments(self, batch_size=1, **kwargs):
    """Initializes the environments."""
    self._steps = np.zeros(batch_size, dtype=np.int32)
    self._last_observations = np.full(
        (batch_size,) + self._observation_space.shape, np.nan)
    self._last_symbols = np.zeros((batch_size, 1), dtype=np.int32)
    input_signature = ShapeDtype((batch_size, 1), np.int32)
    (self._init_model_weights, self._init_model_state) = self._model_initialize(
        input_signature
    )
    super(SerializedSequenceSimulatedEnvProblem, self).initialize_environments(
        batch_size=batch_size, **kwargs)
    self._model_state = self._init_model_state

  def _extract_weights(self, weights):
    return serialization_utils.extract_inner_model(weights)

  def _predict_obs(self, predict_fn, rng):
    obs_repr = np.zeros(
        (self._steps.shape[0], self._obs_repr_length), dtype=np.int32,
    )
    for (i, subrng) in enumerate(jax_random.split(rng, self._obs_repr_length)):
      log_probs = predict_fn(self._last_symbols, rng=subrng)
      self._last_symbols = utils.gumbel_sample(log_probs)
      obs_repr[:, i] = self._last_symbols[:, 0]
    return np.array(self._obs_serializer.deserialize(obs_repr))

  def _consume_act(self, actions, predict_fn, rng):
    act_repr = self._action_serializer.serialize(actions)
    for (i, subrng) in enumerate(jax_random.split(rng, self._act_repr_length)):
      # Run the network to update the inference buffers, but ignore the result.
      predict_fn(self._last_symbols, rng=subrng)
      self._last_symbols = act_repr[:, i:(i + 1)]

  def _reset_model(self, predict_fn, indices, history, rng):
    # TODO(pkozakowski): Random starts.
    del history

    indices = np.array(indices)

    # During reset, we need to predict the first observation for a subset of
    # indices, however inference only works for the full set of indices. To
    # workaround that:
    # 1. Save prior inference state.
    old_model_state = self._model_state
    old_last_symbols = self._last_symbols
    # 2. Reset the entire inference state.
    self._model_state = self._init_model_state
    self._last_symbols[:] = 0
    # 3. Predict the next observation.
    observation = self._predict_obs(predict_fn, rng)[indices]
    self._last_observations[indices] = observation

    # TODO(pkozakowski): Abstract out this primitive e.g. as
    # trax.math.nested_zip_with?
    def reset_recursively(current_state, init_state):
      """Resets the initial state, assuming it's batched by trajectories."""
      if isinstance(current_state, (list, tuple)):
        return [
            reset_recursively(current, init)
            for (current, init) in zip(current_state, init_state)
        ]
      elif isinstance(current_state, dict):
        return {
            key: reset_recursively(current_state[key], init_state[key])
            for key in current_state
        }
      else:
        # current_state might just be a scalar primitive, check.
        if (getattr(current_state, 'shape', ()) and
            current_state.shape[0] == self._batch_size):
          # If the state component is batched, substitute it on appropriate
          # indices.
          # This doesn't work with more than one head in attention layers,
          # because the batch dimension gets multiplied by the number of heads.
          # TODO(pkozakowski): Fix that in trax.layes.attention.
          return jax.ops.index_update(
              current_state, jax.ops.index[indices], init_state[indices]
          )
        else:
          # Otherwise, leave as it is.
          return current_state

    # 4. Assign back the old inference state, updated on the appropriate
    # indices.
    self._model_state = reset_recursively(old_model_state, self._model_state)
    old_last_symbols[indices] = self._last_symbols[indices]
    self._last_symbols = old_last_symbols
    self._steps[indices] = 0
    return observation

  def _step_model(self, predict_fn, actions, rng):
    self._consume_act(actions, predict_fn, rng)
    self._steps += 1
    observation = self._predict_obs(predict_fn, rng)
    reward = self._reward_fn(self._last_observations, observation)
    done = self._done_fn(self._last_observations, observation)
    # Copy the last observations, so that we don't overwrite data stored in a
    # trajectory when resetting the environment (see _reset_model).
    self._last_observations = np.copy(observation)
    done = np.logical_or(done, self._steps == self._max_trajectory_length - 1)
    return (observation, reward, done)

  def trajectory_to_training_examples(self, trajectory):
    padding_length = self._max_trajectory_length - trajectory.num_time_steps
    def pad(x):
      pad_width = [(0, padding_length)] + [(0, 0)] * (x.ndim - 1)
      return np.pad(x, pad_width=pad_width, mode='constant')
    obs = pad(trajectory.observations_np)
    act = pad(trajectory.actions_np)
    mask = np.zeros_like(obs)
    mask[:trajectory.num_time_steps, ...] = 1
    return [(obs, act, obs, mask)]


def cartpole_done_fn(previous_observation, current_observation):
  del previous_observation
  x_threshold = 2.4
  theta_threshold = 12 * 2 * np.pi / 360
  x = current_observation[:, 0]
  theta = current_observation[:, 2]
  return np.logical_or(np.abs(x) > x_threshold, np.abs(theta) > theta_threshold)


def cartpole_reward_fn(previous_observation, current_observation):
  done = cartpole_done_fn(previous_observation, current_observation)
  return 1.0 - done  # Unit reward for every timestep until the end.


def acrobot_done_fn(previous_observation, current_observation):
  del previous_observation
  theta1 = current_observation[:, 0]
  theta2 = current_observation[:, 1]
  return -np.cos(theta1) - np.cos(theta2 + theta1) > 1.0


def acrobot_reward_fn(previous_observation, current_observation):
  done = acrobot_done_fn(previous_observation, current_observation)
  return -1.0 + done  # -1 reward for every timestep until the end.


def onlinetune_done_fn(previous_observation, current_observation):
  del previous_observation
  del current_observation
  # Never return "done" from the environment, rely on max_trajectory_length
  # instead.
  return False


def onlinetune_reward_fn(
    previous_observation,
    current_observation,
    # 2 is the evaluation accuracy metric in the default settings of
    # OnlineTuneEnv.
    dim_index=2,
):
  prev = previous_observation[:, dim_index]
  cur = current_observation[:, dim_index]
  return cur - prev
