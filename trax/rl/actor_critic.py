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
"""Classes for RL training in Trax."""

import functools
import os

import gin
import gym
import numpy as np
import tensorflow as tf

from trax import data
from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax import supervised
from trax.fastmath import numpy as jnp
from trax.optimizers import adam
from trax.rl import advantages as rl_advantages
from trax.rl import distributions
from trax.rl import policy_tasks
from trax.rl import training as rl_training
from trax.rl import value_tasks
from trax.supervised import lr_schedules as lr


class ActorCriticAgent(rl_training.PolicyAgent):
  """Trains policy and value models using actor-critic methods.

  Attrs:
    on_policy (bool): Whether the algorithm is on-policy. Used in the data
      generators. Should be set in derived classes.
  """

  on_policy = None

  def __init__(self, task,
               value_model=None,
               value_optimizer=None,
               value_lr_schedule=lr.multifactor,
               value_batch_size=64,
               value_train_steps_per_epoch=500,
               value_evals_per_epoch=1,
               value_eval_steps=1,
               n_shared_layers=0,
               added_policy_slice_length=0,
               n_replay_epochs=1,
               scale_value_targets=False,
               q_value=False,
               q_value_aggregate='logsumexp',
               q_value_temperature=1.0,
               q_value_n_samples=1,
               q_value_normalization=False,
               offline=False,
               **kwargs):  # Arguments of PolicyAgent come here.
    """Configures the actor-critic trainer.

    Args:
      task: `RLTask` instance to use.
      value_model: Model to use for the value function.
      value_optimizer: Optimizer to train the value model.
      value_lr_schedule: lr schedule for value model training.
      value_batch_size: Batch size for value model training.
      value_train_steps_per_epoch: Number of steps are we using to train the
          value model in each epoch.
      value_evals_per_epoch: Number of value trainer evaluations per RL epoch.
          Every evaluation, we also synchronize the weights of the target
          network.
      value_eval_steps: Number of value trainer steps per evaluation; only
          affects metric reporting.
      n_shared_layers: Number of layers to share between value and policy
          models.
      added_policy_slice_length: How much longer should slices of
          trajectories be for policy than for value training; this
          is useful for TD calculations and only affect the length
          of elements produced for policy batches; value batches
          have maximum length set by `max_slice_length` in `**kwargs`.
      n_replay_epochs: Number of last epochs to take into the replay buffer;
          only makes sense for off-policy algorithms.
      scale_value_targets: If `True`, scale value function targets by
          `1 / (1 - gamma)`.
      q_value: If `True`, use Q-values as baselines.
      q_value_aggregate: How to aggregate Q-values. Options: 'mean', 'max',
        'softmax', 'logsumexp'.
      q_value_temperature: Temperature parameter for the 'softmax' and
        'logsumexp' aggregation methods.
      q_value_n_samples: Number of samples to average over when calculating
          baselines based on Q-values.
      q_value_normalization: How to normalize Q-values before aggregation.
          Allowed values: 'std', 'abs', `None`. If `None`, don't normalize.
      offline: Whether to train in offline mode. This matters for some
        algorithms, e.g. QWR.
      **kwargs: Arguments for `PolicyAgent` superclass.
    """
    self._n_shared_layers = n_shared_layers
    self._value_batch_size = value_batch_size
    self._value_train_steps_per_epoch = value_train_steps_per_epoch
    self._value_evals_per_epoch = value_evals_per_epoch
    self._value_eval_steps = value_eval_steps

    # The 2 below will be initalized in super.__init__ anyway, but are needed
    # to construct value batches which are needed before PolicyAgent init
    # since policy input creation calls the value model -- hence this code.
    self._task = task
    self._max_slice_length = kwargs.get('max_slice_length', 1)
    self._added_policy_slice_length = added_policy_slice_length
    self._n_replay_epochs = n_replay_epochs
    task.set_n_replay_epochs(n_replay_epochs)

    if scale_value_targets:
      self._value_network_scale = 1 / (1 - self._task.gamma)
    else:
      self._value_network_scale = 1

    self._q_value = q_value
    self._q_value_aggregate = q_value_aggregate
    self._q_value_temperature = q_value_temperature
    self._q_value_n_samples = q_value_n_samples
    self._q_value_normalization = q_value_normalization

    is_discrete = isinstance(self._task.action_space, gym.spaces.Discrete)
    self._is_discrete = is_discrete
    self._vocab_size = None
    self._sample_all_discrete_actions = False
    if q_value and is_discrete:
      self._vocab_size = self.task.action_space.n
      # TODO(lukaszkaiser): the code below is specific to AWR, move it.
      # If n_samples = n_actions, we'll take them all in actor and reweight.
      if self._q_value_n_samples == self._vocab_size:
        # TODO(lukaszkaiser): set this explicitly once it's in AWR Trainer.
        self._sample_all_discrete_actions = True
    if offline and is_discrete:
      raise NotImplementedError(
          'Offline training is only supported for continuous action spaces for '
          'now.'
      )
    self._offline = offline

    if q_value:
      value_model = functools.partial(value_model,
                                      inject_actions=True,
                                      is_discrete=is_discrete,
                                      vocab_size=self._vocab_size)
    self._value_eval_model = value_model(mode='eval')
    self._value_eval_model.init(self._value_model_signature)
    self._value_eval_jit = tl.jit_forward(
        self._value_eval_model.pure_fn, fastmath.local_device_count(),
        do_mean=False)

    # Initialize policy training.
    super().__init__(task, **kwargs)

    # Initialize training of the value function.
    value_output_dir = kwargs.get('output_dir', None)
    if value_output_dir is not None:
      value_output_dir = os.path.join(value_output_dir, 'value')
      # If needed, create value_output_dir and missing parent directories.
      if not tf.io.gfile.isdir(value_output_dir):
        tf.io.gfile.makedirs(value_output_dir)
    self._value_inputs = data.inputs.Inputs(
        train_stream=lambda _: self.value_batches_stream())
    self._value_trainer = supervised.Trainer(
        model=value_model,
        optimizer=value_optimizer,
        lr_schedule=value_lr_schedule(),
        loss_fn=tl.L2Loss(),
        inputs=self._value_inputs,
        output_dir=value_output_dir,
        metrics={'value_loss': tl.L2Loss(),
                 'value_mean': self.value_mean})

  @property
  def value_mean(self):
    """The mean value of the value function."""
    # TODO(henrykm): A better solution would take into account the masks
    def f(values):
      return jnp.mean(values)
    return tl.Fn('ValueMean', f)

  @property
  def _value_model_signature(self):
    obs_sig = shapes.signature(self._task.observation_space)
    target_sig = mask_sig = shapes.ShapeDtype(
        shape=(1, 1, 1),
    )
    inputs_sig = (obs_sig.replace(shape=(1, 1) + obs_sig.shape),)
    if self._q_value:
      act_sig = shapes.signature(self._task.action_space)
      inputs_sig += (act_sig.replace(shape=(1, 1) + act_sig.shape),)
    return (*inputs_sig, target_sig, mask_sig)

  @property
  def _replay_epochs(self):
    if self.on_policy:
      assert self._n_replay_epochs == 1, (
          'Non-unit replay buffer size only makes sense for off-policy '
          'algorithms.'
      )
    return [-(ep + 1) for ep in range(self._n_replay_epochs)]

  def _run_value_model(self, observations, dist_inputs):
    if dist_inputs is None:
      dist_inputs = jnp.zeros(
          observations.shape[:2] + (self._policy_dist.n_inputs,)
      )

    actions = None
    if self._q_value:
      if self._sample_all_discrete_actions:
        # Since we want to sample all actions, start by creating their list.
        act = np.arange(self._vocab_size)
        # Now act is a vector [0, ..., vocab_size-1], but we'll need to tile it.
        # Add extra dimenstions so it's the same dimensionality as dist_inputs.
        act = jnp.reshape(act, [-1] + [1] * (len(dist_inputs.shape) - 1))
        # Now act is [vocab_size, 1, ..., 1], dimensionality of dist_inputs.
      dist_inputs = jnp.broadcast_to(
          dist_inputs, (self._q_value_n_samples,) + dist_inputs.shape)
      if self._sample_all_discrete_actions:
        actions = act + jnp.zeros(dist_inputs.shape[:-1], dtype=jnp.int32)
        actions = jnp.swapaxes(actions, 0, 1)
      # Swapping the n_samples and batch_size axes, so the input is split
      # between accelerators along the batch_size axis.
      dist_inputs = jnp.swapaxes(dist_inputs, 0, 1)
      if not self._sample_all_discrete_actions:
        actions = self._policy_dist.sample(dist_inputs)
      log_probs = self._policy_dist.log_prob(dist_inputs, actions)
      obs = observations
      obs = jnp.reshape(obs, [obs.shape[0], 1] + list(obs.shape[1:]))
      inputs = (obs, actions)
    else:
      log_probs = None
      inputs = (observations,)

    n_devices = fastmath.local_device_count()
    weights = tl.for_n_devices(self._value_eval_model.weights, n_devices)
    state = tl.for_n_devices(self._value_eval_model.state, n_devices)
    rng = self._value_eval_model.rng
    values, _ = self._value_eval_jit(inputs, weights, state, rng)
    values *= self._value_network_scale
    values = jnp.squeeze(values, axis=-1)  # Remove the singleton depth dim.
    return (values, actions, log_probs)

  def _aggregate_values(self, values, aggregate, act_log_probs):
    # Normalize the Q-values before aggragetion, so it can adapt to the scale
    # of the returns. This does not affect mean and max aggregation.
    scale = 1
    epsilon = 1e-5
    if self._q_value_normalization == 'std':
      scale = jnp.std(values) + epsilon
    elif self._q_value_normalization == 'abs':
      scale = jnp.mean(jnp.abs(values - jnp.mean(values))) + epsilon
    values /= scale

    temp = self._q_value_temperature
    if self._q_value:
      assert values.shape[:2] == (
          self._value_batch_size, self._q_value_n_samples
      )
      if aggregate == 'max':
        # max_a Q(s, a)
        values = jnp.max(values, axis=1)
      elif aggregate == 'softmax':
        # sum_a (Q(s, a) * w(s, a))
        # where w(s, .) = softmax (Q(s, .) / T)
        weights = tl.Softmax(axis=1)(values / temp)
        values = jnp.sum(values * weights, axis=1)
      elif aggregate == 'logsumexp':
        # log(mean_a exp(Q(s, a) / T)) * T
        n = values.shape[1]
        values = (fastmath.logsumexp(values / temp, axis=1) - jnp.log(n)) * temp
      else:
        assert aggregate == 'mean'
        # mean_a Q(s, a)
        if self._sample_all_discrete_actions:
          values = jnp.sum(values * jnp.exp(act_log_probs), axis=1)
        else:
          values = jnp.mean(values, axis=1)

    # Re-scale the Q-values after aggregation.
    values *= scale
    return np.array(values)  # Move the values to CPU.

  def _get_dist_inputs(self, trajectory):
    if not self._offline:
      return trajectory.dist_inputs
    else:
      return trajectory.action

  def value_batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    max_slice_length = self._max_slice_length + self._added_policy_slice_length
    for np_trajectory in self._task.trajectory_batch_stream(
        self._value_batch_size,
        max_slice_length=max_slice_length,
        min_slice_length=(1 + self._added_policy_slice_length),
        margin=self._added_policy_slice_length,
        epochs=self._replay_epochs,
    ):
      dist_inputs = self._get_dist_inputs(np_trajectory)
      (values, _, act_log_probs) = self._run_value_model(
          np_trajectory.observation, dist_inputs
      )
      values = self._aggregate_values(
          values, self._q_value_aggregate, act_log_probs)

      # TODO(pkozakowski): Add some shape assertions and docs.
      # Calculate targets based on the advantages over the target network - this
      # allows TD learning for value networks.
      advantages = self._advantage_estimator(
          rewards=np_trajectory.reward,
          returns=np_trajectory.return_,
          values=values,
          dones=np_trajectory.done,
          discount_mask=np_trajectory.env_info.discount_mask,
      )
      length = advantages.shape[1]
      values = values[:, :length]
      target_returns = values + advantages

      inputs = (np_trajectory.observation[:, :length],)
      if self._q_value:
        inputs += (np_trajectory.action[:, :length],)

      # Insert an extra depth dimension, so the target shape is consistent with
      # the network output shape.
      yield (
          # Inputs: observations and maybe actions.
          *inputs,
          # Targets: computed returns.
          target_returns[:, :, None] / self._value_network_scale,
          # Mask to zero-out padding.
          np_trajectory.mask[:, :length, None],
      )

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TimeStepBatch and values.

    Args:
      trajectory: a TimeStepBatch, the trajectory to create inputs from
      values: a numpy array: value function computed on trajectory

    Returns:
      a tuple of numpy arrays of the form (inputs, x1, x2, ...) that will be
      passed to the policy model; policy model will compute outputs from
      inputs and (outputs, x1, x2, ...) will be passed to self.policy_loss
      which should be overridden accordingly.
    """
    return NotImplementedError

  def policy_batches_stream(self):
    """Use the RLTask self._task to create inputs to the policy model."""
    # Maximum slice length for policy is max_slice_len + the added policy len.
    max_slice_length = self._max_slice_length + self._added_policy_slice_length
    for np_trajectory in self._task.trajectory_batch_stream(
        self._policy_batch_size,
        epochs=self._replay_epochs,
        max_slice_length=max_slice_length,
        margin=self._added_policy_slice_length,
    ):
      dist_inputs = self._get_dist_inputs(np_trajectory)
      (values, _, act_log_probs) = self._run_value_model(
          np_trajectory.observation, dist_inputs)
      values = self._aggregate_values(values, 'mean', act_log_probs)
      if len(values.shape) != 2:
        raise ValueError('Values are expected to have shape ' +
                         '[batch_size, length], got: %s' % str(values.shape))
      if values.shape[0] != self._policy_batch_size:
        raise ValueError('Values first dimension should = policy batch size, ' +
                         '%d != %d' %(values.shape[0], self._policy_batch_size))
      yield self.policy_inputs(np_trajectory, values)

  def train_epoch(self):
    """Trains RL for one epoch."""
    # Copy policy state accumulated during data collection to the trainer.
    self._policy_trainer.model_state = self._policy_collect_model.state

    # Copy policy weights and state to value trainer.
    if self._n_shared_layers > 0:
      _copy_model_weights_and_state(
          0, self._n_shared_layers, self._policy_trainer, self._value_trainer
      )

    # Update the target value network.
    self._value_eval_model.weights = self._value_trainer.model_weights
    self._value_eval_model.state = self._value_trainer.model_state

    n_value_evals = rl_training.remaining_evals(
        self._value_trainer.step,
        self._epoch,
        self._value_train_steps_per_epoch,
        self._value_evals_per_epoch)
    for _ in range(n_value_evals):
      self._value_trainer.train_epoch(
          self._value_train_steps_per_epoch // self._value_evals_per_epoch,
          self._value_eval_steps,
      )
      # Update the target value network.
      self._value_eval_model.weights = self._value_trainer.model_weights
      self._value_eval_model.state = self._value_trainer.model_state

    # Copy value weights and state to policy trainer.
    if self._n_shared_layers > 0:
      _copy_model_weights_and_state(
          0, self._n_shared_layers, self._value_trainer, self._policy_trainer
      )
    n_policy_evals = rl_training.remaining_evals(
        self._policy_trainer.step,
        self._epoch,
        self._policy_train_steps_per_epoch,
        self._policy_evals_per_epoch)
    # Check if there was a restart after value training finishes and policy not.
    stopped_after_value = (n_value_evals == 0 and
                           n_policy_evals < self._policy_evals_per_epoch)
    should_copy_weights = self._n_shared_layers > 0 and not stopped_after_value
    if should_copy_weights:
      _copy_model_weights_and_state(
          0, self._n_shared_layers, self._value_trainer, self._policy_trainer
      )

    # Update the target value network.
    self._value_eval_model.weights = self._value_trainer.model_weights
    self._value_eval_model.state = self._value_trainer.model_state

    for _ in range(n_policy_evals):
      self._policy_trainer.train_epoch(
          self._policy_train_steps_per_epoch // self._policy_evals_per_epoch,
          self._policy_eval_steps,
      )

  def close(self):
    self._value_trainer.close()
    super().close()


def _copy_model_weights_and_state(  # pylint: disable=invalid-name
    start, end, from_trainer, to_trainer, copy_optimizer_slots=False
):
  """Copy model weights[start:end] from from_trainer to to_trainer."""
  from_weights = from_trainer.model_weights
  to_weights = list(to_trainer.model_weights)
  shared_weights = from_weights[start:end]
  to_weights[start:end] = shared_weights
  to_trainer.model_weights = to_weights

  from_state = from_trainer.model_state
  to_state = list(to_trainer.model_state)
  shared_state = from_state[start:end]
  to_state[start:end] = shared_state
  to_trainer.model_state = to_state

  if copy_optimizer_slots:
    # TODO(lukaszkaiser): make a nicer API in Trainer to support this.
    # Currently we use the hack below. Note [0] since that's the model w/o loss.
    # pylint: disable=protected-access
    from_slots = from_trainer._opt_state.slots[0][start:end]
    to_slots = to_trainer._opt_state.slots[0]
    # The lines below do to_slots[start:end] = from_slots, but on tuples.
    new_slots = to_slots[:start] + from_slots[start:end] + to_slots[end:]
    new_slots = tuple([new_slots] + list(to_trainer._opt_state.slots[1:]))
    to_trainer._opt_state = to_trainer._opt_state._replace(slots=new_slots)
    # pylint: enable=protected-access


class AdvantageBasedActorCriticAgent(ActorCriticAgent):
  """Base class for advantage-based actor-critic algorithms."""

  def __init__(
      self,
      task,
      advantage_estimator=rl_advantages.td_lambda,
      advantage_normalization=True,
      advantage_normalization_epsilon=1e-5,
      advantage_normalization_factor=1.0,
      added_policy_slice_length=0,
      **kwargs
  ):
    self._advantage_estimator = advantage_estimator(
        gamma=task.gamma, margin=added_policy_slice_length
    )
    self._advantage_normalization = advantage_normalization
    self._advantage_normalization_epsilon = advantage_normalization_epsilon
    self._advantage_normalization_factor = advantage_normalization_factor
    super().__init__(
        task, added_policy_slice_length=added_policy_slice_length, **kwargs
    )

  def policy_inputs(self, trajectory, values):
    """Create inputs to policy model from a TimeStepBatch and values."""
    # How much TD to use is determined by the added policy slice length,
    # as the policy batches need to be this much longer to calculate TD.
    advantages = self._advantage_estimator(
        rewards=trajectory.reward,
        returns=trajectory.return_,
        values=values,
        dones=trajectory.done,
        discount_mask=trajectory.env_info.discount_mask,
    )
    # Observations should be the same length as advantages - so if we are
    # using n_extra_steps, we need to trim the length to match.
    obs = trajectory.observation[:, :advantages.shape[1]]
    act = trajectory.action[:, :advantages.shape[1]]
    mask = trajectory.mask[:, :advantages.shape[1]]  # Mask to zero-out padding.
    if trajectory.dist_inputs is not None:
      dist_inputs = self._get_dist_inputs(trajectory)
      dist_inputs = dist_inputs[:, :advantages.shape[1]]
    else:
      dist_inputs = jnp.zeros(advantages.shape + (self._policy_dist.n_inputs,))
    # Shape checks to help debugging.
    if len(advantages.shape) != 2:
      raise ValueError('Advantages are expected to have shape ' +
                       '[batch_size, length], got: %s' % str(advantages.shape))
    if act.shape[0:2] != advantages.shape:
      raise ValueError('First 2 dimensions of actions should be the same as in '
                       'advantages, %s != %s' % (act.shape[0:2],
                                                 advantages.shape))
    if obs.shape[0:2] != advantages.shape:
      raise ValueError('First 2 dimensions of observations should be the same '
                       'as in advantages, %s != %s' % (obs.shape[0:2],
                                                       advantages.shape))
    if dist_inputs.shape[:2] != advantages.shape:
      raise ValueError('First 2 dimensions of dist_inputs should be the same '
                       'as in advantages, %s != %s' % (dist_inputs.shape[:2],
                                                       advantages.shape))
    if mask.shape != advantages.shape:
      raise ValueError('Mask and advantages shapes should be the same'
                       ', %s != %s' % (mask.shape, advantages.shape))
    return (obs, act, advantages, dist_inputs, mask)

  @property
  def policy_loss_given_log_probs(self):
    """Policy loss given action log-probabilities."""
    raise NotImplementedError

  def _preprocess_advantages(self, advantages):
    if self._advantage_normalization:
      advantages = self._advantage_normalization_factor * (
          (advantages - jnp.mean(advantages)) /
          (jnp.std(advantages) + self._advantage_normalization_epsilon)
      )
    return advantages

  @property
  def policy_loss(self, **unused_kwargs):
    """Policy loss."""
    def LossInput(dist_inputs, actions, advantages, old_dist_inputs):  # pylint: disable=invalid-name
      """Calculates action log probabilities and normalizes advantages."""
      advantages = self._preprocess_advantages(advantages)
      log_probs = self._policy_dist.log_prob(dist_inputs, actions)
      old_log_probs = self._policy_dist.log_prob(old_dist_inputs, actions)
      return (log_probs, advantages, old_log_probs)

    return tl.Serial(
        tl.Fn('LossInput', LossInput, n_out=3),
        # Policy loss is expected to consume
        # (log_probs, advantages, old_log_probs, mask).
        self.policy_loss_given_log_probs,
    )

  @property
  def policy_metrics(self):
    metrics = super().policy_metrics
    metrics.update({
        'advantage_mean': self.advantage_mean,
        'advantage_std': self.advantage_std,
    })
    return metrics

  @property
  def advantage_mean(self):
    return tl.Serial([
        # (dist_inputs, advantages, old_dist_inputs, mask)
        tl.Select([1]),  # Select just the advantages.
        tl.Fn('AdvantageMean', lambda x: jnp.mean(x)),  # pylint: disable=unnecessary-lambda
    ])

  @property
  def advantage_std(self):
    return tl.Serial([
        # (dist_inputs, advantages, old_dist_inputs, mask)
        tl.Select([1]),  # Select just the advantages.
        tl.Fn('AdvantageStd', lambda x: jnp.std(x)),  # pylint: disable=unnecessary-lambda
    ])


# TODO(pkozakowski): Move to a better place.
@gin.configurable(module='trax.rl')
def every(n_steps):
  """Returns True every n_steps, for use as *_at functions in various places."""
  return lambda step: step % n_steps == 0


# TODO(pkozakowski): Rewrite all interleaved actor-critic algos to subclass
# this, then rename to ActorCriticAgent and remove the other base classes.
class LoopActorCriticAgent(rl_training.Agent):
  """Base class for actor-critic algorithms based on `Loop`."""

  on_policy = None

  def __init__(
      self, task, model_fn,
      optimizer=adam.Adam,
      policy_lr_schedule=lr.multifactor,
      policy_n_steps_per_epoch=1000,
      policy_weight_fn=(lambda x: x),
      value_lr_schedule=lr.multifactor,
      value_n_steps_per_epoch=1000,
      value_sync_at=(lambda x: x % 100 == 0),
      advantage_estimator=rl_advantages.monte_carlo,
      batch_size=64,
      network_eval_at=None,
      n_eval_batches=1,
      max_slice_length=1,
      margin=0,
      n_replay_epochs=1,
      **kwargs
  ):
    """Initializes LoopActorCriticAgent.

    Args:
      task: `RLTask` instance to use.
      model_fn: Function mode -> Trax model, building a joint policy and value
        network.
      optimizer: Optimizer for the policy and value networks.
      policy_lr_schedule: Learning rate schedule for the policy network.
      policy_n_steps_per_epoch: Number of steps to train the policy network for
        in each epoch.
      policy_weight_fn: Function advantages -> weights for calculating the
        log probability weights in policy training.
      value_lr_schedule: Learning rate schedule for the value network.
      value_n_steps_per_epoch: Number of steps to train the value network for
        in each epoch.
      value_sync_at: Function step -> bool indicating when to synchronize the
        target network with the trained network in value training.
      advantage_estimator: Advantage estimator to use in policy and value
        training.
      batch_size: Batch size for training the networks.
      network_eval_at: Function step -> bool indicating in when to evaluate the
        networks.
      n_eval_batches: Number of batches to compute the network evaluation
        metrics on.
      max_slice_length: Maximum length of a trajectory slice to train on.
      margin: Number of timesteps to add at the end of each trajectory slice for
        better advantage estimation.
      n_replay_epochs: Number of epochs of trajectories to store in the replay
        buffer.
      **kwargs: Keyword arguments forwarded to Agent.
    """
    super().__init__(task, **kwargs)

    self._policy_dist = distributions.create_distribution(
        self.task.action_space
    )
    model_fn = functools.partial(
        model_fn,
        policy_distribution=self._policy_dist,
    )
    train_model = model_fn(mode='train')
    eval_model = model_fn(mode='eval')

    trajectory_batch_stream = self._init_trajectory_batch_stream(
        batch_size, max_slice_length, margin, n_replay_epochs
    )
    advantage_estimator = advantage_estimator(task.gamma, margin=margin)
    (value_train_task, value_eval_task) = self._init_value_tasks(
        trajectory_batch_stream,
        optimizer=optimizer(),
        lr_schedule=value_lr_schedule(),
        advantage_estimator=advantage_estimator,
        train_model=train_model,
        eval_model=eval_model,
        sync_at=value_sync_at,
        n_steps_per_epoch=value_n_steps_per_epoch,
        n_eval_batches=n_eval_batches,
    )
    (policy_train_task, policy_eval_task) = self._init_policy_tasks(
        trajectory_batch_stream,
        optimizer=optimizer(),
        lr_schedule=policy_lr_schedule(),
        advantage_estimator=advantage_estimator,
        value_train_task=value_train_task,
        weight_fn=policy_weight_fn,
        n_eval_batches=n_eval_batches,
    )
    self._init_loop(
        train_model=train_model,
        eval_model=eval_model,
        policy_train_and_eval_task=(policy_train_task, policy_eval_task),
        value_train_and_eval_task=(value_train_task, value_eval_task),
        eval_at=network_eval_at,
        policy_n_steps_per_epoch=policy_n_steps_per_epoch,
        value_n_steps_per_epoch=value_n_steps_per_epoch,
    )
    self._init_collection(model_fn, policy_train_task.sample_batch)

  def _init_trajectory_batch_stream(
      self, batch_size, max_slice_length, margin, n_replay_epochs
  ):
    assert self.on_policy is not None, 'Attribute "on_policy" not set.'
    if self.on_policy:
      assert n_replay_epochs == 1, (
          'Non-unit replay buffer size only makes sense for off-policy '
          'algorithms.'
      )
    self._task.set_n_replay_epochs(n_replay_epochs)
    self._max_slice_length = max_slice_length
    return self._task.trajectory_batch_stream(
        batch_size,
        epochs=[-(ep + 1) for ep in range(n_replay_epochs)],
        min_slice_length=(1 + margin),
        max_slice_length=(self._max_slice_length + margin),
        margin=margin,
    )

  def _init_value_tasks(
      self,
      trajectory_batch_stream,
      optimizer,
      lr_schedule,
      advantage_estimator,
      train_model,
      eval_model,
      sync_at,
      n_steps_per_epoch,
      n_eval_batches,
  ):
    def sync_also_at_epoch_boundaries(step):
      return sync_at(step) or (
          # 0 - end of the epoch, 1 - beginning of the next.
          step % n_steps_per_epoch in (0, 1)
      )

    head_selector = tl.Select([1])
    value_train_task = value_tasks.ValueTrainTask(
        trajectory_batch_stream,
        optimizer,
        lr_schedule,
        advantage_estimator=advantage_estimator,
        model=train_model,
        target_model=eval_model,
        target_scale=(1 - self.task.gamma),
        sync_at=sync_also_at_epoch_boundaries,
        head_selector=head_selector,
    )
    value_eval_task = value_tasks.ValueEvalTask(
        value_train_task, n_eval_batches, head_selector
    )
    return (value_train_task, value_eval_task)

  def _init_policy_tasks(
      self,
      trajectory_batch_stream,
      optimizer,
      lr_schedule,
      advantage_estimator,
      value_train_task,
      weight_fn,
      n_eval_batches,
  ):
    head_selector = tl.Select([0], n_in=2)
    policy_train_task = policy_tasks.PolicyTrainTask(
        trajectory_batch_stream,
        optimizer,
        lr_schedule,
        self._policy_dist,
        advantage_estimator=advantage_estimator,
        value_fn=value_train_task.value,
        weight_fn=weight_fn,
        head_selector=head_selector,
    )
    policy_eval_task = policy_tasks.PolicyEvalTask(
        policy_train_task, n_eval_batches, head_selector
    )
    return (policy_train_task, policy_eval_task)

  def _init_loop(
      self,
      train_model,
      eval_model,
      policy_train_and_eval_task,
      value_train_and_eval_task,
      eval_at,
      policy_n_steps_per_epoch,
      value_n_steps_per_epoch,
  ):
    (policy_train_task, policy_eval_task) = policy_train_and_eval_task
    (value_train_task, value_eval_task) = value_train_and_eval_task

    if self._output_dir is not None:
      model_output_dir = os.path.join(self._output_dir, 'model')
    else:
      model_output_dir = None

    self._n_train_steps_per_epoch = (
        policy_n_steps_per_epoch + value_n_steps_per_epoch
    )

    checkpoint_at = lambda step: step % self._n_train_steps_per_epoch == 0

    def which_task(step):
      if step % self._n_train_steps_per_epoch < value_n_steps_per_epoch:
        return 1
      else:
        return 0

    self._loop = supervised.training.Loop(
        model=train_model,
        tasks=(policy_train_task, value_train_task),
        eval_model=eval_model,
        eval_tasks=(policy_eval_task, value_eval_task),
        output_dir=model_output_dir,
        eval_at=eval_at,
        checkpoint_at=checkpoint_at,
        which_task=which_task,
    )

    # Validate the restored checkpoints.
    # TODO(pkozakowski): Move this to the base class once all Agents use Loop.
    if self._loop.step != self._epoch * self._n_train_steps_per_epoch:
      raise ValueError(
          'The number of Loop steps must equal the number of Agent epochs '
          'times the number of steps per epoch, got {}, {} and {}.'.format(
              self._loop.step, self._epoch, self._n_train_steps_per_epoch
          )
      )

  def _init_collection(self, model_fn, sample_batch):
    self._collect_model = model_fn(mode='collect')
    self._collect_model.init(shapes.signature(sample_batch))
    if self._task._initial_trajectories == 0:  # pylint: disable=protected-access
      self._collect_trajectories()

  @property
  def loop(self):
    """Loop exposed for testing."""
    return self._loop

  def policy(self, trajectory, temperature=1.0):
    """Policy function that allows to play using this agent."""
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    return rl_training.network_policy(
        collect_model=self._collect_model,
        policy_distribution=self._policy_dist,
        loop=self.loop,
        trajectory_np=trajectory_np,
        head_index=0,
        temperature=temperature,
    )

  def train_epoch(self):
    """Trains RL for one epoch."""
    # Copy policy state accumulated during data collection to the trainer.
    self._loop.update_weights_and_state(state=self._collect_model.state)
    # Perform one gradient step per training epoch to ensure we stay on policy.
    self._loop.run(n_steps=self._n_train_steps_per_epoch)


### Implementations of common actor-critic algorithms.


class A2C(AdvantageBasedActorCriticAgent):
  """Trains policy and value models using the A2C algorithm."""

  on_policy = True

  def __init__(self, task, entropy_coeff=0.01, **kwargs):
    """Configures the A2C Trainer."""
    self._entropy_coeff = entropy_coeff
    super().__init__(task, **kwargs)

  @property
  def policy_loss_given_log_probs(self):
    """Definition of the Advantage Actor Critic (A2C) loss."""
    # A2C is one of the most basic actor-critic RL algorithms.
    # TODO(henrykm) re-factor f into rl_layers and finally share code between
    # actor_critic.py and actor_critic_joint.py - requires change of inputs
    # in actor_critic_joint.py from dist_inputs to log_probs.
    def f(log_probs, advantages, old_log_probs, mask):
      del old_log_probs  # Not used in A2C.
      # log_probs of the shape float32[128,1]
      # advantages of the shape int32[128,1]
      # mask of the shape int32[128,1]
      if log_probs.shape != advantages.shape:
        raise ValueError('New log-probs and advantages shapes '
                         'should be the same, %s != %s' % (log_probs.shape,
                                                           advantages.shape))
      if log_probs.shape != mask.shape:
        raise ValueError('New log-probs and mask shapes should be the same'
                         ', %s != %s' % (log_probs.shape, mask.shape))

      a2c_objective = -jnp.sum(log_probs * advantages * mask) / jnp.sum(mask)

      entropy_vec = self._policy_dist.entropy(log_probs) * self._entropy_coeff
      entropy_loss = jnp.mean(entropy_vec)

      combined_loss = a2c_objective - entropy_loss

      return combined_loss

    return tl.Fn('A2CLoss', f)


class PPO(AdvantageBasedActorCriticAgent):
  """The Proximal Policy Optimization Algorithm aka PPO.

  Trains policy and value models using the PPO algorithm.
  """

  on_policy = True

  def __init__(self, task, epsilon=0.2, entropy_coeff=0.01, **kwargs):
    """Configures the PPO Trainer."""
    self._entropy_coeff = entropy_coeff
    self._epsilon = epsilon
    super().__init__(task, **kwargs)

  @property
  def policy_loss_given_log_probs(self):
    """Definition of the Proximal Policy Optimization loss."""
    def f(new_log_probs, advantages, old_log_probs, mask):
      # new_log_probs of the shape float32[128,1]
      # advantages of the shape int32[128,1]
      # old_log_probs of the shape int32[128,1]
      # mask of the shape int32[128,1]
      if new_log_probs.shape != advantages.shape:
        raise ValueError('New log-probs and advantages shapes '
                         'should be the same, %s != %s' % (new_log_probs.shape,
                                                           advantages.shape))
      if new_log_probs.shape != old_log_probs.shape:
        raise ValueError('New log-probs and old log-probs shapes '
                         'should be the same, %s != %s' % (new_log_probs.shape,
                                                           old_log_probs.shape))
      if new_log_probs.shape != mask.shape:
        raise ValueError('New log-probs and mask shapes should be the same'
                         ', %s != %s' % (new_log_probs.shape, mask.shape))

      # The ratio between new_probs and old_probs expressed
      # using log_probs and exponentiation
      probs_ratio = jnp.exp(new_log_probs - old_log_probs)
      if advantages.shape != probs_ratio.shape:
        raise ValueError('New log-probs and old log probs shapes '
                         'should be the same, %s != %s' % (advantages.shape,
                                                           probs_ratio.shape))
      unclipped_objective = probs_ratio * advantages
      clipped_objective = jnp.clip(probs_ratio,
                                   1 - self._epsilon,
                                   1 + self._epsilon) * advantages

      if unclipped_objective.shape != probs_ratio.shape:
        raise ValueError('unclipped_objective and clipped_objective shapes '
                         'should be the same, %s != %s' % (
                             unclipped_objective.shape,
                             clipped_objective.shape))

      ppo_objective = jnp.minimum(unclipped_objective, clipped_objective)

      if ppo_objective.shape != mask.shape:
        raise ValueError('ppo_objective and mask shapes '
                         'should be the same, %s != %s' % (
                             ppo_objective.shape,
                             mask.shape))

      ppo_loss = -jnp.sum(ppo_objective * mask) / jnp.sum(mask)
      entropy_vec = self._policy_dist.entropy(
          new_log_probs) * self._entropy_coeff
      entropy_loss = jnp.mean(entropy_vec)
      combined_loss = ppo_loss - entropy_loss

      return combined_loss
    return tl.Fn('PPOLoss', f)


def _weighted_percentiles(x, thresholds):
  """Calculate weights for x by percentile-and-weights given in thresholds.

  Thresholds contain a list of (p, weight, minumum). For each threshold,
  all elements of x that are above the p-th percentile *and* above minimum
  get the weight weight, and all other get the weight 0.
  The result is the sum over all thresholds.

  Args:
    x: tensor to calculate the weights for
    thresholds: list of triples (percentile, weight, minimum) used to
      calculate the weights (see above how)

  Returns:
    weights, a tensor of the same shape as x
  """
  res = []
  for (percentile, weight, minimum) in thresholds:
    threshold = jnp.percentile(x, percentile)
    if minimum is not None:
      threshold = jnp.maximum(minimum, threshold)
    zero_ones = jnp.where(x < threshold, jnp.zeros_like(x), jnp.ones_like(x))
    res.append(weight * zero_ones)
  return sum(res)


# AWR is an off-policy actor-critic RL algorithm.
def awr_weights(advantages, beta, thresholds):
  if thresholds:
    return _weighted_percentiles(advantages, thresholds)
  return jnp.exp(advantages / beta)


# Helper functions for computing AWR metrics.
def awr_metrics(beta, thresholds, preprocess_layer=None):
  return {  # pylint: disable=g-complex-comprehension
      'awr_weight_' + name: awr_weight_stat(name, fn, beta, thresholds,
                                            preprocess_layer)
      for (name, fn) in [
          ('mean', jnp.mean),
          ('std', jnp.std),
          ('min', jnp.min),
          ('max', jnp.max),
      ]
  }


def awr_weight_stat(stat_name, stat_fn, beta, thresholds, preprocess_layer):
  # Select just the advantages if preprocess layer is not given.
  preprocess = tl.Select([1]) if preprocess_layer is None else preprocess_layer
  return tl.Serial([
      preprocess,
      tl.Fn(
          'AWRWeight' + stat_name.capitalize(),
          lambda x: stat_fn(awr_weights(x, beta, thresholds)),
      ),
  ])


def AWRLoss(beta, w_max, thresholds):  # pylint: disable=invalid-name
  """Definition of the Advantage Weighted Regression (AWR) loss."""
  def f(log_probs, advantages, old_log_probs, mask):
    del old_log_probs  # Not used in AWR.
    weights = jnp.minimum(awr_weights(advantages, beta, thresholds), w_max)
    return -jnp.sum(log_probs * weights * mask) / jnp.sum(mask)
  return tl.Fn('AWRLoss', f)


class AWR(AdvantageBasedActorCriticAgent):
  """Trains policy and value models using AWR."""

  on_policy = False

  def __init__(self, task, beta=1.0, w_max=20.0, thresholds=None, **kwargs):
    """Configures the AWR Trainer."""
    self._beta = beta
    self._w_max = w_max
    self._thresholds = thresholds
    super().__init__(task, **kwargs)

  @property
  def policy_loss_given_log_probs(self):
    """Policy loss."""
    return AWRLoss(beta=self._beta, w_max=self._w_max,
                   thresholds=self._thresholds)  # pylint: disable=no-value-for-parameter


class LoopAWR(LoopActorCriticAgent):
  """Advantage Weighted Regression."""

  on_policy = False

  def __init__(self, task, model_fn, beta=1.0, w_max=20, **kwargs):
    def policy_weight_fn(advantages):
      return jnp.minimum(jnp.exp(advantages / beta), w_max)
    super().__init__(
        task, model_fn, policy_weight_fn=policy_weight_fn, **kwargs
    )


def SamplingAWRLoss(beta, w_max, thresholds,  # pylint: disable=invalid-name
                    reweight=False, sampled_all_discrete=False):
  """Definition of the Advantage Weighted Regression (AWR) loss."""
  def f(log_probs, advantages, old_log_probs, mask):
    if reweight:  # Use new policy weights for sampled actions instead.
      mask *= jnp.exp(fastmath.stop_gradient(log_probs) - old_log_probs)
    if sampled_all_discrete:  # Actions were sampled uniformly; weight them.
      mask *= jnp.exp(old_log_probs)
    weights = jnp.minimum(awr_weights(advantages, beta, thresholds), w_max)
    return -jnp.sum(log_probs * weights * mask) / jnp.sum(mask)
  return tl.Fn('SamplingAWRLoss', f)


class SamplingAWR(AdvantageBasedActorCriticAgent):
  """Trains policy and value models using Sampling AWR."""

  on_policy = False

  def __init__(self, task, beta=1.0, w_max=20.0, thresholds=None,
               reweight=False, **kwargs):
    """Configures the AWR Trainer."""
    self._beta = beta
    self._w_max = w_max
    self._thresholds = thresholds
    self._reweight = reweight
    super().__init__(task, q_value=True, **kwargs)

  def _policy_inputs_to_advantages(self, preprocess):
    """A layer that computes advantages from policy inputs."""
    def fn(dist_inputs, actions, q_values, act_log_probs, mask):
      del dist_inputs, actions, mask
      q_values = jnp.swapaxes(q_values, 0, 1)
      act_log_probs = jnp.swapaxes(act_log_probs, 0, 1)
      if self._sample_all_discrete_actions:
        values = jnp.sum(q_values * jnp.exp(act_log_probs), axis=0)
      else:
        values = jnp.mean(q_values, axis=0)
      advantages = q_values - values  # Broadcasting values over n_samples
      if preprocess:
        advantages = self._preprocess_advantages(advantages)
      return advantages
    return tl.Fn('PolicyInputsToAdvantages', fn)

  @property
  def policy_metrics(self):
    metrics = {
        'policy_loss': self.policy_loss,
        'advantage_mean': tl.Serial(
            self._policy_inputs_to_advantages(False),
            tl.Fn('Mean', lambda x: jnp.mean(x))  # pylint: disable=unnecessary-lambda
        ),
        'advantage_std': tl.Serial(
            self._policy_inputs_to_advantages(False),
            tl.Fn('Std', lambda x: jnp.std(x))  # pylint: disable=unnecessary-lambda
        )
    }
    metrics.update(awr_metrics(
        self._beta, self._thresholds,
        preprocess_layer=self._policy_inputs_to_advantages(True)))
    return metrics

  @property
  def policy_loss(self, **unused_kwargs):
    """Policy loss."""
    def LossInput(dist_inputs, actions, q_values, act_log_probs, mask):  # pylint: disable=invalid-name
      """Calculates action log probabilities and normalizes advantages."""
      # (batch_size, n_samples, ...) -> (n_samples, batch_size, ...)
      q_values = jnp.swapaxes(q_values, 0, 1)
      mask = jnp.swapaxes(mask, 0, 1)
      actions = jnp.swapaxes(actions, 0, 1)
      act_log_probs = jnp.swapaxes(act_log_probs, 0, 1)

      # TODO(pkozakowski,lukaszkaiser): Try max here, or reweighting?
      if self._sample_all_discrete_actions:
        values = jnp.sum(q_values * jnp.exp(act_log_probs), axis=0)
      else:
        values = jnp.mean(q_values, axis=0)
      advantages = q_values - values  # Broadcasting values over n_samples
      advantages = self._preprocess_advantages(advantages)

      # Broadcast inputs and calculate log-probs
      dist_inputs = jnp.broadcast_to(
          dist_inputs, (self._q_value_n_samples,) + dist_inputs.shape)
      log_probs = self._policy_dist.log_prob(dist_inputs, actions)
      return (log_probs, advantages, act_log_probs, mask)

    return tl.Serial(
        tl.Fn('LossInput', LossInput, n_out=4),
        # Policy loss is expected to consume
        # (log_probs, advantages, old_log_probs, mask).
        SamplingAWRLoss(
            beta=self._beta, w_max=self._w_max, thresholds=self._thresholds,
            reweight=self._reweight,
            sampled_all_discrete=self._sample_all_discrete_actions)
    )

  def policy_batches_stream(self):
    """Use the RLTask self._task to create inputs to the policy model."""
    # For now TD-0 estimation of the value. TODO(pkozakowski): Support others?
    for np_trajectory in self._task.trajectory_batch_stream(
        self._policy_batch_size,
        epochs=self._replay_epochs,
        max_slice_length=self._max_slice_length,
    ):
      dist_inputs = self._get_dist_inputs(np_trajectory)
      (q_values, actions, act_log_probs) = self._run_value_model(
          np_trajectory.observation, dist_inputs)
      shapes.assert_same_shape(q_values, act_log_probs)

      # q_values shape: (batch_size, n_samples, length)
      if len(q_values.shape) != 3:
        raise ValueError('Q-values are expected to have shape [batch_size, ' +
                         'n_samples, length], got: %s' % str(q_values.shape))
      if q_values.shape[1] != self._q_value_n_samples:
        raise ValueError('Q-values dimension 1 should = n_samples, %d != %d'
                         % (q_values.shape[1], self._q_value_n_samples))
      if q_values.shape[0] != self._policy_batch_size:
        raise ValueError('Q-values dimension 0 should = policy batch size, ' +
                         '%d!=%d' %(q_values.shape[1], self._policy_batch_size))

      mask = np_trajectory.mask
      mask = np.reshape(mask, [mask.shape[0], 1] + list(mask.shape[1:]))
      mask = jnp.broadcast_to(mask, q_values.shape)
      shapes.assert_same_shape(mask, q_values)
      yield (np_trajectory.observation, actions, q_values, act_log_probs, mask)
