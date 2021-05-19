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

import contextlib
import functools
import os
import pickle
import time

import gin
import numpy as np
import tensorflow as tf

from trax import data
from trax import fastmath
from trax import jaxboard
from trax import layers as tl
from trax import models
from trax import shapes
from trax import supervised
from trax.fastmath import numpy as jnp
from trax.optimizers import adam
from trax.rl import advantages
from trax.rl import distributions
from trax.rl import normalization  # So gin files see it. # pylint: disable=unused-import
from trax.rl import policy_tasks
from trax.rl import task as rl_task
from trax.supervised import lr_schedules as lr


class Agent:
  """Abstract class for RL agents, presenting the required API."""

  def __init__(self,
               task: rl_task.RLTask,
               n_trajectories_per_epoch=None,
               n_interactions_per_epoch=None,
               n_eval_episodes=0,
               eval_steps=None,
               eval_temperatures=(0.0,),
               only_eval=False,
               output_dir=None,
               timestep_to_np=None):
    """Configures the Agent.

    Note that subclasses can have many more arguments, which will be configured
    using defaults and gin. But task and output_dir are passed explicitly.

    Args:
      task: RLTask instance, which defines the environment to train on.
      n_trajectories_per_epoch: How many new trajectories to collect in each
        epoch.
      n_interactions_per_epoch: How many interactions to collect in each epoch.
      n_eval_episodes: Number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only.
      eval_steps: an optional list of max_steps to use for evaluation
        (defaults to task.max_steps).
      eval_temperatures: we always train with temperature 1 and evaluate with
        temperature specified in the eval_temperatures list
        (defaults to [0.0, 0.5])
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      output_dir: Path telling where to save outputs such as checkpoints.
      timestep_to_np: Timestep-to-numpy function to override in the task.
    """
    if n_trajectories_per_epoch is None == n_interactions_per_epoch is None:
      raise ValueError(
          'Exactly one of n_trajectories_per_epoch or '
          'n_interactions_per_epoch should be specified.'
      )
    self._epoch = 0
    self._task = task
    self._eval_steps = eval_steps or [task.max_steps]
    if timestep_to_np is not None:
      self._task.timestep_to_np = timestep_to_np
    self._n_trajectories_per_epoch = n_trajectories_per_epoch
    self._n_interactions_per_epoch = n_interactions_per_epoch
    self._only_eval = only_eval
    self._output_dir = output_dir
    self._avg_returns = []
    self._n_eval_episodes = n_eval_episodes
    self._eval_temperatures = eval_temperatures
    self._avg_returns_temperatures = {
        eval_t: {step: [] for step in self._eval_steps
                } for eval_t in eval_temperatures
    }
    if self._output_dir is not None:
      self.init_from_file()

  @property
  def current_epoch(self):
    """Returns current step number in this training session."""
    return self._epoch

  @property
  def task(self):
    """Returns the task."""
    return self._task

  @property
  def avg_returns(self):
    return self._avg_returns

  def save_gin(self, summary_writer=None):
    assert self._output_dir is not None
    config_path = os.path.join(self._output_dir, 'config.gin')
    config_str = gin.operative_config_str()
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_str)
    if summary_writer is not None:
      summary_writer.text(
          'gin_config', jaxboard.markdownify_operative_config_str(config_str)
      )

  def save_to_file(self, file_name='rl.pkl',
                   task_file_name='trajectories.pkl'):
    """Save current epoch number and average returns to file."""
    assert self._output_dir is not None
    task_path = os.path.join(self._output_dir, task_file_name)
    self._task.save_to_file(task_path)
    file_path = os.path.join(self._output_dir, file_name)
    dictionary = {'epoch': self._epoch, 'avg_returns': self._avg_returns}
    for eval_t in self._eval_temperatures:
      dictionary['avg_returns_temperature_{}'.format(
          eval_t)] = self._avg_returns_temperatures[eval_t]
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      pickle.dump(dictionary, f)

  def init_from_file(self, file_name='rl.pkl',
                     task_file_name='trajectories.pkl'):
    """Initialize epoch number and average returns from file."""
    assert self._output_dir is not None
    task_path = os.path.join(self._output_dir, task_file_name)
    if tf.io.gfile.exists(task_path):
      self._task.init_from_file(task_path)
    file_path = os.path.join(self._output_dir, file_name)
    if not tf.io.gfile.exists(file_path):
      return
    with tf.io.gfile.GFile(file_path, 'rb') as f:
      dictionary = pickle.load(f)
    self._epoch = dictionary['epoch']
    self._avg_returns = dictionary['avg_returns']
    for eval_t in self._eval_temperatures:
      self._avg_returns_temperatures[eval_t] = dictionary[
          'avg_returns_temperature_{}'.format(eval_t)]

  def _collect_trajectories(self):
    return self.task.collect_trajectories(
        self.policy,
        n_trajectories=self._n_trajectories_per_epoch,
        n_interactions=self._n_interactions_per_epoch,
        only_eval=self._only_eval,
        epoch_id=self._epoch
    )

  def policy(self, trajectory, temperature=1.0):
    """Policy function that allows to play using this trainer.

    Args:
      trajectory: an instance of trax.rl.task.Trajectory
      temperature: temperature used to sample from the policy (default=1.0)

    Returns:
      a pair (action, dist_inputs) where action is the action taken and
      dist_inputs is the parameters of the policy distribution, that will later
      be used for training.
    """
    raise NotImplementedError

  def train_epoch(self):
    """Trains this Agent for one epoch -- main RL logic goes here."""
    raise NotImplementedError

  @contextlib.contextmanager
  def _open_summary_writer(self):
    """Opens the Jaxboard summary writer wrapped by a context manager.

    Yields:
      A Jaxboard summary writer wrapped in a GeneratorContextManager object.
      Elements of the lists correspond to the training and evaluation task
      directories created during initialization. If there is no output_dir
      provided, yields None.
    """
    if self._output_dir is not None:
      writer = jaxboard.SummaryWriter(os.path.join(self._output_dir, 'rl'))
      try:
        yield writer
      finally:
        writer.close()
    else:
      yield None

  def run(self, n_epochs=1, n_epochs_is_total_epochs=False):
    """Runs this loop for n epochs.

    Args:
      n_epochs: Stop training after completing n steps.
      n_epochs_is_total_epochs: if True, consider n_epochs as the total
        number of epochs to train, including previously trained ones
    """
    with self._open_summary_writer() as sw:
      n_epochs_to_run = n_epochs
      if n_epochs_is_total_epochs:
        n_epochs_to_run -= self._epoch
      cur_n_interactions = 0
      for _ in range(n_epochs_to_run):
        self._epoch += 1
        cur_time = time.time()
        avg_return = self._collect_trajectories()
        self._avg_returns.append(avg_return)
        if self._n_trajectories_per_epoch:
          supervised.trainer_lib.log(
              'Collecting %d episodes took %.2f seconds.'
              % (self._n_trajectories_per_epoch, time.time() - cur_time))
        else:
          supervised.trainer_lib.log(
              'Collecting %d interactions took %.2f seconds.'
              % (self._n_interactions_per_epoch, time.time() - cur_time))
        supervised.trainer_lib.log(
            'Average return in epoch %d was %.2f.' % (self._epoch, avg_return))
        if self._n_eval_episodes > 0:
          for steps in self._eval_steps:
            for eval_t in self._eval_temperatures:
              avg_return_temperature = self.task.collect_trajectories(
                  functools.partial(self.policy, temperature=eval_t),
                  n_trajectories=self._n_eval_episodes,
                  max_steps=steps,
                  only_eval=True)
              supervised.trainer_lib.log(
                  'Eval return in epoch %d with temperature %.2f was %.2f.'
                  % (self._epoch, eval_t, avg_return_temperature))
              self._avg_returns_temperatures[eval_t][steps].append(
                  avg_return_temperature)

        if sw is not None:
          sw.scalar('timing/collect', time.time() - cur_time,
                    step=self._epoch)
          sw.scalar('rl/avg_return', avg_return, step=self._epoch)
          if self._n_eval_episodes > 0:
            for steps in self._eval_steps:
              for eval_t in self._eval_temperatures:
                sw.scalar(
                    'rl/avg_return_temperature%.2f_steps%d' % (eval_t, steps),
                    self._avg_returns_temperatures[eval_t][steps][-1],
                    step=self._epoch)
          sw.scalar('rl/n_interactions', self.task.n_interactions(),
                    step=self._epoch)
          sw.scalar('rl/n_interactions_per_second',
                    (self.task.n_interactions() - cur_n_interactions)/ \
                    (time.time() - cur_time),
                    step=self._epoch)
          cur_n_interactions = self.task.n_interactions()
          sw.scalar('rl/n_trajectories', self.task.n_trajectories(),
                    step=self._epoch)
          sw.flush()

        cur_time = time.time()
        self.train_epoch()
        supervised.trainer_lib.log(
            'RL training took %.2f seconds.' % (time.time() - cur_time))

        if self._output_dir is not None and self._epoch == 1:
          self.save_gin(sw)
        if self._output_dir is not None:
          self.save_to_file()

  def close(self):
    pass


class PolicyAgent(Agent):
  """Agent that uses a deep learning model for policy.

  Many deep RL methods, such as policy gradient (REINFORCE) or actor-critic fall
  into this category, so a lot of classes will be subclasses of this one. But
  some methods only have a value or Q function, these are different.
  """

  def __init__(self, task, policy_model=None, policy_optimizer=None,
               policy_lr_schedule=lr.multifactor, policy_batch_size=64,
               policy_train_steps_per_epoch=500, policy_evals_per_epoch=1,
               policy_eval_steps=1, n_eval_episodes=0,
               only_eval=False, max_slice_length=1, output_dir=None, **kwargs):
    """Configures the policy trainer.

    Args:
      task: RLTask instance, which defines the environment to train on.
      policy_model: Trax layer, representing the policy model.
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      policy_optimizer: the optimizer to use to train the policy model.
      policy_lr_schedule: learning rate schedule to use to train the policy.
      policy_batch_size: batch size used to train the policy model.
      policy_train_steps_per_epoch: how long to train policy in each RL epoch.
      policy_evals_per_epoch: number of policy trainer evaluations per RL epoch
          - only affects metric reporting.
      policy_eval_steps: number of policy trainer steps per evaluation - only
          affects metric reporting.
      n_eval_episodes: number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
      **kwargs: arguments for the superclass Agent.
    """
    super().__init__(
        task,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        **kwargs
    )
    self._policy_batch_size = policy_batch_size
    self._policy_train_steps_per_epoch = policy_train_steps_per_epoch
    self._policy_evals_per_epoch = policy_evals_per_epoch
    self._policy_eval_steps = policy_eval_steps
    self._only_eval = only_eval
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)

    # Inputs to the policy model are produced by self._policy_batches_stream.
    self._policy_inputs = data.inputs.Inputs(
        train_stream=lambda _: self.policy_batches_stream())

    policy_model = functools.partial(
        policy_model,
        policy_distribution=self._policy_dist,
    )

    # This is the policy Trainer that will be used to train the policy model.
    # * inputs to the trainer come from self.policy_batches_stream
    # * outputs, targets and weights are passed to self.policy_loss
    self._policy_trainer = supervised.Trainer(
        model=policy_model,
        optimizer=policy_optimizer,
        lr_schedule=policy_lr_schedule(),
        loss_fn=self.policy_loss,
        inputs=self._policy_inputs,
        output_dir=output_dir,
        metrics=self.policy_metrics,
    )
    self._policy_collect_model = tl.Accelerate(
        policy_model(mode='collect'), n_devices=1)
    policy_batch = next(self.policy_batches_stream())
    self._policy_collect_model.init(shapes.signature(policy_batch))
    self._policy_eval_model = tl.Accelerate(
        policy_model(mode='eval'), n_devices=1)  # Not collecting stats
    self._policy_eval_model.init(shapes.signature(policy_batch))

  @property
  def policy_loss(self):
    """Policy loss."""
    return NotImplementedError

  @property
  def policy_metrics(self):
    return {'policy_loss': self.policy_loss}

  def policy_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    return NotImplementedError

  def policy(self, trajectory, temperature=1.0):
    """Chooses an action to play after a trajectory."""
    model = self._policy_collect_model
    if temperature != 1.0:  # When evaluating (t != 1.0), don't collect stats
      model = self._policy_eval_model
      model.state = self._policy_collect_model.state
    model.replicate_weights(self._policy_trainer.model_weights)
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observation[None, ...])
    # Pick element 0 from the batch (the only one), last (current) timestep.
    pred = pred[0, -1, :]
    sample = self._policy_dist.sample(pred, temperature=temperature)
    result = (sample, pred)
    if fastmath.is_backend(fastmath.Backend.JAX):
      result = fastmath.nested_map(lambda x: x.copy(), result)
    return result

  def train_epoch(self):
    """Trains RL for one epoch."""
    # When restoring, calculate how many evals are remaining.
    n_evals = remaining_evals(
        self._policy_trainer.step,
        self._epoch,
        self._policy_train_steps_per_epoch,
        self._policy_evals_per_epoch)
    for _ in range(n_evals):
      self._policy_trainer.train_epoch(
          self._policy_train_steps_per_epoch // self._policy_evals_per_epoch,
          self._policy_eval_steps)

  def close(self):
    self._policy_trainer.close()
    super().close()


def remaining_evals(cur_step, epoch, train_steps_per_epoch, evals_per_epoch):
  """Helper function to calculate remaining evaluations for a trainer.

  Args:
    cur_step: current step of the supervised trainer
    epoch: current epoch of the RL trainer
    train_steps_per_epoch: supervised trainer steps per RL epoch
    evals_per_epoch: supervised trainer evals per RL epoch

  Returns:
    number of remaining evals to do this epoch

  Raises:
    ValueError if the provided numbers indicate a step mismatch
  """
  if epoch < 1:
    raise ValueError('Epoch must be at least 1, got %d' % epoch)
  prev_steps = (epoch - 1) * train_steps_per_epoch
  done_steps_this_epoch = cur_step - prev_steps
  if done_steps_this_epoch < 0:
    raise ValueError('Current step (%d) < previously done steps (%d).'
                     % (cur_step, prev_steps))
  train_steps_per_eval = train_steps_per_epoch // evals_per_epoch
  if done_steps_this_epoch % train_steps_per_eval != 0:
    raise ValueError('Done steps (%d) must divide train steps per eval (%d).'
                     % (done_steps_this_epoch, train_steps_per_eval))
  return evals_per_epoch - (done_steps_this_epoch // train_steps_per_eval)


class LoopPolicyAgent(Agent):
  """Base class for policy-only Agents based on Loop."""

  def __init__(
      self,
      task,
      model_fn,
      value_fn,
      weight_fn,
      n_replay_epochs,
      n_train_steps_per_epoch,
      advantage_normalization,
      optimizer=adam.Adam,
      lr_schedule=lr.multifactor,
      batch_size=64,
      network_eval_at=None,
      n_eval_batches=1,
      max_slice_length=1,
      trajectory_stream_preprocessing_fn=None,
      **kwargs
  ):
    """Initializes LoopPolicyAgent.

    Args:
      task: Instance of trax.rl.task.RLTask.
      model_fn: Function (policy_distribution, mode) -> policy_model.
      value_fn: Function TimeStepBatch -> array (batch_size, seq_len)
        calculating the baseline for advantage calculation.
      weight_fn: Function float -> float to apply to advantages when calculating
        policy loss.
      n_replay_epochs: Number of last epochs to take into the replay buffer;
        only makes sense for off-policy algorithms.
      n_train_steps_per_epoch: Number of steps to train the policy network for
        in each epoch.
      advantage_normalization: Whether to normalize the advantages before
        passing them to weight_fn.
      optimizer: Optimizer for network training.
      lr_schedule: Learning rate schedule for network training.
      batch_size: Batch size for network training.
      network_eval_at: Function step -> bool indicating the training steps, when
        network evaluation should be performed.
      n_eval_batches: Number of batches to run during network evaluation.
      max_slice_length: The length of trajectory slices to run the network on.
      trajectory_stream_preprocessing_fn: Function to apply to the trajectory
        stream before batching. Can be used e.g. to filter trajectories.
      **kwargs: Keyword arguments passed to the superclass.
    """
    self._n_train_steps_per_epoch = n_train_steps_per_epoch
    super().__init__(task, **kwargs)

    task.set_n_replay_epochs(n_replay_epochs)
    self._max_slice_length = max_slice_length
    trajectory_batch_stream = task.trajectory_batch_stream(
        batch_size,
        epochs=[-(ep + 1) for ep in range(n_replay_epochs)],
        max_slice_length=self._max_slice_length,
        sample_trajectories_uniformly=True,
        trajectory_stream_preprocessing_fn=trajectory_stream_preprocessing_fn,
    )
    self._policy_dist = distributions.create_distribution(task.action_space)
    train_task = policy_tasks.PolicyTrainTask(
        trajectory_batch_stream,
        optimizer(),
        lr_schedule(),
        self._policy_dist,
        # Without a value network it doesn't make a lot of sense to use
        # a better advantage estimator than MC.
        advantage_estimator=advantages.monte_carlo(task.gamma, margin=0),
        advantage_normalization=advantage_normalization,
        value_fn=value_fn,
        weight_fn=weight_fn,
    )
    eval_task = policy_tasks.PolicyEvalTask(train_task, n_eval_batches)
    model_fn = functools.partial(
        model_fn,
        policy_distribution=self._policy_dist,
    )

    if self._output_dir is not None:
      policy_output_dir = os.path.join(self._output_dir, 'policy')
    else:
      policy_output_dir = None
    # Checkpoint every epoch.
    checkpoint_at = lambda step: step % n_train_steps_per_epoch == 0
    self._loop = supervised.training.Loop(
        model=model_fn(mode='train'),
        tasks=[train_task],
        eval_model=model_fn(mode='eval'),
        eval_tasks=[eval_task],
        output_dir=policy_output_dir,
        eval_at=network_eval_at,
        checkpoint_at=checkpoint_at,
    )
    self._collect_model = model_fn(mode='collect')
    self._collect_model.init(shapes.signature(train_task.sample_batch))

    # Validate the restored checkpoints.
    # TODO(pkozakowski): Move this to the base class once all Agents use Loop.
    if self._loop.step != self._epoch * self._n_train_steps_per_epoch:
      raise ValueError(
          'The number of Loop steps must equal the number of Agent epochs '
          'times the number of steps per epoch, got {}, {} and {}.'.format(
              self._loop.step, self._epoch, self._n_train_steps_per_epoch
          )
      )

  @property
  def loop(self):
    """Loop exposed for testing."""
    return self._loop

  def train_epoch(self):
    """Trains RL for one epoch."""
    # Copy policy state accumulated during data collection to the trainer.
    self._loop.update_weights_and_state(state=self._collect_model.state)
    # Train for the specified number of steps.
    self._loop.run(n_steps=self._n_train_steps_per_epoch)


class PolicyGradient(LoopPolicyAgent):
  """Trains a policy model using policy gradient on the given RLTask."""

  def __init__(self, task, model_fn, **kwargs):
    """Initializes PolicyGradient.

    Args:
      task: Instance of trax.rl.task.RLTask.
      model_fn: Function (policy_distribution, mode) -> policy_model.
      **kwargs: Keyword arguments passed to the superclass.
    """
    super().__init__(
        task, model_fn,
        # We're on-policy, so we can only use data from the last epoch.
        n_replay_epochs=1,
        # Each gradient computation needs a new data sample, so we do 1 step
        # per epoch.
        n_train_steps_per_epoch=1,
        # Very simple baseline: mean return across trajectories.
        value_fn=self._value_fn,
        # Weights are just advantages.
        weight_fn=(lambda x: x),
        # Normalize advantages, because this makes optimization nicer.
        advantage_normalization=True,
        **kwargs
    )

  def policy(self, trajectory, temperature=1.0):
    """Policy function that samples from the trained network."""
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    return network_policy(
        collect_model=self._collect_model,
        policy_distribution=self._policy_dist,
        loop=self.loop,
        trajectory_np=trajectory_np,
        temperature=temperature,
    )

  @staticmethod
  def _value_fn(trajectory_batch):
    # Estimate the value of every state as the mean return across trajectories
    # and timesteps in a batch.
    value = np.mean(trajectory_batch.return_)
    return np.broadcast_to(value, trajectory_batch.return_.shape)


@gin.configurable
def sharpened_network_policy(
    temperature,
    temperature_multiplier=1.0,
    **kwargs
):
  """Expert function that runs a policy network with lower temperature.

  Args:
    temperature: Temperature passed from the Agent.
    temperature_multiplier: Multiplier to apply to the temperature to "sharpen"
      the policy distribution. Should be <= 1, but this is not a requirement.
    **kwargs: Keyword arguments passed to network_policy.

  Returns:
    Pair (action, dist_inputs) where action is the action taken and dist_inputs
    is the parameters of the policy distribution, that will later be used for
    training.
  """
  return network_policy(
      temperature=(temperature_multiplier * temperature),
      **kwargs
  )


class ExpertIteration(LoopPolicyAgent):
  """Trains a policy model using expert iteration with a given expert."""

  def __init__(
      self, task, model_fn,
      expert_policy_fn=sharpened_network_policy,
      quantile=0.9,
      n_replay_epochs=10,
      n_train_steps_per_epoch=1000,
      filter_buffer_size=256,
      **kwargs
  ):
    """Initializes ExpertIteration.

    Args:
      task: Instance of trax.rl.task.RLTask.
      model_fn: Function (policy_distribution, mode) -> policy_model.
      expert_policy_fn: Function of the same signature as `network_policy`, to
        be used as an expert. The policy will be trained to mimic the expert on
        the "solved" trajectories.
      quantile: Quantile of best trajectories to be marked as "solved". They
        will be used to train the policy.
      n_replay_epochs: Number of last epochs to include in the replay buffer.
      n_train_steps_per_epoch: Number of policy training steps to run in each
        epoch.
      filter_buffer_size: Number of trajectories in the trajectory filter
        buffer, used to select the best trajectories based on the quantile.
      **kwargs: Keyword arguments passed to the superclass.
    """
    self._expert_policy_fn = expert_policy_fn
    self._quantile = quantile
    self._filter_buffer_size = filter_buffer_size
    super().__init__(
        task, model_fn,
        # Don't use a baseline - it's not useful in our weights.
        value_fn=(lambda batch: jnp.zeros_like(batch.return_)),
        # Don't weight trajectories - the training signal is provided by
        # filtering trajectories.
        weight_fn=jnp.ones_like,
        # Filter trajectories based on the quantile.
        trajectory_stream_preprocessing_fn=self._filter_trajectories,
        # Advantage normalization is a no-op here.
        advantage_normalization=False,
        n_replay_epochs=n_replay_epochs,
        n_train_steps_per_epoch=n_train_steps_per_epoch,
        **kwargs
    )

  def policy(self, trajectory, temperature=1.0):
    """Policy function that runs the expert."""
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    return self._expert_policy_fn(
        collect_model=self._collect_model,
        policy_distribution=self._policy_dist,
        loop=self.loop,
        trajectory_np=trajectory_np,
        temperature=temperature,
    )

  def _filter_trajectories(self, trajectory_stream):
    """Filter trajectories based on the quantile."""
    def trajectory_return(trajectory):
      return trajectory.timesteps[0].return_

    trajectory_buffer = []
    for trajectory in trajectory_stream:
      trajectory_buffer.append(trajectory)
      if len(trajectory_buffer) == self._filter_buffer_size:
        n_best = int((1 - self._quantile) * self._filter_buffer_size) or 1
        trajectory_buffer.sort(key=trajectory_return, reverse=True)
        yield from trajectory_buffer[:n_best]
        trajectory_buffer.clear()


def network_policy(
    collect_model,
    policy_distribution,
    loop,
    trajectory_np,
    head_index=0,
    temperature=1.0,
):
  """Policy function powered by a neural network.

  Used to implement Agent.policy() in policy-based agents.

  Args:
    collect_model: the model used for collecting trajectories
    policy_distribution: an instance of trax.rl.distributions.Distribution
    loop: trax.supervised.training.Loop used to train the policy network
    trajectory_np: an instance of trax.rl.task.TimeStepBatch
    head_index: index of the policy head a multihead model.
    temperature: temperature used to sample from the policy (default=1.0)

  Returns:
    a pair (action, dist_inputs) where action is the action taken and
    dist_inputs is the parameters of the policy distribution, that will later
    be used for training.
  """
  if temperature == 1.0:
    model = collect_model
  else:
    # When evaluating (t != 1.0), use the evaluation model instead of the
    # collection model - some models accumulate normalization statistics
    # during data collection, and we don't want to do it in eval to avoid data
    # leakage.
    model = loop.eval_model
    model.state = collect_model.state
  # Copying weights from loop.model should work, because the raw model's
  # weights should be updated automatically during training, but it doesn't.
  # TODO(pkozakowski): Debug.
  acc = loop._trainer_per_task[0].accelerated_model_with_loss  # pylint: disable=protected-access
  model.weights = acc._unreplicate(acc.weights[0])  # pylint: disable=protected-access
  # Add batch dimension to trajectory_np and run the model.
  pred = model(trajectory_np.observation[None, ...])
  if isinstance(pred, (tuple, list)):
    # For multihead models, extract the policy head output.
    pred = pred[head_index]
  assert pred.shape == (
      1, trajectory_np.observation.shape[0], policy_distribution.n_inputs
  )
  # Pick element 0 from the batch (the only one), last (current) timestep.
  pred = pred[0, -1, :]
  sample = policy_distribution.sample(pred, temperature=temperature)
  result = (sample, pred)
  if fastmath.is_backend(fastmath.Backend.JAX):
    # The result is composed of mutable numpy arrays. We copy them to avoid
    # accidental modification.
    result = fastmath.nested_map(lambda x: x.copy(), result)
  return result


class ValueAgent(Agent):
  """Trainer that uses a deep learning model for value function.

  Compute the loss using variants of the Bellman equation.
  """

  def __init__(self, task,
               value_body=None,
               value_optimizer=None,
               value_lr_schedule=lr.multifactor,
               value_batch_size=64,
               value_train_steps_per_epoch=500,
               value_evals_per_epoch=1,
               value_eval_steps=1,
               exploration_rate=functools.partial(
                   lr.multifactor,
                   factors='constant * decay_every',
                   constant=1.,  # pylint: disable=redefined-outer-name
                   decay_factor=0.99,
                   steps_per_decay=1,
                   minimum=0.1),
               n_eval_episodes=0,
               only_eval=False,
               n_replay_epochs=1,
               max_slice_length=1,
               sync_freq=1000,
               scale_value_targets=True,
               output_dir=None,
               **kwargs):
    """Configures the value trainer.

    Args:
      task: RLTask instance, which defines the environment to train on.
      value_body: Trax layer, representing the body of the value model.
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      value_optimizer: the optimizer to use to train the policy model.
      value_lr_schedule: learning rate schedule to use to train the policy.
      value_batch_size: batch size used to train the policy model.
      value_train_steps_per_epoch: how long to train policy in each RL epoch.
      value_evals_per_epoch: number of policy trainer evaluations per RL epoch
          - only affects metric reporting.
      value_eval_steps: number of policy trainer steps per evaluation - only
          affects metric reporting.
      exploration_rate: exploration rate schedule - used in the policy method.
      n_eval_episodes: number of episodes to play with policy at
        temperature 0 in each epoch -- used for evaluation only
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      n_replay_epochs: Number of last epochs to take into the replay buffer;
          only makes sense for off-policy algorithms.
      max_slice_length: the maximum length of trajectory slices to use; it is
          the second dimenions of the value network output:
          (batch, max_slice_length, number of actions)
          Higher max_slice_length implies that the network has to predict more
          values into the future.
      sync_freq: frequency when to synchronize the target
        network with the trained network. This is necessary for training the
        network on bootstrapped targets, e.g. using n-step returns.
      scale_value_targets: If `True`, scale value function targets by
          `1 / (1 - gamma)`. We are trying to fix the problem with very large
          returns in some games in a way which does not introduce an additional
          hyperparameters.
      output_dir: Path telling where to save outputs (evals and checkpoints).
      **kwargs: arguments for the superclass RLTrainer.
    """
    super(ValueAgent, self).__init__(
        task,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        **kwargs
    )
    self._value_batch_size = value_batch_size
    self._value_train_steps_per_epoch = value_train_steps_per_epoch
    self._value_evals_per_epoch = value_evals_per_epoch
    self._value_eval_steps = value_eval_steps
    self._only_eval = only_eval
    self._max_slice_length = max_slice_length
    self._policy_dist = distributions.create_distribution(task.action_space)
    self._n_replay_epochs = n_replay_epochs

    self._exploration_rate = exploration_rate()
    self._sync_at = (lambda step: step % sync_freq == 0)

    if scale_value_targets:
      self._value_network_scale = 1 / (1 - self._task.gamma)
    else:
      self._value_network_scale = 1

    value_model = functools.partial(
        models.Quality,
        body=value_body,
        n_actions=self.task.action_space.n)

    self._value_eval_model = value_model(mode='eval')
    self._value_eval_model.init(self._value_model_signature)
    self._value_eval_jit = tl.jit_forward(
        self._value_eval_model.pure_fn, fastmath.local_device_count(),
        do_mean=False)

    # Inputs to the value model are produced by self._values_batches_stream.
    self._inputs = data.inputs.Inputs(
        train_stream=lambda _: self.value_batches_stream())

    # This is the value Trainer that will be used to train the value model.
    # * inputs to the trainer come from self.value_batches_stream
    # * outputs, targets and weights are passed to self.value_loss
    self._value_trainer = supervised.Trainer(
        model=value_model,
        optimizer=value_optimizer,
        lr_schedule=value_lr_schedule(),
        loss_fn=self.value_loss,
        inputs=self._inputs,
        output_dir=output_dir,
        metrics={'value_loss': self.value_loss,
                 'value_mean': self.value_mean,
                 'returns_mean': self.returns_mean}
    )
    value_batch = next(self.value_batches_stream())
    self._eval_model = tl.Accelerate(
        value_model(mode='collect'), n_devices=1)
    self._eval_model.init(shapes.signature(value_batch))

  @property
  def _value_model_signature(self):
    obs_sig = shapes.signature(self._task.observation_space)
    target_sig = mask_sig = shapes.ShapeDtype(
        shape=(1, 1, self._task.action_space),
    )
    inputs_sig = obs_sig.replace(shape=(1, 1) + obs_sig.shape)
    return (inputs_sig, target_sig, mask_sig)

  def value_batches_stream(self):
    """Use self.task to create inputs to the policy model."""
    raise NotImplementedError

  def policy(self, trajectory, temperature=1):
    """Chooses an action to play after a trajectory."""
    raise NotImplementedError

  def train_epoch(self):
    """Trains RL for one epoch."""
    # Update the target value network.
    self._value_eval_model.weights = self._value_trainer.model_weights
    self._value_eval_model.state = self._value_trainer.model_state

    # When restoring, calculate how many evals are remaining.
    n_evals = remaining_evals(
        self._value_trainer.step,
        self._epoch,
        self._value_train_steps_per_epoch,
        self._value_evals_per_epoch)
    for _ in range(n_evals):
      self._value_trainer.train_epoch(
          self._value_train_steps_per_epoch // self._value_evals_per_epoch,
          self._value_eval_steps)
      value_metrics = dict(
          {'exploration_rate': self._exploration_rate(self._epoch)})
      self._value_trainer.log_metrics(value_metrics,
                                      self._value_trainer._train_sw, 'dqn')  # pylint: disable=protected-access
    # Update the target value network.
    # TODO(henrykm) a bit tricky if sync_at does not coincide with epochs
    if self._sync_at(self._value_trainer.step):
      self._value_eval_model.weights = self._value_trainer.model_weights
      self._value_eval_model.state = self._value_trainer.model_state

  def close(self):
    self._value_trainer.close()
    super().close()

  @property
  def value_mean(self):
    """The mean value of actions selected by the behavioral policy."""
    raise NotImplementedError

  @property
  def returns_mean(self):
    """The mean value of actions selected by the behavioral policy."""
    def f(values, index_max, returns, mask):
      del values, index_max
      return jnp.sum(returns) / jnp.sum(mask)
    return tl.Fn('ReturnsMean', f)


class DQN(ValueAgent):
  r"""Trains a value model using DQN on the given RLTask.

  Notice that the algorithm and the parameters signficantly diverge from
  the original DQN paper. In particular we have separated learning and data
  collection.

  The Bellman loss is computed in the value_loss method. The formula takes
  the state-action values tensors Q and n-step returns R:

    .. math::
        L(s,a) = Q(s,a) - R(s,a)

  where R is computed in value_batches_stream. In the simplest case of the
  1-step returns we are getting

    .. math::
        L(s,a) = Q(s,a) - r(s,a) - gamma * \max_{a'} Q'(s',a')

  where s' is the state reached after taking action a in state s, Q' is
  the target network, gamma is the discount factor and the maximum is taken
  with respect to all actions avaliable in the state s'. The tensor Q' is
  updated using the sync_freq parameter.

  In code the maximum is visible in the policy method where we take
  sample = jnp.argmax(values). The epsilon-greedy policy is taking a random
  move with probability epsilon and oterhwise in state s it takes the
  action argmax_a Q(s,a).
  """

  def __init__(self,
               task,
               advantage_estimator=advantages.monte_carlo,
               max_slice_length=1,
               smoothl1loss=True,
               double_dqn=False,
               **kwargs):

    self._max_slice_length = max_slice_length
    self._margin = max_slice_length-1
    # Our default choice of learning targets for DQN are n-step targets
    # implemented in the method td_k. We set the slice used for computation
    # of td_k to max_slice_length and we set the "margin" in td_k
    # to self._max_slice_length-1; in turn it implies that the shape of the
    # returned tensor of n-step targets is
    # values[:, :-(self.margin)] =  values[:, :1]
    self._advantage_estimator = advantage_estimator(
        gamma=task.gamma, margin=self._margin)
    self._smoothl1loss = smoothl1loss
    self._double_dqn = double_dqn
    super(DQN, self).__init__(task=task,
                              max_slice_length=max_slice_length,
                              **kwargs)

  @property
  def value_loss(self):
    """Value loss computed using smooth L1 loss or L2 loss."""
    def f(values, actions, returns, mask):
      ind_0, ind_1 = np.indices(actions.shape)
      # We calculate length using the shape of returns
      # and adequatly remove a superflous slice of values.
      # An analogous operation is done in value_batches_stream.
      length = returns.shape[1]
      values = values[:, :length, :]
      selected_values = values[ind_0, ind_1, actions]
      shapes.assert_same_shape(selected_values, returns)
      shapes.assert_same_shape(selected_values, mask)
      if self._smoothl1loss:
        return tl.SmoothL1Loss().forward((selected_values, returns, mask))
      else:
        return tl.L2Loss().forward((selected_values, returns, mask))
    return tl.Fn('ValueLoss', f)

  @property
  def _replay_epochs(self):
    return [-(ep + 1) for ep in range(self._n_replay_epochs)]

  def value_batches_stream(self):
    """Use the RLTask self._task to create inputs to the value model."""
    max_slice_length = self._max_slice_length
    min_slice_length = 1
    for np_trajectory in self._task.trajectory_batch_stream(
        self._value_batch_size,
        max_slice_length=max_slice_length,
        min_slice_length=min_slice_length,
        margin=0,
        epochs=self._replay_epochs,
    ):
      values_target = self._run_value_model(
          np_trajectory.observation, use_eval_model=True)
      if self._double_dqn:
        values = self._run_value_model(
            np_trajectory.observation, use_eval_model=False
        )
        index_max = np.argmax(values, axis=-1)
        ind_0, ind_1 = np.indices(index_max.shape)
        values_max = values_target[ind_0, ind_1, index_max]
      else:
        values_max = np.array(jnp.max(values_target, axis=-1))

      # The advantage_estimator returns
      #     gamma^n_steps * values_max(s_{i + n_steps}) + discounted_rewards
      #        - values_max(s_i)
      # hence we have to add values_max(s_i) in order to get n-step returns:
      #     gamma^n_steps * values_max(s_{i + n_steps}) + discounted_rewards
      # Notice, that in DQN the tensor values_max[:, :-self._margin]
      # is the same as values_max[:, :-1].
      n_step_returns = values_max[:, :-self._margin] + \
          self._advantage_estimator(
              rewards=np_trajectory.reward,
              returns=np_trajectory.return_,
              values=values_max,
              dones=np_trajectory.done,
              discount_mask=np_trajectory.env_info.discount_mask,
          )

      length = n_step_returns.shape[1]
      target_returns = n_step_returns[:, :length]
      inputs = np_trajectory.observation[:, :length, :]

      yield (
          # Inputs are observations
          # (batch, length, obs)
          inputs,
          # the max indices will be needed to compute the loss
          np_trajectory.action[:, :length],  # index_max,
          # Targets: computed returns.
          # target_returns, we expect here shapes such as
          # (batch, length, num_actions)
          target_returns / self._value_network_scale,
          # TODO(henrykm): mask has the shape (batch, max_slice_length)
          # that is it misses the action dimension; the preferred format
          # would be np_trajectory.mask[:, :length, :] but for now we pass:
          np.ones(shape=target_returns.shape),
      )

  def policy(self, trajectory, temperature=1):
    """Chooses an action to play after a trajectory."""
    tr_slice = trajectory.suffix(self._max_slice_length)
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    obs = trajectory_np.observation[None, ...]
    values = self._run_value_model(obs, use_eval_model=False)
    # We insisit that values and observations have the shape
    # (batch, length, ...), where the length is the number of subsequent
    # observations on a given trajectory
    assert values.shape[:1] == obs.shape[:1]
    # We select the last element in the batch and the value
    # related to the last (current) observation
    values = values[0, -1, :]
    # temperature == 0 is used in another place in order to trigger eval
    if np.random.random_sample() < self._exploration_rate(self._epoch) and \
        temperature == 1:
      sample = np.array(self.task.action_space.sample())
    else:
      # this is our way of doing the argmax
      sample = jnp.argmax(values)
    result = (sample, values)
    if fastmath.backend_name() == 'jax':
      result = fastmath.nested_map(lambda x: x.copy(), result)
    return result

  def _run_value_model(self, obs, use_eval_model=True):
    """Runs value model."""
    n_devices = fastmath.local_device_count()
    if use_eval_model:
      weights = tl.for_n_devices(self._value_eval_model.weights, n_devices)
      state = tl.for_n_devices(self._value_eval_model.state, n_devices)
      rng = self._value_eval_model.rng
    else:
      # TODO(henrykm): this strangely looking solution address the problem that
      # value_batches_stream calls _run_value_model _once_ before
      # the trainer is initialized.
      try:
        weights = tl.for_n_devices(self._value_trainer.model_weights, n_devices)
        state = tl.for_n_devices(self._value_trainer.model_state, n_devices)
        rng = self._value_trainer._rng  # pylint: disable=protected-access
      except AttributeError:
        weights = tl.for_n_devices(self._value_eval_model.weights, n_devices)
        state = tl.for_n_devices(self._value_eval_model.state, n_devices)
        rng = self._value_eval_model.rng
    # TODO(henrykm): the line below fails on TPU with the error
    # ValueError: Number of devices (8) does not evenly divide batch size (1).
    obs_batch = obs.shape[0]
    if n_devices > obs_batch:
      obs = jnp.repeat(obs, int(n_devices / obs_batch), axis=0)
    values, _ = self._value_eval_jit(obs, weights, state, rng)
    values = values[:obs_batch]
    values *= self._value_network_scale
    return values

  @property
  def value_mean(self):
    """The mean value of actions selected by the behavioral policy."""
    def f(values, actions, returns, mask):
      ind_0, ind_1 = np.indices(actions.shape)
      # We calculate length using the shape of returns
      # and adequatly remove a superflous slice of values.
      # An analogous operation is done in value_batches_stream.
      length = returns.shape[1]
      values = values[:, :length, :]
      selected_values = values[ind_0, ind_1, actions]
      shapes.assert_same_shape(selected_values, returns)
      shapes.assert_same_shape(selected_values, mask)
      return jnp.sum(selected_values) / jnp.sum(mask)
    return tl.Fn('ValueMean', f)
