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
from trax import shapes
from trax import supervised
from trax.optimizers import adam
from trax.rl import advantages
from trax.rl import distributions
from trax.rl import normalization  # So gin files see it. # pylint: disable=unused-import
from trax.rl import policy_tasks
from trax.rl import task as rl_task
from trax.supervised import lr_schedules as lr


class Agent:
  """Abstract class for RL agents, presenting the required API."""

  def __init__(self, task: rl_task.RLTask,
               n_trajectories_per_epoch=None,
               n_interactions_per_epoch=None,
               n_eval_episodes=0,
               eval_steps=None,
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
      only_eval: If set to True, then trajectories are collected only for
        for evaluation purposes, but they are not recorded.
      output_dir: Path telling where to save outputs such as checkpoints.
      timestep_to_np: Timestep-to-numpy function to override in the task.
    """
    assert bool(n_trajectories_per_epoch) != bool(n_interactions_per_epoch), (
        'Exactly one of n_trajectories_per_epoch or n_interactions_per_epoch '
        'should be specified.'
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
    self._avg_returns_temperature0 = {step: [] for step in self._eval_steps}
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
    dictionary = {'epoch': self._epoch,
                  'avg_returns': self._avg_returns,
                  'avg_returns_temperature0': self._avg_returns_temperature0}
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
    self._avg_returns_temperature0 = dictionary['avg_returns_temperature0']

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
        self.train_epoch()
        supervised.trainer_lib.log(
            'RL training took %.2f seconds.' % (time.time() - cur_time))
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
            avg_return_temperature0 = self.task.collect_trajectories(
                lambda x: self.policy(x, temperature=0.0),
                n_trajectories=self._n_eval_episodes,
                max_steps=steps, only_eval=True)
            self._avg_returns_temperature0[steps].append(
                avg_return_temperature0
            )
            supervised.trainer_lib.log(
                'Avg return with temperature 0 at %d steps in epoch %d was %.2f.'
                % (steps, self._epoch, avg_return_temperature0))
        if sw is not None:
          sw.scalar('timing/collect', time.time() - cur_time,
                    step=self._epoch)
          sw.scalar('rl/avg_return', avg_return, step=self._epoch)
          if self._n_eval_episodes > 0:
            for steps in self._eval_steps:
              sw.scalar('rl/avg_return_temperature0_steps%d' % steps,
                        self._avg_returns_temperature0[steps][-1],
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
    if self._task._initial_trajectories == 0:
      self._task.remove_epoch(0)
      self._collect_trajectories()

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
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    # Add batch dimension to trajectory_np and run the model.
    pred = model(trajectory_np.observations[None, ...])
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


class PolicyGradient(Agent):
  """Trains a policy model using policy gradient on the given RLTask."""

  def __init__(
      self, task, model_fn,
      optimizer=adam.Adam,
      lr_schedule=lr.multifactor,
      batch_size=64,
      network_eval_at=None,
      n_eval_batches=1,
      max_slice_length=1,
      **kwargs
  ):
    """Initializes PolicyGradient.

    Args:
      task: Instance of trax.rl.task.RLTask.
      model_fn: Function (policy_distribution, mode) -> policy_model.
      optimizer: Optimizer for network training.
      lr_schedule: Learning rate schedule for network training.
      batch_size: Batch size for network training.
      network_eval_at: Function step -> bool indicating the training steps, when
        network evaluation should be performed.
      n_eval_batches: Number of batches to run during network evaluation.
      max_slice_length: The length of trajectory slices to run the network on.
      **kwargs: Keyword arguments passed to the superclass.
    """
    super().__init__(task, **kwargs)

    self._max_slice_length = max_slice_length
    trajectory_batch_stream = task.trajectory_batch_stream(
        batch_size,
        epochs=[-1],
        max_slice_length=self._max_slice_length,
        sample_trajectories_uniformly=True,
    )
    self._policy_dist = distributions.create_distribution(task.action_space)
    train_task = policy_tasks.PolicyTrainTask(
        trajectory_batch_stream,
        optimizer(),
        lr_schedule(),
        self._policy_dist,
        # Policy gradient uses the MC estimator.
        advantage_estimator=advantages.monte_carlo,
        # No need for margin - the MC estimator only uses empirical returns.
        margin=0,
        value_fn=self._value_fn,
        gamma=task.gamma,
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
    # Checkpoint every epoch. We do one step per epoch, so that's every step.
    checkpoint_at = lambda _: True
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

    # Validate the restored checkpoints. The number of network training steps
    # (self.loop.step) should be equal to the number of epochs (self._epoch),
    # because we do exactly one gradient step per epoch.
    # TODO(pkozakowski): Move this to the base class once all Agents use Loop.
    if self.loop.step != self._epoch:
      raise ValueError(
          'The number of Loop steps must equal the number of Agent epochs, '
          'got {} and {}.'.format(self.loop.step, self._epoch)
      )

  @property
  def loop(self):
    """Loop exposed for testing."""
    return self._loop

  def policy(self, trajectory, temperature=1.0):
    """Policy function that allows to play using this agent."""
    tr_slice = trajectory[-self._max_slice_length:]
    trajectory_np = tr_slice.to_np(timestep_to_np=self.task.timestep_to_np)
    return network_policy(
        collect_model=self._collect_model,
        policy_distribution=self._policy_dist,
        loop=self.loop,
        trajectory_np=trajectory_np,
        temperature=temperature,
    )

  def train_epoch(self):
    """Trains RL for one epoch."""
    # Perform one gradient step per training epoch to ensure we stay on policy.
    self._loop.run(n_steps=1)

  @staticmethod
  def _value_fn(trajectory_batch):
    # Estimate the value of every state as the mean return across trajectories
    # and timesteps in a batch.
    value = np.mean(trajectory_batch.returns)
    return np.broadcast_to(value, trajectory_batch.returns.shape)


def network_policy(
    collect_model, policy_distribution, loop, trajectory_np, temperature=1.0
):
  """Policy function powered by a neural network.

  Used to implement Agent.policy() in policy-based agents.

  Args:
    collect_model: the model used for collecting trajectories
    policy_distribution: an instance of trax.rl.distributions.Distribution
    loop: trax.supervised.training.Loop used to train the policy network
    trajectory_np: an instance of trax.rl.task.TrajectoryNp
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
  acc = loop._trainer_per_task[0].accelerated_loss_layer  # pylint: disable=protected-access
  model.weights = acc._unreplicate(acc.weights[0])  # pylint: disable=protected-access
  # Add batch dimension to trajectory_np and run the model.
  pred = model(trajectory_np.observations[None, ...])
  assert pred.shape == (
      1, trajectory_np.observations.shape[0], policy_distribution.n_inputs
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
