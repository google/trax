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
"""Tests for RL training."""

import functools
import math

from absl.testing import absltest

from trax import layers as tl
from trax import models
from trax import optimizers as opt
from trax import test_utils
from trax.rl import actor_critic
from trax.rl import advantages
from trax.rl import task as rl_task
from trax.supervised import lr_schedules


class ActorCriticTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_a2ctrainer_save_restore(self):
    """Check save and restore of A2C trainer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=0,
                          max_steps=20)
    body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    tmp_dir = self.create_tempdir().full_path
    trainer1 = actor_critic.A2CTrainer(
        task,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_batch_size=2,
        value_train_steps_per_epoch=1,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        n_shared_layers=1,
        output_dir=tmp_dir)
    trainer1.run(2)
    self.assertEqual(trainer1.current_epoch, 2)
    self.assertEqual(trainer1._value_trainer.step, 2)
    self.assertEqual(trainer1._policy_trainer.step, 4)
    # Trainer 2 starts where trainer 1 stopped.
    trainer2 = actor_critic.A2CTrainer(
        task,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_batch_size=2,
        value_train_steps_per_epoch=1,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        n_shared_layers=1,
        output_dir=tmp_dir)
    trainer2.run(1)
    self.assertEqual(trainer2.current_epoch, 3)
    self.assertEqual(trainer2._value_trainer.step, 3)
    self.assertEqual(trainer2._policy_trainer.step, 6)
    trainer1.close()
    trainer2.close()

  def test_sanity_a2ctrainer_cartpole(self):
    """Test-runs a2c on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=0,
                          max_steps=2)
    body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-4, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.A2CTrainer(
        task,
        n_shared_layers=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=2,
        value_train_steps_per_epoch=2,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_sanity_ppo_cartpole(self):
    """Run PPO and check whether it correctly runs for 2 epochs.s."""
    task = rl_task.RLTask(
        'CartPole-v1', initial_trajectories=0, max_steps=200)

    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-3,
        warmup_steps=100,
        factors='constant * linear_warmup')

    body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    trainer = actor_critic.PPOTrainer(
        task,
        n_shared_layers=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=128,
        value_train_steps_per_epoch=10,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=128,
        policy_train_steps_per_epoch=10,
        n_trajectories_per_epoch=10)

    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_awrtrainer_cartpole(self):
    """Test-runs AWR on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=1000,
                          max_steps=200)
    body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.AWRTrainer(
        task,
        n_shared_layers=0,
        added_policy_slice_length=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=32,
        value_train_steps_per_epoch=200,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=32,
        policy_train_steps_per_epoch=200,
        n_trajectories_per_epoch=10,
        advantage_estimator=advantages.monte_carlo,
        advantage_normalization=False,
    )
    trainer.run(1)
    self.assertEqual(1, trainer.current_epoch)
    self.assertGreater(trainer.avg_returns[-1], 35.0)

  def test_awrtrainer_cartpole_shared(self):
    """Test-runs AWR on cartpole with shared layers."""
    # This test is flaky, and this is the simplest way to retry in OSS.
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=1000,
                          max_steps=200)
    body = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    # pylint: disable=g-long-lambda
    lr = (
        lambda: lr_schedules.multifactor(
            constant=1e-2, warmup_steps=100,
            factors='constant * linear_warmup')
    )
    # pylint: enable=g-long-lambda
    max_avg_returns = -math.inf
    for _ in range(5):
      trainer = actor_critic.AWRTrainer(
          task,
          n_shared_layers=1,
          added_policy_slice_length=1,
          value_model=value_model,
          value_optimizer=opt.Adam,
          value_lr_schedule=lr,
          value_batch_size=32,
          value_train_steps_per_epoch=200,
          policy_model=policy_model,
          policy_optimizer=opt.Adam,
          policy_lr_schedule=lr,
          policy_batch_size=32,
          policy_train_steps_per_epoch=200,
          n_trajectories_per_epoch=10,
          advantage_estimator=advantages.monte_carlo,
          advantage_normalization=False,
      )
      trainer.run(1)
      self.assertEqual(1, trainer.current_epoch)
      max_avg_returns = (
          max_avg_returns if max_avg_returns > trainer.avg_returns[-1]
          else trainer.avg_returns[-1])
      if trainer.avg_returns[-1] > 35.0:
        return
    self.fail(f'We did not reach a score > 35.0, max was {max_avg_returns}.')

  def test_sanity_awrtrainer_transformer_cartpole(self):
    """Test-runs AWR on cartpole with Transformer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=2,
                          max_steps=2)
    body = lambda mode: models.TransformerDecoder(  # pylint: disable=g-long-lambda
        d_model=2, d_ff=2, n_layers=1, n_heads=1, mode=mode)
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.AWRTrainer(
        task,
        n_shared_layers=0,
        max_slice_length=2,
        added_policy_slice_length=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=2,
        value_train_steps_per_epoch=2,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=1,
        n_eval_episodes=1)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_sampling_awrtrainer_cartpole(self):
    """Test-runs AWR on cartpole with Transformer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=0,
                          max_steps=20)
    body = lambda mode: tl.Serial(tl.Dense(2), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.SamplingAWRTrainer(
        task,
        n_shared_layers=0,
        added_policy_slice_length=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=2,
        value_train_steps_per_epoch=2,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        advantage_estimator=advantages.monte_carlo,
        advantage_normalization=False,
        q_value_n_samples=3,
        q_value_aggregate_max=True,
        reweight=False,
    )
    trainer.run(1)
    self.assertEqual(1, trainer.current_epoch)

  def test_sampling_awrtrainer_cartpole_sample_all_discrete(self):
    """Test-runs AWR on cartpole with Transformer, n_actions = n_samples."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=0,
                          max_steps=20)
    body = lambda mode: tl.Serial(tl.Dense(2), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.SamplingAWRTrainer(
        task,
        n_shared_layers=0,
        added_policy_slice_length=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=2,
        value_train_steps_per_epoch=2,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        advantage_estimator=advantages.monte_carlo,
        advantage_normalization=False,
        q_value_n_samples=2,
        q_value_aggregate_max=True,
        reweight=False,
    )
    trainer.run(1)
    self.assertEqual(1, trainer.current_epoch)

  def test_sampling_awrtrainer_mountain_acr(self):
    """Test-runs Sampling AWR on MountainCarContinuous."""
    task = rl_task.RLTask('MountainCarContinuous-v0', initial_trajectories=0,
                          max_steps=2)
    body = lambda mode: tl.Serial(tl.Dense(2), tl.Relu())
    policy_model = functools.partial(models.Policy, body=body)
    value_model = functools.partial(models.Value, body=body)
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic.SamplingAWRTrainer(
        task,
        n_shared_layers=0,
        added_policy_slice_length=1,
        value_model=value_model,
        value_optimizer=opt.Adam,
        value_lr_schedule=lr,
        value_batch_size=2,
        value_train_steps_per_epoch=2,
        policy_model=policy_model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=2,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        advantage_estimator=advantages.monte_carlo,
        advantage_normalization=False,
        q_value_n_samples=3,
    )
    trainer.run(1)
    self.assertEqual(1, trainer.current_epoch)


if __name__ == '__main__':
  absltest.main()
