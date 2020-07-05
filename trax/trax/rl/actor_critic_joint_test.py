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

from absl.testing import absltest

from trax import layers as tl
from trax import models
from trax import optimizers as opt
from trax import test_utils
from trax.rl import actor_critic_joint
from trax.rl import task as rl_task
from trax.supervised import lr_schedules



class ActorCriticJointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_awrjoint_save_restore(self):
    """Check save and restore of joint AWR trainer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=100,
                          max_steps=200)
    joint_model = functools.partial(
        models.PolicyAndValue,
        body=lambda mode: tl.Serial(tl.Dense(64), tl.Relu()),
    )
    tmp_dir = self.create_tempdir().full_path
    trainer1 = actor_critic_joint.AWRJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.Adam,
        batch_size=4,
        train_steps_per_epoch=1,
        n_trajectories_per_epoch=2,
        output_dir=tmp_dir)
    trainer1.run(2)
    self.assertEqual(trainer1.current_epoch, 2)
    self.assertEqual(trainer1._trainer.step, 2)
    # Trainer 2 starts where trainer 1 stopped.
    trainer2 = actor_critic_joint.AWRJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.Adam,
        batch_size=4,
        train_steps_per_epoch=1,
        n_trajectories_per_epoch=2,
        output_dir=tmp_dir)
    trainer2.run(1)
    self.assertEqual(trainer2.current_epoch, 3)
    self.assertEqual(trainer2._trainer.step, 3)
    trainer1.close()
    trainer2.close()


  def test_jointppotrainer_cartpole(self):
    """Test-runs joint PPO on CartPole."""

    task = rl_task.RLTask('CartPole-v0', initial_trajectories=0,
                          max_steps=2)
    joint_model = functools.partial(
        models.PolicyAndValue,
        body=lambda mode: tl.Serial(tl.Dense(2), tl.Relu()),
    )
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')

    trainer = actor_critic_joint.PPOJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.Adam,
        lr_schedule=lr,
        batch_size=4,
        train_steps_per_epoch=2,
        n_trajectories_per_epoch=5)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_jointawrtrainer_cartpole(self):
    """Test-runs joint AWR on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=100,
                          max_steps=200)
    joint_model = functools.partial(
        models.PolicyAndValue,
        body=lambda mode: tl.Serial(tl.Dense(64), tl.Relu()),
    )
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic_joint.AWRJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.Adam,
        lr_schedule=lr,
        batch_size=4,
        train_steps_per_epoch=2,
        n_trajectories_per_epoch=5)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_jointa2ctrainer_cartpole(self):
    """Test-runs joint A2C on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=100,
                          max_steps=200)
    joint_model = functools.partial(
        models.PolicyAndValue,
        body=lambda mode: tl.Serial(tl.Dense(64), tl.Relu()),
    )
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic_joint.A2CJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.RMSProp,
        lr_schedule=lr,
        batch_size=2,
        train_steps_per_epoch=1,
        n_trajectories_per_epoch=1)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_jointawrtrainer_cartpole_transformer(self):
    """Test-runs joint AWR on cartpole with Transformer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=1,
                          max_steps=2)
    body = lambda mode: models.TransformerDecoder(  # pylint: disable=g-long-lambda
        d_model=32, d_ff=32, n_layers=1, n_heads=1, mode=mode)
    joint_model = functools.partial(models.PolicyAndValue, body=body)
    trainer = actor_critic_joint.AWRJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.Adam,
        batch_size=4,
        train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        max_slice_length=2)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)

  def test_jointa2ctrainer_cartpole_transformer(self):
    """Test-runs joint A2C on cartpole with Transformer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=100,
                          max_steps=200)
    body = lambda mode: models.TransformerDecoder(  # pylint: disable=g-long-lambda
        d_model=32, d_ff=32, n_layers=1, n_heads=1, mode=mode)
    joint_model = functools.partial(models.PolicyAndValue, body=body)
    trainer = actor_critic_joint.A2CJointTrainer(
        task,
        joint_model=joint_model,
        optimizer=opt.RMSProp,
        batch_size=4,
        train_steps_per_epoch=2,
        n_trajectories_per_epoch=2)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)


if __name__ == '__main__':
  absltest.main()
