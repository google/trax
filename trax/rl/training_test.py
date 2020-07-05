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
import os
import pickle

from absl.testing import absltest
import tensorflow as tf

from trax import layers as tl
from trax import models
from trax import optimizers as opt
from trax import test_utils
from trax.rl import task as rl_task
from trax.rl import training
from trax.supervised import lr_schedules


class TrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.ensure_flag('test_tmpdir')

  def test_policytrainer_save_restore(self):
    """Check save and restore of policy trainer."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=10,
                          max_steps=200)
    model = functools.partial(
        models.Policy,
        body=lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
            tl.Dense(64), tl.Relu(), tl.Dense(64), tl.Relu()
        ),
    )
    tmp_dir = self.create_tempdir().full_path
    trainer1 = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_batch_size=128,
        policy_train_steps_per_epoch=1,
        n_trajectories_per_epoch=2,
        n_eval_episodes=1,
        output_dir=tmp_dir)
    trainer1.run(1)
    trainer1.run(1)
    self.assertEqual(trainer1.current_epoch, 2)
    self.assertEqual(trainer1._policy_trainer.step, 2)
    # Trainer 2 starts where trainer 1 stopped.
    trainer2 = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_batch_size=128,
        policy_train_steps_per_epoch=1,
        n_trajectories_per_epoch=2,
        n_eval_episodes=1,
        output_dir=tmp_dir)
    trainer2.run(1)
    self.assertEqual(trainer2.current_epoch, 3)
    self.assertEqual(trainer2._policy_trainer.step, 3)
    # Trainer 3 has 2x steps-per-epoch, but epoch 3, should raise an error.
    trainer3 = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_batch_size=128,
        policy_train_steps_per_epoch=2,
        n_trajectories_per_epoch=2,
        n_eval_episodes=1,
        output_dir=tmp_dir)
    self.assertRaises(ValueError, trainer3.run)
    # Manually set saved epoch to 1.
    dictionary = {'epoch': 1, 'avg_returns': [0.0],
                  'avg_returns_temperature0': {200: [0.0]}}
    with tf.io.gfile.GFile(os.path.join(tmp_dir, 'rl.pkl'), 'wb') as f:
      pickle.dump(dictionary, f)
    # Trainer 3 still should fail as steps between evals are 2, cannot do 1.
    self.assertRaises(ValueError, trainer3.run)
    # Trainer 4 does 1 step per eval, should train 1 step in epoch 2.
    trainer4 = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_batch_size=128,
        policy_train_steps_per_epoch=2,
        policy_evals_per_epoch=2,
        n_trajectories_per_epoch=2,
        n_eval_episodes=1,
        output_dir=tmp_dir)
    trainer4.run(1)
    self.assertEqual(trainer4.current_epoch, 2)
    self.assertEqual(trainer4._policy_trainer.step, 4)
    trainer1.close()
    trainer2.close()
    trainer3.close()
    trainer4.close()

  def test_policytrainer_cartpole(self):
    """Trains a policy on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=1,
                          max_steps=200)
    model = functools.partial(
        models.Policy,
        body=lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
            tl.Dense(64), tl.Relu(), tl.Dense(64), tl.Relu()
        ),
    )
    lr = lambda: lr_schedules.multifactor(  # pylint: disable=g-long-lambda
        constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=128,
        policy_train_steps_per_epoch=1,
        n_trajectories_per_epoch=2)
    # Assert that we get to 200 at some point and then exit so the test is as
    # fast as possible.
    for ep in range(200):
      trainer.run(1)
      self.assertEqual(trainer.current_epoch, ep + 1)
      if trainer.avg_returns[-1] == 200.0:
        return
    self.fail(
        'The expected score of 200 has not been reached. '
        'Maximum was {}.'.format(max(trainer.avg_returns))
    )


if __name__ == '__main__':
  absltest.main()
