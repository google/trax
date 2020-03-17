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

from absl.testing import absltest

from trax import layers as tl
from trax import lr_schedules
from trax import optimizers as opt
from trax.rl import task as rl_task
from trax.rl import training


class TrainingTest(absltest.TestCase):

  def test_policytrainer_cartpole(self):
    """Trains a policy on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=100,
                          max_steps=200)
    # TODO(pkozakowski): Use Distribution.n_inputs to initialize the action
    # head.
    model = lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
        tl.Dense(64), tl.Relu(), tl.Dense(64), tl.Relu(),
        tl.Dense(2), tl.LogSoftmax())
    lr = lambda h: lr_schedules.MultifactorSchedule(  # pylint: disable=g-long-lambda
        h, constant=0.001, factors='constant')
    trainer = training.PolicyGradientTrainer(
        task,
        policy_model=model,
        policy_optimizer=opt.Adam,
        policy_lr_schedule=lr,
        policy_batch_size=32,
        policy_train_steps_per_epoch=100,
        collect_per_epoch=5)
    trainer.run(10)
    self.assertEqual(10, trainer.current_epoch)
    self.assertGreater(max(trainer.avg_returns), 100.0)


if __name__ == '__main__':
  absltest.main()
