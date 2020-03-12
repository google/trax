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
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=750,
                          max_steps=200)
    model = lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
        tl.Dense(64), tl.Relu(), tl.Dense(2), tl.LogSoftmax())
    lr = lambda h: lr_schedules.MultifactorSchedule(  # pylint: disable=g-long-lambda
        h, constant=1e-4, warmup_steps=100, factors='constant * linear_warmup')
    trainer = training.PolicyGradientTrainer(
        task, model, opt.Adam, lr_schedule=lr, batch_size=128,
        train_steps_per_epoch=700, collect_per_epoch=50)
    trainer.run(1)
    # This should *mostly* pass, this means that this test is flaky.
    self.assertGreater(trainer.avg_returns[-1], 35.0)
    self.assertEqual(1, trainer.current_epoch)


if __name__ == '__main__':
  absltest.main()
