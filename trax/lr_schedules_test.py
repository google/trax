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
"""Tests for trax.lr_schedules."""

import functools

import numpy as onp
from tensorflow import test
from trax import history as trax_history
from trax import lr_schedules
from trax.math import numpy as np
from trax.models import transformer
from trax.rl import online_tune
from trax.rl import ppo
from trax.shapes import ShapeDtype


class PolicyScheduleTest(test.TestCase):

  def _make_schedule(
      self,
      history,
      control_configs,
      observation_metrics=(('eval', 'metrics/accuracy'),),
      action_multipliers=(1.0,),
      vocab_size=None,
  ):
    policy_and_value_model = functools.partial(
        transformer.TransformerDecoder,
        d_model=2,
        d_ff=2,
        n_layers=0,
        vocab_size=vocab_size,
    )
    net = ppo.policy_and_value_net(
        n_actions=len(action_multipliers),
        n_controls=len(control_configs),
        vocab_size=None,
        bottom_layers_fn=policy_and_value_model,
        two_towers=False,
    )
    obs_dim = len(observation_metrics)
    if vocab_size is None:
      shape = (1, 1, obs_dim)
      dtype = np.float32
    else:
      shape = (1, 1)
      dtype = np.int32
    input_signature = ShapeDtype(shape, dtype)
    (params, state) = net.init(input_signature)
    policy_dir = self.get_temp_dir()
    # Optimizer slots and parameters should not be used for anything.
    slots = None
    opt_params = None
    opt_state = (params, slots, opt_params)
    ppo.save_opt_state(
        policy_dir, opt_state, state, epoch=0, total_opt_step=0, history=history
    )
    return lr_schedules.PolicySchedule(
        history,
        observation_metrics=observation_metrics,
        include_controls_in_observation=False,
        action_multipliers=action_multipliers,
        control_configs=control_configs,
        policy_and_value_model=policy_and_value_model,
        policy_and_value_two_towers=False,
        policy_and_value_vocab_size=vocab_size,
        policy_dir=policy_dir,
    )

  def test_returns_start_lr_when_there_are_no_metrics(self):
    history = trax_history.History()
    start_lr = 1e-3
    schedule = self._make_schedule(
        history,
        control_configs=(('learning_rate', start_lr, (1e-9, 1.0), False),),
    )
    self.assertEqual(schedule(0)['learning_rate'], start_lr)

  def test_changes_lr_when_there_are_some_metrics(self):
    history = trax_history.History()
    history.append('eval', 'metrics/accuracy', step=0, value=0.8)
    history.append(
        *online_tune.control_metric('learning_rate'), step=0, value=1e-4
    )
    schedule = self._make_schedule(
        history,
        control_configs=(('learning_rate', 1e-3, (1e-9, 1.0), False),),
        observation_metrics=(('eval', 'metrics/accuracy'),),
        action_multipliers=(0.5, 2.0),
    )
    new_lr = schedule(123)['learning_rate']
    self.assertTrue(
        onp.allclose(new_lr, 5e-5) or onp.allclose(new_lr, 2e-4)
    )

  def test_works_with_multiple_controls(self):
    history = trax_history.History()
    history.append('eval', 'metrics/accuracy', step=0, value=0.8)
    history.append(
        *online_tune.control_metric('learning_rate'), step=0, value=1e-4
    )
    history.append(
        *online_tune.control_metric('weight_decay_rate'), step=0, value=1e-5
    )
    schedule = self._make_schedule(
        history,
        observation_metrics=(('eval', 'metrics/accuracy'),),
        control_configs=(
            ('learning_rate', 1e-3, (1e-9, 1.0), False),
            ('weight_decay_rate', 1e-5, (1e-9, 1.0), False),
        ),
        action_multipliers=(1.0,),
    )
    new_controls = schedule(123)
    self.assertIn('learning_rate', new_controls)
    self.assertIn('weight_decay_rate', new_controls)

  def test_works_with_serialized_policy(self):
    history = trax_history.History()
    history.append('eval', 'metrics/accuracy', step=0, value=0.8)
    history.append(
        *online_tune.control_metric('learning_rate'), step=0, value=1e-4
    )
    schedule = self._make_schedule(
        history,
        control_configs=(('learning_rate', 1e-3, (1e-9, 1.0), False),),
        observation_metrics=(('eval', 'metrics/accuracy'),),
        action_multipliers=(0.5, 2.0),
        vocab_size=16,
    )
    new_lr = schedule(123)['learning_rate']
    self.assertTrue(
        onp.allclose(new_lr, 5e-5) or onp.allclose(new_lr, 2e-4)
    )


if __name__ == '__main__':
  test.main()
