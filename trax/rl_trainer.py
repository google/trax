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

r"""Trainer for RL environments.

For now we only support PPO as RL algorithm.

Sample invocation:

.. code-block:: bash

    TRAIN_BATCH_SIZE=32
    python trax/rl_trainer.py \
      --config_file=trax/rl/configs/ppo_acrobot.gin \
      --train_batch_size=${TRAIN_BATCH_SIZE} \
      --output_dir=${HOME}/ppo_acrobot \
      --alsologtostderr
"""

import faulthandler

from absl import app
from absl import flags
from absl import logging
import gin
import jax
from jax.config import config
from trax import fastmath
from trax import rl  # pylint: disable=unused-import
from trax import trainer_flags  # pylint: disable=unused-import
from trax.rl import task as rl_task
from trax.rl import training as light_trainers
from trax.tf_numpy import numpy as tf_np


FLAGS = flags.FLAGS


# Not just 'train' to avoid a conflict with trax.train in GIN files.
@gin.configurable(denylist=[
    'output_dir', 'train_batch_size', 'eval_batch_size', 'trajectory_dump_dir'
])
def train_rl(
    output_dir,
    train_batch_size,
    eval_batch_size,
    env_name='Acrobot-v1',
    max_timestep=None,
    clip_rewards=False,
    rendered_env=False,
    resize=False,
    resize_dims=(105, 80),
    trainer_class=None,
    n_epochs=10000,
    trajectory_dump_dir=None,
    num_actions=None,
    light_rl=True,
    light_rl_trainer=light_trainers.PolicyGradient,
):
  """Train the RL agent.

  Args:
    output_dir: Output directory.
    train_batch_size: Number of parallel environments to use for training.
    eval_batch_size: Number of parallel environments to use for evaluation.
    env_name: Name of the environment.
    max_timestep: Int or None, the maximum number of timesteps in a trajectory.
      The environment is wrapped in a TimeLimit wrapper.
    clip_rewards: Whether to clip and discretize the rewards.
    rendered_env: Whether the environment has visual input. If so, a
      RenderedEnvProblem will be used.
    resize: whether to do resize or not
    resize_dims: Pair (height, width), dimensions to resize the visual
      observations to.
    trainer_class: RLTrainer class to use.
    n_epochs: Number epochs to run the training for.
    trajectory_dump_dir: Directory to dump trajectories to.
    num_actions: None unless one wants to use the discretization wrapper. Then
      num_actions specifies the number of discrete actions.
    light_rl: deprecated, always True, left out for old gin configs.
    light_rl_trainer: which light RL trainer to use (experimental).
  """
  del light_rl
  tf_np.set_allow_float64(FLAGS.tf_allow_float64)
  task = rl_task.RLTask()
  env_name = task.env_name


  if FLAGS.jax_debug_nans:
    config.update('jax_debug_nans', True)

  if FLAGS.use_tpu:
    config.update('jax_platform_name', 'tpu')
  else:
    config.update('jax_platform_name', '')


  trainer = light_rl_trainer(task=task, output_dir=output_dir)
  def light_training_loop():
    """Run the trainer for n_epochs and call close on it."""
    try:
      logging.info('Starting RL training for %d epochs.', n_epochs)
      trainer.run(n_epochs, n_epochs_is_total_epochs=True)
      logging.info('Completed RL training for %d epochs.', n_epochs)
      trainer.close()
      logging.info('Trainer is now closed.')
    except Exception as e:
      raise e
    finally:
      logging.info('Encountered an exception, still calling trainer.close()')
      trainer.close()
      logging.info('Trainer is now closed.')

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    fastmath.disable_jit()
    with jax.disable_jit():
      light_training_loop()
  else:
    light_training_loop()


def main(argv):
  del argv
  logging.info('Starting RL training.')

  gin_configs = FLAGS.config if FLAGS.config is not None else []
  gin.parse_config_files_and_bindings(FLAGS.config_file, gin_configs)

  logging.info('Gin config:')
  logging.info(gin_configs)

  train_rl(
      output_dir=FLAGS.output_dir,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      trajectory_dump_dir=(FLAGS.trajectory_dump_dir or None),
  )

  # TODO(afrozm): This is for debugging.
  logging.info('Dumping stack traces of all stacks.')
  faulthandler.dump_traceback(all_threads=True)

  logging.info('Training is done, should exit.')


if __name__ == '__main__':
  app.run(main)
