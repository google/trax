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
"""Flags for trainer.py and rl_trainer.py.

We keep these flags in sync across the trainer and the rl_trainer binaries.
"""

from absl import flags
from absl import logging

# Common flags.
flags.DEFINE_string('output_dir',
                    None,
                    'Path to the directory to save logs and checkpoints.')
flags.DEFINE_multi_string('config_file',
                          None,
                          'Configuration file with parameters (.gin).')
flags.DEFINE_multi_string('config',
                          None,
                          'Configuration parameters (gin string).')

# TPU Flags
flags.DEFINE_bool('use_tpu', False, "Whether we're running on TPU.")
flags.DEFINE_string('jax_xla_backend',
                    'xla',
                    'Either "xla" for the XLA service directly, or "tpu_driver"'
                    'for a TPU Driver backend.')
flags.DEFINE_string('jax_backend_target',
                    'local',
                    'Either "local" or "rpc:address" to connect to a '
                    'remote service target.')

# trainer.py flags.
flags.DEFINE_string('dataset', None, 'Which dataset to use.')
flags.DEFINE_string('model', None, 'Which model to train.')
flags.DEFINE_string('data_dir', None, 'Path to the directory with data.')
flags.DEFINE_integer('log_level', logging.INFO, 'Log level.')

# TensorFlow Flags
flags.DEFINE_bool('enable_eager_execution',
                  True,
                  "Whether we're running TF in eager mode.")
flags.DEFINE_bool('tf_xla', True, 'Whether to turn on XLA for TF.')
flags.DEFINE_bool('tf_opt_pin_to_host',
                  False,
                  'Whether to turn on TF pin-to-host optimization.')
flags.DEFINE_bool('tf_opt_layout',
                  False,
                  'Whether to turn on TF layout optimization.')
flags.DEFINE_bool('tf_xla_forced_compile',
                  False,
                  'Use forced-compilation instead of auto-clustering for XLA.'
                  'This flag only has effects when --tf_xla is on.')
flags.DEFINE_bool('tf_allow_float64', False, 'Whether to allow float64 for TF.')

# rl_trainer.py flags.
flags.DEFINE_boolean('jax_debug_nans',
                     False,
                     'Setting to true will help to debug nans and disable jit.')
flags.DEFINE_boolean('disable_jit', False, 'Setting to true will disable jit.')
flags.DEFINE_string('envs_output_dir', '', 'Output dir for the envs.')
flags.DEFINE_bool('xm', False, 'Copy atari roms?')
flags.DEFINE_integer('train_batch_size',
                     32,
                     'Number of parallel environments during training.')
flags.DEFINE_integer('eval_batch_size', 4, 'Batch size for evaluation.')
flags.DEFINE_boolean('parallelize_envs',
                     False,
                     'If true, sets parallelism to number of cpu cores.')
flags.DEFINE_string('trajectory_dump_dir',
                    '',
                    'Directory to dump trajectories to.')
flags.DEFINE_bool('async_mode', False, 'Async mode.')
