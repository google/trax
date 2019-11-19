# coding=utf-8
# Copyright 2019 The Trax Authors.
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

"""Trax trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags
from absl import logging

import gin
import jax
import tensorflow.compat.v2 as tf
from trax import backend
from trax import trainer_lib
from trax.tf_numpy import numpy as tf_np

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Which dataset to use.')
flags.DEFINE_string('model', None, 'Which model to train.')
flags.DEFINE_string('data_dir', None, 'Path to the directory with data.')
flags.DEFINE_string('output_dir', None,
                    'Path to the directory to save logs and checkpoints.')
flags.DEFINE_multi_string('config_file', None,
                          'Configuration file with parameters (.gin).')
flags.DEFINE_multi_string('config', None,
                          'Configuration parameters (gin string).')
flags.DEFINE_integer('log_level', logging.INFO, 'Log level.')
# TPU Flags
flags.DEFINE_bool('use_tpu', False, "Whether we're running on TPU.")
flags.DEFINE_string(
    'jax_xla_backend', 'xla',
    'Either "xla" for the XLA service directly, or "tpu_driver"'
    'for a TPU Driver backend.')
flags.DEFINE_string('jax_backend_target', 'local',
                    'Either "local" or "rpc:address" to connect to a '
                    'remote service target.')
# TF Flags
flags.DEFINE_bool('enable_eager_execution', True,
                  "Whether we're running TF in eager mode.")
flags.DEFINE_bool('tf_xla', True, 'Whether to turn on XLA for TF.')
flags.DEFINE_bool('tf_opt_pin_to_host', False, 'Whether to turn on TF '
                  'pin-to-host optimization.')
flags.DEFINE_bool('tf_opt_layout', False, 'Whether to turn on TF layout '
                  'optimization.')
flags.DEFINE_bool('tf_xla_forced_compile', False, 'Use forced-compilation '
                  'instead of auto-clustering for XLA. This flag only has '
                  'effects when --tf_xla is on.')
flags.DEFINE_bool('tf_allow_float64', False, 'Whether to allow float64 for TF.')


def _default_output_dir():
  """Default output directory."""
  try:
    dataset_name = gin.query_parameter('inputs.dataset_name')
  except ValueError:
    dataset_name = 'random'
  dir_name = '{model_name}_{dataset_name}_{timestamp}'.format(
      model_name=gin.query_parameter('train.model').configurable.name,
      dataset_name=dataset_name,
      timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
  )
  dir_path = os.path.join('~', 'trax', dir_name)
  print()
  trainer_lib.log('No --output_dir specified')
  return dir_path


def _setup_gin():
  """Setup gin configuration."""
  # Imports for configurables
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable
  from trax import models as _trax_models
  from trax import optimizers as _trax_opt
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

  configs = FLAGS.config or []
  # Override with --dataset and --model
  if FLAGS.dataset:
    configs.append("inputs.dataset_name='%s'" % FLAGS.dataset)
    if FLAGS.data_dir:
      configs.append("inputs.data_dir='%s'" % FLAGS.data_dir)
  if FLAGS.model:
    configs.append('train.model=@trax.models.%s' % FLAGS.model)
  gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def set_tf_allow_float64(b):
  tf_np.set_allow_float64(b)


@gin.configurable
def tf_init_tpu(worker='', protocol=None):
  """Initializes TPU for TensorFlow.

  Args:
    worker: The BNS address of the remote TPU worker. If it's empty (the default
      value), TF will assume the TPU devices are connected to the local host.
    protocol: The network protocol used to connect to the TPU worker.
  Returns:
    The device name of the TPU worker's CPU.
  """
  protocol = protocol or 'grpc'
  is_local = (worker in ('', 'local'))
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=worker)
  if not is_local:
    tf.config.experimental_connect_to_cluster(resolver, protocol=protocol)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  if is_local:
    return ''
  else:
    return '/job:worker'


def main(_):

  logging.set_verbosity(FLAGS.log_level)

  if FLAGS.enable_eager_execution:
    tf.compat.v1.enable_eager_execution()

  if FLAGS.tf_xla:
    tf.config.optimizer.set_jit(True)
    backend.set_tf_xla_forced_compile(FLAGS.tf_xla_forced_compile)

  tf.config.optimizer.set_experimental_options(
      {'pin_to_host_optimization': FLAGS.tf_opt_pin_to_host}
  )

  tf.config.optimizer.set_experimental_options(
      {'layout_optimizer': FLAGS.tf_opt_layout}
  )

  set_tf_allow_float64(FLAGS.tf_allow_float64)

  _setup_gin()

  if FLAGS.enable_eager_execution and backend.get_name() in ('numpy', 'jax'):
    # Numpy backend doesn't benefit from having the input pipeline run on GPU,
    # and jax backend has GPU memory contention if TF uses the GPU. Gin must be
    # set up first before determining the backend.
    tf.config.experimental.set_visible_devices([], 'GPU')

  # Setup output directory
  output_dir = FLAGS.output_dir or _default_output_dir()
  trainer_lib.log('Using --output_dir %s' % output_dir)
  output_dir = os.path.expanduser(output_dir)

  # If on TPU, let JAX know.
  if FLAGS.use_tpu:
    jax.config.update('jax_platform_name', 'tpu')
    jax.config.update('jax_xla_backend', FLAGS.jax_xla_backend)
    jax.config.update('jax_backend_target', FLAGS.jax_backend_target)

  if FLAGS.use_tpu and backend.get_name() == 'tf':
    worker_cpu = tf_init_tpu()
    with tf.device(worker_cpu):
      if trainer_lib.num_devices() == 1:
        # TF's device priority is GPU > CPU > TPU, so we need to explicitly make
        # the TPU core the default device here.
        with tf.device('/device:TPU:0'):
          trainer_lib.train(output_dir=output_dir)
      else:
        trainer_lib.train(output_dir=output_dir)
  else:
    trainer_lib.train(output_dir=output_dir)

  trainer_lib.log('Finished training.')



if __name__ == '__main__':
  app.run(main)
