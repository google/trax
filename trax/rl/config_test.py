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
"""Tests for gin configs."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import gin

from trax import rl_trainer


# TODO(pkozakowski): Extend to handle the supervised configs as well.
def list_configs():
  pkg_dir = os.path.dirname(__file__)
  config_dir = os.path.join(pkg_dir, 'configs/')
  return [
      # (config name without extension, config path)
      (os.path.splitext(config)[0], os.path.join(config_dir, config))
      for config in os.listdir(config_dir)
  ]


class ConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(list_configs())
  def test_dry_run(self, config):
    """Dry-runs all gin configs."""
    gin.clear_config(clear_constants=True)
    gin.parse_config_file(config)
    try:
      rl_trainer.train_rl(
          output_dir=self.create_tempdir().full_path,
          # Don't run any actual training, just initialize all classes.
          n_epochs=0,
          train_batch_size=1,
          eval_batch_size=1,
      )
    except Exception as e:
      raise AssertionError(
          'Error in gin config {}.'.format(os.path.basename(config))
      ) from e


if __name__ == '__main__':
  absltest.main()
