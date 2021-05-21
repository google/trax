# coding=utf-8
# Copyright 2021 The Trax Authors.
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

"""End to end test for Reformer."""

import os

from absl.testing import absltest
import gin

from trax import test_utils
from trax.models.research import terraformer  # pylint: disable=unused-import
from trax.supervised import trainer_lib
from trax.tf_numpy import numpy as tf_np  # pylint: disable=unused-import

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')
_CONFIG_DIR = os.path.join(pkg_dir, '../../supervised/configs/')


class TerraformerE2ETest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    gin.add_config_file_search_path(_CONFIG_DIR)
    test_utils.ensure_flag('test_tmpdir')

  def test_terraformer_wmt_ende(self):
    batch_size_per_device = 2
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('terraformer_wmt_ende.gin')

    gin.bind_parameter('data_streams.data_dir', _TESTDATA)
    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('batcher.buckets',
                       ([512], [batch_size_per_device, batch_size_per_device]))
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('ConfigurableTerraformer.n_encoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.n_decoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)

  def test_terraformer_copy(self):
    batch_size_per_device = 2
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('terraformer_copy.gin')

    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('batcher.buckets', ([64], [1, 1]))  # batch size 1.
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('ConfigurableTerraformer.n_encoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.n_decoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)

  def test_terraformer_purelsh_copy(self):
    batch_size_per_device = 2
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('terraformer_purelsh_copy.gin')

    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('batcher.buckets', ([64], [1, 1]))  # batch size 1.
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('ConfigurableTerraformer.n_encoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.n_decoder_layers', n_layers)
    gin.bind_parameter('ConfigurableTerraformer.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)


if __name__ == '__main__':
  absltest.main()
