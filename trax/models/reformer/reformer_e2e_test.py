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

"""End to end test for Reformer."""

import os

from absl.testing import absltest
import gin

import trax
from trax import test_utils
from trax.models.reformer import reformer  # pylint: disable=unused-import
from trax.supervised import trainer_lib

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')
_CONFIG_DIR = os.path.join(pkg_dir, '../../supervised/configs/')


class ReformerE2ETest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    gin.add_config_file_search_path(_CONFIG_DIR)
    test_utils.ensure_flag('test_tmpdir')

  def test_reformer_wmt_ende(self):
    trax.fastmath.disable_jit()

    batch_size_per_device = 2
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('reformer_wmt_ende.gin')

    gin.bind_parameter('data_streams.data_dir', _TESTDATA)
    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('Reformer.n_encoder_layers', n_layers)
    gin.bind_parameter('Reformer.n_decoder_layers', n_layers)
    gin.bind_parameter('Reformer.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)

  def test_reformer2_wmt_ende(self):
    trax.fastmath.disable_jit()

    batch_size_per_device = 1  # Ignored, but needs to be set.
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('reformer2_wmt_ende.gin')

    gin.bind_parameter('data_streams.data_dir', _TESTDATA)
    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('batcher.buckets', ([512], [1, 1]))  # batch size 1.
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('Reformer2.n_encoder_layers', n_layers)
    gin.bind_parameter('Reformer2.n_decoder_layers', n_layers)
    gin.bind_parameter('Reformer2.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)

  def test_reformer2_copy(self):
    trax.fastmath.disable_jit()

    batch_size_per_device = 1  # Ignored, but needs to be set.
    steps = 1
    n_layers = 2
    d_ff = 32

    gin.parse_config_file('reformer2_copy.gin')

    gin.bind_parameter('batcher.batch_size_per_device', batch_size_per_device)
    gin.bind_parameter('batcher.buckets', ([64], [1, 1]))  # batch size 1.
    gin.bind_parameter('train.steps', steps)
    gin.bind_parameter('Reformer2.n_encoder_layers', n_layers)
    gin.bind_parameter('Reformer2.n_decoder_layers', n_layers)
    gin.bind_parameter('Reformer2.d_ff', d_ff)

    output_dir = self.create_tempdir().full_path
    _ = trainer_lib.train(output_dir=output_dir)

if __name__ == '__main__':
  absltest.main()
