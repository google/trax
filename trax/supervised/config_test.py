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

# Lint as: python3
"""Tests for trax.supervised.config."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import gin

from trax.supervised import trainer_lib

_CONFIG_FOLDERS = (
    'configs/',
)

# TODO(yuwenyan): do not skip the following configs.
_SKIPPED_CONFIG_FILENAMES = (
    'bert.gin',
    'bert_glue_classification.gin',
    'bert_glue_regression.gin',
    'bert_pretraining.gin',
    'bert_pretraining_onlymlm.gin',
    'bert_pretraining_onlynsp.gin',
    'c4.gin',
    'c4_pretrain_16gb_adafactor.gin',
    'cond_skipping_transformer_lm1b.gin',
    'funnel_imdb_tfds.gin',
    'mira.gin',
    'reformer2_c4_big.gin',
    'reformer2_c4_medium.gin',
    'reformer2_copy.gin',
    'reformer2_copy_self_attn.gin',
    'reformer2_wmt_ende.gin',
    'reformer_bair_robot_pushing.gin',
    'reformer_c4.gin',
    'reformer_pc_enpl.gin',
    'resnet50_frn_imagenet_8gb.gin',
    'scientific_papers_reformer2_favor.gin',
    'sparse_c4_pretrain_16gb_adafactor.gin',
    't5_glue_classification.gin',
    't5_glue_classification_mnli.gin',
    't5_mathqa.gin',
    'transformer_finetune_squad_16gb.gin',
    'transformer_imdb_tfds.gin',
    'transformer_lm_wmt_ende_16gb.gin',
    'transformer_meena_8gb_adafactor.gin',
    'transformer_ptb_16gb.gin',
)


def list_configs():

  def is_skipped(config):
    return config in _SKIPPED_CONFIG_FILENAMES or not config.endswith('.gin')

  pkg_dir = os.path.dirname(__file__)
  for config_folder in _CONFIG_FOLDERS:
    config_dir = os.path.join(pkg_dir, config_folder)
    for config in os.listdir(config_dir):
      if not is_skipped(config):
        yield (os.path.splitext(config)[0], os.path.join(config_dir, config))


class ConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config(clear_constants=True)

  @parameterized.named_parameters(list_configs())
  def test_dry_run(self, config):
    """Dry-runs all gin configs."""
    gin.parse_config_file(config)

    trainer_lib.train(
        self.create_tempdir().full_path,
        # Don't run any actual training, just initialize all classes.
        steps=0,
    )


if __name__ == '__main__':
  absltest.main()
