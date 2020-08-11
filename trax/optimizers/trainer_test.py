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
"""Tests for accelerated optimization of loss layers."""

from absl.testing import absltest

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import optimizers


class TrainerTest(absltest.TestCase):

  def test_run_simple_task(self):
    """Runs an accelerated optimizer on a simple task."""
    inputs_batch = np.arange(8).reshape((8, 1))  # 8 items per batch
    targets_batch = np.pi * np.ones_like(inputs_batch)
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    loss_layer = tl.Serial(tl.Dense(1), tl.L2Loss())
    loss_layer.init(labeled_batch)
    optimizer = optimizers.SGD(.01)
    optimizer.tree_init(loss_layer.weights)
    trainer = optimizers.Trainer(loss_layer, optimizer)
    rng = fastmath.random.get_prng(0)
    trainer.one_step(labeled_batch, rng)

  def test_run_simple_task_tfnp(self):
    """Runs an accelerated optimizer on a simple task, TFNP backend."""
    with fastmath.use_backend(fastmath.Backend.TFNP):
      inputs_batch = np.arange(8).reshape((8, 1))  # 8 items per batch
      targets_batch = np.pi * np.ones_like(inputs_batch)
      labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
      loss_layer = tl.Serial(tl.Dense(1), tl.L2Loss())
      loss_layer.init(labeled_batch)
      optimizer = optimizers.Adam(.01)
      optimizer.tree_init(loss_layer.weights)
      trainer = optimizers.Trainer(loss_layer, optimizer)
      rng = fastmath.random.get_prng(0)
      trainer.one_step(labeled_batch, rng)


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
