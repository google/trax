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

import time
from absl.testing import absltest

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import optimizers
from trax import shapes
from trax.models.reformer import reformer


class TrainerTest(absltest.TestCase):

  def _assert_all_equal(self, t1, t2, tol=1e-5):
    def eq(x1, x2):
      diff = np.maximum(np.abs(x1 - x2) - tol, 0.0)
      self.assertLessEqual(np.sum(diff), 0.0,
                           msg=f'\n{x1}\n !=\n{x2}\n diff:\n{x1-x2}')
    fastmath.nested_map_multiarg(eq, t1, t2)

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

  def test_run_reversible_slots(self):
    """Tests that slots can be read and assigned in reversible trainer."""
    layers = [tl.Dense(4), tl.Dup()]
    rev_layers = [tl.ReversibleHalfResidual(tl.Dense(4)),
                  tl.ReversibleSwap()]
    loss_layer = tl.Serial(tl.Concatenate(), tl.Dense(4),
                           tl.LogSoftmax(), tl.CrossEntropyLoss())
    trainer = optimizers.ReversibleSerialTrainer(
        [(layers, rev_layers)], loss_layer, optimizers.Adam)
    slots = trainer.slots
    trainer.slots = slots
    self.assertEqual(slots, trainer.slots)

  def test_run_reversible_same_as_default_basic(self):
    """Runs the reversible trainer, check results are the same as default."""
    inputs_batch = np.arange(8).reshape((2, 4))
    targets_batch = 2 * inputs_batch
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    # We want to test rng propagation too, so adding some dropout layers.
    first_layer = tl.Serial(tl.Embedding(9, 4), tl.Dropout(0.5), tl.Dup())
    rev_layers = [tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.2)),
                  tl.ReversibleSwap(),
                  tl.ReversibleHalfResidual(tl.Dropout(0.5), tl.Dense(4)),
                  tl.ReversibleSwap()]
    loss_layer = tl.Serial(tl.Concatenate(), tl.Dense(19), tl.Dropout(0.3),
                           tl.LogSoftmax(), tl.CrossEntropyLoss())
    model = tl.Serial([first_layer] + rev_layers + [loss_layer])
    rng_init = fastmath.random.get_prng(12)
    model.init(labeled_batch, rng=rng_init)
    optimizer_fn = optimizers.Adam  # to test slots

    # Make 2 steps with the original trainer.
    optimizer = optimizer_fn()
    optimizer.tree_init(model.weights)
    trainer = optimizers.Trainer(model, optimizer)
    rng_step1 = fastmath.random.get_prng(7)
    rng_step2 = fastmath.random.get_prng(8)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
    first_layer_weights1 = first_layer.weights
    rev_layer0_weights1 = rev_layers[0].weights
    rev_layer2_weights1 = rev_layers[2].weights
    loss_layer_weights1 = loss_layer.weights

    # Now make 2 steps with reversible trainer.
    model.init(labeled_batch, rng=rng_init)
    trainer = optimizers.ReversibleSerialTrainer(
        [(first_layer.sublayers, rev_layers)], loss_layer, optimizer_fn)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)

    # Check that weights end up the same.
    self._assert_all_equal(loss_layer_weights1, loss_layer.weights)
    self._assert_all_equal(rev_layer2_weights1, rev_layers[2].weights)
    self._assert_all_equal(rev_layer0_weights1, rev_layers[0].weights)
    self._assert_all_equal(first_layer_weights1, first_layer.weights)

  def test_run_reversible_same_as_default_extended(self):
    """Runs the reversible trainer, check results are the same as default."""
    inputs_batch = np.arange(8).reshape((2, 4))
    targets_batch = 2 * inputs_batch
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    # We want to test rng propagation too, so adding some dropout layers.
    first_layer = tl.Serial(tl.Embedding(9, 4), tl.Dropout(0.5), tl.Dup())
    rev_layers1 = [tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.2)),
                   tl.ReversibleSwap(),
                   tl.ReversibleHalfResidual(tl.Dropout(0.5), tl.Dense(4)),
                   tl.ReversibleSwap()]
    mid_layer = tl.Serial(tl.Add(), tl.Dense(4), tl.Dup())
    rev_layers2 = [tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.3)),
                   tl.ReversibleSwap()]
    loss_layer = tl.Serial(tl.Concatenate(), tl.Dense(19), tl.Dropout(0.3),
                           tl.LogSoftmax(), tl.CrossEntropyLoss())
    model = tl.Serial([first_layer] + rev_layers1 + [mid_layer] +
                      rev_layers2 + [loss_layer])
    rng_init = fastmath.random.get_prng(12)
    model.init(labeled_batch, rng=rng_init)
    optimizer_fn = optimizers.Adam  # to test slots

    # Make 3 steps with the original trainer.
    optimizer = optimizer_fn()
    optimizer.tree_init(model.weights)
    trainer = optimizers.Trainer(model, optimizer)
    rng_step1 = fastmath.random.get_prng(7)
    rng_step2 = fastmath.random.get_prng(8)
    rng_step3 = fastmath.random.get_prng(9)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
    trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)
    first_layer_weights1 = first_layer.weights
    rev_layer12_weights1 = rev_layers1[2].weights
    mid_layer_weights1 = mid_layer.weights
    rev_layer20_weights1 = rev_layers2[0].weights
    loss_layer_weights1 = loss_layer.weights

    # Now make 3 steps with reversible trainer.
    model.init(labeled_batch, rng=rng_init)
    # TODO(lukaszkaiser): this test seems to fail with memoize_jit, why?
    trainer = optimizers.ReversibleSerialTrainer(
        [(first_layer.sublayers, rev_layers1),
         (mid_layer.sublayers, rev_layers2)],
        loss_layer, optimizer_fn, memoize_jit=False)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
    trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)

    # Check that weights end up the same.
    self._assert_all_equal(loss_layer_weights1, loss_layer.weights)
    self._assert_all_equal(rev_layer20_weights1, rev_layers2[0].weights)
    self._assert_all_equal(mid_layer_weights1, mid_layer.weights)
    self._assert_all_equal(rev_layer12_weights1, rev_layers1[2].weights)
    self._assert_all_equal(first_layer_weights1, first_layer.weights)

  def test_run_reversible_same_as_default_reformer2(self):
    """Runs the reversible trainer, check results are the same as default."""
    inputs_batch = np.arange(8).reshape((2, 4))
    targets_batch = 2 * inputs_batch
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    int_sig = shapes.ShapeDtype((2, 4), dtype=np.int32)
    input_sig = (int_sig, int_sig, int_sig)
    # We want to test rng propagation too, so adding some dropout layers.
    model = reformer.Reformer2(20, d_model=8, d_ff=16, n_heads=1,
                               n_encoder_layers=1, n_decoder_layers=1,
                               loss_sparsity=2)
    loss = tl.Serial(tl.LogSoftmax(), tl.CrossEntropyLoss())
    optimizer_fn = optimizers.SGD  # Adafactor
    blocks, loss_layer = optimizers.trainer.extract_reversible_blocks(
        [model, loss], loss_chunk_size=4)
    blocks_serial = [(tl.Serial(std), rev) for (std, rev) in blocks]
    model_with_loss = tl.Serial(blocks_serial + [loss_layer])
    rng_init = fastmath.random.get_prng(12)
    model_with_loss.init(input_sig, rng=rng_init)

    # Make 3 steps with the original trainer.
    optimizer = optimizer_fn()
    optimizer.tree_init(model_with_loss.weights)
    trainer = optimizers.Trainer(model_with_loss, optimizer)
    rng_step1 = fastmath.random.get_prng(7)
    rng_step2 = fastmath.random.get_prng(8)
    rng_step3 = fastmath.random.get_prng(9)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
    trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)
    first_weights = blocks_serial[0][0].weights
    first_rev_weights = blocks[0][1][0].weights
    loss_weights = loss_layer.weights

    # Now make 3 steps with reversible trainer.
    model_with_loss.init(input_sig, rng=rng_init)
    trainer = optimizers.ReversibleSerialTrainer(
        blocks, loss_layer, optimizer_fn)
    trainer.one_step(labeled_batch, rng_step1)
    trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
    trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)

    # Check that weights end up the same.
    self._assert_all_equal(loss_weights, loss_layer.weights)
    self._assert_all_equal(first_rev_weights, blocks[0][1][0].weights)
    self._assert_all_equal(first_weights, blocks_serial[0][0].weights)

  def test_run_reversible_large_weights(self):
    """Runs the reversible trainer with a lot of weights to test memory use."""
    # This test requires > 18GB RAM, only run on TPUs. It does pass on GPU
    # and CPU when you run it locally, but it's too big for unit-testing.
    ram_limited = True  # Set to False to run this test locally.
    if fastmath.device_count() == 1 and ram_limited:
      return

    # Create inputs and rngs.
    inputs_batch = np.arange(8).reshape((2, 4))
    targets_batch = inputs_batch
    labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
    first_layer = tl.Serial(tl.Embedding(9, 16*1024), tl.Dup())
    rng_init = fastmath.random.get_prng(12)
    rng_step = fastmath.random.get_prng(13)

    # Initialize layers.
    first_layer.init(labeled_batch, rng=rng_init)
    n_layers = 18  # 18 layers each 16K x 16K = 256M weights ~= 1GB, 18GB ram
    rev_layers = []
    int_shape = shapes.ShapeDtype((2, 4), dtype=np.int32)
    shape = shapes.ShapeDtype((2, 4, 16*1024))
    sig = (shape, shape)
    for _ in range(n_layers):
      layer = tl.ReversibleHalfResidual(tl.Dense(16*1024))
      layer.init(sig, rng=rng_init)
      layer.weights = tl.on_cpu(layer.weights)  # store weights in cpu memory
      rev_layers.append(layer)
      rev_layers.append(tl.ReversibleSwap())
    loss_layer = tl.Serial(tl.Concatenate(), tl.Dense(9),
                           tl.LogSoftmax(), tl.CrossEntropyLoss())
    loss_layer.init((shape, shape, int_shape, int_shape))
    optimizer_fn = optimizers.Adafactor

    # Make a step with reversible trainer.
    trainer = optimizers.ReversibleSerialTrainer(
        [(first_layer, rev_layers)], loss_layer, optimizer_fn)
    loss, _ = trainer.one_step(labeled_batch, rng_step)
    self.assertLess(float(loss.sum()), 10000.0)  # Just to get the loss.
    # Set to true to run again, e.g., for profiling.
    run_twice = False
    if run_twice:
      t = time.time()
      loss, _ = trainer.one_step(labeled_batch, rng_step)
      self.assertLess(float(loss.sum()), 10000.0)  # Just to get the loss.
      print('Took %.3f seconds to run, loss %s' % (time.time() - t, loss))


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
