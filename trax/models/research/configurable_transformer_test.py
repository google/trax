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
"""Tests for Transformer models."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import optimizers
from trax import shapes
from trax.models.research import configurable_transformer as ct
from trax.supervised import training


class LayerDuplicateTest(parameterized.TestCase):

  def test_duplicate_dense(self):
    layer = tl.Serial(tl.Serial(), tl.Dense(8))
    layer_duplicate = ct.LayerDuplicate(layer)
    model = tl.Serial(tl.Branch(layer,
                                layer_duplicate),
                      tl.SubtractTop())
    x = np.ones((3, 5))
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 8))
    self.assertAlmostEqual(np.sum(y), 0.0)

  def test_duplicate_state(self):
    layer = tl.Serial(tl.Serial(), tl.SummaryScalar(name='Avg'))
    layer_duplicate1 = ct.LayerDuplicate(layer, own_state=True)
    layer_duplicate2 = ct.LayerDuplicate(layer, own_state=True)
    model = tl.Serial(layer,
                      layer_duplicate1,
                      tl.Fn('Mult2', lambda x: x*2),
                      layer_duplicate2,
                      )
    x = np.ones((3, 5))
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5))
    self.assertEqual(layer.state, layer_duplicate1.state)
    self.assertNotEqual(layer.state, layer_duplicate2.state)


class ConfigurableTransformerTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    vocab_size = 16
    model = ct.ConfigurableTransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    x = np.ones((3, 5)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, vocab_size))

  def test_transformer_duplicates_lm_forward_shape(self):
    vocab_size = 16
    x = np.ones((3, 5)).astype(np.int32)

    # Testing duplicates type: block
    model = ct.ConfigurableTransformerLM(
        vocab_size, duplicates=2, d_model=32, d_ff=64, n_layers=2,
        n_heads=2, duplicates_type='block')
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, vocab_size))

    # Testing duplicates type: model
    model = ct.ConfigurableTransformerLM(
        vocab_size, duplicates=2, d_model=32, d_ff=64, n_layers=2,
        n_heads=2, duplicates_type='model')
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 5, vocab_size))

  def test_train_save_restore_duplicated_block_transformer(self):
    """Saves and restores a checkpoint to check for equivalence."""
    vocab_size = 8
    task = training.TrainTask(
        _very_simple_transformer_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_transformer_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path

    def _make_model_and_session():
      m = ct.ConfigurableTransformerLM(
          vocab_size, d_model=4, d_ff=4, n_layers=1, n_heads=2, dropout=0.,
          ff_dropout=0., duplicates=1, duplicates_type='block')
      ts = training.Loop(m, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
      return m, ts

    model, training_session = _make_model_and_session()
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=1)
    training_session.save_checkpoint()
    model2, training_session2 = _make_model_and_session()

    x = np.ones((2, 2)).astype(np.int32)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertEqual(str(y1), str(y2))

    training_session2.run(n_steps=1)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertNotEqual(str(y1), str(y2))

  def test_train_save_restore_duplicated_model_transformer(self):
    """Saves and restores a checkpoint to check for equivalence."""
    vocab_size = 8
    task = training.TrainTask(
        _very_simple_transformer_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_transformer_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path

    def _make_model_and_session():
      m = ct.ConfigurableTransformerLM(
          vocab_size, d_model=4, d_ff=4, n_layers=1, n_heads=2, dropout=0.,
          ff_dropout=0., duplicates=1, duplicates_type='block')
      ts = training.Loop(m, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
      return m, ts

    model, training_session = _make_model_and_session()
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=1)
    training_session.save_checkpoint()
    model2, training_session2 = _make_model_and_session()

    x = np.ones((2, 2)).astype(np.int32)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertEqual(str(y1), str(y2))

    training_session2.run(n_steps=1)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertNotEqual(str(y1), str(y2))

  def test_train_save_standard_restore_duplicated(self):
    """Saves non-duplicated model and restores weights into duplicated model."""
    vocab_size = 8
    task = training.TrainTask(
        _very_simple_transformer_data(), tl.L2Loss(), optimizers.SGD(.01))
    eval_task = training.EvalTask(
        _very_simple_transformer_data(),  # deliberately re-using training data
        [tl.L2Loss()],
        metric_names=['SGD.L2Loss'])
    tmp_dir = self.create_tempdir().full_path

    def _make_model_and_session(duplicates):
      m = ct.ConfigurableTransformerLM(
          vocab_size, d_model=4, d_ff=4, n_layers=1, n_heads=2, dropout=0.,
          ff_dropout=0., duplicates=duplicates)
      ts = training.Loop(m, [task], eval_tasks=[eval_task],
                         eval_at=lambda step_n: step_n % 2 == 0,
                         output_dir=tmp_dir)
      return m, ts

    # standard model without repeating layers
    model, training_session = _make_model_and_session(0)
    self.assertEqual(0, training_session.step)
    training_session.run(n_steps=1)
    training_session.save_checkpoint()

    # model with repeating blocks
    model2, training_session2 = _make_model_and_session(1)
    x = np.ones((2, 2)).astype(np.int32)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    # Outputs shouldn't be equal!
    self.assertNotEqual(str(y1), str(y2))

    # Training should work.
    training_session2.run(n_steps=1)
    y1 = model(x, rng=fastmath.random.get_prng(0))
    y2 = model2(x, rng=fastmath.random.get_prng(0))
    self.assertNotEqual(str(y1), str(y2))

  def _test_transformer_forward_shape(self, input_vocab_size,
                                      output_vocab_size):
    model = ct.ConfigurableTransformer(
        input_vocab_size,
        output_vocab_size,
        d_model=32,
        d_ff=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=2)
    xs = [np.ones((3, 5)).astype(np.int32), np.ones((3, 5)).astype(np.int32)]
    _, _ = model.init(shapes.signature(xs))
    y, _ = model(xs)

    vocab_size = output_vocab_size or input_vocab_size
    self.assertEqual(y.shape, (3, 5, vocab_size))

  @parameterized.named_parameters(('same_vocab', 16, None),
                                  ('same_size', 16, 16),
                                  ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


  def _test_fast_inference(self, length):
    with fastmath.use_backend(fastmath.Backend.JAX):
      vocab_size = 16
      model_fn = functools.partial(
          ct.ConfigurableTransformerLM,
          vocab_size=vocab_size,
          d_model=4,
          d_ff=8,
          n_layers=2,
          n_heads=2,
      )
      model_slow = model_fn(mode='eval')
      model_fast = model_fn(mode='predict')
      rng = fastmath.random.get_prng(0)
      batch_size = 2
      input_signature = shapes.ShapeDtype((batch_size, 1), np.int32)
      # Given the same rng, both models initialize with the same parameters.
      model_slow.init(input_signature, rng)
      model_fast.init(input_signature, rng)

      buf = np.zeros((batch_size, length), dtype=np.int32)
      next_sym = np.zeros((batch_size, 1), dtype=np.int32)

      for index in range(length):
        logits_slow = model_slow(buf, rng=rng)
        logits_fast = model_fast(next_sym, rng=rng)
        np.testing.assert_array_almost_equal(
            logits_slow[:, index, :],
            logits_fast[:, 0, :],
            decimal=5,
        )
        next_sym = np.random.randint(vocab_size, size=(batch_size, 1))
        buf[:, index] = next_sym[:, 0]

  def test_dot_product_causal_attention_fast_inference(self):
    self._test_fast_inference(length=5)

  @parameterized.named_parameters(
      ('positional_encoding', None),
      ('fixed_base_positional_encoding', 'fixed-base'),
      ('infinite_positional_encoding', 'infinite'),
      ('infinite_affine_positional_encoding', 'infinite-affine'),
      ('axial_positional_encoding', (2, 16)))
  def test_positional_encoder(self, axial_pos_shape):
    # dim should divide FixedBasePositionalEncoding.n_digits
    batch, length, dim = 2, 32, 8
    input_shape = (batch, length, dim)
    vocab_size = 32
    x = np.random.randint(0, vocab_size - 1, input_shape)
    # should sum to dim
    d_axial_pos_embs = (4, 4)

    positional_encoding = ct.PositionalEncoder(
        'train', dropout=0.1, max_len=length, axial_pos_shape=axial_pos_shape,
        d_axial_pos_embs=d_axial_pos_embs)
    _, _ = positional_encoding.init(shapes.signature(x))
    y = positional_encoding(x)
    self.assertEqual(y.shape, input_shape)

  def test_embedding_and_positional_encodings(self):
    d_model = 16
    max_len = 32
    batch = 2
    input_shape = (batch, max_len)
    input_vocab_size = 32
    x = np.random.randint(0, input_vocab_size - 1, input_shape)

    in_encoder, out_encoder, output_vocab_size = (
        ct.EmbeddingAndPositionalEncodings(
            input_vocab_size,
            d_model,
            'train',
            0.1,
            [-2],
            max_len,
            output_vocab_size=None,
            axial_pos_shape=None,
            d_axial_pos_embs=None))

    self.assertEqual(output_vocab_size, input_vocab_size)

    model_in = tl.Serial(in_encoder)
    model_out = tl.Serial(out_encoder)

    model_in.init(shapes.signature(x))
    model_out.init(shapes.signature(x))

    y = model_in(x)
    self.assertEqual(y.shape, input_shape + (d_model,))

    y = model_in(x)
    self.assertEqual(y.shape, input_shape + (d_model,))


def _very_simple_transformer_data():
  """"Returns stream of labeled data that maps small integers to constant pi."""
  inputs_batch = np.ones((2, 2)).astype(np.int32)
  targets_batch = np.ones((2, 2, 8)).astype(np.int32)
  labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
  while True:
    yield labeled_batch


if __name__ == '__main__':
  absltest.main()
