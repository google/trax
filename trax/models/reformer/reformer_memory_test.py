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
"""Test for memory usage in Reformer models.

This test is designed to run on TPUv3 hardware, processing 1 million tokens at a
time while just barely fitting within the 16 GB memory budget.
"""

import functools

from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import test_util  # pylint: disable=unused-import
from jax.config import config

import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.models.reformer import reformer


class ReformerMemoryTest(absltest.TestCase):

  def test_reformer_lm_memory(self):
    lsh_self_attention = functools.partial(
        tl.LSHSelfAttention,
        attention_dropout=0.0,
        chunk_len=64,
        n_buckets=[128, 128],
        n_chunks_after=0,
        n_chunks_before=1,
        n_hashes=1,
        n_parallel_heads=1,
        predict_drop_len=128,
        predict_mem_len=1024,
    )
    timebin_self_attention = functools.partial(
        tl.SelfAttention,
        attention_dropout=0.05,
        chunk_len=64,
        n_chunks_before=1,
        n_parallel_heads=1,
    )

    model = reformer.ReformerLM(
        vocab_size=256,
        d_model=256,
        d_ff=512,
        d_attention_key=64,
        d_attention_value=64,
        n_layers=6,
        n_heads=2,
        dropout=0.05,
        max_len=1048576,
        attention_type=[timebin_self_attention, lsh_self_attention],
        pos_axial_shape=(1024, 1024),
        pos_d_axial_embs=(64, 192),
        ff_activation=tl.Relu,
        ff_use_sru=0,
        ff_chunk_size=131072,
        mode='train',
    )
    x = np.ones((1, 1048576)).astype(np.int32)
    weights, state = model.init(shapes.signature(x))

    @jax.jit
    def mock_training_step(x, weights, state, rng):
      def compute_mock_loss(weights):
        logits, new_state = model.pure_fn(x, weights, state, rng)
        loss = jnp.mean(logits[..., 0])
        return loss, (new_state, logits)
      gradients, (new_state, logits) = jax.grad(
          compute_mock_loss, has_aux=True)(weights)
      new_weights = fastmath.nested_map_multiarg(
          lambda w, g: w - 1e-4 * g, weights, gradients)
      return new_weights, new_state, logits

    weights, state, logits = mock_training_step(
        x, weights, state, jax.random.PRNGKey(0))
    self.assertEqual(logits.shape, (1, 1048576, 256))


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
