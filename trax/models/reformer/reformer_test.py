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

"""Tests for Reformer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as onp

from trax import layers as tl
from trax import math
from trax.math import numpy as np
from trax.models.reformer import reformer
from trax.shapes import ShapeDtype


class PoisonOnRNGMismatchAttention(tl.BaseCausalAttention):
  """Fills gradients with NaNs if reverse rng does not match forward rng."""

  def new_weights_and_state(self, input_signature):
    state = math.random.get_prng(1)
    return self.new_weights(input_signature), state

  def forward_and_backward(self, inputs, ct, state, new_state, rng=None,
                           **kwargs):
    assert math.backend_name() == 'jax', (
        'JAX backend is required to use forward_and_backward.')

    if ct is not None and new_state is not tl.EMPTY_STATE:
      recovered_rng = new_state
      is_same = (rng[0] == recovered_rng[0]) & (rng[1] == recovered_rng[1])
      is_same = is_same.astype(np.float32)
      # Divides by zero if rngs are not the same, which results in NaNs.
      inputs = (inputs[0] / is_same, inputs[1] / is_same, inputs[2] / is_same)

    def _do_forward(x):  # pylint: disable=invalid-name
      res, _ = self.forward_with_state(x, state=state, rng=rng, **kwargs)
      return res
    output, vjpfun = jax.vjp(_do_forward, inputs)
    return output, vjpfun(ct)[0]

  def forward_with_state(self, inputs, weights=(), state=(),
                         rng=None, **kwargs):
    return inputs[2], rng


class ReformerTest(parameterized.TestCase):

  def test_reformer_lm_forward_shape(self):
    """Run the ReformerLM forward and check output shape."""
    vocab_size = 16
    input_sd = ShapeDtype((1, 8), np.int32)
    input_signature = (input_sd, input_sd)
    model = reformer.ReformerLM(
        vocab_size, d_model=32, d_ff=64,
        d_attention_key=16, d_attention_value=16, n_layers=1, n_heads=2,
        max_len=16, n_chunks=2, n_attention_chunks=1)
    final_shape = tl.check_shape_agreement(
        model, input_signature)
    self.assertEqual(((1, 8, 16), (1, 8, 16)), final_shape)

  def test_reformer_rng_consistency(self):
    with math.use_backend('jax'):
      vocab_size = 16
      batch_size = 1
      input_sd = ShapeDtype((batch_size, 8), np.int32)
      input_signature = (input_sd, input_sd)
      model = reformer.ReformerLM(
          vocab_size, d_model=32, d_ff=64,
          d_attention_key=16, d_attention_value=16, n_layers=1, n_heads=2,
          max_len=16, n_chunks=2, n_attention_chunks=1, mode='train',
          attention_type=PoisonOnRNGMismatchAttention)

      rng = math.random.get_prng(0)
      weights, state = model.init(input_signature)

      def dummy_loss_fn(weights):
        inputs = (np.zeros(input_sd.shape, dtype=np.int32),) * 2
        output = model(inputs, weights=weights, state=state, rng=rng)
        dummy_loss = math.numpy.sum(output[0])
        return dummy_loss

      grad_fn = math.grad(dummy_loss_fn)
      grads = grad_fn(weights)
      # PoisonOnRNGMismatchAttention uses NaNs to signal an rng mismatch.
      for grad in jax.tree_util.tree_leaves(grads):
        assert onp.all(onp.isfinite(grad))


if __name__ == '__main__':
  absltest.main()
