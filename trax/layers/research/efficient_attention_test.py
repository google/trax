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
"""Tests for trax.layers.research.efficient_attention."""

import jax
import numpy as np
from tensorflow import test

from trax import fastmath
from trax import shapes
from trax.fastmath import numpy as jnp
import trax.layers as tl
from trax.layers.research import efficient_attention


class EfficientAttentionTest(test.TestCase):

  def test_self_attention(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.SelfAttention(
          n_heads=5, d_qk=7, d_v=17, share_qk=False, causal=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          use_reference_code=True, attention_dropout=0.0, mode='train')
      x = np.ones((3, 32, 8)).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_lsh_ff(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.LSHFF(d_ff=1024*8, n_buckets=[16, 8])
      x = np.ones((3, 7, 1024)).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_self_attention_tf(self):
    with fastmath.use_backend(fastmath.Backend.TFNP):
      layer = efficient_attention.SelfAttention(
          n_heads=5, d_qk=7, d_v=17, share_qk=False, causal=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          use_reference_code=True, attention_dropout=0.0, mode='train')
      x = np.ones((3, 32, 8)).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_lsh_self_attention(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.LSHSelfAttention(
          n_heads=5, d_qk=7, d_v=17, causal=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=2, n_buckets=4,
          use_reference_code=True, attention_dropout=0.0, mode='train')
      x = np.ones((3, 32, 8)).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_lsh_self_attention_tf(self):
    with fastmath.use_backend(fastmath.Backend.TFNP):
      layer = efficient_attention.LSHSelfAttention(
          n_heads=5, d_qk=7, d_v=17, causal=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=2, n_buckets=4,
          use_reference_code=True, attention_dropout=0.0, mode='train')
      x = np.ones((3, 32, 8)).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def _run_forward_and_backward(self, model, inp, weights, state):
    def forward(inp, weights):
      return model.pure_fn(
          inp, weights, state, rng=jax.random.PRNGKey(0))
    out, vjpfun, new_state = jax.vjp(forward, inp, weights, has_aux=True)
    inp_grad, weights_grad = vjpfun(np.ones_like(inp))
    return out, new_state, inp_grad, weights_grad

  def _test_equivalence_to_reference_code(
      self, model_cls, inp, input_signature, common_kwargs, *test_kwargs):
    ref_model = model_cls(use_reference_code=True, **common_kwargs)
    rng = fastmath.random.get_prng(123)
    weights, state = ref_model.init(input_signature, rng)

    ref_all = self._run_forward_and_backward(ref_model, inp, weights, state)
    ref_out, ref_state, ref_inp_grad, ref_weights_grad = ref_all

    for kwargs in test_kwargs:
      test_model = model_cls(**common_kwargs, **kwargs)
      state = test_model.init(input_signature, rng)[1]
      test_all = self._run_forward_and_backward(test_model, inp, weights, state)
      test_out, test_state, test_inp_grad, test_weights_grad = test_all

      self.assertEqual(jax.tree_structure(ref_out),
                       jax.tree_structure(test_out))
      self.assertEqual(jax.tree_structure(ref_state),
                       jax.tree_structure(test_state))
      self.assertEqual(jax.tree_structure(ref_inp_grad),
                       jax.tree_structure(test_inp_grad))
      self.assertEqual(jax.tree_structure(ref_weights_grad),
                       jax.tree_structure(test_weights_grad))

      check_close = lambda x, y: self.assertAllClose(x, y, rtol=1e-3, atol=1e-3)
      fastmath.nested_map_multiarg(check_close, ref_out, test_out)
      fastmath.nested_map_multiarg(check_close, ref_state, test_state)
      fastmath.nested_map_multiarg(check_close, ref_inp_grad, test_inp_grad)
      fastmath.nested_map_multiarg(check_close, ref_weights_grad,
                                   test_weights_grad)

  def test_batching_self_attention(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      common_kwargs = dict(
          n_heads=6, d_qk=7, d_v=17, share_qk=False, causal=True,
          chunk_len=5, n_chunks_before=1, n_chunks_after=0,
          attention_dropout=0.2, output_dropout=0.1, mode='train',
      )
      test_kwargs = []
      for n_parallel_heads in [1, 3, 6, 12]:
        for use_python_loop in [True, False]:
          test_kwargs.append(dict(n_parallel_heads=n_parallel_heads,
                                  use_python_loop=use_python_loop))

      x = jax.random.uniform(
          jax.random.PRNGKey(0), (2, 10, 13), dtype=jnp.float32)
      input_signature = shapes.signature(x)
      self._test_equivalence_to_reference_code(
          efficient_attention.SelfAttention,
          x, input_signature,
          common_kwargs, *test_kwargs)

  def test_batching_lsh_self_attention(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      common_kwargs = dict(
          n_heads=6, d_qk=7, d_v=17, causal=True,
          chunk_len=5, n_chunks_before=1, n_chunks_after=0,
          n_hashes=2, n_buckets=4,
          attention_dropout=0.2, output_dropout=0.1, mode='train',
      )
      test_kwargs = []
      for n_parallel_heads in [1, 3, 6, 12]:
        for use_python_loop in [True, False]:
          test_kwargs.append(dict(n_parallel_heads=n_parallel_heads,
                                  use_python_loop=use_python_loop))

      x = jax.random.uniform(
          jax.random.PRNGKey(0), (2, 10, 13), dtype=jnp.float32)
      input_signature = shapes.signature(x)
      self._test_equivalence_to_reference_code(
          efficient_attention.LSHSelfAttention,
          x, input_signature,
          common_kwargs, *test_kwargs)

  def _test_fast_inference(
      self, model_cls, x, input_signature, common_kwargs, *test_kwargs):
    ref_model = model_cls(use_reference_code=True, mode='eval', **common_kwargs)
    weights, state = ref_model.init(input_signature)

    ref_out, _ = ref_model.pure_fn(
        x, weights, state, rng=jax.random.PRNGKey(0))

    def get_slice(pytree, i):
      def get_slice_for_val(x):
        if isinstance(x, shapes.ShapeDtype):
          return shapes.ShapeDtype(shape=x.shape[:1] + (1,) + x.shape[2:],
                                   dtype=x.dtype)
        else:
          return x[:, i:i+1]
      return jax.tree_map(get_slice_for_val, pytree)

    seqlen = x[0].shape[1] if isinstance(x, (tuple, list)) else x.shape[1]

    for kwargs in test_kwargs:
      test_model = model_cls(mode='predict', **common_kwargs, **kwargs)
      cur_state = test_model.init(get_slice(input_signature, 0))[1]
      out = []
      for i in range(seqlen):
        cur_out, cur_state = test_model.pure_fn(
            get_slice(x, i), weights, cur_state, jax.random.PRNGKey(0))
        out.append(cur_out)
      out = jnp.concatenate(out, axis=1)

      self.assertAllClose(out, ref_out, rtol=1e-3, atol=1e-3)

  def test_fast_inference_self_attention(self):
    with fastmath.use_backend(fastmath.Backend.JAX):
      common_kwargs = dict(
          n_heads=6, d_qk=7, d_v=17, share_qk=False, causal=True,
          chunk_len=5, n_chunks_before=1, n_chunks_after=0,
          attention_dropout=0.0, output_dropout=0.0,
      )
      test_kwargs = []
      for n_parallel_heads in [1, 3, 6, 12]:
        for use_python_loop in [True, False]:
          test_kwargs.append(dict(n_parallel_heads=n_parallel_heads,
                                  use_python_loop=use_python_loop))

      x = jax.random.uniform(
          jax.random.PRNGKey(0), (2, 10, 13), dtype=jnp.float32)
      input_signature = shapes.signature(x)
      self._test_fast_inference(
          efficient_attention.SelfAttention,
          x, input_signature,
          common_kwargs, *test_kwargs)

  def _test_lsh_self_attention_deterministic_given_seed(self, causal=False):
    # Once the initialization and the call seeds are pinned down we have
    # deterministic output.
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.LSHSelfAttention(
          n_heads=5, d_qk=7, d_v=17, causal=causal,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=2, n_buckets=4,
          use_reference_code=True, attention_dropout=0.0, mode='train')
      x = np.ones((3, 32, 8)).astype(np.float32)

      def get_output():
        _, _ = layer.init(shapes.signature(x), jax.random.PRNGKey(0))
        return layer(x, rng=jax.random.PRNGKey(1))

      ys = [get_output() for _ in range(10)]

      self.assertEqual(ys[0].shape, x.shape)

      for y in ys[1:]:
        np.testing.assert_array_almost_equal(ys[0], y, decimal=6)

  def test_lsh_determinism_causal(self):
    self._test_lsh_self_attention_deterministic_given_seed(causal=True)

  def test_lsh_determinism_non_causal(self):
    self._test_lsh_self_attention_deterministic_given_seed(causal=False)

  def test_lsh_self_attention_masked_non_causal(self):
    # Test that when the input that is in the masked area changes the attention
    # for the un-masked outputs doesn't change, but the masked region does
    # change.
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.LSHSelfAttention(
          n_heads=5, d_qk=7, d_v=17, causal=False, masked=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=2, n_buckets=4,
          use_reference_code=True, attention_dropout=0.0, mode='train')

      batch = 5
      max_len = 32
      hidden = 8

      x = np.random.uniform(size=(batch, max_len, hidden))
      mask = np.ones((batch, max_len)).astype(np.bool)
      rngs = jax.random.randint(
          jax.random.PRNGKey(0), (batch,), minval=1, maxval=max_len - 1)

      # Set some suffix of each mask[b] to 0.
      for i in range(batch):
        mask[i, rngs[i]:] = 0

      # Fix rngs and get the output for the LSH layer.
      def get_output(x, mask):
        xs = [x, mask]
        _, _ = layer.init(shapes.signature(xs), jax.random.PRNGKey(0))
        return layer(xs, rng=jax.random.PRNGKey(1))

      # Get the attention output for masked x.
      y = get_output(x, mask)

      # Change x, but only in the masked regions.
      for i in range(batch):
        x[i, rngs[i]:] = np.random.uniform(size=(max_len - rngs[i], hidden))

      y2 = get_output(x, mask)

      for i in range(batch):
        # y and y2 should be identical in the non-masked part.
        np.testing.assert_array_almost_equal(y[i, :rngs[i]], y2[i, :rngs[i]],
                                             decimal=6)

        # In the masked out part, they should be different.
        self.assertGreater(
            np.mean(np.abs(y[i, rngs[i]:] - y2[i, rngs[i]:])), 1e-5)


class EfficientFeedForwardTest(test.TestCase):

  def test_blocksparse_ff_train(self):
    d_model = 1024
    num_experts = 64
    d_ff = d_model * 8
    x_shape = (3, 7, d_model)
    with fastmath.use_backend(fastmath.Backend.JAX):
      layer = efficient_attention.BlockSparseFF(
          d_ff=d_ff, num_experts=num_experts, temperature=0.7, mode='train')
      x = np.ones(x_shape).astype(np.float32)
      _, _ = layer.init(shapes.signature(x))
      y = layer(x)
      self.assertEqual(y.shape, x.shape)

  def test_blocksparse_ff_predict_equals_eval(self):
    d_model = 1024
    num_experts = 64
    d_ff = d_model * 8
    x_shape = (1, 1, d_model)
    temperature = 0.7
    with fastmath.use_backend(fastmath.Backend.JAX):
      x = np.ones(x_shape).astype(np.float32)
      input_signature = shapes.signature(x)
      common_kwargs = dict(
          d_ff=d_ff,
          num_experts=num_experts,
          temperature=temperature,
      )
      eval_model = efficient_attention.BlockSparseFF(
          mode='eval', **common_kwargs)
      weights, state = eval_model.init(input_signature)
      eval_out, _ = eval_model.pure_fn(
          x, weights, state, rng=jax.random.PRNGKey(0))
      pred_model = efficient_attention.BlockSparseFF(
          mode='predict', **common_kwargs)
      _, _ = pred_model.init(input_signature)
      pred_out, _ = pred_model.pure_fn(
          x, weights, state, rng=jax.random.PRNGKey(0))
      self.assertEqual(eval_out.shape, x.shape)
      # eval_out and pred_out should be identical.
      np.testing.assert_array_almost_equal(eval_out[0, 0, :], pred_out[0, 0, :])


class ReversibleReshapePermuteTest(test.TestCase):

  def test_reversible_permute(self):
    layer = efficient_attention.ReversibleReshapePermute()
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    layer.init(shapes.signature(x))
    ys = layer(x)
    self.assertEqual(tl.to_list(ys), [
        [1, 3, 5, 7, 2, 4, 6, 8],
        [0, 2, 4, 6, 1, 3, 5, 7]])
    rev_x = layer.reverse(ys, weights=layer.weights)
    self.assertEqual(tl.to_list(x), tl.to_list(rev_x))


class ReversibleRandomPermuteTest(test.TestCase):

  def test_reversible_permute(self):
    layer = efficient_attention.ReversibleRandomPermute()
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13],
                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  ])
    layer.init(shapes.signature(x))
    ys = layer(x)
    # this assert will fail once per ~87B runs, but it's okay
    self.assertNotEqual(tl.to_list(ys), tl.to_list(x))

    self.assertEqual(tl.to_list(ys[0]), tl.to_list(ys[2]))
    self.assertNotEqual(tl.to_list(ys[0]), tl.to_list(ys[1]))
    rev_x = layer.reverse(ys, weights=layer.weights)
    self.assertEqual(tl.to_list(x), tl.to_list(rev_x))


class LocallyConnectedDenseTest(test.TestCase):

  def test_simple_call(self):
    layer = efficient_attention.LocallyConnectedDense(2, 8)
    x = np.array([[2, 5, 3, 4],
                  [0, 1, 2, 3]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (2, 16))


class ModularCausalAttentionTest(test.TestCase):

  def test_simple_call(self):
    layer = efficient_attention.ModularCausalAttention(
        d_feature=4, n_heads=2, n_modules=2)
    x = np.array([[[2, 5, 3, 4],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3],]])
    _, _ = layer.init(shapes.signature(x))

    y = layer(x)
    self.assertEqual(y.shape, (1, 3, 4))

if __name__ == '__main__':
  test.main()
