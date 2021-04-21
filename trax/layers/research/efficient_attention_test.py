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
"""Tests for trax.layers.research.efficient_attention."""

from absl.testing import parameterized
import jax
import numpy as np
from tensorflow import test

from trax import fastmath
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.layers.research import efficient_attention


class EfficientAttentionTest(test.TestCase, parameterized.TestCase):

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


  def _run_forward_and_backward(self, model, inp, weights, state):
    def forward(inp, weights):
      return model.pure_fn(
          inp, weights, state, rng=jax.random.PRNGKey(0))
    out, vjpfun, new_state = jax.vjp(forward, inp, weights, has_aux=True)
    inp_grad, weights_grad = vjpfun(fastmath.numpy.ones_like(inp))
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

      check_close = lambda x, y: self.assertAllClose(x, y, rtol=2e-3, atol=2e-3)
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

  @parameterized.named_parameters(('_weights_2', 2), ('_weights_3', 3))
  def test_pure_lsh_wrapper_causal_non_masked(self, num_weights):
    with fastmath.use_backend(fastmath.Backend.JAX):
      n_heads = 5
      batch, seqlen, d_head = 3, 32, 8
      n_hashes = 2
      d_model = n_heads * d_head
      layer = efficient_attention.PureLSHSelfAttentionWrapper(
          n_heads=n_heads, d_qk=d_head, d_v=d_head, causal=True, masked=False,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=n_hashes, n_buckets=4, bias=False,
          pure_lsh_implementation=efficient_attention.PureLSHSelfAttention,
          mode='train', num_weights=num_weights)

      rng = jax.random.PRNGKey(0)
      rng, x_rng = jax.random.split(rng)

      input_shape = (batch, seqlen, d_model)
      x = jax.random.uniform(x_rng, input_shape, dtype=jnp.float32)

      inp = x
      w, s = layer.init(shapes.signature(inp))
      o = layer(inp)

      # Get the actual weights.
      weights = fastmath.tree_leaves(w)
      # Assert number of weights is as expected, the extra 1 is for output.
      self.assertLen(weights, num_weights + 1)

      # Assert each weight is of the expected shape.
      for i in range(num_weights + 1):
        self.assertEqual(weights[i].shape, (d_model, d_model))

      # Test that the output and the input shape match.
      self.assertEqual(inp.shape, o.shape)

      # Assert state is the shape expected.
      state = fastmath.tree_leaves(s)
      self.assertLen(state, 2)
      # buckets
      self.assertEqual(state[0].shape, (batch * n_heads, n_hashes * seqlen))
      # rngs
      self.assertEqual(state[1].shape, (batch * n_heads, 2))

  @parameterized.named_parameters(('_weights_2', 2), ('_weights_3', 3))
  def test_pure_lsh_wrapper_non_causal_masked(self, num_weights):
    with fastmath.use_backend(fastmath.Backend.JAX):
      n_heads = 5
      batch, seqlen, d_head = 3, 32, 8
      num_weights = 2
      n_hashes = 2
      d_model = n_heads * d_head
      layer = efficient_attention.PureLSHSelfAttentionWrapper(
          n_heads=n_heads, d_qk=d_head, d_v=d_head, causal=False, masked=True,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=n_hashes, n_buckets=4, bias=False,
          pure_lsh_implementation=efficient_attention.PureLSHSelfAttention,
          mode='train', num_weights=num_weights)

      rng = jax.random.PRNGKey(0)
      rng, x_rng = jax.random.split(rng)

      input_shape = (batch, seqlen, d_model)
      x = jax.random.uniform(x_rng, input_shape, dtype=jnp.float32)
      mask = jnp.ones((batch, seqlen), dtype=jnp.int32)

      inp = (x, mask)
      w, s = layer.init(shapes.signature(inp))
      o = layer(inp)

      # Get the actual weights.
      weights = fastmath.tree_leaves(w)
      # Assert number of weights is as expected, the extra 1 is for output.
      self.assertLen(weights, num_weights + 1)

      # Assert each weight is of the expected shape.
      for i in range(num_weights + 1):
        self.assertEqual(weights[i].shape, (d_model, d_model))

      # Test that the output and the x's shape match.
      self.assertEqual(x.shape, o.shape)

      # Assert state is the shape expected.
      state = fastmath.tree_leaves(s)
      self.assertLen(state, 2)
      # buckets
      self.assertEqual(state[0].shape, (batch * n_heads, n_hashes * seqlen))
      # rngs
      self.assertEqual(state[1].shape, (batch * n_heads, 2))

  def test_lsh_and_pure_lsh_self_attention_equivalence(self):
    # Given the same weight matrices and random numbers, do these produce the
    # same output.
    with fastmath.use_backend(fastmath.Backend.JAX):
      n_heads = 4
      d_head = 4
      d_model = n_heads * d_head
      pure_lsh_layer = efficient_attention.PureLSHSelfAttention(
          n_heads=n_heads, d_qk=d_head, d_v=d_head, causal=True, masked=False,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=4, n_buckets=8,
          use_reference_code=False,
          attention_dropout=0.0,
          use_python_loop=True,
          bias=False, mode='train')
      lsh_layer = efficient_attention.LSHSelfAttention(
          n_heads=n_heads, d_qk=d_head, d_v=d_head, causal=True, masked=False,
          chunk_len=8, n_chunks_before=1, n_chunks_after=0,
          n_hashes=4, n_buckets=8,
          use_reference_code=False,
          attention_dropout=0.0,
          use_python_loop=True,
          mode='train')

      batch, seqlen = 3, 32
      input_shape = (batch, seqlen, d_model)

      x = jax.random.uniform(jax.random.PRNGKey(0), input_shape,
                             dtype=jnp.float32)
      lsh_layer_input = x

      call_rng = jax.random.PRNGKey(42)

      lsh_layer_weights, lsh_layer_state = lsh_layer.init(
          shapes.signature(lsh_layer_input))
      lsh_layer.rng = call_rng
      lsh_layer_output = lsh_layer(lsh_layer_input)

      # Shapes are: (n_heads, d_model, d_head), (n_heads, d_model, d_head),
      # (n_heads, d_head, d_model)
      # Abbreviated as - hmn, hmn, hnm
      w_qk, w_v, w_o = lsh_layer_weights

      qk = jnp.einsum('blm,hmn->bhln', x, w_qk)
      qk = qk.reshape((-1, qk.shape[2], qk.shape[3]))

      v = jnp.einsum('blm,hmn->bhln', x, w_v)
      v = v.reshape((-1, v.shape[2], v.shape[3]))

      pure_lsh_layer_input = (qk, v)
      _, _ = pure_lsh_layer.init(shapes.signature(pure_lsh_layer_input))
      pure_lsh_layer.rng = call_rng
      pure_lsh_layer.state = lsh_layer_state
      pure_lsh_layer_output = pure_lsh_layer(pure_lsh_layer_input)

      # b*h,l,n
      pure_lsh_layer_output = pure_lsh_layer_output.reshape(
          (batch, -1) + pure_lsh_layer_output.shape[1:])
      pure_lsh_layer_output_projected = (
          jnp.einsum('bhld,hdm->blm', pure_lsh_layer_output, w_o))

      diff = pure_lsh_layer_output_projected - lsh_layer_output
      avg_diff = jnp.sum(jnp.abs(diff)) / jnp.sum(jnp.ones_like(diff))

      self.assertLess(avg_diff, 1e-5)

if __name__ == '__main__':
  test.main()
