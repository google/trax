# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for trax.layers.research.efficient_attention."""

import functools
from absl.testing import parameterized
import jax
import numpy as np
from tensorflow import test

from trax import fastmath
from trax import shapes
import trax.layers as tl
from tests.layers import test_utils
from trax.layers.research import sparsity


class EfficientFeedForwardTest(test.TestCase, parameterized.TestCase):
    def test_blocksparse_ff_train(self):
        d_model = 1024
        n_experts = 64
        d_ff = d_model * 8
        x_shape = (3, 7, d_model)
        with fastmath.use_backend(fastmath.Backend.JAX):
            layer = sparsity.BlockSparseFF(
                d_ff=d_ff, n_experts=n_experts, temperature=0.7, mode="train"
            )
            x = np.ones(x_shape).astype(np.float32)
            _, _ = layer.init(shapes.signature(x))
            y = layer(x)
            self.assertEqual(y.shape, x.shape)

    def test_blocksparse_ff_predict_equals_eval(self):
        d_model = 1024
        n_experts = 64
        d_ff = d_model * 8
        x_shape = (1, 1, d_model)
        temperature = 0.7
        with fastmath.use_backend(fastmath.Backend.JAX):
            x = np.ones(x_shape).astype(np.float32)
            input_signature = shapes.signature(x)
            common_kwargs = dict(
                d_ff=d_ff,
                n_experts=n_experts,
                temperature=temperature,
            )
            eval_model = sparsity.BlockSparseFF(mode="eval", **common_kwargs)
            weights, state = eval_model.init(input_signature)
            eval_out, _ = eval_model.pure_fn(
                x, weights, state, rng=jax.random.PRNGKey(0)
            )
            pred_model = sparsity.BlockSparseFF(mode="predict", **common_kwargs)
            _, _ = pred_model.init(input_signature)
            pred_out, _ = pred_model.pure_fn(
                x, weights, state, rng=jax.random.PRNGKey(0)
            )
            self.assertEqual(eval_out.shape, x.shape)
            # eval_out and pred_out should be identical.
            np.testing.assert_array_almost_equal(eval_out[0, 0, :], pred_out[0, 0, :])

    def test_sparse_ff_predict_equals_eval(self):
        with fastmath.use_backend(fastmath.Backend.JAX):
            d_model = 64
            seq_len = 6
            x_shape = (1, seq_len, d_model)
            inp = np.ones(x_shape).astype(np.float32)

            model_fn = functools.partial(
                sparsity.SparseFF,
                d_ff=256,
                temperature=0.7,
                n_elements_in_block=8,
            )

            configs = [
                {"multiply_by_controller_output": True},
                {"multiply_by_controller_output": False},
                {"ff_chunk_size": 2},
            ]

            test_utils.test_eval_equals_predict_configs(inp, model_fn, configs)

    @parameterized.named_parameters(
        ("_mode_train", "train"), ("_mode_eval", "eval"), ("_mode_predict", "predict")
    )
    def test_sparse_ff_with_chunking(self, mode):
        d_model = 8
        n_elements_in_block = 2
        d_ff = 16
        x_shape = (2, 8, d_model)
        temperature = 0.7
        with fastmath.use_backend(fastmath.Backend.JAX):
            x = np.ones(x_shape).astype(np.float32)
            input_signature = shapes.signature(x)
            model = sparsity.SparseFF(
                d_ff=d_ff,
                n_elements_in_block=n_elements_in_block,
                temperature=temperature,
                ff_chunk_size=4,
                mode=mode,
            )
            weights, state = model.init(input_signature)
            out, _ = model.pure_fn(x, weights, state, rng=jax.random.PRNGKey(0))
            self.assertEqual(out.shape, x.shape)

    @parameterized.named_parameters(
        ("_mode_train", "train"), ("_mode_eval", "eval"), ("_mode_predict", "predict")
    )
    def test_sparse_ff_multiply(self, mode):
        d_model = 8
        n_elements_in_block = 2
        d_ff = 16
        x_shape = (2, 8, d_model)
        temperature = 0.7
        with fastmath.use_backend(fastmath.Backend.JAX):
            x = np.ones(x_shape).astype(np.float32)
            input_signature = shapes.signature(x)
            model = sparsity.SparseFF(
                d_ff=d_ff,
                n_elements_in_block=n_elements_in_block,
                temperature=temperature,
                ff_chunk_size=4,
                mode=mode,
                multiply_by_controller_output=True,
            )
            weights, state = model.init(input_signature)
            out, _ = model.pure_fn(x, weights, state, rng=jax.random.PRNGKey(0))
            self.assertEqual(out.shape, x.shape)

    def test_sparse_ff_kernel_scaling(self):
        d_model = 8
        n_elements_in_block = 2
        d_ff = 16
        x_shape = (2, 8, d_model)
        temperature = 0.7
        with fastmath.use_backend(fastmath.Backend.JAX):
            x = np.ones(x_shape).astype(np.float32)
            input_signature = shapes.signature(x)
            model = sparsity.SparseFF(
                d_ff=d_ff,
                n_elements_in_block=n_elements_in_block,
                temperature=temperature,
                ff_chunk_size=4,
                mode="train",
                kernel_scaling=True,
            )
            weights, state = model.init(input_signature)
            out, _ = model.pure_fn(x, weights, state, rng=jax.random.PRNGKey(0))
            self.assertEqual(out.shape, x.shape)

    def test_switchsparse_ff_train(self):
        d_model = 1024
        n_experts = 64
        d_ff = d_model * 8
        x_shape = (3, 7, d_model)
        layer = sparsity.SwitchSparseFF(d_ff=d_ff, n_experts=n_experts, mode="train")
        x = np.ones(x_shape).astype(np.float32)
        layer.init(shapes.signature(x))
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_switchsparse_ff_predict_equals_eval(self):
        d_model = 1024
        n_experts = 64
        d_ff = d_model * 8
        x_shape = (1, 1, d_model)
        x = np.ones(x_shape).astype(np.float32)
        input_signature = shapes.signature(x)
        eval_model = sparsity.SwitchSparseFF(
            mode="eval", d_ff=d_ff, n_experts=n_experts
        )
        weights, state = eval_model.init(input_signature)
        eval_out, _ = eval_model.pure_fn(x, weights, state, rng=jax.random.PRNGKey(0))
        pred_model = sparsity.SwitchSparseFF(
            mode="predict", d_ff=d_ff, n_experts=n_experts
        )
        pred_model.init(input_signature)
        pred_out, _ = pred_model.pure_fn(x, weights, state, rng=jax.random.PRNGKey(0))
        self.assertEqual(eval_out.shape, x.shape)
        # eval_out and pred_out should be identical.
        np.testing.assert_array_almost_equal(eval_out[0, 0, :], pred_out[0, 0, :])


class ReversibleReshapePermuteTest(test.TestCase):
    def test_reversible_permute(self):
        layer = sparsity.ReversibleReshapePermute()
        x = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7]])
        layer.init(shapes.signature(x))
        ys = layer(x)
        self.assertEqual(
            tl.to_list(ys), [[1, 3, 5, 7, 2, 4, 6, 8], [0, 2, 4, 6, 1, 3, 5, 7]]
        )
        rev_x = layer.reverse(ys, weights=layer.weights)
        self.assertEqual(tl.to_list(x), tl.to_list(rev_x))


class ReversibleRandomPermuteTest(test.TestCase):
    def test_reversible_permute(self):
        layer = sparsity.ReversibleRandomPermute()
        x = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ]
        )
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
        layer = sparsity.LocallyConnectedDense(2, 8)
        x = np.array([[2, 5, 3, 4], [0, 1, 2, 3]])
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (2, 16))


class SparseDenseWithOptionsTest(test.TestCase):
    def test_simple_call(self):
        d_input, d_output = 16, 32
        settings = [
            (None, 0, 0, False),
            (None, 0, 0, True),
            ("einsum", 0, 0, False),
            ("lowrank", 0, 8, False),
            ("mult", 2, 0, False),
            ("mult", 2, 0, True),
            ("local", 2, 0, False),
            ("local3", 2, 0, False),
        ]
        for stype, sparsity_level, d_lowrank, use_bfloat16 in settings:
            layer = sparsity.SparseDenseWithOptions(
                d_output,
                d_input=d_input,
                sparsity_type=stype,
                sparsity=sparsity_level,
                d_lowrank=d_lowrank,
                use_bfloat16=use_bfloat16,
            )
            x = np.ones((1, 1, d_input))
            _, _ = layer.init(shapes.signature(x))
            y = layer(x)
            self.assertEqual(
                y.shape,
                (1, 1, d_output),
                msg="[{}->{}] {} - {} - {} - {}".format(
                    d_input, d_output, stype, sparsity_level, d_lowrank, use_bfloat16
                ),
            )


class ModularCausalAttentionTest(test.TestCase):
    def test_simple_call(self):
        layer = sparsity.ModularCausalAttention(d_feature=4, n_heads=2, sparsity=2)
        x = np.array(
            [
                [
                    [2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            ]
        )
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (1, 3, 4))


class LowRankCausalAttentionTest(test.TestCase):
    def test_simple_call(self):
        layer = sparsity.LowRankCausalAttention(d_feature=4, n_heads=2, lowrank=2)
        x = np.array(
            [
                [
                    [2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            ]
        )
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (1, 3, 4))


class MultiplicativeCausalAttentionTest(test.TestCase):
    def test_simple_call(self):
        layer = sparsity.MultiplicativeCausalAttention(
            d_feature=4, n_heads=2, sparsity=2
        )
        x = np.array(
            [
                [
                    [2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            ]
        )
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (1, 3, 4))


class MultiplicativeModularCausalAttentionTest(test.TestCase):
    def test_simple_call(self):
        layer = sparsity.MultiplicativeModularCausalAttention(
            d_feature=4, n_heads=2, sparsity=2
        )
        x = np.array(
            [
                [
                    [2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            ]
        )
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (1, 3, 4))


class MultiplicativeConvCausalAttentionTest(test.TestCase):
    def test_simple_call(self):
        layer = sparsity.MultiplicativeConvCausalAttention(
            d_feature=4, n_heads=2, sparsity=2
        )
        x = np.array(
            [
                [
                    [2, 5, 3, 4],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            ]
        )
        _, _ = layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (1, 3, 4))

    def test_various_calls(self):
        list_kwargs = []
        for share_qk in [True, False]:
            for output in ["none", "mult", "conv", "multconv"]:
                for concat in ["original", "fixed", "none"]:
                    kwargs = {
                        "share_qk": share_qk,
                        "output_layer_type": output,
                        "v_concat_type": concat,
                    }
                    list_kwargs.append(kwargs)
        for kwargs in list_kwargs:
            layer = sparsity.MultiplicativeConvCausalAttention(
                d_feature=4, n_heads=2, sparsity=2, **kwargs
            )
            x = np.array(
                [
                    [
                        [2, 5, 3, 4],
                        [0, 1, 2, 3],
                        [0, 1, 2, 3],
                    ]
                ]
            )
            _, _ = layer.init(shapes.signature(x))

            y = layer(x)
            self.assertEqual(y.shape, (1, 3, 4))

    def test_predict_equals_eval(self):
        with fastmath.use_backend(fastmath.Backend.JAX):
            d_model = 32
            seq_len = 5
            x_shape = (1, seq_len, d_model)
            inp = np.ones(x_shape).astype(np.float32)

            model_fn = functools.partial(
                sparsity.MultiplicativeConvCausalAttention,
                d_feature=d_model,
                n_heads=4,
                sparsity=4,
            )

            list_kwargs = []
            for share_qk in [True, False]:
                for output in ["none", "mult", "conv", "multconv"]:
                    for concat in ["original", "fixed", "none"]:
                        kwargs = {
                            "share_qk": share_qk,
                            "output_layer_type": output,
                            "v_concat_type": concat,
                        }
                        list_kwargs.append(kwargs)

            test_utils.test_eval_equals_predict_configs(inp, model_fn, list_kwargs)


class FavorTest(test.TestCase):
    def test_call_and_grad(self):
        layer_partial = tl.Serial(
            tl.Branch(tl.Embedding(3, 4), tl.PaddingMask()),
            sparsity.Favor(d_feature=4, n_heads=2),
            tl.Select([0], n_in=2),
        )
        layer = tl.Serial(
            tl.Branch(tl.Embedding(3, 4), tl.PaddingMask()),
            sparsity.Favor(d_feature=4, n_heads=2),
            tl.Select([0], n_in=2),
            tl.WeightedCategoryCrossEntropy(),
        )
        x = np.ones((1, 2), dtype=np.int32)
        w = np.ones_like(x).astype(np.float32)
        x_sig = shapes.signature(x)
        w_sig = shapes.signature(w)
        layer_partial.init(x_sig)
        y = layer_partial(x)
        self.assertEqual(y.shape, (1, 2, 4))
        layer.init((x_sig, x_sig, w_sig))
        y = layer((x, x, w))
        self.assertEqual(y.shape, ())
        state = layer.state
        rng = fastmath.random.get_prng(0)
        fwd = lambda weights, inp: layer.pure_fn(inp, weights, state, rng=rng)[0]
        g = fastmath.grad(fwd)(layer.weights, (x, x, w))
        self.assertEqual(g[0][1][0].shape, (3, 4))

    def test_call_and_grad_approximate_softmax(self):
        layer_partial = tl.Serial(
            tl.Branch(tl.Embedding(11, 12), tl.PaddingMask()),
            sparsity.Favor(
                d_feature=12,
                n_heads=3,
                n_random_features=128,
                use_approximate_softmax=True,
            ),
            tl.Select([0], n_in=2),
        )
        layer = tl.Serial(
            tl.Branch(tl.Embedding(11, 12), tl.PaddingMask()),
            sparsity.Favor(
                d_feature=12,
                n_heads=3,
                n_random_features=128,
                use_approximate_softmax=True,
            ),
            tl.Select([0], n_in=2),
            tl.WeightedCategoryCrossEntropy(),
        )
        x = np.ones((3, 5), dtype=np.int32)
        w = np.ones_like(x).astype(np.float32)
        x_sig = shapes.signature(x)
        w_sig = shapes.signature(w)
        layer_partial.init(x_sig)
        y = layer_partial(x)
        self.assertEqual(y.shape, (3, 5, 12))
        layer.init((x_sig, x_sig, w_sig))
        y = layer((x, x, w))
        self.assertEqual(y.shape, ())
        state = layer.state
        rng = fastmath.random.get_prng(0)
        fwd = lambda weights, inp: layer.pure_fn(inp, weights, state, rng=rng)[0]
        g = fastmath.grad(fwd)(layer.weights, (x, x, w))
        self.assertEqual(g[0][1][0].shape, (11, 12))

    def test_causal_call_and_grad(self):
        layer = tl.Serial(
            tl.Dense(4), sparsity.CausalFavor(d_feature=4, n_heads=2), tl.L2Loss()
        )
        x = np.random.uniform(size=(1, 2, 4)).astype(np.float32)
        w = np.ones_like(x)
        x_sig = shapes.signature(x)
        w_sig = shapes.signature(w)
        layer.init((x_sig, x_sig, w_sig))
        y = layer((x, x, w))
        self.assertEqual(y.shape, ())
        state = layer.state
        rng = fastmath.random.get_prng(0)
        fwd = lambda weights, inp: layer.pure_fn(inp, weights, state, rng=rng)[0]
        g = fastmath.grad(fwd)(layer.weights, (x, x, w))
        self.assertEqual(g[0][0].shape, (4, 4))


if __name__ == "__main__":
    test.main()
