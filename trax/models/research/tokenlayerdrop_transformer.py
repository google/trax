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
"""Token-wise Layer-Dropping Transformer Models.

Why not.
"""

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers import initializers as init
from trax.models import transformer


def LessThan(val):
  """Checks if the input is less than a certain value."""
  return tl.Fn('LessThan', lambda x: x < val)


def MultGradient(value):
  return tl.Fn('MultGrad',
               lambda x: (x*value + fastmath.stop_gradient(x)*(1.0 - value)))


def TokenCond(mask, layer, shortcut=None):
  """Wraps a series of layers with a residual connection.

  Args:
    mask: Mask to be computed; it will be applied to a layer.
    layer: Layer to be applied in series.
    shortcut: If None (the usual case), the Residual layer computes the
        element-wise sum of the stack-top input with the output of the layer
        series. If specified, the `shortcut` layer applies to a copy of the
        inputs and (elementwise) adds its output to the output from the main
        layer series.

  Returns:
      A layer representing a residual connection paired with a layer series.
  """
  if shortcut is None:
    shortcut = tl.Serial()

  return tl.Serial(
      tl.Branch(
          mask,
          layer,
          shortcut),
      tl.Fn('MaskCond',
            lambda mask, layer, shortcut: mask * layer + (1.-mask) * shortcut)
  )


def ConfidenceDropTransformerLM(vocab_size,
                                d_model=512,
                                d_ff=2048,
                                n_layers=6,
                                n_heads=8,
                                dropout=0.1,
                                max_len=2048,
                                mode='train',
                                ff_activation=tl.Relu,
                                skip_threshold=0.5):
  """Returns a Confidence-basec ControllerDrop Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer
    skip_threshold: probability threshold for skipping a layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  learnable_part_of_decider = tl.Serial(
      tl.LayerNorm(),
      tl.Dense(vocab_size))

  final_decider = tl.Serial(
      learnable_part_of_decider,
      tl.LogSoftmax())

  def ConditionedBlock(layer_id):
    mask_decider = tl.Serial(
        learnable_part_of_decider,
        tl.StopGradient(),
        tl.Softmax(),
        tl.StateScalar('pr_sumprob_max_l{}'.format(layer_id),
                       lambda x: jnp.max(jnp.sum(x, axis=-1))),
        tl.StateScalar('pr_sumprob_mean_l{}'.format(layer_id),
                       lambda x: jnp.mean(jnp.sum(x, axis=-1))),
        tl.Max(),
        LessThan(skip_threshold),
        tl.StateScalar('pr_mean_decider_l{}'.format(layer_id)),
        )
    return TokenCond(
        mask_decider, transformer._DecoderBlock(  # pylint: disable=g-complex-comprehension,protected-access
            d_model, d_ff, n_heads, dropout, [], mode, ff_activation))

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      embedder,
      [ConditionedBlock(i) for i in range(n_layers)],
      final_decider
  )


def ControllerDropTransformerLM(vocab_size,
                                d_model=512,
                                d_ff=2048,
                                n_layers=6,
                                n_heads=8,
                                dropout=0.1,
                                max_len=2048,
                                mode='train',
                                ff_activation=tl.Relu):
  """Returns a Dense1Sigmoid ControllerDrop Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  final_decider = tl.Serial(
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax())

  def ConditionedBlock(layer_id):
    mask_decider = tl.Serial(
        tl.StopGradient(),
        tl.Dense(1, kernel_initializer=init.RandomNormalInitializer(0.),
                 bias_initializer=init.RandomNormalInitializer(0.),),
        # MultGradient(1./d_model),
        tl.Sigmoid(),
        tl.SummaryScalar('sigmoid_mean_l{}'.format(layer_id)),
        tl.SummaryScalar('sigmoid_std_l{}'.format(layer_id), jnp.std),
        )
    return TokenCond(
        mask_decider, transformer._DecoderBlock(  # pylint: disable=g-complex-comprehension,protected-access
            d_model, d_ff, n_heads, dropout, [], mode, ff_activation))

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      embedder,
      [ConditionedBlock(i) for i in range(n_layers)],
      final_decider
  )
