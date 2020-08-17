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
"""Layer-Skipping Transformer Models.

Like in https://arxiv.org/pdf/1909.11556.pdf
"""

from trax import layers as tl
from trax.models import transformer


def LargerThan(val):
  """Checks if the input is larger than a certain value."""
  return tl.Fn('LargerThan', lambda x: x > val)


def LayerDropSkippingTransformerLM(vocab_size,
                                   d_model=512,
                                   d_ff=2048,
                                   n_layers=6,
                                   n_heads=8,
                                   dropout=0.1,
                                   max_len=2048,
                                   mode='train',
                                   ff_activation=tl.Relu,
                                   skip_fraction=0.4):
  """Returns a Skipping Transformer language model.

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
    skip_fraction: fraction of times to skip some layers

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  def ConditionedBlock(current_layer_num):
    return tl.Serial(
        # stack: embedding, n_layers_to_keep
        tl.Select([1, 0, 1]),  # n_layers_to_keep, embedding, n_layers_to_keep
        tl.Cond(
            # if n_layers_to_keep > current_layer_num
            LargerThan(float(current_layer_num)),
            # then: run block
            tl.Serial(transformer._DecoderBlock(  # pylint: disable=g-complex-comprehension,protected-access
                d_model, d_ff, n_heads, dropout, [], mode, ff_activation)),
            # else: run noop
            tl.Serial()
            )
        # stack: embedding, n_layers_to_keep
        )

  if mode == 'train':
    minimum_layers = 0.0
    maximum_layers = float(n_layers) / skip_fraction
  else:
    minimum_layers = maximum_layers = float(n_layers)

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      embedder,
      # stack: embedding
      tl.RandomUniform(minimum_layers, maximum_layers, sync=True),
      # stack: n_layers_to_keep, embedding
      tl.Swap(),
      # stack: embedding, n_layers_to_keep
      [ConditionedBlock(i) for i in range(n_layers)],
      # stack: embedding, n_layers_to_keep
      tl.Select([0], n_in=2),  # stack: embedding
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )


def LayerDropTransformerLM(vocab_size,
                           d_model=512,
                           d_ff=2048,
                           n_layers=6,
                           n_heads=8,
                           dropout=0.1,
                           max_len=2048,
                           mode='train',
                           ff_activation=tl.Relu,
                           skip_fraction=0.4):
  """Returns a LayerDrop Transformer language model.

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
    skip_fraction: probability of skipping a layer; it can be a single
        probability or a list of probabilities different for each layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  if not isinstance(skip_fraction, (list, tuple)):
    # If we don't get a list of skip_fractions we use the same skip_fraction
    # for each layer.
    skip_fraction = [skip_fraction for i in range(n_layers)]
  if len(skip_fraction) != n_layers:
    raise ValueError('n_layers ({}) must be equal to len(skip_fraction) ({})'
                     .format(n_layers, len(skip_fraction)))

  def ConditionedBlock(current_layer_num):
    return tl.Serial(
        # stack: embedding
        tl.RandomUniform(0., 1, sync=True),
        # stack: random_uniform, embedding
        tl.Cond(
            # if random_uniform > skip_fraction
            LargerThan(skip_fraction[current_layer_num] if mode == 'train'
                       else 0.0),
            # then: run block
            tl.Serial(transformer._DecoderBlock(  # pylint: disable=g-complex-comprehension,protected-access
                d_model, d_ff, n_heads, dropout, [], mode, ff_activation)),
            # else: run noop
            tl.Serial()
            )
        # stack: embedding
        )

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      embedder,
      [ConditionedBlock(i) for i in range(n_layers)],
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )
