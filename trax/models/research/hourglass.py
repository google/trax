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
"""Hourglass - a hierarchical Transformer language model."""

import trax.layers as tl
from trax.layers.research.rel_attention import get_rel_att_inputs
from trax.layers.research.rel_attention import RelativeAttentionWrapper
from trax.layers.research.resampling import AttentionResampling
from trax.layers.research.resampling import AveragePooling
from trax.layers.research.resampling import FeedForwardBlock
from trax.layers.research.resampling import LinearUpsampling
from trax.models.research.configurable_transformer import ApplyAttentionLayer


def _RelativeDecoderBlock(attention_type, d_model, d_ff, n_heads, dropout,
                          dropout_shared_axes, mode, ff_activation,
                          context_bias_layer, location_bias_layer,
                          total_pooling):
  """Returns a list of layers.

    The layers implement a Transformer decoder block with relative attention
  parametrization.

  The input to the block is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    attention_type: attention type.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will pass all
      values through unaltered.
    ff_activation: Type of activation function at the end of each block; must be
      an activation-type subclass of `Layer`.
    context_bias_layer: context bias layer.
    location_bias_layer: location bias layer.
    total_pooling: The combined pool size of previously used funnel blocks.

  Returns:
    A list of layers that maps (activations, att_vecs, mask) to
                               (activations, att_vecs, mask).
  """
  if attention_type == RelativeAttentionWrapper:
    attention = RelativeAttentionWrapper(
        d_model,
        n_heads,
        dropout,
        mode=mode,
        context_bias_layer=context_bias_layer,
        location_bias_layer=location_bias_layer,
        total_pooling=total_pooling)
  else:
    attention = ApplyAttentionLayer(
        attention_type,
        d_model,
        n_heads,
        d_model // n_heads,
        d_model // n_heads,
        causal=True,
        masked=False,
        attention_dropout=dropout,
        output_dropout=dropout,
        attention_chunk_size=0,  # Disables tl.Chunk in ApplyAttentionLayer.
        mode=mode,
    )

  feed_forward = FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes,
                                  mode, ff_activation)

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(  # vecs
          tl.LayerNorm(),
          attention,
          _Dropout(),
      ),  # vecs
      tl.Residual(
          tl.LayerNorm(),
          feed_forward,
          _Dropout(),
      ),  # vecs
  ]


def _parse_hierarchy(hierarchy_str):  # pylint: disable = invalid-name
  """Parse hierarchy for Hourglass definition."""
  levels = hierarchy_str.split(' ')
  if levels != levels[::-1]:
    raise ValueError('Hierarchy is not a palindrome')
  layer_level_pairs = [(x.split('@')) for x in levels[:1 + (len(levels) // 2)]]
  hierarchy_n_layers = [int(x[0]) for x in layer_level_pairs]
  total_sf_per_level = [int(x[1]) for x in layer_level_pairs]

  hierarchy_shorten_factors = []
  for current_sf, prev_sf in zip(total_sf_per_level,
                                 [1] + total_sf_per_level[:-1]):
    if current_sf % prev_sf != 0:
      raise ValueError(
          f'Hierarchy not divisible by previous level: {current_sf}, {prev_sf}')
    hierarchy_shorten_factors.append(current_sf // prev_sf)

  return hierarchy_n_layers, hierarchy_shorten_factors


def HourglassLM(vocab_size,
                d_model=512,
                d_ff=2048,
                vanilla_layers=(1, 1),
                hierarchy='6@3',
                n_heads=8,
                dropout=0.1,
                dropout_shared_axes=None,
                mode='train',
                ff_activation=tl.FastGelu,
                vanilla_attn_type=RelativeAttentionWrapper,
                middle_attn_type=RelativeAttentionWrapper,
                downsampling_fn=AttentionResampling,
                upsampling_fn=AttentionResampling,
                attention_downsampling_fn=AveragePooling,
                attention_upsampling_fn=LinearUpsampling):
  """Returns a hierarchical Transformer language model.

  This model performs autoregressive language modeling:

    - input: rank 2 tensor representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). The tensor
      elements are integers in `range(vocab_size)`, and `0` values mark padding
      positions.

    - output: rank 3 tensor representing a batch of log-probability
      distributions for each sequence position over possible token IDs;
      shape is (batch_size, sequence_length, `vocab_size`).

  This model uses only the decoder part of the overall Transformer.

  Args:
    vocab_size: Input vocabulary size -- each element of the input tensor should
      be an integer in `range(vocab_size)`. These integers typically represent
      token IDs from a vocabulary-based tokenizer.
    d_model: Final dimension of tensors at most points in the model, including
      the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each encoder
      block.
    vanilla_layers: (pre_layers, post_layers) tuple - number of full token-level
      Transformer decoder layers before and after shortening.
    hierarchy: string - shortening hierarchy, as described in the paper.
      Hierarchy levels must form a palindrome, e.g. '1@2 2@6 1@2'.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value when
      applying dropout within an encoder block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask. Sharing
      along batch and sequence axes (`dropout_shared_axes=(0,1)`) is a useful
      way to save memory and apply consistent masks to activation vectors at
      different sequence positions.
    mode: str: 'train' or 'eval'.
    ff_activation: Type of activation function at the end of each encoder block;
      must be an activation-type subclass of `Layer`.
    vanilla_attn_type: class: attention class such as SelfAttention to use in
      the layers before and after shortening (vanilla layers).
    middle_attn_type: class: attention class to use in the middle layers (these
      operating on the shortened sequence).
    downsampling_fn: function that takes full token-level vectors of length `l`
      and transforms them into `l` / `k` vectors, where `k` denotes
      `shorten_factor` parameter.
    upsampling_fn: function that takes shortened representations of a sequence,
      consisting of `l` / `k` vectors and transforms them into full token-level
      representations of length `l`.
    attention_downsampling_fn: Downsampling function that transforms token-level
      vectors into query vectors with reduced length. Necessary only when
      AttentionResampling is used as `downsampling_fn`.
    attention_upsampling_fn: Upsampling function for AttentionResampling. Valid
      only when AttentionResampling is used as a `upsampling_fn`.

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  assert mode != 'predict'  # For now, 'predict' mode is unsupported.
  hierarchy_n_layers, hierarchy_shorten_factors = _parse_hierarchy(hierarchy)

  token_encoder = [
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  ]

  context_bias_layer, location_bias_layer = get_rel_att_inputs(d_model, n_heads)

  n_pre_decoder_blocks, n_post_decoder_blocks = vanilla_layers

  def create_decoder_blocks(n_layers, total_pooling,  # pylint: disable = invalid-name
                            attention_type):
    decoder_blocks = [
        # pylint: disable=g-complex-comprehension
        _RelativeDecoderBlock(attention_type, d_model, d_ff, n_heads, dropout,
                              dropout_shared_axes, mode, ff_activation,
                              context_bias_layer, location_bias_layer,
                              total_pooling) for _ in range(n_layers)
    ]
    return decoder_blocks + [tl.LayerNorm()]

  def create_hourglass_valley(rest_shorten_factors, rest_n_funnel_blocks,  # pylint: disable = invalid-name
                              current_total_pooling):
    assert rest_shorten_factors
    assert len(rest_shorten_factors) == len(rest_n_funnel_blocks)

    current_sf = rest_shorten_factors[0]
    current_n_layers = rest_n_funnel_blocks[0]

    shortening_layer = downsampling_fn(
        current_sf,
        d_model,
        is_upsampling=False,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        dropout_shared_axes=dropout_shared_axes,
        mode=mode,
        ff_activation=ff_activation,
        context_bias_layer=context_bias_layer,
        location_bias_layer=location_bias_layer,
        total_pooling=current_total_pooling,
        resampling_fn=attention_downsampling_fn)

    upsampling_layer = upsampling_fn(
        current_sf,
        d_model=d_model,
        is_upsampling=True,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        dropout_shared_axes=dropout_shared_axes,
        mode=mode,
        ff_activation=ff_activation,
        context_bias_layer=context_bias_layer,
        location_bias_layer=location_bias_layer,
        total_pooling=current_total_pooling,
        resampling_fn=attention_upsampling_fn)

    if len(rest_shorten_factors) > 1:  # we need to go deeper again
      pre_stage_blocks = create_decoder_blocks(
          current_n_layers, current_total_pooling * current_sf,
          middle_attn_type)

      post_stage_blocks = create_decoder_blocks(
          current_n_layers, current_total_pooling * current_sf,
          middle_attn_type)

      return [
          tl.Dup(),
          tl.ShiftRight(current_sf - 1, mode=mode), shortening_layer,
          pre_stage_blocks, *create_hourglass_valley(
              rest_shorten_factors[1:], rest_n_funnel_blocks[1:],
              current_total_pooling * current_sf), post_stage_blocks,
          upsampling_layer,
          tl.LayerNorm(),
          tl.Add()
      ]
    else:
      blocks = create_decoder_blocks(current_n_layers,
                                     current_total_pooling * current_sf,
                                     middle_attn_type)

      return [
          tl.Dup(),
          tl.ShiftRight(current_sf - 1), shortening_layer, blocks,
          upsampling_layer,
          tl.LayerNorm(),
          tl.Add()
      ]

  pre_decoder_blocks = create_decoder_blocks(n_pre_decoder_blocks, 1,
                                             vanilla_attn_type)

  post_decoder_blocks = create_decoder_blocks(n_post_decoder_blocks, 1,
                                              vanilla_attn_type)

  valley = create_hourglass_valley(hierarchy_shorten_factors,
                                   hierarchy_n_layers, 1)

  # Assemble and return the model.
  return tl.Serial(  # tokens (or chunked tuple of tokens)
      tl.ShiftRight(mode=mode),  # toks
      token_encoder,  # vecs
      pre_decoder_blocks,  # vecs
      valley,  # shortened vecs
      post_decoder_blocks,  # vecs
      tl.Dense(vocab_size),  # vecs
  )
