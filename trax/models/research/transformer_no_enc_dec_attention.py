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
"""Transformer variant -- no encoder-decoder attention."""

from trax import layers as tl
from trax.models import transformer


def TransformerNoEncDecAttention(input_vocab_size,
                                 output_vocab_size=None,
                                 d_model=512,
                                 d_ff=2048,
                                 n_encoder_layers=6,
                                 n_decoder_layers=6,
                                 n_heads=8,
                                 dropout=0.1,
                                 dropout_shared_axes=None,
                                 max_len=2048,
                                 mode='train',
                                 ff_activation=tl.Relu):
  """Returns a Transformer model.

  This model expects an input pair: target, source.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_encoder_layers: int: number of encoder layers
    n_decoder_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  def PositionalEncoder(vocab_size):  # tokens --> vectors
    return [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
        tl.PositionalEncoding(max_len=max_len),
    ]

  in_encoder = PositionalEncoder(input_vocab_size)
  out_encoder = (in_encoder if output_vocab_size is None
                 else PositionalEncoder(output_vocab_size))
  if output_vocab_size is None:
    output_vocab_size = input_vocab_size

  encoder_blocks = [
      transformer._EncoderBlock(d_model, d_ff, n_heads, dropout,  # pylint: disable=protected-access
                                dropout_shared_axes, mode, ff_activation)
      for i in range(n_encoder_layers)]

  encoder = tl.Serial(
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  decoder_blocks = [
      transformer._DecoderBlock(d_model, d_ff, n_heads, dropout,  # pylint: disable=protected-access
                                dropout_shared_axes, mode, ff_activation)
      for i in range(n_decoder_layers)]

  # pylint: disable=protected-access
  # Assemble and return the model.
  return tl.Serial(
      # Input: encoder_side_tokens, decoder_side_tokens
      # Copy decoder tokens for use in loss.
      tl.Select([0, 0, 1, 1]),          # tok_e tok_e tok_d tok_d

      # Encode.
      tl.Branch([], tl.PaddingMask()),  # tok_e mask_e tok_e tok_d tok_d
      encoder,                          # vec_e mask_e tok_e tok_d tok_d

      # Simple encoder mask, doesn't contain extra dims.
      tl.Select([2, 0, 2], n_in=3),     # tok_e vec_e tok_e tok_d tok_d
      transformer._MaskOfRightShiftedArray(
          n_shifts=0),                  # mask_e vec_e tok_e tok_d tok_d

      # Decode.
      tl.Select([3, 1, 0, 2]),          #  tok_d vec_e mask_e tok_e tok_d
      tl.ShiftRight(mode=mode),         # stok_d vec_e mask_e tok_e tok_d
      tl.Branch(
          [],
          transformer._MaskOfRightShiftedArray()
      ),                                # stok_d mask_d vec_e mask_e tok_e tok_d
      out_encoder,                      # svec_d mask_d vec_e mask_e tok_e tok_d

      # Concat encoder and decoder.
      tl.Select([2, 0, 3, 1]),          # vec_e svec_d mask_e mask_d tok_e tok_d
      transformer._ConcatWithPadding(),  # vec_ed tok_e tok_d

      # Decoder blocks with causal attention
      decoder_blocks,                   # vec_ed tok_e tok_d
      tl.LayerNorm(),                   # vec_ed tok_e tok_d

      # Separate out the encoder part from the concatenated vector.
      tl.Select([0, 1, 2, 2]),          # vec_ed tok_e tok_d tok_d
      transformer._StripFromConcatenateWithPadding(),  # vec_d tok_d

      # Map to output vocab.
      tl.Dense(output_vocab_size),      # vec_d tok_d
      tl.LogSoftmax(),                  # vec_d tok_d
  )
