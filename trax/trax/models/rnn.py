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
"""RNNs."""

from trax import layers as tl
from trax.fastmath import numpy as jnp


def RNNLM(vocab_size,
          d_model=512,
          n_layers=2,
          rnn_cell=tl.LSTMCell,
          rnn_cell_d_state_multiplier=2,
          dropout=0.1,
          mode='train'):
  """Returns an RNN language model.

  The input to the model is a tensor of tokens (ints).

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding (n_units in the RNN cell)
    n_layers: int: number of RNN layers
    rnn_cell: the RNN cell
    rnn_cell_d_state_multiplier: how many times is RNN cell state larger
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

  Returns:
    An RNN language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  def MultiRNNCell():
    """Multi-layer RNN cell."""
    assert n_layers == 2
    return tl.Serial(
        tl.Parallel([], tl.Split(n_items=n_layers)),
        tl.SerialWithSideOutputs(
            [rnn_cell(n_units=d_model) for _ in range(n_layers)]),
        tl.Parallel([], tl.Concatenate(n_items=n_layers))
    )

  zero_state = tl.MakeZeroState(  # pylint: disable=no-value-for-parameter
      depth_multiplier=n_layers * rnn_cell_d_state_multiplier
  )

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(vocab_size, d_model),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Branch([], zero_state),
      tl.Scan(MultiRNNCell(), axis=1),
      tl.Select([0], n_in=2),  # Drop RNN state.
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )


def GRULM(vocab_size=256,
          d_model=512,
          n_layers=1,
          mode='train'):
  """Returns an GRU language model.

  The input to the model is a tensor of tokens (ints).

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding (n_units in the RNN cell)
    n_layers: int: number of RNN layers
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

  Returns:
    An RNN language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  return tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(vocab_size, d_model),
      [tl.GRU(d_model) for _ in range(n_layers)],
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )


def LSTMSeq2SeqAttn(input_vocab_size=256,
                    target_vocab_size=256,
                    d_model=512,
                    n_encoder_layers=2,
                    n_decoder_layers=2,
                    n_attention_heads=1,
                    attention_dropout=0.0,
                    mode='train'):
  """Returns an LSTM sequence-to-sequence model with attention.

  The input to the model is a pair (input tokens, target tokens), e.g.,
  an English sentence (tokenized) and its translation into German (tokenized).

  The model works as follows:

  * Input encoder runs on the input tokens and creates activations that
    are used as both keys and values in attention.
  * Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.
  * Attention runs on the queries, keys and values masking out input padding.
  * Decoder runs on the result, followed by a cross-entropy loss.

  Args:
    input_vocab_size: int: vocab size of the input
    target_vocab_size: int: vocab size of the target
    d_model: int:  depth of embedding (n_units in the LSTM cell)
    n_encoder_layers: int: number of LSTM layers in the encoder
    n_decoder_layers: int: number of LSTM layers in the decoder after attention
    n_attention_heads: int: number of attention heads
    attention_dropout: float, dropout for the attention layer
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

  Returns:
    An LSTM sequence-to-sequence model with attention.
  """
  input_encoder = tl.Serial(
      tl.Embedding(input_vocab_size, d_model),
      [tl.LSTM(d_model) for _ in range(n_encoder_layers)],
  )

  pre_attention_decoder = tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(target_vocab_size, d_model),
      tl.LSTM(d_model),
  )

  def PrepareAttentionInputs():
    """Layer that prepares queries, keys, values and mask for attention."""
    def F(encoder_activations, decoder_activations, input_tokens):
      keys = values = encoder_activations
      queries = decoder_activations
      # Mask is 1 where inputs are not padding (0) and 0 where they are padding.
      mask = (input_tokens != 0)
      # We need to add axes to the mask for attention heads and decoder length.
      mask = jnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
      # Broadcast so mask is [batch, 1 for heads, decoder-len, encoder-len].
      mask = mask + jnp.zeros((1, 1, decoder_activations.shape[1], 1))
      return queries, keys, values, mask
    return tl.Fn('PrepareAttentionInputs', F, n_out=4)

  return tl.Serial(              # in-toks, target-toks
      tl.Select([0, 1, 0, 1]),   # in-toks, target-toks, in-toks, target-toks
      tl.Parallel(input_encoder, pre_attention_decoder),
      PrepareAttentionInputs(),  # q, k, v, mask, target-toks
      tl.Residual(
          tl.AttentionQKV(d_model, n_heads=n_attention_heads,
                          dropout=attention_dropout, mode=mode)
      ),                         # decoder-vecs, mask, target-toks
      tl.Select([0, 2]),         # decoder-vecs, target-toks
      [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
      tl.Dense(target_vocab_size),
      tl.LogSoftmax()
  )
