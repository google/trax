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

"""Layer-Skipping Transformer Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax import layers as tl
from trax import math
from trax.layers.combinators import _inputs_from_stack
from trax.layers.combinators import _outputs_onto_stack
from trax.layers.combinators import _pop_rng_and_split
from trax.math import numpy as np
from trax.math import random
from trax.models import transformer


class SkippingSerial(tl.Serial):
  """Serial combinator that also skips layers."""

  def __init__(self, *sublayers, **kwargs):
    super(SkippingSerial, self).__init__(*sublayers)
    self._mode = kwargs.get('mode', 'train')
    # Ensure that each layer has the same number of inputs and outputs.
    if self.sublayers:
      n_in_out = self.sublayers[0].n_in
      for layer in self.sublayers:
        assert layer.n_in == n_in_out
        assert layer.n_out == n_in_out

  def forward_with_state(self, xs, weights=tl.EMPTY_WEIGHTS,
                         state=tl.EMPTY_STATE, **kwargs):
    self._validate_forward_inputs(xs)
    # Get N+1 rngs, N for running layers and one extra.
    rngs = _pop_rng_and_split(kwargs, self._n_layers + 1)
    rng0, rngs = rngs[0], rngs[1:]
    if not self.sublayers:  # No-op: leave args unchanged.
      return (xs, state)

    # Prepare the stack and do some safety checks as in the parent class.
    stack = xs
    new_state = []
    n_layers = self._n_layers
    if n_layers != 1 and len(weights) != n_layers:
      raise ValueError('number of weights ({}) not equal to number of layers '
                       '({})'.format(len(weights), n_layers))
    if n_layers != 1 and len(state) != n_layers:
      raise ValueError('length of state ({}) not equal to number of layers '
                       '({})'.format(len(state), n_layers))

    # TODO(chowdhery): try different strategies, also try running not all
    # layers backwards by using math.stop_gradient where needed.

    # Calculate how many layers to run forward.
    if self._mode == 'train':
      n_forward_layers = random.uniform(rng0, (), np.float32, 0.0, n_layers)
    else:
      n_forward_layers = float(n_layers)
    # Run layers skipping after a certain number.
    cur_layer_idx = 0.0
    for layer, p, s, rng in zip(self.sublayers, weights, state, rngs):
      inputs = _inputs_from_stack(layer, stack)
      # TODO(chowdhery): port to jax.lax.cond once it has a JVP rule.
      outputs, s = layer._forward_internal(inputs, p, s, rng)  # pylint: disable=protected-access
      condition = math.lt(cur_layer_idx, n_forward_layers).astype(np.float32)
      outputs = condition * outputs + (1 - condition) * inputs
      stack = _outputs_onto_stack(layer, outputs, stack)
      new_state.append(s)
      cur_layer_idx += 1.0
    return stack, new_state


def SkippingTransformerLM(vocab_size,
                          d_model=512,
                          d_ff=2048,
                          n_layers=6,
                          n_heads=8,
                          d_attention_key=None,
                          d_attention_value=None,
                          attention_type=tl.DotProductCausalAttention,
                          dropout=0.1,
                          share_qk=False,
                          max_len=2048,
                          mode='train',
                          ff_activation=tl.Relu):
  """Returns a Skipping Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head (default
      is d_model // n_heads)
    d_attention_value: int: depth of value vector for each attention head
      (default is d_model // n_heads)
    attention_type: subclass of tl.BaseCausalAttention: attention class to use
    dropout: float: dropout rate (how much to drop out)
    share_qk: bool, whether to share queries and keys in decoder attention
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    ff_activation: the non-linearity in feed-forward layer

  Returns:
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, name='embedding', mode=mode),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
  ]

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      embedder,
      SkippingSerial([
          transformer.DecoderBlock(  # pylint: disable=g-complex-comprehension
              d_model, d_ff, n_heads, d_attention_key, d_attention_value,
              attention_type, dropout, share_qk, i, mode, ff_activation)
          for i in range(n_layers)], mode=mode),
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )
