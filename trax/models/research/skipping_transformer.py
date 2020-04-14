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
"""Layer-Skipping Transformer Models."""

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
    # Parameters for skipping: how many steps to warm-up, how often to skip.
    self._skipping_warmup_steps = kwargs.get('skipping_warmup_steps', 20000)
    self._skip_fraction = kwargs.get('skip_fraction', 0.4)
    # Ensure that each layer has the same number of inputs and outputs.
    if self.sublayers:
      n_in_out = self.sublayers[0].n_in
      for layer in self.sublayers:
        assert layer.n_in == n_in_out
        assert layer.n_out == n_in_out

  def new_weights_and_state(self, input_signature):
    """Add a step-counter to the state. Initialize with 0."""
    weights, state = super(SkippingSerial, self).new_weights_and_state(
        input_signature)
    return weights, (0, state)

  @tl.Layer.state.setter
  def state(self, state):
    """Recursively sets non-param state on this layer and all sublayers."""
    self._state = state
    n_layers = self._n_layers
    if n_layers != 1 and len(state[1]) != n_layers:
      raise ValueError(
          f'Number of state elements ({len(state[1])}) does not equal '
          f'number of sublayers ({n_layers}).')
    for layer, sublayer_state in zip(self.sublayers, state[1]):
      layer.state = sublayer_state

  def forward_with_state(self, xs, weights=tl.EMPTY_WEIGHTS,
                         state=tl.EMPTY_STATE, **kwargs):
    self._validate_forward_inputs(xs)
    (step, layers_state) = state
    # Get N+1 rngs, N for running layers and one extra.
    rngs = _pop_rng_and_split(kwargs, self._n_layers + 1)
    rng0, rngs = rngs[0], rngs[1:]
    if not self.sublayers:  # No-op: leave args unchanged.
      return (xs, (step + 1, layers_state))

    # Prepare the stack and do some safety checks as in the parent class.
    stack = xs
    new_state = []
    n_layers = self._n_layers
    if n_layers != 1 and len(weights) != n_layers:
      raise ValueError('number of weights ({}) not equal to number of layers '
                       '({})'.format(len(weights), n_layers))
    if n_layers != 1 and len(layers_state) != n_layers:
      raise ValueError('length of state ({}) not equal to number of layers '
                       '({})'.format(len(layers_state), n_layers))

    # TODO(chowdhery): try different strategies, also try running not all
    # layers backwards by using math.stop_gradient where needed.

    # Calculate how many layers to run forward.
    if self._mode == 'train':
      # warmup goes from 1.0 at start to 0.0 at skipping_warmup_steps and after
      w_steps = float(self._skipping_warmup_steps)
      warmup = np.maximum(0.0, (w_steps - step.astype(np.float32)) / w_steps)
      # low is the minimum number of layers to *not* skip, from n_layers to 0
      low = warmup * float(n_layers)
      # high should be so that (high - n_layers) / high = 1.0 - skip_fraction
      # because (high - n_layers) / high is the probability we're not skipping
      # (after warmup); so high - n_layers = high - high * skip_fraction
      high = float(n_layers) / self._skip_fraction
      # We want the same rng0 on all cores.
      if math.device_count() > 1:
        rng0 = math.psum(rng0, 'batch')
      n_forward_layers = random.uniform(rng0, (), np.float32, low, high)
    else:
      n_forward_layers = float(n_layers)
    # Run layers skipping after a certain number.
    cur_layer_idx = 0.0
    for layer, p, s, rng in zip(self.sublayers, weights, layers_state, rngs):
      inputs = _inputs_from_stack(layer, stack)
      outputs, s = math.cond(  # Skip (do identity) if > n_forward_layers.
          pred=(math.lt(cur_layer_idx, n_forward_layers)),
          true_operand=(inputs, p, s, rng),  # This tuple is t below.
          true_fun=(lambda t: layer.pure_fn(t[0], t[1], t[2], t[3])),  # pylint: disable=cell-var-from-loop
          false_operand=(inputs, p, s, rng),
          false_fun=(lambda t: (t[0], t[2])),  # return (inputs, state)
      )
      stack = _outputs_onto_stack(layer, outputs, stack)
      new_state.append(s)
      cur_layer_idx += 1.0
    return stack, (step + 1, new_state)


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
          transformer._DecoderBlock(  # pylint: disable=g-complex-comprehension,protected-access
              d_model, d_ff, n_heads, d_attention_key, d_attention_value,
              attention_type, dropout, share_qk, i, mode, ff_activation)
          for i in range(n_layers)], mode=mode),
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )
