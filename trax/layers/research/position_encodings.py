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
"""Experimenting with position encodings."""

import jax
from trax.layers import base as layer_base
from trax.layers import initializers as init
from trax.math import numpy as np


class FixedBasePositionalEncoding(layer_base.Layer):
  """Implements fixed-base positional encoding."""

  def __init__(self, base=16, n_digits=2, mode='train',
               initializer=init.RandomNormalInitializer(1e-6)):
    super(FixedBasePositionalEncoding, self).__init__()
    self._base = base
    self._n_digits = n_digits
    self._mode = mode
    self._initializer = initializer

  def forward_with_state(self, x, weights=layer_base.EMPTY_WEIGHTS,
                         state=layer_base.EMPTY_STATE, rng=None, **kwargs):
    length = np.shape(x)[1]
    max_pos = self._base**self._n_digits
    assert length < max_pos, 'length (%d) >= max_pos (%d)' % (length, max_pos)
    positions = np.arange(0, length)
    if self._mode == 'train':
      positions += jax.random.randint(rng, (), 0, max_pos - length)
    pos_embeddings = []
    cur_positions = positions
    for i in range(self._n_digits):
      cur_indices = np.mod(cur_positions, self._base)
      cur_positions //= self._base
      pos_embeddings.append(np.take(weights[i], cur_indices, axis=0))
    embeddings = np.concatenate(pos_embeddings, axis=-1)
    return (x + embeddings[None, :, :], state)

  def new_weights(self, input_signature):
    d_feature = input_signature.shape[-1]
    assert d_feature % self._n_digits == 0
    d_weight = d_feature // self._n_digits
    return tuple([self._initializer((self._base, d_weight), rng)
                  for rng in self.new_rngs(self._n_digits)])
