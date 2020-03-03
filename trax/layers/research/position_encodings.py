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
from trax import math
from trax.layers import base as layer_base
from trax.layers import initializers as init
from trax.math import numpy as np


class FixedBasePositionalEncoding(layer_base.Layer):
  """Implements fixed-base positional encoding."""

  def __init__(self, bases=[11, 13, 14, 15], n_digits=8,  #  pylint: disable=dangerous-default-value
               start_from_zero_one_in=100, base_dropout_one_in=100,
               mode='train', initializer=init.RandomUniformInitializer(1e-4)):
    super(FixedBasePositionalEncoding, self).__init__()
    self._bases = bases
    self._n_digits = n_digits
    self._mode = mode
    self._initializer = initializer
    self._start_from_zero_one_in = start_from_zero_one_in
    self._base_dropout_one_in = base_dropout_one_in

  def forward_with_state(self, x, weights=layer_base.EMPTY_WEIGHTS,
                         state=layer_base.EMPTY_STATE, rng=None, **kwargs):
    batch_size, length = x.shape[0], x.shape[1]
    max_pos = min(self._bases)**self._n_digits
    rng1, rng2, rng3 = math.random.split(rng, 3)
    assert length < max_pos, 'length (%d) >= max_pos (%d)' % (length, max_pos)
    positions = np.arange(0, length)[None, :]
    if self._mode == 'train':
      # In 1% of training cases still start from 0 to be exactly as in eval.
      start_from_nonzero = jax.random.randint(
          rng1, (batch_size,), 0, self._start_from_zero_one_in)
      start_from_nonzero = np.minimum(1, start_from_nonzero)
      random_start = jax.random.randint(rng2, (batch_size,), 0, max_pos-length)
      random_start *= start_from_nonzero
      positions += random_start[:, None]
    res = []
    for bn, base in enumerate(self._bases):
      pos_embeddings = []
      cur_positions = positions
      for i in range(self._n_digits):
        cur_indices = np.mod(cur_positions, base)
        cur_positions = cur_positions // base
        s = weights[bn][i]
        pos_embeddings.append(cur_indices.astype(np.float32)[:, :, None] * s)
      embeddings = np.concatenate(pos_embeddings, axis=-1)
      if self._mode == 'train':
        base_dropout = jax.random.randint(
            rng3, (batch_size,), 0, self._base_dropout_one_in)
        base_dropout = np.minimum(1, base_dropout).astype(np.float32)
        embeddings *= base_dropout[:, None, None]
      res.append(embeddings)
    res = sum(res) + np.zeros_like(x)
    return np.concatenate([x, res], axis=-1), state

  def new_weights(self, input_signature):
    d_feature = input_signature.shape[-1]
    assert d_feature % self._n_digits == 0
    d_weight = d_feature // self._n_digits
    return [[self._initializer((1, d_weight), rng)
             for rng in self.new_rngs(self._n_digits)]
            for _ in self._bases]
