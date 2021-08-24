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
"""Rotary positional embeddings.

  Rotary positional embedding implementation, as described in:
  https://arxiv.org/pdf/2104.09864.pdf
"""

# from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.layers import core


def rotate(x):
  """Rotate function."""
  _, l, d = x.shape
  inv_freq = jnp.exp(jnp.arange(0, d, 2) * -(jnp.log(10000.0) / d))
  positions = jnp.arange(l)
  freqs = jnp.einsum('i,j->ij', positions, inv_freq)
  emb = jnp.concatenate((freqs, freqs), axis=-1)
  cos = jnp.cos(emb)
  sin = jnp.sin(emb)

  def mul(vecs, pos_emb):
    return jnp.einsum('bld,ld->bld', vecs, pos_emb)

  def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=x1.ndim - 1)

  return mul(x, cos) + mul(rotate_half(x), sin)


def Rotate():  # pylint: disable=invalid-name
  return core.Fn('Rotate', rotate)
