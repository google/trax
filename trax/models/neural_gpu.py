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
"""Implementation of the improved Neural GPU (NGPU)."""

from trax import layers as tl
from trax.fastmath import numpy as jnp


# TODO(ddohan): Combinator to add saturation costs to loss
def SaturationCost(x, limit=0.9):
  return jnp.minimum(0, jnp.abs(x) - limit)


def DiagonalGate():
  """Split channels in 3 parts. Shifts 1st and 3rd sections to left/right."""

  def f(x):  # pylint: disable=invalid-name
    # x : [batch, 1, length, depth]
    x = jnp.pad(x, [(0, 0), (0, 0), (1, 1), (0, 0)],
                mode='constant', constant_values=0.0)
    depth = x.shape[-1] // 3
    assert 3 * depth == x.shape[-1], ('Depth must be divisible by 3', depth,
                                      x.shape)
    xs = [
        x[:, :, :-2, :depth], x[:, :, 1:-1, depth:2 * depth],
        x[:, :, 2:, 2 * depth:3 * depth]
    ]
    return jnp.concatenate(xs, axis=3)
  return tl.Fn('DiagonalGate', f)


def ConvDiagonalGRU(units, kernel_size=(3, 3)):
  """Build convolutional GRU with diagonal gating as in ImprovedNGPU."""

  def BuildConv():
    return tl.Conv(filters=units, kernel_size=kernel_size, padding='SAME')

  return tl.GeneralGRUCell(
      candidate_transform=BuildConv,
      memory_transform_fn=DiagonalGate,
      gate_nonlinearity=tl.HardSigmoid,
      candidate_nonlinearity=tl.HardTanh)


def NeuralGPU(d_feature=96, steps=16, vocab_size=2, mode='train'):
  """Implementation of Neural GPU: https://arxiv.org/abs/1702.08727.

  Args:
    d_feature: Number of memory channels (dimensionality of feature embedding).
    steps: Number of times depthwise recurrence steps.
    vocab_size: Vocabulary size.
    mode: Whether we are training or evaluating or doing inference.

  Returns:
    A NeuralGPU Stax model.
  """
  del mode

  core = ConvDiagonalGRU(units=d_feature)
  return tl.Serial(
      tl.Embedding(vocab_size=vocab_size, d_feature=d_feature),
      [core] * steps,
      tl.Dense(vocab_size),
  )
