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
"""ResNet."""

from trax import layers as tl


def ConvBlock(kernel_size, filters, strides, norm, non_linearity,
              mode='train'):
  """ResNet convolutional striding block."""
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = [
      tl.Conv(filters1, (1, 1), strides),
      norm(mode=mode),
      non_linearity(),
      tl.Conv(filters2, (ks, ks), padding='SAME'),
      norm(mode=mode),
      non_linearity(),
      tl.Conv(filters3, (1, 1)),
      norm(mode=mode),
  ]
  shortcut = [
      tl.Conv(filters3, (1, 1), strides),
      norm(mode=mode),
  ]
  return [
      tl.Residual(main, shortcut=shortcut),
      non_linearity()
  ]


def IdentityBlock(kernel_size, filters, norm, non_linearity,
                  mode='train'):
  """ResNet identical size block."""
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = [
      tl.Conv(filters1, (1, 1)),
      norm(mode=mode),
      non_linearity(),
      tl.Conv(filters2, (ks, ks), padding='SAME'),
      norm(mode=mode),
      non_linearity(),
      tl.Conv(filters3, (1, 1)),
      norm(mode=mode),
  ]
  return [
      tl.Residual(main),
      non_linearity(),
  ]


def Resnet50(d_hidden=64, n_output_classes=1001, mode='train',
             norm=tl.BatchNorm,
             non_linearity=tl.Relu):
  """ResNet.

  Args:
    d_hidden: Dimensionality of the first hidden layer (multiplied later).
    n_output_classes: Number of distinct output classes.
    mode: Whether we are training or evaluating or doing inference.
    norm: `Layer` used for normalization, Ex: BatchNorm or
      FilterResponseNorm.
    non_linearity: `Layer` used as a non-linearity, Ex: If norm is
      BatchNorm then this is a Relu, otherwise for FilterResponseNorm this
      should be ThresholdedLinearUnit.

  Returns:
    The list of layers comprising a ResNet model with the given parameters.
  """

  # A ConvBlock configured with the given norm, non-linearity and mode.
  def Resnet50ConvBlock(filter_multiplier=1, strides=(2, 2)):
    filters = (
        [filter_multiplier * dim for dim in [d_hidden, d_hidden, 4 * d_hidden]])
    return ConvBlock(3, filters, strides, norm, non_linearity, mode)

  # Same as above for IdentityBlock.
  def Resnet50IdentityBlock(filter_multiplier=1):
    filters = (
        [filter_multiplier * dim for dim in [d_hidden, d_hidden, 4 * d_hidden]])
    return IdentityBlock(3, filters, norm, non_linearity, mode)

  return tl.Serial(
      tl.ToFloat(),
      tl.Conv(d_hidden, (7, 7), (2, 2), 'SAME'),
      norm(mode=mode),
      non_linearity(),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),
      Resnet50ConvBlock(strides=(1, 1)),
      [Resnet50IdentityBlock() for _ in range(2)],
      Resnet50ConvBlock(2),
      [Resnet50IdentityBlock(2) for _ in range(3)],
      Resnet50ConvBlock(4),
      [Resnet50IdentityBlock(4) for _ in range(5)],
      Resnet50ConvBlock(8),
      [Resnet50IdentityBlock(8) for _ in range(2)],
      tl.AvgPool(pool_size=(7, 7)),
      tl.Flatten(),
      tl.Dense(n_output_classes),
  )


def WideResnetBlock(channels, strides=(1, 1), bn_momentum=0.9, mode='train'):
  """WideResnet convolutional block."""
  return [
      tl.BatchNorm(momentum=bn_momentum, mode=mode),
      tl.Relu(),
      tl.Conv(channels, (3, 3), strides, padding='SAME'),
      tl.BatchNorm(momentum=bn_momentum, mode=mode),
      tl.Relu(),
      tl.Conv(channels, (3, 3), padding='SAME'),
  ]


def WideResnetGroup(n, channels, strides=(1, 1), bn_momentum=0.9, mode='train'):
  shortcut = [
      tl.Conv(channels, (3, 3), strides, padding='SAME'),
  ]
  return [
      tl.Residual(WideResnetBlock(channels, strides, bn_momentum=bn_momentum,
                                  mode=mode),
                  shortcut=shortcut),
      tl.Residual([WideResnetBlock(channels, (1, 1), bn_momentum=bn_momentum,
                                   mode=mode)
                   for _ in range(n - 1)]),
  ]


def WideResnet(n_blocks=3, widen_factor=1, n_output_classes=10, bn_momentum=0.9,
               mode='train'):
  """WideResnet from https://arxiv.org/pdf/1605.07146.pdf.

  Args:
    n_blocks: int, number of blocks in a group. total layers = 6n + 4.
    widen_factor: int, widening factor of each group. k=1 is vanilla resnet.
    n_output_classes: int, number of distinct output classes.
    bn_momentum: float, momentum in BatchNorm.
    mode: Whether we are training or evaluating or doing inference.

  Returns:
    The list of layers comprising a WideResnet model with the given parameters.
  """
  return tl.Serial(
      tl.ToFloat(),
      tl.Conv(16, (3, 3), padding='SAME'),
      WideResnetGroup(n_blocks, 16 * widen_factor, bn_momentum=bn_momentum,
                      mode=mode),
      WideResnetGroup(n_blocks, 32 * widen_factor, (2, 2),
                      bn_momentum=bn_momentum, mode=mode),
      WideResnetGroup(n_blocks, 64 * widen_factor, (2, 2),
                      bn_momentum=bn_momentum, mode=mode),
      tl.BatchNorm(momentum=bn_momentum, mode=mode),
      tl.Relu(),
      tl.AvgPool(pool_size=(8, 8)),
      tl.Flatten(),
      tl.Dense(n_output_classes),
  )
