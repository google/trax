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
"""Layers: trainable functions as neural network building blocks."""

import gin
# We create a flat layers.* namespace for uniform calling conventions as we
# upstream changes.
# pylint: disable=wildcard-import
from trax.layers.acceleration import *
from trax.layers.activation_fns import *
from trax.layers.assert_shape import *
from trax.layers.attention import *
from trax.layers.base import *
from trax.layers.combinators import *
from trax.layers.convolution import *
from trax.layers.core import *
from trax.layers.deconvolution import *
from trax.layers.initializers import *
from trax.layers.metrics import *
from trax.layers.normalization import *
from trax.layers.pooling import *
from trax.layers.research.efficient_attention import *
from trax.layers.research.position_encodings import *
from trax.layers.research.sparsity import *
from trax.layers.reversible import *
from trax.layers.rnn import *


# Ginify
def layer_configure(*args, **kwargs):
  kwargs['module'] = 'trax.layers'
  return gin.external_configurable(*args, **kwargs)

# pylint: disable=used-before-assignment
# pylint: disable=invalid-name
Relu = layer_configure(Relu)
Gelu = layer_configure(Gelu)
FastGelu = layer_configure(FastGelu)
Sigmoid = layer_configure(Sigmoid)
Tanh = layer_configure(Tanh)
HardSigmoid = layer_configure(HardSigmoid)
HardTanh = layer_configure(HardTanh)
Exp = layer_configure(Exp)
LogSoftmax = layer_configure(LogSoftmax)
Softmax = layer_configure(Softmax)
Softplus = layer_configure(Softplus)
L2Loss = layer_configure(L2Loss)
LSTMCell = layer_configure(LSTMCell)
GRUCell = layer_configure(GRUCell)

BatchNorm = layer_configure(BatchNorm)
LayerNorm = layer_configure(LayerNorm)
FilterResponseNorm = layer_configure(FilterResponseNorm)
ThresholdedLinearUnit = layer_configure(ThresholdedLinearUnit)

Attention = layer_configure(Attention, denylist=['mode'])
CausalAttention = layer_configure(CausalAttention, denylist=['mode'])
Favor = layer_configure(Favor, denylist=['mode'])
CausalFavor = layer_configure(CausalFavor, denylist=['mode'])
DotProductCausalAttention = layer_configure(
    DotProductCausalAttention, denylist=['mode'])
SelfAttention = layer_configure(SelfAttention, denylist=['mode'])
ModularCausalAttention = layer_configure(ModularCausalAttention,
                                         denylist=['mode'])
LowRankCausalAttention = layer_configure(LowRankCausalAttention,
                                         denylist=['mode'])
MultiplicativeCausalAttention = layer_configure(MultiplicativeCausalAttention,
                                                denylist=['mode'])
MultiplicativeModularCausalAttention = layer_configure(
    MultiplicativeModularCausalAttention, denylist=['mode'])
MultiplicativeConvCausalAttention = layer_configure(
    MultiplicativeConvCausalAttention, denylist=['mode'])
ConvTranspose = layer_configure(ConvTranspose)
LSHSelfAttention = layer_configure(LSHSelfAttention, denylist=['mode'])
EncDecAttention = layer_configure(EncDecAttention, denylist=['mode'])

InfinitePositionalEncoding = layer_configure(
    InfinitePositionalEncoding, denylist=['mode'])
TimeBinPositionalEncoding = layer_configure(
    TimeBinPositionalEncoding, denylist=['mode'])

AtariConvInit = layer_configure(AtariConvInit)
CrossEntropyLossWithLogSoftmax = layer_configure(CrossEntropyLossWithLogSoftmax)
WeightedCategoryAccuracy = layer_configure(WeightedCategoryAccuracy)
SequenceAccuracy = layer_configure(SequenceAccuracy)
CategoryCrossEntropy = layer_configure(CategoryCrossEntropy)
WeightedCategoryCrossEntropy = layer_configure(WeightedCategoryCrossEntropy)
