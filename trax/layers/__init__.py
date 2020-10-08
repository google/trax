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
"""Layers defined in trax."""

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
from trax.layers.initializers import *
from trax.layers.metrics import *
from trax.layers.normalization import *
from trax.layers.pooling import *
from trax.layers.research.efficient_attention import *
from trax.layers.research.position_encodings import *
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

CausalAttention = layer_configure(CausalAttention, blacklist=['mode'])
CausalFavor = layer_configure(CausalFavor, blacklist=['mode'])
DotProductCausalAttention = layer_configure(
    DotProductCausalAttention, blacklist=['mode'])
SelfAttention = layer_configure(SelfAttention, blacklist=['mode'])
ModularCausalAttention = layer_configure(ModularCausalAttention,
                                         blacklist=['mode'])
LSHSelfAttention = layer_configure(LSHSelfAttention, blacklist=['mode'])
EncDecAttention = layer_configure(EncDecAttention, blacklist=['mode'])

InfinitePositionalEncoding = layer_configure(
    InfinitePositionalEncoding, blacklist=['mode'])
TimeBinPositionalEncoding = layer_configure(
    TimeBinPositionalEncoding, blacklist=['mode'])

AtariConvInit = layer_configure(AtariConvInit)
