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

"""Layers defined in trax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
# We create a flat layers.* namespace for uniform calling conventions as we
# upstream changes.
# pylint: disable=wildcard-import
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
from trax.layers.reversible import *
from trax.layers.rnn import *
from trax.layers.tracer import symbolic


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
FilterResponseNorm = layer_configure(FilterResponseNorm)
ThresholdedLinearUnit = layer_configure(ThresholdedLinearUnit)

DotProductCausalAttention = layer_configure(
    DotProductCausalAttention, blacklist=['mode'])
MemoryEfficientCausalAttention = layer_configure(
    MemoryEfficientCausalAttention, blacklist=['mode'])
TimeBinCausalAttention = layer_configure(
    TimeBinCausalAttention, blacklist=['mode'])
LSHCausalAttention = layer_configure(
    LSHCausalAttention, blacklist=['mode'])
