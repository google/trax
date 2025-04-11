# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Layers: trainable functions as neural network building blocks."""

import gin

# Still make other modules available in the namespace
from trax.layers.acceleration import *  # noqa: F403, F401

# Import explicitly instead of using star imports
# Activation functions
from trax.layers.activation_fns import (
    Exp,
    FastGelu,
    Gelu,
    HardSigmoid,
    HardTanh,
    Relu,
    Sigmoid,
    Softplus,
    Tanh,
    ThresholdedLinearUnit,
)
from trax.layers.assert_shape import *  # noqa: F403, F401

# Attention mechanisms
from trax.layers.attention import (
    Attention,
    CausalAttention,
    DotProductCausalAttention,
    PositionalEncoding,
)

# Base layers
from trax.layers.combinators import *  # noqa: F403, F401
from trax.layers.convolution import *  # noqa: F403, F401
from trax.layers.core import *  # noqa: F403, F401
from trax.layers.core import (
    LogSoftmax,
    Softmax,
)

# Convolution and related
from trax.layers.deconvolution import ConvTranspose

# Initializers
from trax.layers.initializers import AtariConvInit

# Metrics
from trax.layers.metrics import (
    CategoryCrossEntropy,
    CrossEntropyLossWithLogSoftmax,
    L2Loss,
    MacroAveragedFScore,
    SequenceAccuracy,
    WeightedCategoryAccuracy,
    WeightedCategoryCrossEntropy,
    WeightedFScore,
)

# Normalization layers
from trax.layers.normalization import (
    BatchNorm,
    FilterResponseNorm,
    LayerNorm,
)
from trax.layers.research.efficient_attention import *  # noqa: F403, F401
from trax.layers.research.efficient_attention import (
    EncDecAttention,
    LSHSelfAttention,
    MixedLSHSelfAttention,
    PureLSHSelfAttention,
    PureLSHSelfAttentionWrapper,
    SelfAttention,
)

# Position encodings
from trax.layers.research.position_encodings import (
    InfinitePositionalEncoding,
    TimeBinPositionalEncoding,
)

# Relative attention
from trax.layers.research.rel_attention import (
    RelativeAttentionLayer,
    RelativeAttentionLMLayer,
    RelativeAttentionWrapper,
)
from trax.layers.research.resampling import (
    AttentionResampling,
    AveragePooling,
    LinearPooling,
    LinearUpsampling,
    NaiveUpsampling,
    NoUpsampling,
)

# Positional embedding and resampling
from trax.layers.research.rotary_positional_embedding import Rotate
from trax.layers.research.sparsity import *  # noqa: F403, F401
from trax.layers.research.sparsity import (
    CausalFavor,
    CausalFavorAttention,
    ConvCausalAttention,
    Favor,
    FavorAttention,
    LowRankCausalAttention,
    ModularCausalAttention,
    MultiplicativeCausalAttention,
    MultiplicativeConvCausalAttention,
    MultiplicativeModularCausalAttention,
)
from trax.layers.reversible import *  # noqa: F403, F401

# RNN cells
from trax.layers.rnn import GRUCell, LSTMCell


# Ginify
def layer_configure(*args, **kwargs):
    kwargs["module"] = "trax.layers"
    return gin.external_configurable(*args, **kwargs)


# Configure all the imported functions with gin
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

Attention = layer_configure(Attention, denylist=["mode"])
CausalAttention = layer_configure(CausalAttention, denylist=["mode"])
FavorAttention = layer_configure(FavorAttention, denylist=["mode"])
Favor = layer_configure(Favor, denylist=["mode"])
CausalFavor = layer_configure(CausalFavor, denylist=["mode"])
CausalFavorAttention = layer_configure(CausalFavorAttention, denylist=["mode"])
DotProductCausalAttention = layer_configure(
    DotProductCausalAttention, denylist=["mode"]
)
SelfAttention = layer_configure(SelfAttention, denylist=["mode"])
ModularCausalAttention = layer_configure(ModularCausalAttention, denylist=["mode"])
LowRankCausalAttention = layer_configure(LowRankCausalAttention, denylist=["mode"])
MultiplicativeCausalAttention = layer_configure(
    MultiplicativeCausalAttention, denylist=["mode"]
)
MultiplicativeModularCausalAttention = layer_configure(
    MultiplicativeModularCausalAttention, denylist=["mode"]
)
ConvCausalAttention = layer_configure(ConvCausalAttention, denylist=["mode"])
MultiplicativeConvCausalAttention = layer_configure(
    MultiplicativeConvCausalAttention, denylist=["mode"]
)
ConvTranspose = layer_configure(ConvTranspose)
LSHSelfAttention = layer_configure(LSHSelfAttention, denylist=["mode"])
PureLSHSelfAttention = layer_configure(PureLSHSelfAttention, denylist=["mode"])
MixedLSHSelfAttention = layer_configure(MixedLSHSelfAttention, denylist=["mode"])
PureLSHSelfAttentionWrapper = layer_configure(
    PureLSHSelfAttentionWrapper, denylist=["mode"]
)
EncDecAttention = layer_configure(EncDecAttention, denylist=["mode"])

PositionalEncoding = layer_configure(PositionalEncoding, denylist=["mode"])
InfinitePositionalEncoding = layer_configure(
    InfinitePositionalEncoding, denylist=["mode"]
)
TimeBinPositionalEncoding = layer_configure(
    TimeBinPositionalEncoding, denylist=["mode"]
)

AtariConvInit = layer_configure(AtariConvInit)
CrossEntropyLossWithLogSoftmax = layer_configure(CrossEntropyLossWithLogSoftmax)
WeightedCategoryAccuracy = layer_configure(WeightedCategoryAccuracy)
SequenceAccuracy = layer_configure(SequenceAccuracy)
CategoryCrossEntropy = layer_configure(CategoryCrossEntropy)
WeightedCategoryCrossEntropy = layer_configure(WeightedCategoryCrossEntropy)
MacroAveragedFScore = layer_configure(MacroAveragedFScore)
WeightedFScore = layer_configure(WeightedFScore)
RelativeAttentionLayer = layer_configure(RelativeAttentionLayer)
RelativeAttentionLMLayer = layer_configure(RelativeAttentionLMLayer)
RelativeAttentionWrapper = layer_configure(RelativeAttentionWrapper)
Rotate = layer_configure(Rotate)
AveragePooling = layer_configure(AveragePooling)
LinearPooling = layer_configure(LinearPooling)
LinearUpsampling = layer_configure(LinearUpsampling)
NoUpsampling = layer_configure(NoUpsampling)
NaiveUpsampling = layer_configure(NaiveUpsampling)
AttentionResampling = layer_configure(AttentionResampling)

