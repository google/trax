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

"""Models defined in trax."""
import gin

from trax.models import atari_cnn
from trax.models import mlp
from trax.models import neural_gpu
from trax.models import resnet
from trax.models import rl
from trax.models import rnn
from trax.models import transformer
from trax.models.reformer import reformer
from trax.models.research import bert
from trax.models.research import configurable_transformer
from trax.models.research import hourglass
from trax.models.research import layerdrop_transformer
from trax.models.research import rezero
from trax.models.research import rse
from trax.models.research import terraformer
from trax.models.research import transformer2


# Ginify
def model_configure(*args, **kwargs):
  kwargs['module'] = 'trax.models'
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
AtariCnn = model_configure(atari_cnn.AtariCnn)
AtariCnnBody = model_configure(atari_cnn.AtariCnnBody)
FrameStackMLP = model_configure(atari_cnn.FrameStackMLP)
BERT = model_configure(bert.BERT)
BERTClassifierHead = model_configure(bert.BERTClassifierHead)
BERTRegressionHead = model_configure(bert.BERTRegressionHead)
ConfigurableTerraformer = model_configure(terraformer.ConfigurableTerraformer)
ConfigurableTransformer = model_configure(
    configurable_transformer.ConfigurableTransformer)
ConfigurableTransformerEncoder = model_configure(
    configurable_transformer.ConfigurableTransformerEncoder)
ConfigurableTransformerLM = model_configure(
    configurable_transformer.ConfigurableTransformerLM)
MLP = model_configure(mlp.MLP)
NeuralGPU = model_configure(neural_gpu.NeuralGPU)
Reformer = model_configure(reformer.Reformer)
ReformerLM = model_configure(reformer.ReformerLM)
ReformerShortenLM = model_configure(reformer.ReformerShortenLM)
Resnet50 = model_configure(resnet.Resnet50)
ReZeroTransformer = model_configure(
    rezero.ReZeroTransformer)
ReZeroTransformerDecoder = model_configure(
    rezero.ReZeroTransformerDecoder)
ReZeroTransformerEncoder = model_configure(
    rezero.ReZeroTransformerEncoder)
ReZeroTransformerLM = model_configure(
    rezero.ReZeroTransformerLM)
SkippingTransformerLM = model_configure(
    layerdrop_transformer.SkippingTransformerLM)
LayerDropTransformerLM = model_configure(
    layerdrop_transformer.LayerDropTransformerLM)
EveryOtherLayerDropTransformerLM = model_configure(
    layerdrop_transformer.EveryOtherLayerDropTransformerLM)
Transformer = model_configure(transformer.Transformer)
TransformerDecoder = model_configure(transformer.TransformerDecoder)
TransformerEncoder = model_configure(transformer.TransformerEncoder)
TransformerLM = model_configure(transformer.TransformerLM)
Transformer2 = model_configure(
    transformer2.Transformer2)
WideResnet = model_configure(resnet.WideResnet)
Policy = model_configure(rl.Policy)
PolicyAndValue = model_configure(rl.PolicyAndValue)
Value = model_configure(rl.Value)
Quality = model_configure(rl.Quality)
RNNLM = model_configure(rnn.RNNLM)
GRULM = model_configure(rnn.GRULM)
LSTMSeq2SeqAttn = model_configure(rnn.LSTMSeq2SeqAttn)
ResidualShuffleExchange = model_configure(rse.ResidualShuffleExchange)
HourglassLM = model_configure(hourglass.HourglassLM)
