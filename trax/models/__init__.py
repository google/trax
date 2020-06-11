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

"""Models defined in trax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from trax.models.research import skipping_transformer


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
MLP = model_configure(mlp.MLP)
PureMLP = model_configure(mlp.PureMLP)
NeuralGPU = model_configure(neural_gpu.NeuralGPU)
Reformer = model_configure(reformer.Reformer)
ReformerLM = model_configure(reformer.ReformerLM)
ReformerShortenLM = model_configure(reformer.ReformerShortenLM)
ReformerNoEncDecAttention = model_configure(reformer.ReformerNoEncDecAttention)
Resnet50 = model_configure(resnet.Resnet50)
SkippingTransformerLM = model_configure(
    skipping_transformer.SkippingTransformerLM)
Transformer = model_configure(transformer.Transformer)
TransformerDecoder = model_configure(transformer.TransformerDecoder)
TransformerEncoder = model_configure(transformer.TransformerEncoder)
TransformerLM = model_configure(transformer.TransformerLM)
TransformerNoEncDecAttention = model_configure(
    transformer.TransformerNoEncDecAttention)
WideResnet = model_configure(resnet.WideResnet)
Policy = model_configure(rl.Policy)
PolicyAndValue = model_configure(rl.PolicyAndValue)
Value = model_configure(rl.Value)
RNNLM = model_configure(rnn.RNNLM)
GRULM = model_configure(rnn.GRULM)
LSTMSeq2SeqAttn = model_configure(rnn.LSTMSeq2SeqAttn)
