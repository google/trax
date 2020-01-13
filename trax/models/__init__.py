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

"""Models defined in trax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import six

from trax.models import atari_cnn
from trax.models import mlp
from trax.models import neural_gpu
from trax.models import resnet
from trax.models import rnn
from trax.models import transformer
from trax.models.reformer import reformer
from trax.models.research import skipping_transformer

if six.PY3:
  # uses @ notation:
  from trax.models.research import position_lookup_transformer  # pylint: disable=g-import-not-at-top


# Ginify
def model_configure(*args, **kwargs):
  kwargs['module'] = 'trax.models'
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
AtariCnn = model_configure(atari_cnn.AtariCnn)
FrameStackMLP = model_configure(atari_cnn.FrameStackMLP)
MLP = model_configure(mlp.MLP)
NeuralGPU = model_configure(neural_gpu.NeuralGPU)
ReformerLM = model_configure(reformer.ReformerLM)
ReformerShortenLM = model_configure(reformer.ReformerShortenLM)
Resnet50 = model_configure(resnet.Resnet50)
SkippingTransformerLM = model_configure(
    skipping_transformer.SkippingTransformerLM)
Transformer = model_configure(transformer.Transformer)
TransformerDecoder = model_configure(transformer.TransformerDecoder)
TransformerEncoder = model_configure(transformer.TransformerEncoder)
TransformerLM = model_configure(transformer.TransformerLM)
WideResnet = model_configure(resnet.WideResnet)
RNNLM = model_configure(rnn.RNNLM)


if six.PY3:
  PositionLookupTransformerLM = model_configure(
      position_lookup_transformer.PositionLookupTransformerLM)
