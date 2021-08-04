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

# pylint: disable=line-too-long
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
# pylint: disable=reimported
# pylint: disable=g-too-many-blank-lines
# pylint: disable=g-wrong-blank-lines
# pylint: disable=bad-whitespace
# pylint: disable=missing-function-docstring
# pylint: disable=g-inconsistent-quotes
# pylint: disable=redefined-outer-name
"""bash.

pip install git+git://github.com/google/trax.git@$master

mkdir /tmp/Terraformer
cd /tmp/Terraformer

download the following into /tmp/Terraformer:
https://storage.googleapis.com/trax-ml/vocabs/en_16k.subword
https://storage.cloud.google.com/trax-ml/terraformer/med/config.gin
https://storage.cloud.google.com/trax-ml/terraformer/med/model_200000.opt_slots0.npy.gz
https://storage.cloud.google.com/trax-ml/terraformer/med/model_200000.pkl.gz
https://storage.cloud.google.com/trax-ml/terraformer/med/model_200000.weights.npy.gz

"""


import sys
import time

import os
import random
import time
import numpy as np

import trax
from trax import layers as tl
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.supervised import training
from trax.layers.assert_shape import assert_shape


import copy
import functools
import gc
import os
import time
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np
import psutil
from tensorflow.compat.v2 import test

from trax import fastmath
from trax import layers as tl
from trax import models
from trax import shapes
from trax.supervised import decoding
import gin


# from colabtools import adhoc_import
import json
import gc
import jax
import numpy as np
import os
import time
import gin

import tensorflow_datasets as tfds


# from colabtools import adhoc_import
import functools

from trax.data import tf_inputs
import tensorflow_datasets as tfds
from t5.data import preprocessors as t5_processors
import t5.data

from trax import data
from trax import layers as tl
from trax import models
from trax import optimizers
from trax.data import inputs
from trax.supervised import lr_schedules
from trax.supervised import trainer_lib
from trax.rl import serialization_utils
from trax.rl import space_serializer
import math
from trax.fastmath import numpy as numpy_math
import trax


import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn
from trax.layers.research import sparsity

import functools
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.models.reformer import reformer
from trax.models.research import configurable_transformer as ct
from trax.models.research import transformer2 as t2

#####

og_PositionalEncoding = tl.PositionalEncoding

trax.layers.attention.PositionalEncoding = functools.partial(og_PositionalEncoding, d_feature=64)
trax.layers.PositionalEncoding = functools.partial(og_PositionalEncoding, d_feature=64)
tl.PositionalEncoding = functools.partial(og_PositionalEncoding, d_feature=64)


#####


import gin
gin.enter_interactive_mode()


def model_configure(*args, **kwargs):
  kwargs['module'] = 'trax.models'
  return gin.external_configurable(*args, **kwargs)

####

xm2a_main = '/tmp/Terraformer/model_200000.pkl.gz'
xm2a_weights = '/tmp/Terraformer/model_200000.weights.npy.gz'
xm2a_opt_slots = '/tmp/Terraformer/model_200000.opt_slots0.npy.gz'
xm2a_config = '/tmp/Terraformer/config.gin'

VOCAB_FILE = 'en_16k.subword'
VOCAB_DIR = '/tmp/Terraformer'

####

f = open(xm2a_config)
gin_config = list(f)
f.close()

# gin_config.append(
#     'DotProductCausalAttention.max_inference_length = 16384'
# )
og_DotProductCausalAttention = trax.layers.attention.DotProductCausalAttention
trax.layers.attention.DotProductCausalAttention = functools.partial(
    og_DotProductCausalAttention, max_inference_length=16384,
)

# gin_config.append(
#     '\nMixedLSHSelfAttention.std_length=16384'
# )

gin_config = [l for l in gin_config if 'mira' not in l]
gin_config = [l for l in gin_config if 'okenize' not in l]  # tokenize

gin_config = ''.join(gin_config)
gin.parse_config(gin_config)
gin.operative_config_str().split('\n')

print(gin_config)

####

def model(mode):
  return models.ConfigurableTerraformer(mode=mode)

# ####

padding_fun = trax.data.PadToLength(len_map={0: 15*1024, 1: 15*1024, 2: 15*1024},
                                    pad_value = {0: 0, 1: 0, 2:0})
# padding_fun = lambda x: x
# padding_fun = trax.data.PadToLength(len_map={0: 128, 1: 128, 2:128}, pad_value={0: 0, 1: 0, 2: 0}, multiple=True)


####


dataset = tfds.summarization.scientific_papers.ScientificPapers()
valid = tfds.load(name='scientific_papers/arxiv:1.1.1')['test']
index = 0
xarts = []
for x in valid:
  xarts.append(x)
  index += 1
  if index == 3:
    break

model_file = xm2a_main
shape11 = trax.shapes.ShapeDtype((1, 1), dtype=numpy_math.int32)
shape1l = trax.shapes.ShapeDtype((1, 15*1024), dtype=numpy_math.int32)

with trax.fastmath.use_backend(trax.fastmath.Backend.JAX):
  model = model(mode='eval')
  model.init_from_file(model_file, weights_only=True)
  # in mode='predict' use input_signature=(shape1l, shape11)
  old_state = model.state


# Decode the first article
xart = xarts[2]['article']
question = xart.numpy().decode()
# print(question[:512])

tokenized = next(padding_fun(trax.data.tokenize([question,], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100)))

def detokenize(x):
  return trax.data.detokenize(x, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR,
                              n_reserved_ids=100)

with trax.fastmath.use_backend(trax.fastmath.Backend.JAX):
  model.state = old_state
  counter, tokens, max_length = 0, [], 30
  for token in decoding.autoregressive_sample_stream(
      model, tokenized[None, :15*1024], batch_size=1, temperature=0.0,
      eval_mode=True, eval_min_length=1024):
    print(f'Token {counter}: "{detokenize(token)}" {token}')
    tokens.append(token[:, None])
    counter += 1
    if counter > max_length:
      break
  tokens = np.concatenate(tokens, axis=1)
  print(tokens)
  print(detokenize(tokens[0]))
