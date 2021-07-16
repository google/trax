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

xm2a_main = '/tmp/Terraformer/terraformer_med_model_200000.pkl.gz'
xm2a_weights = '/tmp/Terraformer/terraformer_med_model_200000.weights.npy.gz'
xm2a_opt_slots = '/tmp/Terraformer/terraformer_med_model_200000.opt_slots0.npy.gz'
xm2a_config = '/tmp/Terraformer/terraformer_med_config.gin'

VOCAB_FILE = 'en_16k.subword'
VOCAB_DIR = '/tmp/Terraformer'

####

f = open(xm2a_config)
gin_config = list(f)
f.close()
#
#  Uncomment this part to get the original results from the docs
#
# keep_gin = [l for l in gin_config if 'predict_mem' not in l]
# change_gin = [l for l in gin_config if 'predict_mem' in l]
# changed_gin = [l[:-6] + '2048\n' for l in change_gin]
# gin_config = keep_gin + changed_gin
# keep_gin = [l for l in gin_config if 'predict_drop' not in l]
# change_gin = [l for l in gin_config if 'predict_drop' in l]
# changed_gin = [l[:-6] + '2048\n' for l in change_gin]
# gin_config = keep_gin + changed_gin


# # NOTE: you can change config to 16*1024 to just use standard (non-LSH)
# # attention. It may be useful to debug/check if LSH attention works.
# keep_gin = [l for l in gin_config if 'std_length' not in l]
# change_gin = [l for l in gin_config if 'std_length' in l]
# changed_gin = [l[:-5] + '2048\n' for l in change_gin]
# gin_config = keep_gin + changed_gin


# keep_gin = [l for l in gin_config if 'max_length = 512' not in l]
# change_gin = [l for l in gin_config if 'max_length = 512' in l]
# changed_gin = ['max_length = 2048\n' for l in change_gin]
# gin_config = keep_gin + changed_gin
#
#  End of the part that needs to be uncommented.
#


# NOTE: Change to 16*1024 to predict on complete papers.
gin_config = [l.replace('Reformer2', 'ConfigurableTerraformer') for l in gin_config]
gin_config.append(
    'DotProductCausalAttention.max_inference_length = 2048'
)

og_DotProductCausalAttention = trax.layers.attention.DotProductCausalAttention
trax.layers.attention.DotProductCausalAttention = functools.partial(
    og_DotProductCausalAttention, max_inference_length=2048,
)

# gin_config.append(
#     'MixedLSHSelfAttention.std_length='
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

padding_fun = trax.data.PadToLength(len_map={0: 512, 1: 512, 2: 512}, pad_value = {0: 0, 1: 0, 2:0})
# question = """code:
# def square_list(xs):
#   return [<SENTINEL> for x in xs]
# print(square_list([1, 2, 3, 4]))

# output:
# [1, 4, 9, 16]"""

# tokenized = next(padding_fun(trax.data.tokenize([question,], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100)))
# print(trax.data.detokenize(tokenized, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100))
# print(tokenized.shape)

####


def autoregressive_sample_stream(model, inputs=None,
                                 batch_size=1, temperature=1.0,
                                 start_id=2, accelerate=True, prefix=None):
  if inputs is not None and inputs.shape[0] != batch_size:
    raise ValueError(f'Inputs batch size ({inputs.shape[0]}) does not match '
                     f'batch_size arg ({batch_size}.')

  fast_model = tl.Accelerate(model) if accelerate else model
  if np.isscalar(start_id):
    start_symbol = np.full((batch_size, 1), start_id, dtype=np.int32)
  else:
    start_symbol = start_id
  if model.n_in == 1 and inputs is not None:
    current_symbols = np.concatenate([start_symbol, inputs], axis=1)
  else:
    if prefix is None:
      current_symbols = start_symbol
    else:
      current_symbols = np.concatenate([start_symbol, prefix], axis=1)

  while True:
    t0 = time.time()
    if model.n_in > 1 and inputs is not None:
      # print("inp, curr:", inputs.shape, current_symbols.shape)
      logits = fast_model((inputs, current_symbols))[0]
    else:
      logits = fast_model(current_symbols)
    # print('logits:', str(logits)[:100])
    logits = tl.log_softmax(logits[:, -1, :])
    sample = tl.logsoftmax_sample(logits, temperature=temperature)

    print(trax.data.detokenize(sample, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100))
    print("Time per token: {}".format(time.time() - t0))
    sys.stdout.flush()

    yield sample
    # NOTE: Because the model is autoregressive and in 'predict' mode, its
    # history is cached in the model state and the next input is the single
    # symbol just sampled.
    current_symbols = sample[:, None]


START_INDEX = 10

def autoregressive_sample(model, inputs=None,
                          batch_size=1, temperature=1.0,
                          start_id=0, eos_id=1, max_length=100,
                          accelerate=True, prefix=None):
  result = []
  eos_seen = []
  counter = 0
  index = START_INDEX
  for index, sample in enumerate(autoregressive_sample_stream(
      model, inputs, batch_size=batch_size, temperature=temperature,
      start_id=start_id, accelerate=accelerate, prefix=prefix)):
    if index == START_INDEX:
      start_time = time.time()
    sample = sample[:, None]
    result.append(sample)
    counter += 1
    if counter >= max_length:
      print('decoded one token per {} s'.format(
          (time.time()-start_time)/(index-START_INDEX)))
      return np.concatenate(result, axis=1)
    # Check at which batch positions have we already encountered EOS.
    for j in range(batch_size):
      if int(sample[j, 0]) == eos_id:
        eos_seen.append(j)
    # If EOS has been seen on all positions, stop.
    if all([j in eos_seen for j in range(batch_size)]):
      print('decoded one token per {} s'.format(
          (time.time()-start_time)/(index-START_INDEX)))
      return np.concatenate(result, axis=1)
  print('decoded one token per {} s'.format(
      (time.time()-start_time)/(index-START_INDEX)))
  return np.concatenate(result, axis=1)


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
# The model does not like other numbers than 1024 in the line below.
# In particular 15 * 1024 does not work.
shape1l = trax.shapes.ShapeDtype((1, 1024), dtype=numpy_math.int32)

with trax.fastmath.use_backend(trax.fastmath.Backend.JAX):
  model_predict = model(mode='predict')
  model_predict.init_from_file(model_file, weights_only=True,
                               input_signature=(shape1l, shape11))
  old_state = model_predict.state


# Decode the first article
xart = xarts[2]['article']
question = xart.numpy().decode()
# print(question[:512])

tokenized = next(padding_fun(trax.data.tokenize([question,], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100)))


with trax.fastmath.use_backend(trax.fastmath.Backend.JAX):
  model_predict.state = old_state

  # Putting below 15*1024 does not work.
  tokens = autoregressive_sample(model_predict, tokenized[None, :1024], temperature=0.0, max_length=50)
  print(tokens)
  print(trax.data.detokenize(tokens[0], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR, n_reserved_ids=100))
