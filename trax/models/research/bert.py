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
"""BERT."""

import os

import gin
import jax
import tensorflow as tf

from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as np
from trax.models.download_model import download_weights_if_not_downloaded

# pylint: disable=invalid-name

_MODEL_LINKS = {
    'bert-base-uncased': 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
}


class AddBias(tl.Layer):

  def forward(self, inputs):
    x = inputs
    return x + self.weights

  def init_weights_and_state(self, input_signature):
    self.weights = np.zeros(input_signature.shape[-1])


@gin.configurable()
def BERTClassifierHead(n_classes):
  return tl.Serial([
      tl.Select([0], n_in=2),
      tl.Dense(n_classes,
               kernel_initializer=tl.RandomNormalInitializer(0.02),
               bias_initializer=tl.RandomNormalInitializer(1e-6),
              ),
  ])


@gin.configurable()
def BERTRegressionHead():
  return tl.Serial([
      tl.Select([0], n_in=2),
      tl.Dense(1,
               kernel_initializer=tl.RandomNormalInitializer(0.02),
               bias_initializer=tl.RandomNormalInitializer(1e-6),
              ),
  ])


# TODO(kitaev): masked LM head
# todo(piotrekp1): tests

def BERT(d_model=768,
         vocab_size=30522,
         max_len=512,
         type_vocab_size=2,
         n_heads=12,
         d_ff=3072,
         n_layers=12,
         head=None,
         init_checkpoint=None,
         mode='eval',
        ):
  """BERT (default hparams are for bert-base-uncased)."""
  # todo(piotrekp1): loading config from model_name

  layer_norm_eps = 1e-12
  d_head = d_model // n_heads

  word_embeddings = tl.Embedding(vocab_size, d_model)
  type_embeddings = tl.Embedding(type_vocab_size, d_model)
  position_embeddings = tl.PositionalEncoding(max_len, mode=mode)
  embeddings = [
      tl.Select([0, 1, 0], n_in=3),  # Drops 'idx' input.
      tl.Parallel(
          word_embeddings,
          type_embeddings,
          [tl.PaddingMask(),
           tl.Fn('Squeeze', lambda x: np.squeeze(x, (1, 2)), n_out=1)]
      ),
      tl.Add(),
      position_embeddings,
      tl.LayerNorm(epsilon=layer_norm_eps),
  ]

  encoder = []
  for _ in range(n_layers):
    attn = tl.SelfAttention(n_heads=n_heads, d_qk=d_head, d_v=d_head,
                            bias=True, masked=True, mode=mode)
    feed_forward = [
        tl.Dense(d_ff),
        tl.Gelu(),
        tl.Dense(d_model)
    ]
    encoder += [
        tl.Select([0, 1, 1]),  # Save a copy of the mask
        tl.Residual(attn, AddBias()),  # pylint: disable=no-value-for-parameter
        tl.LayerNorm(epsilon=layer_norm_eps),
        tl.Residual(*feed_forward),
        tl.LayerNorm(epsilon=layer_norm_eps),
    ]

  encoder += [tl.Select([0], n_in=2)]  # Drop the mask

  pooler = [
      tl.Fn('', lambda x: (x[:, 0, :], x), n_out=2),
      tl.Dense(d_model),
      tl.Tanh(),
  ]

  init_checkpoint = init_checkpoint if mode == 'train' else None
  bert = PretrainedBERT(
      embeddings + encoder + pooler, init_checkpoint=init_checkpoint)

  if head is not None:
    bert = tl.Serial(bert, head())

  return bert


def download_model_if_model_name(init_checkpoint):
  """Returns model dir path with model filename.
  if init_checkpoint is a model name and there is no local model with that name
  then it downloads it and returns newly created path."""
  try:
    model_link = _MODEL_LINKS[init_checkpoint]
  except KeyError:
    raise KeyError(f'Not known model name, please make sure the model name'
                     f' is in the list of available models. If this is a path'
                     f' to a model it should contain at least one {os.path.sep}')
  init_checkpoint_dir, checkpoint_filename = download_weights_if_not_downloaded(model_link, init_checkpoint)
  return init_checkpoint_dir, checkpoint_filename


class PretrainedBERT(tl.Serial):
  """Wrapper that always initializes weights from a pre-trained checkpoint."""

  def __init__(self, *sublayers, init_checkpoint=None):
    super().__init__(*sublayers)

    if init_checkpoint is None:
      # initialize model from scratch
      self.init_checkpoint = None
    elif os.path.sep not in init_checkpoint:
      # initialize model from model name
      init_checkpoint_dir, init_checkpoint_filename = download_model_if_model_name(init_checkpoint)
      self.init_checkpoint = os.path.join(init_checkpoint_dir, init_checkpoint_filename)
    else:
      # initialize model from path
      self.init_checkpoint = init_checkpoint

  def init_weights_and_state(self, input_signature):
    super().init_weights_and_state(input_signature)
    if self.init_checkpoint is None:
      return

    print('Loading pre-trained weights from', self.init_checkpoint)
    ckpt = tf.train.load_checkpoint(self.init_checkpoint)

    def reshape_qkv(name):
      x = ckpt.get_tensor(name)
      return x.reshape((x.shape[0], -1, 64)).swapaxes(0, 1)
    def reshape_o(name):
      x = ckpt.get_tensor(name)
      return x.reshape((-1, 64, x.shape[-1]))
    def reshape_bias(name):
      x = ckpt.get_tensor(name)
      return x.reshape((-1, 64))

    new_w = [
        ckpt.get_tensor('bert/embeddings/word_embeddings'),
        ckpt.get_tensor('bert/embeddings/token_type_embeddings'),
        ckpt.get_tensor('bert/embeddings/position_embeddings')[None, ...],
        ckpt.get_tensor('bert/embeddings/LayerNorm/gamma'),
        ckpt.get_tensor('bert/embeddings/LayerNorm/beta'),
    ]

    for i in range(12):  # 12 layers
      new_w += [
          reshape_qkv(f'bert/encoder/layer_{i}/attention/self/query/kernel'),
          reshape_qkv(f'bert/encoder/layer_{i}/attention/self/key/kernel'),
          reshape_qkv(f'bert/encoder/layer_{i}/attention/self/value/kernel'),
          reshape_o(f'bert/encoder/layer_{i}/attention/output/dense/kernel'),
          reshape_bias(f'bert/encoder/layer_{i}/attention/self/query/bias'),
          reshape_bias(f'bert/encoder/layer_{i}/attention/self/key/bias'),
          reshape_bias(f'bert/encoder/layer_{i}/attention/self/value/bias'),
          ckpt.get_tensor(
              f'bert/encoder/layer_{i}/attention/output/dense/bias'),
          ckpt.get_tensor(
              f'bert/encoder/layer_{i}/attention/output/LayerNorm/gamma'),
          ckpt.get_tensor(
              f'bert/encoder/layer_{i}/attention/output/LayerNorm/beta'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/intermediate/dense/kernel'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/intermediate/dense/bias'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/output/dense/kernel'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/output/dense/bias'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/output/LayerNorm/gamma'),
          ckpt.get_tensor(f'bert/encoder/layer_{i}/output/LayerNorm/beta'),
      ]

    new_w += [
        ckpt.get_tensor('bert/pooler/dense/kernel'),
        ckpt.get_tensor('bert/pooler/dense/bias'),
    ]

    for a, b in zip(fastmath.tree_leaves(self.weights), new_w):
      assert a.shape == b.shape, (
          f'Expected shape {a.shape}, got shape {b.shape}')
    self.weights = jax.tree_unflatten(jax.tree_structure(self.weights), new_w)
    move_to_device = jax.jit(lambda x: x)
    self.weights = jax.tree_map(move_to_device, self.weights)
