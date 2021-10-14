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

"""Resampling module."""

from trax.fastmath import numpy as jnp
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers.normalization import LayerNorm
from trax.layers.pooling import AvgPool
from trax.layers.research.rel_attention import RelativeAttentionLMLayer


def AveragePooling(shorten_factor, *args, **kwargs):
  del args, kwargs

  return AvgPool(pool_size=(shorten_factor,), strides=(shorten_factor,))


def LinearPooling(shorten_factor, d_model, *args, dropout=0.0, mode='train',
                  **kwargs):
  del args, kwargs

  return cb.Serial(
      core.Fn(
          'Shorten',
          lambda x: jnp.reshape(  # pylint: disable=g-long-lambda
              # Shorten -- move to depth.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] // shorten_factor, -1)),
          n_out=1),
      core.Dense(d_model),
      core.Dropout(rate=dropout, mode=mode)
  )


def LinearUpsampling(shorten_factor, d_model, *args, dropout=0.0, mode='train',
                     **kwargs):
  del args, kwargs

  return cb.Serial(
      core.Dense(shorten_factor * d_model),
      core.Dropout(rate=dropout, mode=mode),
      core.Fn(
          'ProlongBack',
          lambda x: jnp.reshape(  # pylint: disable=g-long-lambda
              # Prolong back.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] * shorten_factor, -1)),
          n_out=1)
  )


def NaiveUpsampling(shorten_factor, d_model, *args, **kwargs):  # pylint: disable = unused-argument
  return core.Fn('Repeat', lambda x: jnp.repeat(x, shorten_factor, axis=1))


def NoUpsampling(shorten_factor, d_model, *args, **kwargs):
  del d_model, args, kwargs

  return core.Fn('ReturnZero', lambda x: jnp.zeros(  # pylint: disable=g-long-lambda
      (x.shape[0], x.shape[1] * shorten_factor, x.shape[2]), dtype=x.dtype))


def FeedForwardBlock(d_model,
                     d_ff,
                     dropout,
                     dropout_shared_axes,
                     mode,
                     activation):
  # We copy the ff block function because we cannot import it from models
  return [
      core.Dense(d_ff),
      activation(),
      core.Dropout(rate=dropout, shared_axes=dropout_shared_axes,
                   mode=mode),
      core.Dense(d_model),
  ]


def AttentionResampling(shorten_factor, d_model, is_upsampling, d_ff, n_heads,
                        dropout, dropout_shared_axes, mode, ff_activation,
                        context_bias_layer, location_bias_layer, total_pooling,
                        resampling_fn):
  """Attention resampling."""

  attention = RelativeAttentionLMLayer(
      d_model, context_bias_layer, location_bias_layer,
      total_pooling, n_heads=n_heads, dropout=dropout,
      mode=mode)

  feed_forward = FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  resampling = resampling_fn(shorten_factor, d_model,
                             mode=mode)

  def _Dropout():
    return core.Dropout(rate=dropout, shared_axes=dropout_shared_axes,
                        mode=mode)

  return [
      LayerNorm(),               # h
      cb.Branch(cb.Serial(
          resampling,
          LayerNorm(),
      ), None),                  # h', h
      cb.Serial(  # pylint: disable=g-long-ternary
          cb.Select([0, 2, 1, 2]),
          cb.Add(),
      ) if is_upsampling else [],
      cb.Residual(
          cb.Select([0, 1, 1]),  # h', h, h
          attention,
          _Dropout(),
      ),
      cb.Residual(
          LayerNorm(),
          feed_forward,
          _Dropout(),
      ),
  ]
