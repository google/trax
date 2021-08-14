from trax.layers import core
from trax.layers.pooling import AvgPool
from trax.layers.normalization import LayerNorm
from trax.layers.research.rel_attention import RelativeAttentionLMLayer
from trax.layers import combinators as cb
from trax.fastmath import numpy as jnp


def AveragePooling(shorten_factor, *args, **kwargs):
  del args, kwargs

  return AvgPool(pool_size=(shorten_factor,), strides=(shorten_factor,))


def LinearPooling(shorten_factor, d_model, *args, dropout=0.0, mode='train',
                  **kwargs):
  del args, kwargs

  return cb.Serial(
      core.Fn(
          'Shorten',
          lambda x: jnp.reshape(
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
          lambda x: jnp.reshape(
              # Prolong back.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] * shorten_factor, -1)),
          n_out=1)
  )


def _FeedForwardBlock(d_model,
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
  attention = RelativeAttentionLMLayer(
      d_model, context_bias_layer, location_bias_layer,
      total_pooling, n_heads=n_heads, dropout=dropout,
      mode=mode)

  feed_forward = _FeedForwardBlock(
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
      cb.Serial(
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
