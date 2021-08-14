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
          lambda x: jnp.reshape(  # Prolong back.  # pylint: disable=g-long-lambda
              x, (x.shape[0], x.shape[1] * shorten_factor, -1)),
          n_out=1)
  )


def _FeedForwardBlock(d_model,
                      d_ff,
                      dropout,
                      dropout_shared_axes,
                      mode,
                      activation):
  """Returns a list of layers that implements a feedforward block.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A list of layers that maps vectors to vectors.
  """

  def _Dropout():
    return core.Dropout(rate=dropout, shared_axes=dropout_shared_axes,
                        mode=mode)

  return [
      core.Dense(d_ff),
      activation(),
      _Dropout(),
      core.Dense(d_model),
  ]


def AttentionResampling(shorten_factor, d_model, is_upsampling, d_ff, n_heads,
                        dropout, dropout_shared_axes, mode, ff_activation,
                        context_bias_layer, location_bias_layer, total_pooling,
                        resampling_fn):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.
    shorten_factor: by how much shorten/upsample at this funnel block.
    resampling_fn: Type of function that performs funnel upsampling/downsampling
        callable with signature: shorten_factor, d_model;  must return an
         activation-type subclass of `Layer`.

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  attention = RelativeAttentionLMLayer(
      d_model, context_bias_layer, location_bias_layer,
      total_pooling, n_heads=n_heads, dropout=dropout,
      mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  resampling = resampling_fn(shorten_factor, d_model,
                             mode=mode)

  def _Dropout(p=dropout):
    return core.Dropout(rate=p, shared_axes=dropout_shared_axes, mode=mode)

  return [
      LayerNorm(),            # h
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
