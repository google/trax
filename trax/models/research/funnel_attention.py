import numpy as np
from trax.layers.attention import DotProductAttention, ConfigurableAttention

from trax.fastmath import numpy as jnp

from trax import fastmath
from trax.layers import DotProductCausalAttention, core


class FunnelDotProductCausalAttention(DotProductCausalAttention):
  def __init__(self, dropout=0.0, max_inference_length=2048, mode='train'):
    super(FunnelDotProductCausalAttention, self).__init__(
        dropout=dropout,
        max_inference_length=max_inference_length, mode=mode)

  @staticmethod
  def _funnel_mask(n_queries, n_keys):
    numpy_ = jnp if fastmath.is_backend(fastmath.Backend.JAX) else np

    mask = numpy_.tril(numpy_.ones((n_queries, n_queries), dtype=np.bool_))

    shorten_factor = n_keys // n_queries

    mask = numpy_.repeat(mask, shorten_factor, axis=-1)
    mask = numpy_.pad(mask, (0, 0), (0, n_keys - mask.shape[1]), 'constant')
    return mask

  def forward(self, inputs):
    q, k, v = inputs

    n_queries, n_keys = q.shape[-2], k.shape[-2]

    mask = self._funnel_mask(n_queries, n_keys)

    res, dots = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=self.rng)
    if self._mode == 'viz':
      self.state = dots
    return res


def FunnelCausalAttention(d_feature, n_heads=1, dropout=0.0,
                          max_inference_length=2048, mode='train'):
  if d_feature % n_heads != 0:
    raise ValueError(
        f'Dimensionality of feature embedding ({d_feature}) is not a multiple '
        f'of the requested number of attention heads ({n_heads}).')

  return ConfigurableAttention(
      core.Dense(d_feature), core.Dense(d_feature), core.Dense(d_feature),
      core.Dense(d_feature), n_heads=n_heads,
      qkv_attention_layer=FunnelDotProductCausalAttention(
          dropout=dropout, max_inference_length=max_inference_length,
          mode=mode))
