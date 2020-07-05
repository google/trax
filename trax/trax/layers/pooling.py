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
"""Trax pooling layers."""

from trax import fastmath
from trax.layers.base import Fn


# pylint: disable=invalid-name
def MaxPool(pool_size=(2, 2), strides=None, padding='VALID'):
  """Reduces each multi-dimensional window to the max of the window's values.

  Windows, as specified by `pool_size` and `strides`, involve all axes of an
  n-dimensional array except the first and last: :math:`(d_1, ..., d_{n-2})`
  from shape :math:`(d_0, d_1, ..., d_{n-2}, d_{n-1})`.

  Args:
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    padding: 'VALID' or 'SAME'. If 'VALID', no padding is done, and only
        full windows get reduced; partial windows are discarded. If 'SAME',
        padding is added at array edges as needed to avoid partial windows
        but does not otherwise affect the selection of max values.

  Returns:
    N-dimensional array in which each valid (or padded-valid) window position
    in the input is reduced to / replaced by the max value from that window.
    An output array has the same number of dimensions as its input, but has
    fewer elements.
  """
  layer_name = f'MaxPool{pool_size}'.replace(' ', '')
  def f(x):
    return fastmath.max_pool(
        x, pool_size=pool_size, strides=strides, padding=padding)
  return Fn(layer_name, f)


def SumPool(pool_size=(2, 2), strides=None, padding='VALID'):
  """Reduces each multi-dimensional window to the sum of the window's values.

  Windows, as specified by `pool_size` and `strides`, involve all axes of an
  n-dimensional array except the first and last: :math:`(d_1, ..., d_{n-2})`
  from shape :math:`(d_0, d_1, ..., d_{n-2}, d_{n-1})`.

  Args:
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    padding: 'VALID' or 'SAME'. If 'VALID', no padding is done, and only
        full windows get reduced; partial windows are discarded. If 'SAME',
        padding is added at array edges as needed to avoid partial
        windows but does not otherwise affect the computation of sums.

  Returns:
    N-dimensional array in which each valid (or padded-valid) window position
    in the input is reduced to / replaced by the sum of values in that window.
    An output array has the same number of dimensions as its input, but has
    fewer elements.
  """
  layer_name = f'SumPool{pool_size}'.replace(' ', '')
  def f(x):
    return fastmath.sum_pool(
        x, pool_size=pool_size, strides=strides, padding=padding)
  return Fn(layer_name, f)


def AvgPool(pool_size=(2, 2), strides=None, padding='VALID'):
  """Reduces each multi-dimensional window to the mean of the window's values.

  Windows, as specified by `pool_size` and `strides`, involve all axes of an
  n-dimensional array except the first and last: :math:`(d_1, ..., d_{n-2})`
  from shape :math:`(d_0, d_1, ..., d_{n-2}, d_{n-1})`.

  Args:
    pool_size: Shape of window that gets reduced to a single vector value.
        If the layer inputs are :math:`n`-dimensional arrays, then `pool_size`
        must be a tuple of length :math:`n-2`.
    strides: Offsets from the location of one window to the locations of
        neighboring windows along each axis. If specified, must be a tuple of
        the same length as `pool_size`. If None, then offsets of 1 along each
        window axis, :math:`(1, ..., 1)`, will be used.
    padding: 'VALID' or 'SAME'. If 'VALID', no padding is done, and only
        full windows get reduced; partial windows are discarded. If 'SAME',
        padding is added at array edges as needed but is not counted in the
        computation of averages.

  Returns:
    N-dimensional array in which each valid (or padded-valid) window position
    in the input is reduced to / replaced by the mean of values in that window.
    An output array has the same number of dimensions as its input, but has
    fewer elements.
  """
  layer_name = f'AvgPool{pool_size}'.replace(' ', '')
  def f(x):
    return fastmath.avg_pool(
        x, pool_size=pool_size, strides=strides, padding=padding)
  return Fn(layer_name, f)
