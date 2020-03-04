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

from trax import math
from trax.layers import base


@base.layer()
def MaxPool(x, weights, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del weights, kw
  return math.max_pool(x, pool_size=pool_size, strides=strides,
                       padding=padding)


@base.layer()
def SumPool(x, weights, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del weights, kw
  return math.sum_pool(x, pool_size=pool_size, strides=strides,
                       padding=padding)


@base.layer()
def AvgPool(x, weights, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del weights, kw
  return math.avg_pool(x, pool_size=pool_size, strides=strides,
                       padding=padding)
