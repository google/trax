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

"""Array manipulation methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy.array_creation import asarray


def broadcast_to(a, shape):
  """Broadcasts an array to the desired shape if possible.

  Args:
    a: array_like
    shape: a scalar or 1-d tuple/list.

  Returns:
    An ndarray.
  """
  return array_creation.full(shape, a)


def stack(arrays, axis=0):
  return asarray(tf.stack([a.data for a in arrays], axis))
