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

"""Array manipulation methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import arrays as arrays_lib
from trax.tf_numpy.numpy import math as math_lib
from trax.tf_numpy.numpy import utils


def broadcast_to(a, shape):
  """Broadcasts an array to the desired shape if possible.

  Args:
    a: array_like
    shape: a scalar or 1-d tuple/list.

  Returns:
    An ndarray.
  """
  return array_creation.full(shape, a)


@utils.np_doc(np.stack)
def stack(arrays, axis=0):
  arrays = array_creation._promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return array_creation.asarray(tf.stack(unwrapped_arrays, axis))


@utils.np_doc(np.hstack)
def hstack(tup):
  arrays = [math_lib.atleast_1d(a) for a in tup]
  arrays = array_creation._promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  rank = tf.rank(unwrapped_arrays[0])
  return utils.cond(rank == 1, lambda: tf.concat(unwrapped_arrays, axis=0),
                    lambda: tf.concat(unwrapped_arrays, axis=1))


@utils.np_doc(np.vstack)
def vstack(tup):
  arrays = [math_lib.atleast_2d(a) for a in tup]
  arrays = array_creation._promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return tf.concat(unwrapped_arrays, axis=0)


@utils.np_doc(np.dstack)
def dstack(tup):
  arrays = [math_lib.atleast_3d(a) for a in tup]
  arrays = array_creation._promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, arrays_lib.ndarray) else a for a in arrays
  ]
  return tf.concat(unwrapped_arrays, axis=2)
