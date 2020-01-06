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

"""Logical functions."""
import numpy as np

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import utils


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  dtype = utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = array_creation.asarray(x1, dtype=dtype)
  x2 = array_creation.asarray(x2, dtype=dtype)
  x1 = x1.data
  x2 = x2.data
  if cast_bool_to_int and x1.dtype == tf.bool:
    x1 = tf.cast(x1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
  return utils.tensor_to_ndarray(tf_fun(x1, x2))


@utils.np_doc(np.equal)
def equal(x1, x2):
  return _comparison(tf.equal, x1, x2)


@utils.np_doc(np.not_equal)
def not_equal(x1, x2):
  return _comparison(tf.not_equal, x1, x2)


@utils.np_doc(np.greater)
def greater(x1, x2):
  return _comparison(tf.greater, x1, x2, True)


@utils.np_doc(np.greater_equal)
def greater_equal(x1, x2):
  return _comparison(tf.greater_equal, x1, x2, True)


@utils.np_doc(np.less)
def less(x1, x2):
  return _comparison(tf.less, x1, x2, True)


@utils.np_doc(np.less_equal)
def less_equal(x1, x2):
  return _comparison(tf.less_equal, x1, x2, True)


@utils.np_doc(np.array_equal)
def array_equal(a1, a2):
  def f(a1, a2):
    if a1.shape != a2.shape:
      return tf.constant(False)
    return tf.reduce_all(tf.equal(a1, a2))
  return _comparison(f, a1, a2)


def _logical_binary_op(tf_fun, x1, x2):
  x1 = array_creation.asarray(x1, dtype=np.bool_)
  x2 = array_creation.asarray(x2, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf_fun(x1.data, x2.data))


@utils.np_doc(np.logical_and)
def logical_and(x1, x2):
  return _logical_binary_op(tf.logical_and, x1, x2)


@utils.np_doc(np.logical_or)
def logical_or(x1, x2):
  return _logical_binary_op(tf.logical_or, x1, x2)


@utils.np_doc(np.logical_xor)
def logical_xor(x1, x2):
  return _logical_binary_op(tf.math.logical_xor, x1, x2)


@utils.np_doc(np.logical_not)
def logical_not(x):
  x = array_creation.asarray(x, dtype=np.bool_)
  return utils.tensor_to_ndarray(tf.logical_not(x.data))
