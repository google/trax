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
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import array_creation
from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import utils


# Relational operators.
def equal(x1, x2):
  """Compare two arrays for equality element-wise.

  Both arrays must either be of the same shape or one should be broadcastable
  to the other.

  Args:
    x1: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    x2: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray of type bool and broadcasted shape of x1 and x2.
  """
  dtype = utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = array_creation.array(x1, copy=False, dtype=dtype)
  x2 = array_creation.array(x2, copy=False, dtype=dtype)
  return utils.tensor_to_ndarray(tf.equal(x1.data, x2.data))


setattr(arrays.ndarray, '__eq__', equal)
