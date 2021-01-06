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

"""Random functions."""
import numpy as np
import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy_impl import utils


DEFAULT_RANDN_DTYPE = np.float32


def randn(*args):
  """Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  """
  # TODO(wangpeng): Use new stateful RNG
  if utils.isscalar(args):
    args = (args,)
  return utils.tensor_to_ndarray(
      tf.random.normal(args, dtype=DEFAULT_RANDN_DTYPE))


def seed(s):
  """Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  """
  # TODO(wangpeng): make the signature the same as numpy
  tf.random.set_seed(s)
