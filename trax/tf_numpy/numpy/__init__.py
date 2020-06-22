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

"""NumPy like wrapper for Tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
# pylint: disable=g-import-not-at-top

try:
  # pylint: disable=g-direct-tensorflow-import
  from tensorflow.python.ops.numpy_ops import *
  from tensorflow import bfloat16

except ImportError:
  from tensorflow import newaxis

  from trax.tf_numpy.numpy_impl import random

  # pylint: disable=wildcard-import
  from trax.tf_numpy.numpy_impl.array_ops import *
  from trax.tf_numpy.numpy_impl.arrays import *
  from trax.tf_numpy.numpy_impl.dtypes import *
  from trax.tf_numpy.numpy_impl.math_ops import *
  from trax.tf_numpy.numpy_impl.utils import finfo
  from trax.tf_numpy.numpy_impl.utils import promote_types
  from trax.tf_numpy.numpy_impl.utils import result_type
  # pylint: enable=wildcard-import

  max = amax  # pylint: disable=redefined-builtin,undefined-variable
  min = amin  # pylint: disable=redefined-builtin,undefined-variable
  round = around  # pylint: disable=redefined-builtin,undefined-variable
