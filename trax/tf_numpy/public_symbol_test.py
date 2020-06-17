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

"""Tests different ways to use the public tf-numpy module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

import tensorflow as tf
import tensorflow.experimental.numpy as np1
from tensorflow.experimental import numpy as np2  # pylint: disable=reimported


np3 = tf.experimental.numpy


class PublicSymbolTest(tf.test.TestCase):

  def testSimple(self):
    a = 0.1
    b = 0.2
    for op in [np1.add, np2.add, np3.add]:
      self.assertAllClose(onp.add(a, b), op(a, b))


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
