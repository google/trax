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
from tensorflow.experimental import numpy as np1
# Note that `import tensorflow.experimental.numpy as np` doesn't work because
# that usage requires `tensorflow.experimental.numpy` to be a module.


np2 = tf.experimental.numpy


class PublicSymbolTest(tf.test.TestCase):

  def testSimple(self):
    a = 0.1
    b = 0.2
    for m in (np1, np2):
      self.assertEqual(onp.int32, m.int32)
      self.assertAllClose(onp.add(a, b), m.add(a, b))


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
