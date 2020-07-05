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
"""Tests for RNNs."""

from absl.testing import absltest
import numpy as np

from trax import shapes
from trax.models import rnn


class RNNTest(absltest.TestCase):

  def test_rnnlm_forward_shape(self):
    model = rnn.RNNLM(vocab_size=20, d_model=16)
    x = np.ones((3, 28)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 28, 20))

  def test_grulm_forward_shape(self):
    model = rnn.GRULM(vocab_size=20, d_model=16)
    x = np.ones((3, 28)).astype(np.int32)
    _, _ = model.init(shapes.signature(x))
    y = model(x)
    self.assertEqual(y.shape, (3, 28, 20))


if __name__ == '__main__':
  absltest.main()
