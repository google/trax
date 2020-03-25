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
from trax import layers as tl
from trax import math
from trax.models import rnn
from trax.shapes import ShapeDtype


class RNNTest(absltest.TestCase):

  def test_rnnlm_forward_shape(self):
    """Runs the RNN LM forward and checks output shape."""
    input_signature = ShapeDtype((3, 28), dtype=math.numpy.int32)
    model = rnn.RNNLM(vocab_size=20, d_model=16)
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual((3, 28, 20), final_shape)

  def test_grulm_forward_shape(self):
    """Runs the GRU LM forward and checks output shape."""
    input_signature = ShapeDtype((3, 28), dtype=math.numpy.int32)
    model = rnn.GRULM(vocab_size=20, d_model=16)
    model.init(input_signature)
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual((3, 28, 20), final_shape)


if __name__ == '__main__':
  absltest.main()
