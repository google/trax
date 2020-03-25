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
"""Tests for MLP."""

from absl.testing import absltest
from trax import layers as tl
from trax import math
from trax.models import mlp
from trax.shapes import ShapeDtype


class MLPTest(absltest.TestCase):

  def test_pure_mlp_forward_shape(self):
    """Run the PureMLP model forward and check output shape."""
    input_signature = ShapeDtype((7, 28, 28, 3))
    model = mlp.PureMLP(layer_widths=(32, 16, 8))
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual((7, 8), final_shape)

  def test_mlp_forward_shape(self):
    """Run the MLP model forward and check output shape."""
    input_signature = ShapeDtype((3, 28, 28, 1))
    model = mlp.MLP(d_hidden=32, n_output_classes=10)
    final_shape = tl.check_shape_agreement(model, input_signature)
    self.assertEqual((3, 10), final_shape)

  def test_mlp_input_signatures(self):
    input_signature = ShapeDtype((3, 28, 28, 1))
    mlp_block = mlp.MLP(d_hidden=32, n_output_classes=10)
    relu = tl.Relu()
    mlp_and_relu = tl.Serial(mlp_block, relu)
    mlp_and_relu.init(input_signature)

    # Check for correct shapes entering and exiting the mlp_block.
    mlp_and_relu._set_input_signature_recursive(input_signature)
    self.assertEqual(mlp_block.input_signature, input_signature)
    self.assertEqual(relu.input_signature, ShapeDtype((3, 10)))


if __name__ == '__main__':
  absltest.main()
