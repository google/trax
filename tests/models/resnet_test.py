# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for Resnet models."""

import numpy as np

from absl.testing import absltest

from trax import shapes
from trax.models import resnet


class ResnetTest(absltest.TestCase):
    def test_resnet(self):
        model = resnet.Resnet50(d_hidden=8, n_output_classes=10)
        x = np.ones((3, 256, 256, 3)).astype(np.float32)
        _, _ = model.init(shapes.signature(x))
        y = model(x)
        self.assertEqual(y.shape, (3, 10))

    def test_wide_resnet(self):
        model = resnet.WideResnet(n_blocks=1, n_output_classes=10)
        x = np.ones((3, 32, 32, 3)).astype(np.float32)
        _, _ = model.init(shapes.signature(x))
        y = model(x)
        self.assertEqual(y.shape, (3, 10))


if __name__ == "__main__":
    absltest.main()
