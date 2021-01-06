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

# Lint as: python3
"""Tests for initializers."""

from absl.testing import absltest
import numpy as np

from trax import fastmath
from trax import test_utils
import trax.layers as tl


INPUT_SHAPE = (5, 7, 20)


def rng():  # Can't be a constant, because JAX has to init itself in main first.
  return fastmath.random.get_prng(0)


class InitializersTest(absltest.TestCase):

  def test_random_normal(self):
    f = tl.RandomNormalInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_lecun_uniform(self):
    f = tl.LeCunUniformInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_random_uniform(self):
    f = tl.RandomUniformInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_glorot_normal(self):
    f = tl.GlorotNormalInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_glorot_uniform(self):
    f = tl.GlorotUniformInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_lecun_normal(self):
    f = tl.LeCunNormalInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_kaiming_normal(self):
    f = tl.KaimingNormalInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_kaiming_uniform(self):
    f = tl.KaimingUniformInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_orthogonal(self):
    f = tl.OrthogonalInitializer()
    init_value = f(INPUT_SHAPE, rng())
    self.assertEqual(init_value.shape, INPUT_SHAPE)

  def test_from_file(self):
    params = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    # `create_tempfile` needs access to --test_tmpdir, however in the OSS world
    # pytest doesn't run `absltest.main`, so we need to manually parse the flags
    test_utils.ensure_flag('test_tmpdir')
    filename = self.create_tempfile('params.npy').full_path
    with open(filename, 'wb') as f:
      np.save(f, params)
    f = tl.InitializerFromFile(filename)
    init_value = f(params.shape, rng())
    self.assertEqual(tl.to_list(init_value), tl.to_list(params))
    # self.assertEqual('%s' % init_value, '%s' % params)


if __name__ == '__main__':
  absltest.main()
