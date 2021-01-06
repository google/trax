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

"""Tests for trax.fastmath.ops."""

import collections
from absl.testing import parameterized

import gin
import jax.numpy as jnp
import numpy as onp
from tensorflow import test
from trax import fastmath


_TestNamedtuple = collections.namedtuple('_TestNamedtuple', ['x'])


class BackendTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def override_gin(self, bindings):
    gin.parse_config_files_and_bindings(None, bindings)

  def test_backend_imports_correctly(self):
    backend = fastmath.backend()
    self.assertEqual(jnp, backend['np'])
    self.assertNotEqual(onp, backend['np'])

    self.override_gin("backend.name = 'numpy'")

    backend = fastmath.backend()
    self.assertNotEqual(jnp, backend['np'])
    self.assertEqual(onp, backend['np'])

  def test_backend_can_be_set(self):
    self.assertEqual(fastmath.backend_name(), 'jax')
    fastmath.set_backend('tensorflow-numpy')
    self.assertEqual(fastmath.backend_name(), 'tensorflow-numpy')
    fastmath.set_backend(None)
    self.assertEqual(fastmath.backend_name(), 'jax')

  def test_numpy_backend_delegation(self):
    # Assert that we are getting JAX's numpy backend.
    backend = fastmath.backend()
    numpy = fastmath.numpy
    self.assertEqual(jnp, backend['np'])

    # Assert that `numpy` calls the appropriate gin configured functions and
    # properties.
    self.assertTrue(numpy.isinf(numpy.inf))
    self.assertEqual(jnp.isinf, numpy.isinf)
    self.assertEqual(jnp.inf, numpy.inf)

    # Assert that we will now get the pure numpy backend.

    self.override_gin("backend.name = 'numpy'")

    backend = fastmath.backend()
    numpy = fastmath.numpy
    self.assertEqual(onp, backend['np'])

    # Assert that `numpy` calls the appropriate gin configured functions and
    # properties.
    self.assertTrue(numpy.isinf(numpy.inf))
    self.assertEqual(onp.isinf, numpy.isinf)
    self.assertEqual(onp.inf, numpy.inf)

  @parameterized.named_parameters(
      ('_' + b.value, b) for b in (fastmath.Backend.JAX, fastmath.Backend.TFNP))
  def test_fori_loop(self, backend):
    with fastmath.use_backend(backend):
      res = fastmath.fori_loop(2, 5, lambda i, x: x + i, 1)
      self.assertEqual(res, 1 + 2 + 3 + 4)

  def test_nested_map(self):
    inp = {'a': ([0, 1], 2), 'b': _TestNamedtuple(3)}
    out = {'a': ([1, 2], 3), 'b': _TestNamedtuple(4)}
    self.assertEqual(fastmath.nested_map(lambda x: x + 1, inp), out)

  def test_nested_stack(self):
    inp = [
        {'a': ([0, 1], 2), 'b': _TestNamedtuple(3)},
        {'a': ([1, 2], 3), 'b': _TestNamedtuple(4)},
    ]
    out = {'a': ([[0, 1], [1, 2]], [2, 3]), 'b': _TestNamedtuple([3, 4])}
    onp.testing.assert_equal(fastmath.nested_stack(inp), out)

  def test_names_match(self):
    # Names match up.
    for backend_enum, backend_obj in fastmath.ops._backend_dict.items():
      self.assertEqual(backend_enum.value, backend_obj['name'])

    # Every backend appears in the dictionary.
    for backend_enum in fastmath.ops.Backend:
      self.assertIn(backend_enum, fastmath.ops._backend_dict)

  def test_use_backend_str(self):
    with fastmath.use_backend('tensorflow-numpy'):
      self.assertEqual(fastmath.backend_name(), 'tensorflow-numpy')

  def test_use_backend_enum(self):
    with fastmath.use_backend(fastmath.Backend.NUMPY):
      self.assertEqual(fastmath.backend_name(), 'numpy')


if __name__ == '__main__':
  test.main()
