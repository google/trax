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
"""Tests for dataflow tracer."""

import copy
import itertools
import random

from absl.testing import absltest
import jax
import numpy as onp
from trax.layers import activation_fns
from trax.layers import combinators as cb
from trax.layers import tracer
from trax.shapes import ShapeDtype


def shuffled(x):
  """Functional shuffle."""
  y = copy.copy(x)
  random.shuffle(y)
  return y


class TracerTest(absltest.TestCase):

  def test_tracer_apply(self):
    lyr = cb.Add()
    a = tracer.Tracer('a')
    b = tracer.Tracer('b')
    c = lyr @ (a, b)
    result = tracer.ApplyExpr(lyr, ('a', 'b'))
    self.assertEqual(c.expr, result)

  def test_tracer_index(self):
    lyr = cb.Parallel(activation_fns.Tanh(), activation_fns.Tanh())
    a = tracer.Tracer('a')
    b = tracer.Tracer('b')
    d, e = lyr @ (a, b)
    result0 = tracer.IndexExpr(0, tracer.ApplyExpr(lyr, ('a', 'b')))
    result1 = tracer.IndexExpr(1, tracer.ApplyExpr(lyr, ('a', 'b')))
    self.assertEqual(d.expr, result0)
    self.assertEqual(e.expr, result1)

  def test_apply_to_eqn(self):
    lyr = cb.Add()
    a = tracer.Tracer('a')
    b = tracer.Tracer('b')
    c = lyr @ (a, b)
    eqns, outputs = tracer.traces_to_eqns(c)
    result0 = [tracer.ApplyEqn(lyr, ('a', 'b'), ('var0',))]
    result1 = ('var0',)
    self.assertEqual(eqns, result0)
    self.assertEqual(outputs, result1)

  def test_index_to_eqn(self):
    a, b = tracer.Tracer('fake_output', 2)
    eqns, outputs = tracer.traces_to_eqns((a, b))
    result0 = [tracer.IndexEqn(0, 'fake_output', 'var0'),
               tracer.IndexEqn(1, 'fake_output', 'var1')]
    result1 = ('var0', 'var1')
    self.assertEqual(eqns, result0)
    self.assertEqual(outputs, result1)

  def test_apply_index_to_eqn(self):
    lyr = cb.Parallel(activation_fns.Tanh(), activation_fns.Tanh())
    a = tracer.Tracer('a')
    b = tracer.Tracer('b')
    c, d = lyr @ (a, b)
    eqns, outputs = tracer.traces_to_eqns((c, d))
    result0 = [tracer.ApplyEqn(lyr, ('a', 'b'), ('var2',)),
               tracer.IndexEqn(0, 'var2', 'var0'),
               tracer.IndexEqn(1, 'var2', 'var1')]
    result1 = ('var0', 'var1')
    self.assertEqual(eqns, result0)
    self.assertEqual(outputs, result1)

  def test_eqns_merge_outputs(self):
    lyr = cb.Parallel(activation_fns.Tanh(), activation_fns.Tanh())
    eqns = [tracer.ApplyEqn(lyr, ('a', 'b'), ('var2',)),
            tracer.IndexEqn(0, 'var2', 'var0'),
            tracer.IndexEqn(1, 'var2', 'var1')]
    simple_eqns = tracer.merge_output_tuples(eqns)
    result = [tracer.ApplyEqn(lyr, ('a', 'b'), ('var0', 'var1'))]
    self.assertEqual(simple_eqns, result)

  def test_eqns_eval_order1(self):
    # exhustive test of all linear order permutations for lists up to 7 long
    dummy = activation_fns.Tanh()
    for n in range(1, 7):
      eqns = [tracer.ApplyEqn(dummy,
                              ('var%d'%i,),
                              ('var%d'%(i+1),)) for i in range(n)]
      for permuted in itertools.permutations(eqns):
        ordered_eqns = tracer.evaluation_order_sort(permuted, ['var%d'%n])
        self.assertEqual(ordered_eqns, eqns)

  def test_eqns_eval_order2(self):
    dummy = activation_fns.Tanh()
    eqns = [
        tracer.ApplyEqn(dummy, ('var0',), ('var1',)),
        tracer.ApplyEqn(dummy, ('var2',), ('var3',)),
        tracer.ApplyEqn(dummy, ('var4',), ('var5',)),
        tracer.ApplyEqn(dummy, ('var1', 'var3', 'var5',), ('var6',)),
    ]
    for permuted in itertools.permutations(eqns):
      self.assertEqual(
          tracer.evaluation_order_sort(permuted, ['var6'])[-1], eqns[-1])

  def test_eqns_eval_order3(self):
    dummy = activation_fns.Tanh()
    eqns = [
        tracer.ApplyEqn(dummy, ('var0',), ('var1', 'var2', 'var3')),
        tracer.ApplyEqn(dummy, ('var1',), ('var4',)),
        tracer.ApplyEqn(dummy, ('var2',), ('var5',)),
        tracer.ApplyEqn(dummy, ('var3',), ('var6',)),
    ]
    outputs = ['var4', 'var5', 'var6']
    for permuted in itertools.permutations(eqns):
      self.assertEqual(
          tracer.evaluation_order_sort(permuted, outputs)[0], eqns[0])

  def test_recombine(self):
    add_lyr = cb.Add()
    tanh_lyr = activation_fns.Tanh()
    eqns = [
        tracer.ApplyEqn(add_lyr, ('a', 'b'), ('var1',)),
        tracer.ApplyEqn(tanh_lyr, ('var1',), ('var2',)),
    ]
    outputs = ('var2',)
    model = tracer.recombine(eqns, ('a', 'b'), outputs)
    self.assertEqual(type(model), cb.Serial)
    self.assertEqual(model.sublayers[0], add_lyr)
    self.assertEqual(model.sublayers[1], tanh_lyr)

  def test_split_signature_parameters1(self):
    def fn(a, b, c=1, d=1):
      del c, d
      return a + b
    result = tracer.split_signature_parameters(fn)
    expected = (['a', 'b'], {'c': 1, 'd': 1})
    self.assertEqual(result, expected)

  def test_split_signature_parameters2(self):
    def fn(a, b, c=1, d=1, **kw):
      del c, d, kw
      return a + b
    result = tracer.split_signature_parameters(fn)
    expected = (['a', 'b'], {'c': 1, 'd': 1})
    self.assertEqual(result, expected)

  def test_symbolic_decorator1(self):
    add_lyr = cb.Add()
    @tracer.symbolic
    def make_layer(a, b, c):
      d = add_lyr @ (a, b)
      e = add_lyr @ (d, c)
      return e
    layer = make_layer()  # pylint: disable=no-value-for-parameter
    a = onp.random.randint(0, 10, size=(2, 10))
    b = onp.random.randint(0, 10, size=(2, 10))
    c = onp.random.randint(0, 10, size=(2, 10))
    input_sd = ShapeDtype((2, 10), onp.int32)
    input_signature = (input_sd, input_sd, input_sd)
    p, s = layer.new_weights_and_state(input_signature)
    res = layer((a, b, c), weights=p, state=s, rng=jax.random.PRNGKey(0))  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    result = onp.array(res)
    expected = a + b + c
    onp.testing.assert_allclose(result, expected)

  def test_symbolic_decorator2(self):
    add_lyr = cb.Add()
    tanh_lyr = activation_fns.Tanh()
    @tracer.symbolic
    def make_layer(a, b, c):
      a = tanh_lyr @ a
      d = add_lyr @ (a, b)
      e = add_lyr @ (d, c)
      return e
    layer = make_layer()  # pylint: disable=no-value-for-parameter
    a = onp.random.randint(0, 10, size=(2, 10))
    b = onp.random.randint(0, 10, size=(2, 10))
    c = onp.random.randint(0, 10, size=(2, 10))
    input_sd = ShapeDtype((2, 10), onp.int32)
    input_signature = (input_sd, input_sd, input_sd)
    p, s = layer.new_weights_and_state(input_signature)
    res = layer((a, b, c), weights=p, state=s, rng=jax.random.PRNGKey(0))  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    result = onp.array(res)
    expected = onp.tanh(a) + b + c
    onp.testing.assert_allclose(result, expected)

  def test_symbolic_decorator3(self):
    add_lyr = cb.Add()
    tanh_lyr = cb.Parallel(activation_fns.Relu(), activation_fns.Tanh())
    @tracer.symbolic
    def make_layer(a, b, c):
      d = add_lyr @ (a, b)
      e = add_lyr @ (d, c)
      f, g = tanh_lyr @ (d, e)
      return f, g
    layer = make_layer()  # pylint: disable=no-value-for-parameter
    a = onp.random.uniform(-10, 10, size=(2, 10))
    b = onp.random.uniform(-10, 10, size=(2, 10))
    c = onp.random.uniform(-10, 10, size=(2, 10))
    input_sd = ShapeDtype((2, 10), onp.int32)
    input_signature = (input_sd, input_sd, input_sd)
    p, s = layer.new_weights_and_state(input_signature)
    res = layer((a, b, c), weights=p, state=s, rng=jax.random.PRNGKey(0))  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter,not-callable
    result0 = onp.array(res[0])
    expected0 = onp.where(a + b > 0, a + b, 0.0)
    onp.testing.assert_allclose(result0, expected0, rtol=1e-5)
    result1 = onp.array(res[1])
    expected1 = onp.tanh(a + b + c)
    onp.testing.assert_allclose(result1, expected1, rtol=1e-5)

  def test_symbolic_decorator4(self):
    add_lyr = cb.Add()
    @tracer.symbolic
    def make_layer(a, b, n=1):
      for _ in range(n):
        a = add_lyr @ (a, b)
      return a
    layer = make_layer(n=3)  # pylint: disable=no-value-for-parameter
    a = onp.random.randint(0, 10, size=(2, 10))
    b = onp.random.randint(0, 10, size=(2, 10))
    input_sd = ShapeDtype((2, 10), onp.int32)
    input_signature = (input_sd, input_sd)
    p, s = layer.new_weights_and_state(input_signature)
    res = layer((a, b), weights=p, state=s, rng=jax.random.PRNGKey(0))  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter,not-callable
    result = onp.array(res)
    expected = a + 3 * b
    onp.testing.assert_allclose(result, expected)

    layer = make_layer(n=5)  # pylint: disable=no-value-for-parameter
    input_sd = ShapeDtype((2, 10), onp.int32)
    input_signature = (input_sd, input_sd)
    p, s = layer.new_weights_and_state(input_signature)
    res = layer((a, b), weights=p, state=s, rng=jax.random.PRNGKey(0))  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter,not-callable
    result = onp.array(res)
    expected = a + 5 * b
    onp.testing.assert_allclose(result, expected)

if __name__ == '__main__':
  absltest.main()
