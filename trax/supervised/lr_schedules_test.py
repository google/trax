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
"""Tests of learning rate schedules."""

import math

from absl.testing import absltest

from trax.supervised import lr_schedules


class LRFunctionsTest(absltest.TestCase):

  def test_warmup(self):
    lr_fn = lr_schedules.warmup(9, .01)

    # Linear warm-up.
    self.assertAlmostEqual(.001, lr_fn(1))
    self.assertAlmostEqual(.002, lr_fn(2))
    self.assertAlmostEqual(.005, lr_fn(5))
    self.assertAlmostEqual(.009, lr_fn(9))

    # Constant thereafter.
    self.assertAlmostEqual(.01, lr_fn(10))
    self.assertAlmostEqual(.01, lr_fn(11))
    self.assertAlmostEqual(.01, lr_fn(20))
    self.assertAlmostEqual(.01, lr_fn(300))
    self.assertAlmostEqual(.01, lr_fn(4000))

  def test_constant(self):
    lr_fn = lr_schedules.constant(.02)
    self.assertEqual(.02, lr_fn(1))
    self.assertEqual(.02, lr_fn(20))
    self.assertEqual(.02, lr_fn(300))
    self.assertEqual(.02, lr_fn(4000))
    self.assertEqual(.02, lr_fn(50000))
    self.assertEqual(.02, lr_fn(600000))
    self.assertEqual(.02, lr_fn(7000000))
    self.assertEqual(.02, lr_fn(80000000))
    self.assertEqual(.02, lr_fn(900000000))

  def test_warmup_and_rsqrt_decay(self):
    lr_fn = lr_schedules.warmup_and_rsqrt_decay(24, .25)

    # Warm-up.
    self.assertAlmostEqual(.01, lr_fn(1))
    self.assertAlmostEqual(.02, lr_fn(2))
    self.assertAlmostEqual(.23, lr_fn(23))
    self.assertAlmostEqual(.24, lr_fn(24))

    # Reciprocal square-root decay.
    self.assertAlmostEqual(.25 * (5 / math.sqrt(25)), lr_fn(25))
    self.assertAlmostEqual(.25 * (5 / math.sqrt(26)), lr_fn(26))
    self.assertAlmostEqual(.25 * (5 / math.sqrt(27)), lr_fn(27))
    self.assertAlmostEqual(.25 * (5 / math.sqrt(300)), lr_fn(300))
    self.assertAlmostEqual(.25 * (5 / math.sqrt(4000)), lr_fn(4000))
    self.assertAlmostEqual(.25 * (5 / math.sqrt(50000)), lr_fn(50000))

  def test_cosine_sawtooth(self):
    tail_fn = lr_schedules._CosineSawtoothTail(180, min_value=.1)
    lr_fn = lr_schedules._BodyAndTail(.3, tail_start=0, tail_fn=tail_fn)

    # First cycle
    self.assertAlmostEqual(.29998477, lr_fn(1))
    self.assertAlmostEqual(.28660254, lr_fn(30))
    self.assertAlmostEqual(.25, lr_fn(60))
    self.assertAlmostEqual(.20, lr_fn(90))
    self.assertAlmostEqual(.15, lr_fn(120))
    self.assertAlmostEqual(.10001523, lr_fn(179))

    # Second cycle
    self.assertEqual(.3, lr_fn(180))
    self.assertAlmostEqual(.29998477, lr_fn(181))
    self.assertAlmostEqual(.28660254, lr_fn(210))
    self.assertAlmostEqual(.25, lr_fn(240))
    self.assertAlmostEqual(.20, lr_fn(270))
    self.assertAlmostEqual(.15, lr_fn(300))
    self.assertAlmostEqual(.10001523, lr_fn(359))


if __name__ == '__main__':
  absltest.main()
