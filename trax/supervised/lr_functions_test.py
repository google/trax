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
"""Tests of functions that produce learning rate schedules."""

import math

from absl.testing import absltest

from trax.supervised import lr_functions


class LRFunctionsTest(absltest.TestCase):

  def test_linear_warmup_and_body(self):
    lr_schedule = lr_functions.BodyAndTail(.01, body_start=10)

    # Linear warm-up.
    self.assertAlmostEqual(.001, lr_schedule.learning_rate(1))
    self.assertAlmostEqual(.002, lr_schedule.learning_rate(2))
    self.assertAlmostEqual(.005, lr_schedule.learning_rate(5))
    self.assertAlmostEqual(.009, lr_schedule.learning_rate(9))

    # Constant body.
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(10))
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(11))
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(20))
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(300))
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(4000))

  def test_no_warmup(self):
    lr_schedule = lr_functions.BodyAndTail(.02)
    self.assertEqual(.02, lr_schedule.learning_rate(1))
    self.assertEqual(.02, lr_schedule.learning_rate(20))
    self.assertEqual(.02, lr_schedule.learning_rate(300))
    self.assertEqual(.02, lr_schedule.learning_rate(4000))
    self.assertEqual(.02, lr_schedule.learning_rate(50000))
    self.assertEqual(.02, lr_schedule.learning_rate(600000))
    self.assertEqual(.02, lr_schedule.learning_rate(7000000))
    self.assertEqual(.02, lr_schedule.learning_rate(80000000))
    self.assertEqual(.02, lr_schedule.learning_rate(900000000))

  def test_no_body(self):
    lr_schedule = lr_functions.BodyAndTail(.25,
                                           body_start=25,
                                           tail_start=25,
                                           tail_fn=lr_functions.rsqrt)
    # Warm-up.
    self.assertAlmostEqual(.01, lr_schedule.learning_rate(1))
    self.assertAlmostEqual(.02, lr_schedule.learning_rate(2))
    self.assertAlmostEqual(.23, lr_schedule.learning_rate(23))
    self.assertAlmostEqual(.24, lr_schedule.learning_rate(24))

    # Tail
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(25)), lr_schedule.learning_rate(25))
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(26)), lr_schedule.learning_rate(26))
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(27)), lr_schedule.learning_rate(27))
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(300)), lr_schedule.learning_rate(300))
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(4000)), lr_schedule.learning_rate(4000))
    self.assertAlmostEqual(
        .25 * (5 / math.sqrt(50000)), lr_schedule.learning_rate(50000))

  def test_cosine_sawtooth_tail(self):
    steps_per_cycle = 180
    cosine_sawtooth = lr_functions.CosineSawtoothTail(steps_per_cycle,
                                                      min_value=.1)
    lr_schedule = lr_functions.BodyAndTail(.3,
                                           tail_start=1000,
                                           tail_fn=cosine_sawtooth.tail_fn)
    # Body
    self.assertEqual(.3, lr_schedule.learning_rate(1))
    self.assertEqual(.3, lr_schedule.learning_rate(2))
    self.assertEqual(.3, lr_schedule.learning_rate(998))
    self.assertEqual(.3, lr_schedule.learning_rate(999))

    # Tail, first cycle
    self.assertEqual(.3, lr_schedule.learning_rate(1000))
    self.assertAlmostEqual(.29998477, lr_schedule.learning_rate(1001))
    self.assertAlmostEqual(.28660254, lr_schedule.learning_rate(1030))
    self.assertAlmostEqual(.25, lr_schedule.learning_rate(1060))
    self.assertAlmostEqual(.20, lr_schedule.learning_rate(1090))
    self.assertAlmostEqual(.15, lr_schedule.learning_rate(1120))
    self.assertAlmostEqual(.10001523, lr_schedule.learning_rate(1179))

    # Second cycle
    self.assertEqual(.3, lr_schedule.learning_rate(1180))
    self.assertAlmostEqual(.29998477, lr_schedule.learning_rate(1181))
    self.assertAlmostEqual(.28660254, lr_schedule.learning_rate(1210))
    self.assertAlmostEqual(.25, lr_schedule.learning_rate(1240))
    self.assertAlmostEqual(.20, lr_schedule.learning_rate(1270))
    self.assertAlmostEqual(.15, lr_schedule.learning_rate(1300))
    self.assertAlmostEqual(.10001523, lr_schedule.learning_rate(1359))


if __name__ == '__main__':
  absltest.main()
