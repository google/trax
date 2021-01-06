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
"""Tests for trax.supervised.history."""

from absl.testing import absltest

from trax.supervised import history as trax_history


class HistoryTest(absltest.TestCase):

  def test_unknown_mode(self):
    history = trax_history.History()
    history.append('train', 'metric1', 1, 0.1)
    self.assertEqual(history.get('unknown_mode', 'metric1'), [])

  def test_unknown_metric(self):
    history = trax_history.History()
    history.append('train', 'metric1', 1, 0.1)
    self.assertEqual(history.get('train', 'unknown_metric'), [])

  def test_serializer_and_deserializer(self):
    history = trax_history.History()
    history.append('train', 'metric1', 1, 0.1)
    json_object = history.to_dict()
    history2 = trax_history.History.from_dict(json_object)
    self.assertEqual(history2.get('train', 'metric1'), [(1, 0.1)])

  def test_modes(self):
    history = trax_history.History()
    history.append('train', 'metric1', 1, 0.1)
    history.append('test', 'metric2', 2, 0.2)
    self.assertEqual(history.modes, ['test', 'train'])

  def test_metrics_for_mode(self):
    history = trax_history.History()
    history.append('train', 'metric1', 1, 0.1)
    history.append('train', 'metric2', 2, 0.2)
    self.assertEqual(history.metrics_for_mode('train'), ['metric1', 'metric2'])


if __name__ == '__main__':
  absltest.main()
