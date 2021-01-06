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
"""Tests for importing Trax."""

from absl.testing import absltest


class ImportTest(absltest.TestCase):

  def test_import_trax(self):
    try:
      # Import trax
      import trax  # pylint: disable=g-import-not-at-top
      # Access a few symbols.
      dir(trax.fastmath)
      dir(trax.layers)
      dir(trax.models)
    except ImportError as e:
      raise e


if __name__ == '__main__':
  absltest.main()
