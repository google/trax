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
"""Test for memory usage in Terraformer models.

This test is designed to run on TPUv3 hardware, processing 1 million tokens at a
time while just barely fitting within the 16 GB memory budget.
"""


from absl.testing import absltest



class TerraformerMemoryTest(absltest.TestCase):


  def test_terraformer_memory(self):
    pass  # TODO(jonni): Figure out an OSS-compatible memory test.


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
