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

"""NumPy like wrapper for Tensorflow."""


# Enable NumPy behavior globally
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# Make everything from tensorflow.experimental.numpy available
from tensorflow.experimental.numpy import *  # pylint: disable=wildcard-import
