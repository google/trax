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

"""Trax fast math -- NumPy-style math on accelerators."""

from trax.fastmath.numpy import nested_map
from trax.fastmath.numpy import nested_map_multiarg
from trax.fastmath.numpy import nested_stack
from trax.fastmath.numpy import nested_zip
from trax.fastmath.numpy import tree_flatten
from trax.fastmath.numpy import tree_leaves
from trax.fastmath.numpy import tree_unflatten
from trax.fastmath.ops import *  # pylint: disable=wildcard-import
