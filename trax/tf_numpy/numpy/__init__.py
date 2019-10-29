# coding=utf-8
# Copyright 2019 The Trax Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import newaxis

from trax.tf_numpy.numpy import random

# pylint: disable=wildcard-import
from trax.tf_numpy.numpy.array_creation import *
from trax.tf_numpy.numpy.array_manipulation import *
from trax.tf_numpy.numpy.array_methods import *
from trax.tf_numpy.numpy.arrays import ndarray
from trax.tf_numpy.numpy.dtypes import *
from trax.tf_numpy.numpy.logic import *
from trax.tf_numpy.numpy.math import *
from trax.tf_numpy.numpy.utils import finfo
# pylint: enable=wildcard-import
