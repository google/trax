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

"""Dtypes and dtype utilities."""
import numpy as np

# We use numpy's dtypes instead of TF's, because the user expects to use them
# with numpy facilities such as `np.dtype(np.int64)` and
# `if x.dtype.type is np.int64`.
# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
from numpy import float32
from numpy import float64

# TODO(wangpeng): Make bfloat16 a numpy dtype instead of using TF's

# pylint: enable=g-bad-import-order
# pylint: enable=unused-import


_to_float32 = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("complex64"),
}


_allow_float64 = True


def is_allow_float64():
    return _allow_float64


def set_allow_float64(b):
    global _allow_float64
    _allow_float64 = b


def canonicalize_dtype(dtype):
    if not is_allow_float64():
        return _to_float32.get(dtype, dtype)
    else:
        return dtype


def _result_type(*arrays_and_dtypes):
    dtype = np.result_type(*arrays_and_dtypes)
    return canonicalize_dtype(dtype)


def default_float_type():
    """Gets the default float type.

    Returns:
      If `is_allow_float64()` is true, returns float64; otherwise returns float32.
    """
    if is_allow_float64():
        return float64
    else:
        return float32
