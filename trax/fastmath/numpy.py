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

"""Trax fast math: pure numpy backend."""

import numpy as np


def get_prng(seed):
  """JAX-compatible way of getting PRNG seeds."""
  if np.shape(seed):
    raise TypeError('PRNGKey seed must be a scalar.')
  convert = lambda k: np.reshape(np.asarray(k, np.uint32), [1])
  k1 = convert(np.bitwise_and(np.right_shift(seed, 32), 0xFFFFFFFF))
  k2 = convert(np.bitwise_and(seed, 0xFFFFFFFF))
  return np.concatenate([k1, k2], 0)


NUMPY_BACKEND = {
    'name': 'numpy',
    'np': np,
    'jit': lambda f: f,
    'random_get_prng': get_prng,
    'random_split': lambda prng, num=2: (None,) * num,
    'expit': lambda x: 1. / (1. + np.exp(-x)),
}
