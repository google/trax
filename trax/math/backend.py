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

"""Trax math: all the primitive functions needed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import gin
from trax.math.jax import JAX_BACKEND
from trax.math.numpy import NUMPY_BACKEND
from trax.math.tf import TF_BACKEND


def backend_name():
  return backend()['name']


def logsumexp(*args, **kwargs):
  return backend()['logsumexp'](*args, **kwargs)


def expit(*args, **kwargs):
  return backend()['expit'](*args, **kwargs)


def sigmoid(*args, **kwargs):
  return backend()['expit'](*args, **kwargs)


def erf(*args, **kwargs):
  return backend()['erf'](*args, **kwargs)


def conv(*args, **kwargs):
  return backend()['conv'](*args, **kwargs)


def avg_pool(*args, **kwargs):
  return backend()['avg_pool'](*args, **kwargs)


def max_pool(*args, **kwargs):
  return backend()['max_pool'](*args, **kwargs)


def sum_pool(*args, **kwargs):
  return backend()['sum_pool'](*args, **kwargs)


def scan(*args, **kwargs):
  return backend()['scan'](*args, **kwargs)


def cond(*args, **kwargs):
  return backend()['cond'](*args, **kwargs)


def lt(*args, **kwargs):
  return backend()['lt'](*args, **kwargs)


def stop_gradient(*args, **kwargs):
  return backend()['stop_gradient'](*args, **kwargs)


def jit(*args, **kwargs):
  return backend()['jit'](*args, **kwargs)


def grad(*args, **kwargs):
  return backend()['grad'](*args, **kwargs)


def pmap(*args, **kwargs):
  return backend()['pmap'](*args, **kwargs)


def psum(*args, **kwargs):
  return backend()['psum'](*args, **kwargs)


def abstract_eval(*args, **kwargs):
  return backend()['abstract_eval'](*args, **kwargs)


def dataset_as_numpy(*args, **kwargs):
  return backend()['dataset_as_numpy'](*args, **kwargs)


def device_count(*args, **kwargs):
  return backend()['device_count'](*args, **kwargs)


# For numpy and random modules, we need to call "backend()" lazily, only when
# the function is called -- so that it can be set by gin configs.
# (Otherwise, backend() is called on import before gin-config is parsed.)
# To do that, we make objects to encapsulated these modules.


class RandomBackend(object):
  """Backend providing random functions."""

  def get_prng(self, seed):
    return backend()['random_get_prng'](seed)

  def split(self, prng, num=2):
    return backend()['random_split'](prng, num)

  def uniform(self, *args, **kwargs):
    return backend()['random_uniform'](*args, **kwargs)

  def randint(self, *args, **kwargs):
    return backend()['random_randint'](*args, **kwargs)

  def normal(self, *args, **kwargs):
    return backend()['random_normal'](*args, **kwargs)

  def bernoulli(self, *args, **kwargs):
    return backend()['random_bernoulli'](*args, **kwargs)


random = RandomBackend()


# A class that just forwards attribute accesses to backend's numpy object.
class NumpyBackend(object):

  def __getattr__(self, attr):
    return getattr(backend()['np'], attr)


numpy = NumpyBackend()


override_backend_name = None


@gin.configurable()
def backend(name='jax'):
  name = name if not override_backend_name else override_backend_name
  if name == 'numpy':
    return NUMPY_BACKEND
  elif name == 'tf':
    return TF_BACKEND
  return JAX_BACKEND


@contextlib.contextmanager
def use_backend(name):
  global override_backend_name
  prev_name = override_backend_name
  override_backend_name = name
  # Run the decorated function in try-finally in case it throws, e.g. for tests.
  try:
    yield
  finally:
    override_backend_name = prev_name
