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
"""Trax accelerated math operations for fast computing on GPUs and TPUs.

Import these operations directly from fastmath and import fastmath.numpy as np::

    from trax import fastmath
    from trax.fastmath import numpy as np

    x = np.array([1.0, 2.0])  # Use like numpy.
    y = np.exp(x)  # Common numpy ops are available and accelerated.
    z = fastmath.logsumexp(y)  # Special operations available from fastmath.

Trax uses either TensorFlow 2 or JAX as backend for accelerating operations.
You can select which one to use (e.g., for  debugging) with `use_backend`.
"""

import contextlib

import gin
from trax.fastmath.jax import JAX_BACKEND
from trax.fastmath.numpy import NUMPY_BACKEND
from trax.fastmath.tf import TF_BACKEND

# For numpy and random modules, we need to call "backend()" lazily, only when
# the function is called -- so that it can be set by gin configs.
# (Otherwise, backend() is called on import before gin-config is parsed.)
# To do that, we make objects to encapsulated these modules.


# A class that just forwards attribute accesses to backend's numpy object.
class NumpyBackend(object):
  """Numpy functions accelerated to run on GPUs and TPUs. Use like numpy."""

  def __getattr__(self, attr):
    return getattr(backend()['np'], attr)

numpy = NumpyBackend()


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


def logsumexp(*args, **kwargs):
  """Computes the log of the sum of exponentials of input elements."""
  return backend()['logsumexp'](*args, **kwargs)


def expit(*args, **kwargs):
  """Computes the expit (sigmoid) function."""
  return backend()['expit'](*args, **kwargs)


def sigmoid(*args, **kwargs):
  """Computes the sigmoid (expit) function."""
  return backend()['expit'](*args, **kwargs)


def erf(*args, **kwargs):
  """Computes the erf function."""
  return backend()['erf'](*args, **kwargs)


def conv(*args, **kwargs):
  """Computes a generalized convolution."""
  return backend()['conv'](*args, **kwargs)


def avg_pool(*args, **kwargs):
  """Average pooling."""
  return backend()['avg_pool'](*args, **kwargs)


def max_pool(*args, **kwargs):
  """Max pooling."""
  return backend()['max_pool'](*args, **kwargs)


def sum_pool(*args, **kwargs):
  """Sum pooling."""
  return backend()['sum_pool'](*args, **kwargs)


def sort_key_val(*args, **kwargs):
  """Sorts keys along dimension and applies same permutation to values."""
  return backend()['sort_key_val'](*args, **kwargs)


def scan(*args, **kwargs):
  """Scan to make recurrent functions run faster on accelerators."""
  return backend()['scan'](*args, **kwargs)


def cond(*args, **kwargs):
  """Conditional computation to run on accelerators."""
  return backend()['cond'](*args, **kwargs)


def lt(*args, **kwargs):
  """Less-than function for backends that do not override <."""
  return backend()['lt'](*args, **kwargs)


def stop_gradient(*args, **kwargs):
  """Identity on the forward pass but 0 (no gradient) on the backward pass."""
  return backend()['stop_gradient'](*args, **kwargs)


_disable_jit = False


def jit(*args, **kwargs):
  """Just-In-Time compiles the given function for use on accelerators."""
  global _disable_jit
  if _disable_jit:
    return args[0]  # jit(f, **unused_now_jit_kwargs) = f
  return backend()['jit'](*args, **kwargs)


def disable_jit():
  """Disables JIT-compilation; helpful for debugging."""
  global _disable_jit
  _disable_jit = True


def grad(*args, **kwargs):
  """Computes the gradient of the specified function (returns a function)."""
  return backend()['grad'](*args, **kwargs)


def value_and_grad(*args, **kwargs):
  """Computes the gradient of the specified function together with the value."""
  if 'value_and_grad' in backend():
    return backend()['value_and_grad'](*args, **kwargs)
  grad_fn = grad(*args, **kwargs)
  fn = args[0]
  has_aux = False
  if has_aux in kwargs:
    has_aux = kwargs['has_aux']
  if not has_aux:
    def val_and_grad(*fn_args, **fn_kwargs):
      return fn(*fn_args, **fn_kwargs), grad_fn(*fn_args, **fn_kwargs)
    return val_and_grad
  def val_and_grad_aux(*fn_args, **fn_kwargs):
    g, aux = grad_fn(*fn_args, **fn_kwargs)
    res, _ = fn(*fn_args, **fn_kwargs)
    return (res, aux), g
  return val_and_grad_aux


def vjp(*args, **kwargs):
  """Computes the vector-Jacobian product for the specified function."""
  return backend()['vjp'](*args, **kwargs)


def custom_grad(*args, **kwargs):
  """Set a custom gradient computation (override the default) for a function."""
  return backend()['custom_grad'](*args, **kwargs)


def pmap(*args, **kwargs):
  """Parallel-map to apply a function on multiple accelerators in parallel."""
  return backend()['pmap'](*args, **kwargs)


def psum(*args, **kwargs):
  """Parallel-sum to use within a pmap'd function for aggregation."""
  return backend()['psum'](*args, **kwargs)


def abstract_eval(*args, **kwargs):
  """Evaluates function just on signatures of parameters, return signatures."""
  return backend()['abstract_eval'](*args, **kwargs)


def dataset_as_numpy(*args, **kwargs):
  """Convert a tf.data.Dataset to a stream of numpy arrays."""
  return backend()['dataset_as_numpy'](*args, **kwargs)


def device_count(*args, **kwargs):
  """Return the number of accelerators (GPUs or TPUs) available."""
  return backend()['device_count'](*args, **kwargs)


# Backend selection functions.


override_backend_name = None


@gin.configurable()
def backend(name='jax'):
  """Return the backend used to provide fastmath ops ('tf' or 'jax')."""
  name = name if not override_backend_name else override_backend_name
  if name == 'numpy':
    return NUMPY_BACKEND
  elif name == 'tf':
    return TF_BACKEND
  return JAX_BACKEND


@contextlib.contextmanager
def use_backend(name):
  """Call fastmath functions with a specified backend."""
  global override_backend_name
  prev_name = override_backend_name
  override_backend_name = name
  # Run the decorated function in try-finally in case it throws, e.g. for tests.
  try:
    yield
  finally:
    override_backend_name = prev_name


def backend_name():
  """Returns the name of the backend curently in use ('tf' or 'jax')."""
  return backend()['name']
