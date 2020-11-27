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
import enum

import gin
from trax.fastmath.jax import JAX_BACKEND
from trax.fastmath.numpy import NUMPY_BACKEND
from trax.fastmath.tf import TF_BACKEND


@enum.unique
class Backend(enum.Enum):
  JAX = 'jax'
  TFNP = 'tensorflow-numpy'
  NUMPY = 'numpy'


# For numpy and random modules, we need to call "backend()" lazily, only when
# the function is called -- so that it can be set by gin configs.
# (Otherwise, backend() is called on import before gin-config is parsed.)
# To do that, we make objects to encapsulated these modules.


# A class that just forwards attribute accesses to backend's numpy object.
class NumpyBackend:
  """Numpy functions accelerated to run on GPUs and TPUs. Use like numpy."""

  def __getattr__(self, attr):
    return getattr(backend()['np'], attr)

numpy = NumpyBackend()


class RandomBackend:
  """Backend providing random functions."""

  def get_prng(self, seed):
    return backend()['random_get_prng'](seed)

  def split(self, prng, num=2):
    return backend()['random_split'](prng, num)

  def fold_in(self, rng, data):
    return backend()['random_fold_in'](rng, data)

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


def fori_loop(lower, upper, body_fn, init_val):
  """Loop from `lower` to `upper` running `body_fn` starting from `init_val`.

  The semantics of `fori_loop` is as follows::

    def fori_loop(lower, upper, body_fn, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fn(i, val)
      return val

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fn: function of type `(int, a) -> a`.
    init_val: initial loop carry value of type `a`.

  Returns:
    Loop value from the final iteration.
  """
  if 'fori_loop' in backend():
    return backend()['fori_loop'](lower, upper, body_fn, init_val)
  # Use scan otherwise.
  def scanned_fn(loop_carry, _):
    i, x = loop_carry
    return (i + 1, body_fn(i, x)), None
  (_, result), _ = scan(
      scanned_fn, (lower, init_val), None, length=upper - lower)
  return result


def remat(*args, **kwargs):
  """Recompute everything in the backward pass to same memory."""
  return backend()['remat'](*args, **kwargs)


def cond(*args, **kwargs):
  """Conditional computation to run on accelerators."""
  return backend()['cond'](*args, **kwargs)


def lt(*args, **kwargs):
  """Less-than function for backends that do not override <."""
  return backend()['lt'](*args, **kwargs)


def index_update(*args, **kwargs):
  return backend()['index_update'](*args, **kwargs)


def index_add(*args, **kwargs):
  return backend()['index_add'](*args, **kwargs)


def index_min(*args, **kwargs):
  return backend()['index_min'](*args, **kwargs)


def index_max(*args, **kwargs):
  return backend()['index_max'](*args, **kwargs)


def dynamic_slice(*args, **kwargs):
  return backend()['dynamic_slice'](*args, **kwargs)


def dynamic_slice_in_dim(*args, **kwargs):
  return backend()['dynamic_slice_in_dim'](*args, **kwargs)


def dynamic_update_slice(*args, **kwargs):
  return backend()['dynamic_update_slice'](*args, **kwargs)


def dynamic_update_slice_in_dim(*args, **kwargs):
  return backend()['dynamic_update_slice_in_dim'](*args, **kwargs)


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


def vmap(*args, **kwargs):
  """Vectorizes the specified function (returns a function)."""
  return backend()['vmap'](*args, **kwargs)


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


def custom_vjp(f, f_fwd, f_bwd, nondiff_argnums=()):
  """Set a custom vjp computation (override the default) for a function."""
  # Call backend custom_vjp if it exists.
  # TODO(lukaszkaiser): unify the APIs and remove nondiff_argnums altogether.
  if 'custom_vjp' in backend():
    return backend()['custom_vjp'](f, f_fwd, f_bwd)

  # Check that nondiff_argnums is (0, 1, ..., N) for some N.
  # Currently we only support nondiff_argnums at the front.
  counter = -1
  for i in nondiff_argnums:
    counter += 1
    if i != counter:
      raise ValueError('Currently we only support custom_vjps with all nondiff'
                       '_argnums up front, like (0,) or (0, 1) but not (1,) or'
                       ' (1, 2). Found: %s' % str(nondiff_argnums))

  # Use custom_grad.
  if counter == -1:  # no non-diff args
    def f_vjp(*args):
      out, residual = f_fwd(*args)
      def vjpfn(g):
        return f_bwd(residual, g)
      return out, vjpfn
    return backend()['custom_grad'](f_vjp, f)

  # Handle non-diff args by closure.
  def f_joint(*args):
    """This function takes all args, first counter+1 are non-diff ones."""
    nondiff_args = list(args[:counter+1])
    def f_diff(*diff_args):  # Takes only diff args, will define custom grad.
      args = nondiff_args + list(diff_args)
      return f(*args)
    def f_vjp(*diff_args):  # Custom VJP for diff args.
      args = nondiff_args + list(diff_args)
      out, residual = f_fwd(*args)
      def vjpfn(g):
        bwd_args = [residual, g]
        res = f_bwd(*bwd_args)
        return res[counter+1:]
      return out, vjpfn
    # This is the function taking only diff args with custom vjp.
    f_diff_vjp = backend()['custom_grad'](f_vjp, f_diff)
    # Call it on the diff args.
    return f_diff_vjp(*args[counter+1:])
  return f_joint


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
  if 'dataset_as_numpy' in backend():
    return backend()['dataset_as_numpy'](*args, **kwargs)
  return JAX_BACKEND['dataset_as_numpy'](*args, **kwargs)


def device_count(*args, **kwargs):
  """Return the number of accelerators (GPUs or TPUs) available."""
  return backend()['device_count'](*args, **kwargs)


# Backend selection functions.

override_backend = None
default_backend = None
_backend_dict = {
    Backend.JAX: JAX_BACKEND,
    Backend.NUMPY: NUMPY_BACKEND,
    Backend.TFNP: TF_BACKEND,
}


def _assert_valid_backend_name(name):
  for backend_ in Backend:
    if backend_.value == name:
      return
  raise ValueError(f'No backend with name {name}')


def set_backend(name):
  """Sets the default backend to use in Trax."""
  if name:
    _assert_valid_backend_name(name)
  global default_backend
  default_backend = name


def _get_backend_from_string(name_str):
  # name is a string.
  for backend_ in Backend:
    if backend_.value == name_str:
      return _backend_dict[backend_]
  return JAX_BACKEND


@gin.configurable()
def backend(name='jax'):
  """Returns the backend used to provide fastmath ops ('tf' or 'jax')."""
  if override_backend:
    return _get_backend_from_string(override_backend)

  if default_backend:
    return _get_backend_from_string(default_backend)

  if isinstance(name, Backend):
    return _backend_dict[name]

  # name is a string.
  return _get_backend_from_string(name)


@contextlib.contextmanager
def use_backend(name):
  """Call fastmath functions with a specified backend."""
  if isinstance(name, Backend):
    name = name.value

  _assert_valid_backend_name(name)
  global override_backend
  prev_name_or_backend = override_backend
  override_backend = name
  # Run the decorated function in try-finally in case it throws, e.g. for tests.
  try:
    yield
  finally:
    override_backend = prev_name_or_backend


def backend_name():
  """Returns the name of the backend currently in use ('tf' or 'jax')."""
  return backend()['name']


def is_backend(backend_):
  return backend()['name'] == backend_.value
