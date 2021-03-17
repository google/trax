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
r"""Learning rate (LR) schedules.

In Trax a learning rate schedule is a function:
:math:`\text{step} \mapsto \text{learning_rate}`.
This module provides helpers for constructing such functions. For example::

    constant(0.001)

returns a function that always returns `0.001`.
"""

import math
import gin
from trax.fastmath import numpy as jnp


@gin.configurable
def constant(value):
  """Returns an LR schedule that is constant from time (step) 1 to infinity."""
  return _BodyAndTail(value, body_start=1)


@gin.configurable
def warmup(n_warmup_steps, max_value):
  """Returns an LR schedule with linear warm-up followed by constant value.

  Args:
    n_warmup_steps: Number of steps during which the learning rate rises on
        a line connecting (0, 0) and (n_warmup_steps, max_value).
    max_value: Value for learning rate after warm-up has finished.
  """
  return _BodyAndTail(max_value, body_start=n_warmup_steps + 1)


@gin.configurable
def warmup_and_rsqrt_decay(n_warmup_steps, max_value):
  """Returns an LR schedule with warm-up + reciprocal square root decay."""
  return _BodyAndTail(max_value, tail_start=n_warmup_steps + 1, tail_fn=_rsqrt)


@gin.configurable
def multifactor(factors='constant * linear_warmup * rsqrt_decay',
                constant=0.1,  # pylint: disable=redefined-outer-name
                warmup_steps=400,
                decay_factor=0.5,
                steps_per_decay=20000,
                steps_per_cycle=100000,
                second_constant=0.01,
                second_constant_step=10000,
                minimum=0):
  """Factor-based learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_deay: Cyclic cosine decay, uses steps_per_cycle parameter.
  * two_constants: constant until second_constant_step, then switch to
    second_constant.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    constant: float, the starting constant for the learning rate schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.
    second_constant: float, the second constant for the learning rate schedule.
    second_constant_step: the step when the second_constant is triggered.
    minimum: if the computed rate is below the minimum, then return the minimum.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def learning_rate(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= constant
      elif name == 'two_constants':
        if step < second_constant_step:
          ret *= constant
        else:
          ret *= second_constant
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor ** (step//steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(
            0.0, (step - warmup_steps) / float(steps_per_cycle))
        ret *= (0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    # TODO(henrykm): return float(jnp.max(minimum, ret)) would be
    # better but causes TypeError: 'numpy.float64' object cannot
    # be interpreted as an integer
    if ret <= minimum:
      return minimum
    return ret

  return learning_rate


class _BodyAndTail:
  """Defines a curve over time as a linear ramp + constant body + curvy tail.

  The body is a span of constant learning rate, and can be the entire curve.
  The warm-up, if present, is based on the line connecting points (0, 0) and
  (body_start, body_value). The tail, if defined, is a function from time to
  learning rate that is used for all training steps from tail_start on.
  """

  def __init__(
      self, body_value, body_start=None, tail_start=None, tail_fn=None):
    """Specifies a body-and-tail time curve.

    Args:
      body_value: Constant learning rate for the body of the curve (after
          warm-up and before tail). Also is the reference (maximum) value for
          calculating warm-up values and tail values.
      body_start: Training step number at which the body starts. If None, takes
          its value from tail_start, which amounts to there being no body. All
          steps from 1 to body_start - 1 are computed using a linear warm-up.
      tail_start: Training step number at which the tail starts. If None, the
          body value remains until the end of training.
      tail_fn: Function returning a floating point learning rate, given inputs:
            - step_number (absolute step number from the start of training)
            - tail_start (step number at which the tail starts)
            - body_value (value relative to which the tail should be computed)
    """
    if body_start is None and tail_start is None:
      raise ValueError('Both body start and tail start are None.')
    if tail_start is not None and tail_fn is None:
      raise ValueError(
          f'Tail start has value ({tail_start}) but tail_fn is None.')
    if body_start is None:
      body_start = tail_start if tail_start is not None else 1

    self._body_value = body_value
    self._body_start = body_start
    self._tail_start = tail_start
    self._tail_fn = tail_fn

  def __call__(self, step_number):
    """Returns the learning rate for the given step number."""
    if step_number < self._body_start:
      return (step_number / self._body_start) * self._body_value
    elif self._tail_start is not None and step_number >= self._tail_start:
      return self._tail_fn(step_number, self._tail_start, self._body_value)
    else:
      return self._body_value


def _rsqrt(step_number, tail_start, body_value):
  """Computes a tail using a scaled reciprocal square root of step number.

  Args:
    step_number: Absolute step number from the start of training.
    tail_start: Step number at which the tail of the curve starts.
    body_value: Value relative to which the tail should be computed.

  Returns:
    A learning rate value that falls as the reciprocal square root of the step
    number, scaled so that it joins smoothly with the body of a BodyAndTail
    instance.
  """
  return body_value * (math.sqrt(tail_start) / math.sqrt(step_number))


class _CosineSawtoothTail:
  """Cosine-sawtooth-shaped tail that simulates warm restarts.

  Creates a cyclic learning rate curve; each cycle is half of a cosine, falling
  from maximum value to minimum value. For motivation and further details, see
  Loshchilov & Hutter (2017) [https://arxiv.org/abs/1608.03983].
  """

  def __init__(self, steps_per_cycle, min_value=1e-5):
    """Configures the periodic behavior of this learning rate function.

    Args:
      steps_per_cycle: Number of training steps per sawtooth cycle. The
          learning rate will be highest at the start of each cycle, and lowest
          at the end.
      min_value: Minimum value, reached at the end of each cycle.
    """
    self._steps_per_cycle = steps_per_cycle
    self._min_value = min_value

  def __call__(self, step_number, tail_start, body_value):
    """Returns the learning rate for the given step number, when in the tail.

    Args:
      step_number: Absolute step number from the start of training.
      tail_start: Step number at which the tail of the curve starts.
      body_value: Value relative to which the tail should be computed.
    """
    max_value = body_value
    min_value = self._min_value
    position_in_cycle = (
        ((step_number - tail_start) / self._steps_per_cycle) % 1.0)
    theta = math.pi * position_in_cycle
    return min_value + (max_value - min_value) * .5 * (1 + math.cos(theta))
