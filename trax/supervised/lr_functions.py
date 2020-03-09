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
"""Learning rate schedules as functions of time (step number).

This is work in progress, intertwined with ongoing changes in supervised
training and optimizers. When complete, the learning rate schedules in this
file are intended to replace the prior ones in trax/lr_schedules.py. The
current package location/name (trax/supervised/lr_functions.py) is
correspondingly temporary.
"""

import math


class BodyAndTail:
  """Defines a time curve as a linear warm-up, constant body, and curvy tail.

  The body is a span of constant learning rate, and can be the entire curve
  if one wants a simple constant learning rate for the whole training run.
  The warm-up, if present, is based on the line connecting points (0, 0) and
  (body_start, body_value). The tail, if defined, is a function from time to
  learning rate that is used for all training steps from tail_start on.
  """

  def __init__(self, body_value, body_start=1, tail_start=None, tail_fn=None):
    """Specifies a body-and-tail time curve.

    Args:
      body_value: Constant learning rate for the body of the curve (after
          warm-up and before tail). Also serves as the reference value for
          calculating warm-up values and tail values.
      body_start: Training step number at which the body starts. All steps from
          1 to body_step - 1 are computes using a linear warm-up. The default
          value of 1 corresponds to there being no warm-up.
      tail_start: Training step number at which the tail starts; must be >=
          body_start. If `None`, the body value remains until the end of
          training. If `tail_start` equals `body_start`, the curve has no
          constant body, instead goes directly from warm-up to tail.
      tail_fn: Function returning a floating point learning rate, given inputs:
            - step_number (absolute step number from the start of training)
            - tail_start (step number at which the tail starts)
            - body_value (value relative to which the tail should be computed)
    """
    if tail_start is not None and tail_fn is None:
      raise ValueError(
          f'Tail start has a value ({tail_start}) but tail_fn is None.')
    self._body_value = body_value
    self._body_start = body_start
    self._tail_start = tail_start
    self._tail_fn = tail_fn

  def learning_rate(self, step_number):
    """Returns the learning rate for the given step number."""
    if step_number < self._body_start:
      return (step_number / self._body_start) * self._body_value
    elif self._tail_start is not None and step_number >= self._tail_start:
      return self._tail_fn(step_number, self._tail_start, self._body_value)
    else:
      return self._body_value


def rsqrt(step_number, tail_start, body_value):
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


class CosineSawtoothTail:
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

  def tail_fn(self, step_number, tail_start, body_value):
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
