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

"""A debugging decorator for TRAX input pipeline."""

import functools

from absl import logging
import gin


@gin.configurable(denylist=['f'])
def debug_pipeline(f, debug=False, method='pow', log_prefix=None):
  """Decorator for input pipeline generators that logs examples at intervals."""
  if not debug:
    return f

  assert method in ('pow', 'every')
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    count = 0
    prefix = log_prefix or f.__name__
    for example in f(*args, **kwargs):
      count += 1
      if method == 'every' or (method == 'pow' and (count & count - 1 == 0)):
        logging.info('%s example[%d] = %r', prefix, count, example)
      yield example

  return wrapper
