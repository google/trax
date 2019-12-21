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

"""SGD optimizer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax.optimizers import base as opt_base


class SGD(opt_base.Optimizer):
  """Plain SGD optimizer."""

  def init(self, params):
    return None

  def update(self, step, grads, weights, slots, opt_params):
    del step
    del slots
    learning_rate = opt_params['learning_rate']
    return weights - (learning_rate * grads).astype(weights.dtype), None
