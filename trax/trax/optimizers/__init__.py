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
"""Optimizers for use with Trax layers."""

import gin

from trax.optimizers import adafactor
from trax.optimizers import adam
from trax.optimizers import base
from trax.optimizers import momentum
from trax.optimizers import rms_prop
from trax.optimizers import sm3


def opt_configure(*args, **kwargs):
  kwargs['module'] = 'trax.optimizers'
  return gin.external_configurable(*args, **kwargs)

# Optimizers (using upper-case names).
# pylint: disable=invalid-name
SGD = opt_configure(base.SGD)
Momentum = opt_configure(momentum.Momentum)
RMSProp = opt_configure(rms_prop.RMSProp)
Adam = opt_configure(adam.Adam)
Adafactor = opt_configure(adafactor.Adafactor)
SM3 = opt_configure(sm3.SM3)
