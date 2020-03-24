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

"""MLP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trax import layers as tl


def PureMLP(
    hidden_dims=(128, 64),
    activation_fn=tl.Relu,
    out_activation=False,
    flatten=True,
    mode='train',
):
  """A multi-layer feedforward (perceptron) network."""
  del mode

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(tl.Dense(hidden_dim))
    layers.append(activation_fn())

  if not out_activation:
    # Don't need the last activation.
    layers.pop()

  return tl.Serial(
      [tl.Flatten()] if flatten else [],
      layers,
  )


def MLP(d_hidden=512,
        n_hidden_layers=2,
        activation_fn=tl.Relu,
        n_output_classes=10,
        flatten=True,
        mode='train'):
  """A multi-layer feedforward network, with a classification head."""

  return tl.Serial(
      PureMLP(
          hidden_dims=[d_hidden] * n_hidden_layers + [n_output_classes],
          activation_fn=activation_fn,
          flatten=flatten,
          mode=mode),
      tl.LogSoftmax(),
  )
