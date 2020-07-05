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
"""mlp -- functions that assemble "multilayer perceptron" networks."""

from trax import layers as tl


def PureMLP(
    layer_widths=(128, 64),
    activation_fn=tl.Relu,
    out_activation=False,
    flatten=True,
    mode='train'):
  """A "multilayer perceptron" (MLP) network.

  This is a classic fully connected feedforward network, with one or more
  layers and a (nonlinear) activation function between each layer. For
  historical reasons, such networks are often called multilayer perceptrons;
  but they are more accurately described as multilayer networks, where
  each layer + activation function is a perceptron-like unit (see, e.g.,
  [https://en.wikipedia.org/wiki/Multilayer_perceptron#Terminology]).

  Args:
    layer_widths: Tuple of ints telling the number of layers and the width of
        each layer. For example, setting `layer_widths=(128, 64, 32)` would
        yield 3 layers with successive widths of 128, 64, and 32.
    activation_fn: Layer that computes a nonlinear activation between pairs
        of fully connnected layers. An activation function typically acts
        elementwise, and its output has the same shape and dtype as its input.
    out_activation: If True, include a copy of the activation function as the
        last layer in the network.
    flatten: If True, insert a layer at the head of the network to flatten the
        input tensor into a matrix of shape (batch_size. -1).
    mode: Ignored.

  Returns:
    An assembled MLP network with the specified layers. This network can either
    be initialized and trained as a full model, or can be used as a building
    block in a larger network.
  """
  del mode

  layers = []
  for width in layer_widths:
    layers.append(tl.Dense(width))
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
  """An MLP network, with a final layer for n-way classification."""
  # TODO(jonni): Revisit naming of PureMLP vs. MLP.
  # TODO(jonni): Redo params for MLP to align with PureMLP?

  return tl.Serial(
      PureMLP(
          layer_widths=[d_hidden] * n_hidden_layers + [n_output_classes],
          activation_fn=activation_fn,
          flatten=flatten,
          mode=mode),
      tl.LogSoftmax(),
  )
