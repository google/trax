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

"""Data imports in Trax."""
import gin

from trax.data import inputs
from trax.data import tf_inputs


# Ginify
def data_configure(*args, **kwargs):
  kwargs['module'] = 'trax.data'
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
AddLossWeights = data_configure(inputs.AddLossWeights)
add_loss_weights = inputs.add_loss_weights
Batch = data_configure(inputs.Batch)
batch = inputs.batch
BucketByLength = data_configure(inputs.BucketByLength)
bucket_by_length = inputs.bucket_by_length
FilterByLength = data_configure(inputs.FilterByLength)
TruncateToLength = data_configure(inputs.TruncateToLength)
AppendValue = data_configure(inputs.AppendValue)
PadToLength = data_configure(inputs.PadToLength)
ConcatenateToLMInput = data_configure(inputs.ConcatenateToLMInput)
FilterEmptyExamples = data_configure(inputs.FilterEmptyExamples)
Log = data_configure(inputs.Log)
Serial = data_configure(inputs.Serial)
Shuffle = data_configure(inputs.Shuffle)
shuffle = inputs.shuffle
TFDS = data_configure(tf_inputs.TFDS)
Tokenize = data_configure(tf_inputs.Tokenize)
ConvertToUnicode = data_configure(tf_inputs.ConvertToUnicode)
tokenize = tf_inputs.tokenize
detokenize = tf_inputs.detokenize
vocab_size = tf_inputs.vocab_size
