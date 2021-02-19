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

"""Data imports in Trax."""
import gin

from trax.data import debug_data_pipeline
from trax.data import inputs
from trax.data import tf_inputs


# Ginify
def data_configure(*args, **kwargs):
  kwargs['module'] = 'trax.data'
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
AddLossWeights = data_configure(inputs.AddLossWeights)
AppendValue = data_configure(inputs.AppendValue)
Batch = data_configure(inputs.Batch)
BertNextSentencePredictionInputs = data_configure(
    tf_inputs.BertNextSentencePredictionInputs)
BucketByLength = data_configure(inputs.BucketByLength)
CastTo = data_configure(inputs.CastTo)
ConcatenateToLMInput = data_configure(inputs.ConcatenateToLMInput)
ConvertToUnicode = data_configure(tf_inputs.ConvertToUnicode)
CorpusToRandomChunks = data_configure(tf_inputs.CorpusToRandomChunks)
CountAndSkip = data_configure(inputs.CountAndSkip)
CreateAnnotatedDropInputs = data_configure(tf_inputs.CreateAnnotatedDropInputs)
CreateAquaInputs = data_configure(tf_inputs.CreateAquaInputs)
CreateBertInputs = data_configure(tf_inputs.CreateBertInputs)
CreateDropInputs = data_configure(tf_inputs.CreateDropInputs)
CreateMathQAInputs = data_configure(tf_inputs.CreateMathQAInputs)
CreateT5GlueInputs = data_configure(tf_inputs.CreateT5GlueInputs)
FilterByLength = data_configure(inputs.FilterByLength)
FilterEmptyExamples = data_configure(inputs.FilterEmptyExamples)
Log = data_configure(inputs.Log)
MixMLMAndPrefixLM = data_configure(inputs.MixMLMAndPrefixLM)
PadToLength = data_configure(inputs.PadToLength)
Parallel = data_configure(inputs.Parallel)
Prefetch = data_configure(inputs.Prefetch)
SentencePieceTokenize = data_configure(tf_inputs.SentencePieceTokenize)
Serial = data_configure(inputs.Serial)
Shuffle = data_configure(inputs.Shuffle)
TFDS = data_configure(tf_inputs.TFDS)
Tokenize = data_configure(tf_inputs.Tokenize)
TruncateToLength = data_configure(inputs.TruncateToLength)
UniformlySeek = data_configure(inputs.UniformlySeek)
add_loss_weights = inputs.add_loss_weights
batch = inputs.batch
bucket_by_length = inputs.bucket_by_length
consume_noise_mask = data_configure(inputs.consume_noise_mask)
debug_pipeline = debug_data_pipeline.debug_pipeline
detokenize = tf_inputs.detokenize
generate_prefix_lm_sequential_chunks = data_configure(
    inputs.generate_prefix_lm_sequential_chunks)
generate_random_noise_mask = data_configure(inputs.generate_random_noise_mask)
generate_sequential_chunks = data_configure(inputs.generate_sequential_chunks)
mask_random_tokens = data_configure(tf_inputs.mask_random_tokens)
random_spans_noise_mask = inputs.random_spans_noise_mask
sentencepiece_tokenize = tf_inputs.sentencepiece_tokenize
shuffle = inputs.shuffle
tokenize = tf_inputs.tokenize
vocab_size = tf_inputs.vocab_size
