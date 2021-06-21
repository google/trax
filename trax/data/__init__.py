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

"""Functions and classes for obtaining and preprocesing data.

The ``trax.data`` module presents a flattened (no subpackages) public API.
(Many of the functions and class initilizers in the API are also accessible for
gin configuration.) To use as a client, import ``trax.data`` and access
functions using ``data.foo`` qualified names; for example::

   from trax import data
   ...
   training_inputs = data.Serial(
     ...
     data.Tokenize(),
     data.Shuffle(),
     ...
  )

"""

from trax.data.debug_data_pipeline import debug_pipeline

from trax.data.inputs import add_loss_weights
from trax.data.inputs import addition_inputs
from trax.data.inputs import AddLossWeights
from trax.data.inputs import AppendValue
from trax.data.inputs import batch
from trax.data.inputs import Batch
from trax.data.inputs import bucket_by_length
from trax.data.inputs import BucketByLength
from trax.data.inputs import CastTo
from trax.data.inputs import ConcatenateToLMInput
from trax.data.inputs import consume_noise_mask
from trax.data.inputs import CountAndSkip
from trax.data.inputs import Dup
from trax.data.inputs import FilterByLength
from trax.data.inputs import FilterEmptyExamples
from trax.data.inputs import generate_random_noise_mask
from trax.data.inputs import generate_sequential_chunks
from trax.data.inputs import Log
from trax.data.inputs import MLM
from trax.data.inputs import PadToLength
from trax.data.inputs import Parallel
from trax.data.inputs import Prefetch
from trax.data.inputs import PrefixLM
from trax.data.inputs import random_spans_noise_mask
from trax.data.inputs import sequence_copy_inputs
from trax.data.inputs import Serial
from trax.data.inputs import shuffle
from trax.data.inputs import Shuffle
from trax.data.inputs import simple_sequence_copy_inputs
from trax.data.inputs import sine_inputs
from trax.data.inputs import TruncateToLength
from trax.data.inputs import UnBatch
from trax.data.inputs import UniformlySeek

from trax.data.tf_inputs import add_eos_to_output_features
from trax.data.tf_inputs import BertGlueEvalStream
from trax.data.tf_inputs import BertGlueTrainStream
from trax.data.tf_inputs import BertNextSentencePredictionInputs
from trax.data.tf_inputs import cifar10_augmentation_flatten_preprocess
from trax.data.tf_inputs import cifar10_augmentation_preprocess
from trax.data.tf_inputs import ConvertToUnicode
from trax.data.tf_inputs import CorpusToRandomChunks
from trax.data.tf_inputs import CreateAnnotatedDropInputs
from trax.data.tf_inputs import CreateAquaInputs
from trax.data.tf_inputs import CreateBertInputs
from trax.data.tf_inputs import CreateDropInputs
from trax.data.tf_inputs import CreateMathQAInputs
from trax.data.tf_inputs import data_streams
from trax.data.tf_inputs import detokenize
from trax.data.tf_inputs import downsampled_imagenet_flatten_bare_preprocess
from trax.data.tf_inputs import filter_dataset_on_len
from trax.data.tf_inputs import lm1b_preprocess
from trax.data.tf_inputs import mask_random_tokens
from trax.data.tf_inputs import sentencepiece_tokenize
from trax.data.tf_inputs import SentencePieceTokenize
from trax.data.tf_inputs import squeeze_targets_preprocess
from trax.data.tf_inputs import T5GlueEvalStream
from trax.data.tf_inputs import T5GlueEvalStreamsParallel
from trax.data.tf_inputs import T5GlueEvalTasks
from trax.data.tf_inputs import T5GlueTrainStream
from trax.data.tf_inputs import T5GlueTrainStreamsParallel
from trax.data.tf_inputs import TFDS
from trax.data.tf_inputs import tokenize
from trax.data.tf_inputs import Tokenize
from trax.data.tf_inputs import truncate_dataset_on_len
from trax.data.tf_inputs import vocab_size
from trax.data.tf_inputs import wmt_concat_preprocess
from trax.data.tf_inputs import wmt_preprocess

