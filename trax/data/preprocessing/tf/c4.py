# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""TensorFlow data sources and associated prepocessing functions."""

import functools

import gin
import tensorflow as tf
import tensorflow_text as tf_text

from trax.data.loader.tf.base import (
    add_eos_to_output_features,
    pad_dataset_to_length,
    truncate_dataset_on_len,
    unsupervised_preprocessors,
)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def c4_preprocess(
    dataset, training, max_target_length=-1, tokenization=None, spm_path=None
):
    """Pre-processing function for C4 dataset."""
    del training

    def unicode_decode_chars(features, targets):
        targets = tf.strings.unicode_decode(features["text"], "UTF-8")
        targets = tf.cast(targets, tf.int64)
        features["targets"] = targets
        features["inputs"] = targets
        return (features, targets)

    def spc_tokenize(tokenizer, features, targets):
        del targets
        tokenized_text = tokenizer.tokenize(features["text"])
        features["targets"] = tf.cast(tokenized_text, tf.int64)
        features["inputs"] = features["targets"]
        return features, features["targets"]

    if tokenization == "spc":
        if not spm_path:
            raise ValueError(
                "A valid SentencePiece model path (`spm_path`) must be provided."
            )

        with tf.io.gfile.GFile(spm_path, "rb") as f:
            spc_model = f.read()
        tokenizer = tf_text.SentencepieceTokenizer(model=spc_model)
        dataset = dataset.map(functools.partial(spc_tokenize, tokenizer))
    else:
        dataset = dataset.map(unicode_decode_chars)

    def target_right_length(_, target):
        return tf.less(tf.shape(target)[0], max_target_length + 1)

    if max_target_length > 0:
        dataset = dataset.filter(target_right_length)

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def c4_bare_preprocess_fn(
    dataset,
    training=True,
    spm_path=None,
    copy_pretokenized=True,
    sequence_length=None,
):
    """
    Preprocess C4 dataset to generate 'inputs' and 'targets' using SentencePiece.
    This version is T5-free and uses tensorflow_text for tokenization.
    """

    # Load SentencePiece model
    with tf.io.gfile.GFile(spm_path, "rb") as f:
        sp_model = f.read()
    tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)

    # Rekey: move "text" to "targets", and optionally to "inputs"
    def rekey(example):
        ret = {"targets": example["text"]}
        if copy_pretokenized:
            ret["targets_pretokenized"] = example["text"]
        return ret

    dataset = dataset.map(rekey, num_parallel_calls=tf.data.AUTOTUNE)

    # Tokenize using SentencePiece
    def tokenize(example):
        tokens = tokenizer.tokenize(example["targets"])
        tokens = tf.cast(tokens, tf.int64)
        example["inputs"] = tokens
        example["targets"] = tokens
        return example

    dataset = dataset.map(tokenize, num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocess the tokens - the exact preprocessors are set via gin.
    dataset = unsupervised_preprocessors(dataset, training, sequence_length=sequence_length)

    # Add EOS.
    dataset = add_eos_to_output_features(dataset, training)

    # Truncate and then pad the examples -- all examples have the same shape.
    dataset = truncate_dataset_on_len(dataset, training, sequence_length, True)
    dataset = pad_dataset_to_length(dataset, training, sequence_length)

    return dataset
