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
import itertools
import json
import os
import random
import re

import gin
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging

from trax import fastmath
from trax.data.encoder.encoder import SentencePieceEncoder
from trax.data.preprocessing.tf.math import (
    convert_float_to_mathqa,
    convert_to_subtract,
)

# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
_MAX_SKIP_EXAMPLES = 1e5

_T2T_TO_TFDS_MAP = {
    "t2t_translate_ende_wmt32k": "wmt14_translate/de-en",
    # Add more legacy mappings here if needed
}

def t5_data():
    """Get the T5 data module if available."""
    module = None
    try:
        import t5.data  # pylint: disable=g-import-not-at-top

        module = t5.data
    except AttributeError as e:
        logging.error("pip install t5")
        raise e
    return module

def random_split_text_tf(max_words_per_segment=512, text_key="text"):
    """
    Returns a TFDS preprocessing function that chunks long text randomly.
    """

    def preprocess_fn(dataset):
        def random_chunk(example):
            text = example[text_key]
            # Basic whitespace tokenizer (can be replaced with SentencePiece)
            tokens = tf.strings.split([text]).values
            length = tf.size(tokens)

            max_len = tf.minimum(length, max_words_per_segment)
            start = tf.random.uniform(
                shape=[], maxval=length - max_len + 1, dtype=tf.int32
            )
            chunk = tokens[start : start + max_len]

            # Rejoin into string or keep as tokens depending on downstream
            example[text_key] = tf.strings.reduce_join(chunk, separator=" ")
            return example

        return dataset.map(random_chunk, num_parallel_calls=tf.data.AUTOTUNE)

    return preprocess_fn


def _select_features(example, feature_list=None):
    """Select a subset of features from the example dict."""
    feature_list = feature_list or ["inputs", "targets"]
    return {f: example[f] for f in feature_list if f in example}

def next_sentence_prediction_tf(text_key="text", label_sentences=True, buffer_size=50000):
    """
    Returns a TFDS preprocessing function for NSP.
    Each example must contain a text_key (e.g., 'text') with paragraph(s).
    """
    def preprocess_fn(dataset):
        # First, buffer examples into memory
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        # Create a second shuffled dataset for random next sentences
        other_dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        # Zip datasets together
        combined = tf.data.Dataset.zip((dataset, other_dataset))

        def create_nsp_example(a, b):
            # Get the raw text tensors
            text_a = a[text_key]
            text_b = b[text_key]

            # Split into sentences
            sent_a = tf.compat.v1.strings.split(
                text_a, sep=". ", result_type="RaggedTensor"
            )
            sent_b = tf.compat.v1.strings.split(
                text_b, sep=". ", result_type="RaggedTensor"
            )

            # Safe access to RaggedTensor A
            sentences_a = sent_a.values
            row_splits_a = sent_a.row_splits
            has_sentences_a = tf.greater(tf.size(row_splits_a), 1)

            # Safe access to RaggedTensor B
            sentences_b = sent_b.values
            row_splits_b = sent_b.row_splits
            has_sentences_b = tf.greater(tf.size(row_splits_b), 1)

            # Get first sentence from A safely
            first_sentence = tf.cond(
                has_sentences_a,
                lambda: tf.cond(
                    tf.greater(row_splits_a[1], row_splits_a[0]),
                    lambda: sentences_a[row_splits_a[0]],
                    lambda: tf.constant("Empty first sentence."),
                ),
                lambda: tf.constant("No sentences in A."),
            )

            # Random decision: use text from B or a subsequent sentence from A
            use_random = tf.random.uniform(()) < 0.5

            # Function to get the second sentence from A (if available)
            def get_next_from_a():
                has_second_sentence = tf.logical_and(
                    has_sentences_a,
                    tf.greater(tf.size(row_splits_a), 2)
                )
                return tf.cond(
                    has_second_sentence,
                    lambda: tf.cond(
                        tf.greater(row_splits_a[2], row_splits_a[1]),
                        lambda: sentences_a[row_splits_a[1]],
                        lambda: first_sentence,
                    ),
                    lambda: first_sentence,
                )

            # Function to get first sentence from B
            def get_next_from_b():
                return tf.cond(
                    has_sentences_b,
                    lambda: tf.cond(
                        tf.greater(row_splits_b[1], row_splits_b[0]),
                        lambda: sentences_b[row_splits_b[0]],
                        lambda: tf.constant("Empty sentence from B."),
                    ),
                    lambda: tf.constant("No sentences in B."),
                )

            # Select the second sentence
            second_sentence = tf.cond(use_random, get_next_from_b, get_next_from_a)

            # Format as requested in the second implementation
            input_text = tf.strings.join(["sentence1: ", first_sentence, " sentence2: ", second_sentence])
            label = tf.cond(use_random,
                           lambda: tf.constant("not_next"),
                           lambda: tf.constant("next"))

            return {"inputs": input_text, "targets": label}

        return combined.map(create_nsp_example)

    return preprocess_fn


def no_preprocess(dataset, training):
    del training
    return dataset


def download_and_prepare(dataset_name, data_dir):
    """Downloads and prepares TFDS dataset, mapping from T2T if needed.

    Args:
      dataset_name: tfds dataset or t2t problem name prefixed by 't2t_'.
      data_dir: location of existing dataset or None.

    Returns:
      data_dir: path string of downloaded data.
    """
    # Translate legacy T2T dataset names to TFDS equivalents
    if dataset_name in _T2T_TO_TFDS_MAP:
        dataset_name = _T2T_TO_TFDS_MAP[dataset_name]

    if not data_dir:
        data_dir = os.path.expanduser("~/tensorflow_datasets/")
        dl_dir = os.path.join(data_dir, "download")
        logging.info(
            "No dataset directory provided. "
            "Downloading and generating dataset for %s inside data directory %s "
            "For large datasets it is better to prepare datasets manually!",
            dataset_name,
            data_dir,
        )

        tf.io.gfile.makedirs(data_dir)
        tf.io.gfile.makedirs(dl_dir)
        # Download and prepare TFDS dataset.
        tfds_builder = tfds.builder(dataset_name)
        tfds_builder.download_and_prepare(download_dir=dl_dir)
    else:
        data_dir = os.path.expanduser(data_dir)
    return data_dir


def dataset_to_stream(dataset, input_name):
    """Takes a tf.Dataset and creates a numpy stream of ready batches."""
    # All input-pipeline processing should be on CPU.
    for example in fastmath.dataset_as_numpy(dataset):
        features = example[0]
        inp, out = features[input_name], example[1]
        mask = features["mask"] if "mask" in features else None
        # Some accelerators don't handle uint8 well, cast to int.
        if isinstance(inp, np.uint8):
            inp = inp.astype(np.int32)
        if isinstance(out, np.uint8):
            out = out.astype(np.int32)
        yield (inp, out) if mask is None else (inp, out, mask)


@gin.configurable(module="trax.data")
def data_streams(
    dataset_name,
    data_dir=None,
    preprocess_fn=no_preprocess,
    bare_preprocess_fn=None,
    shuffle_buffer_size=1024,
    eval_holdout_size=0,
    input_name=None,
    target_name=None,
):
    """Creates `(train, eval)` data sources from ``dataset_name``.

    Args:
      dataset_name: Name of dataset belonging to TFDS or T2T. T2T dataset names
        must start with ``'t2t_'``.
      data_dir: Directory where the data is located.
      preprocess_fn: Function to use for pre-processing after appending targets to
        inputs.
      bare_preprocess_fn: Function to use for pre-processing before appending
        targets to inputs.
      shuffle_buffer_size: Size of the shuffle buffer.
      eval_holdout_size: If greater than 0, specifies a fraction of training data
        to siphon off and use as eval data, in place of an separate eval split.
      input_name: Name of the inputs from the dictionary.
      target_name: Name of the outputs either from the dictionary or as a result
        of post-processing.

    Returns:
      A pair of functions, `(f, g)` for use as data sources; call `f()` to get an
      iterator of training data samples, and call `g()` to get an iterator of eval
      data samples.
    """
    data_dir = download_and_prepare(dataset_name, data_dir)

    cache = []

    def stream(which):
        """Create the stream, cache TF streams if needed."""
        if not cache:
            cache.append(
                _train_and_eval_streams(
                    dataset_name,
                    data_dir,
                    preprocess_fn,
                    bare_preprocess_fn,
                    shuffle_buffer_size,
                    eval_holdout_size,
                    input_name,
                    target_name,
                )
            )

        (train_ds, eval_ds, input_name_c) = cache[0]
        dataset = eval_ds if which == "eval" else train_ds
        return dataset_to_stream(dataset, input_name_c)

    train_stream = lambda: stream("train")
    eval_stream = lambda: stream("eval")
    return train_stream, eval_stream


def load_translation_dataset(
    dataset_name="wmt14_translate/de-en",
    data_dir=None,
    train_shuffle_files=True,
    eval_shuffle_files=False,
    input_key="en",
    target_key="de",
):
    """
    Loads translation dataset and prepares train/eval tf.data.Datasets with mapped (inputs, targets).
    """
    data_dir = os.path.expanduser(data_dir or "~/tensorflow_datasets")
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()

    def _map_example(example):
        return {"inputs": example[input_key], "targets": example[target_key]}

    # Load and preprocess splits
    train_ds = tfds.load(
        dataset_name,
        split="train",
        shuffle_files=train_shuffle_files,
        data_dir=data_dir,
    ).map(_map_example)

    eval_ds = tfds.load(
        dataset_name,
        split="validation",
        shuffle_files=eval_shuffle_files,
        data_dir=data_dir,
    ).map(_map_example)

    supervised_keys = (["inputs"], ["targets"])
    return train_ds, eval_ds, supervised_keys


def _train_and_eval_streams(
    dataset,
    data_dir,
    preprocess_fn,
    bare_preprocess_fn,
    shuffle_buffer_size,
    eval_holdout_size,
    input_name,
    target_name,
):
    """Return train and eval batches with input name and shape."""
    (train_data, eval_data, keys) = _train_and_eval_dataset(
        dataset, data_dir, eval_holdout_size
    )
    # If provided select input_name/target_name else fall back to keys if that is
    # available, else [None].
    input_names = (
        [input_name]
        if input_name is not None
        else keys[0]
        if keys is not None
        else [None]
    )
    target_names = (
        [target_name]
        if target_name is not None
        else keys[1]
        if keys is not None
        else [None]
    )

    train_batches = _shuffle_data(
        train_data,
        target_names,
        True,
        shuffle_buffer_size,
        preprocess_fn,
        bare_preprocess_fn,
    )
    eval_batches = _shuffle_data(
        eval_data,
        target_names,
        False,
        shuffle_buffer_size,
        preprocess_fn,
        bare_preprocess_fn,
    )
    return (train_batches, eval_batches, input_names[0])


def _train_and_eval_dataset(
    dataset_name,
    data_dir,
    eval_holdout_size,
    train_shuffle_files=True,
    eval_shuffle_files=False,
    use_alt_eval=False,
    subsplit=None,
):
    """Return train and evaluation datasets, feature info and supervised keys.

    Args:
      dataset_name: a string, the name of the dataset; if it starts with 't2t_'
        then we'll search T2T Problem registry for it, otherwise we assume it is a
        dataset from TFDS and load it from there.
      data_dir: directory where the data is located.
      eval_holdout_size: float from 0 to <1; if >0 use this much of training data
        for evaluation (instead of looking for a pre-specified VALIDATION split).
      train_shuffle_files: Boolean determining whether or not to shuffle the train
        files at startup. Set to False if you want data determinism.
      eval_shuffle_files: Boolean determining whether or not to shuffle the test
        files at startup. Set to False if you want data determinism.
      use_alt_eval: If True, use the dataset's alternate/secondary eval split;
        else use the dataset's default/only eval split. Currently, only the
        `glue/mnli` dataset provides an alternate eval split, and this arg is
        ignored for other datasets.
      subsplit: a pair of floats (x, y), both in [0, 1], saying which part of the
        full training dataset we should return (default: all of it, [0, 1]).

    Returns:
      a 4-tuple consisting of:
       * the train tf.Dataset
       * the eval tf.Dataset
       * information about features: a python dictionary with feature names
           as keys and an object as value that provides .shape and .n_classes.
       * supervised_keys: information what's the input and what's the target,
           ie., a pair of lists with input and target feature names.
    """
    logging.info("Building TF data pipeline for %s", dataset_name)
    if dataset_name.startswith("t2t_"):
        return _train_and_eval_dataset_v1(
            dataset_name[4:], data_dir, train_shuffle_files, eval_shuffle_files
        )
    dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
    info = dataset_builder.info
    splits = dataset_builder.info.splits
    if dataset_name != "c4/multilingual" and tfds.Split.TRAIN not in splits:
        raise ValueError("To train we require a train split in the dataset.")
    train_split = tfds.Split.TRAIN if dataset_name != "c4/multilingual" else "en"
    eval_split = None
    train_examples = info.splits[train_split].num_examples
    eval_holdout_examples = int(train_examples * eval_holdout_size)
    if eval_holdout_examples > 0 or subsplit is not None:
        if subsplit is None:
            subsplit = (0, 1)
        n_train = train_examples - eval_holdout_examples
        train_start = int(n_train * subsplit[0])
        train_end = int(n_train * subsplit[1])
        if train_end - train_start < 1:
            raise ValueError(
                "Requested train subsplit has no examples: "
                "n_train %d subsplit %s" % (n_train, subsplit)
            )
        # Eval holdout examples from the end of the training set.
        if eval_holdout_examples > 0:
            eval_split = f"{train_split}[-{eval_holdout_examples}:]"
        # Shard the training set for this host.
        train_split = f"{train_split}[{train_start}:{train_end}]"

    if dataset_name == "glue/mnli":
        eval_split = "validation_mismatched" if use_alt_eval else "validation_matched"
    elif dataset_name == "c4/multilingual":
        eval_split = "en-validation"
    elif eval_split is None:
        if tfds.Split.VALIDATION not in splits and "test" not in splits:
            raise ValueError("We require a validation or test split in the dataset.")
        eval_split = tfds.Split.VALIDATION
        if tfds.Split.VALIDATION not in splits:
            eval_split = tfds.Split.TEST

    train = tfds.load(
        name=dataset_name,
        split=train_split,
        data_dir=data_dir,
        shuffle_files=train_shuffle_files,
    )
    valid = tfds.load(
        name=dataset_name,
        split=eval_split,
        data_dir=data_dir,
        shuffle_files=eval_shuffle_files,
    )
    keys = None
    if info.supervised_keys:
        keys = ([info.supervised_keys[0]], [info.supervised_keys[1]])
    return train, valid, keys


def _train_and_eval_dataset_v1(
    dataset_name="wmt14_translate/de-en",
    data_dir=None,
    train_shuffle_files=True,
    eval_shuffle_files=False,
):
    """Return train and evaluation datasets, feature info and supervised keys."""
    train_ds, eval_ds, supervised_keys = load_translation_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_shuffle_files=train_shuffle_files,
        eval_shuffle_files=eval_shuffle_files,
        input_key="en",
        target_key="de",
    )

    # You can take an example to determine input key if needed
    examples = list(tfds.as_numpy(train_ds.take(1)))
    input_key = "inputs" if "inputs" in examples[0] else "targets"
    return train_ds, eval_ds, ([input_key], ["targets"])


def _shuffle_data(
    dataset,
    target_names,
    training,
    shuffle_buffer_size,
    preprocess_fn,
    bare_preprocess_fn,
):
    """Shuffle the given dataset and run pre-processing."""

    def append_targets(example):
        """Append targets to the example dictionary. Needed for Keras."""
        if len(target_names) == 1:
            return (example, example[target_names[0]])
        targets = {}
        for name in target_names:
            targets[name] = example[name]
        return (example, targets)

    # `bare_preprocess_fn` is called before appending targets etc.
    if bare_preprocess_fn is not None:
        dataset = bare_preprocess_fn(dataset, training)
    dataset = dataset.map(append_targets)
    # TODO(pkozakowski): Repeat both the training and evaluation set, so we don't
    # have incomplete batches during evaluation. This will be a problem when we
    # add an option to evaluate on the whole dataset, then we'll need to think of
    # a different solution.
    dataset = dataset.repeat()
    if training:
        # Skip a random fraction at the beginning of the stream.  The skip is
        # essential for synchronous highly-parallel training to avoid multiple
        # replicas reading the same data in lock-step.
        dataset = dataset.skip(random.randint(0, _MAX_SKIP_EXAMPLES))
    dataset = preprocess_fn(dataset, training)
    dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.prefetch(8)


@gin.configurable(module="trax.data")
def TFDS(  # pylint: disable=invalid-name
    dataset_name,
    data_dir=None,
    tfds_preprocess_fn=None,
    keys=None,
    train=True,
    use_alt_eval=False,
    shuffle_train=True,
    host_id=None,
    n_hosts=None,
    eval_holdout_size=0,
):
    """Creates a data source from TensorFlow dataset ``dataset_name``.

    Args:
      dataset_name: Name of the dataset, as registered in TensorFlow datasets
        (e.g., ``'glue/mnli'``).
      data_dir: Directory where the data is located.
      tfds_preprocess_fn: If specified, function that applies to items in raw
        dataset (before selecting specific features).
      keys: Tuple of dataset-specific strings that select features from the
        dataset.
      train: If True, select the training split from the dataset; else select an
        eval split.
      use_alt_eval: If True, and if ``train`` is False, select the dataset's
        alternate eval split if it has one (or fall back to the dataset's only
        eval split). This currently affects only the `glue/mnli` dataset.
      shuffle_train: If True, have TensorFlow pre-shuffle the training data; else
        receive training data in deterministic sequence.
      host_id: Integer id used for tracking data subsplits, in cases where
        ``n_hosts`` > 1.
      n_hosts: If greater than 1, prepare data subsplits for the given number of
        hosts.
      eval_holdout_size: If greater than 0, specifies a fraction of training data
        to siphon off and use as eval data, in place of an separate eval split.

    Returns:
      A function `f` for use as a training or eval data source; call `f()` to get
      an iterator of data samples.
    """
    data_dir = download_and_prepare(dataset_name, data_dir)

    host_id = jax.process_index() if host_id is None else host_id
    n_hosts = n_hosts or jax.host_count()
    if n_hosts > 1:
        subsplit = (host_id / n_hosts, (host_id + 1) / n_hosts)
    else:
        subsplit = None
    train_data, eval_data, _ = _train_and_eval_dataset(
        dataset_name,
        data_dir,
        eval_holdout_size,
        train_shuffle_files=shuffle_train,
        use_alt_eval=use_alt_eval,
        subsplit=subsplit,
    )
    dataset = train_data if train else eval_data
    dataset = dataset if tfds_preprocess_fn is None else tfds_preprocess_fn(dataset)

    def select_from(example):
        return tuple(example[k] for k in keys)

    dataset = dataset.map(select_from)
    dataset = dataset.repeat()

    def gen(generator=None):
        del generator
        for example in fastmath.dataset_as_numpy(dataset):
            yield example

    return gen


@gin.configurable(module="trax.data")
def CorpusToRandomChunks(dataset_name, num_tokens=512, train=True):  # pylint: disable=invalid-name
    return TFDS(
        dataset_name,
        tfds_preprocess_fn=random_split_text_tf(
            max_words_per_segment=num_tokens,
            text_key="text",
        ),
        train=train,
        keys=["text"],
    )


@gin.configurable(module="trax.data")
def CreateAquaInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    cumulative=False,
    rationale=False,
    correct_answer=False,
    correct_answer_given_reasoning=False,
    partial_reasoning=True,
    order_prediction=False,
):
    """Prepares Aqua inputs.

    Args:
      dataset_path: a path with the Aqua dataset.
      train: if True, then generate training examples, otherwhise generate
        validation examples (the dataset has also a test set).
      cumulative: if set to True, then generate examples in the format input -
        problem + step1 + step3 + step3 target - step4 If set to False, then
        examples are in the format input - problem, target - all operations.
      rationale: if set to True, then input is the problem and the target is the
        rationale.
      correct_answer: if set to True, then input is the problem plus all possible
        answers and the target is the correct answer.
      correct_answer_given_reasoning: if set to True, then input is the problem
        plus reasoning (aka rationale) plus all possible answers and the target is
        the correct answer.
      partial_reasoning: an additional option related to
        correct_answer_given_reasoning; if set to True, then we take a random
        prefix of the reasoning.
      order_prediction: if set to True, then input is the problem and a list of
        all operations; with probability 0.5 two operations are swapped; the task
        consists in detecting whether the operations were swapped. A similar
        additional task was considered in https://arxiv.org/pdf/1909.11942.pdf and
          in a recent work of Piotr Piękos, henrykm@ and mateuszm@.

    Returns:
      aqua_yield_examples: a generator of Aqua examples; the generator yields
      non-tokenized examples - they can be further processed using for example
      the tokenize function from this module
    """
    if train:
        dataset_path = os.path.join(dataset_path, "train.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev.json")
    # Opening with GFile allows to use remotely stored files, e.g.
    # in a gs bucket.
    dataset_handle = tf.io.gfile.GFile(dataset_path, "r")
    dataset = []
    for line in dataset_handle:
        dataset.append(json.loads(line))

    def aqua_yield_examples(generator=None):
        del generator
        while True:
            for example in itertools.cycle(dataset):
                input_prefix = example["question"]
                steps = example["rationale"].split("\n")
                if cumulative:
                    for i in range(len(steps)):
                        input_values = "infer cumulative rationale: " + input_prefix
                        target_values = steps[i]
                        input_prefix += " " + steps[i]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                elif rationale:
                    input_values = "infer full rationale: " + input_prefix
                    target_values = example["rationale"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif correct_answer:
                    input_values = "infer correct answer: " + input_prefix
                    input_values += " " + " ".join(example["options"])
                    target_values = example["correct"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif correct_answer_given_reasoning:
                    input_values = (
                        "infer correct answer given reasoning: " + input_prefix
                    )
                    if partial_reasoning:
                        reasoning_list = example["rationale"].split("\n")
                        reasoning_list = reasoning_list[
                            0 : np.random.randint(0, len(reasoning_list))
                        ]
                        reasoning = "\n".join(reasoning_list)
                    else:
                        reasoning = example["rationale"]
                    input_values += (
                        " " + example["rationale"] + " " + " ".join(example["options"])
                    )
                    target_values = example["correct"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif order_prediction:
                    if np.random.uniform() < 0.5 and len(steps) >= 2:
                        idx = range(len(steps))
                        i1, i2 = random.sample(idx, 2)
                        steps[i1], steps[i2] = steps[i2], steps[i1]
                        target_values = "not_ordered"
                    else:
                        target_values = "ordered"
                    input_values = (
                        "order prediction: " + input_prefix + " " + "\n".join(steps)
                    )
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                else:
                    raise ValueError(
                        "One of the boolean parameters of the Aqua generator must be set to True."
                    )

    return aqua_yield_examples


@gin.configurable(module="trax.data")
def CreateAnnotatedDropInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    single_file=True,
    unique=False,
    total_number_of_samples=None,
    percentile=1.0,
):
    r"""Prepares annotated Drop inputs.

    Example of an annotated input which can be used with this interface:

    {
      'passage': 'The Armenian Prelature of Cyprus was established in 973 by
      Catholicos Khatchig I. Historically, the Prelature has been under the
      jurisdiction of the Catholicosate of the Great House of Cilicia, while today
      it is the oldest theme that falls under its jurisdiction. Since 2014 the
      Prelate, a Catholicosal Vicar General, has been Archbishop Nareg Alemezian.
      The parish priest in Nicosia is Fr. Momik Habeshian, while the parish priest
      in Larnaca and Limassol is Fr. Mashdots Ashkarian. For centuries, the
      Prelature building was located within the Armenian compound in Victoria
      street in walled Nicosia; when that area was taken over by Turkish-Cypriot
      extremists in 1963-1964, the Prelature was temporarily housed in Aram
      Ouzounian street and, later on, in Kyriakos Matsis street in Ayios
      Dhometios. Thanks to the efforts of Bishop Zareh Aznavorian and with
      financial aid from the Evangelical Church of Westphalia, the new Prelature
      building was erected in 1983, next to the Virgin Mary church and the Nareg
      school in Nicosia, by architects Athos Dikaios & Alkis Dikaios; it was
      officially inaugurated on 4 March 1984, during the pastoral visit of
      Catholicos Karekin II. By initiative of Archbishop Varoujan Hergelian, in
      1998 the basement of the building was renovated and the "Vahram Utidjian"
      Hall was formed; previously a store room, it became a reality from the
      proceeds of the auction in 1994 of the art collection that Vahram Utidjian
      had donated to the Prelature in 1954. It was inaugurated on 3 February 1999
      by Catholicos Aram I; numerous charity, communal and cultural events take
      place there. The Prelature\'s consistory houses a collection of
      ecclesiastical relics, some of which were previously in the old Virgin Mary
      church or the Magaravank.',
      'question': 'How many years after the Vahram Utidjian was donated to the
      Prelature was it sold at an auction?',
      'answer': 40,
      'calculation': 'subtract(n8,n9)'
    }

    In this example the calculation is formulated using the notation from the
    MathQA dataset, but this is not required. subtract(n8,n9) means that the
    answer 40 can be obtained through the substraction of the 9th and and the 10th
    number in the input. The input consists of the passage concatened with the
    question. The annotations can be generated using, for example, a method
    from the paper https://arxiv.org/abs/1909.00109.

    Args:
      dataset_path: a path with the Aqua dataset.
      train: if True, then generate training examples, otherwhise generate
        validation examples (the dataset has also a test set).
      single_file: if True, then look just for one file. If False, read all
        json files in a given directory and assume that each file contains one
        example. Applied only to training data.
      unique: if set to True, then the generator will provide at most one question
        per passage.
      total_number_of_samples: if set to a positive integer, then the total number
        of unique samples will be bounded total_number_of_samples.
      percentile: the percentile of the train dataset used for training; default
        set to 1., though setting to a lower value can be interesting when
        combined train is combined with another source of data.

    Returns:
      drop_annotated_yield_examples: a generator of annotated Drop examples;
      the generator yields non-tokenized examples - they can be further processed
      using for example the tokenize function from this module.
    """
    if train:
        if single_file:
            dataset_path = os.path.join(dataset_path, "train_annotated.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev_annotated.json")

    def load_dataset():
        dataset = []
        if single_file:
            # Opening with GFile allows to use remotely stored files, e.g.
            # in a gs bucket.
            dataset_handle = tf.io.gfile.GFile(dataset_path, "r")
            for line in dataset_handle:
                dataset.append(json.loads(line))
        else:
            all_files = tf.io.gfile.listdir(dataset_path)
            for filename in all_files:
                if "json" in filename:
                    print("Loading data from file {}".format(filename))
                    with tf.io.gfile.GFile(os.path.join(dataset_path, filename)) as f:
                        for line in f:
                            dataset.append(json.loads(line))
        print("The total size of the dataset {}".format(len(dataset)))
        return dataset[: int(len(dataset) * percentile)]

    def drop_annotated_yield_examples(generator=None):
        del generator
        while True:
            passages = set()
            unique_examples = set()
            # Notice that below we enable a poor man RL loop
            # aka the DAgger algorithm: https://arxiv.org/pdf/1011.0686.pdf
            # tl;dr: after parsing all examples we re-load the dataset - this
            # may become handy if a prediction service generates new examples.
            dataset = load_dataset()
            for example in dataset:
                # If total_number_of_samples is not None and we have reached this
                # number of samples, then we re-load the dataset.
                if total_number_of_samples:
                    if len(unique_examples) >= total_number_of_samples:
                        break
                # Do we have a pre-calculated input in the example?
                if "input" in example.keys():
                    question = example["input"]
                    # Remove the old prompt
                    question = question[question.find(":") + 2 :]
                else:
                    # If input is not present, then we expect that this is an
                    # original drop example.
                    if unique and example["passage"] in passages:
                        continue
                    passages.add(example["passage"])
                    question = example["passage"] + " " + example["question"]
                    list_num = [
                        float(num.replace(",", "").rstrip(".").lstrip("."))  # pylint: disable=g-complex-comprehension
                        for num in re.findall(
                            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                            question,
                        )
                    ]
                    for i in range(len(list_num)):
                        question += " n{} = {}".format(i, list_num[i])
                input_values = "drop annotated question: " + question
                target_values = example["calculation"]
                unique_examples.add((input_values, target_values))
                yield (
                    input_values,
                    target_values,
                    np.array([1] * len(target_values), dtype=np.int32),
                )

    return drop_annotated_yield_examples


@gin.configurable(module="trax.data")
def CreateDropInputs(train=True, mathqa_format=False):  # pylint: disable=invalid-name
    """Prepares Drop inputs.

    Args:
      train: if True, then generate training examples, otherwhise generate
        validation examples (the dataset has also a test set).
      mathqa_format: if True, then floats in targets are converted to the
        the MathQA convention and wrapped in the subtract operation.
        E.g. "3.13" is converted to "subtract(const_3_13,const_0)".

    Returns:
      drop_yield_examples: a generator of Drop examples; the generator yields
      non-tokenized examples - they can be further processed using for example
      the tokenize function from this module
    """
    if train:
        dataset = tfds.load(name="drop", split="train")
    else:
        dataset = tfds.load(name="drop", split="dev")
    dataset = tfds.as_numpy(dataset)

    def drop_yield_examples(generator=None):
        del generator
        while True:
            for example in itertools.cycle(dataset):
                input_values = (
                    "drop question: "
                    + example["passage"].decode("utf-8")
                    + " "
                    + example["question"].decode("utf-8")
                )
                target_values = example["answer"].decode("utf-8")
                # Apparently the dataset has some empty "target values" -
                # when such a value is encountered, the Tokenizer decides to assign
                # to it a float32 tensor and the training fails.
                if not target_values:
                    continue
                if mathqa_format:
                    if target_values.replace(".", "", 1).isdigit():
                        target_values = convert_to_subtract(
                            convert_float_to_mathqa(target_values)
                        )
                yield input_values, target_values, np.array(
                    [1] * len(target_values), dtype=np.int32
                )

    return drop_yield_examples


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def downsampled_imagenet_flatten_bare_preprocess(dataset, training):
    """Preprocessing for downsampled_imagenet.

    Args:
      dataset: the dataset.
      training: unused option.

    Returns:
      Flattened dataset.

    Preprocessing for downsampled_imagenet 32x32 and 64x64 generation from
    http://arxiv.org/abs/1601.06759 (page 8).
    """
    del training

    def flatten_image(features):
        img = features["image"]
        flat = tf.cast(tf.reshape(img, [-1]), tf.int64)

        new_features = {"image": flat}
        return new_features

    return dataset.map(flatten_image)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def concat_preprocess(dataset, training, pad_symbol=0):
    """Pre-processing function that concatenates input and target for LM."""
    del training

    def concat(features, targets):
        inp = features["inputs"]
        pad = tf.expand_dims(tf.zeros_like(inp[0]) + pad_symbol, axis=0)
        concat = tf.concat([pad, inp, pad, targets], axis=0)
        # Note: we're updating existing features dictionary here, so make sure
        # it is not re-used in some other ways outside of this function.
        features["inputs"] = concat
        return features, concat

    dataset = dataset.map(concat)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def squeeze_targets_preprocess(dataset, training):
    """Pre-processing function that squeezes last axis of targets."""
    del training

    def squeeze(features, targets):
        if targets.shape[-1] == 1:
            targets = tf.squeeze(targets, axis=-1)
        return features, targets

    dataset = dataset.map(squeeze)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def lm1b_preprocess(dataset, training, max_target_length=-1, max_eval_target_length=-1):
    """Preprocessing for LM1B: filter out targets exceeding maximum length."""

    def target_right_length(_, target):
        return tf.less(tf.shape(target)[0], max_target_length + 1)

    def eval_target_right_length(_, target):
        return tf.less(tf.shape(target)[0], max_eval_target_length + 1)

    if max_target_length > 0 and training:
        dataset = dataset.filter(target_right_length)

    if max_eval_target_length > 0 and not training:
        dataset = dataset.filter(eval_target_right_length)

    return dataset



@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def lm_token_preprocessing(dataset, training):
    """Concatenates inputs, 0, targets, with masking only for targets."""
    del training

    def concat_and_add_mask(x):
        inp = x["inputs"]
        targets = x["targets"]
        pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
        concat = tf.concat([inp, pad, targets], axis=0)
        mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
        x["inputs"] = concat
        x["targets"] = concat
        x["mask"] = mask
        return x

    dataset = dataset.map(concat_and_add_mask)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def bair_robot_pushing_preprocess(dataset, training):
    """Pre-processing function that concatenates input and target frames."""
    del training

    def concat_and_add_mask(features, targets):
        """Concatenate input and output frames to form a language modeling setup."""
        inp = features["inputs"]
        concat = tf.concat([inp, targets], axis=0)
        mask = tf.concat([tf.zeros_like(inp), tf.ones_like(targets)], axis=0)
        concat = tf.reshape(concat, (-1,))
        mask = tf.reshape(mask, (-1,))
        concat = tf.cast(concat, tf.int32)
        mask = tf.cast(mask, tf.float32)
        features["inputs"] = features["targets"] = concat
        features["mask"] = mask
        return features, concat

    dataset = dataset.map(concat_and_add_mask)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def filter_dataset_on_len(dataset, training, len_map=None, filter_on_eval=False):
    """Filters a dataset of lengths given in `len_map`.

    Args:
      dataset: `tf.data.Dataset` the dataset to filter.
      training: bool, true if we are in training mode.
      len_map: optional dict of str to (int, int). We filter examples where a
        feature's size is beyond the specified bounds. Ex:
        {'inputs': (1, 512), 'targets': (64, 128)} will keep only those examples
          where 1 <= len(inputs) <= 512 and 64 <= len(targets) <= 128.
      filter_on_eval: bool if true, we will filter in eval mode also.

    Returns:
      a filtered `tf.data.Dataset`.
    """
    if (len_map is None) or (not training and not filter_on_eval):
        return dataset

    assert isinstance(len_map, dict)
    for k, bounds in len_map.items():
        # pylint: disable=cell-var-from-loop
        # TODO(afrozm): Investigate `cell-var-from-loop` - since this is WAI and
        # there is a test too.
        def within_bounds(x, key, len_bounds):
            size = tf.shape(x[key])[0]
            min_len, max_len = len_bounds
            return (min_len <= size) and (size <= max_len)

        dataset = dataset.filter(lambda x: within_bounds(x, k, bounds))
        # pylint: enable=cell-var-from-loop

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def truncate_dataset_on_len(dataset, training, len_map=None, truncate_on_eval=False):
    """Truncates features in an example to lengths given in `len_map`.

    Args:
      dataset: `tf.data.Dataset` the dataset to filter.
      training: bool, true if we are in training mode.
      len_map: optional dict of str to int, we truncate examples where a feature's
        size is beyond the max. Ex: {'inputs': 512, 'targets': 64} will truncate
          examples to be within those bounds.
      truncate_on_eval: bool if true, we will truncate in eval mode also.

    Returns:
      a filtered `tf.data.Dataset`.
    """
    if (len_map is None) or (not training and not truncate_on_eval):
        return dataset

    assert isinstance(len_map, dict)

    def truncate_example(x):
        for key, max_len in len_map.items():
            x_len = tf.shape(x[key])[0]
            if x_len > max_len:
                x[key] = x[key][:max_len, ...]
        return x

    return dataset.map(truncate_example)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def pad_dataset_to_length(dataset, training, len_map=None):
    """Pad features less than specified length to specified length."""
    del training
    if len_map is None:
        return dataset

    def pad_to_len(x):
        for key, max_len in len_map.items():
            x_shape = tf.shape(x[key])
            x_len = x_shape[0]
            if x_len < max_len:
                pad_shape = [
                    max_len - x_len,
                ]
                zeros = tf.zeros(pad_shape, dtype=x[key].dtype)
                x[key] = tf.concat([x[key], zeros], 0)
        return x

    return dataset.map(pad_to_len)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def add_eos_to_output_features(dataset, training, output_features="targets", eos=1):
    """Adds `EOS` to all features in `output_features`."""
    del training
    if not isinstance(output_features, (list, tuple)):
        output_features = [output_features]

    def add_eos(x):
        for output_feature in output_features:
            x[output_feature] = tf.concat([x[output_feature], [eos]], axis=0)
        return x

    return dataset.map(add_eos)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def select_random_chunk_t5(dataset, training, sequence_length=None, output_features=None):
    """Select a random chunk from the input tokens."""
    del training

    def select_chunk(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]
        seq_len = tf.shape(tokens)[0]

        max_start = tf.maximum(seq_len - sequence_length, 0)
        start_index = tf.random.uniform(
            [], minval=0, maxval=max_start + 1, dtype=tf.int32
        )

        chunk = tokens[start_index : start_index + sequence_length]

        features["inputs"] = chunk
        features["targets"] = chunk

        return features

    return dataset.map(select_chunk, num_parallel_calls=tf.data.experimental.AUTOTUNE)

@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def split_tokens_t5(dataset, training, sequence_length=None, output_features=None):
    """Split tokens into two parts."""
    del training

    def split(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]
        seq_len = tf.shape(tokens)[0]

        split_point = seq_len // 2

        features["inputs"] = tokens[:split_point]
        features["targets"] = tokens[split_point:]

        return features

    return dataset.map(split, num_parallel_calls=tf.data.experimental.AUTOTUNE)

@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def denoise_t5(
        dataset, training, sequence_length=None, output_features=None, noise_density=0.15
    ):
    """Apply denoising to the tokens."""
    del training

    def apply_noise(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]

        mask = tf.random.uniform(tf.shape(tokens), minval=0, maxval=1) < noise_density
        noisy_tokens = tf.where(mask, tf.zeros_like(tokens), tokens)

        features["inputs"] = noisy_tokens
        features["targets"] = tokens

        return features

    return dataset.map(apply_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def _pad_punctuation(text):
  """Adds spaces around punctuation."""
  # Add space around punctuation.
  text = tf.strings.regex_replace(text, r'([[:punct:]])', r' \1 ')
  # Collapse consecutive whitespace into one space.
  text = tf.strings.regex_replace(text, r'\s+', ' ')
  return text

def _string_join(lst):
  # Join on space, but collapse consecutive spaces.
  out = tf.strings.join(lst, separator=' ')
  return tf.strings.regex_replace(out, r'\s+', ' ')

@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def squad_t5(dataset, training, include_context=True):
    """Convert SQuAD examples to a text2text pair.

    SQuAD produces examples with this form:
      {'id': <id>, context': <article>, 'question': <question>,
       'answers': { 'text': [<n answers>] }}
    This function will return examples of the format:
      {'inputs': 'question: <question> context: <article>',
       'targets': '<answer_0>',
       'id': <id>, 'question': <question>, 'context': <context>,
       'answers': [<n answers>]},

    Args:
      x: an example to process.
      include_context: a boolean
    Returns:
      A preprocessed example with the format listed above.
    """

    """Apply squad to the tokens."""
    del training

    def squad(x):
        a = _pad_punctuation(x["answers"]["text"])
        q = _pad_punctuation(x["question"])
        c = _pad_punctuation(x["context"])
        if include_context:
            inputs = _string_join(["question:", q, "context:", c])
        else:
            inputs = _string_join(["squad trivia question:", q])
        return {
            "inputs": inputs,
            "targets": a[0],
            "id": x["id"],
            "context": c,
            "question": q,
            "answers": a,
        }

    return dataset.map(squad, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def rekey_t5(dataset, training, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`.

    For example, if the dataset returns examples of the format:
    {'foo': 'something', 'bar': 'something else'}
    and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
    examples with the format
    {'boo': 'something', 'spar': 'something else'}

    If a mapping is to an empty key name or None, the new value is set to an empty
    string.

    Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys

    Returns:
      A preprocessed example with the format listed above.
    """

    del training

    def rekey(x):
        if key_map:
            return {
                new_key: x[old_key] if old_key else ""
                for new_key, old_key in key_map.items()
            }
        return x

    return dataset.map(rekey, num_parallel_calls=tf.data.experimental.AUTOTUNE)


_PREPROCESSOR_REGISTRY = {
    "next_sentence_prediction_tf": next_sentence_prediction_tf,
    "random_split_text_tf": random_split_text_tf,
    "select_random_chunk_t5": select_random_chunk_t5,
    "split_tokens_t5": split_tokens_t5,
    "denoise_t5": denoise_t5,
    "squad_t5": squad_t5,
    "rekey_t5": rekey_t5,
}


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def unsupervised_preprocessors(
    dataset, training, sequence_length=None, output_features=None, preprocessors=None
):
    """
    Apply a series of unsupervised preprocessors.

    Args:
        dataset: Input TensorFlow dataset
        sequence_length: Maximum sequence length
        output_features: Optional output features dictionary
        preprocessors: List of preprocessing functions to apply

    Returns:
        Preprocessed dataset
    """
    del training

    if preprocessors is None:
        return dataset

    for preprocessor in preprocessors:
        dataset = preprocessor(
            dataset, None, sequence_length=sequence_length, output_features=output_features
        )

    return dataset

@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def generic_text_dataset_preprocess_fn(
    dataset,
    training=True,
    text_preprocess_fns=None,
    token_preprocess_fns=None,
    spm_path=None,
    copy_pretokenized=False,
    debug_print_examples=False,
    debug_print_examples_rate=0.01,
):
    """Pre-processes, tokenizes and post-processes a `tf.data.Dataset`.

    Args:
      dataset: `tf.data.Dataset` to process.
      training: boolean, set to True if training, False otherwise.
      text_preprocess_fns: None or list of callables: `tf.data.Dataset`, bool ->
        `tf.data.Dataset` this operates before tokenization. Typically used to
        select which fields we want to learn over or change something into "text
        to text" form.
      token_preprocess_fns: None or list of callables: `tf.data.Dataset`, bool ->
        `tf.data.Dataset`, this operates after tokenization. Since this can view
        the tokenized fields, this can be used to filter on length etc.
      spm_path: None or str, path to a sentencepiece model to use for tokenization
        by default uses the 32k vocabulary from T5.
      copy_pretokenized: bool, if True retains the original fields after
        tokenization.
      debug_print_examples: bool, if True this prints examples to the logging
        stream for inspection, both before and after tokenization.
      debug_print_examples_rate: float, [0, 1.0], on average this fraction of
        dataset examples will be printed out in each phase i.e. pre and post
        tokenization.

    Returns:
      a `tf.data.Dataset` with all the preprocessing and tokenization performed.
    """

    # The assumption is that `text_preprocess_fns` finally gives us a dataset
    # which has `inputs` and `targets`.
    if text_preprocess_fns is not None:
        for text_preprocess_fn in text_preprocess_fns:
            dataset = text_preprocess_fn(dataset, training)

    # Print debugging examples if needed before tokenization.
    if debug_print_examples:
        def print_examples(x):
            if np.random.uniform() < debug_print_examples_rate:
                tf.print(x, output_stream=logging.info)
            return x

        dataset = dataset.map(print_examples)

    # Vocabulary for tokenization.
    tokenizer = SentencePieceEncoder(spm_path)

    # Tokenize the inputs and targets.
    def tokenize_fields(example):
        inputs = example.get("inputs", example["targets"])
        targets = example["targets"]

        tokenized_inputs = tf.cast(tokenizer.encode(inputs), tf.int64)
        tokenized_targets = tf.cast(tokenizer.encode(targets), tf.int64)

        new_example = {
            "inputs": tokenized_inputs,
            "targets": tokenized_targets,
        }
        if copy_pretokenized:
            new_example["inputs_pretokenized"] = inputs
            new_example["targets_pretokenized"] = targets

        return new_example

    dataset = dataset.map(tokenize_fields)

    # Apply the token-preprocessors.
    if token_preprocess_fns is not None:
        for token_preprocess_fn in token_preprocess_fns:
            dataset = token_preprocess_fn(dataset, training)

    if debug_print_examples:
        def print_examples_and_shapes(x):
            if np.random.uniform() < debug_print_examples_rate:
                tf.print(
                    "inputs_shape:",
                    tf.size(x["inputs"]),
                    "targets_shape:",
                    tf.size(x["targets"]),
                    "inputs:",
                    x["inputs"],
                    "targets:",
                    x["targets"],
                    output_stream=logging.info,  # or use a custom stream that writes to logging.info
                )
            return x

        dataset = dataset.map(print_examples_and_shapes)

    return dataset


@gin.configurable(module="trax.data")
def get_t5_preprocessor_by_name(name=None, fn_kwargs=None):
    """Returns a closure of any T5 preprocessor function with its arguments.

    The main use-case is to use this (with gin scopes) to make any preprocessor
    function available in a gin file to configure and use.

    See: `TFInputs.test_gin_configurable_preprocessors`

    Args:
      name: str, name of the preprocessor function to configure.
      fn_kwargs: optional dictionary, the arguments to configure, these will be
        partially applied to the function given by `name`.

    Returns:
      a closure of the preprocessor function along with its arguments, this
      function takes two arguments only, dataset and boolean training and ignores
      the training and calls the t5 processor with the dataset (and closed over
      arguments only).
    """

    if name is None or name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(f"Unknown or missing preprocessor name: '{name}'.")

    fn = _PREPROCESSOR_REGISTRY[name]
    if fn_kwargs:
        fn = functools.partial(fn, **fn_kwargs)

    # Ensure compatibility with trax data preprocessing signature
    return lambda ds, training: fn(ds, training)
