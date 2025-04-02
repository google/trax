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


import gin
import tensorflow_datasets as tfds

from data.preprocessing.tokenizer import tokenizer as tokenizer

from trax import data, supervised
from trax import layers as tl
from trax.data.loader.tf.base import TFDS, generic_text_dataset_preprocess_fn, t5_data
from trax.data.preprocessing.tf import bert as bert
from trax.fastmath import numpy as jnp

_GLUE_KEYS = {
    "cola": ("sentence",),
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


# Labels inferred from the T5 paper: https://arxiv.org/pdf/1910.10683.pdf
_GLUE_LABELS = {
    "cola": ("unacceptable", "acceptable"),
    "sst2": ("negative", "positive"),
    "mrpc": ("not_equivalent", "equivalent"),
    "qqp": ("not_duplicate", "duplicate"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "qnli": ("entailment", "not_entailment"),
    "rte": ("entailment", "not_entailment"),
    "wnli": ("sentence1", "sentence2"),
}

# Defining separate <Foo>TrainStream and <Foo>EvalStream functions (below)
# makes gin configuration expressions more direct. A single gin line can
# configure each; for example:
#
#   BertGlueTrainStream.benchmark= 'mnli'
#   BertGlueEvalStream.benchmark = 'mnli'


# pylint: disable=invalid-name
@gin.configurable(module="trax.data")
def BertGlueTrainStream(benchmark=gin.REQUIRED):
    """Returns a Bert-preprocessed training stream for ``benchmark``.

    Args:
      benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
          ``'mnli'``, ``'rte'``.
    """
    return _BertGlueDataStream(benchmark + "_t")


# GLUE evals need special handling because one eval in particular, MNLI, has
# two different eval sets: "matched" and "mismatched". The code in this module
# distinguishes between the two using the suffixes '_e' versus '_e2',
# respectively.
def _ensure_eval_suffix(benchmark):
    """Returns a string ending in an eval suffix; adds ``'_e'`` suffix if needed.

    Args:
      benchmark: Name of a benchmark or task, that might already include an
          eval-indicating suffix (``'_e'`` or ``'_e2'``).
    """
    if benchmark.endswith("_e") or benchmark.endswith("_e2"):
        return benchmark
    else:
        return benchmark + "_e"


@gin.configurable(module="trax.data")
def BertGlueEvalStream(benchmark=gin.REQUIRED):
    """Returns a Bert-preprocessed eval data stream for ``benchmark``.

    Args:
      benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
          ``'mnli'``, ``'rte'``. If the benchmark includes an alternate
          eval (e.g., MNLI's "mismatched" eval/validation split), you can
          specify it with an ``'_e2'`` suffix, e.g., ``'mnli_e2'``.
    """
    return _BertGlueDataStream(_ensure_eval_suffix(benchmark))


def _BertGlueDataStream(benchmark_id):
    """Returns a Bert-preprocessed data stream for ``benchmark_id``.

    Args:
      benchmark_id: String that indicates the name and data split of a GLUE
          benchmark. Data splits are indicated as underscore suffixes, e.g.,
          ``'cola_t'`` (Cola benchmark, training split), ``'rte_e'`` (RTE
          benchmark, eval/validation split), and ``'mnli_e2'`` (MNLI benchmark,
          alternate "mismatched" eval/validation split).
    """
    benchmark_id = _ensure_eval_suffix(benchmark_id)
    benchmark, split = benchmark_id.rsplit("_", 1)
    glue_data = TFDS(
        f"glue/{benchmark}",
        keys=_GLUE_KEYS[benchmark],
        train=(split == "t"),
        use_alt_eval=(split == "e2"),
    )
    return data.Serial(
        glue_data,
        tokenizer.Tokenize(),
        bert.CreateBertInputs(),
        data.Shuffle(),
        data.PadToLength(),
        data.TruncateToLength(),
        data.Batch(),
    )


@gin.configurable(module="trax.data")
def T5GlueTrainStream(benchmark=gin.REQUIRED):
    """Returns a T5-preprocessed training data stream for ``benchmark``.

    Args:
      benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
          ``'mnli'``, ``'rte'``.
    """
    return _T5GlueDataStream(benchmark + "_t")


@gin.configurable(module="trax.data")
def T5GlueTrainStreamsParallel(
    benchmark_list=gin.REQUIRED,
    counters=None,
    reweight_by_minimum=False,
    gradually_reweight=False,
):
    """Returns a parallel set of training streams, based on ``benchmark_list``.

    Args:
      benchmark_list: List of simple lower-case names of GLUE benchmarks, e.g.,
          ``'cola'``, ``'mnli'``, ``'rte'``.
      counters: a list of counters to be passed to data.Parallel, e.g.,
      [8551, 392702, 2490] would be a reasonable counterpart to
      benchmark_list = ["cola", "mnli", "rte"], see
      https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/glue_utils.py#L42
      for more details on counters.
      reweight_by_minimum: divide by the minimal counter.
      gradually_reweight: a more refined reweighting policy, see inputs.py
        for more details.
    """
    stream_list = list(map(T5GlueTrainStream, benchmark_list))
    return data.Parallel(
        stream_list,
        counters=counters,
        reweight_by_minimum=reweight_by_minimum,
        gradually_reweight=gradually_reweight,
    )()


@gin.configurable(module="trax.data")
def T5GlueEvalStream(benchmark=gin.REQUIRED):
    """Returns a T5-preprocessed eval data stream for ``benchmark``.

    Args:
      benchmark: Simple lower-case name of a GLUE benchmark, e.g., ``'cola'``,
          ``'mnli'``, ``'rte'``. If the benchmark includes an alternate
          eval (e.g., MNLI's "mismatched" eval/validation split), you can
          specify it with an ``'_e2'`` suffix, e.g., ``'mnli_e2'``.
    """
    return _T5GlueDataStream(_ensure_eval_suffix(benchmark))


@gin.configurable(module="trax.data")
def T5GlueEvalStreamsParallel(benchmark_list=gin.REQUIRED):
    """Returns a parallel set of T5 eval streams, based on ``benchmark_list``.

    Args:
      benchmark_list: List of strings, each of which is a simple lower-case name
          of a GLUE benchmark, e.g., ``'cola'``, ``'mnli'``, ``'rte'``. If a
          benchmark includes an alternate eval (e.g., MNLI's "mismatched"
          eval/validation split), you can specify it with an ``'_e2'`` suffix,
          e.g., ``'mnli_e2'``.
    """
    stream_list = list(map(T5GlueEvalStream, benchmark_list))
    return data.Parallel(stream_list)()


def _T5GlueDataStream(benchmark_id, t5_tokenization=False):
    """Returns a T5-preprocessed data stream for ``benchmark_id``.

    Args:
      benchmark_id: String that indicates the name and data split of a GLUE
          benchmark. Data splits are indicated as underscore suffixes, e.g.,
          ``'cola_t'`` (Cola benchmark, training split), ``'rte_e'`` (RTE
          benchmark, eval/validation split), and ``'mnli_e2'`` (MNLI benchmark,
          alternate "mismatched" eval/validation split).
      t5_tokenization: if true, then use t5_tokenization.
    """
    return data.Serial(
        _t5_glue_data_split(benchmark_id)
        if t5_tokenization
        else _t5_glue_data_split_no_token(benchmark_id),
        tokenizer.Tokenize(),
        data.Shuffle(),
        data.PadToLength(),
        data.TruncateToLength(),
        data.Batch(),
    )


@gin.configurable(module="trax.data")
def T5GlueEvalTasks(benchmark_list=gin.REQUIRED):
    """Returns a list of T5 GLUE eval tasks, based on ``benchmark_list``.

    Args:
      benchmark_list: List of strings, each of which indicates the name and
          data split of a GLUE benchmark. Data splits are indicated as underscore
          suffixes, e.g., ``'cola_t'`` (Cola benchmark, training split),
          ``'rte_e'`` (RTE benchmark, eval/validation split), and ``'mnli_e2'``
          (MNLI alternate "mismatched" eval/validation split).
    """
    task_list = list(map(_T5GlueEvalTask, benchmark_list))
    return task_list


def _T5GlueEvalTask(benchmark_id):
    """Returns a T5 GLUE eval task, based on ``benchmark_id``."""
    eval_data = T5GlueEvalStream(benchmark_id)
    benchmark_id = _ensure_eval_suffix(benchmark_id)
    metrics = [tl.WeightedCategoryAccuracy(), tl.SequenceAccuracy()]
    benchmark, split = benchmark_id.rsplit("_", 1)
    if benchmark == "cola":
        name_upper = "Cola"
    elif benchmark == "mnli":
        name_upper = "MNLI_matched" if split == "e" else "MNLI_mismatched"
    else:
        name_upper = benchmark.upper()
    return supervised.training.EvalTask(
        eval_data(),
        metrics,
        metric_names=[f"{name_upper} accuracy", f"{name_upper} sequence accuracy"],
    )


def _t5_glue_data_split_no_token(benchmark_id):
    """Returns a GLUE data split prepared with the standard T5 preprocessor."""
    benchmark, split = _t5_glue_benchmark_and_split(benchmark_id)
    dataset = tfds.load(name=f"glue/{benchmark}", split=split)
    processed_dataset = t5_data().preprocessors.glue(  # pylint: disable=g-long-lambda
        dataset, benchmark_name=benchmark, label_names=_GLUE_LABELS[benchmark]
    )

    def stream_of_inputs_targets_weights(generator=None):
        del generator
        while True:
            for example in processed_dataset:
                input_values = example["inputs"].numpy()
                target_values = example["targets"].numpy()
                yield (input_values, target_values, jnp.array([1] * len(target_values)))

    return stream_of_inputs_targets_weights


def _t5_glue_data_split(benchmark_id):
    """Returns a GLUE data split prepared with the standard T5 preprocessor."""
    benchmark, split = _t5_glue_benchmark_and_split(benchmark_id)
    dataset = tfds.load(name=f"glue/{benchmark}", split=split)
    processed_dataset = generic_text_dataset_preprocess_fn(
        dataset,
        spm_path=t5_data().DEFAULT_SPM_PATH,
        text_preprocess_fns=[
            lambda ds, training: t5_data().preprocessors.glue(  # pylint: disable=g-long-lambda
                ds, benchmark_name=benchmark, label_names=_GLUE_LABELS[benchmark]
            )
        ],
        copy_pretokenized=True,
        debug_print_examples=True,
        debug_print_examples_rate=0.05,
    )
    dataset_as_numpy = tfds.as_numpy(processed_dataset)

    def stream_of_inputs_targets_weights(generator=None):
        del generator
        while True:
            for example in dataset_as_numpy:
                input_values = example["inputs"]
                target_values = example["targets"]
                yield (
                    jnp.array(input_values),
                    jnp.array(target_values),
                    jnp.array([1] * len(target_values)),
                )

    return stream_of_inputs_targets_weights


def _t5_glue_benchmark_and_split(benchmark_id):
    benchmark, mode = benchmark_id.rsplit("_", 1)
    if mode == "t":
        split = "train"
    elif benchmark == "mnli":
        split = "validation_mismatched" if mode == "e2" else "validation_matched"
    else:
        split = "validation"
    return benchmark, split
