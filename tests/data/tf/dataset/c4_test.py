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

"""Tests for trax.data.tf_inputs."""
import collections

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tests.data.utils import (  # relative import
    _TESTDATA,
    _c4_dataset,
    _spm_path,
    _t5_gin_config,
)
from trax.data.loader.tf import base as ds
from trax.data.preprocessing.inputs import batcher  # noqa: F401
from trax.data.preprocessing.tf.c4 import c4_bare_preprocess_fn, c4_preprocess


class TFDatasetC4Test(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_c4_bare_preprocess_fn(self):
        dataset = _c4_dataset()

        example = list(tfds.as_numpy(dataset.take(1)))[0]

        # Targets are NOT in the example.
        self.assertNotIn("targets", example)
        self.assertIn("text", example)
        text = example["text"]

        # This should convert the dataset to an inputs/targets that are tokenized.
        dataset = c4_bare_preprocess_fn(dataset, spm_path=_spm_path())

        example = list(tfds.as_numpy(dataset.take(1)))[0]

        # Earlier text is now stored in targets_pretokenized
        self.assertIn("targets_pretokenized", example)
        self.assertEqual(example["targets_pretokenized"], text)

        # Targets are now tokenized.
        self.assertIn("targets", example)
        self.assertIsInstance(example["targets"], np.ndarray)
        self.assertEqual(example["targets"].dtype, np.int64)
        self.assertGreater(len(example["targets"]), 0)
        self.assertEqual(example["targets"][-1], 1)  # we add EOS at the end.

        self.assertIn("inputs", example)
        self.assertEqual(len(example["inputs"]), 171)

    def test_c4_preprocess(self):
        def load_c4_dataset(split="train"):
            dataset = _c4_dataset(split=split)
            return dataset.map(lambda example: (example, example["text"]))

        def examine_processed_dataset(proc_dataset):
            count = 0
            lengths = []
            for example in tfds.as_numpy(proc_dataset):
                count += 1
                ex = example[0]
                # Targets are in the example.
                self.assertIn("targets", ex)
                self.assertEqual(ex["targets"].dtype, np.int64)
                lengths.append(len(ex["targets"]))
            return count, lengths

        unfiltered_count = 0
        for example in tfds.as_numpy(load_c4_dataset()):
            unfiltered_count += 1
            # Targets are NOT in the example.
            self.assertNotIn("targets", example[0])

        proc_dataset = c4_preprocess(load_c4_dataset(), False, 2048)

        # `examine_processed_dataset` has some asserts in it.
        proc_count, char_lengths = examine_processed_dataset(proc_dataset)

        # Both the original and filtered datasets have examples.
        self.assertGreater(unfiltered_count, 0)
        self.assertGreater(proc_count, 0)

        # Because we filter out some entries on length.
        self.assertLess(proc_count, unfiltered_count)

        # Preprocess using the sentencepiece model in testdata.
        spc_proc_dataset = c4_preprocess(
            load_c4_dataset(), False, 2048, tokenization="spc", spm_path=_spm_path()
        )

        spc_proc_count, spc_lengths = examine_processed_dataset(spc_proc_dataset)

        # spc shortens the target sequence a lot, should be almost equal to
        # unfiltered
        self.assertLessEqual(proc_count, spc_proc_count)
        self.assertEqual(unfiltered_count, spc_proc_count)

        # Assert all spc_lengths are lesser than their char counterparts.
        for spc_len, char_len in zip(spc_lengths, char_lengths):
            self.assertLessEqual(spc_len, char_len)

    def test_c4(self):
        gin.bind_parameter("c4_preprocess.max_target_length", 2048)
        gin.bind_parameter("c4_preprocess.tokenization", "spc")
        gin.bind_parameter("c4_preprocess.spm_path", _spm_path())

        result = None

        try:
            # Just make sure this doesn't throw.
            result = ds.data_streams(
                "c4",
                data_dir=_TESTDATA,
                input_name="targets",
                target_name="text",
                preprocess_fn=c4_preprocess,
            )
        except Exception as e:
            self.fail(f"data_streams() raised an unexpected exception: {e}")

        self.assertIsNotNone(result, "data_streams() returned None unexpectedly")

    def test_c4_bare_preprocess_fn_denoising_objective(self):
        _t5_gin_config()

        dataset = _c4_dataset()
        dataset = c4_bare_preprocess_fn(dataset, spm_path=_spm_path())

        example = list(tfds.as_numpy(dataset.take(1)))[0]

        # Assertions now.
        self.assertIn("targets", example)
        targets = example["targets"]
        self.assertIsInstance(targets, np.ndarray)
        self.assertEqual(targets.dtype, np.int64)
        self.assertGreater(len(targets), 0)

        self.assertIn("inputs", example)
        _inputs = example["inputs"]  # pylint: disable=invalid-name
        self.assertIsInstance(_inputs, np.ndarray)
        self.assertEqual(_inputs.dtype, np.int64)
        self.assertGreater(len(_inputs), 0)

        # WHP inputs will have the bulk of the text.
        self.assertGreater(len(targets), len(_inputs))

        # WHP there will be one sentinel token in the inputs and targets.
        # We new tokenizer so there is no sentinel any more
        inputs_counter = collections.Counter(_inputs.tolist())
        targets_counter = collections.Counter(targets.tolist())
        self.assertEqual(0, inputs_counter[31999])
        self.assertEqual(0, targets_counter[31999])

        self.assertEqual(0, inputs_counter[1])
        self.assertEqual(1, targets_counter[1])

    def test_c4_pretrain(self):
        _t5_gin_config()

        gin.bind_parameter("c4_bare_preprocess_fn.spm_path", _spm_path())

        gin.bind_parameter("batcher.batch_size_per_device", 8)
        gin.bind_parameter("batcher.eval_batch_size", 8)
        gin.bind_parameter("batcher.max_eval_length", 50)
        gin.bind_parameter("batcher.buckets", ([51], [8, 1]))

        result = None

        try:
            # Just make sure this doesn't throw.
            result = ds.data_streams(
                "c4",
                data_dir=_TESTDATA,
                input_name="inputs",
                target_name="targets",
                bare_preprocess_fn=c4_bare_preprocess_fn,
            )
        except Exception as e:
            self.fail(f"data_streams() raised an unexpected exception: {e}")

        self.assertIsNotNone(result, "data_streams() returned None unexpectedly")


if __name__ == "__main__":
    tf.test.main()
