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

"""Tests for trax.data.tf.datasets."""
from unittest import mock

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tests.data.utils import (  # relative import
    _TESTDATA,
    _load_dataset,
    _spm_path,
    _test_dataset_ints,
    assert_dataset,
)
from trax.data.loader.tf import base as ds
from trax.data.preprocessing import inputs
from trax.data.preprocessing.inputs import batcher  # noqa: F401


class TFDatasetTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_TFDS_single_host_with_eval_holdout(self):
        train_ds_gen = ds.TFDS(
            "c4/en:2.3.0",
            data_dir=_TESTDATA,
            train=True,
            host_id=0,
            keys=("text",),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in train_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

        valid_ds_gen = ds.TFDS(
            "c4/en:2.3.0",
            data_dir=_TESTDATA,
            train=False,
            host_id=0,
            keys=("text",),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in valid_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

    def test_TFDS_single_host_with_eval_holdout_no_valid_split(self):
        train_ds_gen = ds.TFDS(
            "para_crawl/ende",
            data_dir=_TESTDATA,
            train=True,
            host_id=0,
            keys=("en", "de"),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in train_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

        # para_crawl doesn't have a validation set, see that this still doesn't
        # crash because of eval_holdout_set.
        valid_ds_gen = ds.TFDS(
            "para_crawl/ende",
            data_dir=_TESTDATA,
            train=False,
            host_id=0,
            keys=("en", "de"),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in valid_ds_gen():
                break
            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

    def test_TFDS_mnli_split_is_eval(self):
        with mock.patch("tensorflow_datasets.load") as tfds_load:
            with mock.patch(
                "trax.data.loader.tf.base.download_and_prepare",
                lambda _, data_dir: data_dir,
            ):
                _ = ds.TFDS("glue/mnli", keys=("premise", "hypothesis"), train=False)
            call_kwargs = tfds_load.call_args[1]
            self.assertEqual(call_kwargs["split"], "validation_matched")

    def test_TFDS_mnli_split_is_alt_eval(self):
        with mock.patch("tensorflow_datasets.load") as tfds_load:
            with mock.patch(
                "trax.data.loader.tf.base.download_and_prepare",
                lambda _, data_dir: data_dir,
            ):
                _ = ds.TFDS(
                    "glue/mnli",
                    keys=("premise", "hypothesis"),
                    train=False,
                    use_alt_eval=True,
                )
            call_kwargs = tfds_load.call_args[1]
            self.assertEqual(call_kwargs["split"], "validation_mismatched")

    def test_generic_text_dataset_preprocess_fn(self):
        # self.skipTest("google.protobuf.json_format.ParseError ...")
        dataset = _load_dataset("squad/v1.1:3.0.0")

        (example,) = tfds.as_numpy(dataset.take(1))

        self.assertNotIn("inputs", example)
        self.assertNotIn("targets", example)

        proc_dataset = ds.generic_text_dataset_preprocess_fn(
            dataset,
            spm_path=_spm_path(),
            text_preprocess_fns=[lambda _ds, training: ds.squad_t5(_ds, None)],
            copy_pretokenized=True,
            debug_print_examples=True,
            debug_print_examples_rate=1.0,
        )

        (proc_example,) = tfds.as_numpy(proc_dataset.take(1))

        self.assertIn("inputs", proc_example)
        self.assertIn("targets", proc_example)

        self.assertEqual(proc_example["inputs"].dtype, tf.int64)
        self.assertEqual(proc_example["targets"].dtype, tf.int64)

    # TODO(afrozm): Why does this test take so much time?
    def test_inputs_using_generic_text_dataset_preprocess_fn(self):
        gin.bind_parameter("generic_text_dataset_preprocess_fn.spm_path", _spm_path())
        gin.bind_parameter(
            "generic_text_dataset_preprocess_fn.text_preprocess_fns",
            [lambda _ds, training: ds.squad_t5(_ds, None)],
        )

        # Just make sure this doesn't throw.
        def data_streams():
            return ds.data_streams(
                "squad",
                data_dir=_TESTDATA,
                input_name="inputs",
                target_name="targets",
                bare_preprocess_fn=ds.generic_text_dataset_preprocess_fn,
                shuffle_buffer_size=1,
            )

        n_devices = 3

        squad_inputs = inputs.batcher(
            data_streams=data_streams,
            max_eval_length=512,
            buckets=(
                [
                    513,
                ],
                [n_devices, n_devices],
            ),
        )

        eval_stream = squad_inputs.eval_stream(n_devices)
        inps, tgts, _ = next(eval_stream)

        # We can only assert that the batch dim gets divided by n_devices.
        self.assertEqual(inps.shape[0] % n_devices, 0)
        self.assertEqual(tgts.shape[0] % n_devices, 0)

    def test_filter_dataset_on_len(self):
        # {1, 2}, {2, 4}, {3, 6} ... {10, 20}
        dataset = _test_dataset_ints(range(1, 11), range(2, 21, 2))

        ds1 = ds.filter_dataset_on_len(
            dataset, True, {"inputs": [4, 8], "targets": [14, 20]}
        )
        # Only {7, 14} and {8, 16} satisfy this.
        self.assertLen(list(ds1.as_numpy_iterator()), 2)

        ds2 = ds.filter_dataset_on_len(
            dataset,
            False,
            len_map={"inputs": [4, 8], "targets": [14, 20]},
            filter_on_eval=False,
        )
        # This is eval and we aren't supposed to filter it.
        self.assertLen(list(ds2.as_numpy_iterator()), 10)

        ds3 = ds.filter_dataset_on_len(
            dataset,
            False,
            len_map={"inputs": [4, 8], "targets": [14, 20]},
            filter_on_eval=True,
        )
        # This is eval and we are asked to filter it.
        self.assertLen(list(ds3.as_numpy_iterator()), 2)

    def test_truncate_dataset_on_len(self):
        dataset = _test_dataset_ints([5, 6, 7], [8, 9, 10])
        ds1 = ds.truncate_dataset_on_len(
            dataset, True, len_map={"inputs": 6, "targets": 4}
        )
        expected_ds = _test_dataset_ints([5, 6, 6], [4, 4, 4])

        # training, should filter.
        assert_dataset(ds1, list(expected_ds.as_numpy_iterator()))

        # not Training, shouldn't filter.
        ds2 = ds.truncate_dataset_on_len(
            dataset, False, len_map={"inputs": 6, "targets": 4}
        )
        assert_dataset(ds2, list(dataset.as_numpy_iterator()))

        # not Training, but asked to filter, should filter.
        ds3 = ds.truncate_dataset_on_len(
            dataset, False, len_map={"inputs": 6, "targets": 4}, truncate_on_eval=True
        )
        assert_dataset(ds3, list(expected_ds.as_numpy_iterator()))

    def test_get_t5_preprocessor_by_name(self):
        gin.clear_config()

        gin.parse_config(
            """
            get_t5_preprocessor_by_name.name = 'rekey_t5'
            get_t5_preprocessor_by_name.fn_kwargs = {'key_map': {'inputs': 'other', 'targets': 'text'}}
            """
        )

        prep_rekey = ds.get_t5_preprocessor_by_name()
        og_dataset = tf.data.Dataset.from_tensors(
            {"text": "That is good.", "other": "That is bad."}
        )
        training = True
        dataset = prep_rekey(og_dataset, training)
        assert_dataset(dataset, {"inputs": "That is bad.", "targets": "That is good."})

    def test_pad_dataset_to_length(self):
        dataset = _test_dataset_ints([5, 6, 7], [6, 7, 8])
        ds1 = ds.pad_dataset_to_length(
            dataset, True, len_map={"inputs": 7, "targets": 10}
        )

        expected_ds = [
            {
                "inputs": np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.int64),
                "targets": np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64),
            },
            {
                "inputs": np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.int64),
                "targets": np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int64),
            },
            {
                "inputs": np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
                "targets": np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64),
            },
        ]

        assert_dataset(ds1, expected_ds)

    def test_lm_token_preprocessing(self):
        dataset = _test_dataset_ints([1, 2, 3], [3, 2, 1])
        ds1 = ds.lm_token_preprocessing(dataset, True)

        expected_ds = [
            {
                "inputs": np.array([1, 0, 1, 1, 1], dtype=np.int64),
                "targets": np.array([1, 0, 1, 1, 1], dtype=np.int64),
                "mask": np.array([0, 0, 1, 1, 1], dtype=np.int64),
            },
            {
                "inputs": np.array([1, 1, 0, 1, 1], dtype=np.int64),
                "targets": np.array([1, 1, 0, 1, 1], dtype=np.int64),
                "mask": np.array([0, 0, 0, 1, 1], dtype=np.int64),
            },
            {
                "inputs": np.array([1, 1, 1, 0, 1], dtype=np.int64),
                "targets": np.array([1, 1, 1, 0, 1], dtype=np.int64),
                "mask": np.array([0, 0, 0, 0, 1], dtype=np.int64),
            },
        ]

        assert_dataset(ds1, expected_ds)


if __name__ == "__main__":
    tf.test.main()
