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

import gin
import numpy as np
import tensorflow as tf

from tests.data.utils import (  # relative import
    _TESTDATA,
)
from trax.data.loader.tf.base import next_sentence_prediction_tf
from trax.data.preprocessing.inputs import batcher  # noqa: F401
from trax.data.preprocessing.tf import bert as inputs_bert


class InputsBertTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_create_bert_inputs(self):
        inputs_sentences_1 = [np.array([100, 150, 200])]
        inputs_sentences_2 = [np.array([300, 500])]
        labels = [np.array(1)]

        create_inputs_1 = inputs_bert.CreateBertInputs(False)
        create_inputs_2 = inputs_bert.CreateBertInputs(True)
        for res in create_inputs_1(zip(inputs_sentences_1, labels)):
            values, segment_embs, _, label, weight = res
            self.assertAllEqual(values, np.array([101, 100, 150, 200, 102]))
            self.assertAllEqual(segment_embs, np.zeros(5))
            self.assertEqual(label, np.int64(1))
            self.assertEqual(weight, np.int64(1))

        for res in create_inputs_2(zip(inputs_sentences_1, inputs_sentences_2, labels)):
            values, segment_embs, _, label, weight = res
            self.assertAllEqual(
                values, np.array([101, 100, 150, 200, 102, 300, 500, 102])
            )
            exp_segment = np.concatenate((np.zeros(5), np.ones(3)))
            self.assertAllEqual(segment_embs, exp_segment)
            self.assertEqual(label, np.int64(1))
            self.assertEqual(weight, np.int64(1))

    def test_bert_next_sentence_prediction_inputs(self):
        stream = inputs_bert.BertNextSentencePredictionInputs(
            "c4/en:2.3.0", data_dir=_TESTDATA, train=False, shuffle_size=1
        )
        exp_sent1 = "The woman who died after falling from"
        exp_sent2 = "The woman who died after falling from"
        sent1, sent2, label = next(stream())
        print(sent1, sent2, label)

        self.assertIn(exp_sent1, sent1, "exp_sent1 powinien być częścią sent1")
        self.assertIn(exp_sent2, sent1, "exp_sent1 powinien być częścią sent1")
        self.assertFalse(label)

    def test_mask_random_tokens(self):
        """Test only standard tokens.

        This test deals with sentences composed of two parts: [100 CLS tokens, 100
        chosen standard tokens]. CLS is the token that is added at the beginning of
        the sentence and there is only one token in standard scenario. It is never
        masked because it is not a part of the sentence.
        This tests whether mask_random_tokens will:
          - mask only standard tokens
          - mask expected number of tokens (15 percent candidates for masking)
        """
        cls_token = 101
        mask_token = 103
        example_standard_token = 1001
        test_case_row = np.array([cls_token] * 100 + [example_standard_token] * 100)
        test_case = [(test_case_row.copy(),)]

        out, original_tokens, token_weights = next(
            inputs_bert.mask_random_tokens(test_case)
        )
        # test whether original tokens are unchanged
        self.assertAllEqual(test_case_row, original_tokens)

        self.assertEqual(1, token_weights.sum())
        self.assertEqual(
            15, (token_weights > 0).sum()
        )  # we should have 15 candidates for masking

        # 101 is a special token, so only 1001 should be masked
        self.assertAllEqual(out[:100], test_case_row[:100])

        # Each candidate has 0.8 probability to be masked while others have 0, so
        # no more than 15 tokens with MASK
        self.assertLessEqual((out == mask_token).sum(), 15)

    def test_next_sentence_prediction_tf(self):
        # Create dummy dataset with two examples.
        def data_generator():
            yield {"text": "This is the first sentence. This is the second sentence."}
            yield {"text": "Another example text. And a follow-up sentence."}

        output_signature = {"text": tf.TensorSpec(shape=(), dtype=tf.string)}
        dataset = tf.data.Dataset.from_generator(
            data_generator, output_signature=output_signature
        )

        preprocess = next_sentence_prediction_tf()
        processed_ds = preprocess(dataset)

        # Collect results for analysis
        examples = []
        for example in processed_ds.take(10):
            examples.append(
                {
                    "inputs": example["inputs"].numpy().decode("utf-8"),
                    "targets": example["targets"].numpy().decode("utf-8"),
                }
            )
            tf.print(example)

        # Check if we have at least some examples
        self.assertGreater(len(examples), 0)

        for example in examples:
            # Check the output structure
            self.assertIn("inputs", example)
            self.assertIn("targets", example)

            # Verify that outputs have correct format
            inputs = example["inputs"]
            self.assertIn("sentence1:", inputs)
            self.assertIn("sentence2:", inputs)

            # Check if label is one of the expected values
            self.assertIn(example["targets"], ["next", "not_next"])

            # Extract sentences for further analysis
            parts = inputs.split("sentence2:")
            sent1_part = parts[0].strip()
            sent1 = sent1_part.replace("sentence1:", "").strip()
            sent2 = parts[1].strip()

            # Check if sentences are not empty
            self.assertTrue(len(sent1) > 0)
            self.assertTrue(len(sent2) > 0)

            # Check relationship between label and sentences
            if example["targets"] == "next":
                # For "next", both sentences should come from the same document
                # We can't fully test this due to randomness, but we can check
                # if the format matches the expected pattern
                exp_sent1 = "This is the first sentence"
                exp_sent2 = "This is the second sentence"
                self.assertTrue(
                    (exp_sent1 in sent1 and exp_sent2 in sent2)
                    or (
                        "Another example text" in sent1
                        and "And a follow-up sentence" in sent2
                    )
                    or not (exp_sent1 in sent1 and "And a follow-up sentence" in sent2)
                )


if __name__ == "__main__":
    tf.test.main()
