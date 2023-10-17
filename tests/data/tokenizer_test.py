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

"""Tests for trax.data..tokenizer."""
import os
import random

import six
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from trax.data import tokenizer


pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.normpath(os.path.join(pkg_dir, "../../resources/data/testdata"))


class TokenizerTest(tf.test.TestCase):
    def test_encode(self):
        self.assertListEqual(
            ["Dude", " - ", "that", "'", "s", "so", "cool", "."],
            tokenizer.encode("Dude - that's so cool."),
        )
        self.assertListEqual(
            ["Łukasz", "est", "né", "en", "1981", "."],
            tokenizer.encode("Łukasz est né en 1981."),
        )
        self.assertListEqual(
            [" ", "Spaces", "at", "the", "ends", " "],
            tokenizer.encode(" Spaces at the ends "),
        )
        self.assertListEqual(["802", ".", "11b"], tokenizer.encode("802.11b"))
        self.assertListEqual(["two", ". \n", "lines"], tokenizer.encode("two. \nlines"))

    def test_decode(self):
        self.assertEqual(
            "Dude - that's so cool.",
            tokenizer.decode(["Dude", " - ", "that", "'", "s", "so", "cool", "."]),
        )

    def test_invertibility_on_random_strings(self):
        for _ in range(1000):
            s = "".join(six.unichr(random.randint(0, 65535)) for _ in range(10))
            self.assertEqual(s, tokenizer.decode(tokenizer.encode(s)))


class TestTokenCounts(tf.test.TestCase):
    def setUp(self):
        super(TestTokenCounts, self).setUp()
        self.corpus_path = os.path.join(_TESTDATA, "corpus-*.txt")
        self.vocab_path = os.path.join(_TESTDATA, "vocab-*.txt")

    def test_corpus_token_counts_split_on_newlines(self):
        token_counts = tokenizer.corpus_token_counts(
            self.corpus_path, corpus_max_lines=0, split_on_newlines=True
        )

        expected = {
            "'": 2,
            ".": 2,
            ". ": 1,
            "... ": 1,
            "Groucho": 1,
            "Marx": 1,
            "Mitch": 1,
            "Hedberg": 1,
            "I": 3,
            "in": 2,
            "my": 2,
            "pajamas": 2,
        }
        self.assertDictContainsSubset(expected, token_counts)
        self.assertNotIn(".\n\n", token_counts)
        self.assertNotIn("\n", token_counts)

    def test_corpus_token_counts_no_split_on_newlines(self):
        token_counts = tokenizer.corpus_token_counts(
            self.corpus_path, corpus_max_lines=0, split_on_newlines=False
        )

        if ".\r\n\r\n" or "\r\n" in token_counts.keys():
            token_counts.update({"\n\n": token_counts.pop(".\r\n\r\n")})
            token_counts.update({"\n": token_counts.pop("\r\n")})

        self.assertDictContainsSubset({"\n\n": 2, "\n": 3}, token_counts)

    def test_corpus_token_counts_split_with_max_lines(self):
        token_counts = tokenizer.corpus_token_counts(
            self.corpus_path, corpus_max_lines=5, split_on_newlines=True
        )

        self.assertIn("slept", token_counts)
        self.assertNotIn("Mitch", token_counts)

    def test_corpus_token_counts_no_split_with_max_lines(self):
        token_counts = tokenizer.corpus_token_counts(
            self.corpus_path, corpus_max_lines=5, split_on_newlines=False
        )

        self.assertIn("slept", token_counts)
        self.assertNotIn("Mitch", token_counts)
        self.assertDictContainsSubset({".\n\n": 1, "\n": 2, ".\n": 1}, token_counts)

    def test_vocab_token_counts(self):
        token_counts = tokenizer.vocab_token_counts(self.vocab_path, 0)

        expected = {
            "lollipop": 8,
            "reverberated": 12,
            "kattywampus": 11,
            "balderdash": 10,
            "jiggery-pokery": 14,
        }
        self.assertDictEqual(expected, token_counts)

    def test_vocab_token_counts_with_max_lines(self):
        # vocab-1 has 2 lines, vocab-2 has 3
        token_counts = tokenizer.vocab_token_counts(self.vocab_path, 5)

        expected = {
            "lollipop": 8,
            "reverberated": 12,
            "kattywampus": 11,
            "balderdash": 10,
        }
        self.assertDictEqual(expected, token_counts)


if __name__ == "__main__":
    tf.test.main()
