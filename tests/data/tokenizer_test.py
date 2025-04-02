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

import gin
import numpy as np
import six
import tensorflow as tf

from six.moves import range  # pylint: disable=redefined-builtin

from tests.data.utils import (  # relative import
    _TESTDATA,
    _spm_path,
)
from trax.data.preprocessing.tokenizer import tokenizer


class TokenizerTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

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

    def test_tokenize_detokenize_character_level(self):
        def dataset():
            yield "I have a cat."

        # Character-level.
        tok_char = list(tokenizer.tokenize(dataset(), vocab_type="char"))
        self.assertAllEqual(tok_char[0], np.array([ord(c) for c in "I have a cat."]))
        detok = tokenizer.detokenize(tok_char[0], vocab_type="char")
        self.assertEqual(detok, "I have a cat.")

    def test_tokenize_detokenize_sentencepiece(self):
        def dataset():
            yield "I have a cat."

            # Sentencepiece.
            tok_spc = list(
                tokenizer.tokenize(
                    dataset(),
                    vocab_type="sentencepiece",
                    vocab_dir=_TESTDATA,
                    vocab_file="sentencepiece.model",
                )
            )

            self.assertAllEqual(tok_spc[0], np.array([[27, 43, 3, 9, 1712, 5]]))

            detok = tokenizer.detokenize(
                list(tok_spc[0]),
                vocab_type="sentencepiece",
                vocab_dir=_TESTDATA,
                vocab_file="sentencepiece.model",
            )

            self.assertEqual(detok, "I have a cat.")

    def test_tokenize_detokenize_subword(self):
        def dataset():
            yield "I have a cat."

        # Subword.
        tok_sbw = list(
            tokenizer.tokenize(
                dataset(),
                vocab_type="subword",
                vocab_dir=_TESTDATA,
                vocab_file="en_8k.subword",
            )
        )
        self.assertAllEqual(tok_sbw[0], np.array([139, 96, 12, 2217, 2, 21]))
        detok = tokenizer.detokenize(
            tok_sbw[0],
            vocab_type="subword",
            vocab_dir=_TESTDATA,
            vocab_file="en_8k.subword",
        )
        self.assertEqual(detok, "I have a cat.")

    def test_tokenize_detokenize_bert_lowercase(self):
        def dataset():
            yield "I have a cat."

        # bert-lowercase
        tok_sbw = list(
            tokenizer.tokenize(
                dataset(),
                vocab_type="bert-lowercase",
                vocab_dir=_TESTDATA,
                vocab_file="bert_uncased_vocab.txt",
            )
        )
        self.assertAllEqual(tok_sbw[0], np.array([1045, 2031, 1037, 4937, 1012]))

        detok = tokenizer.detokenize(
            tok_sbw[0],
            vocab_type="bert-lowercase",
            vocab_dir=_TESTDATA,
            vocab_file="bert_uncased_vocab.txt",
        )
        self.assertEqual(detok, "i have a cat .")
        # note: BERT tokenizer is not reversible, therefore
        # difference between original input

    def test_tokenize_keys_reservedids(self):
        def dataset():
            yield ("Cat.", "Dog.")

        tok_char1 = list(
            tokenizer.tokenize(dataset(), vocab_type="char", n_reserved_ids=5)
        )
        self.assertAllEqual(tok_char1[0][0], np.array([ord(c) + 5 for c in "Cat."]))
        self.assertAllEqual(tok_char1[0][1], np.array([ord(c) + 5 for c in "Dog."]))

        tok_char2 = list(
            tokenizer.tokenize(dataset(), keys=[0], vocab_type="char", n_reserved_ids=2)
        )
        self.assertAllEqual(tok_char2[0][0], np.array([ord(c) + 2 for c in "Cat."]))
        self.assertEqual(tok_char2[0][1], "Dog.")

    def test_tokenize_dict(self):
        def dataset():
            yield {"a": "Cat.", "b": "Dog."}

        tok_char1 = list(tokenizer.tokenize(dataset(), vocab_type="char"))
        self.assertAllEqual(tok_char1[0]["a"], np.array([ord(c) for c in "Cat."]))
        self.assertAllEqual(tok_char1[0]["b"], np.array([ord(c) for c in "Dog."]))

        tok_char2 = list(tokenizer.tokenize(dataset(), keys=["a"], vocab_type="char"))
        self.assertAllEqual(tok_char2[0]["a"], np.array([ord(c) for c in "Cat."]))
        self.assertEqual(tok_char2[0]["b"], "Dog.")

    def test_vocab_size_character_level(self):
        # Character-level.
        char_size = tokenizer.vocab_size(vocab_type="char", n_reserved_ids=11)
        self.assertEqual(char_size, 256 + 11)

    def test_vocab_size_sentencepiece(self):
        # Sentencepiece.
        spc_size = tokenizer.vocab_size(
            vocab_type="sentencepiece",
            vocab_dir=_TESTDATA,
            vocab_file="sentencepiece.model",
        )
        self.assertEqual(spc_size, 32000)

    def test_vocab_size_subword_level(self):
        sbw_size = tokenizer.vocab_size(
            vocab_type="subword",
            vocab_dir=_TESTDATA,
            vocab_file="en_8k.subword",
        )
        self.assertEqual(sbw_size, 8183)

    def test_vocab_size_bert_uncased(self):
        # Bert_uncased.
        sbw_size = tokenizer.vocab_size(
            vocab_type="bert-lowercase",
            vocab_dir=_TESTDATA,
            vocab_file="bert_uncased_vocab.txt",
        )
        self.assertEqual(sbw_size, 30522)

    def test_sentencepiece_tokenize(self):
        def dataset():
            yield "I have a cat."

        # Assume _spm_path() returns the correct path to your SentencePiece model.
        # Use the new name: SentencePieceTokenizer.
        tokenizer_fn = tokenizer.SentencePieceTokenizer(_spm_path())

        # tokenizer_fn is now a function that expects a generator (stream) of examples.
        tokenized_gen = tokenizer_fn(dataset())

        # Get the first tokenized example using next()
        first_example = next(tokenized_gen)
        # Convert to list if needed
        toks = list(first_example)
        self.assertSequenceEqual([27, 43, 3, 9, 1712, 5], toks)


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

        if ".\r\n\r\n" in token_counts.keys():
            token_counts.update({"\n\n": token_counts.pop(".\r\n\r\n")})

        if "\r\n" in token_counts.keys():
            token_counts.update({"\n": token_counts.pop("\r\n")})

        if ".\n\n" in token_counts.keys():
            token_counts.update({"\n\n": token_counts.pop(".\n\n")})

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
