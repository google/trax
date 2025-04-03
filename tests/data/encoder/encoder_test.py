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

"""Tests for trax.data.text_encoder."""

import collections
import io
import os
import random
import shutil
import string

import gin
import mock
import numpy as np
import six
import tensorflow.compat.v1 as tf


# import tensorflow.compat.v1 as tf
from six.moves import (
    range,  # pylint: disable=redefined-builtin  # pylint: disable=redefined-builtin
)

from tests.data.utils import (  # relative import
    _TESTDATA,
    _spm_path,
)
from trax.data.encoder import encoder as text_encoder


class NativeToUnicodeTest(tf.test.TestCase):
    def test_native_to_unicode(self):
        s = r"foo bar"
        s_unicode = text_encoder.native_to_unicode(s)
        self.assertEqual(s_unicode, "foo bar")


class EscapeUnescapeTokenTest(tf.test.TestCase):
    def test_escape_token(self):
        escaped = text_encoder._escape_token(
            "Foo! Bar.\nunder_score back\\slash",
            set("abcdefghijklmnopqrstuvwxyz .\n") | text_encoder._ESCAPE_CHARS,
        )

        self.assertEqual(
            "\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_", escaped
        )

    def test_unescape_token(self):
        unescaped = text_encoder._unescape_token(
            "\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_"
        )

        self.assertEqual("Foo! Bar.\nunder_score back\\slash", unescaped)


class TokenTextEncoderTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        """Make sure the test dir exists and is empty."""
        cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), "encoder_test")
        shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
        tf.gfile.MakeDirs(cls.test_temp_dir)

    def test_save_and_reload(self):
        """Test that saving and reloading doesn't change the vocab.

        Note that this test reads and writes to the filesystem, which necessitates
        that this test size be "large".
        """

        corpus = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        vocab_filename = os.path.join(self.test_temp_dir, "abc.vocab")

        # Make text encoder from a list and store vocab to fake filesystem.
        encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())
        encoder.store_to_file(vocab_filename)

        # Load back the saved vocab file from the fake_filesystem.
        new_encoder = text_encoder.TokenTextEncoder(vocab_filename)

        self.assertEqual(encoder._id_to_token, new_encoder._id_to_token)
        self.assertEqual(encoder._token_to_id, new_encoder._token_to_id)

    def test_reserved_tokens_in_corpus(self):
        """Test that we handle reserved tokens appearing in the corpus."""
        corpus = "A B {} D E F {} G {}".format(
            text_encoder.EOS, text_encoder.EOS, text_encoder.PAD
        )

        encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())

        all_tokens = encoder._id_to_token.values()

        # If reserved tokens are removed correctly, then the set of tokens will
        # be unique.
        self.assertEqual(len(all_tokens), len(set(all_tokens)))


class SubwordTextEncoderTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        """Make sure the test dir exists and is empty."""
        cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), "encoder_test")
        shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
        tf.gfile.MakeDirs(cls.test_temp_dir)

    def test_encode_decode(self):
        corpus = (
            "This is a corpus of text that provides a bunch of tokens from which "
            "to build a vocabulary. It will be used when strings are encoded "
            "with a TextEncoder subclass. The encoder was coded by a coder."
        )
        token_counts = collections.Counter(corpus.split(" "))
        alphabet = set(corpus) - {" "}

        original = "This is a coded sentence encoded by the SubwordTextEncoder."
        token_counts.update(original.split(" "))

        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )

        # Encoding should be reversible.
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        self.assertEqual(original, decoded)

        # The substrings coded and coder are frequent enough in the corpus that
        # they should appear in the vocabulary even though they are substrings
        # of other included strings.
        subtoken_strings = {encoder.all_subtoken_strings[i] for i in encoded}
        self.assertIn("encoded_", subtoken_strings)
        self.assertIn("coded_", subtoken_strings)
        self.assertIn("TextEncoder", encoder.all_subtoken_strings)
        self.assertIn("coder", encoder.all_subtoken_strings)

        # Every character in the corpus should be in the encoders alphabet and
        # its subtoken vocabulary.
        self.assertTrue(alphabet.issubset(encoder._alphabet))
        for a in alphabet:
            self.assertIn(a, encoder.all_subtoken_strings)

    def test_unicode(self):
        corpus = "Cat emoticons. \U0001F638 \U0001F639 \U0001F63A \U0001F63B"
        token_counts = collections.Counter(corpus.split(" "))

        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )

        self.assertIn("\U0001F638", encoder._alphabet)
        self.assertIn("\U0001F63B", encoder.all_subtoken_strings)

    def test_small_vocab(self):
        corpus = "The quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))
        alphabet = set(corpus) - {" "}

        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            10, token_counts, 2, 10
        )

        # All vocabulary elements are in the alphabet and subtoken strings even
        # if we requested a smaller vocabulary to assure all expected strings
        # are encodable.
        self.assertTrue(alphabet.issubset(encoder._alphabet))
        for a in alphabet:
            self.assertIn(a, encoder.all_subtoken_strings)

    def test_long_tokens(self):
        """Subword tokenization should still run efficiently with long tokens.

        To make it run efficiently, we need to use the `max_subtoken_length`
        argument when calling SubwordTextEncoder.build_to_target_size.
        """
        token_length = 4000
        num_tokens = 50
        target_vocab_size = 600
        max_subtoken_length = 10  # Set this to `None` to get problems.
        max_count = 500

        # Generate some long random strings.
        random.seed(0)
        long_tokens = []
        for _ in range(num_tokens):
            long_token = "".join(
                [random.choice(string.ascii_uppercase) for _ in range(token_length)]
            )
            long_tokens.append(long_token)

        corpus = " ".join(long_tokens)
        token_counts = collections.Counter(corpus.split(" "))
        alphabet = set(corpus) - {" "}

        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            target_vocab_size,
            token_counts,
            1,
            max_count,
            num_iterations=1,
            max_subtoken_length=max_subtoken_length,
        )

        # All vocabulary elements are in the alphabet and subtoken strings even
        # if we requested a smaller vocabulary to assure all expected strings
        # are encodable.
        self.assertTrue(alphabet.issubset(encoder._alphabet))
        for a in alphabet:
            self.assertIn(a, encoder.all_subtoken_strings)

    def test_custom_reserved_tokens(self):
        """Test that we can pass custom reserved tokens to SubwordTextEncoder."""
        corpus = "The quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))

        start_symbol = "<S>"
        end_symbol = "<E>"
        reserved_tokens = text_encoder.RESERVED_TOKENS + [start_symbol, end_symbol]
        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            10, token_counts, 2, 10, reserved_tokens=reserved_tokens
        )

        # Make sure that reserved tokens appear in the right places.
        self.assertEqual(encoder.decode([2]), start_symbol)
        self.assertEqual(encoder.decode([3]), end_symbol)

        # Make sure that we haven't messed up the ability to reconstruct.
        reconstructed_corpus = encoder.decode(encoder.encode(corpus))
        self.assertEqual(corpus, reconstructed_corpus)

    def test_encodable_when_not_in_alphabet(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))

        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )
        original = "This has UPPER CASE letters that are out of alphabet"

        # Early versions could have an infinite loop when breaking into subtokens
        # if there was any out-of-alphabet characters in the encoded string.
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        self.assertEqual(original, decoded)
        encoded_str = "".join(encoder.all_subtoken_strings[i] for i in encoded)
        self.assertIn("\\84;", encoded_str)

    @mock.patch.object(text_encoder, "_ESCAPE_CHARS", new=set("\\_;13579"))
    def test_raises_exception_when_not_encodable(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))

        # Deliberately exclude some required encoding chars from the alphabet
        # and token list, making some strings unencodable.
        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )
        original = "This has UPPER CASE letters that are out of alphabet"

        # Previously there was a bug which produced an infinite loop in this case.
        with self.assertRaises(AssertionError):
            encoder.encode(original)

    def test_load_from_file(self):
        # Test a vocab file with words not wrapped with single quotes
        encoder = text_encoder.SubwordTextEncoder()
        correct_vocab = ["the", "and", "of"]
        vocab = io.StringIO("the\n" "and\n" "of\n")
        encoder._load_from_file_object(vocab)
        self.assertAllEqual(encoder.all_subtoken_strings, correct_vocab)

        # Test a vocab file with words wrapped in single quotes
        encoder = text_encoder.SubwordTextEncoder()
        vocab = io.StringIO('"the"\n' '"and"\n' '"of"\n')
        encoder._load_from_file_object(vocab)
        self.assertAllEqual(encoder.all_subtoken_strings, correct_vocab)

    def test_reserved_token_chars_not_in_alphabet(self):
        corpus = "dog"
        token_counts = collections.Counter(corpus.split(" "))
        encoder1 = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 100
        )
        filename = os.path.join(self.test_temp_dir, "out.voc")
        encoder1.store_to_file(filename)
        encoder2 = text_encoder.SubwordTextEncoder(filename=filename)

        self.assertEqual(encoder1._alphabet, encoder2._alphabet)

        for t in text_encoder.RESERVED_TOKENS:
            for c in t:
                # Verify that encoders can encode all reserved token chars.
                encoder1.encode(c)
                encoder2.encode(c)

    def test_save_and_reload(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))

        # Deliberately exclude some required encoding chars from the alphabet
        # and token list, making some strings unencodable.
        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )

        filename = os.path.join(self.test_temp_dir, "out.voc")
        encoder.store_to_file(filename)
        new_encoder = text_encoder.SubwordTextEncoder(filename)

        self.assertEqual(encoder._alphabet, new_encoder._alphabet)
        self.assertEqual(encoder.all_subtoken_strings, new_encoder.all_subtoken_strings)
        self.assertEqual(
            encoder._subtoken_string_to_id, new_encoder._subtoken_string_to_id
        )
        self.assertEqual(encoder._max_subtoken_len, new_encoder._max_subtoken_len)

    def test_save_and_reload_no_single_quotes(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        token_counts = collections.Counter(corpus.split(" "))

        # Deliberately exclude some required encoding chars from the alphabet
        # and token list, making some strings unencodable.
        encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
            100, token_counts, 2, 10
        )

        filename = os.path.join(self.test_temp_dir, "out.voc")
        encoder.store_to_file(filename, add_single_quotes=False)
        new_encoder = text_encoder.SubwordTextEncoder(filename)

        self.assertEqual(encoder._alphabet, new_encoder._alphabet)
        self.assertEqual(encoder.all_subtoken_strings, new_encoder.all_subtoken_strings)
        self.assertEqual(
            encoder._subtoken_string_to_id, new_encoder._subtoken_string_to_id
        )
        self.assertEqual(encoder._max_subtoken_len, new_encoder._max_subtoken_len)

    def test_build_from_generator(self):
        corpus = "The quick brown fox jumps over the lazy dog"

        def gen():
            for _ in range(3):
                yield corpus

        start_symbol = "<S>"
        end_symbol = "<E>"
        reserved_tokens = text_encoder.RESERVED_TOKENS + [start_symbol, end_symbol]
        encoder = text_encoder.SubwordTextEncoder.build_from_generator(
            gen(), 10, reserved_tokens=reserved_tokens
        )

        # Make sure that reserved tokens appear in the right places.
        self.assertEqual(encoder.decode([2]), start_symbol)
        self.assertEqual(encoder.decode([3]), end_symbol)

        self.assertEqual(
            "hi%s" % start_symbol, encoder.decode(encoder.encode("hi") + [2])
        )

        # Make sure that we haven't messed up the ability to reconstruct.
        reconstructed_corpus = encoder.decode(encoder.encode(corpus))
        self.assertEqual(corpus, reconstructed_corpus)


class OneHotClassLabelEncoderTest(tf.test.TestCase):
    def test_one_hot_encode(self):
        encoder = text_encoder.OneHotClassLabelEncoder(
            class_labels=["zero", "one", "two"]
        )
        self.assertEqual(encoder.encode("zero"), [1, 0, 0])
        self.assertEqual(encoder.encode("one"), [0, 1, 0])
        self.assertEqual(encoder.encode("two"), [0, 0, 1])

    def test_one_hot_decode(self):
        encoder = text_encoder.OneHotClassLabelEncoder(
            class_labels=["zero", "one", "two"]
        )
        self.assertEqual(encoder.decode([1, 0, 0]), "zero")
        self.assertEqual(encoder.decode([0, 1, 0]), "one")
        self.assertEqual(encoder.decode([0, 0, 1]), "two")


class TokenizerTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_encode(self):
        self.assertListEqual(
            ["Dude", " - ", "that", "'", "s", "so", "cool", "."],
            text_encoder.encode("Dude - that's so cool."),
        )
        self.assertListEqual(
            ["Łukasz", "est", "né", "en", "1981", "."],
            text_encoder.encode("Łukasz est né en 1981."),
        )
        self.assertListEqual(
            [" ", "Spaces", "at", "the", "ends", " "],
            text_encoder.encode(" Spaces at the ends "),
        )
        self.assertListEqual(["802", ".", "11b"], text_encoder.encode("802.11b"))
        self.assertListEqual(
            ["two", ". \n", "lines"], text_encoder.encode("two. \nlines")
        )

    def test_decode(self):
        self.assertEqual(
            "Dude - that's so cool.",
            text_encoder.decode(["Dude", " - ", "that", "'", "s", "so", "cool", "."]),
        )

    def test_invertibility_on_random_strings(self):
        for _ in range(1000):
            s = "".join(six.unichr(random.randint(0, 65535)) for _ in range(10))
            self.assertEqual(s, text_encoder.decode(text_encoder.encode(s)))

    def test_tokenize_detokenize_character_level(self):
        def dataset():
            yield "I have a cat."

        # Character-level.
        tok_char = list(text_encoder.tokenize(dataset(), vocab_type="char"))
        self.assertAllEqual(tok_char[0], np.array([ord(c) for c in "I have a cat."]))
        detok = text_encoder.detokenize(tok_char[0], vocab_type="char")
        self.assertEqual(detok, "I have a cat.")

    def test_tokenize_detokenize_sentencepiece(self):
        def dataset():
            yield "I have a cat."

            # Sentencepiece.
            tok_spc = list(
                text_encoder.tokenize(
                    dataset(),
                    vocab_type="sentencepiece",
                    vocab_dir=_TESTDATA,
                    vocab_file="sentencepiece.model",
                )
            )

            self.assertAllEqual(tok_spc[0], np.array([[27, 43, 3, 9, 1712, 5]]))

            detok = text_encoder.detokenize(
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
            text_encoder.tokenize(
                dataset(),
                vocab_type="subword",
                vocab_dir=_TESTDATA,
                vocab_file="en_8k.subword",
            )
        )
        self.assertAllEqual(tok_sbw[0], np.array([139, 96, 12, 2217, 2, 21]))
        detok = text_encoder.detokenize(
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
            text_encoder.tokenize(
                dataset(),
                vocab_type="bert-lowercase",
                vocab_dir=_TESTDATA,
                vocab_file="bert_uncased_vocab.txt",
            )
        )
        self.assertAllEqual(tok_sbw[0], np.array([1045, 2031, 1037, 4937, 1012]))

        detok = text_encoder.detokenize(
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
            text_encoder.tokenize(dataset(), vocab_type="char", n_reserved_ids=5)
        )
        self.assertAllEqual(tok_char1[0][0], np.array([ord(c) + 5 for c in "Cat."]))
        self.assertAllEqual(tok_char1[0][1], np.array([ord(c) + 5 for c in "Dog."]))

        tok_char2 = list(
            text_encoder.tokenize(
                dataset(), keys=[0], vocab_type="char", n_reserved_ids=2
            )
        )
        self.assertAllEqual(tok_char2[0][0], np.array([ord(c) + 2 for c in "Cat."]))
        self.assertEqual(tok_char2[0][1], "Dog.")

    def test_tokenize_dict(self):
        def dataset():
            yield {"a": "Cat.", "b": "Dog."}

        tok_char1 = list(text_encoder.tokenize(dataset(), vocab_type="char"))
        self.assertAllEqual(tok_char1[0]["a"], np.array([ord(c) for c in "Cat."]))
        self.assertAllEqual(tok_char1[0]["b"], np.array([ord(c) for c in "Dog."]))

        tok_char2 = list(
            text_encoder.tokenize(dataset(), keys=["a"], vocab_type="char")
        )
        self.assertAllEqual(tok_char2[0]["a"], np.array([ord(c) for c in "Cat."]))
        self.assertEqual(tok_char2[0]["b"], "Dog.")

    def test_vocab_size_character_level(self):
        # Character-level.
        char_size = text_encoder.vocab_size(vocab_type="char", n_reserved_ids=11)
        self.assertEqual(char_size, 256 + 11)

    def test_vocab_size_sentencepiece(self):
        # Sentencepiece.
        spc_size = text_encoder.vocab_size(
            vocab_type="sentencepiece",
            vocab_dir=_TESTDATA,
            vocab_file="sentencepiece.model",
        )
        self.assertEqual(spc_size, 32000)

    def test_vocab_size_subword_level(self):
        sbw_size = text_encoder.vocab_size(
            vocab_type="subword",
            vocab_dir=_TESTDATA,
            vocab_file="en_8k.subword",
        )
        self.assertEqual(sbw_size, 8183)

    def test_vocab_size_bert_uncased(self):
        # Bert_uncased.
        sbw_size = text_encoder.vocab_size(
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
        tokenizer_fn = text_encoder.SentencePieceTokenizer(_spm_path())

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
        token_counts = text_encoder.corpus_token_counts(
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
        token_counts = text_encoder.corpus_token_counts(
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
        token_counts = text_encoder.corpus_token_counts(
            self.corpus_path, corpus_max_lines=5, split_on_newlines=True
        )

        self.assertIn("slept", token_counts)
        self.assertNotIn("Mitch", token_counts)

    def test_corpus_token_counts_no_split_with_max_lines(self):
        token_counts = text_encoder.corpus_token_counts(
            self.corpus_path, corpus_max_lines=5, split_on_newlines=False
        )

        self.assertIn("slept", token_counts)
        self.assertNotIn("Mitch", token_counts)
        self.assertDictContainsSubset({".\n\n": 1, "\n": 2, ".\n": 1}, token_counts)

    def test_vocab_token_counts(self):
        token_counts = text_encoder.vocab_token_counts(self.vocab_path, 0)

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
        token_counts = text_encoder.vocab_token_counts(self.vocab_path, 5)

        expected = {
            "lollipop": 8,
            "reverberated": 12,
            "kattywampus": 11,
            "balderdash": 10,
        }
        self.assertDictEqual(expected, token_counts)

if __name__ == "__main__":
    tf.test.main()
