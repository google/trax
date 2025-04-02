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

"""A simple invertible tokenizer.

Converts from a unicode string to a list of tokens
(represented as Unicode strings).

This tokenizer has the following desirable properties:
 - It is invertible.
 - Alphanumeric characters are broken away from non-alphanumeric characters.
 - A single space between words does not produce an extra token.
 - The full Unicode punctuation and separator set is recognized.

The tokenization algorithm is as follows:

1.  Split the text into a list of tokens, splitting at every boundary of an
    alphanumeric character and a non-alphanumeric character.  This produces
    a list which alternates between "alphanumeric tokens"
    (strings of alphanumeric characters) and "non-alphanumeric tokens"
    (strings of non-alphanumeric characters).

2.  Remove every token consisting of a single space, unless it is
    the very first or very last token in the list.  These tokens are now
    implied by the fact that there are two adjacent alphanumeric tokens.

e.g.  u"Dude - that's so cool."
        -> [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]
"""

import collections
import os
import sys
import unicodedata

import gin
import numpy as np
import six
import tensorflow as tf

from absl import logging

from trax.data.debugger import data_pipeline as debug_data_pipeline
from trax.data.encoder import encoder
from trax.data.utils.text_utils import native_to_unicode, whitespace_tokenize

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i)
    for i in range(sys.maxunicode)
    if (
        unicodedata.category(six.unichr(i)).startswith("L")
        or unicodedata.category(six.unichr(i)).startswith("N")
    )
)


def encode(text):
    """Encode a unicode string as a list of tokens.

    Args:
      text: a unicode string
    Returns:
      a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != " " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def decode(tokens):
    """Decode a list of tokens to a unicode string.

    Args:
      tokens: a list of Unicode strings
    Returns:
      a unicode string
    """
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(" ")
        ret.append(token)
    return "".join(ret)


def _read_filepattern(filepattern, max_lines=None, split_on_newlines=True):
    """Reads files matching a wildcard pattern, yielding the contents.

    Args:
      filepattern: A wildcard pattern matching one or more files.
      max_lines: If set, stop reading after reading this many lines.
      split_on_newlines: A boolean. If true, then split files by lines and strip
          leading and trailing whitespace from each line. Otherwise, treat each
          file as a single string.

    Yields:
      The contents of the files as lines, if split_on_newlines is True, or
      the entire contents of each file if False.
    """
    filenames = sorted(tf.io.gfile.glob(filepattern))
    lines_read = 0
    for filename in filenames:
        with tf.io.gfile.GFile(filename) as f:
            if split_on_newlines:
                for line in f:
                    yield line.strip()
                    lines_read += 1
                    if max_lines and lines_read >= max_lines:
                        return

            else:
                if max_lines:
                    doc = []
                    for line in f:
                        doc.append(line)
                        lines_read += 1
                        if max_lines and lines_read >= max_lines:
                            yield "".join(doc)
                            return
                    yield "".join(doc)

                else:
                    yield f.read()


def corpus_token_counts(text_filepattern, corpus_max_lines, split_on_newlines=True):
    """Read the corpus and compute a dictionary of token counts.

    Args:
      text_filepattern: A pattern matching one or more files.
      corpus_max_lines: An integer; maximum total lines to read.
      split_on_newlines: A boolean. If true, then split files by lines and strip
          leading and trailing whitespace from each line. Otherwise, treat each
          file as a single string.

    Returns:
      a dictionary mapping token to count.
    """
    counts = collections.Counter()
    for doc in _read_filepattern(
        text_filepattern,
        max_lines=corpus_max_lines,
        split_on_newlines=split_on_newlines,
    ):
        counts.update(encode(doc))

    return counts


def vocab_token_counts(text_filepattern, max_lines):
    """Read a vocab file and return a dictionary of token counts.

    Reads a two-column CSV file of tokens and their frequency in a dataset. The
    tokens are presumed to be generated by encode() or the equivalent.

    Args:
      text_filepattern: A pattern matching one or more files.
      max_lines: An integer; maximum total lines to read.

    Returns:
      a dictionary mapping token to count.
    """
    ret = {}
    for i, line in enumerate(_read_filepattern(text_filepattern, max_lines=max_lines)):
        if "," not in line:
            logging.warning("Malformed vocab line #%d '%s'", i, line)
            continue

        token, count = line.rsplit(",", 1)
        ret[token] = int(count)

    return ret


def vocab_size(vocab_type="subword", vocab_file=None, vocab_dir=None, n_reserved_ids=0):
    """Returns the size of the vocabulary (number of symbols used).

    This function can be used to set the size of the final layers of a model that
    needs to predict symbols from a given vocabulary. More precisely, if this
    function returns N then the last layer size should be set to at least N (it
    can be more). Note that this function does take reserved IDs into account.

    Args:
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused.

    Returns:
      An integer, the number of symbols used (including reserved IDs).
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    return vocab.vocab_size + n_reserved_ids


def _get_vocab(vocab_type="subword", vocab_file=None, vocab_dir=None, extra_ids=0):
    """Gets the vocabulary object for tokenization; see tokenize for details."""
    if vocab_type not in ["char", "subword", "sentencepiece", "bert", "bert-lowercase"]:
        raise ValueError(
            'vocab_type must be "subword", "char", "sentencepiece", "bert" or "bert-lowercase" '
            f"but got {vocab_type}"
        )

    if vocab_type == "char":
        # Note that we set num_reserved_ids=0 below. We could instead pass
        # the value n_reserved_ids from tokenize here -- ByteTextEncoder does
        # exactly the same thing as tokenize above, ie., adds num_reserved_ids.
        return encoder.ByteTextEncoder(num_reserved_ids=0)

    vocab_dir = vocab_dir or "gs://trax-ml/vocabs/"
    path = os.path.join(vocab_dir, vocab_file)

    if vocab_type == "subword":
        return encoder.SubwordTextEncoder(path)

    if vocab_type == "bert":
        return encoder.BertEncoder(path, do_lower_case=False)

    if vocab_type == "bert-lowercase":
        return encoder.BertEncoder(path, do_lower_case=True)

    if vocab_type == "sentencepiece":
        return encoder.SentencePieceEncoder(path, extra_ids=extra_ids)


# Tokenization.
@debug_data_pipeline.debug_pipeline
def tokenize(
    stream,
    keys=None,
    vocab_type="subword",
    vocab_file=None,
    vocab_dir=None,
    n_reserved_ids=0,
):
    """Tokenize examples from the stream.

    This function assumes that `stream` generates either strings or tuples/dicts
    containing strings at some `keys`. This function maps these strings to
    numpy arrays of integers -- the tokenized version of each string.

    Args:
      stream: A python generator yielding strings, tuples or dicts.
      keys: which keys of the tuple/dict to tokenize (by default: all)
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
        This is common for example when reserving the 0 for padding and 1 for EOS,
        but it's only needed if these symbols are not already included (and thus
        reserved) in the vocab_file.

    Yields:
      Examples from stream with strings at `keys` replaced by np.arrays of
      integers -- the tokenized version of these strings.
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    for example in stream:
        if isinstance(example, (list, tuple)):
            new_example = []
            for i, x in enumerate(example):
                if keys is None or i in keys:
                    new_example.append(np.array(vocab.encode(x)) + n_reserved_ids)
                else:
                    new_example.append(x)
            output = tuple(new_example)
            yield output
        elif isinstance(example, dict):
            new_example = {}
            for k in example:
                if keys is None or k in keys:
                    new_example[k] = np.array(vocab.encode(example[k])) + n_reserved_ids
                else:
                    new_example[k] = example[k]
            yield new_example
        else:
            output = np.array(vocab.encode(example)) + n_reserved_ids
            yield output


@gin.configurable(module="trax.data")
def Tokenize(  # pylint: disable=invalid-name
    keys=None,
    vocab_type="subword",  # pylint: disable=invalid-name
    vocab_file=None,
    vocab_dir=None,
    n_reserved_ids=0,
):
    """Returns a function that maps text to integer arrays; see `tokenize`."""
    return lambda g: tokenize(  # pylint: disable=g-long-lambda
        g,
        keys=keys,
        vocab_type=vocab_type,
        vocab_file=vocab_file,
        vocab_dir=vocab_dir,
        n_reserved_ids=n_reserved_ids,
    )


def detokenize(
    x, vocab_type="subword", vocab_file=None, vocab_dir=None, n_reserved_ids=0
):
    """Maps integer arrays to text; the opposite of `tokenize`.

    In many cases (all char- and subword-type vocabularies and most sentencepiece
    ones) the tokenization is invertible, so detokenize(tokenize(x)) = x. In some
    more rare cases this can remove some spacing, but it is still often useful
    to run detokenize to get a readable version for a tokenized string.

    Args:
      x: a list or numpy array of integers.
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
        This is common for example when reserving the 0 for padding and 1 for EOS,
        but it's only needed if these symbols are not already included (and thus
        reserved) in the vocab_file.

    Returns:
      A string corresponding to the de-tokenized version of x.
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    x_unreserved = np.array(x) - n_reserved_ids
    return str(vocab.decode(x_unreserved.tolist()))


@gin.configurable(module="trax.data")
def SentencePieceTokenizer(spm_path=None, extra_ids=0):
    """
    Returns a generator function that tokenizes a stream of text using
    SentencePiece and supports extra IDs.

    Args:
      spm_path: Path to the SentencePiece model file. Must be provided.
      extra_ids: Number of extra IDs to reserve.

    Returns:
      A function that takes a generator of text examples and yields tokenized
      numpy arrays.
    """
    if spm_path is None:
        raise ValueError("spm_path must be provided.")

    def tokenize(stream, spm_path, extra_ids):
        vocab_file = os.path.basename(spm_path)
        vocab_dir = os.path.dirname(spm_path)
        vocab = _get_vocab(
            vocab_type="sentencepiece",
            vocab_file=vocab_file,
            vocab_dir=vocab_dir,
            extra_ids=extra_ids,
        )
        for example in stream:
            # Optionally replace print with logging.debugger
            # logging.debugger("Tokenizing example: %s", example)
            if isinstance(example, tuple):
                example = example[0]
            yield np.array(vocab.encode(example), dtype=np.int64)

    return lambda g: tokenize(g, spm_path=spm_path, extra_ids=extra_ids)


class BertWordpieceTokenizer:
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = native_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
