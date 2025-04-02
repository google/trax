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
import numpy as np

from trax.data.loader.tf.base import TFDS, next_sentence_prediction_tf


@gin.configurable(module="trax.data")
def BertNextSentencePredictionInputs(
    dataset_name,  # pylint: disable=invalid-name
    data_dir=None,
    text_key="text",
    train=True,
    shuffle_size=50000,
):
    """Defines a stream for the next sentence prediction task."""
    stream = TFDS(
        dataset_name,
        data_dir=data_dir,
        tfds_preprocess_fn=next_sentence_prediction_tf(
            text_key=text_key,
            label_sentences=True,
            buffer_size=shuffle_size,
        ),
        keys=["inputs", "targets"],
        train=train,
    )

    def split_stream(generator=None):
        # split string with 'sentence1:' and 'sentence2:' into two separate strings
        for inputs, targets in stream(generator):
            # Extract inputs and targets from the dictionary

            text_str = str(inputs)[:-1]  # removes last '"' which is always at the end
            print(text_str)
            sentences = text_str.split("sentence1: ")[1].split(" sentence2: ")
            if len(sentences) != 2:
                # 'sentence2:' appeared in the text and got mixed up with the label
                continue
            sent1, sent2 = sentences
            yield sent1, sent2, targets == "next"

    return split_stream


def BertSingleSentenceInputs(
    batch,
    labeled=True,
    cls_id=101,
    sep_id=102,  # pylint: disable=invalid-name
):
    """Prepares inputs for BERT: add [SEP], [CLS] and create embeddings."""
    if labeled:
        for sent1, label in batch:
            value_vector = np.concatenate(([cls_id], sent1, [sep_id]))
            segment_embs = np.zeros(sent1.shape[0] + 2, dtype=np.int32)
            yield value_vector, segment_embs, segment_embs, label, np.int32(1)
    else:
        for (sent1,) in batch:  # row is a tuple with 1 element
            value_vector = np.concatenate(([cls_id], sent1, [sep_id]))
            segment_embs = np.zeros(sent1.shape[0] + 2, dtype=np.int32)
            yield value_vector, segment_embs, segment_embs


def BertDoubleSentenceInputs(
    batch, labeled=True, cls_id=101, sep_id=102  # pylint: disable=invalid-name
):
    """Prepares inputs for BERT models by adding [SEP] and [CLS] tokens and creating segment embeddings."""
    if labeled:
        for sent1, sent2, label in batch:
            value_vector = np.concatenate(([cls_id], sent1, [sep_id], sent2, [sep_id]))

            segment_embs = np.zeros(sent1.shape[0] + sent2.shape[0] + 3, dtype=np.int32)
            second_sent_start = sent1.shape[0] + 2
            segment_embs[second_sent_start:] = 1
            yield value_vector, segment_embs, segment_embs, label, np.int32(1)
    else:
        for sent1, sent2 in batch:
            value_vector = np.concatenate(([cls_id], sent1, [sep_id], sent2, [sep_id]))

            segment_embs = np.zeros(sent1.shape[0] + sent2.shape[0] + 3, dtype=np.int32)
            second_sent_start = sent1.shape[0] + 2
            segment_embs[second_sent_start:] = 1
            yield value_vector, segment_embs, segment_embs


@gin.configurable(module="trax.data")
def CreateBertInputs(
    double_sentence=True,  # pylint: disable=invalid-name
    labeled=True,
    cls_id=101,
    sep_id=102,
):
    bert_inputs_fn = (
        BertDoubleSentenceInputs if double_sentence else BertSingleSentenceInputs
    )
    return functools.partial(
        bert_inputs_fn, labeled=labeled, cls_id=cls_id, sep_id=sep_id
    )


@gin.configurable(module="trax.data")
def mask_random_tokens(
    batch,
    explicit_vocab_size=30522,
    masking_prob=0.15,
    cls_id=101,
    sep_id=102,
    mask_id=103,
    vocab_start_id=999,
):
    """Prepares input for the masking task.

    Preparation consist in masking masking_prob percentage of non-special tokens
    at each input row; round(masking_prob * num_nonspecial_tokens) random tokens
    are selected out of which each token is either
    - replaced with [MASK] token with 80% probability,
    - replaced with random token with 10% probability,
    - or unchanged with 10%.
    The implentation is based on
    https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L342

    Examples:
    - batch is a stream with each row having tuple (token_ids,). Function yields
    rows of form (modified_token_ids, original_tokens, token_weights), where
    modified_token_ids have [MASK] tokens or random tokens according to the
    procedure described above.
    - batch is a stream with each row having tuple (token_ids, segment_embeddings,
    nsp_label, nsp_weight).Function yields rows of form (modified_token_ids,
    segment_embeddings, nsp_label, nsp_weight, original_tokens, token_weights).

    Args:
      batch: stream of inputs. Each row in the stream is a tuple which first
        element is an array of tokens
      explicit_vocab_size: the total size of the vocabulary.
      masking_prob: Determines percent of non-special tokens to be selected for
        masking.
      cls_id: id of the special CLS token.
      sep_id: id of the special SEP token.
      mask_id: id of the special MASK token.
      vocab_start_id: id of first non-special token in the vocabulary.

    Yields:
      a stream with tokens masked for MLM training and 2 appended arrays:
        - original tokens: a copy of original tokens used as a label for mlm
        training
        - token_weights: weights distributed uniformly over selected tokens (sum
        is 1). Other tokens have 0 weight.
    """
    for token_ids, *row_rest in batch:
        original_tokens = token_ids.copy()

        # choose tokens for prediction. Chooses 0.15 of
        # all non-special tokens
        is_special_token = np.logical_or(
            token_ids == cls_id, token_ids == sep_id
        )  # CLS and SEP tokens
        is_special_token = np.logical_or(is_special_token, token_ids == 0)  # padding
        viable_ids = np.arange(token_ids.shape[0])[~is_special_token]
        num_to_sample = round(masking_prob * viable_ids.shape[0])
        if num_to_sample == 0:
            # sentence is too short to select given percentage of tokens to mask
            continue
        candidate_ids = np.random.choice(viable_ids, num_to_sample, replace=False)

        # create weights
        token_weights = np.zeros(token_ids.shape)
        token_weights[candidate_ids] = 1 / candidate_ids.shape[0]

        prob_scores = np.random.random(candidate_ids.shape)

        # change 80 % of tokens to [MASK]
        mask_token_ids = candidate_ids[prob_scores < 0.8]
        token_ids[mask_token_ids] = mask_id

        # change 10% of tokens to random token
        random_token_ids = candidate_ids[(0.8 <= prob_scores) & (prob_scores < 0.9)]
        token_ids[random_token_ids] = np.random.randint(
            vocab_start_id, explicit_vocab_size, random_token_ids.shape[0]
        )

        # rest (10%) is left unchaged
        yield (token_ids, *row_rest, original_tokens, token_weights)
