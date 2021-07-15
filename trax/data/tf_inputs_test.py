# coding=utf-8
# Copyright 2021 The Trax Authors.
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

# Lint as: python3
"""Tests for trax.data.tf_inputs."""

import collections
import os
from unittest import mock

import gin
import numpy as np
from t5.data import assert_dataset
from t5.data import preprocessors as t5_processors
import tensorflow as tf
import tensorflow_datasets as tfds
from trax.data import inputs  # pylint: disable=unused-import
from trax.data import tf_inputs

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, 'testdata')


def _test_dataset_ints(inp_lengths, tgt_lengths):
  """Create a test dataset of int64 tensors of given shapes."""

  def generator():
    for inp_len, tgt_len in zip(inp_lengths, tgt_lengths):
      inp = np.ones([inp_len], dtype=np.int64)
      tgt = np.ones([tgt_len], dtype=np.int64)
      yield {'inputs': inp, 'targets': tgt}

  types = {'inputs': tf.int64, 'targets': tf.int64}
  shapes = {'inputs': tf.TensorShape([None]), 'targets': tf.TensorShape([None])}
  return tf.data.Dataset.from_generator(
      generator, output_types=types, output_shapes=shapes)


def _load_dataset(name, split='train'):
  return tfds.load(
      name=name, split=split, data_dir=_TESTDATA, shuffle_files=False)


def _c4_dataset(split='train'):
  return _load_dataset('c4:2.3.0', split=split)


def _spm_path():
  return os.path.join(_TESTDATA, 'sentencepiece.model')


def _t5_gin_config():
  # The following pages worth of gin configuration are required because a lot
  # of T5 functions have `gin.REQUIRED` in code, i.e. you cannot use these
  # functions at all without having configured gin.

  noise_density = 0.15
  max_input_length = 50

  # What preprocessors to apply - we select a random chunk of the document if
  # it exceeds a certain lengths (`select_random_chunk`), then split up long
  # examples (`split_tokens`) and finally the denoising objective (`denoise`).
  #
  # In addition to this T5 concates multiple documents together to reduce
  # padding (`reduce_concat_tokens`) after `select_random_chunk`, but we skip
  # that since we don't do sequence packing.
  gin.bind_parameter('unsupervised.preprocessors', [
      t5_processors.select_random_chunk,
      t5_processors.split_tokens,
      t5_processors.denoise,
  ])

  # select_random_chunk
  gin.bind_parameter('select_random_chunk.feature_key', 'targets')
  gin.bind_parameter('select_random_chunk.max_length', max_input_length)

  # reduce_concat_tokens
  gin.bind_parameter('random_spans_helper.extra_tokens_per_span_inputs', 1)
  gin.bind_parameter('random_spans_helper.extra_tokens_per_span_targets', 1)
  gin.bind_parameter('random_spans_helper.inputs_length', max_input_length)
  gin.bind_parameter('random_spans_helper.mean_noise_span_length', 3.0)
  gin.bind_parameter('random_spans_helper.noise_density', noise_density)

  # split_tokens
  gin.bind_parameter('split_tokens.max_tokens_per_segment',
                     t5_processors.random_spans_tokens_length())

  # denoise
  gin.bind_parameter('denoise.inputs_fn',
                     t5_processors.noise_span_to_unique_sentinel)
  gin.bind_parameter('denoise.noise_density', noise_density)
  gin.bind_parameter('denoise.noise_mask_fn',
                     t5_processors.random_spans_noise_mask)
  gin.bind_parameter('denoise.targets_fn',
                     t5_processors.nonnoise_span_to_unique_sentinel)


class TFInputsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()


  def test_TFDS_single_host_with_eval_holdout(self):
    train_ds_gen = tf_inputs.TFDS(
        'c4/en:2.3.0',
        data_dir=_TESTDATA,
        train=True,
        host_id=0,
        keys=('text',),
        n_hosts=1,
        eval_holdout_size=0.1)

    # Just ensure that this doesn't crash.
    for d in train_ds_gen():
      print(f'Train: {d}')
      break

    valid_ds_gen = tf_inputs.TFDS(
        'c4/en:2.3.0',
        data_dir=_TESTDATA,
        train=False,
        host_id=0,
        keys=('text',),
        n_hosts=1,
        eval_holdout_size=0.1)

    # Just ensure that this doesn't crash.
    for d in valid_ds_gen():
      print(f'Eval: {d}')
      break

  def test_TFDS_single_host_with_eval_holdout_no_valid_split(self):
    train_ds_gen = tf_inputs.TFDS(
        'para_crawl/ende',
        data_dir=_TESTDATA,
        train=True,
        host_id=0,
        keys=('en', 'de'),
        n_hosts=1,
        eval_holdout_size=0.1)

    # Just ensure that this doesn't crash.
    for d in train_ds_gen():
      print(f'Train: {d}')
      break

    # para_crawl doesn't have a validation set, see that this still doesn't
    # crash because of eval_holdout_set.
    valid_ds_gen = tf_inputs.TFDS(
        'para_crawl/ende',
        data_dir=_TESTDATA,
        train=False,
        host_id=0,
        keys=('en', 'de'),
        n_hosts=1,
        eval_holdout_size=0.1)

    # Just ensure that this doesn't crash.
    for d in valid_ds_gen():
      print(f'Eval: {d}')
      break

  def test_TFDS_mnli_split_is_eval(self):
    with mock.patch('tensorflow_datasets.load') as tfds_load:
      with mock.patch('trax.data.tf_inputs.download_and_prepare',
                      lambda _, data_dir: data_dir):
        _ = tf_inputs.TFDS('glue/mnli',
                           keys=('premise', 'hypothesis'),
                           train=False)
      call_kwargs = tfds_load.call_args[1]
      self.assertEqual(call_kwargs['split'], 'validation_matched')

  def test_TFDS_mnli_split_is_alt_eval(self):
    with mock.patch('tensorflow_datasets.load') as tfds_load:
      with mock.patch('trax.data.tf_inputs.download_and_prepare',
                      lambda _, data_dir: data_dir):
        _ = tf_inputs.TFDS('glue/mnli',
                           keys=('premise', 'hypothesis'),
                           train=False,
                           use_alt_eval=True)
      call_kwargs = tfds_load.call_args[1]
      self.assertEqual(call_kwargs['split'], 'validation_mismatched')

  def test_convert_to_unicode(self):

    def dataset1():
      yield (b'Audentes fortuna iuvat.', b'Fortune favors the bold.')

    def dataset2():
      yield (b'\x81aabb', b'Value')

    convert_function1 = tf_inputs.ConvertToUnicode(keys=[0])
    convert_output1 = next(convert_function1(dataset1()))
    self.assertEqual(convert_output1[0], 'Audentes fortuna iuvat.')
    self.assertEqual(convert_output1[1], b'Fortune favors the bold.')
    self.assertIsInstance(convert_output1[0], str)
    self.assertIsInstance(convert_output1[1], bytes)

    # Contains an invalid bytes array from the point of view of UTF-8.
    try:
      convert_function2 = tf_inputs.ConvertToUnicode(keys=[0])
      convert_output2 = next(convert_function2(dataset2()))
    except UnicodeDecodeError:
      self.fail('ConvertToUnicode threw UnicodeDecodeError.')
    self.assertEqual(convert_output2[0], 'aabb')
    self.assertIsInstance(convert_output2[0], str)

  def test_tokenize_detokenize(self):

    def dataset():
      yield 'I have a cat.'

    # Character-level.
    tok_char = list(tf_inputs.tokenize(dataset(), vocab_type='char'))
    self.assertAllEqual(tok_char[0],
                        np.array([ord(c) for c in 'I have a cat.']))
    detok = tf_inputs.detokenize(tok_char[0], vocab_type='char')
    self.assertEqual(detok, 'I have a cat.')

    # Sentencepiece.
    tok_spc = list(
        tf_inputs.tokenize(
            dataset(),
            vocab_type='sentencepiece',
            vocab_dir=_TESTDATA,
            vocab_file='sentencepiece.model'))
    self.assertAllEqual(tok_spc[0], np.array([27, 43, 3, 9, 1712, 5]))
    detok = tf_inputs.detokenize(
        list(tok_spc[0]),
        vocab_type='sentencepiece',
        vocab_dir=_TESTDATA,
        vocab_file='sentencepiece.model')
    self.assertEqual(detok, 'I have a cat.')

    # Subword.
    tok_sbw = list(
        tf_inputs.tokenize(
            dataset(),
            vocab_type='subword',
            vocab_dir=_TESTDATA,
            vocab_file='en_8k.subword'))
    self.assertAllEqual(tok_sbw[0], np.array([139, 96, 12, 2217, 2, 21]))
    detok = tf_inputs.detokenize(
        tok_sbw[0],
        vocab_type='subword',
        vocab_dir=_TESTDATA,
        vocab_file='en_8k.subword')
    self.assertEqual(detok, 'I have a cat.')

    # bert-lowercase
    tok_sbw = list(
        tf_inputs.tokenize(
            dataset(),
            vocab_type='bert-lowercase',
            vocab_dir=_TESTDATA,
            vocab_file='bert_uncased_vocab.txt'))
    self.assertAllEqual(tok_sbw[0], np.array([1045, 2031, 1037, 4937, 1012]))
    detok = tf_inputs.detokenize(
        tok_sbw[0],
        vocab_type='bert-lowercase',
        vocab_dir=_TESTDATA,
        vocab_file='bert_uncased_vocab.txt')
    self.assertEqual(detok, 'i have a cat .')
    # note: BERT tokenizer is not reversible, therefore
    # difference between original input

  def test_tokenize_keys_reservedids(self):

    def dataset():
      yield ('Cat.', 'Dog.')

    tok_char1 = list(
        tf_inputs.tokenize(dataset(), vocab_type='char', n_reserved_ids=5))
    self.assertAllEqual(tok_char1[0][0], np.array([ord(c) + 5 for c in 'Cat.']))
    self.assertAllEqual(tok_char1[0][1], np.array([ord(c) + 5 for c in 'Dog.']))

    tok_char2 = list(
        tf_inputs.tokenize(
            dataset(), keys=[0], vocab_type='char', n_reserved_ids=2))
    self.assertAllEqual(tok_char2[0][0], np.array([ord(c) + 2 for c in 'Cat.']))
    self.assertEqual(tok_char2[0][1], 'Dog.')

  def test_tokenize_dict(self):

    def dataset():
      yield {'a': 'Cat.', 'b': 'Dog.'}

    tok_char1 = list(tf_inputs.tokenize(dataset(), vocab_type='char'))
    self.assertAllEqual(tok_char1[0]['a'], np.array([ord(c) for c in 'Cat.']))
    self.assertAllEqual(tok_char1[0]['b'], np.array([ord(c) for c in 'Dog.']))

    tok_char2 = list(
        tf_inputs.tokenize(dataset(), keys=['a'], vocab_type='char'))
    self.assertAllEqual(tok_char2[0]['a'], np.array([ord(c) for c in 'Cat.']))
    self.assertEqual(tok_char2[0]['b'], 'Dog.')

  def test_vocab_size(self):
    # Character-level.
    char_size = tf_inputs.vocab_size(vocab_type='char', n_reserved_ids=11)
    self.assertEqual(char_size, 256 + 11)
    # Sentencepiece.
    spc_size = tf_inputs.vocab_size(
        vocab_type='sentencepiece',
        vocab_dir=_TESTDATA,
        vocab_file='sentencepiece.model')
    self.assertEqual(spc_size, 32000)
    # Subword.
    sbw_size = tf_inputs.vocab_size(
        vocab_type='subword', vocab_dir=_TESTDATA, vocab_file='en_8k.subword')
    self.assertEqual(sbw_size, 8183)
    # Bert_uncased.
    sbw_size = tf_inputs.vocab_size(
        vocab_type='bert-lowercase',
        vocab_dir=_TESTDATA,
        vocab_file='bert_uncased_vocab.txt')
    self.assertEqual(sbw_size, 30522)

  def test_c4_bare_preprocess_fn(self):
    dataset = _c4_dataset()

    example = list(tfds.as_numpy(dataset.take(1)))[0]

    # Targets are NOT in the example.
    self.assertNotIn('targets', example)
    self.assertIn('text', example)
    text = example['text']

    # This should convert the dataset to an inputs/targets that are tokenized.
    dataset = tf_inputs.c4_bare_preprocess_fn(dataset, spm_path=_spm_path())

    example = list(tfds.as_numpy(dataset.take(1)))[0]

    # Earlier text is now stored in targets_pretokenized
    self.assertIn('targets_pretokenized', example)
    self.assertEqual(example['targets_pretokenized'], text)

    # Targets are now tokenized.
    self.assertIn('targets', example)
    self.assertIsInstance(example['targets'], np.ndarray)
    self.assertEqual(example['targets'].dtype, np.int64)
    self.assertGreater(len(example['targets']), 0)
    self.assertEqual(example['targets'][-1], 1)  # we add EOS at the end.

    # Inputs exist but is empty because t5 preprocessors' unsupervised wasn't
    # gin configured with any.
    self.assertIn('inputs', example)
    self.assertEqual(len(example['inputs']), 0)

  def test_c4_preprocess(self):

    def load_c4_dataset(split='train'):
      dataset = _c4_dataset(split=split)
      return dataset.map(lambda example: (example, example['text']))

    def examine_processed_dataset(proc_dataset):
      count = 0
      lengths = []
      for example in tfds.as_numpy(proc_dataset):
        count += 1
        ex = example[0]
        # Targets are in the example.
        self.assertIn('targets', ex)
        self.assertEqual(ex['targets'].dtype, np.int64)
        lengths.append(len(ex['targets']))
      return count, lengths

    unfiltered_count = 0
    for example in tfds.as_numpy(load_c4_dataset()):
      unfiltered_count += 1
      # Targets are NOT in the example.
      self.assertNotIn('targets', example[0])

    proc_dataset = tf_inputs.c4_preprocess(load_c4_dataset(), False, 2048)

    # `examine_processed_dataset` has some asserts in it.
    proc_count, char_lengths = examine_processed_dataset(proc_dataset)

    # Both the original and filtered datasets have examples.
    self.assertGreater(unfiltered_count, 0)
    self.assertGreater(proc_count, 0)

    # Because we filter out some entries on length.
    self.assertLess(proc_count, unfiltered_count)

    # Preprocess using the sentencepiece model in testdata.
    spc_proc_dataset = tf_inputs.c4_preprocess(
        load_c4_dataset(),
        False,
        2048,
        tokenization='spc',
        spm_path=_spm_path())

    spc_proc_count, spc_lengths = examine_processed_dataset(spc_proc_dataset)

    # spc shortens the target sequence a lot, should be almost equal to
    # unfiltered
    self.assertLessEqual(proc_count, spc_proc_count)
    self.assertEqual(unfiltered_count, spc_proc_count)

    # Assert all spc_lengths are lesser than their char counterparts.
    for spc_len, char_len in zip(spc_lengths, char_lengths):
      self.assertLessEqual(spc_len, char_len)

  def test_c4(self):
    gin.bind_parameter('c4_preprocess.max_target_length', 2048)
    gin.bind_parameter('c4_preprocess.tokenization', 'spc')
    gin.bind_parameter('c4_preprocess.spm_path', _spm_path())

    # Just make sure this doesn't throw.
    _ = tf_inputs.data_streams(
        'c4',
        data_dir=_TESTDATA,
        input_name='targets',
        target_name='text',
        preprocess_fn=tf_inputs.c4_preprocess)

  def test_c4_bare_preprocess_fn_denoising_objective(self):
    _t5_gin_config()

    dataset = _c4_dataset()
    dataset = tf_inputs.c4_bare_preprocess_fn(dataset, spm_path=_spm_path())

    example = list(tfds.as_numpy(dataset.take(1)))[0]

    # Assertions now.

    self.assertIn('targets', example)
    targets = example['targets']
    self.assertIsInstance(targets, np.ndarray)
    self.assertEqual(targets.dtype, np.int64)
    self.assertGreater(len(targets), 0)

    self.assertIn('inputs', example)
    _inputs = example['inputs']  # pylint: disable=invalid-name
    self.assertIsInstance(_inputs, np.ndarray)
    self.assertEqual(_inputs.dtype, np.int64)
    self.assertGreater(len(_inputs), 0)

    # WHP inputs will have the bulk of the text.
    self.assertGreater(len(_inputs), len(targets))

    # WHP there will be one sentinel token in the inputs and targets.
    inputs_counter = collections.Counter(_inputs.tolist())
    targets_counter = collections.Counter(targets.tolist())
    self.assertEqual(1, inputs_counter[31999])
    self.assertEqual(1, targets_counter[31999])

  def test_c4_pretrain(self):
    _t5_gin_config()

    gin.bind_parameter('c4_bare_preprocess_fn.spm_path', _spm_path())

    gin.bind_parameter('batcher.batch_size_per_device', 8)
    gin.bind_parameter('batcher.eval_batch_size', 8)
    gin.bind_parameter('batcher.max_eval_length', 50)
    gin.bind_parameter('batcher.buckets', ([51], [8, 1]))

    # Just make sure this doesn't throw.
    _ = tf_inputs.data_streams(
        'c4',
        data_dir=_TESTDATA,
        input_name='inputs',
        target_name='targets',
        bare_preprocess_fn=tf_inputs.c4_bare_preprocess_fn)

  def test_generic_text_dataset_preprocess_fn(self):
    dataset = _load_dataset('squad/v1.1:3.0.0')

    example, = tfds.as_numpy(dataset.take(1))

    self.assertNotIn('inputs', example)
    self.assertNotIn('targets', example)

    proc_dataset = tf_inputs.generic_text_dataset_preprocess_fn(
        dataset,
        spm_path=_spm_path(),
        text_preprocess_fns=[lambda ds, training: t5_processors.squad(ds)],
        copy_pretokenized=True,
        debug_print_examples=True,
        debug_print_examples_rate=1.0)

    proc_example, = tfds.as_numpy(proc_dataset.take(1))

    self.assertIn('inputs', proc_example)
    self.assertIn('targets', proc_example)

    self.assertEqual(proc_example['inputs'].dtype, np.int32)
    self.assertEqual(proc_example['targets'].dtype, np.int32)

  # TODO(afrozm): Why does this test take so much time?
  def test_inputs_using_generic_text_dataset_preprocess_fn(self):
    gin.bind_parameter('generic_text_dataset_preprocess_fn.spm_path',
                       _spm_path())
    gin.bind_parameter('generic_text_dataset_preprocess_fn.text_preprocess_fns',
                       [lambda ds, training: t5_processors.squad(ds)])

    # Just make sure this doesn't throw.
    def data_streams():
      return tf_inputs.data_streams(
          'squad',
          data_dir=_TESTDATA,
          input_name='inputs',
          target_name='targets',
          bare_preprocess_fn=tf_inputs.generic_text_dataset_preprocess_fn,
          shuffle_buffer_size=1)

    n_devices = 3

    squad_inputs = inputs.batcher(
        data_streams=data_streams,
        max_eval_length=512,
        buckets=([
            513,
        ], [n_devices, n_devices]))

    eval_stream = squad_inputs.eval_stream(n_devices)
    inps, tgts, _ = next(eval_stream)

    # We can only assert that the batch dim gets divided by n_devices.
    self.assertEqual(inps.shape[0] % n_devices, 0)
    self.assertEqual(tgts.shape[0] % n_devices, 0)

  def test_filter_dataset_on_len(self):
    # {1, 2}, {2, 4}, {3, 6} ... {10, 20}
    ds = _test_dataset_ints(range(1, 11), range(2, 21, 2))

    ds1 = tf_inputs.filter_dataset_on_len(ds, True, {
        'inputs': [4, 8],
        'targets': [14, 20]
    })
    # Only {7, 14} and {8, 16} satisfy this.
    self.assertLen(list(ds1.as_numpy_iterator()), 2)

    ds2 = tf_inputs.filter_dataset_on_len(
        ds,
        False,
        len_map={
            'inputs': [4, 8],
            'targets': [14, 20]
        },
        filter_on_eval=False)
    # This is eval and we aren't supposed to filter it.
    self.assertLen(list(ds2.as_numpy_iterator()), 10)

    ds3 = tf_inputs.filter_dataset_on_len(
        ds,
        False,
        len_map={
            'inputs': [4, 8],
            'targets': [14, 20]
        },
        filter_on_eval=True)
    # This is eval and we are asked to filter it.
    self.assertLen(list(ds3.as_numpy_iterator()), 2)

  def test_truncate_dataset_on_len(self):
    ds = _test_dataset_ints([5, 6, 7], [8, 9, 10])
    ds1 = tf_inputs.truncate_dataset_on_len(
        ds, True, len_map={
            'inputs': 6,
            'targets': 4
        })
    expected_ds = _test_dataset_ints([5, 6, 6], [4, 4, 4])

    # training, should filter.
    assert_dataset(ds1, list(expected_ds.as_numpy_iterator()))

    # not Training, shouldn't filter.
    ds2 = tf_inputs.truncate_dataset_on_len(
        ds, False, len_map={
            'inputs': 6,
            'targets': 4
        })
    assert_dataset(ds2, list(ds.as_numpy_iterator()))

    # not Training, but asked to filter, should filter.
    ds3 = tf_inputs.truncate_dataset_on_len(
        ds, False, len_map={
            'inputs': 6,
            'targets': 4
        }, truncate_on_eval=True)
    assert_dataset(ds3, list(expected_ds.as_numpy_iterator()))

  def test_get_t5_preprocessor_by_name(self):
    gin.clear_config()

    gin.parse_config("""
      get_t5_preprocessor_by_name.name = 'rekey'
      get_t5_preprocessor_by_name.fn_kwargs = {'key_map': {'inputs': 'other', 'targets': 'text'}}
    """)
    prep_rekey = tf_inputs.get_t5_preprocessor_by_name()
    og_dataset = tf.data.Dataset.from_tensors({
        'text': 'That is good.',
        'other': 'That is bad.'
    })
    training = True
    dataset = prep_rekey(og_dataset, training)
    assert_dataset(dataset, {
        'inputs': 'That is bad.',
        'targets': 'That is good.'
    })

  def test_pad_dataset_to_length(self):
    ds = _test_dataset_ints([5, 6, 7], [6, 7, 8])
    ds1 = tf_inputs.pad_dataset_to_length(
        ds, True, len_map={
            'inputs': 7,
            'targets': 10
        })

    expected_ds = [
        {
            'inputs': np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.int64),
            'targets': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        },
        {
            'inputs': np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.int64),
            'targets': np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int64),
        },
        {
            'inputs': np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
            'targets': np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64),
        },
    ]

    assert_dataset(ds1, expected_ds)

  def test_lm_token_preprocessing(self):
    ds = _test_dataset_ints([1, 2, 3], [3, 2, 1])
    ds1 = tf_inputs.lm_token_preprocessing(ds, True)

    # pylint: disable=bad-whitespace
    expected_ds = [
        {
            'inputs': np.array([1, 0, 1, 1, 1], dtype=np.int64),
            'targets': np.array([1, 0, 1, 1, 1], dtype=np.int64),
            'mask': np.array([0, 0, 1, 1, 1], dtype=np.int64),
        },
        {
            'inputs': np.array([1, 1, 0, 1, 1], dtype=np.int64),
            'targets': np.array([1, 1, 0, 1, 1], dtype=np.int64),
            'mask': np.array([0, 0, 0, 1, 1], dtype=np.int64),
        },
        {
            'inputs': np.array([1, 1, 1, 0, 1], dtype=np.int64),
            'targets': np.array([1, 1, 1, 0, 1], dtype=np.int64),
            'mask': np.array([0, 0, 0, 0, 1], dtype=np.int64),
        },
    ]
    # pylint: enable=bad-whitespace

    assert_dataset(ds1, expected_ds)

  def test_create_bert_inputs(self):
    inputs_sentences_1 = [np.array([100, 150, 200])]
    inputs_sentences_2 = [np.array([300, 500])]
    labels = [np.array(1)]

    create_inputs_1 = tf_inputs.CreateBertInputs(False)
    create_inputs_2 = tf_inputs.CreateBertInputs(True)
    for res in create_inputs_1(zip(inputs_sentences_1, labels)):
      values, segment_embs, _, label, weight = res
      self.assertAllEqual(values, np.array([101, 100, 150, 200, 102]))
      self.assertAllEqual(segment_embs, np.zeros(5))
      self.assertEqual(label, np.int64(1))
      self.assertEqual(weight, np.int64(1))

    for res in create_inputs_2(
        zip(inputs_sentences_1, inputs_sentences_2, labels)):
      values, segment_embs, _, label, weight = res
      self.assertAllEqual(values,
                          np.array([101, 100, 150, 200, 102, 300, 500, 102]))
      exp_segment = np.concatenate((np.zeros(5), np.ones(3)))
      self.assertAllEqual(segment_embs, exp_segment)
      self.assertEqual(label, np.int64(1))
      self.assertEqual(weight, np.int64(1))

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
        tf_inputs.mask_random_tokens(test_case))
    # test whether original tokens are unchanged
    self.assertAllEqual(test_case_row, original_tokens)

    self.assertEqual(1, token_weights.sum())
    self.assertEqual(
        15,
        (token_weights > 0).sum())  # we should have 15 candidates for masking

    # 101 is a special token, so only 1001 should be masked
    self.assertAllEqual(out[:100], test_case_row[:100])

    # Each candidate has 0.8 probability to be masked while others have 0, so
    # no more than 15 tokens with MASK
    self.assertLessEqual((out == mask_token).sum(), 15)

  def test_bert_next_sentence_prediction_inputs(self):
    stream = tf_inputs.BertNextSentencePredictionInputs(
        'c4/en:2.3.0', data_dir=_TESTDATA, train=False, shuffle_size=1)
    exp_sent1 = 'Police were called to the carriageway around 6.'
    exp_sent2 = 'I am sorry we did not see how lost and alone you felt.'
    sent1, sent2, label = next(stream())
    self.assertEqual(exp_sent1, sent1)
    self.assertEqual(exp_sent2, sent2)
    self.assertFalse(label)

  def test_process_single_mathqa_example_0(self):
    # This is the first problem in the MathQA dataset.
    example = {
        'Problem':
            "the banker ' s gain of a certain sum due 3 years hence at 10 % "
            'per annum is rs . 36 . what is the present worth ?',
        'Rationale':
            '"explanation : t = 3 years r = 10 % td = ( bg × 100 ) / tr = ( '
            '36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs . 120 td = ( pw × tr )'
            ' / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = '
            '1200 / 3 = rs . 400 answer : option a"',
        'options':
            'a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) '
            'none of these',
        'correct':
            'a',
        'annotated_formula':
            'divide(multiply(const_100, divide(multiply(36, const_100), '
            'multiply(3, 10))), multiply(3, 10))',
        'linear_formula':
            'multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|multiply(#2,const_100)|divide(#3,#1)|',
        'category':
            'gain'
    }

    answer_num, python_result, python_program, list_op, list_num = tf_inputs.process_single_mathqa_example(
        example)
    self.assertEqual(answer_num,
                     400)  # we know it, because correct answer is a)
    self.assertEqual(python_result, [3600.0, 30.0, 120.0, 12000.0, 400.0])

    self.assertEqual(python_program, [
        't0 = n2 * 100.0', 't1 = n0 * n1', 't2 = t0 / t1', 't3 = t2 * 100.0',
        't4 = t3 / t1'
    ])
    self.assertEqual(list_op, [
        'multiply(n2,const_100)', 'multiply(n0,n1)', 'divide(#0,#1)',
        'multiply(#2,const_100)', 'divide(#3,#1)'
    ])
    self.assertEqual(list_num, [3.0, 10.0, 36.0])

  def test_process_single_mathqa_example_1(self):
    # This is the third problem in the MathQA dataset.
    example = {
        'Problem':
            'sophia finished 2 / 3 of a book . she calculated that she '
            'finished 90 more pages than she has yet to read . how long is her'
            ' book ?',
        'Rationale':
            'let xx be the total number of pages in the book , then she '
            'finished 23 ⋅ x 23 ⋅ x pages . then she has x − 23 ⋅ x = '
            '13 ⋅ xx − 23 ⋅ x = 13 ⋅ x pages left . 23 ⋅ x − 13 '
            '⋅ x = 9023 ⋅ x − 13 ⋅ x = 90 13 ⋅ x = 9013 ⋅ x = 90 x'
            ' = 270 x = 270 so the book is 270 pages long . answer : b',
        'options': 'a ) 229 , b ) 270 , c ) 877 , d ) 266 , e ) 281',
        'correct': 'b',
        'annotated_formula': 'divide(90, subtract(const_1, divide(2, 3)))',
        'linear_formula': 'divide(n0,n1)|subtract(const_1,#0)|divide(n2,#1)',
        'category': 'general'
    }

    answer_num, python_result, python_program, list_op, list_num = tf_inputs.process_single_mathqa_example(
        example)
    self.assertEqual(answer_num,
                     270)  # we know it, because correct answer is b)
    self.assertAllClose(
        python_result,
        [0.6666666666666666, 0.33333333333333337, 269.99999999999994])
    self.assertEqual(python_program,
                     ['t0 = n0 / n1', 't1 = 1.0 - t0', 't2 = n2 / t1'])
    self.assertEqual(list_op,
                     ['divide(n0,n1)', 'subtract(const_1,#0)', 'divide(n2,#1)'])
    self.assertEqual(list_num, [2.0, 3.0, 90.0])

  def test_process_single_mathqa_example_with_import(self):
    # This is a training MathQA problem which involve an import.
    example = {
        'Problem':
            'the length of a rectangular garden is three times its width . if '
            'the area of the rectangular garden is 588 square meters , then '
            'what is the width of the rectangular garden ?',
        'Rationale':
            '\"let x be the width of the garden . 3 x ^ 2 = 588 x ^ 2 = 196 x '
            '= 14 the answer is c .\"',
        'options':
            'a ) 12 , b ) 13 , c ) 14 , d ) 15 , e ) 16',
        'correct':
            'c',
        'annotated_formula':
            'sqrt(divide(588, const_3))',
        'linear_formula':
            'divide(n0,const_3)|sqrt(#0)|',
        'category':
            'geometry'
    }

    answer_num, python_result, python_program, list_op, list_num = tf_inputs.process_single_mathqa_example(
        example)
    self.assertEqual(answer_num, 14)  # we know it, because correct answer is c)
    self.assertAllClose(python_result, [196, 14])
    self.assertEqual(
        python_program,
        ['t0 = n0 / 3.0', 't1 = math.sqrt(max(0, t0))'])
    self.assertEqual(list_op, ['divide(n0,const_3)', 'sqrt(#0)'])
    self.assertEqual(list_num, [588])

    # Below we execute twice the Python program and once the DSL program.
    target_values = 'import math\n'
    problem = example['Problem']
    for i in range(len(list_num)):
      target_values += 'n{} = {}\n'.format(i, list_num[i])
      problem += ' n{} = {}'.format(i, list_num[i])
    target_values += '\n'.join(python_program[:-1])
    final_line = python_program[-1].split('=')[1]
    target_values += '\nanswer ={}'.format(final_line)
    var_dict = {}
    exec(target_values, globals(), var_dict)  # pylint: disable=exec-used
    self.assertAllClose(var_dict['answer'], 14)
    self.assertAllClose(
        tf_inputs.execute_mathqa_program(problem, target_values.split('\n')),
        14)
    self.assertAllClose(
        tf_inputs.execute_mathqa_dsl_program(problem,
                                             [example['linear_formula']]), 14)


  def test_sentencepiece_tokenize(self):
    def dataset():
      yield 'I have a cat.'

    examples = []
    for example in tf_inputs.sentencepiece_tokenize(dataset(), _spm_path()):
      examples.append(example)
    toks = list(examples[0])
    self.assertSequenceEqual([27, 43, 3, 9, 1712, 5], toks)


if __name__ == '__main__':
  tf.test.main()
