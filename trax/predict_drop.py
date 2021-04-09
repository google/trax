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
"""Prediction binary for the Drop task.

Binary that loads a checkpoint and runs inference on selected problems
from the Drop dataset. For more details about Drop see
https://arxiv.org/pdf/1903.00161.pdf.
"""

import json
import os
import re
import time

from absl import app as absl_app
from absl import flags
import gin
import jax
import numpy as np
from seqio import vocabularies as t5_spc_vocab
from t5 import data
import tensorflow as tf
from trax import data as trax_data
from trax import layers as tl
from trax import shapes
from trax.supervised import decoding


FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '',
                    'Path to model checkpoint.')
flags.DEFINE_integer('max_answer_len', 1024,
                     'Maximum length of answers to produce.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for eval.')
flags.DEFINE_integer('num_examples', 1, 'Number of examples to infer.')
flags.DEFINE_integer('n_hashes', None,
                     'n_hashes parameter to override in attentions.')
flags.DEFINE_integer('example_repetitions', 1,
                     'How many times to infer an example.')
flags.DEFINE_bool('use_eval_mode', False,
                  'If True, use the slower but easier to debug eval mode.')
flags.DEFINE_bool('use_eval_set', False,
                  'If True, use eval set for evaluation.')
flags.DEFINE_bool(
    'use_beam_search', False,
    'If True, use beam search, otherwise use autoregresive sampling.')
flags.DEFINE_float('autoregressive_sample_temp', 1,
                   'The temperature for autoregressive sampling.')
flags.DEFINE_integer('n_beams', 4, 'How many beams to use in beam search.')
flags.DEFINE_string(
    'output_dir', '', 'Path to the output directory where articles, abstracts, '
    'and predictions would be stored.')
flags.DEFINE_integer('starting_example', 0,
                     'Example index for starting decoding.')
flags.DEFINE_integer('reload_after', 1000,
                     'Reload checkpoint after reload_after examples.')
flags.DEFINE_multi_string('config_file', None,
                          'Configuration file with parameters (.gin).')


def _check_exists(file_path):
  if not tf.io.gfile.exists(file_path):
    print('No such file: %s' % file_path, flush=True)
    exit(1)


def multiply_examples(example):
  for i in range(FLAGS.example_repetitions):
    yield i, example


def prepare_model(model_file, batch_size=1):
  """Prepare the model."""
  mode = 'eval' if FLAGS.use_eval_mode else 'predict'
  print('Initializing the model in %s mode.' % mode, flush=True)

  # Read the model name from the gin file
  model_reference = gin.query_parameter(
      'trax.supervised.trainer_lib.train.model')
  model = model_reference.scoped_configurable_fn(mode=mode)

  dec_len = 32 if FLAGS.use_eval_mode else 1
  batch_size_pd = max(1, batch_size // jax.local_device_count())
  shape11 = shapes.ShapeDtype((batch_size_pd, dec_len), dtype=np.int32)
  # shape11 = shapes.ShapeDtype((1, 1), dtype=np.int32)
  model.init_from_file(
      model_file, weights_only=True, input_signature=(shape11, shape11))
  model = tl.Accelerate(model)

  initial_state = model.state
  vocab = t5_spc_vocab.SentencePieceVocabulary(data.DEFAULT_SPM_PATH)

  return vocab, model, initial_state


def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def main(argv):
  if len(argv) > 1:
    raise absl_app.UsageError('Too many command-line arguments.')
  if not FLAGS.output_dir:
    raise absl_app.UsageError('--output_dir needs to be provided.')

  tf.compat.v1.enable_eager_execution()

  # Check that checkpoint_dir is correct: should contain model.pkl.gz file.
  model_file = os.path.join(FLAGS.checkpoint_dir, 'model.pkl.gz')
  _check_exists(model_file)

  gin.parse_config_file(os.path.join(FLAGS.checkpoint_dir, 'config.gin'))
  # Batching on our own because of possible repetitions of examples.
  gin.bind_parameter('data.Batch.batch_size', 1)
  if FLAGS.n_hashes is not None:
    gin.bind_parameter('LSHSelfAttention.n_hashes', FLAGS.n_hashes)
    gin.bind_parameter('ref2_encoder/LSHSelfAttention.n_hashes', FLAGS.n_hashes)

  vocab, model, initial_state = prepare_model(model_file, FLAGS.batch_size)

  host_id, host_count = jax.host_id(), jax.host_count()
  print('Running on host %d out of %d.' % (host_id, host_count))

  example_count = 0
  start_time = time.time()

  # Creates all intermediate directories if they do not exist
  tf.io.gfile.makedirs(FLAGS.output_dir)

  json_to_write = os.path.join(FLAGS.output_dir, 'output%d.json' % host_id)
  all_jsons = []

  # In a case of a reset we have to check how much work was already done.
  # We can check whether the processing of an example was finished, but
  # currently we are only checking whether it was started.
  done = FLAGS.starting_example
  reload_count = 0
  all_existing_files = tf.io.gfile.listdir(FLAGS.output_dir)
  for filename in all_existing_files:
    if 'processing' in filename:
      # The definition of digits looks for a number after the infix "processing"
      # in the file name. Example: tom_processing_532 will lead to
      # digits = "processing_532" and number equal to "532".
      digits = filename[filename.find('processing'):]
      number = ''.join(d for d in digits if d.isdigit())
      if is_number(
          number) and int(number) < FLAGS.num_examples + FLAGS.starting_example:
        done = max(done, int(number))
  print('The done number is {}'.format(done))

  if FLAGS.use_eval_set:
    drop_gen = trax_data.CreateDropInputs(train=False)()
  else:
    drop_gen = trax_data.CreateDropInputs(train=True)()
  padding_fun = trax_data.PadToLength()

  # TODO(henrykm): improve managment of the counters.
  # example_count_total - all numeric examples
  # example_count - all numeric examples above starting_example
  # reload_count - if we processed FLAGS.reload_after examples,
  #   then the checkpoint should be reloaded.
  # idx - total number of exaples
  example_count_total = 0
  reload_count += 1
  for idx, e in enumerate(drop_gen):
    if reload_count >= FLAGS.reload_after:
      vocab, model, initial_state = prepare_model(model_file, FLAGS.batch_size)
      reload_count = 0
    if example_count >= FLAGS.num_examples:
      print('Reached the example_count {} - breaking'.format(example_count))
      break
    if not is_number(e[1]):
      continue
    target_answer = float(e[1])

    # We count numeric starting examples
    example_count_total += 1
    if example_count_total <= FLAGS.starting_example:
      print('Skipping example_count_total {} because it is below {}'.format(
          example_count_total, FLAGS.starting_example))
      continue

    if example_count % 10 == 0:
      elapsed_time = time.time() - start_time
      start_time = time.time()
      print('Starting inference on example %d, %.2fs since last log' %
            (example_count, elapsed_time), flush=True)

    example_count += 1
    if example_count <= done - FLAGS.starting_example + 1:
      print('Skipping example_count {} because it is below {}'.format(
          example_count, done - FLAGS.starting_example))
      # We are increasing the example_count because the example
      # was processed before
      continue

    if example_count % host_count != host_id:
      continue

    # At this point we are committed to the processing of an example with
    # index example_count
    processing_file = os.path.join(FLAGS.output_dir, 'processing_')
    data_id = str(example_count + FLAGS.starting_example)
    with tf.io.gfile.GFile(processing_file + data_id, 'w') as w:
      w.write('Procesing started.')
    for repetition_id, example in multiply_examples(e):
      question = example[0]
      question_text = question[question.find(':') + 2:]
      question_text = question_text.replace('-', ' - ')
      question = 'infer full calculation: ' + question_text

      list_num = [
          float(num.replace(',', '').rstrip('.')) for num in re.findall(
              r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', question)
      ]
      for i in range(len(list_num)):
        question += ' n{} = {}'.format(i, list_num[i])

      # print('Question {}'.format(question))
      tokenized_question = next(
          padding_fun(
              trax_data.tokenize([
                  question,
              ],
                                 vocab_file=gin.query_parameter(
                                     'trax.data.Tokenize.vocab_file'))))
      state = model.state
      if FLAGS.use_beam_search:
        answer_beams = decoding.beam_search(
            model,
            tokenized_question[None, :],
            n_beams=FLAGS.n_beams,
            max_length=FLAGS.max_answer_len,
            accelerate=False)
        model.state = state
      else:
        answer_beams = []
        # We recycle the n_beams flag to control the number
        # of autoregressive samples.
        for i in range(FLAGS.n_beams):
          answer = decoding.autoregressive_sample(
              model,
              tokenized_question[None, :],
              temperature=FLAGS.autoregressive_sample_temp,
              max_length=FLAGS.max_answer_len,
              accelerate=False)
          model.state = state
          answer_beams.append(answer)

      correct_example_index = -1

      for i in range(len(answer_beams)):
        if FLAGS.use_beam_search:
          answer = trax_data.detokenize(
              answer_beams[i][0][0],
              vocab_file=gin.query_parameter('trax.data.Tokenize.vocab_file'))
        else:
          answer = trax_data.detokenize(
              answer_beams[i][0],
              vocab_file=gin.query_parameter('trax.data.Tokenize.vocab_file'))
        print('Proposed computation {}'.format(answer))
        list_op = answer.split('|')
        if not list_op[-1]:
          list_op = list_op[:-1]

        try:
          result = trax_data.tf_inputs.compute_result(list_op, list_num)
          if target_answer in result:
            correct_example_index = result.index(target_answer)
            break
        # This is a temporary hack with "broad" exceptions - the computations
        # must fail sometime, because we evaluate arbitrary sequences; I am in
        # the process of checking what are possible failure modes.
        except Exception as e:  # pylint: disable=broad-except
          print(e)
          try:
            result = trax_data.tf_inputs.compute_result(list_op[:-1], list_num)
            if target_answer in result:
              correct_example_index = result.index(target_answer)
              break
          except Exception as e:  # pylint: disable=broad-except
            print(e)
            print('Infered incorrect computation.')

      if correct_example_index == -1:
        continue

      json_record = {
          'question': question_text,
          'input': question,
          'calculation': '|'.join(list_op[:correct_example_index + 1]),
          'target_answer': target_answer
      }
      all_jsons.append(json.dumps(json_record) + '\n')
      # Outputting the inferred data in JSONL format.
      data_id = str(example_count + FLAGS.starting_example)
      with tf.io.gfile.GFile(json_to_write + data_id, 'w') as w:
        w.write(json.dumps(json_record) + '\n')
    with tf.io.gfile.GFile(processing_file + data_id, 'w') as w:
      w.write('Procesing finished.')

  with tf.io.gfile.GFile(json_to_write + '_' + str(FLAGS.starting_example),
                         'w') as w:
    for record in all_jsons:
      w.write(record)


if __name__ == '__main__':
  absl_app.run(main)
