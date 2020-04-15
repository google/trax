# coding=utf-8
# Copyright 2020 The Trax Authors.
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
"""Inputs and training loops for pre-training and fine-tuning.

For now, this file only supports fine-tuning bert-base-uncased on GLUE.
"""
import functools
import os

import gin
import jax
import numpy as onp

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import trax.layers as tl
import trax.layers.base
from trax.math import numpy as np
import trax.models
import trax.optimizers
from trax.supervised.inputs import Inputs
from trax.supervised.trainer_lib import Trainer


class MultiInputs(Inputs):
  """Supports tuples and not just single tensors for model input."""

  def __init__(self, train_stream, eval_stream=None, train_eval_stream=None,
               extra_streams=None):
    self._train_stream = train_stream
    self._eval_stream = eval_stream or self._train_stream
    self._extra_streams = extra_streams

    # TODO(lukaszkaiser): should we get rid of this one day?
    self._train_eval_stream = train_eval_stream or self._train_stream

    # Peek into the train stream to get an example shape.
    example_train_batch = next(train_stream(1))
    def get_shape(x):
      return tuple(x.shape)[1:]
    def get_dtype(x):
      return x.dtype

    self._input_shape = jax.tree_map(get_shape, example_train_batch[:-1])
    self._input_dtype = jax.tree_map(get_dtype, example_train_batch[:-1])
    self._target_shape = jax.tree_map(get_shape, example_train_batch[-1])
    self._target_dtype = jax.tree_map(get_dtype, example_train_batch[-1])

  def extra_streams(self, n_devices):
    return [stream(n_devices) for stream in self._extra_streams]


def _tfds_stream(n_devices, dataset_name, split, batch_size, data_dir,
                 shuffle_files, shuffle_buffer_size, batch_shuffle_size,
                 preprocess_fun, repeat=True):
  """Streams batches of examples from tfds, with pure-python preprocessing."""
  if batch_size % n_devices != 0:
    raise ValueError(f'Batch size ({batch_size}) not divisible'
                     ' by number of devices ({n_devices})')
  ds = tfds.load(
      name=dataset_name, split=split, data_dir=data_dir,
      shuffle_files=shuffle_files)
  if repeat:
    ds = ds.repeat()
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.batch(batch_size)
  if batch_shuffle_size is not None:
    ds = ds.shuffle(batch_shuffle_size)

  for batch in tfds.as_numpy(ds):
    if preprocess_fun is not None:
      yield preprocess_fun(batch)
    else:
      yield batch


@gin.configurable()
def tfds_inputs(
    dataset_name,
    preprocess_fun,
    batch_size,
    eval_batch_size=None,
    data_dir=None,
    train_split=tfds.Split.TRAIN,
    eval_split=tfds.Split.VALIDATION,
    extra_splits=(tfds.Split.VALIDATION, tfds.Split.TEST,),
    shuffle_buffer_size=1024,
    batch_shuffle_size=128,
    ):
  """Tensorflow Datasets input pipeline, with pure-python preprocessing."""
  if eval_batch_size is None:
    eval_batch_size = batch_size
  return MultiInputs(
      train_stream=functools.partial(
          _tfds_stream,
          dataset_name=dataset_name,
          split=train_split,
          batch_size=batch_size,
          data_dir=data_dir,
          shuffle_files=True,
          shuffle_buffer_size=shuffle_buffer_size,
          batch_shuffle_size=batch_shuffle_size,
          preprocess_fun=preprocess_fun,
          ),
      eval_stream=functools.partial(
          _tfds_stream,
          dataset_name=dataset_name,
          split=eval_split,
          batch_size=eval_batch_size,
          data_dir=data_dir,
          shuffle_files=False,
          shuffle_buffer_size=None,
          batch_shuffle_size=None,
          preprocess_fun=preprocess_fun,
          ),
      extra_streams=[functools.partial(  # pylint: disable=g-complex-comprehension
          _tfds_stream,
          dataset_name=dataset_name,
          split=split,
          batch_size=eval_batch_size,
          data_dir=data_dir,
          shuffle_files=False,
          shuffle_buffer_size=None,
          batch_shuffle_size=None,
          preprocess_fun=preprocess_fun,
          repeat=False,
          ) for split in extra_splits]
  )


@gin.configurable()
def bert_tokenizer(vocab_path=None):
  """Constructs a BERT tokenizer."""

  # This import is from https://github.com/google-research/bert which is not
  # listed as a dependency in trax.
  from bert.tokenization import FullTokenizer
  if vocab_path is None:
    raise ValueError('vocab_path is required to construct the BERT tokenizer.')
  tokenizer = FullTokenizer(vocab_path, do_lower_case=True)
  return tokenizer


@gin.configurable()
def glue_inputs(dataset_name, batch_size, eval_batch_size=None, data_dir=None,
                max_len=128, tokenizer=bert_tokenizer):
  """Input pipeline for fine-tuning BERT on GLUE tasks."""
  if callable(tokenizer):  # If we pass a function, e.g., through gin, call it.
    tokenizer = bert_tokenizer()

  eval_split = tfds.Split.VALIDATION
  extra_splits = (tfds.Split.VALIDATION, tfds.Split.TEST,)
  if dataset_name == 'glue/mnli':
    eval_split = 'validation_matched'
    # TODO(kitaev): Support diagnostic dataset (AX)
    extra_splits = ['validation_matched', 'validation_mismatched',
                    'test_matched', 'test_mismatched']

  keys_lookup = {
      'glue/cola': ('sentence', None),
      'glue/sst2': ('sentence', None),
      'glue/mrpc': ('sentence1', 'sentence2'),
      'glue/qqp': ('question1', 'question2'),
      'glue/stsb': ('sentence1', 'sentence2'),
      'glue/mnli': ('premise', 'hypothesis'),   # TODO(kitaev): swap the two?
      'glue/qnli': ('question', 'sentence'),  # TODO(kitaev) swap the two?
      'glue/rte': ('sentence1', 'sentence2'),
      'glue/wnli': ('sentence1', 'sentence2'),
  }

  key_a, key_b = keys_lookup[dataset_name]

  if key_b is None:
    def preprocess(batch):
      """Tokenize and convert text to model inputs."""
      batch_size = batch['idx'].shape[0]
      input_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
      type_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)

      for i in range(batch_size):
        sentence_a = batch[key_a][i]
        tokens_a = [101] + tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sentence_a)) + [102]
        input_ids[i, :len(tokens_a)] = tokens_a[:max_len]

      return input_ids, type_ids, batch['idx'].astype(np.int32), batch['label']
  else:
    def preprocess(batch):
      """Tokenize and convert text to model inputs."""
      batch_size = batch['idx'].shape[0]
      input_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)
      type_ids = onp.zeros((batch_size, max_len), dtype=onp.int32)

      for i in range(batch_size):
        sentence_a = batch[key_a][i]
        sentence_b = batch[key_b][i]
        tokens_a = [101] + tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sentence_a)) + [102]
        tokens_b = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sentence_b)) + [102]

        ex_input_ids = (tokens_a + tokens_b)[:max_len]
        ex_type_ids = ([0] * len(tokens_a) + [1] * len(tokens_b))[:max_len]

        input_ids[i, :len(ex_input_ids)] = ex_input_ids
        type_ids[i, :len(ex_type_ids)] = ex_type_ids

      return input_ids, type_ids, batch['idx'].astype(np.int32), batch['label']

  return tfds_inputs(
      dataset_name=dataset_name,
      preprocess_fun=preprocess,
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      data_dir=data_dir,
      train_split=tfds.Split.TRAIN,
      eval_split=eval_split,
      extra_splits=extra_splits,
      )


def get_accuracy(guess, gold):
  return (guess == gold).mean()


def get_mcc(guess, gold):
  tp = ((guess == 1) & (gold == 1)).sum()
  tn = ((guess == 0) & (gold == 0)).sum()
  fp = ((guess == 1) & (gold == 0)).sum()
  fn = ((guess == 0) & (gold == 1)).sum()
  mcc_denom = onp.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc = (tp * tn - fp * fn) / (mcc_denom + 1e-6)
  return mcc


def get_f1(guess, gold):
  tp = ((guess == 1) & (gold == 1)).sum()
  fp = ((guess == 1) & (gold == 0)).sum()
  fn = ((guess == 0) & (gold == 1)).sum()
  f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
  return f1


def get_f1_accuracy_mean(guess, gold):
  return (get_f1(guess, gold) + get_accuracy(guess, gold)) / 2.0


def get_pearsonr(x, y):
  return onp.corrcoef(x, y)[0, 1]


@gin.configurable(blacklist=['output_dir'])
def finetune(output_dir, model=gin.REQUIRED, dataset_name=gin.REQUIRED,
             batch_size=16, num_train_epochs=3.0):
  """Fine-tuning loop for GLUE, largely following the BERT recipe."""
  ds_info = tfds.builder(dataset_name).info
  is_regression_task = (ds_info.features.dtype['label'] == onp.float32)

  if is_regression_task:
    # Regression task
    loss_fn = tl.L2Loss()
    metrics = {
        'loss': tl.L2Loss(),
        'weights_per_batch_per_core': tl.SumOfWeights(),
    }
    model = functools.partial(model, head=trax.models.BERTRegressionHead)
  else:
    # Classification task
    loss_fn = tl.CrossEntropyLoss()
    metrics = {
        'loss': tl.CrossEntropyLoss(),
        'accuracy': tl.AccuracyScalar(),
        'weights_per_batch_per_core': tl.SumOfWeights(),
    }
    n_classes = ds_info.features['label'].num_classes
    with gin.unlock_config():
      gin.parse_config(f'BERTClassifierHead.n_classes = {n_classes}')
    model = functools.partial(model, head=trax.models.BERTClassifierHead)

  num_train_examples = ds_info.splits[tfds.Split.TRAIN].num_examples
  total_steps = int(num_train_examples * num_train_epochs // batch_size)
  warmup_steps = int(0.1 * total_steps)
  cooldown_steps = total_steps - warmup_steps

  # TODO(kitaev): Re-think how configuration works for this setup.
  with gin.unlock_config():
    gin.parse_config(f"""
    # TODO(kitaev): Devlin et al. use linear decay, not cosine decay
    MultifactorSchedule.factors = 'constant * linear_warmup * cosine_decay'
    MultifactorSchedule.warmup_steps = {warmup_steps}
    MultifactorSchedule.steps_per_cycle = {cooldown_steps}

    # TODO(kitaev): Devlin et al. use 0.01, but exclude biases from weight decay
    Adam.weight_decay_rate=0.0
    Adam.b1 = 0.9
    Adam.b2 = 0.999
    Adam.eps = 1e-6

    glue_inputs.dataset_name = '{dataset_name}'
    glue_inputs.batch_size = {batch_size}
    glue_inputs.tokenizer = @bert_tokenizer
    """)

  trainer = Trainer(
      model, loss_fn,
      optimizer=trax.optimizers.Adam,
      lr_schedule=trax.lr_schedules.MultifactorSchedule,
      inputs=glue_inputs,
      output_dir=output_dir,
      random_seed=None,
      n_devices=None,  # Use all available.
      checkpoints_at=None,
      nontrainable_param_map=None,
      metrics=metrics,
      id_to_mask=None,
      checkpoint_lowest=None,
      checkpoint_highest=None,
      )

  trainer.log_step('Starting training using %d devices' % trainer.n_devices)
  trainer.print_n_weights()

  trainer.train_epoch(n_steps=1, n_eval_steps=10)
  trainer.save_gin()
  trainer.train_epoch(n_steps=warmup_steps - 1, n_eval_steps=10)
  trainer.train_epoch(n_steps=cooldown_steps, n_eval_steps=10)

  trainer.log_step('Training done')

  # Evaluation
  # pylint: disable=protected-access
  def my_jit(forward, n_devices):
    """Returns a JIT-compiled forward function running on n_devices."""
    model_predict = trax.layers.base._accelerate(forward, n_devices)
    if n_devices == 1:
      def predict1(x, weights, state):
        res, state = model_predict(x, weights, state, rng=jax.random.PRNGKey(0))
        return res
      return predict1

    def predict(x, weights, state):
      """Predict function jited and parallelized as requested."""
      res, state = trax.layers.base._combine_devices(model_predict(
          trax.layers.base.reshape_by_device(x, n_devices),
          weights,
          state,
          np.broadcast_to(jax.random.PRNGKey(0)[None, :], (8, 2))))
      return res

    return predict

  fwd = functools.partial(
      my_jit(trainer._model_predict_eval.pure_fn, trainer._n_devices),
      weights=trainer._opt_state[0][0],
      state=trainer._model_state[0])

  def run_model(stream):
    """Run forward pass on a dataset."""
    all_out = []
    all_idx = []
    all_labels = []
    for input_ids, type_ids, idx, labels in stream:
      remainder = labels.shape[0] % trainer._n_devices
      if remainder:
        pad_amount = trainer._n_devices - remainder
        input_ids = onp.pad(
            input_ids, ((0, pad_amount), (0, 0)), mode='constant')
        type_ids = onp.pad(type_ids, ((0, pad_amount), (0, 0)), mode='constant')
        padded_idx = onp.pad(idx, ((0, pad_amount),), mode='constant')
      else:
        padded_idx = idx
      out = onp.array(fwd((input_ids, type_ids, padded_idx)))
      if remainder:
        out = out[:-pad_amount]
      all_out.append(out)
      all_idx.append(idx)
      all_labels.append(labels)
    all_out = onp.concatenate(all_out, axis=0)
    all_idx = onp.concatenate(all_idx, axis=0)
    all_labels = onp.concatenate(all_labels, axis=0)

    return all_out, all_labels, all_idx

  eval_metrics = {}
  if is_regression_task:
    eval_metrics['pearsonr'] = get_pearsonr
  else:
    eval_metrics['accuracy'] = get_accuracy

  if dataset_name == 'glue/cola':
    eval_metrics['mcc'] = get_mcc
  elif dataset_name in ('glue/mrpc', 'glue/qqp'):
    eval_metrics['f1_accuracy_mean'] = get_f1_accuracy_mean

  preds_labels_idxs = [
      run_model(stream) for stream in trainer._inputs.extra_streams(
          trainer._n_devices)]

  # Log results on development data
  eval_results_path = os.path.join(trainer._output_dir, 'eval_results.txt')
  with tf.io.gfile.GFile(eval_results_path, 'w') as f:
    guess, gold, _ = preds_labels_idxs[0]
    if is_regression_task:
      guess = guess[:, 0]
    else:
      guess = guess.argmax(-1)
    for name, fn in sorted(eval_metrics.items()):
      val = fn(guess, gold)
      f.write(f'eval_{name} = {val:.06f}\n')
      trainer.log_step(f'eval_{name} = {val:.06f}\n')

    if dataset_name == 'glue/mnli':
      guess, gold, _ = preds_labels_idxs[1]
      guess = guess.argmax(-1)
      for name, fn in sorted(eval_metrics.items()):
        val = fn(guess, gold)
        f.write(f'eval_mismatched_{name} = {val:.06f}\n')
        trainer.log_step(f'eval_mismatched_{name} = {val:.06f}\n')

    f.write(f'global_step = {trainer.step}\n')

  # Write predictions for test data
  path_map = {
      'glue/cola': 'CoLA.tsv',
      'glue/mrpc': 'MRPC.tsv',
      'glue/qqp': 'QQP.tsv',
      'glue/sst2': 'SST-2.tsv',
      'glue/mnli': 'MNLI-mm.tsv',
      'glue/qnli': 'QNLI.tsv',
      'glue/rte': 'RTE.tsv',
      # No eval on WNLI for now. BERT accuracy on WNLI is below baseline, unless
      # special training recipe is used.
      # 'glue/wnli': 'WNLI.tsv',
  }

  if dataset_name == 'glue/stsb':
    test_results_path = os.path.join(trainer._output_dir, 'STS-B.tsv')
    idxs = preds_labels_idxs[-1][2]
    guess = preds_labels_idxs[-1][0][:, 0]
    with tf.io.gfile.GFile(test_results_path, 'w') as f:
      f.write('index\tprediction\n')
      for idx, val in zip(idxs, guess):
        f.write(f'{idx}\t{val:.06f}\n')
  elif dataset_name in path_map:
    if dataset_name in ('glue/cola', 'glue/mrpc', 'glue/qqp', 'glue/sst2'):
      label_set = ['0', '1']
    elif dataset_name in ('glue/qnli', 'glue/rte'):
      label_set = ['entailment', 'not_entailment']
    elif dataset_name == 'glue/mnli':
      label_set = ['entailment', 'neutral', 'contradiction']
    else:
      assert False, f'Unexpected dataset_name {dataset_name}'

    test_results_path = os.path.join(
        trainer._output_dir, path_map[dataset_name])

    idxs = preds_labels_idxs[-1][2]
    guess = preds_labels_idxs[-1][0].argmax(-1)
    with tf.io.gfile.GFile(test_results_path, 'w') as f:
      f.write('index\tprediction\n')
      for idx, val in zip(idxs, guess):
        f.write(f'{idx}\t{label_set[val]}\n')

    trainer.log_step(f'Predictions written to {test_results_path}')

    if dataset_name == 'glue/mnli':
      test_results_path = os.path.join(trainer._output_dir, 'MNLI-m.tsv')
      idxs = preds_labels_idxs[-2][2]
      guess = preds_labels_idxs[-2][0].argmax(-1)
      with tf.io.gfile.GFile(test_results_path, 'w') as f:
        f.write('index\tprediction\n')
        for idx, val in zip(idxs, guess):
          f.write(f'{idx}\t{label_set[val]}\n')
      trainer.log_step(f'Predictions written to {test_results_path}')

  return trainer, preds_labels_idxs
