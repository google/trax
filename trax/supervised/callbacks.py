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
"""Loop callbacks.

Callbacks can be used to customize the behavior of `supervised.training.Loop`
to accomodate a variety of use-cases.

Examples include:
  - custom evaluation schemes
  - logging metrics to external servers
  - sending model checkpoints to external servers
  - updating the target network in RL algorithms and other non-stationary
    problems
"""

import collections
import os

import gin
import numpy as np

from trax import jaxboard
from trax import layers as tl
from trax import shapes
from trax.rl import serialization_utils
from trax.supervised import decoding


class TrainingStepCallback:
  """Callback triggered before and after a training step."""

  def __init__(self, loop):
    """Initializes the callback with a `supervised.training.Loop` instance."""
    self._loop = loop

  def call_at(self, step):
    """Returns whether the callback should be called at a given step."""
    raise NotImplementedError

  def on_step_begin(self, step):
    """Called by Loop before training steps, when call_at returned True."""
    raise NotImplementedError

  def on_step_end(self, step):
    """Called by Loop after training steps, when call_at returned True."""
    raise NotImplementedError


@gin.configurable
class SerializedModelEvaluation(TrainingStepCallback):
  """Evaluates serialized sequence prediction models.

  Example: time series prediction. We can serialize a time series into
  a sequence of discrete tokens and model this sequence using an autoregressive
  sequence model, such as Transformer - see
  `trax.rl.serialization_utils.SerializedModel`. Then we can use this callback
  to evaluate long-horizon predictions of such a model.
  """

  def __init__(
      self,
      loop,
      model=None,
      eval_at=1000,
      eval_task=None,
      context_lengths=(1,),
      horizon_lengths=(1,),
      n_steps=1,
      accelerate_model=True,
  ):
    """Initializes SerializedModelEvaluation.

    Args:
      loop: Instance of `trax.supervised.training.Loop` or `None`. Can be set to
        `None` for testing - in such a case, `model` and `eval_task` must be
        provided.
      model: Instance of `trax.rl.serialization_utils.SerializedModel`. Not
        required if `loop` is provided.
      eval_at: When to evaluate. Either int (every how many steps to evaluate),
        or a list of ints (step numbers), or a function int -> bool (step
        predicate).
      eval_task: Instance of `trax.supervised.training.EvalTask` with the
        evaluation data, or None. If not provided, the task will be taken from
        `loop`.
      context_lengths: List of lengths of the context sequence fed into the
        model before starting prediction.
      horizon_lengths: List of lengths of the predicted sequence.
      n_steps: Number of batches to run evaluation for.
      accelerate_model (bool): Whether to wrap the model in `tl.Accelerate`.
    """
    super().__init__(loop)

    if model is None:
      model = loop.model

    observation_serializer = model.observation_serializer
    action_serializer = model.action_serializer

    predict_model = model.make_predict_model()
    if accelerate_model:
      predict_model = tl.Accelerate(predict_model)
    self._predict_model = predict_model
    self._obs_serializer = observation_serializer
    self._act_serializer = action_serializer

    if isinstance(eval_at, int):
      self._eval_at = lambda step: step % eval_at == 1
    elif hasattr(eval_at, '__in__'):
      self._eval_at = lambda step: step in eval_at
    elif callable(eval_at):
      self._eval_at = eval_at
    else:
      raise TypeError(f'Unsupported type for eval_at: {type(eval_at)}.')

    if eval_task is None:
      if len(loop.eval_tasks) != 1:
        raise ValueError(
            'If eval_task is not provided, the number of eval_tasks registered '
            'in Loop must be exactly 1.'
        )
      eval_task = loop.eval_tasks[0]
    self._eval_task = eval_task

    self._context_lengths = list(sorted(context_lengths))
    self._horizon_lengths = list(sorted(horizon_lengths))
    self._n_steps = n_steps

    self._batch_size = eval_task.sample_batch[0].shape[0]
    (_, self._init_state) = predict_model.init(
        shapes.ShapeDtype((self._batch_size, 1), dtype=np.int32)
    )

  @property
  def predict_model(self):
    return self._predict_model

  def call_at(self, step):
    return self._eval_at(step)

  def on_step_begin(self, step):
    pass

  def on_step_end(self, step):
    summary_writer = jaxboard.SummaryWriter(
        os.path.join(self._loop.output_dir, 'srl_eval')
    )
    try:
      weights = self._loop.eval_model.seq_model_weights
      metrics = self.evaluate(weights)
      self._loop.log_summary(metrics, summary_writer, '', 'srl_eval')
    finally:
      summary_writer.close()

  def evaluate(self, weights):
    """Evaluates the model and returns the metrics."""
    self._predict_model.weights = weights

    metrics = collections.defaultdict(list)
    for _ in range(self._n_steps):
      batch = self._eval_task.next_batch()
      step_metrics = self._evaluate_batch(batch)
      for (key, value) in step_metrics.items():
        metrics[key].append(value)

    metrics = {k: np.array(v) for (k, v) in metrics.items()}

    def metric_name(context, horizon):
      return f'pred_error/context_{context}/horizon_{horizon}'

    return {
        metric_name(context, horizon):
            np.sum(errors) / (np.sum(errors != 0) + 1e-6)
        for ((context, horizon), errors) in metrics.items()
    }

  def _evaluate_batch(self, batch):
    """Performs evaluation on a single batch."""
    (obs, act, _, mask) = batch
    obs_repr = serialization_utils.Serialize(self._obs_serializer)(obs)
    act_repr = serialization_utils.Serialize(self._act_serializer)(act)

    errors = {}
    last_context = 0
    last_state = self._init_state
    last_start_id = 0
    for context in self._context_lengths:
      self._predict_model.state = last_state
      start_id = last_start_id

      if context > last_context:
        context_seq = serialization_utils.Interleave()((
            obs_repr[:, last_context:context], act_repr[:, last_context:context]
        ))
        consume_sequence(self._predict_model, start_id, context_seq[:, :-1])
        last_start_id = start_id = context_seq[:, -1:]
        last_state = self._predict_model.state
        last_context = context

      for timestep in range(max(self._horizon_lengths)):
        pred_repr = decoding.autoregressive_sample(
            self._predict_model,
            start_id=start_id,
            eos_id=-1,
            batch_size=self._batch_size,
            max_length=self._obs_serializer.representation_length,
            accelerate=False,
        )
        horizon = timestep + 1
        if horizon in self._horizon_lengths:
          pred = self._obs_serializer.deserialize(pred_repr)
          error = self._calculate_error(pred, obs[:, context + timestep])
          errors[context, horizon] = error * mask[:, context + timestep]

        start_id = pred_repr[:, -1:]
        consume_sequence(
            self._predict_model, start_id, act_repr[:, context + timestep, :-1]
        )
        start_id = act_repr[:, context + timestep, -1:]

    return errors

  def _calculate_error(self, prediction, ground_truth):
    return (prediction - ground_truth) ** 2


def consume_sequence(model, start_id, sequence):
  decoding.autoregressive_sample(
      model,
      start_id=start_id,
      eos_id=-1,
      inputs=sequence,
      batch_size=sequence.shape[0],
      max_length=1,
      accelerate=False,
  )
