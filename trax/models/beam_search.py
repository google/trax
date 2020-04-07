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
"""Inference methods for autoregressive models (see the Search class)."""
# TODO(kitaev): this file needs style cleanup.

import collections
import functools

import jax
from jax import lax
from jax import numpy as jnp
import numpy as onp

from trax import layers as tl
import trax.math
from trax.math import numpy as np
from trax.shapes import ShapeDtype

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = onp.array(-1.0e7)


# Beam Search
#
# We try to match the logic of the original t2t implementation and the mlperf
# reference tensorflow implementation at:
# https://github.com/mlperf/training/blob/master/translation/tensorflow/transformer/model/beam_search.py
#
# Using JAX we are directly programming with the XLA computation model,
# so here we initialize and update static-sized arrays inside an XLA while loop
# rather than concatenating onto a growing chain of sequences.


def brevity_penalty(alpha, length):
  """Brevity penalty function for beam search penalizing short sequences.

  Args:
    alpha: float: brevity-penalty scaling parameter.
    length: int: length of considered sequence.

  Returns:
    Brevity penalty score as jax scalar.
  """
  return jnp.power(((5.0 + length) / 6.0), alpha)


def top_k(x, k):
  """Select the top k slices from the last dimension."""
  bcast_idxs = jnp.broadcast_to(np.arange(x.shape[-1]), x.shape)
  sorted_vals, sorted_idxs = lax.sort_key_val(x, bcast_idxs)
  # TODO(levskaya): use lax.slice here instead to benefit from XLA optimization
  return sorted_vals[..., -k:], sorted_idxs[..., -k:]


# Beam handling utility functions:


def add_beam_dim(x, beam_size):
  """Creates new beam dimension in non-scalar array and tiles into it."""
  if isinstance(x, int):
    return jnp.array(x)
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  x = jnp.expand_dims(x, axis=1)
  tile_dims = [1] * x.ndim
  tile_dims[1] = beam_size
  return jnp.tile(x, tile_dims)


def flatten_beam_dim(x, batch_size=None):
  """Flattens the first two dimensions of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  if batch_size is not None and x.shape[0] != batch_size:
    assert x.shape[0] % batch_size == 0
    res = x.reshape((batch_size, -1, x.shape[1]) + x.shape[2:])
    res = np.swapaxes(res, 1, 2)
    res = res.reshape(
        (res.shape[0] * res.shape[1] * res.shape[2],) + res.shape[3:])
    return res
  return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unflatten_beam_dim(x, batch_size, beam_size):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  if batch_size * beam_size == x.shape[0]:
    return x.reshape((batch_size, beam_size) + x.shape[1:])
  else:
    assert x.shape[0] % (batch_size * beam_size) == 0
    res = x.reshape((batch_size, beam_size, -1) + x.shape[1:])
    res = np.swapaxes(res, 1, 2)
    res = res.reshape((-1, beam_size) + res.shape[3:])
    return res


def gather_beams(nested, beam_indices, batch_size, new_beam_size):
  """Gathers the beam slices indexed by beam_indices into new beam array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    beam_indices: array of beam_indices
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ beam dimension.

  Returns:
    New pytree with new beam arrays.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  batch_indices = jnp.reshape(
      jnp.arange(batch_size * new_beam_size) // new_beam_size,
      (batch_size, new_beam_size))
  def gather_fn(x):
    """Gather slices for a single tensor."""
    if x.ndim == 0:  # ignore scalars (e.g. cache index)
      return x
    elif x.shape[0] != batch_size:
      assert x.shape[0] % batch_size == 0
      res = x.reshape((batch_size, -1,) + x.shape[1:])
      res = np.swapaxes(res, 1, 2)
      res = res[batch_indices, beam_indices]
      res = np.swapaxes(res, 1, 2)
      res = res.reshape((-1,) + res.shape[2:])
      return res
    else:
      return x[batch_indices, beam_indices]
  return jax.tree_map(gather_fn, nested)


def gather_topk_beams(nested, score_or_log_prob, batch_size, new_beam_size):
  """Gathers the top-k beam slices given by score_or_log_prob array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    score_or_log_prob: [batch_size, old_beam_size] array of values to sort by
      for top-k selection of beam slices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ top-k selected beam dimension

  Returns:
    New pytree with new beam arrays containing top k new_beam_size slices.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  _, topk_indexes = top_k(score_or_log_prob, k=new_beam_size)
  return gather_beams(nested, topk_indexes, batch_size, new_beam_size)


# Beam search state:

BeamState = collections.namedtuple('BeamState', [
    # The position of the decoding loop in the length dimension.
    'cur_index',  # scalar int32: current decoded length index
    # The active sequence log probabilities and finished sequence scores.
    'live_logprobs',  # float32: [batch_size, beam_size]
    'finished_scores',  # float32: [batch_size, beam_size]
    # The current active-beam-searching and finished sequences.
    'live_seqs',  # int32: [batch_size, beam_size, max_decode_len]
    'finished_seqs',  # int32: [batch_size, beam_size, max_decode_len]
    # Records which of the 'finished_seqs' is occupied and not a filler slot.
    'finished_flags',  # bool: [batch_size, beam_size]
    # The current state of the autoregressive decoding caches.
    'cache',  # Any pytree of arrays, e.g. flax attention Cache object
])


def beam_init(batch_size, beam_size, max_decode_len, cache, start_tokens=None):
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (beam_size - 1)),
      [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * NEG_INF
  if start_tokens is None:
    live_seqs0 = jnp.zeros(
        (batch_size, beam_size, max_decode_len), jnp.int32)
  else:
    live_seqs0 = add_beam_dim(
        np.pad(start_tokens[:, None],
               ((0, 0), (0, max_decode_len - 1)), mode='constant'),
        beam_size)
  finished_seqs0 = jnp.zeros(
      (batch_size, beam_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  beam_cache0 = jax.tree_map(lambda x: add_beam_dim(x, beam_size), cache)
  return BeamState(cur_index=cur_index0,
                   live_logprobs=live_logprobs0,
                   finished_scores=finished_scores0,
                   live_seqs=live_seqs0,
                   finished_seqs=finished_seqs0,
                   finished_flags=finished_flags0,
                   cache=beam_cache0)


# Beam search routine:


def beam_search(batch_size,
                cache,
                tokens_to_logits,
                max_decode_len,
                beam_size=4,
                alpha=0.6,
                eos_token=-1,
                start_tokens=None):
  """Beam search for transformer machine translation.

  Args:
    batch_size: int: batch size for decoding
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    max_decode_len: int: maximum length of decoded translations.
    beam_size: int: number of beams to use in beam search.
    alpha: float: scaling factor for brevity penalty.
    eos_token: int: end-of-sentence token for target vocabulary.
    start_tokens: (optional) array: [batch_size] int32 start tokens

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  # We liberally annotate shape information for clarity below.

  end_marker = jnp.array(eos_token)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size,
                                     beam_size,
                                     max_decode_len,
                                     cache,
                                     start_tokens)

  def beam_search_loop_cond_fn(state):
    """Beam search loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = (state.cur_index <= max_decode_len)

    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_decode_len)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores = jnp.min(
        state.finished_scores, axis=1, keepdims=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = jnp.where(
        state.finished_flags, worst_finished_scores, NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = jnp.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)

  def beam_search_loop_body_fn(state):
    """Beam search loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the beam dimension into batch
    # dimension for feeding into the model.
    # --> [batch * beam, 1]
    flat_ids = flatten_beam_dim(lax.dynamic_slice(
        state.live_seqs,
        (0, 0, state.cur_index),
        (batch_size, beam_size, 1)))
    # Flatten beam dimension into batch to be compatible with model.
    # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
    flat_cache = jax.tree_map(
        lambda x: flatten_beam_dim(x, batch_size), state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * beam, vocab]
    flat_logits, new_flat_cache = tokens_to_logits(
        flat_ids, flat_cache, jax.random.PRNGKey(state.cur_index))

    # unflatten beam dimension
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    # Unflatten beam dimension in attention cache arrays
    # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
    new_cache = jax.tree_map(
        lambda x: unflatten_beam_dim(x, batch_size, beam_size), new_flat_cache)

    # Gather log probabilities from logits
    candidate_log_probs = jax.nn.log_softmax(logits)
    # Add new logprobs to existing prefix logprobs.
    # --> [batch, beam, vocab]
    log_probs = (candidate_log_probs +
                 jnp.expand_dims(state.live_logprobs, axis=2))

    # We'll need the vocab size, gather it from the log probability dimension.
    vocab_size = log_probs.shape[2]

    # Each item in batch has beam_size * vocab_size candidate sequences.
    # For each item, get the top 2*k candidates with the highest log-
    # probabilities. We gather the top 2*K beams here so that even if the best
    # K sequences reach EOS simultaneously, we have another K sequences
    # remaining to continue the live beam search.
    beams_to_keep = 2 * beam_size
    # Flatten beam and vocab dimensions.
    flat_log_probs = log_probs.reshape((batch_size, beam_size * vocab_size))
    # Gather the top 2*K scores from _all_ beams.
    # --> [batch, 2*beams], [batch, 2*beams]
    topk_log_probs, topk_indices = top_k(flat_log_probs, k=beams_to_keep)
    # Recover the beam index by floor division.
    topk_beam_indices = topk_indices // vocab_size
    # Gather 2*k top beams and beam-associated caches.
    # --> [batch, 2*beams, length], {[batch, 2*beams, ...], ...}
    topk_seq, new_cache = gather_beams([state.live_seqs, new_cache],
                                       topk_beam_indices,
                                       batch_size, beams_to_keep)

    # Append the most probable 2*K token IDs to the top 2*K sequences
    # Recover token id by modulo division and expand Id array for broadcasting.
    # --> [batch, 2*beams, 1]
    topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(
        topk_seq, topk_ids, (0, 0, state.cur_index + 1))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index + 1] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # --> [batch, beams, length], [batch, beams], {[batch, beams, ...], ...}
    top_alive_seq, top_alive_log_probs, top_alive_cache = gather_topk_beams(
        [topk_seq, new_log_probs, new_cache],
        new_log_probs,
        batch_size, beam_size)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq], axis=1)
    finished_scores = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], axis=1)
    finished_flags = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], axis=1)
    # --> [batch, beams, length], [batch, beams], [batch, beams]
    top_finished_seq, top_finished_scores, top_finished_flags = (
        gather_topk_beams([finished_seqs, finished_scores, finished_flags],
                          finished_scores, batch_size, beam_size))

    return BeamState(cur_index=state.cur_index + 1,
                     live_logprobs=top_alive_log_probs,
                     finished_scores=top_finished_scores,
                     live_seqs=top_alive_seq,
                     finished_seqs=top_finished_seq,
                     finished_flags=top_finished_flags,
                     cache=top_alive_cache)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn,
                               beam_search_loop_body_fn,
                               beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(none_finished[:, None, None],
                            final_state.finished_seqs,
                            final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(none_finished[:, None],
                              final_state.finished_scores,
                              final_state.live_logprobs)

  return finished_seqs, finished_scores


class Search:
  """Provides inference for autoregressive models."""

  def __init__(self, model, weights, max_decode_len, beam_size=1, temperature=0,
               alpha=0.0, eos_id=None):
    """Construct an inference wrapper for an autoregressive model.

    The default behavior is to do greedy decoding:
      s = Search(model, weights, max_decode_len, eos_id=eos_id)
    Passing a temperature parameter will switch to sampling:
      s = Search(model, weights, max_decode_len, temperature=1, eos_id=eos_id)
    Passing a beam_size parameter will switch to beam search. For machine
    translation with Transformer models, Vaswani et al. (2017) recommend a beam
    size of 4 and length normalization with alpha=0.6.
      s = Search(model, weights, max_decode_len, beam_size=4, alpha=0.6,
                 eos_id=eos_id)

    After constructing the class, see Search.decode for how to decode a batch
    of examples.

    Args:
      model: function to construct a model (e.g. trax.models.Reformer)
      weights: model weights
      max_decode_len: maximum length to decode
      beam_size: beam size, for beam search
      temperature: temperature parameter for sampling; set to nonzero to switch
        from greedy/beam-search behavior to sampling.
      alpha: length penalty alpha coefficient for beam search.
      eos_id: end-of-sentence token for target vocabulary.
    """
    # TODO(kitaev): k and p parameters for top-k and nucleus sampling.
    self.model = model
    self.model_infer = model(mode='predict')
    # Weights are stored on device, but not replicated.
    self.model_weights = jax.tree_map(jax.jit(lambda x: x), weights)

    self.sample = (temperature != 0)
    self.temperature = temperature

    if self.sample and beam_size > 1:
      # TODO(kitaev): perform stochastic beam search in this case
      # (https://arxiv.org/abs/1903.06059)
      raise ValueError('beam_size parameter is not supported when sampling')

    is_cache = [isinstance(l, tl.Cache) for l in self.model_infer.sublayers]
    if any(is_cache):
      assert sum([int(x) for x in is_cache]) == 1, (
          'At most one usage of tl.Cache currently supported')
      self.encoder_idx = is_cache.index(True) + 1
    else:
      self.encoder_idx = None

    beam_search_partial = functools.partial(
        self._unreplicated_beam_search,
        beam_size=beam_size, alpha=alpha,
        eos_token=eos_id if eos_id is not None else -1,
        max_decode_len=max_decode_len + 1)  # Add 1 to account for start token.

    if trax.math.device_count() == 1:
      self._jit_beam_search = jax.jit(beam_search_partial, static_argnums=(2,))
    else:
      self._jit_beam_search = jax.pmap(beam_search_partial, axis_name='batch',
                                       static_broadcasted_argnums=(2,))

  def _get_initial_state(self, inputs, targets_prefix, batch_size):
    """Get initial state for beam search."""
    if targets_prefix is None:
      prompt = np.zeros((batch_size, 1), dtype=np.int32)
    else:
      prompt = np.pad(
          targets_prefix[:, :-1], ((0, 0), (1, 0)), mode='constant')

    # Get state prior to running the encoder or incorporating targets_prefix
    if inputs is None:
      signature = ShapeDtype((batch_size, 1), prompt.dtype)
    else:
      signature = (ShapeDtype(inputs.shape, inputs.dtype),
                   ShapeDtype((batch_size, 1), prompt.dtype))
    # Trax's model.init is stateful as opposed to functional. Calling it on an
    # already-existing model instance doesn't work.
    # TODO(lukaszkaiser): add purely functional init to Trax.
    _, initial_state = self.model(mode='predict').init(signature)

    # Incorporate encoder and prompt into state
    _, prompted_state = self.model_infer.pure_fn(
        prompt if inputs is None else (inputs, prompt),
        self.model_weights,
        initial_state,
        jax.random.PRNGKey(0))
    state_structure = jax.tree_structure(prompted_state)

    if targets_prefix is not None:
      initial_state = prompted_state
    elif self.encoder_idx is not None:
      initial_state = (tuple(prompted_state[:self.encoder_idx])
                       + tuple(initial_state[self.encoder_idx:]))

    # Fix tree structure of the state (there's a tuple vs. list mismatch)
    initial_state = jax.tree_unflatten(
        state_structure, jax.tree_leaves(initial_state))

    return initial_state

  def _unreplicated_beam_search(self, inputs, targets_prefix, batch_size, dummy,
                                beam_size, alpha, eos_token, max_decode_len):
    """Beam search, on one device."""
    del dummy  # Used to signal pmap axis size in the fully unconditional case
    def tokens_to_logits(flat_ids, flat_cache, rng):
      """Autoregressive decoding step: map from previous tokens to logits."""
      rng, subrng = jax.random.split(rng)
      flat_logits, new_flat_cache = self.model_infer.pure_fn(
          flat_ids if inputs is None else (inputs, flat_ids),
          self.model_weights,
          flat_cache,
          rng)
      if isinstance(flat_logits, (list, tuple)):
        flat_logits = flat_logits[0]  # Keep only logits from output stack
      flat_logits = flat_logits[:, 0, :]  # Squeeze along seqlen dim

      if self.sample:
        flat_logits = (flat_logits / self.temperature
                       + jax.random.gumbel(subrng, flat_logits.shape))

      return flat_logits, new_flat_cache
    return beam_search(
        batch_size,
        self._get_initial_state(inputs, targets_prefix, batch_size),
        tokens_to_logits,
        max_decode_len,
        start_tokens=None if targets_prefix is None else targets_prefix[:, -1],
        beam_size=beam_size, alpha=alpha, eos_token=eos_token)

  def decode(self, inputs=None, targets_prefix=None, batch_size=None):
    """Performs decoding for a batch of examples.

    Args:
      inputs: [batch_size, encoder_input_length] int32 numpy array: Inputs to
        the encoder portion of the model. If the model does not have an encoder,
        leave this set to None.
      targets_prefix: [batch_size, target_prefix_length] int32 numpy array:
        Optional prefix to initialize the decoder with. The start token should
        never be included in the prefix. Note that all examples in the batch
        must use the same prefix length.
      batch_size: If both inputs and targets_prefix are None, the batch_size
        argument is required and will determine the batch size for decoding.
        Otherwise, this argument serves as an optional hint for the batch size
        that the inputs should be padded out to before running inference. The
        XLA computation for inference needs to be re-jitted every time a new
        batch size is encountered, so passing a constant batch_size argument can
        speed up inference by avoiding recompilation.

    Returns:
      Tuple of:
        [batch_size, beam_size, max_decode_len] top-scoring sequences
        [batch_size, beam_size] beam-search scores.
      The highest-scoring sequence will be at index -1 along the beam_size axis.
    """
    n_devices = trax.math.device_count()
    if inputs is not None and targets_prefix is not None:
      pad_to = batch_size
      batch_size = inputs.shape[0]
      assert targets_prefix.shape[0] == batch_size
    elif inputs is not None:
      pad_to = batch_size
      batch_size = inputs.shape[0]
    elif targets_prefix is not None:
      pad_to = batch_size
      batch_size = targets_prefix.shape[0]
    else:
      pad_to = None

    if pad_to is None:
      pad_amount = (n_devices - (batch_size % n_devices)) % n_devices
    else:
      assert pad_to % n_devices == 0, (
          'When specifying batch_size for the purposes of padding,'
          'batch_size must be divisible by the number of devices.')
      pad_amount = pad_to - batch_size
      assert pad_amount >= 0

    if inputs is not None:
      if pad_amount:
        inputs = onp.concatenate([inputs] + [inputs[0:1]] * pad_amount, 0)
      inputs = inputs.reshape((n_devices, -1) + inputs.shape[1:])
    if targets_prefix is not None:
      if pad_amount:
        targets_prefix = onp.concatenate(
            [targets_prefix] + [targets_prefix[0:1]] * pad_amount, 0)
      targets_prefix = targets_prefix.reshape(
          (n_devices, -1) + targets_prefix.shape[1:])

    seqs, scores = self._jit_beam_search(
        inputs, targets_prefix, (batch_size + pad_amount) // n_devices,
        dummy=np.zeros(n_devices))
    seqs = onp.asarray(seqs)
    scores = onp.asarray(scores)
    seqs = seqs.reshape((-1,) + seqs.shape[2:])
    scores = scores.reshape((-1,) + scores.shape[2:])
    seqs = seqs[:, :, 1:]  # Strip start token
    if pad_amount:
      seqs = seqs[:batch_size]
      scores = scores[:batch_size]
    return seqs, scores
