import os

from typing import Any, Mapping, Optional, Sequence, Union

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl.testing import absltest
from t5.data import preprocessors as t5_processors

from trax.data.loader.tf import base as ds

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.normpath(os.path.join(pkg_dir, "../../resources/data/testdata"))
_CONFIG_DIR = os.path.normpath(os.path.join(pkg_dir, "../../resources/supervised/configs/"))
_SUPERVISED_TESTDATA = os.path.normpath(os.path.join(pkg_dir, "../../resources/supervised/testdata"))

# _ProxyTest is required because py2 does not allow instantiating
# absltest.TestCase directly.
class _ProxyTest(absltest.TestCase):
  """Instance of TestCase to reuse methods for testing."""

  maxDiff = None

  def runTest(self):
    pass


_pyunit_proxy = _ProxyTest()

def _test_dataset_ints(inp_lengths, tgt_lengths):
    """Create a test dataset of int64 tensors of given shapes."""

    def generator():
        for inp_len, tgt_len in zip(inp_lengths, tgt_lengths):
            inp = np.ones([inp_len], dtype=np.int64)
            tgt = np.ones([tgt_len], dtype=np.int64)
            yield {"inputs": inp, "targets": tgt}

    types = {"inputs": tf.int64, "targets": tf.int64}
    shapes = {"inputs": tf.TensorShape([None]), "targets": tf.TensorShape([None])}
    return tf.data.Dataset.from_generator(
        generator, output_types=types, output_shapes=shapes
    )


def _load_dataset(name, split="train"):
    return tfds.load(name=name, split=split, data_dir=_TESTDATA, shuffle_files=False)


def _c4_dataset(split="train"):
    return _load_dataset("c4:2.3.0", split=split)


def _spm_path():
    return os.path.join(_TESTDATA, "sentencepiece.model")


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
    gin.bind_parameter(
        "unsupervised_preprocessors.preprocessors",
        [
            ds._PREPROCESSOR_REGISTRY["select_random_chunk_t5"],
            ds._PREPROCESSOR_REGISTRY["split_tokens_t5"],
            ds._PREPROCESSOR_REGISTRY["denoise_t5"],
        ],
    )

    # select_random_chunk
    gin.bind_parameter("select_random_chunk.feature_key", "targets")
    gin.bind_parameter("select_random_chunk.max_length", max_input_length)

    # reduce_concat_tokens
    gin.bind_parameter("random_spans_helper.extra_tokens_per_span_inputs", 1)
    gin.bind_parameter("random_spans_helper.extra_tokens_per_span_targets", 1)
    gin.bind_parameter("random_spans_helper.inputs_length", max_input_length)
    gin.bind_parameter("random_spans_helper.mean_noise_span_length", 3.0)
    gin.bind_parameter("random_spans_helper.noise_density", noise_density)

    # split_tokens
    gin.bind_parameter(
        "split_tokens.max_tokens_per_segment",
        t5_processors.random_spans_tokens_length(),
    )

    # denoise
    gin.bind_parameter("denoise.inputs_fn", t5_processors.noise_span_to_unique_sentinel)
    gin.bind_parameter("denoise.noise_density", noise_density)
    gin.bind_parameter("denoise.noise_mask_fn", t5_processors.random_spans_noise_mask)
    gin.bind_parameter(
        "denoise.targets_fn", t5_processors.nonnoise_span_to_unique_sentinel
    )


def _maybe_as_bytes(v):
  if isinstance(v, list):
    return [_maybe_as_bytes(x) for x in v]
  if isinstance(v, str):
    return tf.compat.as_bytes(v)
  return v

def assert_dataset(
    dataset: tf.data.Dataset,
    expected: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    expected_dtypes: Optional[Mapping[str, tf.DType]] = None,
    rtol=1e-7,
    atol=0,
):
  """Tests whether the entire dataset == expected or [expected].

  Args:
    dataset: a tf.data dataset
    expected: either a single example, or a list of examples. Each example is a
      dictionary.
    expected_dtypes: an optional mapping from feature key to expected dtype.
    rtol: the relative tolerance.
    atol: the absolute tolerance.
  """

  if not isinstance(expected, list):
    expected = [expected]
  actual = list(tfds.as_numpy(dataset))
  _pyunit_proxy.assertEqual(len(actual), len(expected))

  def _compare_dict(actual_dict, expected_dict):
    _pyunit_proxy.assertEqual(
        set(actual_dict.keys()), set(expected_dict.keys())
    )
    for key, actual_value in actual_dict.items():
      if isinstance(actual_value, dict):
        _compare_dict(actual_value, expected_dict[key])
      elif isinstance(actual_value, tf.RaggedTensor) or isinstance(
          actual_value, tf.compat.v1.ragged.RaggedTensorValue
      ):
        actual_value = actual_value.to_list()
        np.testing.assert_array_equal(
            np.array(actual_value, dtype=object),
            np.array(_maybe_as_bytes(expected_dict[key]), dtype=object),
            key,
        )
      elif (
          isinstance(actual_value, np.floating)
          or isinstance(actual_value, np.ndarray)
          and np.issubdtype(actual_value.dtype, np.floating)
      ):
        np.testing.assert_allclose(
            actual_value, expected_dict[key], err_msg=key, rtol=rtol, atol=atol
        )
      else:
        np.testing.assert_array_equal(
            actual_value, _maybe_as_bytes(expected_dict[key]), key
        )

  for actual_ex, expected_ex in zip(actual, expected):
    _compare_dict(actual_ex, expected_ex)

  if expected_dtypes:
    actual_dtypes = {k: dataset.element_spec[k].dtype for k in expected_dtypes}
    _pyunit_proxy.assertDictEqual(expected_dtypes, actual_dtypes)
