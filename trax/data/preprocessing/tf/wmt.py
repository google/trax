
import gin
import numpy as np
import tensorflow as tf


# TODO(lukaszkaiser): find a single more abstract way of text pre-processing.
@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def wmt_preprocess(dataset, training, max_length=-1, max_eval_length=-1, tokenizer=None):
    """Preprocessing for LM1B: filter out targets exceeding maximum length."""
    def train_right_length(example):
        input_length = tf.strings.length(example["inputs"])
        target_length = tf.strings.length(example["targets"])
        max_tensor_length = tf.maximum(input_length, target_length)
        return tf.less(max_tensor_length, max_length + 1)

    def eval_right_length(example):
        input_length = tf.strings.length(example["inputs"])
        target_length = tf.strings.length(example["targets"])
        max_tensor_length = tf.maximum(input_length, target_length)
        return tf.less(max_tensor_length, max_eval_length + 1)

    dataset = dataset.map(lambda x, y: x)

    if max_length > 0 and training:
        dataset = dataset.filter(train_right_length)

    if max_eval_length > 0 and not training:
        dataset = dataset.filter(eval_right_length)

    def tokenize_example(encoder, example):
        """Tokenize examples using a SubwordTextEncoder.

        Args:
            encoder: A trax.data.encoder.encoder.SubwordTextEncoder instance
            example: A dictionary with 'inputs' and 'targets' keys containing text tensors

        Returns:
            A dictionary with tokenized 'inputs' and 'targets'
        """
        def _encode_text(text_tensor):
            # Convert tensor to string
            if hasattr(text_tensor, "numpy"):
                # Handle TensorFlow tensor
                text = text_tensor.numpy()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
            else:
                # Already string or bytes
                text = text_tensor
                if isinstance(text, bytes):
                    text = text.decode("utf-8")

            # Use the encoder's encode method directly
            return np.array(encoder.encode(text), dtype=np.int64)

        # Use tf.py_function to handle the Python code within TensorFlow graph
        encoded_inputs = tf.py_function(_encode_text, [example["inputs"]], tf.int64)

        encoded_targets = tf.py_function(_encode_text, [example["targets"]], tf.int64)

        # Update the example with encoded data
        return {"inputs": encoded_inputs, "targets": encoded_targets}, encoded_targets


    # Apply to your dataset
    dataset = dataset.map(
        lambda example: tokenize_example(tokenizer, example),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def wmt_concat_preprocess(dataset, training, max_length=-1, max_eval_length=-1):
    """Preprocessing for WMT: filter exceeding maximum length and concatenate."""
    dataset = wmt_preprocess(dataset, training, max_length, max_eval_length)

    def concat_and_add_mask(features, targets):
        inp = features["inputs"]
        pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
        concat = tf.concat([inp, pad, targets], axis=0)
        mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
        features["inputs"] = concat
        features["mask"] = mask
        return features, concat

    dataset = dataset.map(concat_and_add_mask)
    return dataset

