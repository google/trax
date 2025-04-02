import gin
import tensorflow as tf


# TODO(lukaszkaiser): find a single more abstract way of text pre-processing.
@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def wmt_preprocess(dataset, training, max_length=-1, max_eval_length=-1):
    """Preprocessing for LM1B: filter out targets exceeding maximum length."""

    def train_right_length(example, target):
        l = tf.maximum(tf.shape(example["inputs"])[0], tf.shape(target)[0])
        return tf.less(l, max_length + 1)

    def eval_right_length(example, target):
        l = tf.maximum(tf.shape(example["inputs"])[0], tf.shape(target)[0])
        return tf.less(l, max_eval_length + 1)

    if max_length > 0 and training:
        dataset = dataset.filter(train_right_length)

    if max_eval_length > 0 and not training:
        dataset = dataset.filter(eval_right_length)

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

