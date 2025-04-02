import gin
import tensorflow as tf


#  Makes the function accessible in gin configs, even with all args denylisted.
@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def cifar10_no_augmentation_preprocess(dataset, training):
    del training

    def cast_image(features, targets):
        features["image"] = tf.cast(features["image"], tf.float32) / 255.0
        return features, targets

    dataset = dataset.map(cast_image)
    return dataset


def _cifar_augment_image(image):
    """Image augmentation suitable for CIFAR-10/100.

    As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).

    Args:
      image: a Tensor.

    Returns:
      Tensor of the same shape as image.
    """
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image


# Makes the function accessible in gin configs, even with all args denylisted.
@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def cifar10_augmentation_preprocess(dataset, training):
    """Preprocessing for cifar10 with augmentation (see below)."""

    def augment(features, targets):
        features["image"] = _cifar_augment_image(features["image"])
        return features, targets

    def cast_image(features, targets):
        features["image"] = tf.cast(features["image"], tf.float32) / 255.0
        return features, targets

    if training:
        dataset = dataset.map(augment)
    dataset = dataset.map(cast_image)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def cifar10_augmentation_flatten_preprocess(
    dataset, training, predict_image_train_weight=0.01
):
    """Preprocessing for cifar10 that flattens it and appends targets."""

    def augment(features, targets):
        features["image"] = _cifar_augment_image(features["image"])
        return features, targets

    def flatten_image(features, targets):
        """Flatten the image."""
        img = features["image"]
        flat = tf.cast(tf.reshape(img, [-1]), tf.int64)
        tgt = tf.expand_dims(targets, axis=0)
        flat_with_target = tf.concat([flat, tgt], axis=0)
        new_features = {}
        new_features["image"] = flat_with_target
        predict_image_weight = predict_image_train_weight if training else 0.0
        mask_begin = tf.ones_like(flat)
        mask_begin = tf.cast(mask_begin, tf.float32) * predict_image_weight
        mask_end = tf.cast(tf.ones_like(tgt), tf.float32)
        new_features["mask"] = tf.concat([mask_begin, mask_end], axis=0)
        return new_features, flat_with_target

    if training:
        dataset = dataset.map(augment)
    dataset = dataset.map(flatten_image)

    return dataset

