# src/preprocessing.py
import tensorflow as tf


def get_rescaling_layer():
    """
    Normalize pixel values from [0, 255] to [0, 1].
    """

    return tf.keras.layers.Rescaling(1.0 / 255)


def get_data_augmentation_layer():
    """
    Basic data augmentation for CNN baseline.
    This is optional in Week 1, but useful for reducing overfitting later.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )