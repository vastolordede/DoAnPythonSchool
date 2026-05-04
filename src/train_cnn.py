# src/train_cnn.py
import tensorflow as tf

from src.config import IMG_HEIGHT, IMG_WIDTH
from src.preprocessing import get_rescaling_layer


def build_cnn_baseline(num_classes: int) -> tf.keras.Model:
    """
    Build a simple CNN baseline model using Keras Sequential API.

    Input:
        num_classes: number of traffic sign classes

    Output:
        compiled CNN model
    """

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

            get_rescaling_layer(),

            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_baseline",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    from src.dataset import load_train_val_test_datasets

    train_ds, val_ds, test_ds, class_names = load_train_val_test_datasets()

    num_classes = len(class_names)

    model = build_cnn_baseline(num_classes)

    print("Class names:")
    print(class_names)

    print("\nNumber of classes:", num_classes)

    print("\nCNN Baseline Summary:")
    model.summary()