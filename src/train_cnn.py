import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CNN_MODEL_PATH,
    CNN_HISTORY_FIGURE_PATH,
    MODELS_DIR,
    FIGURES_DIR,
)
from src.preprocessing import get_rescaling_layer


def build_cnn_baseline(num_classes: int) -> tf.keras.Model:
    """
    Build CNN baseline model using Keras Sequential API.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

            get_rescaling_layer(),

            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_baseline",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_cnn_callbacks():
    """
    Create callbacks for CNN training.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=CNN_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [early_stopping, model_checkpoint]


def plot_training_history(history, save_path=CNN_HISTORY_FIGURE_PATH):
    """
    Plot training and validation accuracy/loss.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("CNN Baseline Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("CNN Baseline Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Training history figure saved to: {save_path}")


def train_cnn_model(train_ds, val_ds, num_classes: int, epochs: int = 20):
    """
    Build and train CNN baseline.
    """

    model = build_cnn_baseline(num_classes)
    callbacks = get_cnn_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    plot_training_history(history)

    return model, history


if __name__ == "__main__":
    from src.dataset import load_train_val_test_datasets

    train_ds, val_ds, test_ds, class_names = load_train_val_test_datasets()
    num_classes = len(class_names)

    model, history = train_cnn_model(
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        epochs=20,
    )

    print("CNN training completed.")
    print(f"Best model saved to: {CNN_MODEL_PATH}")