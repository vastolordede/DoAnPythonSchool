import tensorflow as tf
import numpy as np

from src.config import CNN_MODEL_PATH, CNN_FINAL_RESULT_PATH, REPORTS_DIR


def load_cnn_model(model_path=CNN_MODEL_PATH):
    """
    Load the best CNN baseline model saved by ModelCheckpoint.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the CNN model first in Week 2."
        )

    model = tf.keras.models.load_model(model_path)

    return model


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.
    """

    test_loss, test_accuracy = model.evaluate(test_ds)

    return test_loss, test_accuracy

def collect_predictions(model, test_ds):
    """
    Collect true labels and predicted labels from test dataset.
    """
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    return np.array(y_true), np.array(y_pred)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def save_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate and save classification report.
    """
    from src.config import REPORTS_DIR
    if save_path is None:
        save_path = REPORTS_DIR / "classification_report.txt"

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to: {save_path}")

    return report

def save_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and save confusion matrix.
    """
    from src.config import FIGURES_DIR
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("Confusion Matrix — CNN Baseline")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix saved to: {save_path}")

def save_cnn_final_result(test_loss, test_accuracy, save_path=CNN_FINAL_RESULT_PATH):
    """
    Save final CNN baseline result to text file.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    content = f"""CNN Baseline Final Result

Model:
- CNN Baseline using TensorFlow/Keras Sequential API

Saved model:
- models/cnn_baseline.keras

Final evaluation:
- Test loss: {test_loss:.4f}
- Test accuracy: {test_accuracy:.4f}
- Test accuracy percent: {test_accuracy * 100:.2f}%

Note:
- The model was saved using ModelCheckpoint during Week 2.
- This result is used as the CNN baseline for comparison with transfer learning models.
"""

    save_path.write_text(content, encoding="utf-8")

    print(f"Final result saved to: {save_path}")


def evaluate_cnn_final(test_ds):
    """
    Load best CNN model, evaluate on test dataset, and save final result.
    """

    model = load_cnn_model()
    test_loss, test_accuracy = evaluate_model(model, test_ds)
    save_cnn_final_result(test_loss, test_accuracy)

    return model, test_loss, test_accuracy