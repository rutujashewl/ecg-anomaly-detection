"""
evaluate.py
-----------
Loads the trained CNN model and test data, then reports proper metrics
for an imbalanced medical classification problem (accuracy alone is
misleading here - recall matters most, since missing an abnormal beat
is worse than a false alarm).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score
)
import tensorflow as tf
from train import patient_wise_split

MODELS_DIR = "models"
REPORTS_DIR = "reports"
RANDOM_STATE = 42


def load_test_split():
    """Rebuilds the SAME patient-wise test split used during training,
    so evaluation reflects performance on patients the model never saw."""
    data = np.load("data/processed/beats.npz")
    X_beats, y_labels, patient_ids = data["X"], data["y"], data["patient_ids"]

    _, test_mask = patient_wise_split(patient_ids, y_labels, test_size=0.2)
    X_test, y_test = X_beats[test_mask], y_labels[test_mask]

    return X_test[..., np.newaxis], y_test


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_model.keras"))
    X_test, y_test = load_test_split()

    y_probs = model.predict(X_test).ravel()
    y_preds = (y_probs >= 0.5).astype(int)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_preds, target_names=["Normal", "Abnormal"]))

    cm = confusion_matrix(y_test, y_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    auc = roc_auc_score(y_test, y_probs)
    print(f"\nROC-AUC: {auc:.3f}")

    # --- Save confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Abnormal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # --- Save ROC curve plot ---
    RocCurveDisplay.from_predictions(y_test, y_probs)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
    plt.close()

    print(f"\nPlots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
