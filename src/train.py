"""
train.py
--------
Trains two models on the ECG dataset:
  1. Baseline: Random Forest on the BEST selected features (fast, interpretable)
  2. Deep Learning: 1D-CNN on the raw beat signal (usually more accurate)

Handles class imbalance (abnormal beats are much rarer than normal ones)
using SMOTE for the baseline model and class weights for the CNN.

Beginner note: we try the simple model first (baseline) so we have something
to compare the more complex deep learning model against. If the CNN isn't
meaningfully better, the simple model is usually the better real-world choice.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from features import build_feature_dataframe, select_best_features

MODELS_DIR = "models"
RANDOM_STATE = 42


def patient_wise_split(patient_ids: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    Splits data so that ALL beats from a given patient go entirely into
    either train or test - never both. This prevents the model from
    'memorizing' a patient's signal style and gives a realistic estimate
    of how it'll perform on a brand new patient.

    Returns boolean masks (train_mask, test_mask) aligned with the input arrays.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X=np.zeros(len(y)), y=y, groups=patient_ids))

    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    train_patients = set(patient_ids[train_mask])
    test_patients = set(patient_ids[test_mask])
    print(f"Train patients ({len(train_patients)}): {sorted(train_patients)}")
    print(f"Test patients  ({len(test_patients)}): {sorted(test_patients)}")
    assert train_patients.isdisjoint(test_patients), "Patient leakage detected between train/test!"

    return train_mask, test_mask


def train_baseline(df: pd.DataFrame, best_features: list, train_mask: np.ndarray, test_mask: np.ndarray):
    """Trains a Random Forest on the selected top features, using a
    patient-wise train/test split (passed in as boolean masks)."""
    X = df[best_features]
    y = df["label"]

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Balance the training set only (never balance test data - it must stay real)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_bal, y_train_bal)

    preds = model.predict(X_test)
    print("\n=== Baseline (Random Forest) Results ===")
    print(classification_report(y_test, preds, target_names=["Normal", "Abnormal"]))

    return model, (X_test, y_test)


def build_cnn(input_length: int):
    """Defines a simple 1D-CNN for raw ECG beat classification."""
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(32, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return model


def train_cnn(X_beats: np.ndarray, y_labels: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray):
    """Trains the CNN directly on raw beat windows, using class weights
    (instead of SMOTE) to handle imbalance - this is the standard approach
    for raw signal/image-like deep learning inputs. Uses the same
    patient-wise split as the baseline model for a fair comparison."""
    import tensorflow as tf

    X_train, X_test = X_beats[train_mask], X_beats[test_mask]
    y_train, y_test = y_labels[train_mask], y_labels[test_mask]

    # Reshape for CNN input: (samples, timesteps, channels)
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # Class weights give more importance to the rare (abnormal) class
    n_normal = (y_train == 0).sum()
    n_abnormal = (y_train == 1).sum()
    class_weight = {
        0: 1.0,
        1: n_normal / max(n_abnormal, 1),
    }

    model = build_cnn(input_length=X_train_cnn.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train_cnn, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    print("\n=== CNN Results ===")
    loss, acc, auc = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f} | Test AUC: {auc:.3f}")

    return model, (X_test_cnn, y_test)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = np.load("data/processed/beats.npz")
    X_beats, y_labels, patient_ids = data["X"], data["y"], data["patient_ids"]

    # Patient-wise split - computed ONCE, reused for both models so results
    # are directly comparable and neither model sees a test patient's data.
    train_mask, test_mask = patient_wise_split(patient_ids, y_labels, test_size=0.2)

    df = build_feature_dataframe(X_beats, y_labels)
    best_features, _ = select_best_features(df, k=8)

    # --- Baseline model ---
    rf_model, _ = train_baseline(df, best_features, train_mask, test_mask)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "baseline_rf.pkl"))

    # --- Deep learning model ---
    cnn_model, _ = train_cnn(X_beats, y_labels, train_mask, test_mask)
    cnn_model.save(os.path.join(MODELS_DIR, "cnn_model.keras"))

    # Save metadata so app.py / evaluate.py know what the model expects
    metadata = {
        "best_features": best_features,
        "beat_window_size": X_beats.shape[1],
        "sampling_rate": 360,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: {MODELS_DIR}/baseline_rf.pkl, {MODELS_DIR}/cnn_model.keras, {MODELS_DIR}/metadata.json")


if __name__ == "__main__":
    main()
