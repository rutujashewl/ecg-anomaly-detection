"""
app.py
------
Streamlit web app for ECG anomaly detection inference.
User uploads a beat/signal (CSV of raw values), the app predicts
normal vs abnormal using the trained CNN model.

Run locally:  streamlit run app.py
Deployed on Render using the Procfile in this repo.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = "models/cnn_model.keras"
METADATA_PATH = "models/metadata.json"

st.set_page_config(page_title="ECG Anomaly Detector", layout="centered")


@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def predict_beat(model, beat: np.ndarray) -> tuple:
    """Runs the model on a single beat window and returns (label, confidence)."""
    beat_input = beat.reshape(1, -1, 1)
    prob = float(model.predict(beat_input, verbose=0)[0][0])
    label = "Abnormal" if prob >= 0.5 else "Normal"
    confidence = prob if label == "Abnormal" else 1 - prob
    return label, confidence, prob


def main():
    st.title("🫀 ECG Anomaly Detection")
    st.write(
        "Upload a single-beat ECG segment (CSV with one column of raw signal "
        "values) to check whether it looks normal or abnormal."
    )

    model, metadata = load_model_and_metadata()
    expected_length = metadata["beat_window_size"]

    uploaded_file = st.file_uploader("Upload beat CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            beat_df = pd.read_csv(uploaded_file, header=None)
            beat = beat_df.values.flatten().astype(float)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        if len(beat) != expected_length:
            st.warning(
                f"Expected a beat of length {expected_length}, got {len(beat)}. "
                "Results may be unreliable."
            )

        # Plot the signal
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(beat, color="crimson")
        ax.set_title("Uploaded ECG Beat")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        label, confidence, raw_prob = predict_beat(model, beat)

        if label == "Abnormal":
            st.error(f"**Prediction: {label}**  (confidence: {confidence:.1%})")
        else:
            st.success(f"**Prediction: {label}**  (confidence: {confidence:.1%})")

        st.caption(f"Raw model output (probability of abnormal): {raw_prob:.3f}")

    st.divider()
    st.caption(
        "⚠️ This is a portfolio/educational project, not a medical diagnostic tool."
    )


if __name__ == "__main__":
    main()
