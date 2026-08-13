"""
app.py
------
Streamlit web app for ECG anomaly detection inference, with patient
database integration - patients can be registered, ECG results are
saved to their history, and past records can be reviewed.

IMPORTANT: This is a portfolio/educational project. Use synthetic/fake
demo patient data only - do not enter real patient information.

Run locally:  streamlit run app.py
Deployed on Render using the Procfile in this repo.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import database as db

MODEL_PATH = "models/cnn_model.keras"
METADATA_PATH = "models/metadata.json"

st.set_page_config(page_title="ECG Anomaly Detector", layout="centered")
db.init_db()


@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def predict_beat(model, beat: np.ndarray) -> tuple:
    """Runs the model on a single beat window and returns (label, confidence, raw_prob)."""
    beat_input = beat.reshape(1, -1, 1)
    prob = float(model.predict(beat_input, verbose=0)[0][0])
    label = "Abnormal" if prob >= 0.5 else "Normal"
    confidence = prob if label == "Abnormal" else 1 - prob
    return label, confidence, prob


def patient_registration_form():
    """Sidebar form to register a new patient."""
    st.sidebar.subheader("➕ Register New Patient")
    st.sidebar.caption("⚠️ Demo data only - do not enter real patient info.")

    with st.sidebar.form("new_patient_form", clear_on_submit=True):
        name = st.text_input("Full Name*")
        col1, col2 = st.columns(2)
        age = col1.number_input("Age*", min_value=0, max_value=120, step=1)
        gender = col2.selectbox("Gender*", ["Male", "Female", "Other"])

        contact = st.text_input("Contact Number")
        blood_group = st.selectbox(
            "Blood Group", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        )

        col3, col4 = st.columns(2)
        height = col3.number_input("Height (cm)", min_value=0.0, step=0.5)
        weight = col4.number_input("Weight (kg)", min_value=0.0, step=0.5)

        conditions = st.text_area("Existing Medical Conditions", height=68)
        allergies = st.text_area("Allergies", height=68)
        medications = st.text_area("Current Medications", height=68)

        submitted = st.form_submit_button("Register Patient")

        if submitted:
            if not name or not age:
                st.sidebar.error("Name and Age are required.")
            else:
                patient_id = db.add_patient(
                    name=name, age=int(age), gender=gender,
                    contact_number=contact, blood_group=blood_group,
                    height_cm=height or None, weight_kg=weight or None,
                    existing_conditions=conditions, allergies=allergies,
                    current_medications=medications,
                )
                st.sidebar.success(f"Registered '{name}' (ID: {patient_id})")
                st.rerun()


def patient_selector():
    """Sidebar dropdown to select an existing patient. Returns patient dict or None."""
    patients = db.get_all_patients()

    if not patients:
        st.sidebar.info("No patients registered yet. Add one below.")
        return None

    options = {f"{p['name']} (ID: {p['patient_id']}, Age {p['age']})": p for p in patients}
    selected_label = st.sidebar.selectbox("Select Patient", list(options.keys()))
    return options[selected_label]


def show_patient_summary(patient: dict):
    """Displays a patient's personal + health details."""
    bmi = db.calculate_bmi(patient.get("height_cm"), patient.get("weight_kg"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {patient['name']}")
        st.markdown(f"**Age:** {patient['age']} | **Gender:** {patient['gender']}")
        st.markdown(f"**Blood Group:** {patient['blood_group'] or '—'}")
        st.markdown(f"**Contact:** {patient['contact_number'] or '—'}")
    with col2:
        st.markdown(f"**Height:** {patient['height_cm'] or '—'} cm")
        st.markdown(f"**Weight:** {patient['weight_kg'] or '—'} kg")
        st.markdown(f"**BMI:** {bmi or '—'}")

    if patient.get("existing_conditions"):
        st.markdown(f"**Existing Conditions:** {patient['existing_conditions']}")
    if patient.get("allergies"):
        st.markdown(f"**Allergies:** {patient['allergies']}")
    if patient.get("current_medications"):
        st.markdown(f"**Current Medications:** {patient['current_medications']}")


def show_patient_history(patient_id: int):
    """Displays a table of past ECG test results for a patient."""
    history = db.get_patient_history(patient_id)

    if not history:
        st.caption("No previous ECG records for this patient.")
        return

    df = pd.DataFrame(history)[["tested_at", "prediction", "confidence", "notes"]]
    df["tested_at"] = pd.to_datetime(df["tested_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
    df.columns = ["Date/Time", "Result", "Confidence", "Notes"]
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.title("🫀 ECG Anomaly Detection")
    st.caption(
        "⚠️ Portfolio/educational project with a demo patient database. "
        "Not a medical diagnostic tool - do not enter real patient data."
    )

    model, metadata = load_model_and_metadata()
    expected_length = metadata["beat_window_size"]

    # --- Sidebar: patient management ---
    st.sidebar.header("Patient Management")
    selected_patient = patient_selector()
    st.sidebar.divider()
    patient_registration_form()

    if selected_patient is None:
        st.info("👈 Register a patient in the sidebar to get started.")
        return

    st.divider()
    st.subheader(f"Patient: {selected_patient['name']}")
    show_patient_summary(selected_patient)

    st.divider()
    st.subheader("📈 Run New ECG Test")

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

        notes = st.text_input("Doctor's notes (optional)", key="notes_input")

        if st.button("💾 Save to Patient History"):
            db.add_ecg_record(
                patient_id=selected_patient["patient_id"],
                prediction=label, confidence=confidence,
                raw_probability=raw_prob, notes=notes,
            )
            st.success("Saved to patient history.")
            st.rerun()

    st.divider()
    st.subheader("📋 Previous ECG History")
    show_patient_history(selected_patient["patient_id"])


if __name__ == "__main__":
    main()
