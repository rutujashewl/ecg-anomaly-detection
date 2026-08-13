"""
database.py
------------
SQLite-backed patient database for the ECG Anomaly Detection app.

Stores:
  - patients: personal + health details
  - ecg_records: every prediction made, linked to a patient

IMPORTANT: This is a portfolio/educational project. Do NOT enter real
patient data here - use synthetic/fake demo data only. Storing real
Protected Health Information (PHI) has legal and ethical requirements
(e.g. HIPAA, India's DPDP Act) that this simple demo does not meet.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "patients.db")


@contextmanager
def get_connection():
    """Provides a database connection that auto-closes and commits/rolls back."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates the patients and ecg_records tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                contact_number TEXT,
                blood_group TEXT,
                height_cm REAL,
                weight_kg REAL,
                existing_conditions TEXT,
                allergies TEXT,
                current_medications TEXT,
                registered_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ecg_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_probability REAL NOT NULL,
                notes TEXT,
                tested_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                    ON DELETE CASCADE
            )
        """)


def add_patient(name: str, age: int, gender: str, contact_number: str = "",
                 blood_group: str = "", height_cm: float = None,
                 weight_kg: float = None, existing_conditions: str = "",
                 allergies: str = "", current_medications: str = "") -> int:
    """Inserts a new patient record and returns the new patient_id."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO patients (
                name, age, gender, contact_number, blood_group,
                height_cm, weight_kg, existing_conditions, allergies,
                current_medications, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, age, gender, contact_number, blood_group,
            height_cm, weight_kg, existing_conditions, allergies,
            current_medications, datetime.now().isoformat()
        ))
        return cursor.lastrowid


def get_all_patients() -> list:
    """Returns all patients as a list of dicts, most recently registered first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY registered_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_patient(patient_id: int) -> dict:
    """Returns a single patient's details, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Standard BMI formula: weight(kg) / height(m)^2"""
    if not height_cm or not weight_kg:
        return None
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def add_ecg_record(patient_id: int, prediction: str, confidence: float,
                    raw_probability: float, notes: str = "") -> int:
    """Saves an ECG test result linked to a patient."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO ecg_records (
                patient_id, prediction, confidence, raw_probability,
                notes, tested_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_id, prediction, confidence, raw_probability,
            notes, datetime.now().isoformat()
        ))
        return cursor.lastrowid


def get_patient_history(patient_id: int) -> list:
    """Returns all ECG records for a patient, most recent first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM ecg_records
            WHERE patient_id = ?
            ORDER BY tested_at DESC
        """, (patient_id,)).fetchall()
        return [dict(row) for row in rows]


def get_all_records_with_patient_names() -> list:
    """Returns every ECG record joined with the patient's name - useful for
    a global dashboard view across all patients."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT r.*, p.name as patient_name
            FROM ecg_records r
            JOIN patients p ON r.patie0nt_id = p.patient_id
            ORDER BY r.tested_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {os.path.abspath(DB_PATH)}")
