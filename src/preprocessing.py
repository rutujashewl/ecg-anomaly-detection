"""
preprocessing.py
-----------------
Loads raw ECG records (MIT-BIH format), cleans the signal, detects heartbeats,
and saves segmented beats + labels for feature extraction / model training.

Beginner note:
- An ECG record is just a long list of numbers (voltage readings over time).
- We remove noise, find each heartbeat's peak (R-peak), cut out a small window
  around each peak (one "beat"), and label it normal/abnormal using the
  dataset's official annotations (.atr files).
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SAMPLING_RATE = 360          # MIT-BIH standard sampling rate (samples/sec)
WINDOW_SIZE = 180            # samples on each side of R-peak (~0.5 sec total)
LOWCUT = 0.5                 # Hz - removes baseline wander
HIGHCUT = 40.0                # Hz - removes muscle/high-freq noise

# MIT-BIH annotation symbols: 'N' = normal beat. Everything else we treat as
# abnormal for a simple binary classification problem.
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}


def bandpass_filter(signal: np.ndarray, fs: int = SAMPLING_RATE,
                     lowcut: float = LOWCUT, highcut: float = HIGHCUT) -> np.ndarray:
    """
    Removes noise outside the useful ECG frequency range.

    Beginner note: think of this like a radio tuner - it keeps only the
    frequencies that matter for heartbeats and throws away static/noise.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="band")
    return filtfilt(b, a, signal)


def get_record_names(raw_dir: str = RAW_DIR) -> list:
    """
    Finds all record names in data/raw/ by looking for .hea header files.
    e.g. '100.hea' -> record name '100'
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"'{raw_dir}' does not exist. Put your MIT-BIH .dat/.hea/.atr files there."
        )

    records = sorted({
        f.split(".")[0] for f in os.listdir(raw_dir) if f.endswith(".hea")
    })

    if not records:
        raise FileNotFoundError(
            f"No .hea files found in '{raw_dir}'. Check that the dataset was extracted there."
        )

    return records


def process_record(record_name: str, raw_dir: str = RAW_DIR):
    """
    Loads one record, filters it, finds beats using the annotation file,
    and returns segmented beat windows + binary labels.

    We use the dataset's own annotation (.atr) locations for R-peaks instead
    of detecting them ourselves - this is the standard, more reliable approach
    for MIT-BIH and avoids implementing Pan-Tompkins from scratch as a beginner.
    """
    record_path = os.path.join(raw_dir, record_name)

    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, extension="atr")
    except Exception as exc:
        print(f"[WARN] Skipping record '{record_name}': {exc}")
        return None, None

    # Use the first channel (lead) of the ECG signal.
    raw_signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(raw_signal, fs=record.fs)

    beats = []
    labels = []
    patient_ids = []  # NEW: track which patient each beat came from

    for peak_idx, symbol in zip(annotation.sample, annotation.symbol):
        start = peak_idx - WINDOW_SIZE
        end = peak_idx + WINDOW_SIZE

        # Skip beats too close to the start/end of the recording.
        if start < 0 or end > len(filtered_signal):
            continue

        beat_window = filtered_signal[start:end]
        label = 1 if symbol not in NORMAL_SYMBOLS else 0  # 1 = abnormal, 0 = normal

        beats.append(beat_window)
        labels.append(label)
        patient_ids.append(record_name)  # NEW: every beat tagged with its patient/record

    if not beats:
        return None, None, None

    return np.array(beats), np.array(labels), np.array(patient_ids)


def build_dataset(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR):
    """
    Processes every record in data/raw/, combines all beats into one dataset,
    and saves it as a compressed .npz file in data/processed/.
    """
    os.makedirs(processed_dir, exist_ok=True)
    record_names = get_record_names(raw_dir)

    all_beats = []
    all_labels = []
    all_patient_ids = []  # NEW
    processed_count = 0

    for name in record_names:
        beats, labels, patient_ids = process_record(name, raw_dir)
        if beats is None:
            continue
        all_beats.append(beats)
        all_labels.append(labels)
        all_patient_ids.append(patient_ids)  # NEW
        processed_count += 1
        print(f"[OK] Processed record {name}: {len(labels)} beats")

    if not all_beats:
        raise RuntimeError("No records were successfully processed. Check dataset files.")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patient_ids, axis=0)  # NEW

    out_path = os.path.join(processed_dir, "beats.npz")
    np.savez_compressed(out_path, X=X, y=y, patient_ids=patient_ids)  # NEW: patient_ids saved too

    print(f"\nDone. Processed {processed_count}/{len(record_names)} records.")
    print(f"Total beats: {len(y)} | Normal: {(y==0).sum()} | Abnormal: {(y==1).sum()}")
    print(f"Saved to: {out_path}")

    return X, y


if __name__ == "__main__":
    build_dataset()
