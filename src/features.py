"""
features.py
------------
Turns raw beat segments (numbers) into meaningful features a model can learn
from, then selects the BEST features using statistical feature selection
(so we don't feed the model useless/noisy columns).

Beginner note:
- A "feature" is just a single measurable number that describes the signal,
  e.g. "average heart rate" or "how spread out the frequencies are".
- "Feature selection" means: out of all the features we calculate, keep only
  the ones that actually help tell normal vs abnormal beats apart.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.feature_selection import SelectKBest, f_classif

SAMPLING_RATE = 360


def extract_features_single(beat: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Extracts a set of time-domain and frequency-domain features from ONE
    beat window. Used both for training (many beats) and inference (1 beat).
    """
    features = {}

    # --- Time-domain features ---
    features["mean"] = np.mean(beat)
    features["std"] = np.std(beat)
    features["min"] = np.min(beat)
    features["max"] = np.max(beat)
    features["range"] = features["max"] - features["min"]
    features["skewness"] = pd.Series(beat).skew()
    features["kurtosis"] = pd.Series(beat).kurtosis()

    # R-peak is at the center of the window (see preprocessing.py WINDOW_SIZE)
    r_peak_amplitude = beat[len(beat) // 2]
    features["r_peak_amplitude"] = r_peak_amplitude

    # Energy = how much "power" the signal has overall
    features["energy"] = np.sum(beat ** 2)

    # --- Frequency-domain features (FFT) ---
    fft_vals = np.abs(rfft(beat))
    fft_freqs = rfftfreq(len(beat), d=1 / fs)

    # Power in clinically relevant ECG frequency bands
    def band_power(low, high):
        mask = (fft_freqs >= low) & (fft_freqs < high)
        return np.sum(fft_vals[mask] ** 2)

    features["power_low"] = band_power(0.5, 5)     # low-frequency components
    features["power_mid"] = band_power(5, 15)      # QRS complex range
    features["power_high"] = band_power(15, 40)     # high-frequency noise range

    # Dominant frequency = the strongest frequency present in the beat
    features["dominant_freq"] = fft_freqs[np.argmax(fft_vals)]

    return features


def build_feature_dataframe(X_beats: np.ndarray, y_labels: np.ndarray) -> pd.DataFrame:
    """
    Applies extract_features_single() to every beat and returns a labeled
    DataFrame ready for model training.
    """
    rows = [extract_features_single(beat) for beat in X_beats]
    df = pd.DataFrame(rows)
    df["label"] = y_labels
    return df


def select_best_features(df: pd.DataFrame, k: int = 8, target_col: str = "label"):
    """
    Statistically ranks features by how well they separate normal vs abnormal
    beats (ANOVA F-test), and keeps only the top `k`.

    Beginner note: this answers "which of my 12 features actually matter?"
    instead of guessing. Features with a high F-score are strong signals;
    low-score features are close to random noise for this task.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
    selected_features = scores.head(k).index.tolist()

    print("Feature importance ranking (higher = more useful):")
    print(scores.round(2).to_string())
    print(f"\nSelected top {k} features: {selected_features}")

    return selected_features, scores


if __name__ == "__main__":
    data = np.load("data/processed/beats.npz")
    X_beats, y_labels = data["X"], data["y"]

    df = build_feature_dataframe(X_beats, y_labels)
    df.to_csv("data/processed/features.csv", index=False)
    print(f"Saved features to data/processed/features.csv | shape={df.shape}")

    best_features, scores = select_best_features(df, k=8)
    scores.to_csv("data/processed/feature_scores.csv")
