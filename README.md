# ECG Anomaly Detection

Detects abnormal heartbeats (arrhythmia) from ECG signals using the MIT-BIH
Arrhythmia Database. Combines a Random Forest baseline (on statistically
selected top features) with a 1D-CNN deep learning model trained on raw
signal segments.

## Project Structure

```
ecg-anomaly-detection/
├── data/
│   ├── raw/            # put MIT-BIH .dat/.hea/.atr files here
│   └── processed/       # generated: beats.npz, features.csv
├── src/
│   ├── preprocessing.py # filtering + beat segmentation
│   ├── features.py      # feature extraction + best-feature selection
│   ├── train.py          # baseline RF + CNN training
│   └── evaluate.py       # metrics + plots
├── models/                # generated: trained model files
├── reports/                # generated: evaluation plots
├── app.py                  # Streamlit inference app
├── requirements.txt
└── Procfile                # Render start command
```

## Setup (Local)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put your MIT-BIH dataset files in data/raw/
#    (already done if you downloaded from PhysioNet)

# 4. Run the pipeline, in order
cd src
python preprocessing.py     # -> data/processed/beats.npz
python features.py          # -> data/processed/features.csv + feature ranking
python train.py              # -> models/baseline_rf.pkl, models/cnn_model.keras
python evaluate.py           # -> reports/confusion_matrix.png, roc_curve.png

# 5. Run the app locally
cd ..
streamlit run app.py
```

## Deploying on Render

1. Push this whole folder to a GitHub repository.
2. Go to [render.com](https://render.com) → **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Render should auto-detect Python. Set these manually if needed:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
     (already defined in the `Procfile`, Render will pick it up automatically)
5. Deploy. Render will give you a public URL for the Streamlit app.

**Important:** Do NOT commit `data/raw/` or `data/processed/` to GitHub — the
dataset is large and not needed in production, only the trained model files
in `models/` are needed for the deployed app to run. Add a `.gitignore`:

```
data/raw/*
data/processed/*
venv/
__pycache__/
```

Make sure `models/cnn_model.keras` and `models/metadata.json` ARE committed
(they're small) since `app.py` needs them to run on Render.

## Notes

- Labels are simplified to **binary**: Normal vs Abnormal (MIT-BIH has many
  arrhythmia sub-types; this can be extended to multi-class later).
- Class imbalance is handled with **SMOTE** for the Random Forest and
  **class weights** for the CNN.
- Evaluation prioritizes **recall/sensitivity** over raw accuracy, since in a
  medical context, missing an abnormal beat is worse than a false alarm.
- This is a portfolio/educational project — not a certified medical device.
