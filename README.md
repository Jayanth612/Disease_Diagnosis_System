# VitaScan — Disease Diagnosis System

AI-assisted Flask web app that screens **heart disease**, **stroke**, and **hepatitis** risk from clinical inputs. Vanilla HTML/CSS/JS frontend (no React).

> Screening aid only — not a medical diagnosis.

## Features

- Disease-specific forms (only relevant fields per model)
- Calibrated low / high risk probabilities with a simple result meter
- Models accept **raw** clinical values (scaler stored with each model)
- Class imbalance handled with **SMOTE** during training
- Demo high/low risk buttons for quick smoke tests

## Project layout

```
app.py                 # Flask API + static page
feature.py             # Feature column order for each disease
model_bundle.py        # DiseaseModel (scaler + classifier) for pickle load/save
train_models.py        # Retrain all models from raw CSVs
templates/index.html   # VitaScan UI
Datasets/              # Raw + cleaned CSVs
Trained_Models/        # *.pkl model bundles
problems.txt           # Manual test cases (before vs after fix)
```

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python train_models.py    # rebuild models (scaling + SMOTE)
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Docker

```bash
docker build -t vitascan .
docker run -p 5000:5000 vitascan
```

## API

### `GET /health`

```json
{ "status": "ok", "models": ["heart", "stroke", "hepatitis"] }
```

### `POST /predict`

```json
{
  "disease": "heart",
  "features": {
    "Age": 67,
    "Sex": 1,
    "RestingBP": 150,
    "Cholesterol": 256,
    "FastingBS": 1,
    "MaxHR": 120,
    "Oldpeak": 2.8,
    "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 0,
    "ChestPainType_TA": 1,
    "RestingECG_Normal": 0,
    "RestingECG_ST": 1,
    "ExerciseAngina_Y": 1,
    "ST_Slope_Flat": 1,
    "ST_Slope_Up": 0
  }
}
```

Response includes `prediction_text`, `raw.prediction`, `raw.probability`, and `raw.risk`.

## Models

| Disease   | Algorithm                         | Notes                                      |
|-----------|-----------------------------------|--------------------------------------------|
| Heart     | Random Forest (+ calibration)     | SMOTE on train; threshold 0.45             |
| Stroke    | XGBoost                           | Heavy imbalance → SMOTE ratio 0.45; thr 0.28 |
| Hepatitis | XGBoost (+ calibration)           | Binary healthy vs disease; SMOTE; thr 0.45 |

Retrain anytime:

```bash
python train_models.py
```

## Hepatitis lab units

Use dataset units (HCV / UCI-style), not mg/dL-style placeholders:

| Feature | Unit        | Typical healthy range (approx.) |
|---------|-------------|----------------------------------|
| ALB     | g/L         | ~35–50                           |
| ALP/ALT/AST/GGT | U/L | varies                           |
| BIL     | µmol/L      | low teens / lower                |
| CHOL    | mmol/L      | ~4–6                             |
| CREA    | µmol/L      | ~60–100                          |
| PROT    | g/L         | ~65–80                           |

See `problems.txt` for before/after scores on the original test payloads.

## Requirements

See `requirements.txt` (`flask`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `joblib`, …).
