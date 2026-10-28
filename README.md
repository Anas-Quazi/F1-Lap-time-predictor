# F1 Lap Time Prediction with PyTorch ANN

An end-to-end machine learning workflow that predicts Formula 1 lap time (seconds) from engineered race telemetry features using a PyTorch feedforward neural network.

---

## Table of Contents

1. [Overview](#1-overview)
2. [ML System Architecture](#2-ml-system-architecture)
3. [Project Pipeline](#3-project-pipeline)
4. [Model](#4-model)
5. [Model Integrity](#5-model-integrity)
6. [Results](#6-results)
7. [Repository Structure](#7-repository-structure)
8. [Tech Stack](#8-tech-stack)
9. [Usage](#9-usage)
10. [Conclusion](#10-conclusion)

---

## 1. Overview

Lap time is a nonlinear regression target shaped by interacting factors such as tire life, stint length, compound, pit timing, race position, previous-lap pace, and driver/team relative pace.

**Objectives**
- Load and preprocess race telemetry data
- Engineer features capturing pace, degradation, and race progression
- Split data without leakage
- Train and validate a PyTorch ANN regressor
- Persist the model and preprocessing artifacts
- Run reproducible inference on new inputs

**Applications:** tire degradation analysis, strategy evaluation, driver/team pace comparison, expected lap time estimation.

---

## 2. ML System Architecture

```
                    TRAINING
┌──────────────┐   ┌───────────────┐   ┌────────────────┐
│  Telemetry   │──▶│ Preprocessing │──▶│    Feature     │
│     Data     │   │ (clean, NaNs) │   │  Engineering   │
└──────────────┘   └───────────────┘   └───────┬────────┘
                                               ▼
                                    ┌────────────────────┐
                                    │  Grouped Split     │
                                    │ (race id + driver) │
                                    └─────────┬──────────┘
                                              ▼
                  ┌─────────────────┐   ┌────────────────┐
                  │ StandardScaler  │──▶│   ANN Training │
                  │ (fit on train)  │   │ + Early Stop   │
                  └────────┬────────┘   └───────┬────────┘
                           │                    ▼
                           │            ┌────────────────┐
                           └───────────▶│   Artifacts    │
                                        │ model • scaler │
                                        │ features • KPIs│
                                        └───────┬────────┘
                    INFERENCE                   ▼
┌──────────────┐   ┌───────────────┐   ┌────────────────┐
│   New Input  │──▶│ Load Features │──▶│  Scale → ANN   │──▶ Lap Time (s)
└──────────────┘   │  + Scaler     │   └────────────────┘
                   └───────────────┘
```

---

## 3. Project Pipeline

| Step | Stage | Description |
|------|-------|-------------|
| 1 | Load data | Read telemetry feature CSV |
| 2 | Clean | Drop rows with missing target |
| 3 | Split | Grouped train/validation/test split by race id + driver |
| 4 | Feature selection | Select numeric features |
| 5 | Imputation | Fill missing values |
| 6 | Scaling | Fit `StandardScaler` on training data; transform all splits |
| 7 | Tensors | Convert to PyTorch tensors and batch |
| 8 | Train | Adam optimizer, MSE loss |
| 9 | Early stopping | Monitor validation loss; restore best weights |
| 10 | Evaluate | Compute MAE, RMSE, R² on the test set |
| 11 | Persist | Save model, scaler, feature columns, metrics |
| 12 | Predict | Load artifacts and infer on new data |

**Engineered features include:** lap number, position and position change, tire compound, tire life, stint, pit duration, previous lap time, lap-to-lap delta, rolling mean/std, degradation indicators, relative pace metrics, driver/team pace context, and race progression variables.

---

## 4. Model

```python
class F1ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
```

**Architecture:** `input → 128 → 64 → 32 → 1` with ReLU activations and dropout (0.2, 0.1)

**Training configuration**

| Parameter | Value |
|-----------|-------|
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Max epochs | 120 |
| Early stopping patience | 12 |

---

## 5. Model Integrity

- **Grouped splitting:** data is split by race id + driver, so the same race-driver context never appears in more than one of train, validation, and test.
- **Train-only preprocessing:** the scaler is fit on training data only, then applied to validation and test sets.
- **Early stopping:** validation loss is monitored with a patience window, and the best weights are restored to limit overfitting.
- **Regularization:** dropout layers reduce co-adaptation of neurons.
- **Reproducible inference:** model weights, scaler, and feature column order are persisted and reloaded together.

---

## 6. Results

Evaluated on the held-out test set:

| Metric | Value |
|--------|-------|
| MAE | 0.3122 s |
| RMSE | 0.5434 s |
| R² | 0.9978 |

The model achieves low absolute error and explains nearly all variance in lap time on unseen race-driver groups.

---

## 7. Repository Structure

```
F1 Lap Time Predictor/
│
├── .vscode/
│   └── settings.json
│
├── artifacts/
│   ├── feat/
│   │   └── feature_columns.json
│   ├── metrics/
│   │   └── metrics.json
│   ├── model/
│   │   └── ann_best.pt
│   └── scaler/
│       └── scaler.pkl
│
├── cache/
│
├── data/
│   ├── feat/
│   │   ├── baseline_features.csv
│   │   └── telemetry_features.csv
│   ├── processed/
│   │   ├── baseline_pace.csv
│   │   └── race_telemetry.csv
│   └── raw/
│       └── raw_data.csv
│
├── src/
│   ├── data/
│   │   ├── clean.py
│   │   ├── ingest.py
│   │   └── split_data.py
│   ├── feat/
│   │   └── features.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   ├── train_ann_...
│   │   └── train_ann.py
│   └── __init__.py
│
├── test/
│   └── test.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt

```

---

## 8. Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep learning | PyTorch |
| ML utilities | scikit-learn (`StandardScaler`, grouped splitting) |
| Data | pandas, NumPy |

---

## 9. Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/models/train_ann.py

# Predict on new data
python src/predict.py
```

---

## 10. Conclusion

This project delivers a complete lap-time prediction workflow: feature engineering, grouped leakage-safe splitting, validation-based early stopping, evaluation, artifact persistence, and inference. It serves as a solid baseline for motorsport analytics, race strategy analysis, and performance modeling.