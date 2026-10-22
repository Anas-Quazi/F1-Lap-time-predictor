# Lap Time Prediction in Formula 1 Using Deep Learning

## Problem
F1 lap time depends on non-linear interactions between circuit, tyre
compound, tyre age, fuel load and race context. This project predicts
lap time from these factors and compares a neural network with
classical ML baselines.

## Dataset
Combined F1 lap dataset (Kaggle + FastF1), ~120k laps, 2019-2024. <br>
Columns used: Race, Year, Driver, Team, Lap, Position, Compound,
TyreLife, Stint, LapTime.

## Approach
1. Clean data (remove pit laps, lap 1, safety-car outliers)
2. Engineer features (laps remaining, race progress, delta from circuit median)
3. Split train/test by race to avoid leakage
4. Train: Linear Regression, Random Forest, MLP with embeddings
5. Evaluate with MAE / RMSE (seconds) and R²

## Results
| Model | MAE (s) | RMSE (s) |
|---|---|---|
| Circuit median baseline | - | - |
| Linear Regression | - | - |
| Random Forest | - | - |
| Neural Network (MLP) | - | - |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```
Run notebooks in order 01 to 04.

## Limitations
- No weather data in this dataset
- Only SOFT/MEDIUM/HARD compounds
- Traffic and driver intent are not modeled

## Future work
Add weather data, sequence models (LSTM), per-driver stint simulation.

## Author
<your name>
