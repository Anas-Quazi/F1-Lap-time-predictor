import json
import pickle
import pandas as pd
import torch


#? load trained artifacts
with open("artifacts/scaler/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/feat/feature_columns.json", "r") as f:
    engineered_cols = json.load(f)

#^ model architecture
class F1ANN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = F1ANN(len(engineered_cols))
model.load_state_dict(torch.load("artifacts/model/ann_best.pt", map_location="cpu"))
model.eval()


#* core features user must input
core_features = [
    "lap",
    "position",
    "compound",
    "tyrelife",
    "stint",
    "pit_duration",
    "is_in_lap",
    "is_out_lap",
    "is_lap1",
    "previous_lap_time",
    "delta_prev_lap",
]

#& ptional history-based features if user has them
optional_features = [
    "rolling_mean_last_3",
    "rolling_std_last_3",
    "tyre_degradation_rate",
    "race_reference_pace",
    "compound_reference_pace",
    "driver_reference_pace",
    "team_reference_pace",
    "driver_relative_pace",
    "team_relative_pace",
    "lap_progress",
    "laps_remaining",
    "stint_progress",
    "previous_position",
    "position_change",
    "previous_lap_special_flag",
]

#~ feature engineering function
def make_engineered_row(raw):
    row = {}

    #todo copy base values
    for c in core_features:
        row[c] = raw.get(c, 0.0)

    #? optional history values
    for c in optional_features:
        row[c] = raw.get(c, 0.0)

    #* convert compound to numeric if needed
    compound_map = {
        "SOFT": 3,
        "MEDIUM": 2,
        "HARD": 1,
        "INTERMEDIATE": 4,
        "WET": 5,
    }
    row["compound"] = compound_map.get(str(raw.get("compound", "MEDIUM")).upper(), 2)

    #^ if history is not given, approximate with defaults
    if row["previous_lap_time"] == 0:
        row["previous_lap_time"] = row.get("lap", 1) * 90.0

    row["delta_prev_lap"] = row.get("delta_prev_lap", 0.0)

    #todo rolling values if not provided
    row["rolling_mean_last_3"] = row.get("rolling_mean_last_3", row["previous_lap_time"])
    row["rolling_std_last_3"] = row.get("rolling_std_last_3", 0.0)

    #! simple degradation approximation
    row["tyre_degradation_rate"] = row.get("tyre_degradation_rate", 0.05)

    #? reference pace approximation
    row["race_reference_pace"] = row.get("race_reference_pace", row["previous_lap_time"])
    row["compound_reference_pace"] = row.get("compound_reference_pace", row["previous_lap_time"])
    row["driver_reference_pace"] = row.get("driver_reference_pace", row["previous_lap_time"])
    row["team_reference_pace"] = row.get("team_reference_pace", row["previous_lap_time"])

    row["driver_relative_pace"] = row.get("driver_relative_pace", 0.0)
    row["team_relative_pace"] = row.get("team_relative_pace", 0.0)

    row["lap_progress"] = row.get("lap_progress", 0.5)
    row["laps_remaining"] = row.get("laps_remaining", 20)
    row["stint_progress"] = row.get("stint_progress", 0.5)
    row["previous_position"] = row.get("previous_position", row["position"])
    row["position_change"] = row.get("position_change", 0)
    row["previous_lap_special_flag"] = row.get("previous_lap_special_flag", 0)
    row["is_in_lap_flag"] = row.get("is_in_lap_flag", row["is_in_lap"])
    row["is_out_lap_flag"] = row.get("is_out_lap_flag", row["is_out_lap"])
    row["is_lap1_flag"] = row.get("is_lap1_flag", row["is_lap1"])

    return row


#& user input
print("Enter core features for one lap prediction")
print("If you do not know optional history features, press Enter to use defaults.")
print("")

raw = {}

for c in core_features:
    val = input(f"{c}: ").strip()
    if val == "":
        raw[c] = 0.0
    else:
        raw[c] = float(val)

for c in optional_features:
    val = input(f"{c} (optional, press Enter for default): ").strip()
    if val == "":
        raw[c] = 0.0
    else:
        raw[c] = float(val)

#! compound input is sometimes string, so handle separately
comp = input("compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET): ").strip()
raw["compound"] = comp

#todo build engineered row
engineered = make_engineered_row(raw)

#* keep only columns used by model
df = pd.DataFrame([engineered])
for col in engineered_cols:
    if col not in df.columns:
        df[col] = 0.0

df = df[engineered_cols]

#~ scale
X = scaler.transform(df)
X_tensor = torch.tensor(X, dtype=torch.float32)

#? predict
with torch.no_grad():
    pred = model(X_tensor).numpy()[0]

print("")
print(f"Predicted lap time: {pred:.4f} seconds")