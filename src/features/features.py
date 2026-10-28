import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def to_seconds(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip()
    if ":" in v:
        m, s = v.split(":")
        return float(m) * 60 + float(s)
    return float(v)


def engineer_features(df, include_same_lap_flags=False):
    """
    Build lap-level feature table from your cleaned F1 CSVs.
    This is the compact feature set for a first deep learning model.
    """
    df = df.copy()
    df["laptime"] = df["laptime"].apply(to_seconds)

    # sort
    df = df.sort_values(["raceid", "driver", "lap"], ascending=[True, True, True]).reset_index(drop=True)

    # ---- 1) Race-level positional and lap context ----
    race_max_lap = df.groupby("raceid")["lap"].transform("max")
    df["total_laps_in_race"] = race_max_lap
    df["lap_progress"] = df["lap"] / df["total_laps_in_race"]
    df["laps_remaining"] = df["total_laps_in_race"] - df["lap"]

    # ---- 2) Previous lap and delta features ----
    df["previous_lap_time"] = df.groupby(["raceid", "driver"])["laptime"].shift(1)
    df["delta_prev_lap"] = df["laptime"] - df["previous_lap_time"]

    # rolling pace over previous laps
    df["rolling_mean_last_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["rolling_std_last_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).std().fillna(0))
    )

    df["rolling_median_last_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    )

    # ---- 3) Degradation / trend features ----
    df["tyre_degradation_rate"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.diff().fillna(0))
    )

    # slope of previous 3-lap trend
    def slope_3(group):
        s = group.shift(1).rolling(3, min_periods=2).mean()
        # this is a stable proxy for degradation trend
        return s

    df["degradation_slope_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=2).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=True))
    )

    # ---- 4) Stint / pit / strategy context ----
    df["pit_flag"] = df["pit_duration"].notna().astype(int)

    # tyre progress within stint
    stint_max_tyre = df.groupby(["raceid", "driver", "stint"])["tyrelife"].transform("max")
    df["stint_progress"] = df["tyrelife"] / stint_max_tyre

    # previous lap special event flag
    special_cols = ["is_in_lap", "is_out_lap", "is_lap1"]
    if all(col in df.columns for col in special_cols):
        prev_special = (
            df.groupby(["raceid", "driver"])[special_cols]
            .shift(1)
            .fillna(0)
        )
        df["previous_lap_special_flag"] = prev_special.sum(axis=1)
    else:
        df["previous_lap_special_flag"] = 0

    # ---- 5) Race baselines and relative pace features ----
    df["race_reference_pace"] = df.groupby("raceid")["laptime"].transform("median")
    df["compound_reference_pace"] = df.groupby(["raceid", "compound"])["laptime"].transform("median")
    df["driver_reference_pace"] = df.groupby(["raceid", "driver"])["laptime"].transform("median")
    df["team_reference_pace"] = df.groupby(["raceid", "team"])["laptime"].transform("median")

    df["driver_relative_pace"] = df["laptime"] - df["driver_reference_pace"]
    df["team_relative_pace"] = df["laptime"] - df["team_reference_pace"]

    # ---- 6) Position-based context ----
    df["previous_position"] = df.groupby(["raceid", "driver"])["position"].shift(1)
    df["position_change"] = df["position"] - df["previous_position"]

    # ---- 7) Optional same-lap flags: keep only if you want a retrospective/context-aware model ----
    if include_same_lap_flags:
        df["is_in_lap_flag"] = df["is_in_lap"]
        df["is_out_lap_flag"] = df["is_out_lap"]
        df["is_lap1_flag"] = df["is_lap1"]
    else:
        df["is_in_lap_flag"] = 0
        df["is_out_lap_flag"] = 0
        df["is_lap1_flag"] = 0

    # ---- 8) Final model-ready feature set ----
    feature_cols = [
        "raceid", "year", "race", "driver", "team", "lap",
        "compound", "tyrelife", "stint", "position",
        "pit_duration", "lap_progress", "laps_remaining",
        "previous_lap_time", "delta_prev_lap",
        "rolling_mean_last_3", "rolling_std_last_3", "rolling_median_last_3",
        "tyre_degradation_rate", "degradation_slope_3",
        "race_reference_pace", "compound_reference_pace",
        "driver_reference_pace", "team_reference_pace",
        "driver_relative_pace", "team_relative_pace",
        "pit_flag", "stint_progress",
        "previous_position", "position_change",
        "previous_lap_special_flag",
        "is_in_lap_flag", "is_out_lap_flag", "is_lap1_flag"
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    out = df[feature_cols].copy()

    # fill NaNs for model input
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)

    return out


def build_feature_dataset(input_path, output_path, include_same_lap_flags=False):
    df = pd.read_csv(input_path)
    X = engineer_features(df, include_same_lap_flags=include_same_lap_flags)
    X.to_csv(output_path, index=False)
    print(f"Saved {output_path} | shape={X.shape}")


if __name__ == "__main__":
    baseline_in = PROCESSED_DIR / "baseline_pace.csv"
    telemetry_in = PROCESSED_DIR / "race_telemetry.csv"

    baseline_out = FEATURES_DIR / "baseline_features.csv"
    telemetry_out = FEATURES_DIR / "telemetry_features.csv"

    # Baseline dataset: strict/clean version
    build_feature_dataset(baseline_in, baseline_out, include_same_lap_flags=False)

    # Telemetry dataset: context-aware version
    build_feature_dataset(telemetry_in, telemetry_out, include_same_lap_flags=True)

    print("Feature pipeline complete.")