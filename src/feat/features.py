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
    s = str(value).strip()
    if ":" in s:
        m, sec = s.split(":")
        return float(m) * 60 + float(sec)
    return float(s)


def engineer_features(df, include_same_lap_flags=False):
    df = df.copy()
    df["laptime"] = df["laptime"].map(to_seconds)

    df = df.sort_values(["raceid", "driver", "lap"]).reset_index(drop=True)

    # basic lap context
    max_lap = df.groupby("raceid")["lap"].transform("max")
    df["lap_progress"] = df["lap"] / max_lap
    df["laps_remaining"] = max_lap - df["lap"]

    # previous lap and delta
    df["previous_lap_time"] = df.groupby(["raceid", "driver"])["laptime"].shift(1)
    df["delta_prev_lap"] = df["laptime"] - df["previous_lap_time"]

    # rolling pace
    df["rolling_mean_last_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_std_last_3"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).std().fillna(0))
    )

    # degradation
    df["tyre_degradation_rate"] = (
        df.groupby(["raceid", "driver"])["laptime"]
        .transform(lambda s: s.diff().fillna(0))
    )

    # pit / stint context
    df["pit_flag"] = df["pit_duration"].notna().astype(int)
    stint_max = df.groupby(["raceid", "driver", "stint"])["tyrelife"].transform("max")
    df["stint_progress"] = df["tyrelife"] / stint_max

    # race and compound baselines
    df["race_reference_pace"] = df.groupby("raceid")["laptime"].transform("median")
    df["compound_reference_pace"] = df.groupby(["raceid", "compound"])["laptime"].transform("median")

    df["driver_reference_pace"] = df.groupby(["raceid", "driver"])["laptime"].transform("median")
    df["team_reference_pace"] = df.groupby(["raceid", "team"])["laptime"].transform("median")

    df["driver_relative_pace"] = df["laptime"] - df["driver_reference_pace"]
    df["team_relative_pace"] = df["laptime"] - df["team_reference_pace"]

    # position
    df["previous_position"] = df.groupby(["raceid", "driver"])["position"].shift(1)
    df["position_change"] = df["position"] - df["previous_position"]

    # special-lap flags (only if you want them)
    if include_same_lap_flags:
        df["is_in_lap_flag"] = df["is_in_lap"]
        df["is_out_lap_flag"] = df["is_out_lap"]
        df["is_lap1_flag"] = df["is_lap1"]
    else:
        df["is_in_lap_flag"] = 0
        df["is_out_lap_flag"] = 0
        df["is_lap1_flag"] = 0

    if {"is_in_lap", "is_out_lap", "is_lap1"}.issubset(df.columns):
        prev_special = (
            df.groupby(["raceid", "driver"])[["is_in_lap", "is_out_lap", "is_lap1"]]
            .shift(1)
            .fillna(0)
        )
        df["previous_lap_special_flag"] = prev_special.sum(axis=1)
    else:
        df["previous_lap_special_flag"] = 0

    keep = [
        "raceid", "year", "race", "driver", "team", "lap",
        "compound", "tyrelife", "stint", "position", "pit_duration",
        "lap_progress", "laps_remaining",
        "previous_lap_time", "delta_prev_lap",
        "rolling_mean_last_3", "rolling_std_last_3",
        "tyre_degradation_rate",
        "race_reference_pace", "compound_reference_pace",
        "driver_reference_pace", "team_reference_pace",
        "driver_relative_pace", "team_relative_pace",
        "pit_flag", "stint_progress",
        "previous_position", "position_change",
        "previous_lap_special_flag",
        "is_in_lap_flag", "is_out_lap_flag", "is_lap1_flag",
        "laptime"
    ]

    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    out = out.fillna(0)
    return out


def build_feature_dataset(input_path, output_path, include_same_lap_flags=False):
    df = pd.read_csv(input_path)
    feat = engineer_features(df, include_same_lap_flags=include_same_lap_flags)
    feat.to_csv(output_path, index=False)
    print(f"Saved {output_path} | shape={feat.shape}")


if __name__ == "__main__":
    baseline_in = PROCESSED_DIR / "baseline_pace.csv"
    telemetry_in = PROCESSED_DIR / "race_telemetry.csv"

    baseline_out = FEATURES_DIR / "baseline_features.csv"
    telemetry_out = FEATURES_DIR / "telemetry_features.csv"

    # strict baseline version
    build_feature_dataset(baseline_in, baseline_out, include_same_lap_flags=False)

    # telemetry version
    build_feature_dataset(telemetry_in, telemetry_out, include_same_lap_flags=True)

    print("Done.")