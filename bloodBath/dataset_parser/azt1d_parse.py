from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import re
from bloodBath.dataset_parser.save_splits import save_chronological_splits

DATASET = "azt1d"
DATASET_PATH = Path("/home/desktop/Sneior_Project/datasets/azt1d/cgm")
OUTPUT_DIR = Path("/home/desktop/Sneior_Project/bloodBender/bloodBath/bloodBank/raw/datasets/")
TZ = "America/Los_Angeles"


def localize_timestamp_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")

    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(TZ)

    return ts.dt.tz_localize(
        TZ,
        ambiguous="NaT",
        nonexistent="shift_forward"
    )


def extract_subject_id(csv_file: Path) -> str:
    folder_name = csv_file.parent.name
    match = re.search(r"Subject\s+(\d+)", folder_name, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"Subject\s+(\d+)", csv_file.stem, re.IGNORECASE)
    if match:
        return match.group(1)

    return csv_file.stem


def build_bg_grid(df: pd.DataFrame, timestamp_col: str, bg_col: str) -> pd.DataFrame:
    raw = pd.DataFrame({
        "timestamp_raw": localize_timestamp_series(df[timestamp_col]),
        "bg_raw": pd.to_numeric(df[bg_col], errors="coerce")
    }).dropna(subset=["timestamp_raw"]).copy()

    raw["timestamp"] = raw["timestamp_raw"].dt.floor("5min")
    raw["bg_observed"] = raw["bg_raw"].notna()

    grouped = (
        raw.groupby("timestamp", as_index=False)
        .agg(
            bg=("bg_raw", "mean"),
            bg_observed=("bg_observed", "max")
        )
        .sort_values("timestamp")
    )

    full_grid = pd.DataFrame({
        "timestamp": pd.date_range(
            start=grouped["timestamp"].min(),
            end=grouped["timestamp"].max(),
            freq="5min",
            tz=TZ
        )
    })

    out = full_grid.merge(grouped, on="timestamp", how="left")
    out["bg_observed"] = out["bg_observed"].fillna(False).astype(bool)

    # mask_bg=True means there was no original BG sample in this bucket
    out["mask_bg"] = ~out["bg_observed"]

    # Fill only after mask creation
    out["bg"] = out["bg"].interpolate(method="linear", limit_direction="both")

    return out


def build_basal_grid(df: pd.DataFrame, timestamp_col: str, basal_col: str, full_timestamps: pd.Series) -> pd.DataFrame:
    if basal_col not in df.columns:
        return pd.DataFrame({
            "timestamp": full_timestamps,
            "basal_rate": 0.0
        })

    raw = pd.DataFrame({
        "timestamp_raw": localize_timestamp_series(df[timestamp_col]),
        "basal_rate": pd.to_numeric(df[basal_col], errors="coerce")
    }).dropna(subset=["timestamp_raw"]).copy()

    raw["timestamp"] = raw["timestamp_raw"].dt.floor("5min")

    grouped = (
        raw.groupby("timestamp", as_index=False)
        .agg(basal_rate=("basal_rate", "last"))
        .sort_values("timestamp")
    )

    out = pd.DataFrame({"timestamp": full_timestamps}).merge(grouped, on="timestamp", how="left")

    # Basal persists until changed
    out["basal_rate"] = out["basal_rate"].ffill().fillna(0.0)

    return out


def build_bolus_grid(df: pd.DataFrame, timestamp_col: str, bolus_col: str, full_timestamps: pd.Series) -> pd.DataFrame:
    if bolus_col not in df.columns:
        return pd.DataFrame({
            "timestamp": full_timestamps,
            "bolus_dose": 0.0
        })

    raw = pd.DataFrame({
        "timestamp_raw": localize_timestamp_series(df[timestamp_col]),
        "bolus_dose": pd.to_numeric(df[bolus_col], errors="coerce").fillna(0.0)
    }).dropna(subset=["timestamp_raw"]).copy()

    raw["timestamp"] = raw["timestamp_raw"].dt.floor("5min")

    grouped = (
        raw.groupby("timestamp", as_index=False)
        .agg(bolus_dose=("bolus_dose", "sum"))
        .sort_values("timestamp")
    )

    out = pd.DataFrame({"timestamp": full_timestamps}).merge(grouped, on="timestamp", how="left")
    out["bolus_dose"] = out["bolus_dose"].fillna(0.0)

    return out


def parse_csv_to_df(csv_file: Path):
    df = pd.read_csv(csv_file)

    timestamp_col = "EventDateTime"
    bg_col = "CGM"
    basal_col = "Basal"
    bolus_col = "TotalBolusInsulinDelivered"

    missing = [c for c in [timestamp_col, bg_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_file}: {missing}")

    out = build_bg_grid(df, timestamp_col, bg_col)

    basal_df = build_basal_grid(df, timestamp_col, basal_col, out["timestamp"])
    out = out.merge(basal_df, on="timestamp", how="left")

    bolus_df = build_bolus_grid(df, timestamp_col, bolus_col, out["timestamp"])
    out = out.merge(bolus_df, on="timestamp", how="left")

    out["basal_delta"] = out["basal_rate"].diff().fillna(0.0)

    last_bolus_time = pd.NaT
    tslb = []
    for ts, dose in zip(out["timestamp"], out["bolus_dose"]):
        if dose > 0:
            last_bolus_time = ts

        if pd.isna(last_bolus_time):
            tslb.append(1440.0)
        else:
            tslb.append((ts - last_bolus_time).total_seconds() / 60.0)

    out["time_since_last_bolus"] = tslb

    out["bg_slope_15min"] = (out["bg"] - out["bg"].shift(3)) / 15.0
    out["bg_slope_30min"] = (out["bg"] - out["bg"].shift(6)) / 30.0
    out["bg_slope_15min"] = out["bg_slope_15min"].fillna(0.0)
    out["bg_slope_30min"] = out["bg_slope_30min"].fillna(0.0)

    minutes_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    angle = 2 * np.pi * minutes_of_day / (24 * 60)
    out["sin_time"] = np.sin(angle)
    out["cos_time"] = np.cos(angle)

    # Final masks
    out["mask_bg"] = out["mask_bg"].astype(bool)
    out["mask_label"] = (~out["mask_bg"]).astype(bool)

    subject_id = extract_subject_id(csv_file)

    out = out[
        [
            "timestamp",
            "bg",
            "basal_rate",
            "bolus_dose",
            "basal_delta",
            "time_since_last_bolus",
            "bg_slope_15min",
            "bg_slope_30min",
            "sin_time",
            "cos_time",
            "mask_bg",
            "mask_label",
        ]
    ]

    return out, subject_id


def traverse_dataset(dataset_dir: Path, output_dir: Path) -> list[str]:
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dataset_dir}")

    results = []

    for csv_file in dataset_path.rglob("*.csv"):
        try:
            df, subject_id = parse_csv_to_df(csv_file)

            # save the full parsed file
            out_path = output_dir / "az1td" / f"{subject_id}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)

            # derive date range from parsed dataframe
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            start_date = ts.min().strftime("%Y-%m-%d")
            end_date = ts.max().strftime("%Y-%m-%d")

            # save train/validate/test splits
            save_chronological_splits(df, subject_id, DATASET, start_date, end_date)

            print(f"Parsed {csv_file} for subject {subject_id}, saved to {out_path}")
            results.append(subject_id)

        except Exception as e:
            print(f"Failed to parse {csv_file}: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse AZ1TD CSV dataset and output processed dataframe"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to input AZ1TD dataset directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to output directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_directory = args.dataset
    output_dir = args.output

    all_data = traverse_dataset(dataset_directory, output_dir)

    for item in all_data:
        print(f"Processed subject: {item}")