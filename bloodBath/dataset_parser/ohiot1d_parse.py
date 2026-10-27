from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
from bloodBath.dataset_parser.save_splits import save_chronological_splits
from bloodBath.core.patient_state import build_feature_frame, ALL_COLUMN_NAMES

DATASET = "ohiot1d"
DATASET_PATH = "/home/desktop/Sneior_Project/datasets/ohiot1d/"
OUTPUT_DIR = "/home/desktop/Sneior_Project/bloodBender/bloodBath/bloodBank/raw/datasets"
TZ = "America/Los_Angeles"

def localize(series):
    return series.dt.tz_localize(
        "America/Los_Angeles",
        ambiguous="NaT",
        nonexistent="shift_forward"
    )

def parse_xml_to_df(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    def parse_dt(x):
        return pd.to_datetime(x, format="%d-%m-%Y %H:%M:%S", errors="coerce")

    # ----------------------------
    # 1) Glucose
    # ----------------------------
    glucose = []
    for ev in root.findall("./glucose_level/event"):
        ts = parse_dt(ev.get("ts"))
        value = ev.get("value")
        glucose.append({
            "timestamp_raw": ts,
            "bg_raw": float(value) if value is not None else np.nan
        })

    glucose_df = pd.DataFrame(glucose).dropna(subset=["timestamp_raw"])
    glucose_df["timestamp_raw"] = localize(glucose_df["timestamp_raw"])

    # put onto 5-minute buckets
    glucose_df["timestamp"] = glucose_df["timestamp_raw"].dt.floor("5min")

    # if multiple readings land in same bucket, average them
    glucose_df = (
        glucose_df.groupby("timestamp", as_index=False)
        .agg(bg=("bg_raw", "mean"))
    )
    glucose_df["bg_observed"] = True

    # ----------------------------
    # 2) Full 5-minute grid
    # ----------------------------
    full_grid = pd.DataFrame({
        "timestamp": pd.date_range(
            start=glucose_df["timestamp"].min(),
            end=glucose_df["timestamp"].max(),
            freq="5min",
            tz=TZ
        )
    })

    df = full_grid.merge(glucose_df, on="timestamp", how="left")
    df["bg_observed"] = df["bg_observed"].fillna(False)

    # mask_bg = True when BG should be ignored
    # i.e. missing/imputed/not originally observed
    df["mask_bg"] = ~df["bg_observed"]

    # fill BG by time interpolation
    df["bg"] = df["bg"].interpolate(method="linear", limit_direction="both")

    # ----------------------------
    # 3) Basal schedule
    # ----------------------------
    basal = []
    for ev in root.findall("./basal/event"):
        ts = parse_dt(ev.get("ts"))
        value = ev.get("value")
        basal.append({
            "timestamp": ts,
            "basal_rate": float(value) if value is not None else np.nan
        })

    basal_df = pd.DataFrame(basal)
    if not basal_df.empty:
        basal_df["timestamp"] = localize(basal_df["timestamp"])
        basal_df = basal_df.sort_values("timestamp")

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            basal_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
    else:
        df["basal_rate"] = np.nan

    # default fill if nothing active yet
    df["basal_rate"] = df["basal_rate"].fillna(0.0)

    # ----------------------------
    # 4) Temp basal overrides
    # ----------------------------
    temp_basal = []
    for ev in root.findall("./temp_basal/event"):
        ts_begin = parse_dt(ev.get("ts_begin"))
        ts_end = parse_dt(ev.get("ts_end"))
        value = ev.get("value")
        temp_basal.append({
            "ts_begin": ts_begin,
            "ts_end": ts_end,
            "value": float(value) if value is not None else np.nan
        })

    temp_df = pd.DataFrame(temp_basal)
    if not temp_df.empty:
        temp_df["ts_begin"] = localize(temp_df["ts_begin"])
        temp_df["ts_end"] = localize(temp_df["ts_end"])

        for _, row in temp_df.iterrows():
            mask = (df["timestamp"] >= row["ts_begin"]) & (df["timestamp"] <= row["ts_end"])
            df.loc[mask, "basal_rate"] = row["value"]

    # ----------------------------
    # 5) Bolus
    # ----------------------------
    bolus = []
    for ev in root.findall("./bolus/event"):
        ts = parse_dt(ev.get("ts_begin"))
        dose = ev.get("dose")
        bolus.append({
            "timestamp": ts,
            "bolus_dose": float(dose) if dose is not None else 0.0
        })

    bolus_df = pd.DataFrame(bolus)
    df["bolus_dose"] = 0.0

    if not bolus_df.empty:
        bolus_df["timestamp"] = localize(bolus_df["timestamp"])
        bolus_df["timestamp"] = bolus_df["timestamp"].dt.floor("5min")

        bolus_df = (
            bolus_df.groupby("timestamp", as_index=False)
            .agg(bolus_dose=("bolus_dose", "sum"))
        )

        df = df.merge(bolus_df, on="timestamp", how="left", suffixes=("", "_new"))
        df["bolus_dose"] = df["bolus_dose_new"].fillna(df["bolus_dose"])
        df = df.drop(columns=["bolus_dose_new"])

    # label mask — True where row can be used in training
    df["mask_label"] = ~df["mask_bg"]

    # optional patient id
    patient_id = root.get("id")

    # All derived features computed from the canonical patient_state module
    df = build_feature_frame(df)
    cols = ["timestamp"] + [c for c in ALL_COLUMN_NAMES if c in df.columns]
    df = df[[c for c in cols if c in df.columns]]

    return df, patient_id


def traverse_dataset(dataset_dir: Path, output_file: Path) -> list[dict]:
    """
    Recursively traverse the directory and parse every XML file.
    """
    dataset_path = dataset_dir

    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dataset_dir}")

    results = []

    for xml_file in dataset_path.rglob("*.xml"):
        df, patient_id = parse_xml_to_df(xml_file)
        file_type = Path(xml_file).parent.name
        out_path = Path(output_file) / f"ohio"/f"{patient_id}_{file_type}.csv"
        
        
        # derive date range from parsed dataframe
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = ts.dropna()

        if ts.empty:
            print(f"Skipping split save for {xml_file}: no valid timestamps")
            continue

        start_date = ts.min().strftime("%Y-%m-%d")
        end_date = ts.max().strftime("%Y-%m-%d")

        # save chronological train/validate/test splits
        save_chronological_splits(df, patient_id, DATASET, start_date, end_date)

        print(f"Parsed {xml_file} for patient {patient_id}, saved to {out_path}")
        results.append(patient_id)

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse XML dataset and output processed dataframe"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to input dataset directory (XML files)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to output file (e.g., CSV)"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    dataset_directory = args.dataset
    output_dir = args.output

    all_data = traverse_dataset(dataset_directory, output_dir)

    for item in all_data:
        print(f"Rows: {item}")