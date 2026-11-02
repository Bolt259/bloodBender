"""
BT-08: Baseline mean-reversion bias audit.

Tests whether the LSTM's 0U baseline prediction shows BG trending toward the
patient's typical range even when current IOB ≈ 0 AND no recent bolus — i.e.,
when there is no physiological mechanism to explain a downward trend.

Usage (from repo root):
    nix develop .#inference --command python scripts/audit_baseline_meanreversion.py

The script loads the latest inference artifacts, finds windows from the merged
test CSVs where:
  - Current BG is elevated (>= HIGH_BG_THRESHOLD mg/dL)
  - Current IOB is negligible (< MAX_IOB_U)
  - No bolus in the last BOLUS_LOOKBACK_STEPS steps
Then runs the 0U baseline prediction and measures predicted_bg at 30/60 min.

A systematic prediction of BG < current_bg in these windows (with no insulin
on board) would confirm the mean-reversion bias hypothesis.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import json
import numpy as np
import csv as csvmod
import itertools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloodTwin.inference.conformal import ConformalPredictor

# Feature schema v1.3/v1.4 — hardcoded to avoid importing bloodBath (which needs arrow)
FEATURE_NAMES = [
    "bg", "basal_rate", "bolus_dose", "basal_delta", "time_since_last_bolus",
    "bg_slope_15min", "bg_slope_30min", "sin_time", "cos_time", "iob",
    "bg_acceleration", "mask_bg", "mask_label", "hypothetical_dose", "site_change",
]

# ── Config ──────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(os.environ.get(
    "BLOODTWIN_ARTIFACTS",
    ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3"
))
DATA_DIR = Path(os.environ.get(
    "BLOODBENDER_DATA",
    ROOT / "bloodBath/bloodBank/merged"
))

HIGH_BG_THRESHOLD   = 180.0  # mg/dL — clearly elevated
LOW_BG_THRESHOLD    = 80.0   # mg/dL — clearly low
MAX_IOB_U           = 0.5    # U — low enough to exclude active correction
BOLUS_LOOKBACK_STEPS = 6     # 30 min back (no recent bolus in last 30 min)
LOOKBACK            = 288    # 24h

IOB_COL             = FEATURE_NAMES.index("iob")
BOLUS_COL           = FEATURE_NAMES.index("bolus_dose")
BG_COL              = FEATURE_NAMES.index("bg")
MASK_LABEL_COL      = FEATURE_NAMES.index("mask_label")


def load_qualifying_windows(pump_id: str, high_bg: bool = True, max_windows: int = 500):
    """
    Load windows where the current timestep (last row) qualifies for the audit:
    - high_bg=True:  current BG >= HIGH_BG_THRESHOLD, IOB < MAX_IOB_U, no recent bolus
    - high_bg=False: current BG <= LOW_BG_THRESHOLD,  IOB < MAX_IOB_U, no recent bolus

    Scans ALL rows in the test CSV, not just the first N.
    """
    import pandas as pd

    pump_dir = DATA_DIR / f"pump_{pump_id}" / "test"
    csvs = sorted(pump_dir.glob("lstm_test_*.csv"))
    if not csvs:
        print(f"  No test CSVs for pump {pump_id}")
        return []

    frames = [pd.read_csv(c, comment="#") for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure all feature columns exist
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0
    if "hypothetical_dose" not in df.columns:
        df["hypothetical_dose"] = df["bolus_dose"].fillna(0.0)

    data_vals = df[FEATURE_NAMES].values.astype(np.float32)
    n = len(df)

    windows = []
    for i in range(LOOKBACK, n):
        # Check current state (last row of window)
        cur_bg  = data_vals[i - 1, BG_COL]
        cur_iob = data_vals[i - 1, IOB_COL]

        if np.isnan(cur_bg):
            continue
        if high_bg and cur_bg < HIGH_BG_THRESHOLD:
            continue
        if not high_bg and cur_bg > LOW_BG_THRESHOLD:
            continue
        if cur_iob >= MAX_IOB_U:
            continue

        # No recent bolus in last BOLUS_LOOKBACK_STEPS
        recent = data_vals[i - BOLUS_LOOKBACK_STEPS : i, BOLUS_COL]
        if np.any(recent > 0):
            continue

        # Build lookback window — NaN-fill missing BG (model uses mask_bg for this)
        window = data_vals[i - LOOKBACK : i].copy()
        window = np.nan_to_num(window, nan=0.0)

        windows.append((df.iloc[i - 1]["timestamp"], window))
        if len(windows) >= max_windows:
            break

    return windows


def run_audit():
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Data:      {DATA_DIR}")
    print()

    predictor = ConformalPredictor(
        onnx_path=ARTIFACTS_DIR / "model.onnx",
        scaler_path=ARTIFACTS_DIR / "scaler.pkl",
        cal_path=ARTIFACTS_DIR / "conformal_cal.pkl",
    )

    pump_ids = ["881235", "901161470", "1539514"]
    results_high = []   # elevated BG, low IOB, no recent bolus
    results_low  = []   # low BG, low IOB, no recent bolus (reverse direction check)

    for pump_id in pump_ids:
        for high_bg in (True, False):
            label = "elevated" if high_bg else "low"
            print(f"Loading pump {pump_id} {label}-BG qualifying windows...")
            windows = load_qualifying_windows(pump_id, high_bg=high_bg, max_windows=500)
            print(f"  {len(windows)} qualifying windows")

            for ts, arr in windows:
                current_bg  = float(arr[-1, BG_COL])
                current_iob = float(arr[-1, IOB_COL])

                try:
                    import pandas as pd
                    pred = predictor.run_raw(arr, window_end_utc=pd.Timestamp(ts))
                except Exception as e:
                    continue

                pred_30  = float(pred.predicted_bg[5])
                pred_60  = float(pred.predicted_bg[11])
                delta_30 = pred_30 - current_bg
                delta_60 = pred_60 - current_bg

                entry = {
                    "pump": pump_id,
                    "timestamp": str(ts),
                    "current_bg": round(current_bg, 1),
                    "iob": round(current_iob, 3),
                    "pred_30": round(pred_30, 1),
                    "pred_60": round(pred_60, 1),
                    "delta_30": round(delta_30, 1),
                    "delta_60": round(delta_60, 1),
                }

                if high_bg:
                    results_high.append(entry)
                else:
                    results_low.append(entry)

    print()
    _report("ELEVATED BG (>= 180 mg/dL), IOB ~ 0, no recent bolus", results_high)
    _report("LOW BG (<= 80 mg/dL), IOB ~ 0, no recent bolus", results_low)

    # Save full results
    out = ROOT / "bloodTwin/analytics/plots/bt08_meanreversion_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"high_bg": results_high, "low_bg": results_low}, f, indent=2)
    print(f"\nFull results → {out}")


def _report(label: str, results: list):
    if not results:
        print(f"{label}: no qualifying windows found")
        return

    deltas_30 = [r["delta_30"] for r in results]
    deltas_60 = [r["delta_60"] for r in results]
    n_down_30 = sum(1 for d in deltas_30 if d < 0)
    n_down_60 = sum(1 for d in deltas_60 if d < 0)

    print(f"{label}")
    print(f"  Windows: {len(results)}")
    print(f"  Mean delta at +30min: {np.mean(deltas_30):+.1f} mg/dL  "
          f"(median {np.median(deltas_30):+.1f})")
    print(f"  Mean delta at +60min: {np.mean(deltas_60):+.1f} mg/dL  "
          f"(median {np.median(deltas_60):+.1f})")
    print(f"  Fraction predicting BG decline at +30min: {n_down_30}/{len(results)} "
          f"({100*n_down_30/len(results):.0f}%)")
    print(f"  Fraction predicting BG decline at +60min: {n_down_60}/{len(results)} "
          f"({100*n_down_60/len(results):.0f}%)")

    # Mean-reversion bias signal: elevated BG with predicted decline despite no insulin
    if "ELEVATED" in label or ">= 180" in label:
        pct = 100 * n_down_60 / len(results)
        if pct > 70:
            print(f"  ⚠ BIAS LIKELY: {pct:.0f}% of elevated-BG/zero-IOB windows "
                  f"predict decline without insulin input.")
        elif pct < 40:
            print(f"  ✓ NO BIAS: only {pct:.0f}% predict decline — model is "
                  f"not systematically mean-reverting.")
        else:
            print(f"  ? AMBIGUOUS: {pct:.0f}% predict decline — inspect individual "
                  f"windows to confirm or rule out bias.")
    print()


if __name__ == "__main__":
    run_audit()
