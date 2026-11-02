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

from bloodBath.core.patient_state import build_feature_frame, FEATURE_NAMES
from bloodTwin.inference.conformal import ConformalPredictor

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
MAX_IOB_U           = 0.1    # U — "no meaningful IOB"
BOLUS_LOOKBACK_STEPS = 18    # 90 min back from current step
LOOKBACK            = 288    # 24h

IOB_COL             = FEATURE_NAMES.index("iob")
BOLUS_COL           = FEATURE_NAMES.index("bolus_dose")
BG_COL              = FEATURE_NAMES.index("bg")
MASK_LABEL_COL      = FEATURE_NAMES.index("mask_label")


def load_test_windows(pump_id: str, max_windows: int = 2000):
    """
    Load up to max_windows rows from the test CSV for pump_id.
    Returns list of (timestamp, DataFrame[LOOKBACK rows]) ready for inference.
    """
    pump_dir = DATA_DIR / f"pump_{pump_id}" / "test"
    csvs = sorted(pump_dir.glob("lstm_test_*.csv"))
    if not csvs:
        print(f"  No test CSVs for pump {pump_id}")
        return []

    import pandas as pd
    frames = [pd.read_csv(c, comment="#") for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    windows = []
    feat_cols = [f for f in FEATURE_NAMES if f in df.columns]
    if len(feat_cols) < len(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(feat_cols)
        for m in missing:
            df[m] = 0.0

    for i in range(LOOKBACK, len(df)):
        window = df.iloc[i - LOOKBACK : i][FEATURE_NAMES].values.astype(np.float32)
        if np.isnan(window[:, BG_COL]).any():
            continue
        if not df.iloc[i - 1].get("mask_label", True):
            continue
        windows.append((df.iloc[i - 1]["timestamp"], window))
        if len(windows) >= max_windows:
            break

    return windows


def run_audit():
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Data:      {DATA_DIR}")
    print()

    predictor = ConformalPredictor(str(ARTIFACTS_DIR))

    pump_ids = ["881235", "901161470", "1539514"]
    results_high = []   # elevated BG, low IOB, no recent bolus
    results_low  = []   # low BG, low IOB, no recent bolus (reverse direction check)

    for pump_id in pump_ids:
        print(f"Loading pump {pump_id} test windows...")
        windows = load_test_windows(pump_id)
        print(f"  {len(windows)} windows loaded")

        for ts, arr in windows:
            last = arr[-1]
            current_bg  = float(last[BG_COL])
            current_iob = float(last[IOB_COL])

            # Skip if IOB meaningful
            if current_iob >= MAX_IOB_U:
                continue

            # Skip if recent bolus
            recent_bolus = arr[-BOLUS_LOOKBACK_STEPS:, BOLUS_COL]
            if np.any(recent_bolus > 0):
                continue

            try:
                pred = predictor.run_raw(arr)
            except Exception as e:
                continue

            pred_30  = float(pred.predicted_bg[5])   # +30 min
            pred_60  = float(pred.predicted_bg[11])  # +60 min
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

            if current_bg >= HIGH_BG_THRESHOLD:
                results_high.append(entry)
            elif current_bg <= LOW_BG_THRESHOLD:
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
