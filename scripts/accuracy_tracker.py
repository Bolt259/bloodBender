"""
BT-09: Production accuracy tracker.

Computes rolling signed error + MAE per horizon from archived inference
predictions vs actual BG in merged CSVs.

Writes bloodBath/logs/accuracy_summary.json.

Usage (from repo root):
    nix develop .#inference --command python scripts/accuracy_tracker.py
or: nix run .#bloodbender-accuracy
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Runtime data lives in BLOODBENDER_ROOT (data-sync dir in production).
# Auto-detect data-sync as a neighbour of the source repo when not set.
_data_sync = ROOT.parent / "bloodBender-data-sync"
DATA_ROOT = Path(os.environ.get(
    "BLOODBENDER_ROOT",
    str(_data_sync) if _data_sync.exists() else str(ROOT),
))

ARCHIVE_DIR = DATA_ROOT / "bloodBath/logs/inference_archive"
MERGED_DIR  = DATA_ROOT / "bloodBath/bloodBank/merged"
OUT_PATH    = DATA_ROOT / "bloodBath/logs/accuracy_summary.json"
PUMP        = os.environ.get("PUMP", "1539514")

# predicted_bg index (0-based) for each key horizon:
#   index 0 → +5 min, index 5 → +30 min, index 11 → +60 min
HORIZONS: dict[int, int] = {30: 5, 60: 11}
MATCH_TOL_S = 150   # seconds: accept an actual-BG reading within ±2.5 min


def load_bg_lookup(pump_id: str) -> pd.Series:
    """UTC-indexed Series of BG readings from all merged split CSVs."""
    pump_dir = MERGED_DIR / f"pump_{pump_id}"
    frames = []
    for split in ("train", "validate", "test"):
        split_dir = pump_dir / split
        if not split_dir.exists():
            continue
        for csv in sorted(split_dir.glob(f"lstm_{split}_*.csv")):
            frames.append(pd.read_csv(csv, comment="#", usecols=["timestamp", "bg"]))
    if not frames:
        raise FileNotFoundError(f"No merged CSVs for pump {pump_id} in {MERGED_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df["bg"].dropna()


def main() -> None:
    print(f"Loading BG lookup for pump {PUMP} ...")
    bg_series = load_bg_lookup(PUMP)
    print(f"  {len(bg_series)} BG readings "
          f"[{bg_series.index[0].isoformat()} → {bg_series.index[-1].isoformat()}]")

    files = sorted(ARCHIVE_DIR.glob("inference_*.json")) if ARCHIVE_DIR.exists() else []
    if not files:
        print("No archive files found — nothing to compute.")
        return
    print(f"Archive files: {len(files)}")

    errors: dict[int, list[float]] = {h: [] for h in HORIZONS}
    n_matched = 0
    n_skipped = 0

    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        ts_str = data.get("timestamp")
        if not ts_str:
            continue
        pred_ts = pd.Timestamp(ts_str)
        if pred_ts.tzinfo is None:
            pred_ts = pred_ts.tz_localize("UTC")

        fan = data.get("fan", [])
        fan0 = next((r for r in fan if r.get("dose_u", -1) == 0.0), None)
        if fan0 is None:
            continue
        preds = fan0.get("predicted_bg", [])
        if not preds:
            continue

        any_matched = False
        for h_min, idx in HORIZONS.items():
            if idx >= len(preds):
                continue
            target_ts = pred_ts + pd.Timedelta(minutes=h_min)

            pos = bg_series.index.searchsorted(target_ts)
            best: tuple[float, float] | None = None
            for p in (max(0, pos - 1), pos):
                if p >= len(bg_series):
                    continue
                diff = abs((bg_series.index[p] - target_ts).total_seconds())
                if diff <= MATCH_TOL_S:
                    if best is None or diff < best[0]:
                        best = (diff, float(bg_series.iloc[p]))
            if best is None:
                continue
            _, actual_bg = best
            errors[h_min].append(float(preds[idx]) - actual_bg)
            any_matched = True

        if any_matched:
            n_matched += 1
        else:
            n_skipped += 1

    summary: dict = {
        "n_archives": len(files),
        "n_matched": n_matched,
        "n_unmatched": n_skipped,
        "pump": PUMP,
        "horizons": {},
    }

    for h_min, errs in errors.items():
        if not errs:
            summary["horizons"][str(h_min)] = None
            continue
        arr = np.array(errs, dtype=np.float32)
        summary["horizons"][str(h_min)] = {
            "n":        int(len(arr)),
            "mae":      round(float(np.mean(np.abs(arr))), 2),
            "bias":     round(float(np.mean(arr)), 2),
            "rmse":     round(float(np.sqrt(np.mean(arr ** 2))), 2),
            "p90_abs":  round(float(np.percentile(np.abs(arr), 90)), 2),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Written: {OUT_PATH}")
    print(f"  Matched {n_matched}/{len(files)} archives, {n_skipped} unmatched (future BG not yet observed)")
    for h_min, stats in summary["horizons"].items():
        if stats:
            print(f"  @{h_min}min: MAE={stats['mae']:.1f} mg/dL, "
                  f"bias={stats['bias']:+.1f}, RMSE={stats['rmse']:.1f} "
                  f"({stats['n']} samples)")


if __name__ == "__main__":
    main()
