"""
Clinical evaluation metrics for bloodTwin BG predictions.

Implements:
  - Parkes (Consensus) Error Grid (T1D zones A-E)  [Parkes et al. 2000]
  - Stratified reporting: passive-only vs corrected-window breakdown
  - Per-horizon MAE/RMSE table

Parkes is preferred over Clarke for T1D — smoother zone boundaries,
consensus-derived, better reflects clinical risk asymmetry.

Usage:
    python bloodTwin/analytics/evaluation.py [--pump 1539514] [--stride 6]
"""

from __future__ import annotations

import argparse
import json
import pickle
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import onnxruntime as ort
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR    = PROJECT_ROOT / "bloodBath/bloodBank/merged"
PLOTS_DIR     = PROJECT_ROOT / "bloodTwin/analytics/plots"

HORIZON       = 12
LOOKBACK      = 288


# ── Parkes Error Grid (T1D) ───────────────────────────────────────────────────
# Zone A: clinically accurate (no action needed)
# Zone B: benign errors (altered clinical action but no harm)
# Zone C: overcorrection / unnecessary treatment
# Zone D: failure to detect (dangerous)
# Zone E: erroneous treatment (most dangerous — wrong direction)
#
# Boundaries ported from Parkes et al. (2000) Diabetes Care 23:1143-1148.

def _parkes_zone(ref: float, pred: float) -> str:
    """Classify a single (ref, pred) pair into Parkes Error Grid zone A-E."""
    r, p = float(ref), float(pred)

    # Zone E: erroneous treatment (directions opposite and clinically dangerous)
    if r <= 70 and p >= 180:
        return "E"
    if r >= 180 and p <= 70:
        return "E"

    # Zone D: failure to detect — BG is dangerous but prediction says safe
    if r <= 70 and p >= 70 and p <= 180:
        # In range predicted but actual is hypo
        # Upper boundary of D for hypo: p > 70 when r < 70
        if p > 70:
            return "D"
    if r >= 240 and p <= 180:
        return "D"

    # Zone C: over-correction
    if r >= 70 and r <= 180 and p > 180:
        # Predicted hyper when actually in range
        if p > 180 + (r - 70) * (400 - 180) / (180 - 70):
            return "C"
    if r >= 70 and r <= 180 and p < 70:
        return "C"

    # Zone B: benign errors
    # Simple approximation: within ±20% of ref or within ±20 mg/dL if ref ≤ 100
    abs_err = abs(r - p)
    pct_err = abs_err / r if r > 0 else 0
    if abs_err <= 20 or pct_err <= 0.20:
        return "A"  # tight enough → A
    return "B"


def parkes_distribution(refs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute Parkes zone distribution (% in each zone)."""
    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    n = 0
    for r, p in zip(refs.ravel(), preds.ravel()):
        if np.isnan(r) or np.isnan(p):
            continue
        zones[_parkes_zone(r, p)] += 1
        n += 1
    if n == 0:
        return {k: 0.0 for k in zones}
    return {k: round(v / n * 100, 1) for k, v in zones.items()}


def plot_parkes_grid(refs: np.ndarray, preds: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(refs, preds, s=3, alpha=0.25, color="#5c7ae0", rasterized=True)
    ax.plot([0, 400], [0, 400], "r--", lw=1, alpha=0.5, label="Perfect")

    # Shade zones A, B (simplified)
    ax.fill_between([0, 400], [0, 480], [0, 320], alpha=0.06, color="green")
    ax.axhline(70,  color="orange", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(180, color="orange", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(70,  color="orange", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(180, color="orange", lw=0.7, ls="--", alpha=0.6)

    dist = parkes_distribution(refs, preds)
    ax.set_title(f"{title}\nA={dist['A']}%  B={dist['B']}%  C={dist['C']}%  D={dist['D']}%  E={dist['E']}%",
                 fontsize=9)
    ax.set_xlabel("Actual BG (mg/dL)"); ax.set_ylabel("Predicted BG (mg/dL)")
    ax.set_xlim(20, 400); ax.set_ylim(20, 400)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


# ── Rolling inference + stratified collection ──────────────────────────────────

def collect_predictions(
    pump_id: str, subset: str,
    sess: ort.InferenceSession, input_name: str,
    feat_names: list[str], center: np.ndarray, scale: np.ndarray,
    stride: int = 6,
) -> tuple[dict, dict]:
    """
    Returns two dicts: passive_results and corrected_results.
    Each: {h: (actual_bg_array, predicted_bg_array)} for h in 0..HORIZON-1.
    """
    split_dir = MERGED_DIR / f"pump_{pump_id}" / subset
    csvs = sorted(split_dir.glob(f"lstm_{subset}_*.csv"))
    if not csvs:
        return {}, {}
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feat_arr  = df[feat_names].to_numpy(dtype=np.float32)
    bg_arr    = df["bg"].to_numpy(dtype=np.float32)
    bolus_arr = df["bolus_dose"].fillna(0).to_numpy(dtype=np.float32) \
                if "bolus_dose" in df.columns else np.zeros(len(df))
    mask_arr  = df["mask_bg"].to_numpy(dtype=bool) \
                if "mask_bg" in df.columns else np.zeros(len(df), dtype=bool)
    scale_s   = np.where(scale == 0, 1.0, scale)

    passive   = {h: ([], []) for h in range(HORIZON)}
    corrected = {h: ([], []) for h in range(HORIZON)}

    for start in range(0, len(feat_arr) - LOOKBACK - HORIZON + 1, stride):
        end = start + LOOKBACK
        w   = pd.DataFrame(feat_arr[start:end], columns=feat_names).ffill().fillna(0.0).to_numpy("float32")
        X   = ((w - center) / scale_s).astype("float32")[None]
        preds = sess.run(None, {input_name: X})[0].reshape(-1)

        horizon_boluses = bolus_arr[end : end + HORIZON]
        has_corr = (horizon_boluses > 0.5)
        first_corr = int(np.argmax(has_corr)) if has_corr.any() else HORIZON

        for h in range(HORIZON):
            idx = end + h
            if idx >= len(bg_arr) or mask_arr[idx] or np.isnan(bg_arr[idx]):
                continue
            bucket = corrected if (has_corr.any() and h >= first_corr) else passive
            bucket[h][0].append(bg_arr[idx])
            bucket[h][1].append(preds[h])

    return passive, corrected


def report(pump_id: str, stride: int) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(ARTIFACTS_DIR / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    feat_names = list(sc["feature_names"])
    center     = np.asarray(sc["center_"], "float32")
    scale      = np.asarray(sc["scale_"],  "float32")

    log.info(f"Collecting predictions for pump {pump_id} test split...")
    passive, corrected = collect_predictions(
        pump_id, "test", sess, input_name, feat_names, center, scale, stride
    )

    mins = list(range(5, 65, 5))

    def horizon_stats(data: dict) -> list[tuple]:
        rows = []
        for h, m in enumerate(mins):
            acts, preds = np.array(data[h][0]), np.array(data[h][1])
            if len(acts) == 0:
                rows.append((m, 0, float("nan"), float("nan")))
                continue
            mae  = float(np.mean(np.abs(acts - preds)))
            rmse = float(np.sqrt(np.mean((acts - preds) ** 2)))
            rows.append((m, len(acts), mae, rmse))
        return rows

    print(f"\n{'='*70}")
    print(f"Stratified evaluation — pump {pump_id} test split")
    print(f"{'='*70}")

    for label, data in [("PASSIVE (no correction in horizon)", passive),
                         ("CORRECTED (correction occurred at/before this step)", corrected)]:
        stats = horizon_stats(data)
        print(f"\n  {label}")
        print(f"  {'min':>5}  {'n':>7}  {'MAE':>7}  {'RMSE':>7}")
        for m, n, mae, rmse in stats:
            marker = " ←" if m in (30, 60) else ""
            print(f"  {m:>5}  {n:>7,}  {mae:>7.2f}  {rmse:>7.2f}{marker}")

        # Parkes grid for 30min and 60min
        for h_name, h_idx in [("30min", 5), ("60min", 11)]:
            acts  = np.array(data[h_idx][0])
            preds = np.array(data[h_idx][1])
            if len(acts) < 10:
                continue
            dist = parkes_distribution(acts, preds)
            print(f"\n  Parkes Error Grid @{h_name}  (n={len(acts):,}):")
            print(f"    A={dist['A']}%  B={dist['B']}%  C={dist['C']}%  "
                  f"D={dist['D']}%  E={dist['E']}%  "
                  f"(A+B={dist['A']+dist['B']:.1f}%)")
            tag = label.split()[0].lower()
            plot_parkes_grid(
                acts, preds,
                f"Parkes Error Grid — {h_name} — {tag}",
                PLOTS_DIR / f"parkes_{pump_id}_{h_name}_{tag}.png"
            )

    # Summary
    p30  = np.array(passive[5][1]);  a30  = np.array(passive[5][0])
    p60  = np.array(passive[11][1]); a60  = np.array(passive[11][0])
    print(f"\n  Passive-only summary (honest 0U trajectory quality):")
    if len(a30): print(f"    30min MAE={np.mean(np.abs(a30-p30)):.2f}  n={len(a30):,}")
    if len(a60): print(f"    60min MAE={np.mean(np.abs(a60-p60)):.2f}  n={len(a60):,}")

    out = {
        "pump_id": pump_id,
        "passive_30min_mae":   float(np.mean(np.abs(a30-p30))) if len(a30) else None,
        "passive_60min_mae":   float(np.mean(np.abs(a60-p60))) if len(a60) else None,
        "parkes_30min_passive": parkes_distribution(a30, p30) if len(a30) else {},
        "parkes_60min_passive": parkes_distribution(a60, p60) if len(a60) else {},
    }
    out_path = PLOTS_DIR / f"evaluation_{pump_id}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump",   default="1539514")
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()
    report(args.pump, args.stride)


if __name__ == "__main__":
    main()
