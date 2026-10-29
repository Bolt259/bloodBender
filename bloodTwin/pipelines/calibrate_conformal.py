"""
Calibrate conformal prediction intervals from the held-out validation split.

Runs rolling ONNX inference on all three pumps' val splits, collects
per-horizon absolute residuals, fits a ConformalCalibrator, and saves
conformal_cal.pkl alongside model.onnx / scaler.pkl.

Usage (from repo root, inside the inference nix shell):
    python bloodTwin/pipelines/calibrate_conformal.py [--coverage 0.90] [--stride 6]

Outputs:
    bloodTwin/artifacts/bloodTwin_unified_lstm/conformal_cal.pkl
    bloodTwin/artifacts/bloodTwin_unified_lstm/conformal_summary.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from bloodTwin.inference.conformal import ConformalCalibrator, LOOKBACK, HORIZON

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR  = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR     = PROJECT_ROOT / "bloodBath/bloodBank/merged"
PUMP_IDS       = ["1539514", "901161470", "881235"]


def load_scaler(path: Path):
    with open(path, "rb") as f:
        s = pickle.load(f)
    if isinstance(s, dict):
        return list(s["feature_names"]), np.asarray(s["center_"], dtype=np.float32), np.asarray(s["scale_"], dtype=np.float32)
    return list(getattr(s, "feature_names_in_", [])), np.asarray(s.center_, dtype=np.float32), np.asarray(s.scale_, dtype=np.float32)


def load_val_csv(pump_id: str) -> pd.DataFrame:
    split_dir = MERGED_DIR / f"pump_{pump_id}" / "validate"
    csvs = sorted(split_dir.glob("lstm_validate_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No validate CSVs in {split_dir}")
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Dose-conditioned schema v1.2: CSVs predate hypothetical_dose; it equals
    # the actual bolus delivered each step for calibration windows.
    if "hypothetical_dose" not in df.columns:
        df["hypothetical_dose"] = df["bolus_dose"].fillna(0.0) \
            if "bolus_dose" in df.columns else 0.0
    return df.sort_values("timestamp").reset_index(drop=True)


def collect_residuals(
    df: pd.DataFrame,
    feat_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    sess: ort.InferenceSession,
    input_name: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a [LOOKBACK] window over df with the given stride, run inference,
    and collect absolute residuals at each horizon step.

    Returns
    -------
    residuals  : [n_valid, HORIZON]  absolute errors (NaN where BG not observed)
    bg_stds    : [n_valid]  BG std of lookback window (for volatility scaling)
    """
    feat_arr = df[feat_names].to_numpy(dtype=np.float32)
    bg_arr   = df["bg"].to_numpy(dtype=np.float32)
    mask_label = df["mask_label"].to_numpy(dtype=bool) if "mask_label" in df.columns \
                 else np.ones(len(df), dtype=bool)

    scale_safe = np.where(scale == 0, 1.0, scale)

    all_residuals, all_stds = [], []
    total = len(feat_arr)

    for start in range(0, total - LOOKBACK - HORIZON + 1, stride):
        end = start + LOOKBACK
        window = feat_arr[start:end]

        # Skip if >10% NaN in features
        if np.isnan(window).mean() > 0.10:
            continue

        # Forward pass
        w = pd.DataFrame(window, columns=feat_names).ffill().fillna(0.0).to_numpy(dtype=np.float32)
        w_scaled = ((w - center) / scale_safe).astype(np.float32)
        X = w_scaled[np.newaxis]
        preds = sess.run(None, {input_name: X})[0].reshape(-1).astype(np.float32)

        # Collect per-horizon residuals (NaN where not observed / masked)
        residuals_row = np.full(HORIZON, np.nan, dtype=np.float32)
        for h in range(HORIZON):
            idx = end + h
            if idx >= total:
                break
            if not mask_label[idx]:
                continue
            a = bg_arr[idx]
            if not np.isnan(a):
                residuals_row[h] = abs(a - preds[h])

        all_residuals.append(residuals_row)

        # BG std of lookback window (for volatility scaling)
        bg_win = bg_arr[start:end]
        bg_std = float(np.nanstd(bg_win)) if not np.all(np.isnan(bg_win)) else np.nan
        all_stds.append(bg_std)

    return np.array(all_residuals, dtype=np.float32), np.array(all_stds, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage",   type=float, default=0.90)
    ap.add_argument("--stride",     type=int,   default=6,
                    help="Row stride for rolling inference (6 = every 30 min)")
    ap.add_argument("--artifacts", "--model", dest="artifacts", default=str(ARTIFACTS_DIR))
    ap.add_argument("--pumps",      nargs="+",  default=PUMP_IDS,
                    help="Pump serials to include in calibration set")
    ap.add_argument("--max-buffer", type=int,   default=None,
                    help="Keep only the most recent N residuals (online/sliding window). "
                         "Recommended: ~2000 for ~6 weeks of 30-min stride data.")
    args = ap.parse_args()

    art = Path(args.artifacts)
    onnx_path   = art / "model.onnx"
    scaler_path = art / "scaler.pkl"
    cal_out     = art / "conformal_cal.pkl"
    summary_out = art / "conformal_summary.json"

    log.info(f"Loading model from {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    log.info(f"Loading scaler from {scaler_path}")
    feat_names, center, scale = load_scaler(scaler_path)
    log.info(f"  {len(feat_names)} features: {feat_names}")

    all_residuals, all_stds = [], []

    for pump_id in args.pumps:
        log.info(f"Loading val split for pump {pump_id} ...")
        try:
            df = load_val_csv(pump_id)
        except FileNotFoundError as e:
            log.warning(f"  Skipping: {e}")
            continue
        log.info(f"  {len(df)} rows  [{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}]")

        log.info(f"  Running rolling inference (stride={args.stride}) ...")
        res, stds = collect_residuals(df, feat_names, center, scale, sess, input_name, args.stride)
        valid = ~np.isnan(res).all(axis=1)
        log.info(f"  {valid.sum()} valid windows (of {len(res)} total)")
        all_residuals.append(res[valid])
        all_stds.append(stds[valid])

    if not all_residuals:
        raise RuntimeError("No calibration data collected — check val split paths")

    residuals_all = np.concatenate(all_residuals, axis=0)   # [N, HORIZON]
    stds_all      = np.concatenate(all_stds,      axis=0)   # [N]
    log.info(f"Total calibration windows: {len(residuals_all)}")

    log.info(f"Fitting ConformalCalibrator at {args.coverage:.0%} coverage "
             f"(max_buffer={args.max_buffer}) ...")
    cal = ConformalCalibrator(coverage=args.coverage)
    cal.fit(residuals_all, cal_bg_stds=stds_all, max_buffer=args.max_buffer)

    summary = cal.summary()
    log.info("Calibration summary:")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    cal.save(cal_out)
    log.info(f"Saved: {cal_out}")

    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved: {summary_out}")

    # Quick empirical coverage check on the calibration set itself (optimistic — same data)
    mins = list(range(5, (HORIZON + 1) * 5, 5))
    log.info("\nEmpirical coverage on cal set (optimistic, same data):")
    log.info(f"  {'horizon':>8}  {'q_hat':>7}  {'coverage':>9}")
    for h, (m, q) in enumerate(zip(mins, cal.q_hat)):
        col = residuals_all[:, h]
        col = col[~np.isnan(col)]
        cov = float(np.mean(col <= q)) if len(col) > 0 else float("nan")
        marker = " ←" if m in (30, 60) else ""
        log.info(f"  {m:>5} min  {q:>7.2f}  {cov:>8.1%}{marker}")


if __name__ == "__main__":
    main()
