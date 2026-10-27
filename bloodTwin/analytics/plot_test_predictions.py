"""
Plot actual vs predicted BG on the held-out test split.

Runs rolling ONNX inference across the test CSV, collecting horizon-wise
predictions, then produces:
  1. A time-series plot: actual BG + predicted ribbon (30 / 60 min horizons)
  2. Per-horizon MAE / RMSE bar charts
  3. Scatter: actual vs predicted at 30 and 60 min
  4. Error distribution (residual histogram) at 30 and 60 min

Usage (from repo root, inside the inference nix shell):
  python bloodTwin/analytics/plot_test_predictions.py [--pump 1539514] [--days 7]

Outputs PNGs to bloodTwin/analytics/plots/.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import onnxruntime as ort
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR    = PROJECT_ROOT / "bloodBath/bloodBank/merged"
PLOTS_DIR     = PROJECT_ROOT / "bloodTwin/analytics/plots"

LOOKBACK = 288   # 24 h @ 5 min
HORIZON  = 12    # 60 min
STEP_MIN = 5
HORIZONS_PLOT = {
    "30min": 5,   # step index 5 (0-based) = 6th step × 5min = 30 min
    "60min": 11,  # step index 11 = 12th step × 5min = 60 min
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_scaler(path: Path):
    with open(path, "rb") as f:
        s = pickle.load(f)
    if isinstance(s, dict):
        return (
            s["feature_names"],
            np.asarray(s["center_"], dtype=np.float32),
            np.asarray(s["scale_"],  dtype=np.float32),
        )
    return (
        list(getattr(s, "feature_names_in_", [])),
        np.asarray(s.center_, dtype=np.float32),
        np.asarray(s.scale_,  dtype=np.float32),
    )


def load_test_csv(pump_id: str) -> pd.DataFrame:
    split_dir = MERGED_DIR / f"pump_{pump_id}" / "test"
    csvs = sorted(split_dir.glob("lstm_test_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No test CSVs in {split_dir}")
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_rolling_inference(df, feat_names, center, scale, sess, stride=6):
    """
    Slide a [LOOKBACK] window across df in steps of `stride` rows.
    Returns a DataFrame of predictions indexed by the window's end timestamp.

    stride=6 → one prediction every 30 min (keeps runtime manageable; full
    stride=1 would give ~11k forward passes on 39 days of data).
    """
    feat_arr = df[feat_names].to_numpy(dtype=np.float32)
    ts_arr   = df["timestamp"].to_numpy()
    bg_arr   = df["bg"].to_numpy(dtype=np.float32)  # NaN where imputed/missing

    total = len(feat_arr)
    records = []

    for start in range(0, total - LOOKBACK - HORIZON + 1, stride):
        end = start + LOOKBACK
        window = feat_arr[start:end]

        # skip if >10% of feature values are NaN
        if np.isnan(window).mean() > 0.1:
            continue

        # scale (forward-fill + zero-fill NaNs before scaling)
        w = pd.DataFrame(window, columns=feat_names).ffill().fillna(0.0).to_numpy(dtype=np.float32)
        w_scaled = (w - center) / np.where(scale == 0, 1.0, scale)
        X = w_scaled[np.newaxis].astype(np.float32)  # [1, T, F]

        preds = sess.run(None, {sess.get_inputs()[0].name: X})[0].reshape(-1)  # [H]

        # collect actual BG for the horizon steps
        actuals = bg_arr[end : end + HORIZON]
        anchor_ts = ts_arr[end]  # timestamp of the first predicted step

        records.append({
            "anchor_ts": pd.Timestamp(anchor_ts),
            "preds":     preds,          # array [H]
            "actuals":   actuals,        # array [H], NaN where not observed
        })

    return records


def compute_metrics(records, horizon_steps):
    """Compute MAE and RMSE for each horizon step index."""
    maes, rmses = [], []
    for h in range(horizon_steps):
        ys, yhats = [], []
        for r in records:
            a = r["actuals"][h] if h < len(r["actuals"]) else np.nan
            p = r["preds"][h]   if h < len(r["preds"])   else np.nan
            if not np.isnan(a) and not np.isnan(p):
                ys.append(a); yhats.append(p)
        if ys:
            err = np.array(ys) - np.array(yhats)
            maes.append(np.mean(np.abs(err)))
            rmses.append(np.sqrt(np.mean(err**2)))
        else:
            maes.append(np.nan); rmses.append(np.nan)
    return np.array(maes), np.array(rmses)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_time_series(records, days_shown, pump_id, out_dir):
    if not records:
        return
    cutoff = records[-1]["anchor_ts"] - pd.Timedelta(days=days_shown)
    subset = [r for r in records if r["anchor_ts"] >= cutoff]
    if not subset:
        subset = records[-min(400, len(records)):]

    ts_30  = [r["anchor_ts"] + pd.Timedelta(minutes=30) for r in subset]
    ts_60  = [r["anchor_ts"] + pd.Timedelta(minutes=60) for r in subset]
    pred30 = [r["preds"][HORIZONS_PLOT["30min"]] for r in subset]
    pred60 = [r["preds"][HORIZONS_PLOT["60min"]] for r in subset]
    act30  = [r["actuals"][HORIZONS_PLOT["30min"]] for r in subset]
    act60  = [r["actuals"][HORIZONS_PLOT["60min"]] for r in subset]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    fig.suptitle(f"bloodTwin — Actual vs Predicted BG  (pump {pump_id}, last {days_shown}d of test split)", fontsize=13)

    for ax, ts_pred, preds, actuals, label, color in [
        (axes[0], ts_30, pred30, act30, "30-min horizon", "#e05c5c"),
        (axes[1], ts_60, pred60, act60, "60-min horizon", "#5c7ae0"),
    ]:
        ax.plot(ts_pred, actuals, color="#444", lw=1.0, label="Actual BG", zorder=3)
        ax.plot(ts_pred, preds,   color=color,  lw=1.2, alpha=0.85, label=f"Predicted ({label})", zorder=2)
        ax.axhspan(70, 180, alpha=0.07, color="green", label="Target range (70–180)")
        ax.axhline(70,  color="orange", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(180, color="orange", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylabel("BG (mg/dL)")
        ax.set_ylim(20, 400)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"timeseries_{pump_id}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_mae_per_horizon(maes, rmses, pump_id, out_dir):
    mins = np.arange(1, HORIZON + 1) * STEP_MIN
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mins - 1.2, maes,  width=2.0, label="MAE",  color="#5c7ae0", alpha=0.85)
    ax.bar(mins + 1.2, rmses, width=2.0, label="RMSE", color="#e05c5c", alpha=0.85)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Error (mg/dL)")
    ax.set_title(f"bloodTwin — MAE & RMSE per horizon  (pump {pump_id} test split)")
    ax.set_xticks(mins)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # annotate 30 and 60 min
    for h_label, h_step in HORIZONS_PLOT.items():
        ax.annotate(f"{h_label}\nMAE={maes[h_step]:.1f}", xy=(mins[h_step], maes[h_step]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color="#5c7ae0")
    plt.tight_layout()
    path = out_dir / f"mae_per_horizon_{pump_id}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_scatter(records, pump_id, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"bloodTwin — Actual vs Predicted scatter  (pump {pump_id})", fontsize=12)
    for ax, (h_label, h_step) in zip(axes, HORIZONS_PLOT.items()):
        ys, yhats = [], []
        for r in records:
            a = r["actuals"][h_step] if h_step < len(r["actuals"]) else np.nan
            p = r["preds"][h_step]
            if not np.isnan(a):
                ys.append(a); yhats.append(p)
        ys = np.array(ys); yhats = np.array(yhats)
        mae  = np.mean(np.abs(ys - yhats))
        rmse = np.sqrt(np.mean((ys - yhats)**2))
        lo, hi = max(20, min(ys.min(), yhats.min()) - 10), min(400, max(ys.max(), yhats.max()) + 10)
        ax.scatter(ys, yhats, s=2, alpha=0.25, color="#5c7ae0", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect")
        ax.set_xlabel("Actual BG (mg/dL)"); ax.set_ylabel("Predicted BG (mg/dL)")
        ax.set_title(f"{h_label}  MAE={mae:.1f}  RMSE={rmse:.1f}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / f"scatter_{pump_id}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_error_hist(records, pump_id, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"bloodTwin — Residual distribution  (pump {pump_id})", fontsize=12)
    for ax, (h_label, h_step) in zip(axes, HORIZONS_PLOT.items()):
        errs = []
        for r in records:
            a = r["actuals"][h_step] if h_step < len(r["actuals"]) else np.nan
            p = r["preds"][h_step]
            if not np.isnan(a):
                errs.append(a - p)
        errs = np.array(errs)
        ax.hist(errs, bins=60, color="#5c7ae0", alpha=0.8, edgecolor="none")
        ax.axvline(0,              color="red",    lw=1.2, ls="--")
        ax.axvline(np.mean(errs),  color="orange", lw=1.0, ls="-", label=f"mean={np.mean(errs):.1f}")
        ax.set_xlabel("Residual: actual − predicted (mg/dL)")
        ax.set_ylabel("Count")
        ax.set_title(f"{h_label}  std={np.std(errs):.1f} mg/dL")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / f"residuals_{pump_id}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump",   default="1539514", help="Pump serial to evaluate")
    ap.add_argument("--days",   type=int, default=7, help="Days of test data to show in time-series plot")
    ap.add_argument("--stride", type=int, default=6, help="Row stride for rolling inference (6=every 30min)")
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading scaler from {ARTIFACTS_DIR}/scaler.pkl ...")
    feat_names, center, scale = load_scaler(ARTIFACTS_DIR / "scaler.pkl")
    print(f"  Features ({len(feat_names)}): {feat_names}")

    print(f"Loading test split for pump {args.pump} ...")
    df = load_test_csv(args.pump)
    print(f"  {len(df)} rows  [{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}]")

    print(f"Loading ONNX model from {ARTIFACTS_DIR}/model.onnx ...")
    sess = ort.InferenceSession(
        str(ARTIFACTS_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    print(f"Running rolling inference (stride={args.stride} rows = {args.stride*5} min) ...")
    records = run_rolling_inference(df, feat_names, center, scale, sess, stride=args.stride)
    print(f"  Generated {len(records)} prediction windows")

    print("Computing per-horizon metrics ...")
    maes, rmses = compute_metrics(records, HORIZON)
    mins = np.arange(1, HORIZON + 1) * STEP_MIN
    print(f"\n  {'Horizon':>8}  {'MAE':>7}  {'RMSE':>7}")
    for m, mae, rmse in zip(mins, maes, rmses):
        marker = " ←" if m in (30, 60) else ""
        print(f"  {m:>5} min  {mae:>7.2f}  {rmse:>7.2f}{marker}")

    print("\nGenerating plots ...")
    plot_time_series(records, args.days, args.pump, PLOTS_DIR)
    plot_mae_per_horizon(maes, rmses, args.pump, PLOTS_DIR)
    plot_scatter(records, args.pump, PLOTS_DIR)
    plot_error_hist(records, args.pump, PLOTS_DIR)

    print(f"\nDone. Plots written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
