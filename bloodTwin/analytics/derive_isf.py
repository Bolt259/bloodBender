"""
Derive patient-specific ISF (Insulin Sensitivity Factor) from pump history.

ISF = how many mg/dL one unit of insulin lowers BG, for this patient.

Method:
  For each isolated correction bolus (no meal within ±90 min, BG above
  target at delivery), measure the BG nadir reached within 2-4 hours.
  ISF estimate = (BG_at_bolus − BG_nadir) / dose_units

Aggregation:
  Rolling 30-day median (robust to outliers).  The output is a time-series
  of ISF estimates and a summary suitable for use as a derived patient config
  parameter by the inference service.

Usage (from repo root, inference nix shell):
    python bloodTwin/analytics/derive_isf.py [--pump 1539514] [--plot]

Outputs:
    bloodTwin/analytics/isf/isf_<pump_id>.json   — summary stats + time series
    bloodTwin/analytics/isf/isf_<pump_id>.png    — rolling ISF plot (if --plot)
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR   = PROJECT_ROOT / "bloodBath/bloodBank/merged"
OUT_DIR      = PROJECT_ROOT / "bloodTwin/analytics/isf"

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_BOLUS_U       = 0.5    # ignore tiny boluses (basal micro-boluses, noise)
MAX_BOLUS_U       = 10.0   # cap — very large boluses are meal corrections, not pure ISF
MIN_BG_AT_BOLUS   = 140.0  # only use correction boluses delivered when BG is elevated
NADIR_WINDOW_MIN  = 90     # look for BG nadir within 90 min of bolus
MEAL_EXCLUSION_MIN = 90    # exclude boluses with another bolus within this window
ROLLING_DAYS      = 30     # rolling window for ISF estimate
MIN_SAMPLES       = 5      # need at least this many events in a rolling window


def load_merged_data(pump_id: str) -> pd.DataFrame:
    """Load ALL split files for a pump (train+val+test) into one sorted DataFrame."""
    frames = []
    for subset in ("train", "validate", "test"):
        d = MERGED_DIR / f"pump_{pump_id}" / subset
        for csv in sorted(d.glob(f"lstm_{subset}_*.csv")):
            df = pd.read_csv(csv, comment="#")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found for pump {pump_id}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    log.info(f"Loaded {len(combined):,} rows for pump {pump_id}  "
             f"[{combined['timestamp'].iloc[0]} → {combined['timestamp'].iloc[-1]}]")
    return combined


def find_correction_boluses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify correction boluses: isolated (no other bolus within ±meal_exclusion_min),
    delivered when BG ≥ MIN_BG_AT_BOLUS, within the plausible correction dose range.

    Returns a DataFrame of candidate events with columns:
        timestamp, bolus_dose, bg_at_bolus, nadir_bg, isf_estimate
    """
    STEP = 5  # minutes per row
    NADIR_STEPS = NADIR_WINDOW_MIN // STEP          # rows to look ahead for nadir
    MEAL_EXCL_STEPS = MEAL_EXCLUSION_MIN // STEP    # isolation window in rows

    bolus_mask = (df["bolus_dose"] >= MIN_BOLUS_U) & (df["bolus_dose"] <= MAX_BOLUS_U)
    bolus_idx  = df.index[bolus_mask].tolist()

    records = []
    for i in bolus_idx:
        dose = df.at[i, "bolus_dose"]
        bg   = df.at[i, "bg"]
        if pd.isna(bg) or bg < MIN_BG_AT_BOLUS:
            continue

        # Check isolation: no other bolus within MEAL_EXCL_STEPS before or after
        lo = max(0, i - MEAL_EXCL_STEPS)
        hi = min(len(df) - 1, i + MEAL_EXCL_STEPS)
        nearby_boluses = df.loc[lo:hi, "bolus_dose"]
        # Count boluses other than the one at i
        other_boluses = nearby_boluses.drop(index=i, errors="ignore")
        if (other_boluses >= MIN_BOLUS_U).any():
            continue  # not isolated — likely a meal bolus

        # Measure BG nadir in the next NADIR_STEPS rows (observed BG only)
        future_slice = df.loc[i + 1 : i + NADIR_STEPS]
        obs_bg = future_slice["bg"].dropna()
        if obs_bg.empty or len(obs_bg) < 3:
            continue  # need a few readings to trust the nadir
        nadir_bg = float(obs_bg.min())

        # BG must actually drop for this to be a valid estimate
        bg_drop = float(bg) - nadir_bg
        if bg_drop <= 10.0:
            continue  # negligible drop — site issue or low sensitivity

        isf_est = bg_drop / float(dose)
        records.append({
            "timestamp":    df.at[i, "timestamp"],
            "bolus_dose":   dose,
            "bg_at_bolus":  float(bg),
            "nadir_bg":     nadir_bg,
            "bg_drop":      bg_drop,
            "isf_estimate": isf_est,
        })

    events = pd.DataFrame(records)
    log.info(f"Found {len(events)} correction bolus events")
    return events


def rolling_isf(events: pd.DataFrame, rolling_days: int = ROLLING_DAYS) -> pd.DataFrame:
    """
    Compute rolling median ISF over a rolling_days window (minimum MIN_SAMPLES events).
    Returns the events DataFrame with a 'rolling_isf' column added.
    """
    if events.empty:
        return events
    events = events.sort_values("timestamp").copy()
    events["rolling_isf"] = np.nan

    window = pd.Timedelta(days=rolling_days)
    for i, row in events.iterrows():
        t = row["timestamp"]
        mask = (events["timestamp"] >= t - window) & (events["timestamp"] <= t)
        window_data = events.loc[mask, "isf_estimate"]
        if len(window_data) >= MIN_SAMPLES:
            events.at[i, "rolling_isf"] = float(window_data.median())

    return events


def summarise(events: pd.DataFrame, pump_id: str) -> dict:
    """Produce a JSON-serialisable summary of ISF estimates."""
    if events.empty or events["rolling_isf"].isna().all():
        return {"pump_id": pump_id, "status": "insufficient_data"}

    rolling_vals = events["rolling_isf"].dropna()
    all_vals     = events["isf_estimate"]

    # Recent ISF: median of last 90 days of events
    latest_ts = events["timestamp"].max()
    recent = events[events["timestamp"] >= latest_ts - pd.Timedelta(days=90)]
    recent_isf = float(recent["isf_estimate"].median()) if len(recent) >= MIN_SAMPLES else None

    return {
        "pump_id":              pump_id,
        "status":               "ok",
        "n_events":             len(events),
        "date_range":           [str(events["timestamp"].min().date()),
                                 str(events["timestamp"].max().date())],
        "isf_overall_median":   round(float(all_vals.median()), 1),
        "isf_overall_p25_p75":  [round(float(all_vals.quantile(0.25)), 1),
                                  round(float(all_vals.quantile(0.75)), 1)],
        "isf_recent_90d":       round(recent_isf, 1) if recent_isf else None,
        "isf_latest_rolling":   round(float(rolling_vals.iloc[-1]), 1),
        "rolling_window_days":  ROLLING_DAYS,
        "method":               "correction_bolus_nadir",
        "params": {
            "min_bolus_u":         MIN_BOLUS_U,
            "max_bolus_u":         MAX_BOLUS_U,
            "min_bg_at_bolus":     MIN_BG_AT_BOLUS,
            "nadir_window_min":    NADIR_WINDOW_MIN,
            "meal_exclusion_min":  MEAL_EXCLUSION_MIN,
        },
    }


def plot_isf(events: pd.DataFrame, pump_id: str, out_path: Path) -> None:
    if events.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(f"ISF derivation from correction boluses  (pump {pump_id})", fontsize=12)

    ax = axes[0]
    ax.scatter(events["timestamp"], events["isf_estimate"],
               s=8, alpha=0.4, color="#5c7ae0", label="individual event")
    valid_roll = events.dropna(subset=["rolling_isf"])
    ax.plot(valid_roll["timestamp"], valid_roll["rolling_isf"],
            color="#e05c5c", lw=1.8, label=f"{ROLLING_DAYS}d rolling median")
    ax.set_ylabel("ISF (mg/dL / U)")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    ax2 = axes[1]
    ax2.hist(events["isf_estimate"].clip(0, 120), bins=40, color="#5c7ae0", alpha=0.8)
    ax2.axvline(events["isf_estimate"].median(), color="red", lw=1.2, ls="--",
                label=f"median={events['isf_estimate'].median():.1f}")
    ax2.set_xlabel("ISF estimate (mg/dL / U)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump", default="1539514")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df     = load_merged_data(args.pump)
    events = find_correction_boluses(df)
    events = rolling_isf(events)

    summary = summarise(events, args.pump)
    log.info("ISF summary:")
    for k, v in summary.items():
        if k != "params":
            log.info(f"  {k}: {v}")

    out_json = OUT_DIR / f"isf_{args.pump}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved: {out_json}")

    if args.plot:
        plot_isf(events, args.pump, OUT_DIR / f"isf_{args.pump}.png")


if __name__ == "__main__":
    main()
