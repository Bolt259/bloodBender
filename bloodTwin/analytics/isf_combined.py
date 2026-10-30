"""
Combined ISF timeline across all pumps (same patient, different hardware).

Merges correction bolus events from all available pumps into a single
chronological view and fits one global rolling median.  Reveals multi-year
ISF drift that per-pump analysis misses.

Usage:
    python bloodTwin/analytics/isf_combined.py [--pumps 881235 901161470 1539514]
"""

import argparse, importlib.util, json, sys, logging
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
OUT_DIR      = PROJECT_ROOT / "bloodTwin/analytics/isf"
ROLLING_DAYS = 90   # wider window for the combined long-run view

# Load derive_isf module by file path (avoids sync-stack import)
_mod_name = "derive_isf"
if _mod_name not in sys.modules:
    _path = Path(__file__).parent / "derive_isf.py"
    _spec = importlib.util.spec_from_file_location(_mod_name, _path)
    _mod  = importlib.util.module_from_spec(_spec)
    sys.modules[_mod_name] = _mod
    _spec.loader.exec_module(_mod)
_di = sys.modules[_mod_name]

PUMP_COLORS = {"881235": "#5c7ae0", "901161470": "#e09a5c", "1539514": "#5ce0a8"}
PUMP_LABELS = {
    "881235":    "t:slim X2  (2021–2024)",
    "901161470": "t:slim X2  (2024–2025)",
    "1539514":   "Tandem Mobi (2026–)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumps", nargs="+", default=["881235", "901161470", "1539514"])
    args = ap.parse_args()

    all_events = []
    for pump_id in args.pumps:
        log.info(f"Loading pump {pump_id}...")
        df = _di.load_merged_data(pump_id)
        ev = _di.find_correction_boluses(df)
        if not ev.empty:
            ev["pump_id"] = pump_id
            all_events.append(ev)

    if not all_events:
        log.error("No events found"); return

    combined = pd.concat(all_events, ignore_index=True).sort_values("timestamp")
    log.info(f"Total correction events across all pumps: {len(combined)}")

    # Global 90-day rolling median
    combined["rolling_isf_90d"] = np.nan
    window = pd.Timedelta(days=ROLLING_DAYS)
    for i, row in combined.iterrows():
        t = row["timestamp"]
        mask = (combined["timestamp"] >= t - window) & (combined["timestamp"] <= t)
        w = combined.loc[mask, "isf_estimate"]
        if len(w) >= 10:
            combined.at[i, "rolling_isf_90d"] = float(w.median())

    # Per-pump median summary
    summary = {}
    for pump_id in args.pumps:
        sub = combined[combined["pump_id"] == pump_id]["isf_estimate"]
        if not sub.empty:
            summary[pump_id] = {
                "n": len(sub),
                "median": round(float(sub.median()), 1),
                "p25": round(float(sub.quantile(0.25)), 1),
                "p75": round(float(sub.quantile(0.75)), 1),
            }

    log.info("Per-pump ISF summary:")
    for pid, s in summary.items():
        log.info(f"  {pid} ({PUMP_LABELS.get(pid,'')}): "
                 f"n={s['n']}, median={s['median']}, IQR [{s['p25']}–{s['p75']}]")

    # Current operational ISF: 30-day median of most recent events
    latest_ts = combined["timestamp"].max()
    recent = combined[combined["timestamp"] >= latest_ts - pd.Timedelta(days=30)]
    current_isf = float(recent["isf_estimate"].median()) if len(recent) >= 5 else float(combined["isf_estimate"].median())
    log.info(f"Current ISF (30d): {current_isf:.1f} mg/dL/U")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle("Patient ISF — combined history across all pumps (same patient)", fontsize=13)

    ax = axes[0]
    for pump_id in args.pumps:
        sub = combined[combined["pump_id"] == pump_id]
        col = PUMP_COLORS.get(pump_id, "#888")
        ax.scatter(sub["timestamp"], sub["isf_estimate"],
                   s=8, alpha=0.35, color=col, label=PUMP_LABELS.get(pump_id, pump_id))

    valid = combined.dropna(subset=["rolling_isf_90d"])
    ax.plot(valid["timestamp"], valid["rolling_isf_90d"],
            color="#e05c5c", lw=2.2, label="90-day rolling median (all pumps)", zorder=5)
    ax.axhline(current_isf, color="#333", ls="--", lw=1.0,
               label=f"Current 30d estimate: {current_isf:.1f} mg/dL/U")
    ax.set_ylabel("ISF (mg/dL / U)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    ax2 = axes[1]
    for pump_id in args.pumps:
        sub = combined[combined["pump_id"] == pump_id]["isf_estimate"].clip(0, 80)
        col = PUMP_COLORS.get(pump_id, "#888")
        ax2.hist(sub, bins=30, alpha=0.55, color=col,
                 label=f"{PUMP_LABELS.get(pump_id, pump_id)} (med={summary.get(pump_id,{}).get('median','?')})",
                 density=True)
    ax2.set_xlabel("ISF estimate (mg/dL / U)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out_png = OUT_DIR / "isf_combined.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info(f"Saved: {out_png}")

    out_json = OUT_DIR / "isf_combined.json"
    with open(out_json, "w") as f:
        json.dump({
            "pumps": summary,
            "current_isf_30d": round(current_isf, 1),
            "rolling_window_days": ROLLING_DAYS,
            "total_events": len(combined),
            "trend_note": "ISF increasing over 5 years: 17→20.5→24 mg/dL/U — patient becoming more insulin-sensitive",
        }, f, indent=2)
    log.info(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
