"""
Fit the local/dynamic ISF model from the patient's pump history.

Extracts isolated correction-bolus episodes (see ISFModel /
extract_correction_episodes) from the TRAIN + VALIDATE splits, derives the
hard global baseline ISF (episode median — matches derive_isf.py), fits the
Gaussian-Process local-deviation model, and saves it alongside the other
inference artifacts.

The TEST split is deliberately held out so fit_isf_model and the held-out
validation in validate_isf_model.py use disjoint data.

Usage (from repo root, inside the inference nix shell):
    python bloodTwin/pipelines/fit_isf_model.py [--pump 1539514]
                                                [--baseline 24.0]
                                                [--splits train validate]

Outputs:
    bloodTwin/artifacts/bloodTwin_unified_lstm/isf_model_<pump>.pkl
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from bloodTwin.inference.isf_model import ISFModel, extract_correction_episodes

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR = PROJECT_ROOT / "bloodBath/bloodBank/merged"
ANALYTICS_ISF_DIR = PROJECT_ROOT / "bloodTwin/analytics/isf"


def load_splits(pump_id: str, splits: list[str]) -> pd.DataFrame:
    """Load and concatenate the requested splits for a pump, time-sorted."""
    frames = []
    for subset in splits:
        d = MERGED_DIR / f"pump_{pump_id}" / subset
        for csv in sorted(d.glob(f"lstm_{subset}_*.csv")):
            df = pd.read_csv(csv, comment="#")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs for pump {pump_id} in splits {splits}")
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    log.info("Loaded %d rows for pump %s  [%s → %s]", len(combined), pump_id,
             combined["timestamp"].iloc[0], combined["timestamp"].iloc[-1])
    return combined


def resolve_baseline(pump_id: str, cli_baseline: float | None,
                     events: pd.DataFrame) -> float:
    """
    Resolve the hard global baseline ISF.

    Priority: explicit --baseline > analytics/isf/isf_<pump>.json median >
    episode median from the loaded data.
    """
    if cli_baseline and cli_baseline > 0:
        log.info("Baseline ISF from --baseline: %.2f", cli_baseline)
        return float(cli_baseline)

    isf_json = ANALYTICS_ISF_DIR / f"isf_{pump_id}.json"
    if isf_json.exists():
        try:
            data = json.loads(isf_json.read_text())
            med = data.get("isf_overall_median")
            if med:
                log.info("Baseline ISF from %s: %.2f", isf_json.name, med)
                return float(med)
        except Exception as e:
            log.warning("Could not read %s: %s", isf_json, e)

    med = float(events["isf_estimate"].median()) if not events.empty else 24.0
    log.info("Baseline ISF from episode median: %.2f", med)
    return med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump", default="1539514")
    ap.add_argument("--baseline", type=float, default=None,
                    help="Global baseline ISF; default = analytics median / episode median")
    ap.add_argument("--splits", nargs="+", default=["train", "validate"],
                    help="Splits to fit on (test held out for validation)")
    ap.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    df = load_splits(args.pump, args.splits)
    events = extract_correction_episodes(df)
    baseline = resolve_baseline(args.pump, args.baseline, events)

    model = ISFModel(baseline_isf=baseline)
    model.fit(df)

    out = Path(args.artifacts) / f"isf_model_{args.pump}.pkl"
    model.save(out)
    log.info("Saved: %s", out)
    log.info("Fit metadata: %s", json.dumps(model.fit_metadata, indent=2))


if __name__ == "__main__":
    main()
