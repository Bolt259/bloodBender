"""
Held-out validation: does the local ISF model beat the global baseline?

Loads the fitted isf_model_<pump>.pkl, extracts correction episodes from the
TEST split (never seen during fit), and compares, per episode, the absolute
error of:
    * baseline-only   : predict ISF = global median for every episode
    * local model     : predict ISF = ISFModel.predict(state at bolus)
against the observed per-episode ISF (bg_drop / dose).

Usage (from repo root, inside the inference nix shell):
    python bloodTwin/pipelines/validate_isf_model.py [--pump 1539514]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bloodTwin.inference.isf_model import (
    ISFModel, extract_correction_episodes, FEATURE_ORDER,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR = PROJECT_ROOT / "bloodBath/bloodBank/merged"


def load_split(pump_id: str, subset: str) -> pd.DataFrame:
    d = MERGED_DIR / f"pump_{pump_id}" / subset
    csvs = sorted(d.glob(f"lstm_{subset}_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No {subset} CSVs for pump {pump_id}")
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump", default="1539514")
    ap.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    model = ISFModel.load(Path(args.artifacts) / f"isf_model_{args.pump}.pkl")
    log.info("Loaded model: baseline=%.2f  n_train=%d  meta=%s",
             model.baseline_isf, model.n_train, model.fit_metadata)

    test_df = load_split(args.pump, "test")
    events = extract_correction_episodes(test_df)
    if events.empty:
        log.error("No test episodes extracted — cannot validate")
        return
    log.info("Held-out test episodes: %d  [%s → %s]", len(events),
             events["timestamp"].min(), events["timestamp"].max())

    isf_true = events["isf_estimate"].to_numpy(dtype=float)
    isf_base = np.full_like(isf_true, model.baseline_isf)

    isf_pred = np.array([
        model.predict({feat: float(events.iloc[i][feat]) for feat in FEATURE_ORDER})
        for i in range(len(events))
    ])

    def report(name: str, pred: np.ndarray) -> dict:
        mae = float(np.mean(np.abs(pred - isf_true)))
        rmse = float(np.sqrt(np.mean((pred - isf_true) ** 2)))
        medae = float(np.median(np.abs(pred - isf_true)))
        log.info("  %-14s MAE=%6.3f  RMSE=%6.3f  MedAE=%6.3f", name, mae, rmse, medae)
        return {"mae": mae, "rmse": rmse, "medae": medae}

    log.info("Held-out test performance (ISF mg/dL/U, per episode):")
    base = report("baseline", isf_base)
    local = report("local model", isf_pred)

    impr_mae = 100.0 * (base["mae"] - local["mae"]) / base["mae"]
    impr_rmse = 100.0 * (base["rmse"] - local["rmse"]) / base["rmse"]
    log.info("Improvement: MAE %+.1f%%   RMSE %+.1f%%", impr_mae, impr_rmse)
    log.info("Predicted ISF range: [%.1f, %.1f]  (baseline %.1f)",
             isf_pred.min(), isf_pred.max(), model.baseline_isf)
    n_moved = int(np.sum(np.abs(isf_pred - model.baseline_isf) > 0.5))
    log.info("Episodes where model moved >0.5 off baseline: %d / %d",
             n_moved, len(events))


if __name__ == "__main__":
    main()
