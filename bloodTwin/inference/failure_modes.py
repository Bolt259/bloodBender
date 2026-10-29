"""
Failure-mode residual classifier for bloodBender.

Produces calibrated per-reading probabilities for five patient states that
explain why BG may be out of range:

  post_meal_rise      — BG accelerating from food absorption
  dawn_phenomenon     — early-morning rise without meal/bolus input
  site_failure        — insulin not absorbing; BG rising despite IOB
  exercise_sensitive  — unusually fast drop relative to IOB
  nominal             — no anomaly detected

Architecture: a lightweight GradientBoostingClassifier (sklearn) trained on
per-window engineered features derived from the model's own prediction
residuals.  Calibrated with isotonic regression (Platt scaling) so displayed
probabilities are honest.  Deliberately not a neural head — GBM is auditable,
fast to retrain, and easy to calibrate.

The single-class "nominal" is residual: 1 − Σ other_probs.

Features used (all computed from the last LOOKBACK window + its prediction):
  residual_30min      — actual − predicted BG at 30 min horizon
  residual_60min      — actual − predicted BG at 60 min horizon
  iob_at_window_end   — insulin on board at current time
  bg_slope_15min_end  — current BG velocity
  bg_acceleration_end — current BG acceleration
  sin_time / cos_time — time of day (circular)
  interval_width_30   — conformal interval width at 30 min (proxy for uncertainty)

Labels (weak supervision from heuristics — intended to be updated as ground
truth becomes available):
  post_meal_rise:     meal_probability ≥ 0.5 (from aux head) AND residual_60 < +20
  dawn_phenomenon:    3am≤hour≤8am AND residual_60 > +15 AND iob < 0.3
  site_failure:       residual_60 > +40 AND iob > 1.0 (bg high despite active insulin)
  exercise_sensitive: residual_30 < -20 AND iob < 0.5 (bg dropping faster than iob explains)
  nominal:            none of the above

Usage:
    python bloodTwin/inference/failure_modes.py --fit   # fit on val split, save model
    python bloodTwin/inference/failure_modes.py --demo  # demo on test split

The FailureModeClassifier class is imported by ConformalPredictor in a future
integration step.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR    = PROJECT_ROOT / "bloodBath/bloodBank/merged"

LOOKBACK = 288
HORIZON  = 12
STATES   = ["post_meal_rise", "dawn_phenomenon", "site_failure",
            "new_site_lag", "exercise_sensitive", "nominal"]


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class FailureModeProbs:
    """Calibrated probability for each patient state, per reading."""
    post_meal_rise:     float
    dawn_phenomenon:    float
    site_failure:       float
    new_site_lag:       float   # transient absorption delay after site change (~90 min)
    exercise_sensitive: float
    nominal:            float
    top_state:          str
    top_prob:           float

    def to_dict(self) -> dict:
        return {s: round(getattr(self, s), 4) for s in STATES} | {
            "top_state": self.top_state,
            "top_prob":  round(self.top_prob, 4),
        }


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(
    window_raw: np.ndarray,     # [LOOKBACK, n_features] unscaled
    preds: np.ndarray,          # [HORIZON] BG predictions (unscaled)
    actuals: np.ndarray,        # [HORIZON] actual BG (NaN where missing)
    meal_prob: float,
    q_hat_30: float,
    feat_names: list[str],
) -> dict:
    """Compute failure-mode features from one window."""
    def _get(name: str) -> int:
        return feat_names.index(name) if name in feat_names else -1

    bg_idx    = _get("bg")
    iob_idx   = _get("iob")
    slope_idx = _get("bg_slope_15min")
    accel_idx = _get("bg_acceleration")
    sin_idx   = _get("sin_time")
    cos_idx   = _get("cos_time")

    last = window_raw[-1]

    # Residuals (positive = model under-predicted, BG higher than expected)
    res_30 = float(actuals[5] - preds[5]) if not np.isnan(actuals[5]) else np.nan
    res_60 = float(actuals[11] - preds[11]) if not np.isnan(actuals[11]) else np.nan

    iob_val   = float(last[iob_idx])   if iob_idx >= 0 else 0.0
    slope_val = float(last[slope_idx]) if slope_idx >= 0 else 0.0
    accel_val = float(last[accel_idx]) if accel_idx >= 0 else 0.0
    sin_val   = float(last[sin_idx])   if sin_idx >= 0 else 0.0
    cos_val   = float(last[cos_idx])   if cos_idx >= 0 else 0.0

    return {
        "residual_30min":      res_30,
        "residual_60min":      res_60,
        "iob":                 iob_val,
        "bg_slope_15min":      slope_val,
        "bg_acceleration":     accel_val,
        "sin_time":            sin_val,
        "cos_time":            cos_val,
        "meal_prob":           meal_prob,
        "interval_width_30":   q_hat_30 * 2,
    }


def _assign_label(row: dict) -> str:
    """Heuristic state label for one window (weak supervision)."""
    r60   = row.get("residual_60min", np.nan)
    r30   = row.get("residual_30min", np.nan)
    iob   = row.get("iob", 0.0)
    meal  = row.get("meal_prob", 0.0)
    sin_t = row.get("sin_time", 0.0)
    cos_t = row.get("cos_time", 1.0)
    # Time-of-day hour approximation from sin/cos
    import math
    hour = (math.atan2(sin_t, cos_t) % (2 * math.pi)) / (2 * math.pi) * 24

    if np.isnan(r60) or np.isnan(r30):
        return "nominal"

    if meal >= 0.5 and r60 < 20.0:
        return "post_meal_rise"
    if 3.0 <= hour <= 8.0 and r60 > 15.0 and iob < 0.3:
        return "dawn_phenomenon"
    if r60 > 40.0 and iob > 1.0:
        return "site_failure"
    if r30 < -20.0 and iob < 0.5:
        return "exercise_sensitive"
    return "nominal"


# ── Classifier ────────────────────────────────────────────────────────────────

class FailureModeClassifier:
    """
    GBM + isotonic calibration for failure-mode state probabilities.

    Train: fit(feature_rows, labels)
    Infer: predict(feature_dict) → FailureModeProbs
    Persist: save / load
    """

    def __init__(self) -> None:
        self._clf: CalibratedClassifierCV | None = None
        self._feat_cols: list[str] = [
            "residual_30min", "residual_60min", "iob",
            "bg_slope_15min", "bg_acceleration",
            "sin_time", "cos_time",
            "meal_prob", "interval_width_30",
        ]

    def fit(self, rows: list[dict], labels: list[str]) -> "FailureModeClassifier":
        X = self._to_matrix(rows)
        y = np.array(labels)
        log.info(f"Fitting on {len(y)} windows; class distribution: "
                 + str({s: int((y == s).sum()) for s in STATES}))

        base = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self._clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._clf.fit(X, y)
        log.info("Fitting complete")
        return self

    def predict(self, row: dict) -> FailureModeProbs:
        if self._clf is None:
            raise RuntimeError("Call fit() first")
        X = self._to_matrix([row])
        # Fill NaN with 0 for inference
        X = np.nan_to_num(X, nan=0.0)
        proba = self._clf.predict_proba(X)[0]
        classes = list(self._clf.classes_)
        prob_dict = {s: 0.0 for s in STATES}
        for cls, p in zip(classes, proba):
            if cls in prob_dict:
                prob_dict[cls] = float(p)
        top = max(prob_dict, key=prob_dict.get)
        return FailureModeProbs(**prob_dict, top_state=top, top_prob=prob_dict[top])

    def _to_matrix(self, rows: list[dict]) -> np.ndarray:
        return np.array([
            [row.get(c, 0.0) if not (isinstance(row.get(c), float) and np.isnan(row.get(c, 0.0))) else 0.0
             for c in self._feat_cols]
            for row in rows
        ], dtype=np.float32)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Saved: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "FailureModeClassifier":
        # The classifier is fit via `python failure_modes.py --fit`, so the pkl
        # stores the class as `__main__.FailureModeClassifier`.  When loaded from
        # a different entry point (e.g. the inference service), `__main__` is that
        # entry point and the attribute is absent — unpickling then fails.  Alias
        # the class into __main__ before load so the reference always resolves.
        import sys as _sys
        _main = _sys.modules.get("__main__")
        if _main is not None and not hasattr(_main, "FailureModeClassifier"):
            _main.FailureModeClassifier = cls
            _main.FailureModeProbs = FailureModeProbs
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FailureModeClassifier, got {type(obj)}")
        return obj


# ── Fit pipeline ──────────────────────────────────────────────────────────────

def collect_features_and_labels(
    pump_id: str,
    subset: str,
    sess: ort.InferenceSession,
    input_name: str,
    feat_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    q_hat_30: float,
    stride: int = 6,
) -> tuple[list[dict], list[str]]:
    split_dir = MERGED_DIR / f"pump_{pump_id}" / subset
    csvs = sorted(split_dir.glob(f"lstm_{subset}_*.csv"))
    if not csvs:
        return [], []
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Dose-conditioned schema v1.2: synthesize hypothetical_dose from the actual
    # bolus when the CSV predates the feature (train/eval: == bolus_dose).
    if "hypothetical_dose" in feat_names and "hypothetical_dose" not in df.columns:
        df["hypothetical_dose"] = df["bolus_dose"].fillna(0.0) \
            if "bolus_dose" in df.columns else 0.0

    feat_arr = df[feat_names].to_numpy(dtype=np.float32)
    bg_arr   = df["bg"].to_numpy(dtype=np.float32)
    scale_s  = np.where(scale == 0, 1.0, scale)

    rows, labels = [], []
    for start in range(0, len(feat_arr) - LOOKBACK - HORIZON + 1, stride):
        end = start + LOOKBACK
        w = pd.DataFrame(feat_arr[start:end], columns=feat_names).ffill().fillna(0.).to_numpy("float32")
        X = ((w - center) / scale_s).astype("float32")[None]
        outs   = sess.run(None, {input_name: X})
        preds  = outs[0].reshape(-1)
        meal_p = float(1 / (1 + np.exp(-outs[1].reshape(-1)[0]))) if len(outs) > 1 else 0.0
        actuals = bg_arr[end: end + HORIZON].copy()
        feat_row = _build_features(feat_arr[start:end], preds, actuals,
                                   meal_p, q_hat_30, feat_names)
        if np.isnan(feat_row["residual_60min"]) and np.isnan(feat_row["residual_30min"]):
            continue
        label = _assign_label(feat_row)
        rows.append(feat_row)
        labels.append(label)

    return rows, labels


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    global ARTIFACTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit",  action="store_true", help="Fit classifier on val splits")
    ap.add_argument("--demo", action="store_true", help="Demo on Mobi test split")
    ap.add_argument("--pumps", nargs="+", default=["1539514", "901161470", "881235"])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--model", "--artifacts", dest="artifacts", default=None,
                    help="Artifacts dir containing model.onnx/scaler.pkl "
                         "(default: bloodTwin_unified_lstm)")
    args = ap.parse_args()
    if args.artifacts:
        ARTIFACTS_DIR = Path(args.artifacts)

    sess = ort.InferenceSession(str(ARTIFACTS_DIR / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    feat_names = list(sc["feature_names"])
    center     = np.asarray(sc["center_"], dtype="float32")
    scale      = np.asarray(sc["scale_"],  dtype="float32")

    with open(ARTIFACTS_DIR / "conformal_summary.json") as f:
        conf_summary = json.load(f)
    q_hat_30 = conf_summary["per_horizon_q_hat"]["30min"]

    cal_path = ARTIFACTS_DIR / "failure_mode_clf.pkl"

    if args.fit:
        all_rows, all_labels = [], []
        for pump_id in args.pumps:
            log.info(f"Collecting val features for pump {pump_id}")
            rows, labels = collect_features_and_labels(
                pump_id, "validate", sess, input_name,
                feat_names, center, scale, q_hat_30, args.stride,
            )
            log.info(f"  {len(rows)} windows")
            all_rows.extend(rows)
            all_labels.extend(labels)

        clf = FailureModeClassifier()
        clf.fit(all_rows, all_labels)
        clf.save(cal_path)

        # Quick accuracy on val set itself (optimistic but sanity-check)
        correct = sum(clf.predict(r).top_state == l
                      for r, l in zip(all_rows, all_labels))
        log.info(f"Val self-accuracy: {correct}/{len(all_labels)} "
                 f"= {correct/len(all_labels):.1%} (optimistic — same data)")

    if args.demo:
        if not cal_path.exists():
            log.error("Run --fit first to create failure_mode_clf.pkl")
            return
        clf = FailureModeClassifier.load(cal_path)
        rows, labels = collect_features_and_labels(
            "1539514", "test", sess, input_name,
            feat_names, center, scale, q_hat_30, args.stride,
        )
        print(f"\nFailure-mode demo on Mobi test split ({len(rows)} windows)\n")
        print(f"{'state':<22}  {'heuristic label count':>22}  {'clf predicted':>14}")
        from collections import Counter
        heuristic_counts = Counter(labels)
        clf_counts = Counter(clf.predict(r).top_state for r in rows)
        for s in STATES:
            print(f"  {s:<20}  {heuristic_counts.get(s,0):>22}  {clf_counts.get(s,0):>14}")


if __name__ == "__main__":
    main()
