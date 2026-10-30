"""
Inference service wrapper for bloodBender.

Single clean entry point that ties every inference component into one callable:

    service = InferenceService()
    result  = service.run(pump_id="1539514")
    print(json.dumps(result.to_dict(), indent=2))   # fully JSON-serialisable

Pipeline (one responsibility per private step):

    _load_window   read + concat + dedupe the pump's split CSVs → last 300 rows
    _predict       build_feature_frame → ConformalPredictor.predict → ConformalPrediction
    _classify_state engineer features → FailureModeClassifier → top-3 StateProbs
    _derive_isf    learned ISFModel on current state (static fallback)
    _simulate      ScenarioSimulator.run with the failure-mode probabilities

Hard boundaries
---------------
* ZERO bloodBath sync logic — this module only ever READS merged CSV files.
  It never touches the Tandem API or any sync stack.
* to_dict() is pure-Python: every numpy scalar is cast to float/int so that
  json.dumps(result.to_dict()) works with no custom encoder.
* Graceful degradation: a missing ISF model pkl falls back to the static
  PatientConfig ISF; a missing failure-mode pkl yields an empty state list.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import ConformalPredictor, ConformalPrediction, LOOKBACK, HORIZON
from .failure_modes import FailureModeClassifier, FailureModeProbs
from .isf_model import ISFModel, FEATURE_ORDER as _ISF_FEATURE_ORDER
from .scenarios import ScenarioSimulator, ScenarioFan, PatientConfig

log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / os.environ.get(
    "BLOODTWIN_ARTIFACTS",
    "bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3",
)
DATA_DIR      = PROJECT_ROOT / "bloodBath/bloodBank/merged"

WINDOW_ROWS: int = 300   # rows handed to the simulator (≥ LOOKBACK=288)

# Key horizons surfaced in the compact conformal_intervals field.
_KEY_HORIZONS: tuple[int, ...] = (15, 30, 60)


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_float(x) -> float:
    """Cast any numpy/py scalar to a plain, finite-safe Python float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _jsonify(obj):
    """Recursively strip numpy types so json.dumps works with no encoder."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _to_float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return _to_float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Aggregated inference output for a single pump at a single point in time."""
    timestamp: datetime
    current_bg: float
    fan: list                       # scenario dicts (one per candidate dose)
    meal_probability: float
    failure_modes: list             # top-3 state dicts, descending probability
    isf: float                      # effective ISF used for the correction dose
    ecd_u: float                    # estimated correction dose (units)
    model_trigger: str
    corrected_in_horizon: bool      # does any simulated dose bring BG to target in-horizon?
    conformal_intervals: dict       # {horizon_min: [lower, upper]} for key horizons

    def to_dict(self) -> dict:
        """Fully JSON-serialisable dict — no numpy types, safe for json.dumps."""
        return _jsonify({
            "timestamp":            self.timestamp.isoformat()
                                    if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "current_bg":           _to_float(self.current_bg),
            "fan":                  self.fan,
            "meal_probability":     _to_float(self.meal_probability),
            "failure_modes":        self.failure_modes,
            "isf":                  _to_float(self.isf),
            "ecd_u":                _to_float(self.ecd_u),
            "model_trigger":        self.model_trigger,
            "corrected_in_horizon": bool(self.corrected_in_horizon),
            "conformal_intervals":  self.conformal_intervals,
        })


# ── Service ──────────────────────────────────────────────────────────────────

class InferenceService:
    """
    Loads every inference artifact once and exposes a single run() method.

    Parameters
    ----------
    artifacts_dir : directory holding model.onnx, scaler.pkl, conformal_cal.pkl,
        failure_mode_clf.pkl, isf_model_<pump>.pkl.
    data_dir : bloodBath/bloodBank/merged — read-only source of split CSVs.
    patient_config : per-patient ISF / correction target; defaults derived from
        the artifact directory's isf json where available.
    """

    def __init__(
        self,
        artifacts_dir: Path = ARTIFACTS_DIR,
        data_dir: Path = DATA_DIR,
        patient_config: PatientConfig | None = None,
    ) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._data_dir = Path(data_dir)
        self._config = patient_config or PatientConfig()

        # LSTM + conformal (owns model.onnx + scaler.pkl + conformal_cal.pkl)
        self._lstm = ConformalPredictor(
            onnx_path=self._artifacts_dir / "model.onnx",
            scaler_path=self._artifacts_dir / "scaler.pkl",
            cal_path=self._artifacts_dir / "conformal_cal.pkl",
        )

        # Failure-mode classifier — optional; degrade to empty state list.
        self._failure_clf: FailureModeClassifier | None = None
        clf_path = self._artifacts_dir / "failure_mode_clf.pkl"
        if clf_path.exists():
            try:
                self._failure_clf = FailureModeClassifier.load(clf_path)
            except Exception as e:  # pragma: no cover - defensive
                log.warning("failure_mode_clf load failed (%s)", e)

        # Learned ISF model — optional; predict() itself falls back to baseline,
        # and if the pkl is missing entirely we use the static config ISF.
        self._isf_model: ISFModel | None = None
        self._isf_model_path = self._resolve_isf_path()
        if self._isf_model_path is not None:
            try:
                self._isf_model = ISFModel.load(self._isf_model_path)
            except Exception as e:  # pragma: no cover - defensive
                log.warning("isf_model load failed (%s) — static ISF fallback", e)

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        pump_id: str,
        as_of: datetime | None = None,
    ) -> InferenceResult:
        """Run the full inference pipeline for one pump at one point in time."""
        window = self._load_window(pump_id, as_of)
        prediction = self._predict(window)
        states = self._classify_state(window, prediction)
        fan = self._simulate(window, prediction, states)
        isf_eff = self._derive_isf(window, states)

        # Observed current BG (last non-imputed reading); fall back to the
        # fan's current_bg or the anchored prediction start.
        current_bg = fan.current_bg
        if current_bg is None:
            current_bg = _to_float(prediction.predicted_bg[0])

        # corrected_in_horizon: does ANY non-zero dose bring the 60-min median
        # down to (or below) the correction target within the forecast horizon?
        target = self._config.correction_target
        corrected = any(
            s.dose_u > 0 and s.predicted_bg_at_60min <= target
            for s in fan.scenarios
        )

        # Compact conformal intervals for the key horizons, from the 0U baseline.
        baseline = fan.baseline.prediction
        intervals: dict[int, list[float]] = {}
        for h_min in _KEY_HORIZONS:
            if h_min in baseline.horizon_minutes:
                idx = baseline.horizon_minutes.index(h_min)
                intervals[h_min] = [
                    round(_to_float(baseline.lower[idx]), 1),
                    round(_to_float(baseline.upper[idx]), 1),
                ]

        return InferenceResult(
            timestamp=self._window_end_datetime(window),
            current_bg=round(_to_float(current_bg), 1),
            fan=[s.to_dict() for s in fan.scenarios],
            meal_probability=_to_float(fan.meal_probability),
            failure_modes=states,
            isf=round(_to_float(isf_eff), 1),
            ecd_u=round(_to_float(fan.estimated_correction_dose), 2),
            model_trigger=fan.model_trigger,
            corrected_in_horizon=corrected,
            conformal_intervals=intervals,
        )

    # ── pipeline steps ──────────────────────────────────────────────────────

    def _load_window(self, pump_id: str, as_of: datetime | None) -> pd.DataFrame:
        """
        Read all split CSVs for the pump, concat, dedupe on timestamp, and return
        the last WINDOW_ROWS rows ending at or before ``as_of`` (latest if None).

        READ-ONLY — no sync, no API. Files are parsed with comment='#'.
        """
        pump_dir = self._data_dir / f"pump_{pump_id}"
        if not pump_dir.exists():
            raise FileNotFoundError(f"No merged data for pump {pump_id}: {pump_dir}")

        frames: list[pd.DataFrame] = []
        for split in ("train", "validate", "test"):
            split_dir = pump_dir / split
            if not split_dir.exists():
                continue
            for csv in sorted(split_dir.glob(f"lstm_{split}_*.csv")):
                frames.append(pd.read_csv(csv, comment="#"))

        if not frames:
            raise FileNotFoundError(f"No lstm_*.csv files found under {pump_dir}")

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (
            df.sort_values("timestamp")
              .drop_duplicates(subset="timestamp", keep="last")
              .reset_index(drop=True)
        )

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            df = df[df["timestamp"] <= cutoff].reset_index(drop=True)

        if len(df) < LOOKBACK:
            raise ValueError(
                f"Need ≥{LOOKBACK} rows for pump {pump_id} up to {as_of or 'latest'}, "
                f"got {len(df)}"
            )

        return df.iloc[-WINDOW_ROWS:].reset_index(drop=True)

    def _predict(self, window: pd.DataFrame) -> ConformalPrediction:
        """Run the LSTM + conformal interval on the window (features built lazily)."""
        pump_serial = getattr(self._config, "pump_serial", "") or ""
        return self._lstm.predict_from_df(
            window, coverage=self._lstm.calibration_summary.get("coverage", 0.90),
            pump_serial=pump_serial,
        )

    def _classify_state(
        self,
        window: pd.DataFrame,
        prediction: ConformalPrediction,
    ) -> list[dict]:
        """
        Classify the current patient state and return the top-3 states as dicts
        sorted by probability descending.  Returns [] if no classifier loaded.

        At live inference the horizon actuals are unknown, so the residual
        features are left NaN — FailureModeClassifier.predict() zero-fills them.
        """
        if self._failure_clf is None:
            return []

        last = window.iloc[-1]
        q_hat_30 = float(prediction.q_hat[5]) if len(prediction.q_hat) > 5 else 0.0
        row = {
            "residual_30min":    np.nan,   # future unknown at inference time
            "residual_60min":    np.nan,
            "iob":               _to_float(last.get("iob", 0.0)),
            "bg_slope_15min":    _to_float(last.get("bg_slope_15min", 0.0)),
            "bg_acceleration":   _to_float(last.get("bg_acceleration", 0.0)),
            "sin_time":          _to_float(last.get("sin_time", 0.0)),
            "cos_time":          _to_float(last.get("cos_time", 1.0)),
            "meal_prob":         _to_float(prediction.meal_probability),
            "interval_width_30": _to_float(q_hat_30) * 2,
        }
        try:
            probs: FailureModeProbs = self._failure_clf.predict(row)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("failure classifier predict failed (%s)", e)
            return []

        full = probs.to_dict()
        states = [
            {"state": s, "probability": round(_to_float(full[s]), 4)}
            for s in full
            if s not in ("top_state", "top_prob")
        ]
        states.sort(key=lambda d: d["probability"], reverse=True)
        return states[:3]

    def _derive_isf(self, window: pd.DataFrame, states: list[dict]) -> float:
        """
        Effective ISF for the current window.

        Base ISF: learned local ISFModel on the current state (last row), or the
        static config ISF when the model pkl is absent.  The hard failure-mode
        multipliers are then layered via ScenarioSimulator.effective_isf, so the
        value returned matches the ISF the simulator actually used.
        """
        base_isf = float(self._config.isf)
        if self._isf_model is not None:
            try:
                last = window.iloc[-1]
                features = {}
                for feat in _ISF_FEATURE_ORDER:
                    col = "bg" if feat == "bg_at_bolus" else feat
                    val = last.get(col)
                    features[feat] = _to_float(val) if val is not None and pd.notna(val) else 0.0
                base_isf = float(self._isf_model.predict(features))
            except Exception as e:  # pragma: no cover - defensive
                log.warning("ISF derive failed (%s) — static fallback", e)
                base_isf = float(self._config.isf)

        probs = self._states_to_probs(states)
        isf_eff, _ = ScenarioSimulator.effective_isf(base_isf, probs)
        return float(isf_eff)

    def _simulate(
        self,
        window: pd.DataFrame,
        prediction: ConformalPrediction,
        states: list[dict],
    ) -> ScenarioFan:
        """
        Build the scenario fan.  The simulator derives its own base ISF (same
        learned model) and layers the failure-mode multipliers we pass in, so the
        fan's isf_used is consistent with _derive_isf.
        """
        simulator = ScenarioSimulator(
            predictor=self._lstm,
            patient_config=self._config,
            isf_model_path=self._isf_model_path,
        )
        pump_serial = prediction.pump_serial or ""
        coverage = self._lstm.calibration_summary.get("coverage", 0.90)
        return simulator.run(
            window,
            pump_serial=pump_serial,
            coverage=coverage,
            failure_mode_probs=self._states_to_probs(states),
        )

    # ── internal utilities ──────────────────────────────────────────────────

    @staticmethod
    def _states_to_probs(states: list[dict]) -> dict[str, float] | None:
        """Convert the top-3 state dicts into a {state: prob} map for the simulator."""
        if not states:
            return None
        return {d["state"]: _to_float(d["probability"]) for d in states}

    def _resolve_isf_path(self) -> Path | None:
        """Locate the pump's ISF model pkl in the artifacts dir, if any."""
        matches = sorted(self._artifacts_dir.glob("isf_model_*.pkl"))
        return matches[0] if matches else None

    @staticmethod
    def _window_end_datetime(window: pd.DataFrame) -> datetime:
        ts = pd.Timestamp(window["timestamp"].iloc[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
