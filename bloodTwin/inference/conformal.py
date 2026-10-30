"""
Adaptive split conformal prediction for bloodTwin BG forecasts.

Wraps any trained ONNX model and produces per-horizon calibrated prediction
intervals — no retraining required.  Uses the held-out validation split as
the calibration set.

Theory (split conformal, Vovk et al. / Angelopoulos & Bates 2022):
  1. Calibrate: for each validation window, run the model, collect
     nonconformity score  s_i = |actual_bg_h − predicted_bg_h|  per horizon h
  2. Predict: interval = point_prediction ± q̂_{1-α}(h)
     where q̂ is the ⌈(1-α)(1 + 1/n)⌉-th quantile of calibration scores

Coverage guarantee: at least (1-α) of future test windows will have the
true BG inside the interval, assuming exchangeability with the cal set.

Volatility adaptation: scale q̂ by the patient's recent BG turbulence
  ratio = std(bg[-volatility_window:]) / median_cal_std
so intervals widen automatically during post-meal spikes or sensor noise.

Usage
-----
# Once, after training:
    calibrator = ConformalCalibrator()
    calibrator.fit(residuals)           # residuals: [n_cal, 12]
    calibrator.save("artifacts/.../conformal_cal.pkl")

# At inference (every 5 min):
    predictor = ConformalPredictor(onnx_path, scaler_path, cal_path)
    pred = predictor.predict(window_df, coverage=0.90)
    # pred.predicted_bg  → [12]  point estimates  (mg/dL)
    # pred.lower/upper   → [12]  90% interval bounds
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import pandas as pd

from bloodTwin._feature_utils import ensure_passthrough_features

# Keep inference module free of bloodBath sync-stack dependencies so it
# runs in the lightweight onnxruntime-only Nix env.  build_feature_frame
# is imported lazily inside predict_from_df only.
FEATURE_NAMES: tuple[str, ...] = (
    "bg", "basal_rate", "bolus_dose", "basal_delta",
    "time_since_last_bolus", "bg_slope_15min", "bg_slope_30min",
    "sin_time", "cos_time", "mask_bg", "mask_label",
)
CGM_STEP_MINUTES: int = 5
HORIZON  = 12    # prediction steps
LOOKBACK = 288   # 24 h @ 5 min



# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class ConformalPrediction:
    """
    Model output for a single inference window.

    All BG values are in mg/dL in the original (unscaled) space.
    Timestamps mark the start of each 5-min forecast interval.
    """
    window_end_utc: pd.Timestamp
    pump_serial: str
    coverage: float                        # e.g. 0.90

    predicted_bg: np.ndarray               # [HORIZON] point estimates
    lower: np.ndarray                      # [HORIZON] lower bound
    upper: np.ndarray                      # [HORIZON] upper bound
    horizon_minutes: list[int] = field(
        default_factory=lambda: list(range(CGM_STEP_MINUTES, (HORIZON + 1) * CGM_STEP_MINUTES, CGM_STEP_MINUTES))
    )

    volatility_scale: float = 1.0          # how much intervals were expanded
    q_hat: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))  # raw quantiles used
    meal_probability: float = 0.0          # P(significant BG rise in horizon) from aux head

    @property
    def predicted_timestamps(self) -> list[pd.Timestamp]:
        return [
            self.window_end_utc + pd.Timedelta(minutes=m)
            for m in self.horizon_minutes
        ]

    def interval_width(self) -> np.ndarray:
        return self.upper - self.lower

    def to_dict(self) -> dict:
        return {
            "window_end_utc":   self.window_end_utc.isoformat(),
            "pump_serial":      self.pump_serial,
            "coverage":         self.coverage,
            "predicted_bg":     self.predicted_bg.tolist(),
            "lower":            self.lower.tolist(),
            "upper":            self.upper.tolist(),
            "horizon_minutes":  self.horizon_minutes,
            "volatility_scale": round(self.volatility_scale, 3),
            "meal_probability": round(self.meal_probability, 4),
        }


# ── Calibrator ────────────────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Fits per-horizon conformal quantiles from validation-set residuals.

    Parameters
    ----------
    coverage : float
        Target coverage level (default 0.90 → 90% prediction intervals).
    volatility_window : int
        Number of lookback rows used to measure recent BG turbulence (default
        36 = last 3 h @ 5 min).
    """

    def __init__(self, coverage: float = 0.90, volatility_window: int = 36) -> None:
        self.coverage = coverage
        self.volatility_window = volatility_window

        # Filled by fit()
        self.q_hat: np.ndarray | None = None      # [HORIZON] raw quantiles
        self.median_cal_std: float | None = None  # median BG std across cal windows
        self.n_cal: int = 0
        self._fitted = False

    def fit(
        self,
        residuals: np.ndarray,
        cal_bg_stds: np.ndarray | None = None,
        max_buffer: int | None = None,
    ) -> "ConformalCalibrator":
        """
        Calibrate from pre-computed residuals.

        Parameters
        ----------
        residuals : ndarray [n_cal, HORIZON]
            Absolute errors |actual_bg − predicted_bg| for each calibration
            window at each horizon step.  NaN rows are silently dropped.
        cal_bg_stds : ndarray [n_cal] or None
            Per-window BG std in the lookback window.  Used to compute the
            median calibration volatility for scaling.  If None, scaling is
            disabled (volatility_scale = 1.0 always).
        max_buffer : int or None
            If set, keep only the most recent max_buffer residuals (sliding
            window online calibration).  Older residuals are discarded so the
            calibrator tracks recent BG dynamics rather than fitting to
            historical data from potentially different patient states.
        """
        r = np.asarray(residuals, dtype=np.float32)
        valid = ~np.isnan(r).any(axis=1)
        r = r[valid]

        if max_buffer is not None and len(r) > max_buffer:
            r = r[-max_buffer:]  # keep most recent

        self.n_cal = len(r)
        if self.n_cal == 0:
            raise ValueError("No valid calibration residuals after NaN drop")

        # Store the buffer for online updates
        self._residual_buffer: list[np.ndarray] = list(r)
        if max_buffer is not None:
            self._max_buffer: int | None = max_buffer
        else:
            self._max_buffer = None

        self._refit()

        if cal_bg_stds is not None:
            stds = np.asarray(cal_bg_stds, dtype=np.float32)[valid]
            if max_buffer is not None and len(stds) > max_buffer:
                stds = stds[-max_buffer:]
            stds = stds[~np.isnan(stds)]
            self.median_cal_std = float(np.median(stds)) if len(stds) > 0 else None

        self._fitted = True
        return self

    def update(self, residuals_row: np.ndarray) -> None:
        """
        Online update: add a new [HORIZON] residual vector and recompute quantiles.

        Call this every time a new actual BG reading becomes available to keep
        the calibration aligned with the patient's current dynamics.  The
        sliding window (max_buffer) automatically discards old residuals.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before update()")
        r = np.asarray(residuals_row, dtype=np.float32)
        if r.shape != (HORIZON,):
            raise ValueError(f"Expected shape ({HORIZON},), got {r.shape}")
        if not np.isnan(r).any():
            self._residual_buffer.append(r)
            if self._max_buffer is not None and len(self._residual_buffer) > self._max_buffer:
                self._residual_buffer.pop(0)
            self.n_cal = len(self._residual_buffer)
            self._refit()

    def _refit(self) -> None:
        r = np.stack(self._residual_buffer, axis=0)
        level = min(1.0, self.coverage * (1.0 + 1.0 / len(r)))
        self.q_hat = np.quantile(r, level, axis=0).astype(np.float32)

    def interval(
        self,
        predicted: np.ndarray,
        recent_bg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute lower/upper bounds around point predictions.

        Parameters
        ----------
        predicted : ndarray [HORIZON]
        recent_bg : ndarray (1-D, last volatility_window BG values) or None
            If provided, scales the interval by local turbulence.

        Returns
        -------
        lower, upper : ndarray [HORIZON]
        scale : float  (volatility scaling factor applied)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before interval()")

        scale = 1.0
        if recent_bg is not None and self.median_cal_std is not None:
            local_std = float(np.nanstd(recent_bg[-self.volatility_window:]))
            if self.median_cal_std > 0:
                scale = max(0.5, min(3.0, local_std / self.median_cal_std))

        q_scaled = self.q_hat * scale
        lower = predicted - q_scaled
        upper = predicted + q_scaled
        return lower.astype(np.float32), upper.astype(np.float32), scale

    def summary(self) -> dict:
        if not self._fitted:
            return {"fitted": False}
        mins = list(range(CGM_STEP_MINUTES, (HORIZON + 1) * CGM_STEP_MINUTES, CGM_STEP_MINUTES))
        return {
            "fitted": True,
            "n_cal": self.n_cal,
            "coverage": self.coverage,
            "median_cal_std": self.median_cal_std,
            "per_horizon_q_hat": {f"{m}min": round(float(q), 2) for m, q in zip(mins, self.q_hat)},
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "ConformalCalibrator":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ConformalCalibrator, got {type(obj)}")
        return obj


# ── Predictor ─────────────────────────────────────────────────────────────────

class ConformalPredictor:
    """
    End-to-end inference: feature assembly → ONNX forward pass → conformal interval.

    Parameters
    ----------
    onnx_path : path to model.onnx
    scaler_path : path to scaler.pkl  (dict with center_/scale_/feature_names)
    cal_path : path to conformal_cal.pkl
    """

    def __init__(
        self,
        onnx_path: Path | str,
        scaler_path: Path | str,
        cal_path: Path | str,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self._sess = ort.InferenceSession(str(onnx_path), providers=list(providers))
        self._input_name = self._sess.get_inputs()[0].name

        with open(Path(scaler_path), "rb") as f:
            scaler = pickle.load(f)
        if isinstance(scaler, dict):
            self._feature_names: list[str] = list(scaler["feature_names"])
            self._center = np.asarray(scaler["center_"], dtype=np.float32)
            self._scale  = np.asarray(scaler["scale_"],  dtype=np.float32)
        else:
            self._feature_names = list(getattr(scaler, "feature_names_in_", FEATURE_NAMES))
            self._center = np.asarray(scaler.center_, dtype=np.float32)
            self._scale  = np.asarray(scaler.scale_,  dtype=np.float32)

        self._cal = ConformalCalibrator.load(cal_path)

    # ── public ────────────────────────────────────────────────────────────────

    def predict_from_window(
        self,
        window,  # PatientStateWindow from bloodBath.core.patient_state
        coverage: float = 0.90,
    ) -> ConformalPrediction:
        """
        Run inference on a pre-assembled PatientStateWindow.

        The window must already be in raw (unscaled) feature space with
        columns ordered per FEATURE_NAMES and shape [LOOKBACK, n_features].
        """
        arr = window.array  # [LOOKBACK, n_features], raw
        return self._run(arr, window.window_end_utc, window.pump_serial, coverage)

    def predict_from_df(
        self,
        df: pd.DataFrame,
        coverage: float = 0.90,
        pump_serial: str = "",
    ) -> ConformalPrediction:
        """
        Run inference on a raw DataFrame of at least LOOKBACK rows.

        build_feature_frame() is loaded directly from patient_state.py by
        file path so this method works in the lightweight inference env
        (no arrow/sync-stack dependency via bloodBath.__init__).
        """
        import importlib.util as _ilu, sys as _sys
        _mod_name = "bloodBath.core.patient_state"
        if _mod_name not in _sys.modules:
            _ps_path = Path(__file__).parents[2] / "bloodBath" / "core" / "patient_state.py"
            _spec = _ilu.spec_from_file_location(_mod_name, _ps_path)
            _mod  = _ilu.module_from_spec(_spec)
            _sys.modules[_mod_name] = _mod   # register before exec so dataclasses can find it
            _spec.loader.exec_module(_mod)
        build_feature_frame = _sys.modules[_mod_name].build_feature_frame  # noqa
        feature_df = build_feature_frame(df)
        if len(feature_df) < LOOKBACK:
            raise ValueError(f"Need ≥{LOOKBACK} rows, got {len(feature_df)}")
        feature_df = ensure_passthrough_features(feature_df, self._feature_names)
        arr = feature_df[list(self._feature_names)].iloc[-LOOKBACK:].to_numpy(dtype=np.float32)
        if "timestamp" in df.columns:
            ts_raw = pd.Timestamp(df["timestamp"].iloc[-1])
            ts = ts_raw if ts_raw.tzinfo else ts_raw.tz_localize("UTC")
        else:
            ts = pd.Timestamp.utcnow()
        return self._run(arr, ts, pump_serial, coverage)

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature columns this model/scaler expects (read-only copy)."""
        return list(self._feature_names)

    def build_raw_window(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Timestamp]:
        """
        Assemble the raw (unscaled) [LOOKBACK, n_features] array and its end
        timestamp from a DataFrame — the same preparation predict_from_df does,
        but WITHOUT running the model.

        Exposed so callers (e.g. the dose-conditioned scenario fan) can edit
        individual feature columns of the raw window before inference, then run
        the model repeatedly via run_raw().
        """
        import importlib.util as _ilu, sys as _sys
        _mod_name = "bloodBath.core.patient_state"
        if _mod_name not in _sys.modules:
            _ps_path = Path(__file__).parents[2] / "bloodBath" / "core" / "patient_state.py"
            _spec = _ilu.spec_from_file_location(_mod_name, _ps_path)
            _mod  = _ilu.module_from_spec(_spec)
            _sys.modules[_mod_name] = _mod
            _spec.loader.exec_module(_mod)
        build_feature_frame = _sys.modules[_mod_name].build_feature_frame  # noqa
        feature_df = build_feature_frame(df)
        if len(feature_df) < LOOKBACK:
            raise ValueError(f"Need ≥{LOOKBACK} rows, got {len(feature_df)}")
        feature_df = ensure_passthrough_features(feature_df, self._feature_names)
        arr = feature_df[list(self._feature_names)].iloc[-LOOKBACK:].to_numpy(dtype=np.float32)
        if "timestamp" in df.columns:
            ts_raw = pd.Timestamp(df["timestamp"].iloc[-1])
            ts = ts_raw if ts_raw.tzinfo else ts_raw.tz_localize("UTC")
        else:
            ts = pd.Timestamp.utcnow()
        return arr, ts

    def run_raw(
        self,
        raw_arr: np.ndarray,
        window_end_utc: pd.Timestamp,
        pump_serial: str = "",
        coverage: float = 0.90,
    ) -> ConformalPrediction:
        """Public entry point to run inference on an already-assembled raw
        [LOOKBACK, n_features] array (feature order == self.feature_names)."""
        return self._run(raw_arr, window_end_utc, pump_serial, coverage)

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        raw_arr: np.ndarray,
        window_end_utc: pd.Timestamp,
        pump_serial: str,
        coverage: float,
    ) -> ConformalPrediction:
        # Scale
        scale_safe = np.where(self._scale == 0, 1.0, self._scale)
        scaled = ((raw_arr - self._center) / scale_safe).astype(np.float32)
        X = scaled[np.newaxis]  # [1, LOOKBACK, n_features]

        # ONNX forward — outputs: [bg_prediction, meal_logit] (meal_logit may be absent
        # in older models exported before the aux head was added)
        onnx_out  = self._sess.run(None, {self._input_name: X})
        predicted = onnx_out[0].reshape(-1).astype(np.float32)
        meal_logit = float(onnx_out[1].reshape(-1)[0]) if len(onnx_out) > 1 else 0.0
        meal_prob  = float(1.0 / (1.0 + np.exp(-meal_logit)))  # sigmoid

        # Conformal interval
        bg_col_idx = self._feature_names.index("bg") if "bg" in self._feature_names else 0
        recent_bg = raw_arr[-self._cal.volatility_window:, bg_col_idx]
        lower, upper, vscale = self._cal.interval(predicted, recent_bg=recent_bg)

        return ConformalPrediction(
            window_end_utc=window_end_utc,
            pump_serial=pump_serial,
            coverage=coverage,
            predicted_bg=predicted,
            lower=lower,
            upper=upper,
            volatility_scale=vscale,
            q_hat=self._cal.q_hat.copy(),
            meal_probability=meal_prob,
        )

    @property
    def calibration_summary(self) -> dict:
        return self._cal.summary()
