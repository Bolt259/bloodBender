"""
Canonical patient state schema for bloodBender.

Single source of truth for the feature set fed into bloodTwin models. Any
pipeline that produces training data or assembles a live inference window
imports from here — resampler.py, external dataset parsers, the inference
runner — so a field rename or new derived feature only changes one place.

Design principles
-----------------
* Schema: `PatientStateField` registry declares every field — name, dtype,
  unit, description, and whether it is a raw sensor value, a derived
  feature, or a quality mask.  `SCHEMA` is the canonical ordered tuple.
* Pure functions: each derived feature has its own module-level function
  that accepts only the Series it needs and returns a new Series.  No
  hidden state, no side-effects, easy to unit-test in isolation.
* Builder: `build_feature_frame` is the one place that calls all derivation
  functions and returns a DataFrame in canonical column order.  Callers
  that already have raw streams call this directly.
* Window carrier: `PatientStateWindow` is a thin typed dataclass that
  wraps a single [lookback × n_features] numpy array together with its
  metadata, used by the inference runner.

Adding a new feature
--------------------
1. Add a `PatientStateField` entry to SCHEMA (is_model_input=True to
   include it in model input; bump SCHEMA_VERSION).
2. Write a pure derivation function below.
3. Call it inside `build_feature_frame`.
4. Retrain the model (input width changes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# ── Schema version ────────────────────────────────────────────────────────────
# Bump when the model-input feature set changes (field added/removed/renamed).
# model.onnx artifacts are tagged against this version at export time.
SCHEMA_VERSION: str = "1.0"

# Minutes per CGM sample interval (5 min for Dexcom G-series / Tandem CGM)
CGM_STEP_MINUTES: int = 5


# ── Field descriptor ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatientStateField:
    """Metadata for a single column in the patient state DataFrame."""
    name: str
    dtype: str          # numpy dtype string, e.g. "float32"
    unit: str           # physical unit or "bool" / "dimensionless"
    description: str
    is_model_input: bool = True   # False for mask/quality fields
    is_derived: bool = False      # True if computed from other fields


# ── Canonical schema ──────────────────────────────────────────────────────────
# Order here defines column order in all output DataFrames and the ONNX input.

SCHEMA: tuple[PatientStateField, ...] = (
    PatientStateField(
        name="bg",
        dtype="float32",
        unit="mg/dL",
        description="Blood glucose from CGM, NaN where imputed/missing",
        is_model_input=True,
        is_derived=False,
    ),
    PatientStateField(
        name="basal_rate",
        dtype="float32",
        unit="U/hr",
        description="Continuous basal insulin delivery rate",
        is_model_input=True,
        is_derived=False,
    ),
    PatientStateField(
        name="bolus_dose",
        dtype="float32",
        unit="U",
        description="Bolus insulin delivered within the 5-min interval",
        is_model_input=True,
        is_derived=False,
    ),
    PatientStateField(
        name="basal_delta",
        dtype="float32",
        unit="U/hr",
        description="Change in basal rate from previous interval",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="time_since_last_bolus",
        dtype="float32",
        unit="minutes",
        description="Minutes since last non-zero bolus delivery, capped at 1440",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="bg_slope_15min",
        dtype="float32",
        unit="mg/dL/min",
        description="BG rate of change over the past 15 minutes",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="bg_slope_30min",
        dtype="float32",
        unit="mg/dL/min",
        description="BG rate of change over the past 30 minutes",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="sin_time",
        dtype="float32",
        unit="dimensionless",
        description="Sine component of time-of-day circular encoding",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="cos_time",
        dtype="float32",
        unit="dimensionless",
        description="Cosine component of time-of-day circular encoding",
        is_model_input=True,
        is_derived=True,
    ),
    PatientStateField(
        name="mask_bg",
        dtype="bool",
        unit="bool",
        description="True where BG was imputed (not a real CGM reading); "
                    "False means observed",
        is_model_input=True,
        is_derived=False,
    ),
    PatientStateField(
        name="mask_label",
        dtype="bool",
        unit="bool",
        description="True where row is valid for training loss/metrics; "
                    "False for long-gap rows where BG target is unreliable",
        is_model_input=True,
        is_derived=False,
    ),
)

# Derived conveniences
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in SCHEMA if f.is_model_input)
ALL_COLUMN_NAMES: tuple[str, ...] = tuple(f.name for f in SCHEMA)

# Raw sensor columns that must be present before derivation
_RAW_REQUIRED: frozenset[str] = frozenset({"bg", "basal_rate", "bolus_dose"})


# ── Pure derivation functions ─────────────────────────────────────────────────
# Each function has exactly one job and depends only on its arguments.
# All return float32 Series with the same index as the input.

def derive_basal_delta(basal_rate: pd.Series) -> pd.Series:
    """Change in basal rate between consecutive 5-min intervals (U/hr)."""
    return basal_rate.diff().fillna(0.0).astype("float32")


def derive_time_since_last_bolus(
    bolus_dose: pd.Series,
    timestamps: pd.DatetimeIndex,
    cap_minutes: float = 1440.0,
) -> pd.Series:
    """
    Minutes elapsed since the most recent non-zero bolus delivery.

    Capped at cap_minutes (default 24 h) so the scaler does not see unbounded
    values at the start of a long fasting window.  Returns cap_minutes if no
    prior bolus exists in the series.
    """
    result = np.full(len(bolus_dose), cap_minutes, dtype="float32")
    last_bolus_ts = None

    for i, (ts, dose) in enumerate(zip(timestamps, bolus_dose)):
        if dose > 0:
            last_bolus_ts = ts
        if last_bolus_ts is not None:
            elapsed = (ts - last_bolus_ts).total_seconds() / 60.0
            result[i] = min(elapsed, cap_minutes)

    return pd.Series(result, index=bolus_dose.index, name="time_since_last_bolus")


def derive_bg_slope(
    bg: pd.Series,
    window_minutes: int,
    step_minutes: int = CGM_STEP_MINUTES,
) -> pd.Series:
    """
    BG rate of change (mg/dL/min) over `window_minutes` using a backward
    difference at the 5-min sample rate.

    Missing BG at either endpoint → slope set to 0.0 (conservative; avoids
    propagating NaN into the feature vector where the resampler has masked
    the gap).
    """
    n_steps = window_minutes // step_minutes
    shifted = bg.shift(n_steps)
    slope = (bg - shifted) / float(window_minutes)
    return slope.fillna(0.0).astype("float32")


def derive_time_of_day(
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    Circular encoding of time-of-day onto the unit circle.

    Returns (sin_time, cos_time), both in [-1, 1].  Using both components
    preserves metric continuity across midnight (23:59 → 00:00).
    """
    minutes = timestamps.hour * 60 + timestamps.minute
    angle = 2.0 * math.pi * minutes / 1440.0
    sin_t = pd.Series(np.sin(angle).astype("float32"), index=timestamps, name="sin_time")
    cos_t = pd.Series(np.cos(angle).astype("float32"), index=timestamps, name="cos_time")
    return sin_t, cos_t


# ── Feature builder ───────────────────────────────────────────────────────────

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features and return a DataFrame in canonical column
    order (matching FEATURE_NAMES).

    Parameters
    ----------
    df : DataFrame
        Must contain at minimum: timestamp (column or index), bg, basal_rate,
        bolus_dose.  mask_bg and mask_label are passed through unchanged if
        present; set to False/True respectively if absent (assumes all rows
        are observed and usable — caller should supply these for training data).

    Returns
    -------
    DataFrame with columns ordered exactly as FEATURE_NAMES, same row count
    as input.  Index is preserved.
    """
    missing = _RAW_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"build_feature_frame: missing required columns {missing}")

    # Resolve timestamps for derivations that need a DatetimeIndex
    if "timestamp" in df.columns:
        ts_index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_index = df.index
    else:
        raise ValueError(
            "build_feature_frame: need a 'timestamp' column or DatetimeIndex"
        )

    out = df.copy()

    # Derived features
    out["basal_delta"] = derive_basal_delta(out["basal_rate"])
    out["time_since_last_bolus"] = derive_time_since_last_bolus(
        out["bolus_dose"], ts_index
    )
    out["bg_slope_15min"] = derive_bg_slope(out["bg"], window_minutes=15)
    out["bg_slope_30min"] = derive_bg_slope(out["bg"], window_minutes=30)
    out["sin_time"], out["cos_time"] = derive_time_of_day(ts_index)

    # Quality masks: pass through if present, otherwise default to all-observed
    if "mask_bg" not in out.columns:
        out["mask_bg"] = False
    if "mask_label" not in out.columns:
        out["mask_label"] = True

    # Return only the canonical columns in canonical order
    return out[list(ALL_COLUMN_NAMES)].astype(
        {f.name: f.dtype for f in SCHEMA}
    )


# ── Inference window carrier ──────────────────────────────────────────────────

@dataclass
class PatientStateWindow:
    """
    A single ONNX-ready inference window: [lookback × n_features] float32.

    Produced by the inference runner after assembling the live feature frame
    and applying the RobustScaler.  Carries metadata for logging/debugging;
    the model only sees `array`.
    """
    array: np.ndarray                    # shape: [lookback, n_features], float32
    feature_names: Sequence[str]         # must match FEATURE_NAMES
    window_end_utc: pd.Timestamp         # UTC timestamp of the last row
    pump_serial: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.array.ndim != 2:
            raise ValueError(f"PatientStateWindow: expected 2-D array, got {self.array.ndim}-D")
        if self.array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"PatientStateWindow: array has {self.array.shape[1]} features "
                f"but feature_names has {len(self.feature_names)}"
            )
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                f"PatientStateWindow: feature_names do not match SCHEMA v{SCHEMA_VERSION}.\n"
                f"  Expected: {list(FEATURE_NAMES)}\n"
                f"  Got:      {list(self.feature_names)}"
            )
