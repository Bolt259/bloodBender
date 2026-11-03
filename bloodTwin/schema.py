"""
bloodTwin canonical feature schema — single source of truth.

Import this from both training and inference environments. Has no bloodBath
dependencies, so it is safe in the lightweight `.#inference` Nix shell.

For the embedded port (bareMetalBender), this module defines the algorithm
contract that must be reproduced identically in Rust/C:
  - Feature order (index-sensitive: LSTM input is a packed float32 array)
  - Model time constants (LOOKBACK, HORIZON, CGM_STEP_MINUTES)
  - Hampel filter parameters (must match resampler.py BgHampelFilter defaults)
"""
from __future__ import annotations

# ── Feature schema v1.5 ───────────────────────────────────────────────────────
#
# Exact column order fed to the LSTM. Any change here requires a full retrain.
# Authoritative source: bloodBath/core/patient_state.py SCHEMA (SCHEMA_VERSION="1.5")
# This module republishes those names for bloodTwin (no bloodBath import needed
# in the inference shell). A test (bloodBath/tests/test_patient_state.py) asserts
# that FEATURE_NAMES here matches patient_state.FEATURE_NAMES exactly.
#
# Fields 1-13: produced by patient_state.build_feature_frame()
# Field 14 (hypothetical_dose): added by dataset._load_data (training copy of bolus_dose)
# Field 15 (site_change): read from merged CSV, defaults 0.0 if absent
#
FEATURE_NAMES: tuple[str, ...] = (
    "bg",
    "basal_rate",
    "bolus_dose",
    "basal_delta",
    "time_since_last_bolus",
    "bg_slope_15min",
    "bg_slope_30min",
    "sin_time",
    "cos_time",
    "iob",
    "bg_acceleration",
    "mask_bg",
    "mask_label",
    "hypothetical_dose",
    "site_change",
)

N_FEATURES: int = len(FEATURE_NAMES)

# Convenience index map (avoids repeated .index() calls in hot paths)
FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

# ── Model time constants ───────────────────────────────────────────────────────

CGM_STEP_MINUTES: int = 5   # resampling cadence
LOOKBACK:         int = 288  # 24 h of history fed to the LSTM
HORIZON:          int = 12   # 60 min of prediction (12 × 5 min)

# ── Hampel filter contract ────────────────────────────────────────────────────
#
# BgHampelFilter (bloodBath/data/resampler.py) runs on raw CGM before slope
# and acceleration features are derived. The embedded port must reproduce
# these parameters exactly or the feature distribution will shift.
#
# Algorithm:
#   for each sample i:
#     window = vals[max(0,i-W) : i+W+1]  (one-sided half-width W)
#     med    = median(window)
#     mad    = median(|window - med|)
#     σ̂     = max(K_SCALE * mad, 1.0 mg/dL)
#     if |vals[i] - med| > THRESHOLD * σ̂:
#         vals[i] = med   # replace spike with local median
#
HAMPEL_WINDOW:    int   = 2      # one-sided half-width in samples (±10 min)
HAMPEL_THRESHOLD: float = 3.0   # σ̂ multiples before a point is a spike
HAMPEL_K_SCALE:   float = 1.4826 # consistency constant: MAD → σ̂ (Gaussian)
HAMPEL_MIN_BG:    float = 20.0  # hard floor (mg/dL)
HAMPEL_MAX_BG:    float = 600.0 # hard ceiling (mg/dL)
