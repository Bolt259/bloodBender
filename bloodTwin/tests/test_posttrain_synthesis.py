"""
Tests that post-train scripts (calibrate_conformal, failure_modes, evaluation)
correctly synthesize passthrough features via ensure_passthrough_features.

These scripts load DataFrames from merged CSVs and pass them to ONNX inference.
For v1.2+ models, columns like hypothetical_dose and site_change must be present
before indexing into feat_names.  The scripts now delegate to ensure_passthrough_features
rather than inlining the synthesis logic.

We don't invoke the full scripts (they require ONNX sessions and live CSVs) — instead
we verify that the shared utility correctly handles all the column combinations that
arise in practice when these scripts load different vintage CSVs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bloodTwin._feature_utils import ensure_passthrough_features


# Feature lists mirroring what the scaler.pkl embeds for each schema version
FEAT_V11 = ["bg", "basal_rate", "bolus_dose", "basal_delta",
            "time_since_last_bolus", "bg_slope_15min", "bg_slope_30min",
            "sin_time", "cos_time", "iob", "bg_acceleration", "mask_bg", "mask_label"]

FEAT_V12 = FEAT_V11 + ["hypothetical_dose"]

FEAT_V13 = FEAT_V12 + ["site_change"]


def _merged_csv_v11(n=10):
    """DataFrame as loaded from a v1.1-era merged CSV (no hypo_dose, no site_change)."""
    return pd.DataFrame({
        "bg": np.linspace(120, 130, n),
        "basal_rate": [0.8] * n,
        "bolus_dose": [0.0, 1.5] + [0.0] * (n - 2),
        "basal_delta": [0.0] * n,
        "time_since_last_bolus": np.linspace(0, 100, n),
        "bg_slope_15min": [0.1] * n,
        "bg_slope_30min": [0.05] * n,
        "sin_time": [0.3] * n,
        "cos_time": [0.95] * n,
        "iob": [0.5] * n,
        "bg_acceleration": [0.0] * n,
        "mask_bg": [False] * n,
        "mask_label": [True] * n,
    })


def _merged_csv_v13(n=10):
    """DataFrame from a current merged CSV (has all v1.3 columns)."""
    df = _merged_csv_v11(n)
    df["hypothetical_dose"] = df["bolus_dose"]
    df["site_change"] = [0.0] * (n - 1) + [1.0]  # site change at last step
    return df


class TestCalibrationSynthesis:
    """calibrate_conformal.py path: loads val CSV, then calls collect_residuals(df, feat_names)."""

    def test_v11_csv_with_v12_model_synthesizes_hypo_dose(self):
        df = _merged_csv_v11()
        out = ensure_passthrough_features(df, FEAT_V12)
        assert "hypothetical_dose" in out.columns
        assert list(out["hypothetical_dose"]) == pytest.approx(list(df["bolus_dose"]))

    def test_v11_csv_with_v13_model_synthesizes_both(self):
        df = _merged_csv_v11()
        out = ensure_passthrough_features(df, FEAT_V13)
        assert "hypothetical_dose" in out.columns
        assert "site_change" in out.columns
        assert (out["site_change"] == 0.0).all()   # default: no site change

    def test_v13_csv_with_v13_model_passes_through_unchanged(self):
        df = _merged_csv_v13()
        out = ensure_passthrough_features(df, FEAT_V13)
        # existing columns must not be overwritten
        assert out["site_change"].iloc[-1] == 1.0   # event preserved
        assert list(out["hypothetical_dose"]) == list(df["hypothetical_dose"])

    def test_all_feat_names_selectable_after_synthesis_v12(self):
        df = _merged_csv_v11()
        out = ensure_passthrough_features(df, FEAT_V12)
        arr = out[FEAT_V12].to_numpy(dtype="float32")
        assert arr.shape == (len(df), len(FEAT_V12))
        assert not np.isnan(arr).any()

    def test_all_feat_names_selectable_after_synthesis_v13(self):
        df = _merged_csv_v11()
        out = ensure_passthrough_features(df, FEAT_V13)
        arr = out[FEAT_V13].to_numpy(dtype="float32")
        assert arr.shape == (len(df), len(FEAT_V13))
        assert not np.isnan(arr).any()


class TestFailureModeSynthesis:
    """failure_modes.py path: loads val CSV per pump, calls _load_val_split."""

    def test_nan_bolus_zeroed_in_hypo_dose(self):
        df = _merged_csv_v11()
        df["bolus_dose"] = [float("nan"), 1.0, float("nan"), 2.0] + [0.0] * 6
        out = ensure_passthrough_features(df, FEAT_V12)
        assert out["hypothetical_dose"].iloc[0] == 0.0
        assert out["hypothetical_dose"].iloc[1] == pytest.approx(1.0)

    def test_v13_csv_preserves_real_site_change_events(self):
        df = _merged_csv_v13()
        df["site_change"] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        out = ensure_passthrough_features(df, FEAT_V13)
        assert out["site_change"].iloc[2] == 1.0


class TestEvaluationSynthesis:
    """evaluation.py path: loads test CSV, runs rolling passive/corrected windows."""

    def test_original_df_not_mutated_v13(self):
        df = _merged_csv_v11()
        original_cols = set(df.columns)
        ensure_passthrough_features(df, FEAT_V13)
        assert set(df.columns) == original_cols

    def test_v11_csv_v11_model_no_changes(self):
        df = _merged_csv_v11()
        out = ensure_passthrough_features(df, FEAT_V11)
        assert set(out.columns) == set(df.columns)
        assert "hypothetical_dose" not in out.columns
        assert "site_change" not in out.columns
