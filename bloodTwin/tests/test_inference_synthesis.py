"""
Unit tests for _ensure_passthrough_features in bloodTwin.inference.conformal.

The v1.2 dose-conditioned model adds hypothetical_dose as a 14th feature.
patient_state.build_feature_frame (bloodBath) never emits it; the inference
path synthesizes it from bolus_dose (API source of truth) at runtime.

These tests verify:
  - synthesis uses bolus_dose values from the dataframe (API data)
  - NaN bolus entries are filled with 0.0 (no bolus delivered)
  - pre-existing hypothetical_dose is never overwritten
  - missing bolus_dose column falls back to all-zero (edge case, not normal)
  - 13-feature (v1.1) model paths are unaffected
  - returned df is a copy (original not mutated)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bloodTwin._feature_utils import ensure_passthrough_features as _ensure_passthrough_features


def _df(bolus=None, hypo=None, n=5):
    data = {"bg": np.linspace(100, 120, n)}
    if bolus is not None:
        data["bolus_dose"] = bolus if hasattr(bolus, "__len__") else [bolus] * n
    if hypo is not None:
        data["hypothetical_dose"] = hypo if hasattr(hypo, "__len__") else [hypo] * n
    return pd.DataFrame(data)


FEAT_V12 = ["bg", "bolus_dose", "hypothetical_dose"]
FEAT_V11 = ["bg", "bolus_dose"]


class TestEnsurePassthroughFeatures:

    def test_synthesizes_from_bolus_dose(self):
        df = _df(bolus=[0.0, 1.5, 0.0, 2.0, 0.0])
        out = _ensure_passthrough_features(df, FEAT_V12)
        pd.testing.assert_series_equal(
            out["hypothetical_dose"].reset_index(drop=True),
            df["bolus_dose"].reset_index(drop=True),
            check_names=False,
        )

    def test_nan_bolus_filled_with_zero(self):
        df = _df(bolus=[1.0, float("nan"), 2.0, float("nan"), 0.5])
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert not out["hypothetical_dose"].isna().any()
        assert out["hypothetical_dose"].iloc[1] == 0.0
        assert out["hypothetical_dose"].iloc[3] == 0.0

    def test_existing_hypothetical_dose_not_overwritten(self):
        existing = [9.9, 8.8, 7.7, 6.6, 5.5]
        df = _df(bolus=[1.0, 2.0, 3.0, 4.0, 5.0], hypo=existing)
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert list(out["hypothetical_dose"]) == existing

    def test_missing_bolus_dose_falls_back_to_zero(self):
        df = pd.DataFrame({"bg": [100.0, 110.0, 120.0]})
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert "hypothetical_dose" in out.columns
        assert (out["hypothetical_dose"] == 0.0).all()

    def test_v11_model_unaffected(self):
        df = _df(bolus=[1.0, 0.0, 2.0, 0.0, 0.5])
        out = _ensure_passthrough_features(df, FEAT_V11)
        assert "hypothetical_dose" not in out.columns

    def test_original_dataframe_not_mutated(self):
        df = _df(bolus=[0.0, 1.0, 0.0, 2.0, 0.0])
        original_cols = set(df.columns)
        _ensure_passthrough_features(df, FEAT_V12)
        assert set(df.columns) == original_cols

    def test_zero_bolus_throughout(self):
        df = _df(bolus=[0.0] * 5)
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert (out["hypothetical_dose"] == 0.0).all()

    def test_all_nonzero_bolus(self):
        bolus = [0.5, 1.0, 1.5, 2.0, 2.5]
        df = _df(bolus=bolus)
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert list(out["hypothetical_dose"]) == pytest.approx(bolus)

    def test_returns_dataframe(self):
        df = _df(bolus=[1.0] * 5)
        out = _ensure_passthrough_features(df, FEAT_V12)
        assert isinstance(out, pd.DataFrame)


FEAT_V13 = ["bg", "bolus_dose", "hypothetical_dose", "site_change"]


class TestSiteChangeSynthesis:

    def test_site_change_copied_from_raw_df(self):
        feature_df = _df(bolus=[0.0] * 5)
        raw_df = pd.DataFrame({"bg": [100.0]*5, "bolus_dose": [0.0]*5,
                                "site_change": [0.0, 0.0, 1.0, 0.0, 0.0]})
        out = _ensure_passthrough_features(feature_df, FEAT_V13, raw_df=raw_df)
        assert list(out["site_change"]) == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_site_change_defaults_to_zero_without_raw_df(self):
        feature_df = _df(bolus=[0.0] * 5)
        out = _ensure_passthrough_features(feature_df, FEAT_V13)
        assert "site_change" in out.columns
        assert (out["site_change"] == 0.0).all()

    def test_site_change_not_overwritten_if_present(self):
        feature_df = pd.DataFrame({"bg": [100.0]*5, "bolus_dose": [0.0]*5,
                                    "site_change": [1.0, 0.0, 0.0, 0.0, 0.0]})
        out = _ensure_passthrough_features(feature_df, FEAT_V13)
        assert list(out["site_change"]) == [1.0, 0.0, 0.0, 0.0, 0.0]

    def test_site_change_zeros_when_absent_from_raw_df(self):
        feature_df = _df(bolus=[0.0] * 5)
        raw_df = pd.DataFrame({"bg": [100.0]*5})  # no site_change column
        out = _ensure_passthrough_features(feature_df, FEAT_V13, raw_df=raw_df)
        assert (out["site_change"] == 0.0).all()

    def test_v12_model_unaffected_by_site_change(self):
        feature_df = _df(bolus=[1.0] * 5)
        out = _ensure_passthrough_features(feature_df, FEAT_V12)
        assert "site_change" not in out.columns
