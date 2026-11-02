"""
Shared fixtures for bloodBath unit tests.

All fixtures use synthetic/mocked data — no real pump data, no network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def make_ts_index(n: int, start: str = "2024-01-15 08:00:00", freq: str = "5min") -> pd.DatetimeIndex:
    """Return a UTC DatetimeIndex of *n* rows at 5-min intervals."""
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


# ── Minimal valid feature-frame DataFrame ─────────────────────────────────────

@pytest.fixture()
def minimal_raw_df():
    """
    Minimal DataFrame with the three raw columns required by build_feature_frame.
    100 rows of synthetic data with a DatetimeIndex.
    """
    n = 100
    idx = make_ts_index(n)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "bg": rng.uniform(80, 200, n).astype("float32"),
            "basal_rate": rng.uniform(0.5, 1.5, n).astype("float32"),
            "bolus_dose": np.where(rng.random(n) < 0.1, rng.uniform(1, 8, n), 0.0).astype("float32"),
        },
        index=idx,
    )
    return df


@pytest.fixture()
def raw_df_with_masks(minimal_raw_df):
    """Same as minimal_raw_df but with mask columns pre-populated."""
    df = minimal_raw_df.copy()
    df["mask_bg"] = False
    df["mask_label"] = True
    return df


# ── Bolus-only series for IOB tests ──────────────────────────────────────────

@pytest.fixture()
def single_bolus_series():
    """
    300-step (25 h) bolus series with exactly one 5-U bolus at t=0.
    Long enough to cover the full 300-min DIA of the kernel.
    """
    n = 300
    idx = make_ts_index(n)
    doses = np.zeros(n, dtype="float32")
    doses[0] = 5.0
    return pd.Series(doses, index=idx, name="bolus_dose")


@pytest.fixture()
def zero_bolus_series():
    """All-zero bolus series."""
    n = 100
    idx = make_ts_index(n)
    return pd.Series(np.zeros(n, dtype="float32"), index=idx, name="bolus_dose")


# ── BG series with gaps for resampler tests ───────────────────────────────────

@pytest.fixture()
def clean_bg_series():
    """Uniformly spaced BG readings with no gaps."""
    n = 48  # 4 hours
    idx = make_ts_index(n)
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(90, 180, n).astype(float), index=idx, name="bg")


@pytest.fixture()
def bg_series_short_gap(clean_bg_series):
    """BG series with a 2-reading (10-min) gap — small enough to impute."""
    bg = clean_bg_series.copy()
    bg.iloc[10] = np.nan
    bg.iloc[11] = np.nan
    return bg


@pytest.fixture()
def bg_series_long_gap(clean_bg_series):
    """BG series with a 7-reading (35-min) gap — too long to impute (>30 min / 6 intervals)."""
    bg = clean_bg_series.copy()
    for i in range(15, 22):
        bg.iloc[i] = np.nan
    return bg


# ── Synthetic pump event objects ──────────────────────────────────────────────

class _SimpleEvent:
    """Minimal duck-typed pump event for extractor tests."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.fixture()
def cgm_event_camel():
    """CGM event that uses the correct camelCase attribute name."""
    return _SimpleEvent(currentGlucoseDisplayValue=142)


@pytest.fixture()
def cgm_event_lower():
    """CGM event using the lowercase fallback attribute."""
    return _SimpleEvent(currentglucosedisplayvalue=98)


@pytest.fixture()
def cgm_event_high():
    """CGM event with glucoseValueStatusRaw=1 (HIGH), BG should map to 400."""
    return _SimpleEvent(currentGlucoseDisplayValue=0, glucoseValueStatusRaw=1)


@pytest.fixture()
def cgm_event_low():
    """CGM event with glucoseValueStatusRaw=2 (LOW), BG should map to 40."""
    return _SimpleEvent(currentGlucoseDisplayValue=0, glucoseValueStatusRaw=2)


@pytest.fixture()
def cgm_event_missing():
    """CGM event with no recognizable BG attribute."""
    return _SimpleEvent(someOtherField=99)


@pytest.fixture()
def bolus_event_delivered():
    """Bolus event with deliveredTotal in thousandths (3500 → 3.5 U)."""
    return _SimpleEvent(deliveredTotal=3500)


@pytest.fixture()
def bolus_event_small():
    """Bolus event where deliveredTotal is already in units (≤10), no division."""
    return _SimpleEvent(deliveredTotal=4)


@pytest.fixture()
def basal_event_commanded():
    """Basal event with commandedRate in milliunits/hr (1200 mU/hr → 1.2 U/hr)."""
    return _SimpleEvent(commandedRate=1200)


@pytest.fixture()
def basal_event_commanded_basal_rate():
    """Basal event with commandedBasalRate in units/hr (LidBasalRateChange style)."""
    return _SimpleEvent(commandedBasalRate=0.85)


# ── Synthetic LSTM DataFrame for save_splits tests ───────────────────────────

def _make_lstm_df(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a synthetic LSTM-ready DataFrame with n rows."""
    idx = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "bg": rng.uniform(80, 200, n).astype("float32"),
            "basal_rate": rng.uniform(0.5, 1.5, n).astype("float32"),
            "bolus_dose": np.zeros(n, dtype="float32"),
        }
    )


@pytest.fixture()
def lstm_df_200():
    """200-row synthetic LSTM DataFrame."""
    return _make_lstm_df(200)


@pytest.fixture()
def lstm_df_100():
    """First 100 rows of the synthetic LSTM DataFrame (chronologically earlier)."""
    return _make_lstm_df(100)
