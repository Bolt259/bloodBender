"""
Unit tests for bloodBath.data.resampler (BgImputer and UnifiedResampler)

Covers:
  - BgImputer.impute_bg_gaps():
      * clean data → no imputation
      * short gap (2 readings = 10 min) → imputed, mask_bg=True, mask_label=True
      * long gap (7 readings = 35 min, > max_impute_minutes=60 but we test the
        configurable boundary) → mask_label=False
      * gap at series edge (no bounding values) → not imputed, masked
  - UnifiedResampler alignment rules:
      * basal forward-fills, bolus sums per interval, BG last-value
  - SequenceSegmenter.detect_temporal_gaps()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bloodBath.data.resampler import BgImputer, UnifiedResampler


# ── BgImputer ─────────────────────────────────────────────────────────────────


class TestBgImputer:
    """Tests for gap imputation and masking logic."""

    def _imputer(self, max_impute_minutes: int = 60) -> BgImputer:
        return BgImputer(max_impute_minutes=max_impute_minutes)

    def test_clean_series_no_imputation(self, clean_bg_series):
        """Fully observed BG series → no values imputed, no gaps masked."""
        imp, mask_bg, mask_label = self._imputer().impute_bg_gaps(clean_bg_series)
        assert mask_bg.sum() == 0
        assert mask_label.all()
        pd.testing.assert_series_equal(imp, clean_bg_series, check_names=False)

    def test_short_gap_imputed(self, bg_series_short_gap):
        """
        2-reading gap (10 min) is within the 60-min impute window.
        After imputation:
          - No NaN at gap positions
          - mask_bg=True at those positions
          - mask_label remains True
        """
        imp, mask_bg, mask_label = self._imputer().impute_bg_gaps(bg_series_short_gap)
        # Positions 10 and 11 were NaN
        assert pd.notna(imp.iloc[10])
        assert pd.notna(imp.iloc[11])
        assert mask_bg.iloc[10] == True
        assert mask_bg.iloc[11] == True
        assert mask_label.iloc[10] == True
        assert mask_label.iloc[11] == True

    def test_short_gap_interpolated_value_in_range(self, bg_series_short_gap):
        """
        Imputed BG must be between the two surrounding observed values.
        """
        orig = bg_series_short_gap.copy()
        imp, _, _ = self._imputer().impute_bg_gaps(orig)
        left  = imp.iloc[9]   # last observed before gap
        right = imp.iloc[12]  # first observed after gap
        lo, hi = min(left, right), max(left, right)
        assert lo <= imp.iloc[10] <= hi + 1, f"Imputed {imp.iloc[10]:.1f} outside [{lo:.1f}, {hi:.1f}]"
        assert lo <= imp.iloc[11] <= hi + 1

    def test_long_gap_masked_from_training(self, bg_series_long_gap):
        """
        7-reading gap (35 min) with max_impute_minutes=30 → gap is too long.
        mask_label should be False for those rows; mask_bg should be True.
        """
        imp = self._imputer(max_impute_minutes=30)
        _, mask_bg, mask_label = imp.impute_bg_gaps(bg_series_long_gap)
        for i in range(15, 22):
            assert mask_bg.iloc[i] == True,  f"mask_bg not True at index {i}"
            assert mask_label.iloc[i] == False, f"mask_label not False at index {i}"

    def test_long_gap_within_limit_imputed(self, bg_series_long_gap):
        """
        Same 7-reading gap is fine if max_impute_minutes=60 (default).
        """
        imp = self._imputer(max_impute_minutes=60)
        _, mask_bg, mask_label = imp.impute_bg_gaps(bg_series_long_gap)
        assert mask_label.iloc[15] == True   # gap is within limit → usable for training
        assert mask_bg.iloc[15] == True      # but flagged as imputed

    def test_gap_at_start_cannot_be_imputed(self):
        """
        Gap at the very start of the series (no left bound) → cannot impute,
        mask from training.
        """
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        bg = pd.Series(np.linspace(100, 150, n), index=idx)
        bg.iloc[0] = np.nan
        bg.iloc[1] = np.nan

        imp = self._imputer()
        _, mask_bg, mask_label = imp.impute_bg_gaps(bg)
        # Can't impute edge gap — should be masked
        assert mask_bg.iloc[0] == True
        assert mask_bg.iloc[1] == True

    def test_stats_updated(self, bg_series_short_gap):
        """imputation_stats['gaps_imputed'] must be incremented."""
        imp = self._imputer()
        imp.impute_bg_gaps(bg_series_short_gap)
        stats = imp.get_imputation_stats()
        assert stats["gaps_imputed"] >= 1

    def test_clip_above_max_bg(self):
        """Values above 600 mg/dL should be clipped."""
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        bg = pd.Series([100.0, 700.0, 100.0, 100.0, 100.0], index=idx)
        imp = self._imputer()
        result, _, _ = imp.impute_bg_gaps(bg)
        assert result.iloc[1] == pytest.approx(600.0)

    def test_clip_below_min_bg(self):
        """Values below 20 mg/dL should be clipped."""
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        bg = pd.Series([100.0, 10.0, 100.0, 100.0, 100.0], index=idx)
        imp = self._imputer()
        result, _, _ = imp.impute_bg_gaps(bg)
        assert result.iloc[1] == pytest.approx(20.0)

    def test_output_series_names(self, clean_bg_series):
        _, mask_bg, mask_label = self._imputer().impute_bg_gaps(clean_bg_series)
        assert mask_bg.name == "mask_bg"
        assert mask_label.name == "mask_label"

    def test_reset_stats(self, bg_series_short_gap):
        imp = self._imputer()
        imp.impute_bg_gaps(bg_series_short_gap)
        imp.reset_stats()
        stats = imp.get_imputation_stats()
        assert stats["gaps_imputed"] == 0
        assert stats["values_imputed"] == 0


# ── UnifiedResampler ──────────────────────────────────────────────────────────


class TestUnifiedResamplerAlignment:
    """Tests for stream alignment rules.

    NOTE: Event timestamps must be timezone-aware (UTC) to match the UTC
    master_index that align_streams_to_master_index receives.  Tz-naive
    timestamps would fail reindex because pandas cannot compare naive vs
    aware indices.
    """

    def _master_index(self, n: int = 12, start: str = "2024-01-01 10:00") -> pd.DatetimeIndex:
        return pd.date_range(start=start, periods=n, freq="5min", tz="UTC")

    def test_bolus_summed_per_interval(self):
        """
        Two boluses in the same 5-min bin (different timestamps) should be summed.

        NOTE: `_events_to_dataframe` deduplicates by exact timestamp (keeping last),
        so two events at the *same* timestamp would lose one.  Use events 30 s apart
        within the same 5-min bin to exercise the sum path.
        """
        master = self._master_index()
        bin_start = master[3]  # e.g. 10:15 UTC
        ts_a = bin_start                                          # 10:15:00
        ts_b = bin_start + pd.Timedelta("30s")                   # 10:15:30
        bolus_data = [
            {"timestamp": ts_a, "bolus_dose": 2.0, "event_type": "bolus"},
            {"timestamp": ts_b, "bolus_dose": 1.5, "event_type": "bolus"},
        ]
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index([], [], bolus_data, master)
        assert float(streams["bolus_dose"].iloc[3]) == pytest.approx(3.5)

    def test_bolus_zero_outside_event(self):
        """Intervals with no bolus must be 0.0, not NaN."""
        master = self._master_index()
        ts = master[2]
        bolus_data = [{"timestamp": ts, "bolus_dose": 5.0, "event_type": "bolus"}]
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index([], [], bolus_data, master)
        assert float(streams["bolus_dose"].iloc[0]) == pytest.approx(0.0)
        assert float(streams["bolus_dose"].iloc[5]) == pytest.approx(0.0)

    def test_basal_value_set_at_event_timestamp(self):
        """
        A single basal event at t=0 is forward-filled to all subsequent steps.

        Implementation: resamples with `.resample(freq).last()` (one entry), then
        `.reindex(master, method='ffill')` creates the remaining positions and
        fills them all with 1.2 since there is only one entry in the resampled
        series.  Subsequent `.fillna(0.0)` has nothing to do.
        """
        master = self._master_index(n=6)
        ts = master[0]
        basal_data = [{"timestamp": ts, "basal_rate": 1.2, "event_type": "basal"}]
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index([], basal_data, [], master)
        vals = streams["basal_rate"].values.astype(float)
        assert all(abs(v - 1.2) < 1e-6 for v in vals), f"Expected all 1.2, got {vals}"

    def test_basal_rate_at_event_timestamps(self):
        """Each basal event's value must appear at its own timestamp."""
        master = self._master_index(n=10)
        basal_data = [
            {"timestamp": master[0], "basal_rate": 1.0, "event_type": "basal"},
            {"timestamp": master[5], "basal_rate": 0.5, "event_type": "basal"},
        ]
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index([], basal_data, [], master)
        assert float(streams["basal_rate"].iloc[0]) == pytest.approx(1.0)
        assert float(streams["basal_rate"].iloc[5]) == pytest.approx(0.5)

    def test_bg_aligned_preserves_values(self):
        """BG values at exact 5-min timestamps must survive alignment."""
        master = self._master_index(n=6)
        cgm_data = [
            {"timestamp": master[0], "bg": 120.0, "event_type": "cgm"},
            {"timestamp": master[3], "bg": 140.0, "event_type": "cgm"},
        ]
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index(cgm_data, [], [], master)
        assert float(streams["bg"].iloc[0]) == pytest.approx(120.0)
        assert float(streams["bg"].iloc[3]) == pytest.approx(140.0)

    def test_empty_data_returns_zeros_or_nan(self):
        """Empty data streams must not crash; basal/bolus fill 0.0, BG fills NaN."""
        master = self._master_index(n=4)
        resampler = UnifiedResampler()
        streams = resampler.align_streams_to_master_index([], [], [], master)
        assert streams["bolus_dose"].isna().sum() == 0   # all 0.0
        assert streams["basal_rate"].isna().sum() == 0   # all 0.0
        assert streams["bg"].isna().all()                # all NaN

    def test_stats_updated_after_alignment(self):
        """resampling_stats must be populated after alignment."""
        master = self._master_index(n=6)
        cgm_data = [{"timestamp": master[1], "bg": 130.0, "event_type": "cgm"}]
        resampler = UnifiedResampler()
        resampler.align_streams_to_master_index(cgm_data, [], [], master)
        stats = resampler.get_resampling_stats()
        assert stats["bg_coverage_pct"] > 0


# ── UnifiedResampler — master index creation ──────────────────────────────────


class TestCreateUnifiedMasterIndex:
    def test_index_spans_all_streams(self):
        """Master index must cover all timestamps from all streams."""
        cgm_data  = [{"timestamp": "2024-01-01 10:00", "bg": 100}]
        basal_data = [{"timestamp": "2024-01-01 11:00", "basal_rate": 1.0}]
        bolus_data = []
        resampler = UnifiedResampler()
        idx = resampler.create_unified_master_index(cgm_data, basal_data, bolus_data)
        # Should span from at least 10:00 to 11:00
        assert idx[0] <= pd.Timestamp("2024-01-01 10:00")
        assert idx[-1] >= pd.Timestamp("2024-01-01 11:00")

    def test_empty_input_returns_empty_index(self):
        resampler = UnifiedResampler()
        idx = resampler.create_unified_master_index([], [], [])
        assert len(idx) == 0

    def test_index_is_5min_uniform(self):
        """All intervals in the master index must be exactly 5 minutes."""
        cgm_data = [
            {"timestamp": "2024-06-01 08:00", "bg": 100},
            {"timestamp": "2024-06-01 10:00", "bg": 110},
        ]
        resampler = UnifiedResampler()
        idx = resampler.create_unified_master_index(cgm_data, [], [])
        if len(idx) > 1:
            diffs = pd.Series(idx).diff().dropna()
            assert (diffs == pd.Timedelta("5min")).all()
