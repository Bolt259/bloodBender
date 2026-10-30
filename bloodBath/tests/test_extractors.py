"""
Unit tests for bloodBath.data.extractors.EventExtractor

Covers:
  - _extract_bg_value(): camelCase primary key, lowercase fallback, HIGH/LOW mapping,
    missing key, numeric conversion
  - _extract_basal_rate(): commandedRate with unit conversion (hundredths)
  - _extract_bolus_dose(): deliveredTotal with heuristic unit detection,
    missing attribute behaviour (the None-return bug is documented)
  - extract_events(): routing of typed event objects
  - deduplicate_events(): deduplication by timestamp+event_type
"""

from __future__ import annotations

import pytest

from bloodBath.data.extractors import EventExtractor


class _Evt:
    """Duck-typed pump event stub."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


# ── _extract_bg_value ─────────────────────────────────────────────────────────


class TestExtractBgValue:
    def setup_method(self):
        self.ex = EventExtractor()

    def test_camel_case_primary_key(self, cgm_event_camel):
        """currentGlucoseDisplayValue (camelCase) must be extracted correctly."""
        result = self.ex._extract_bg_value(cgm_event_camel)
        assert result == pytest.approx(142.0)

    def test_lowercase_fallback(self, cgm_event_lower):
        """currentglucosedisplayvalue (all-lowercase) must work as fallback."""
        result = self.ex._extract_bg_value(cgm_event_lower)
        assert result == pytest.approx(98.0)

    def test_high_status_maps_to_400(self, cgm_event_high):
        """glucoseValueStatusRaw=1 (HIGH) with zero display value → 400."""
        result = self.ex._extract_bg_value(cgm_event_high)
        assert result == pytest.approx(400.0)

    def test_low_status_maps_to_40(self, cgm_event_low):
        """glucoseValueStatusRaw=2 (LOW) with zero display value → 40."""
        result = self.ex._extract_bg_value(cgm_event_low)
        assert result == pytest.approx(40.0)

    def test_missing_key_returns_none(self, cgm_event_missing):
        """Event with no recognised BG attribute must return None."""
        result = self.ex._extract_bg_value(cgm_event_missing)
        assert result is None

    def test_sgv_attribute(self):
        """sgv attribute (Nightscout-style) should be picked up."""
        evt = _Evt(sgv=110)
        result = self.ex._extract_bg_value(evt)
        assert result == pytest.approx(110.0)

    def test_bg_attribute(self):
        """bg attribute should be picked up."""
        evt = _Evt(bg=95)
        result = self.ex._extract_bg_value(evt)
        assert result == pytest.approx(95.0)

    def test_glucose_value_attribute(self):
        """glucoseValue attribute should be picked up."""
        evt = _Evt(glucoseValue=130)
        result = self.ex._extract_bg_value(evt)
        assert result == pytest.approx(130.0)

    def test_high_status_with_non_zero_display_uses_display(self):
        """If glucoseValueStatusRaw=1 but display value is non-zero, use display value."""
        # The implementation only substitutes 400 when bg_value is None or 0
        evt = _Evt(currentGlucoseDisplayValue=385, glucoseValueStatusRaw=1)
        result = self.ex._extract_bg_value(evt)
        # 385 != 0, so substitution should NOT happen — use raw display value
        assert result == pytest.approx(385.0)

    def test_returns_float(self, cgm_event_camel):
        result = self.ex._extract_bg_value(cgm_event_camel)
        assert isinstance(result, float)


# ── _extract_basal_rate ───────────────────────────────────────────────────────


class TestExtractBasalRate:
    def setup_method(self):
        self.ex = EventExtractor()

    def test_commanded_rate_division(self, basal_event_commanded):
        """commandedRate=125 → 1.25 U/hr (divide by 100)."""
        result = self.ex._extract_basal_rate(basal_event_commanded)
        assert result == pytest.approx(1.25)

    def test_commanded_rate_zero(self):
        """commandedRate=0 → 0.0 U/hr."""
        evt = _Evt(commandedRate=0)
        result = self.ex._extract_basal_rate(evt)
        assert result == pytest.approx(0.0)

    def test_fallback_commandedbasalrate(self):
        """commandedbasalrate (lowercase) should be returned as-is."""
        evt = _Evt(commandedbasalrate=0.85)
        result = self.ex._extract_basal_rate(evt)
        assert result == pytest.approx(0.85)

    def test_no_attribute_returns_zero(self):
        """Event with no recognised basal attribute should return 0.0."""
        evt = _Evt(someField=1)
        result = self.ex._extract_basal_rate(evt)
        assert result == pytest.approx(0.0)

    def test_value_attribute_large_heuristic(self):
        """value > 10 treated as hundredths → divided by 100."""
        evt = _Evt(value=150)
        result = self.ex._extract_basal_rate(evt)
        assert result == pytest.approx(1.5)

    def test_value_attribute_small_passthrough(self):
        """value ≤ 10 treated as already in U/hr → pass through."""
        evt = _Evt(value=0.8)
        result = self.ex._extract_basal_rate(evt)
        assert result == pytest.approx(0.8)


# ── _extract_bolus_dose ───────────────────────────────────────────────────────


class TestExtractBolusDose:
    def setup_method(self):
        self.ex = EventExtractor()

    def test_delivered_total_large_divided_by_1000(self, bolus_event_delivered):
        """deliveredTotal=3500 (>10) → 3.5 U (divide by 1000)."""
        result = self.ex._extract_bolus_dose(bolus_event_delivered)
        assert result == pytest.approx(3.5)

    def test_delivered_total_small_passthrough(self, bolus_event_small):
        """deliveredTotal=4 (≤10) → 4.0 U (no division)."""
        result = self.ex._extract_bolus_dose(bolus_event_small)
        assert result == pytest.approx(4.0)

    def test_delivered_total_none_falls_through(self):
        """
        deliveredTotal=None: the condition `event.deliveredTotal is not None` is False,
        so the function falls through and returns None (same as missing attribute).
        The caller guards against None with `bolus_dose > 0`.
        """
        evt = _Evt(deliveredTotal=None)
        result = self.ex._extract_bolus_dose(evt)
        assert result is None

    def test_missing_attribute_returns_none(self):
        """
        Event with no deliveredTotal attribute → _extract_bolus_dose returns None
        (implicit fall-through — no attribute matched).

        This is a known behaviour of the current implementation: the function body
        only has one active branch (deliveredTotal).  If the event lacks it, the
        function implicitly returns None.  The caller (normalize_bolus_events) guards
        against this with `bolus_dose > 0` so None will not propagate to output.
        We document this here rather than changing the production code.
        """
        evt = _Evt(someOtherField=5)
        result = self.ex._extract_bolus_dose(evt)
        assert result is None  # implicit None — documented behaviour

    def test_delivered_total_zero(self):
        """deliveredTotal=0 → 0.0 U."""
        evt = _Evt(deliveredTotal=0)
        result = self.ex._extract_bolus_dose(evt)
        assert result == pytest.approx(0.0)


# ── extract_events routing ────────────────────────────────────────────────────


class TestExtractEventsRouting:
    """Verify events are correctly categorised by extract_events."""

    def setup_method(self):
        self.ex = EventExtractor()

    def test_cgm_event_categorised(self):
        evt = _Evt(currentGlucoseDisplayValue=120)
        result = self.ex.extract_events([evt])
        assert len(result["cgm_events"]) == 1
        assert len(result["basal_events"]) == 0
        assert len(result["bolus_events"]) == 0

    def test_basal_event_categorised(self):
        evt = _Evt(commandedRate=100)
        result = self.ex.extract_events([evt])
        assert len(result["basal_events"]) == 1

    def test_bolus_event_categorised(self):
        """
        _is_bolus_event checks for 'Bolus' or 'bolus' in the class name, or
        presence of 'bolusAmount' or 'insulin' attributes.
        Use bolusAmount to trigger the attribute-based detection path.
        """
        evt = _Evt(bolusAmount=3.5)
        result = self.ex.extract_events([evt])
        assert len(result["bolus_events"]) == 1

    def test_unknown_event_categorised(self):
        evt = _Evt(unrelatedField=99)
        result = self.ex.extract_events([evt])
        assert self.ex.event_counts["unknown"] == 1

    def test_generator_input_accepted(self):
        """extract_events must accept a generator (not just a list)."""
        def gen():
            yield _Evt(currentGlucoseDisplayValue=100)
            yield _Evt(currentGlucoseDisplayValue=110)
        result = self.ex.extract_events(gen())
        assert len(result["cgm_events"]) == 2

    def test_empty_input_returns_empty_dicts(self):
        result = self.ex.extract_events([])
        assert result["cgm_events"] == []
        assert result["basal_events"] == []
        assert result["bolus_events"] == []


# ── deduplicate_events ────────────────────────────────────────────────────────


class TestDeduplicateEvents:
    def setup_method(self):
        self.ex = EventExtractor()

    def test_exact_duplicates_removed(self):
        ts = "2024-01-01T10:00:00"
        events = [
            {"timestamp": ts, "event_type": "cgm", "bg": 100.0},
            {"timestamp": ts, "event_type": "cgm", "bg": 105.0},
        ]
        result = self.ex.deduplicate_events(events)
        assert len(result) == 1

    def test_different_types_same_ts_kept(self):
        ts = "2024-01-01T10:00:00"
        events = [
            {"timestamp": ts, "event_type": "cgm", "bg": 100.0},
            {"timestamp": ts, "event_type": "basal", "basal_rate": 1.0},
        ]
        result = self.ex.deduplicate_events(events)
        assert len(result) == 2

    def test_empty_returns_empty(self):
        result = self.ex.deduplicate_events([])
        assert result == []

    def test_sorted_by_timestamp(self):
        events = [
            {"timestamp": "2024-01-01T10:05:00", "event_type": "cgm", "bg": 110.0},
            {"timestamp": "2024-01-01T10:00:00", "event_type": "cgm", "bg": 100.0},
        ]
        result = self.ex.deduplicate_events(events)
        assert len(result) == 2
        assert result[0]["bg"] == 100.0


# ── normalize_site_change_events ──────────────────────────────────────────────


class LidCannulaFilled:
    """Stub mirroring the parsed LID_CANNULA_FILLED dataclass.

    normalize_site_change_events matches on the class name
    (``type(event).__name__ == 'LidCannulaFilled'``), so this class must keep
    that exact name.
    """

    NAME = "LID_CANNULA_FILLED"

    def __init__(self, timestamp="2024-01-15T09:00:00"):
        self.timestamp = timestamp


class TestSiteChangeExtraction:
    """Tests for extracting LID_CANNULA_FILLED (infusion-site change) events."""

    def setup_method(self):
        self.ex = EventExtractor()

    def test_lid_cannula_filled_extracted(self):
        """A single LID_CANNULA_FILLED event yields one site_change record."""
        events = [LidCannulaFilled(timestamp="2024-01-15T09:00:00")]
        result = self.ex.normalize_site_change_events(events)
        assert len(result) == 1

    def test_non_site_events_filtered(self):
        """Mixed cgm/bolus/site events → only the site event is extracted."""
        events = [
            _Evt(currentGlucoseDisplayValue=120, timestamp="2024-01-15T08:00:00"),
            _Evt(deliveredTotal=3500, timestamp="2024-01-15T08:30:00"),
            LidCannulaFilled(timestamp="2024-01-15T09:00:00"),
        ]
        result = self.ex.normalize_site_change_events(events)
        assert len(result) == 1
        assert result[0]["event_type"] == "site_change"

    def test_empty_events_returns_empty(self):
        """Empty input → empty output."""
        assert self.ex.normalize_site_change_events([]) == []

    def test_site_change_has_required_fields(self):
        """Returned dict carries timestamp, site_change, and event_type."""
        events = [LidCannulaFilled(timestamp="2024-01-15T09:00:00")]
        record = self.ex.normalize_site_change_events(events)[0]
        assert "timestamp" in record
        assert record["site_change"] == 1
        assert record["event_type"] == "site_change"

    def test_event_counts_updated(self):
        """event_counts['site_change'] is incremented per extracted event."""
        assert self.ex.event_counts["site_change"] == 0
        events = [
            LidCannulaFilled(timestamp="2024-01-15T09:00:00"),
            LidCannulaFilled(timestamp="2024-01-16T09:00:00"),
        ]
        self.ex.normalize_site_change_events(events)
        assert self.ex.event_counts["site_change"] == 2

    def test_matches_on_name_attribute(self):
        """An event whose class name differs but exposes NAME is still matched."""
        other = _Evt(NAME="LID_CANNULA_FILLED", timestamp="2024-01-15T09:00:00")
        result = self.ex.normalize_site_change_events([other])
        assert len(result) == 1
