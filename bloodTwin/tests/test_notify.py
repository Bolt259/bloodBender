"""Tests for bloodTwin/inference/notify.py trigger logic."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

# notify.py is stdlib-only; import directly to avoid bloodTwin/__init__.py
# pulling in onnxruntime which is only available in the .#inference shell.
_NOTIFY_PATH = Path(__file__).resolve().parents[2] / "bloodTwin" / "inference" / "notify.py"
_spec = importlib.util.spec_from_file_location("notify", _NOTIFY_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

COOLDOWN_SECONDS       = _mod.COOLDOWN_SECONDS
FAILURE_MODE_ALERTABLE = _mod.FAILURE_MODE_ALERTABLE
FAILURE_MODE_MIN_PROB  = _mod.FAILURE_MODE_MIN_PROB
HYPO_THRESHOLD_MGDL    = _mod.HYPO_THRESHOLD_MGDL
_in_cooldown           = _mod._in_cooldown
_mark_sent             = _mod._mark_sent
evaluate               = _mod.evaluate


def _make_latest(
    current_bg: float = 120.0,
    baseline_predicted: list[float] | None = None,
    top_state: str = "normal",
    top_prob: float = 0.5,
) -> dict:
    if baseline_predicted is None:
        baseline_predicted = [120.0] * 12
    return {
        "timestamp": "2026-10-31T12:00:00+00:00",
        "current_bg": current_bg,
        "fan": [
            {
                "dose_u": 0.0,
                "predicted_bg": baseline_predicted,
                "lower": [v - 30 for v in baseline_predicted],
                "upper": [v + 30 for v in baseline_predicted],
            }
        ],
        "failure_modes": [
            {"state": top_state, "probability": top_prob},
            {"state": "normal", "probability": 1.0 - top_prob},
        ],
        "iob": 0.5,
        "ecd_u": 0.0,
    }


def test_no_triggers_normal(tmp_path):
    latest = _make_latest(current_bg=130.0, baseline_predicted=[128.0] * 12)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert fired == []


def test_hypo_trigger_fires(tmp_path):
    predicted = [120.0, 110.0, 100.0, 85.0, 72.0, 65.0] + [60.0] * 6
    latest = _make_latest(current_bg=120.0, baseline_predicted=predicted)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "hypo" in fired


def test_hypo_just_above_threshold_no_fire(tmp_path):
    predicted = [HYPO_THRESHOLD_MGDL + 1.0] * 12
    latest = _make_latest(current_bg=120.0, baseline_predicted=predicted)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "hypo" not in fired


def test_hypo_only_beyond_30min_no_fire(tmp_path):
    # dips below threshold only at step 7+ (>30 min) — outside the 6-step window
    predicted = [120.0] * 6 + [60.0] * 6
    latest = _make_latest(current_bg=120.0, baseline_predicted=predicted)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "hypo" not in fired


def test_failure_mode_site_failure_fires(tmp_path):
    latest = _make_latest(top_state="site_failure", top_prob=0.92)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "failure_site_failure" in fired


def test_failure_mode_exercise_fires(tmp_path):
    latest = _make_latest(top_state="exercise_induced_sensitivity", top_prob=0.87)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "failure_exercise_induced_sensitivity" in fired


def test_failure_mode_below_prob_threshold_no_fire(tmp_path):
    latest = _make_latest(top_state="site_failure", top_prob=FAILURE_MODE_MIN_PROB - 0.01)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert not any("failure" in f for f in fired)


def test_failure_mode_non_alertable_no_fire(tmp_path):
    latest = _make_latest(top_state="post_meal_rise", top_prob=0.95)
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert not any("failure" in f for f in fired)


def test_cooldown_suppresses_repeat(tmp_path):
    predicted = [60.0] * 12
    latest = _make_latest(current_bg=80.0, baseline_predicted=predicted)

    # First call — fires
    fired1 = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                      dry_run=False)
    # Patch: mark sent manually (dry_run skips _mark_sent, so do it explicitly)
    _mark_sent(tmp_path, "hypo")

    # Second call within cooldown — suppressed
    fired2 = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                      dry_run=True)
    assert "hypo" not in fired2


def test_cooldown_expires(tmp_path):
    _mark_sent(tmp_path, "hypo")
    # Backdate the stamp
    stamp_path = tmp_path / ".notify_last_hypo.ts"
    stamp_path.write_text(str(time.time() - COOLDOWN_SECONDS - 1))

    assert not _in_cooldown(tmp_path, "hypo")


def test_missing_fan_no_crash(tmp_path):
    latest = {"timestamp": "t", "current_bg": 80.0, "fan": [], "failure_modes": []}
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert fired == []


def test_both_triggers_fire_together(tmp_path):
    predicted = [60.0] * 12
    latest = _make_latest(
        current_bg=80.0,
        baseline_predicted=predicted,
        top_state="site_failure",
        top_prob=0.90,
    )
    fired = evaluate(latest, tmp_path, topic="test", ntfy_url="http://fake",
                     dry_run=True)
    assert "hypo" in fired
    assert "failure_site_failure" in fired
