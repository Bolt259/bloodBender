"""
Push notification hook for bloodBender hypoglycemia risk alerts.

Reads bloodBath/logs/inference_latest.json and fires a ntfy.sh notification
when either trigger condition is met:
  1. Hypo risk: baseline (0 U) predicted BG dips below HYPO_THRESHOLD within 30 min
  2. Failure mode: top predicted state is site_failure or exercise_induced_sensitivity
     with probability >= FAILURE_MODE_MIN_PROB

Deduplication: each trigger type has a 15-min cooldown stored in
LOG_DIR/.notify_last_{trigger}.ts so the 5-min inference cron does not flood.

Usage (from repo root, with NTFY_TOPIC set):
    python bloodTwin/inference/notify.py [--latest PATH] [--log-dir PATH]

Environment variables:
    NTFY_TOPIC   (required) ntfy.sh topic name — e.g. "bloodbender-lkshply"
    NTFY_URL     (optional) ntfy server base URL, default https://ntfy.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Thresholds ──────────────────────────────────────────────────────────────
HYPO_THRESHOLD_MGDL    = 70.0   # predicted BG below this triggers hypo alert
HYPO_HORIZON_STEPS     = 6      # 6 × 5 min = 30 min look-ahead
FAILURE_MODE_MIN_PROB  = 0.85   # minimum probability to fire failure-mode alert
FAILURE_MODE_ALERTABLE = {"site_failure", "exercise_induced_sensitivity"}
COOLDOWN_SECONDS       = 900    # 15 minutes between re-alerts for the same trigger

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_LOG_DIR     = Path(__file__).resolve().parents[2] / "bloodBath" / "logs"
DEFAULT_LATEST_JSON = DEFAULT_LOG_DIR / "inference_latest.json"
NTFY_URL_DEFAULT    = "https://ntfy.sh"


def _cooldown_path(log_dir: Path, trigger: str) -> Path:
    return log_dir / f".notify_last_{trigger}.ts"


def _in_cooldown(log_dir: Path, trigger: str) -> bool:
    p = _cooldown_path(log_dir, trigger)
    if not p.exists():
        return False
    try:
        last = float(p.read_text().strip())
        return (time.time() - last) < COOLDOWN_SECONDS
    except (ValueError, OSError):
        return False


def _mark_sent(log_dir: Path, trigger: str) -> None:
    _cooldown_path(log_dir, trigger).write_text(f"{time.time():.0f}")


def _post(topic: str, title: str, body: str, tags: list[str],
          priority: str, ntfy_url: str) -> bool:
    url = f"{ntfy_url.rstrip('/')}/{topic}"
    data = json.dumps({
        "title":    title,
        "message":  body,
        "tags":     tags,
        "priority": priority,
    }).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.URLError as exc:
        print(f"[notify] POST failed: {exc}", file=sys.stderr)
        return False


def evaluate(latest: dict, log_dir: Path, topic: str, ntfy_url: str,
             dry_run: bool = False) -> list[str]:
    """
    Evaluate triggers against latest inference result. Returns list of trigger
    names that fired (empty list = nothing sent).
    """
    fired: list[str] = []
    current_bg  = float(latest.get("current_bg", 0))
    fan         = latest.get("fan", [])
    failure_modes = latest.get("failure_modes", [])
    timestamp   = latest.get("timestamp", "unknown")

    # ── Trigger 1: Hypo risk within 30 min ──────────────────────────────────
    baseline = next((s for s in fan if float(s.get("dose_u", -1)) == 0.0), None)
    if baseline:
        predicted = baseline.get("predicted_bg", [])
        window    = predicted[:HYPO_HORIZON_STEPS]
        min_bg    = min(window) if window else current_bg

        if min_bg < HYPO_THRESHOLD_MGDL:
            trigger = "hypo"
            if _in_cooldown(log_dir, trigger):
                print(f"[notify] hypo trigger active but in cooldown; skipping")
            else:
                eta_steps = next(
                    (i for i, v in enumerate(window) if v < HYPO_THRESHOLD_MGDL), 0
                )
                eta_min = (eta_steps + 1) * 5
                title = f"Low BG risk — {int(min_bg)} mg/dL in {eta_min} min"
                body  = (
                    f"Current: {int(current_bg)} mg/dL\n"
                    f"Predicted min within 30 min: {int(min_bg)} mg/dL "
                    f"(~{eta_min} min)\n"
                    f"As of: {timestamp}"
                )
                if not dry_run:
                    ok = _post(topic, title, body,
                               tags=["rotating_light", "pill"],
                               priority="urgent",
                               ntfy_url=ntfy_url)
                    if ok:
                        _mark_sent(log_dir, trigger)
                        print(f"[notify] sent: {trigger}")
                else:
                    print(f"[notify] DRY RUN — would send: {title}")
                fired.append(trigger)

    # ── Trigger 2: Failure mode alert ────────────────────────────────────────
    if failure_modes:
        top = failure_modes[0]
        state = top.get("state", "")
        prob  = float(top.get("probability", 0.0))

        if state in FAILURE_MODE_ALERTABLE and prob >= FAILURE_MODE_MIN_PROB:
            trigger = f"failure_{state}"
            if _in_cooldown(log_dir, trigger):
                print(f"[notify] {trigger} active but in cooldown; skipping")
            else:
                state_label = state.replace("_", " ").title()
                title = f"Alert: {state_label} ({int(prob * 100)}% confidence)"
                body  = (
                    f"Predicted state: {state_label}\n"
                    f"Confidence: {prob:.0%}\n"
                    f"Current BG: {int(current_bg)} mg/dL\n"
                    f"As of: {timestamp}"
                )
                if not dry_run:
                    ok = _post(topic, title, body,
                               tags=["warning"],
                               priority="high",
                               ntfy_url=ntfy_url)
                    if ok:
                        _mark_sent(log_dir, trigger)
                        print(f"[notify] sent: {trigger}")
                else:
                    print(f"[notify] DRY RUN — would send: {title}")
                fired.append(trigger)

    if not fired:
        print("[notify] no triggers fired")
    return fired


def main() -> None:
    parser = argparse.ArgumentParser(description="bloodBender push notification hook")
    parser.add_argument("--latest",  default=str(DEFAULT_LATEST_JSON))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate triggers but do not send notifications")
    args = parser.parse_args()

    topic    = os.environ.get("NTFY_TOPIC", "").strip()
    ntfy_url = os.environ.get("NTFY_URL", NTFY_URL_DEFAULT).strip()
    log_dir  = Path(args.log_dir)

    if not topic and not args.dry_run:
        print("[notify] NTFY_TOPIC not set; skipping notifications", file=sys.stderr)
        sys.exit(0)

    latest_path = Path(args.latest)
    if not latest_path.exists():
        print(f"[notify] {latest_path} not found; skipping", file=sys.stderr)
        sys.exit(0)

    try:
        latest = json.loads(latest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[notify] failed to read {latest_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    evaluate(latest, log_dir=log_dir, topic=topic, ntfy_url=ntfy_url,
             dry_run=args.dry_run)


if __name__ == "__main__":
    main()
