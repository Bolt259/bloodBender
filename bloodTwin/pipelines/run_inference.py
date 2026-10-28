"""
Inference CLI entry point for bloodBender.

Usage:
    python bloodTwin/pipelines/run_inference.py --pump 1539514
    python bloodTwin/pipelines/run_inference.py --pump 1539514 --as-of 2026-09-20T08:00:00

Outputs the full InferenceResult as formatted JSON to stdout; errors to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from bloodTwin.inference.service import ARTIFACTS_DIR, InferenceService


def main() -> int:
    ap = argparse.ArgumentParser(description="Run bloodBender inference for one pump.")
    ap.add_argument("--pump", required=True, help="Pump serial, e.g. 1539514")
    ap.add_argument("--as-of", default=None, help="ISO datetime cutoff (default: latest row)")
    ap.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Artifacts directory")
    args = ap.parse_args()

    try:
        as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
        service = InferenceService(artifacts_dir=Path(args.artifacts_dir))
        result = service.run(pump_id=args.pump, as_of=as_of)
        print(json.dumps(result.to_dict(), indent=2))
    except Exception as e:
        print(f"inference failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
