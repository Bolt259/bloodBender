"""
Archive inference output to bloodBath/logs/inference_archive/.

Called by the 5-min inference cron after each successful inference.
Can also be imported by server.py for backwards-compatible reads.
"""
import json
from pathlib import Path

ARCHIVE_MAX = 2016  # 7 days @ 5-min intervals


def maybe_archive(data: dict, archive_dir: Path) -> Path | None:
    """
    Write a timestamped copy of ``data`` to ``archive_dir``.

    Returns the written Path, or None if this timestamp was already archived.
    Prunes files beyond ARCHIVE_MAX (oldest first).
    """
    ts = data.get("timestamp", "")
    if not ts:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    safe = ts.replace(":", "").replace("+", "").replace("-", "")[:15]
    dest = archive_dir / f"inference_{safe}.json"
    if dest.exists():
        return None
    try:
        dest.write_text(json.dumps(data))
        files = sorted(archive_dir.glob("inference_*.json"))
        for old in files[:-ARCHIVE_MAX]:
            old.unlink(missing_ok=True)
    except OSError:
        return None
    return dest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Archive inference_latest.json")
    ap.add_argument("--latest", required=True, help="Path to inference_latest.json")
    ap.add_argument("--archive-dir", required=True, help="Directory to archive into")
    args = ap.parse_args()

    with open(args.latest) as f:
        data = json.load(f)
    result = maybe_archive(data, Path(args.archive_dir))
    if result:
        print(f"Archived: {result}")
    else:
        print("Already archived (duplicate timestamp)")
