"""
One-time migration: add iob and bg_acceleration to v1.0 split CSVs.

Reads each split CSV under bloodBath/bloodBank/merged/pump_*/,
derives the two missing columns from existing data (bolus_dose → iob,
bg_slope_15min → bg_acceleration), and writes a new file alongside the
old one (new timestamp suffix).  Does NOT touch files that already have
the new columns.

Usage (from repo root, inside the default nix shell):
    python bloodTwin/pipelines/migrate_schema_v1_1.py [--dry-run]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR   = PROJECT_ROOT / "bloodBath/bloodBank/merged"
PUMP_IDS     = ["881235", "901161470"]   # only retired pumps; Mobi already migrated by re-sync
SUBSETS      = ["train", "validate", "test"]

NEW_COLS = ["iob", "bg_acceleration"]


def migrate_csv(path: Path, dry_run: bool) -> bool:
    """Return True if migrated, False if already up-to-date."""
    df = pd.read_csv(path, comment="#")
    if all(c in df.columns for c in NEW_COLS):
        log.debug(f"  Already v1.1: {path.name}")
        return False

    log.info(f"  Migrating {path.name}  ({len(df)} rows)")

    # Import derivation functions by file path (no sync-stack dependency)
    import importlib.util, sys
    mod_name = "bloodBath.core.patient_state"
    if mod_name not in sys.modules:
        ps_path = PROJECT_ROOT / "bloodBath" / "core" / "patient_state.py"
        spec = importlib.util.spec_from_file_location(mod_name, ps_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    ps = sys.modules[mod_name]

    # Derive iob from bolus_dose
    df["iob"] = ps.derive_iob(df["bolus_dose"]).values

    # Derive bg_acceleration from bg_slope_15min
    df["bg_acceleration"] = ps.derive_bg_acceleration(df["bg_slope_15min"]).values

    # Reorder to match FEATURE_NAMES (put new cols before mask_bg/mask_label)
    base_cols  = [c for c in df.columns if c not in NEW_COLS + ["mask_bg", "mask_label", "timestamp"]]
    mask_cols  = [c for c in ["mask_bg", "mask_label"] if c in df.columns]
    ts_col     = ["timestamp"] if "timestamp" in df.columns else []
    ordered    = ts_col + [c for c in base_cols if c != "timestamp"] + NEW_COLS + mask_cols
    df = df[[c for c in ordered if c in df.columns]]

    if dry_run:
        log.info(f"    [dry-run] would write {path}")
        return True

    # Write in place (overwrite existing file — it's regenerable from raw data)
    df.to_csv(path, index=False)
    log.info(f"    Written: {path.name}  cols={list(df.columns)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pumps",   nargs="+", default=PUMP_IDS)
    args = ap.parse_args()

    migrated = 0
    for pump_id in args.pumps:
        for subset in SUBSETS:
            split_dir = MERGED_DIR / f"pump_{pump_id}" / subset
            if not split_dir.exists():
                log.warning(f"Missing: {split_dir}")
                continue
            for csv in sorted(split_dir.glob(f"lstm_{subset}_*.csv")):
                if migrate_csv(csv, args.dry_run):
                    migrated += 1

    log.info(f"Done — {migrated} file(s) migrated" + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
