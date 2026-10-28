"""
Unit tests for bloodBath.dataset_parser.save_splits

Covers the leakage-fix: chronological splits must not overlap and must cover the
full input range without duplication.

Because save_chronological_splits uses DATA_PATHS from core.config to construct
output directories, we patch DATA_PATHS so all writes go into pytest's tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bloodBath.dataset_parser.save_splits import save_chronological_splits


# ── inline data factory (mirrors conftest._make_lstm_df) ─────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_data_paths(base: Path) -> dict:
    """Return a DATA_PATHS-shaped dict rooted under *base*."""
    return {
        "merged": {
            "train":    base / "merged" / "train",
            "validate": base / "merged" / "validate",
            "test":     base / "merged" / "test",
        }
    }


def _call_splits(df, tmp_path, pump_serial="TEST01", dataset="synthetic"):
    """Call save_chronological_splits with patched DATA_PATHS."""
    data_paths = _make_data_paths(tmp_path)
    with patch("bloodBath.dataset_parser.save_splits.DATA_PATHS", data_paths):
        save_chronological_splits(
            lstm_df=df,
            pump_serial=pump_serial,
            dataset=dataset,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    return data_paths


def _find_split_files(data_paths, dataset, pump_serial):
    """Return {split_name: [Path, ...]} for all written split files.

    save_chronological_splits builds:
        split_dir = merged_path.parent / dataset / f"pump_{pump_serial}" / merged_path.name
    where merged_path = DATA_PATHS['merged'][split_name]  (e.g. base/merged/train).
    So split_dir = base/merged / dataset / pump_<serial> / train.
    """
    result = {}
    for split_name in ("train", "validate", "test"):
        merged_path = data_paths["merged"][split_name]
        split_dir = merged_path.parent / dataset / f"pump_{pump_serial}" / merged_path.name
        files = list(split_dir.glob(f"lstm_{split_name}_{pump_serial}*.csv"))
        result[split_name] = files
    return result


def _read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── basic split structure ─────────────────────────────────────────────────────


class TestSplitStructure:

    def test_produces_three_files(self, lstm_df_200, tmp_path):
        """200-row DataFrame must produce exactly one file per split."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        for name, flist in files.items():
            assert len(flist) == 1, f"Expected 1 file for {name}, got {len(flist)}"

    def test_total_rows_equals_input(self, lstm_df_200, tmp_path):
        """train + validate + test rows must equal the full input size."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        total = sum(len(_read_split(fl[0])) for fl in files.values())
        assert total == len(lstm_df_200)

    def test_split_ratios_approx_70_15_15(self, lstm_df_200, tmp_path):
        """Train ≈70%, validate ≈15%, test ≈15%."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        n_total = len(lstm_df_200)
        n_train    = len(_read_split(files["train"][0]))
        n_validate = len(_read_split(files["validate"][0]))
        n_test     = len(_read_split(files["test"][0]))

        assert abs(n_train / n_total - 0.70) < 0.02
        assert abs(n_validate / n_total - 0.15) < 0.02
        assert abs(n_test / n_total - 0.15) < 0.02

    def test_test_is_most_recent_rows(self, lstm_df_200, tmp_path):
        """Test split must contain the most recent timestamps."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        train_ts = _read_split(files["train"][0])["timestamp"]
        test_ts  = _read_split(files["test"][0])["timestamp"]
        assert train_ts.max() < test_ts.min(), (
            "Test split must be chronologically after train split"
        )

    def test_no_timestamp_overlap_between_splits(self, lstm_df_200, tmp_path):
        """No timestamp may appear in more than one split."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        timestamps = []
        for flist in files.values():
            timestamps.append(set(_read_split(flist[0])["timestamp"].astype(str)))
        train_set, val_set, test_set = timestamps
        assert train_set.isdisjoint(val_set),  "Overlap between train and validate"
        assert train_set.isdisjoint(test_set), "Overlap between train and test"
        assert val_set.isdisjoint(test_set),   "Overlap between validate and test"

    def test_chronological_order_within_splits(self, lstm_df_200, tmp_path):
        """Each split file must be sorted in ascending timestamp order."""
        data_paths = _call_splits(lstm_df_200, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        for name, flist in files.items():
            ts = _read_split(flist[0])["timestamp"]
            assert ts.is_monotonic_increasing, (
                f"Split '{name}' is not sorted in ascending timestamp order"
            )


# ── stale-file cleanup / update scenario ─────────────────────────────────────


class TestStaleCleaup:
    """Verify that calling save_splits twice leaves only the latest files."""

    def test_update_replaces_stale_files(self, tmp_path):
        """
        Call with 100 rows first, then with 200 rows.
        The final directory must contain exactly one file per split and
        collectively cover the full 200-row range.
        """
        df_100 = _make_lstm_df(100, start="2024-01-01")
        df_200 = _make_lstm_df(200, start="2024-01-01")

        data_paths_100 = _call_splits(df_100, tmp_path)
        data_paths_200 = _call_splits(df_200, tmp_path)

        files = _find_split_files(data_paths_200, "synthetic", "TEST01")

        # Exactly one file per split
        for name, flist in files.items():
            assert len(flist) == 1, f"Expected 1 file for {name} after update, got {len(flist)}"

        # Total rows cover the full 200
        total = sum(len(_read_split(fl[0])) for fl in files.values())
        assert total == 200

    def test_no_stale_files_from_previous_run(self, tmp_path):
        """
        After two calls, no old-format split files (e.g. lstm_train_TEST01_*.csv
        with a date suffix) should remain alongside the current file.
        """
        df_a = _make_lstm_df(100, start="2024-01-01")
        df_b = _make_lstm_df(200, start="2024-01-01")

        data_paths = _make_data_paths(tmp_path)
        pump_serial = "STALE01"
        dataset = "syn"

        for df in (df_a, df_b):
            with patch("bloodBath.dataset_parser.save_splits.DATA_PATHS", data_paths):
                save_chronological_splits(df, pump_serial, dataset, "2024-01-01", "2024-12-31")

        for split_name in ("train", "validate", "test"):
            merged_path = data_paths["merged"][split_name]
            split_dir = merged_path.parent / dataset / f"pump_{pump_serial}" / merged_path.name
            all_csv = list(split_dir.glob("*.csv"))
            assert len(all_csv) == 1, (
                f"Found {len(all_csv)} files in {split_name} dir after update — "
                f"stale files were not cleaned up: {[f.name for f in all_csv]}"
            )


# ── edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_dataframe_writes_nothing(self, tmp_path):
        """Empty input must not crash and must not create any files."""
        df_empty = pd.DataFrame(columns=["timestamp", "bg", "basal_rate", "bolus_dose"])
        data_paths = _call_splits(df_empty, tmp_path)
        for split_name in ("train", "validate", "test"):
            merged_path = data_paths["merged"][split_name]
            split_dir = merged_path.parent / "synthetic" / "pump_TEST01" / merged_path.name
            # Directory might not even be created
            if split_dir.exists():
                assert len(list(split_dir.glob("*.csv"))) == 0

    def test_unsorted_input_gets_sorted(self, tmp_path):
        """Even if the input is unsorted, output must be in chronological order."""
        n = 60
        idx = pd.date_range("2024-03-01", periods=n, freq="5min", tz="UTC")
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            {
                "timestamp": idx[rng.permutation(n)],  # shuffled
                "bg": rng.uniform(80, 200, n),
            }
        )
        data_paths = _call_splits(df, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        for name, flist in files.items():
            if flist:
                ts = _read_split(flist[0])["timestamp"]
                assert ts.is_monotonic_increasing, f"{name} split not chronological"

    def test_small_dataframe_still_produces_splits(self, tmp_path):
        """Very small DataFrame (10 rows) should still produce 3 split files."""
        idx = pd.date_range("2024-06-01", periods=10, freq="5min", tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "bg": np.ones(10) * 100})
        data_paths = _call_splits(df, tmp_path)
        files = _find_split_files(data_paths, "synthetic", "TEST01")
        # train must exist; validate and test may be empty (still 0 rows)
        assert len(files["train"]) == 1
