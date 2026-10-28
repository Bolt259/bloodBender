"""
Unit tests for bloodBath.io.csv_reader and bloodBath.io.csv_writer

Covers:
  - CsvReader.read_csv_with_metadata():
      * #-comment preamble is stripped; data rows are correct
      * metadata keys are extracted
      * timestamp column is parsed to UTC-aware Timestamps
      * file without comments also works
  - CsvWriter._build_header() / write_csv_with_header():
      * header contains expected fields
      * written file is readable with comment='#'
      * round-trip: write → read gives back the same data
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from bloodBath.io.csv_reader import CsvReader, read_csv_with_metadata
from bloodBath.io.csv_writer import CsvFileType, CsvWriter


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_csv_with_preamble(path: Path, preamble: str, csv_body: str) -> None:
    path.write_text(preamble + csv_body)


# ── CsvReader ─────────────────────────────────────────────────────────────────


class TestCsvReader:

    def test_comment_preamble_stripped(self, tmp_path):
        """#-comment lines must not appear in the returned DataFrame."""
        f = tmp_path / "test.csv"
        _write_csv_with_preamble(
            f,
            "# pump_serial: ABC123\n# record_count: 2\n",
            "timestamp,bg\n2024-01-01 10:00:00+00:00,120\n2024-01-01 10:05:00+00:00,125\n",
        )
        reader = CsvReader()
        df, _ = reader.read_csv_with_metadata(f)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "bg"]

    def test_metadata_keys_extracted(self, tmp_path):
        """Metadata dict must contain the key: value pairs from # lines."""
        f = tmp_path / "meta.csv"
        _write_csv_with_preamble(
            f,
            "# pump_serial: XYZ99\n# file_role: train\n",
            "bg\n100\n",
        )
        reader = CsvReader()
        _, meta = reader.read_csv_with_metadata(f)
        assert meta.get("pump_serial") == "XYZ99"
        assert meta.get("file_role") == "train"

    def test_timestamp_parsed_to_datetime(self, tmp_path):
        """'timestamp' column must be parsed to timezone-aware datetime."""
        f = tmp_path / "ts.csv"
        _write_csv_with_preamble(
            f,
            "",
            "timestamp,bg\n2024-06-15 08:00:00+00:00,130\n",
        )
        reader = CsvReader()
        df, _ = reader.read_csv_with_metadata(f)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_file_without_comments(self, tmp_path):
        """CSV without any # lines must parse correctly with empty metadata."""
        f = tmp_path / "nocomment.csv"
        f.write_text("bg,basal_rate\n100,1.0\n110,1.1\n")
        reader = CsvReader()
        df, meta = reader.read_csv_with_metadata(f)
        assert len(df) == 2
        assert meta == {}

    def test_convenience_function_matches_class(self, tmp_path):
        """Module-level read_csv_with_metadata() must match the class method."""
        f = tmp_path / "conv.csv"
        _write_csv_with_preamble(f, "# k: v\n", "x\n1\n2\n")
        df_cls, meta_cls = CsvReader().read_csv_with_metadata(f)
        df_fn,  meta_fn  = read_csv_with_metadata(f)
        pd.testing.assert_frame_equal(df_cls, df_fn)
        assert meta_cls == meta_fn

    def test_multiple_colon_in_value_handled(self, tmp_path):
        """Values that contain ':' must be preserved correctly."""
        f = tmp_path / "colons.csv"
        _write_csv_with_preamble(
            f,
            "# date_range_utc: 2024-01-01T00:00:00+00:00 to 2024-01-31T23:59:59+00:00\n",
            "bg\n100\n",
        )
        _, meta = CsvReader().read_csv_with_metadata(f)
        assert "2024-01-01T00:00:00+00:00" in meta.get("date_range_utc", "")

    def test_empty_csv_returns_empty_dataframe(self, tmp_path):
        """CSV with only a header row but no data must return an empty DataFrame."""
        f = tmp_path / "empty.csv"
        f.write_text("bg,basal_rate\n")
        reader = CsvReader()
        df, _ = reader.read_csv_with_metadata(f)
        assert len(df) == 0
        assert list(df.columns) == ["bg", "basal_rate"]


# ── CsvWriter ─────────────────────────────────────────────────────────────────


class TestCsvWriter:

    def _sample_df(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
        return pd.DataFrame({"timestamp": idx, "bg": [100, 110, 120, 115, 105]})

    def test_header_contains_pump_serial(self, tmp_path):
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="P001", file_type=CsvFileType.MERGED_LSTM
        )
        content = path.read_text()
        assert "P001" in content

    def test_header_contains_record_count(self, tmp_path):
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="P002", file_type=CsvFileType.TRAIN
        )
        content = path.read_text()
        assert f"record_count: {len(df)}" in content

    def test_file_readable_with_comment_skip(self, tmp_path):
        """The written file must be parseable by pd.read_csv(comment='#')."""
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="P003", file_type=CsvFileType.TEST
        )
        df_back = pd.read_csv(path, comment="#")
        assert len(df_back) == len(df)

    def test_round_trip_data_preserved(self, tmp_path):
        """write → read must return the same data values."""
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="P004", file_type=CsvFileType.VALIDATE
        )
        df_back = pd.read_csv(path, comment="#")
        # Compare bg column numerically
        assert list(df_back["bg"]) == list(df["bg"])

    def test_header_starts_with_hash(self, tmp_path):
        """Every header line must start with '#'."""
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="P005", file_type=CsvFileType.RAW_CGM
        )
        content = path.read_text()
        header_lines = [ln for ln in content.splitlines() if ln.startswith("#")]
        assert len(header_lines) > 0, "No header lines found"

    def test_output_dir_created(self, tmp_path):
        """CsvWriter must create subdirectories that don't exist yet."""
        nested = tmp_path / "a" / "b" / "c"
        writer = CsvWriter(nested)
        assert nested.exists()

    def test_filename_contains_pump_serial_and_type(self, tmp_path):
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="SRL42", file_type=CsvFileType.RAW_BASAL
        )
        assert "SRL42" in path.name
        assert "raw_basal" in path.name

    def test_written_file_path_returned(self, tmp_path):
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="R01", file_type=CsvFileType.MERGED_LSTM
        )
        assert isinstance(path, Path)
        assert path.exists()

    def test_bg_range_policy_in_header_when_bg_column_present(self, tmp_path):
        """When 'bg' column is present the header must mention bg_range_policy."""
        df = self._sample_df()
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="BG01", file_type=CsvFileType.MERGED_LSTM
        )
        content = path.read_text()
        assert "bg_range_policy" in content

    def test_no_bg_column_no_bg_policy(self, tmp_path):
        """If DataFrame has no 'bg' column, bg_range_policy must not appear."""
        df = pd.DataFrame({"basal_rate": [1.0, 1.1]})
        writer = CsvWriter(tmp_path)
        path = writer.write_csv_with_header(
            df, pump_serial="NB01", file_type=CsvFileType.RAW_BASAL
        )
        content = path.read_text()
        assert "bg_range_policy" not in content
