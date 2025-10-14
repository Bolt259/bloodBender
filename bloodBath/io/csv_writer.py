"""
CSV Writer with Standardized Headers and Naming Conventions

Enforces bloodBath v2.0 CSV format:
- Comment header block with metadata
- Standardized naming conventions
- UTC timestamp storage
- Outlier flagging and metadata
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CsvFileType(Enum):
    """Enum for CSV file types"""
    RAW_CGM = "raw_cgm"
    RAW_BASAL = "raw_basal"
    RAW_BOLUS = "raw_bolus"
    MERGED_LSTM = "merged_lstm"
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


class CsvWriter:
    """Writes CSV files with standardized headers and naming conventions"""
    
    # BG outlier policy
    BG_MIN = 20  # mg/dL
    BG_MAX = 600  # mg/dL
    
    def __init__(self, output_dir: Path):
        """
        Initialize CSV writer
        
        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_csv_with_header(
        self,
        df: pd.DataFrame,
        pump_serial: str,
        file_type: CsvFileType,
        year_month: Optional[str] = None,
        date_range_start: Optional[datetime] = None,
        date_range_end: Optional[datetime] = None,
        notes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Write CSV file with standardized header
        
        Args:
            df: DataFrame to write
            pump_serial: Pump serial number
            file_type: Type of CSV file
            year_month: Month in YYYYMM format (optional)
            date_range_start: Start of date range (optional)
            date_range_end: End of date range (optional)
            notes: Additional notes for header
            metadata: Additional metadata for header
            
        Returns:
            Path to written file
        """
        # Build filename
        filename = self._build_filename(pump_serial, file_type, year_month, 
                                       date_range_start, date_range_end)
        filepath = self.output_dir / filename
        
        # Build header
        header = self._build_header(
            pump_serial=pump_serial,
            file_type=file_type,
            year_month=year_month,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            df=df,
            notes=notes,
            metadata=metadata
        )
        
        # Write file with header
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(header)
            df.to_csv(f, index=False)
        
        logger.info(f"✅ Written {len(df)} records to {filepath}")
        return filepath
    
    def _build_filename(
        self,
        pump_serial: str,
        file_type: CsvFileType,
        year_month: Optional[str],
        date_range_start: Optional[datetime],
        date_range_end: Optional[datetime]
    ) -> str:
        """Build standardized filename"""
        # Base pattern: pump_<serial>_<YYYYMM>_{type}.csv
        parts = [f"pump_{pump_serial}"]
        
        if year_month:
            parts.append(year_month)
        elif date_range_start and date_range_end:
            start_str = date_range_start.strftime("%Y-%m-%d")
            end_str = date_range_end.strftime("%Y-%m-%d")
            parts.append(f"{start_str}_to_{end_str}")
        
        parts.append(file_type.value)
        
        return "_".join(parts) + ".csv"
    
    def _build_header(
        self,
        pump_serial: str,
        file_type: CsvFileType,
        year_month: Optional[str],
        date_range_start: Optional[datetime],
        date_range_end: Optional[datetime],
        df: pd.DataFrame,
        notes: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Build standardized CSV header"""
        lines = []
        lines.append("# bloodBath v2.0 CSV Data File")
        lines.append("# ==============================")
        lines.append(f"# data_version: v2")
        lines.append(f"# generated_at_utc: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"# pump_serial: {pump_serial}")
        lines.append(f"# file_role: {file_type.value}")
        
        # Date range
        if date_range_start and date_range_end:
            start_iso = date_range_start.isoformat() if hasattr(date_range_start, 'isoformat') else str(date_range_start)
            end_iso = date_range_end.isoformat() if hasattr(date_range_end, 'isoformat') else str(date_range_end)
            lines.append(f"# date_range_utc: {start_iso} → {end_iso}")
        elif year_month:
            lines.append(f"# year_month: {year_month}")
        
        # Timezone handling
        lines.append(f"# tz_handling: stored_in=UTC")
        
        # Record count
        lines.append(f"# record_count: {len(df)}")
        
        # Columns
        if not df.empty:
            lines.append(f"# columns: {', '.join(df.columns)}")
        
        # BG outlier policy
        if 'bg' in df.columns:
            lines.append(f"# bg_range_policy: clipped to [{self.BG_MIN}, {self.BG_MAX}] mg/dL")
            if 'bg_clip_flag' in df.columns:
                clipped_count = df['bg_clip_flag'].sum()
                lines.append(f"# bg_clipped_count: {clipped_count}")
        
        # Additional notes
        if notes:
            lines.append("# notes:")
            for note in notes:
                lines.append(f"#   - {note}")
        
        # Metadata
        if metadata:
            lines.append("# metadata:")
            for key, value in metadata.items():
                lines.append(f"#   {key}: {value}")
        
        lines.append("# ==============================")
        lines.append("")  # Empty line before data
        
        return "\n".join(lines) + "\n"


def write_csv_with_header(
    df: pd.DataFrame,
    output_path: Path,
    pump_serial: str,
    file_type: CsvFileType,
    **kwargs
) -> Path:
    """
    Convenience function to write CSV with header
    
    Args:
        df: DataFrame to write
        output_path: Output file path
        pump_serial: Pump serial number
        file_type: Type of CSV file
        **kwargs: Additional arguments for header
        
    Returns:
        Path to written file
    """
    writer = CsvWriter(output_path.parent)
    return writer.write_csv_with_header(
        df=df,
        pump_serial=pump_serial,
        file_type=file_type,
        **kwargs
    )
