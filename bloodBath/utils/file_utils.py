"""
File utilities for CSV writing, folder creation, and metadata tracking
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(path: Path) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataframe_to_csv(df: pd.DataFrame, 
                          output_file: Path,
                          append_mode: bool = False) -> None:
    """
    Write DataFrame to CSV file
    
    Args:
        df: DataFrame to write
        output_file: Output file path
        append_mode: Whether to append to existing file
    """
    if df.empty:
        logger.warning(f"No data to save for {output_file}")
        return
    
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = output_file.exists()
    
    # Write CSV
    mode = 'a' if append_mode else 'w'
    header = not (file_exists and append_mode)
    
    df.to_csv(output_file, mode=mode, header=header, index=False)
    
    logger.info(f"{'Appended' if append_mode else 'Wrote'} {len(df)} records to {output_file}")


def write_dicts_to_csv(data: List[Dict[str, Any]], 
                      output_file: Path,
                      append_mode: bool = False) -> None:
    """
    Write list of dictionaries to CSV file
    
    Args:
        data: List of dictionaries to write
        output_file: Output file path  
        append_mode: Whether to append to existing file
    """
    if not data:
        logger.warning(f"No data to save for {output_file}")
        return
    
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all field names
    fieldnames = set()
    for record in data:
        fieldnames.update(record.keys())
    fieldnames = sorted(fieldnames)
    
    # Check if file exists to determine if we need headers
    file_exists = output_file.exists()
    mode = 'a' if append_mode else 'w'
    
    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write headers if new file or not appending
        if not (file_exists and append_mode):
            writer.writeheader()
        
        writer.writerows(data)
    
    logger.info(f"{'Appended' if append_mode else 'Wrote'} {len(data)} records to {output_file}")


def get_output_filename(output_dir: Path,
                       pump_serial: str, 
                       start_date: str, 
                       end_date: str,
                       is_update: bool = False) -> Path:
    """
    Generate output filename for a date range
    
    Args:
        output_dir: Base output directory
        pump_serial: Pump serial number
        start_date: Start date string
        end_date: End date string
        is_update: Whether this is an update operation
        
    Returns:
        Path to output file
    """
    pump_dir = output_dir / f'pump_{pump_serial}'
    pump_dir.mkdir(exist_ok=True)
    
    if is_update:
        filename = f'update_{start_date}_to_{end_date}.csv'
    else:
        filename = f'{start_date}_to_{end_date}.csv'
    
    return pump_dir / filename


def get_lstm_output_path(output_dir: Path, pump_serial: str) -> Path:
    """
    Get the output path for LSTM-ready data using structured approach
    
    Args:
        output_dir: Base output directory
        pump_serial: Pump serial number
        
    Returns:
        Path to LSTM-ready CSV file
    """
    # Check if this is a structured directory
    lstm_pump_data_dir = output_dir / "lstm_pump_data"
    if lstm_pump_data_dir.exists():
        # Use structured approach
        from .structure_utils import get_lstm_pump_data_file
        structure = {
            'base': output_dir,
            'lstm_pump_data': lstm_pump_data_dir,
            'metadata': output_dir / 'metadata',
            'models': output_dir / 'models',
            'logs': output_dir / 'logs'
        }
        return get_lstm_pump_data_file(structure, pump_serial)
    else:
        # Fall back to old structure for backward compatibility
        lstm_dir = output_dir / "data" / "lstm_ready"
        lstm_dir.mkdir(parents=True, exist_ok=True)
        return lstm_dir / f"lstm_ready_{pump_serial}.csv"


def save_lstm_ready_data(df: pd.DataFrame,
                        output_file: Path,
                        pump_serial: str,
                        also_save_chunks: bool = True) -> None:
    """
    Save LSTM-ready data with proper formatting
    
    Args:
        df: DataFrame with LSTM-ready data
        output_file: Chunk output file path
        pump_serial: Pump serial number
        also_save_chunks: Whether to also save individual chunks
    """
    if df.empty:
        logger.warning(f"No data to save for {output_file}")
        return
    
    # Get LSTM output path
    lstm_file = get_lstm_output_path(output_file.parent.parent, pump_serial)
    
    # Check if file exists to determine if we need headers
    file_exists = lstm_file.exists()
    
    # Save/append to LSTM-ready CSV
    df.to_csv(lstm_file, mode='a', header=not file_exists, index=False)
    
    logger.info(f"Saved {len(df)} records to {lstm_file}")
    
    # Also save the chunk for debugging/compatibility if needed
    if also_save_chunks:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.debug(f"Also saved chunk to {output_file}")


def load_csv_with_datetime(csv_file: Path, 
                          datetime_col: str = 'created_at') -> pd.DataFrame:
    """
    Load CSV file and parse datetime column
    
    Args:
        csv_file: Path to CSV file
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with parsed datetime
    """
    if not csv_file.exists():
        logger.warning(f"File not found: {csv_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        if datetime_col not in df.columns:
            logger.warning(f"Column '{datetime_col}' not found in {csv_file}")
            return pd.DataFrame()
        
        # Parse timestamps and convert to UTC
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df = df.dropna(subset=[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_file}: {e}")
        return pd.DataFrame()


def find_most_recent_files(base_dir: Path, 
                          pattern: str) -> List[Path]:
    """
    Find the most recent files matching a pattern
    
    Args:
        base_dir: Base directory to search
        pattern: Glob pattern to match
        
    Returns:
        List of file paths sorted by modification time (newest first)
    """
    files = list(base_dir.glob(pattern))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def merge_csv_files(input_files: List[Path],
                   output_file: Path,
                   add_source_column: bool = False) -> None:
    """
    Merge multiple CSV files into one
    
    Args:
        input_files: List of input CSV files
        output_file: Output merged CSV file
        add_source_column: Whether to add source filename column
    """
    if not input_files:
        logger.warning("No input files to merge")
        return
    
    combined_data = []
    
    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            
            if add_source_column:
                df['source_file'] = file_path.name
            
            combined_data.append(df)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not combined_data:
        logger.warning("No data loaded for merging")
        return
    
    # Combine all data
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Save merged data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    logger.info(f"Merged {len(combined_df)} records from {len(input_files)} files to {output_file}")


def get_structured_output_filename(structure: Dict[str, Path],
                                   pump_serial: str,
                                   start_date: str,
                                   end_date: str,
                                   data_type: str = 'lstm_ready') -> Path:
    """
    Get output filename using structured directory layout
    
    Args:
        structure: Directory structure mapping
        pump_serial: Pump serial number
        start_date: Start date string
        end_date: End date string
        data_type: Type of data (defaults to 'lstm_ready')
        
    Returns:
        Path to output file
    """
    
    # Use lstm_pump_data directory for simplified structure
    data_dir = structure['lstm_pump_data']
    
    # Create filename with pump serial and timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"pump_{pump_serial}_{timestamp}.csv"
    
    return data_dir / filename


def save_structured_lstm_data(structure: Dict[str, Path],
                             df: pd.DataFrame,
                             pump_serial: str,
                             start_date: str,
                             end_date: str,
                             validate_data: bool = True) -> Optional[Path]:
    """
    Save LSTM-ready data using structured directory layout with optional validation
    
    Args:
        structure: Directory structure mapping
        df: DataFrame to save
        pump_serial: Pump serial number
        start_date: Start date string
        end_date: End date string
        validate_data: Whether to validate data quality before saving
        
    Returns:
        Path to saved file or None if no data or validation fails
    """
    if df.empty:
        logger.warning(f"No LSTM data to save for pump {pump_serial}")
        return None
    
    output_file = get_structured_output_filename(
        structure, pump_serial, start_date, end_date, 'lstm_ready'
    )
    
    # ✅ DATA QUALITY VALIDATION - Validate before saving
    validation_result = None
    if validate_data and 'bg' in df.columns:
        try:
            from ..data.csv_validator import CSVValidator
            
            # Create a temporary file for validation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                
                validator = CSVValidator()
                validation_result = validator.validate_csv(temp_file.name)
                
                # Clean up temp file
                import os
                os.unlink(temp_file.name)
                
            if not validation_result.is_valid:
                logger.warning(f"🚨 SYNTHETIC DATA DETECTED for pump {pump_serial} ({start_date} to {end_date})")
                logger.warning(f"   Confidence: {validation_result.confidence_score:.3f}")
                logger.warning(f"   Issues: {', '.join(validation_result.issues)}")
                
                # Generate default template instead of saving synthetic data
                return _generate_default_template(output_file, pump_serial, start_date, end_date, validation_result)
                
            else:
                logger.info(f"✅ Data validation passed for pump {pump_serial} (confidence: {validation_result.confidence_score:.3f})")
                
        except Exception as e:
            logger.warning(f"Data validation failed with error: {e}. Proceeding with save...")
    
    # Save with metadata header including validation results
    with open(output_file, 'w') as f:
        f.write(f"# bloodBath LSTM-ready dataset\n")
        f.write(f"# Pump Serial: {pump_serial}\n")
        f.write(f"# Date Range: {start_date} to {end_date}\n")
        f.write(f"# Records: {len(df)}\n")
        f.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"# Columns: {', '.join(df.columns)}\n")
        if validation_result:
            f.write(f"# Validation: {'PASSED' if validation_result.is_valid else 'FAILED'}\n")
            f.write(f"# Confidence: {validation_result.confidence_score:.3f}\n")
        f.write("\n")
    
    # Append the actual data
    df.to_csv(output_file, mode='a', index=False)
    
    logger.info(f"Saved {len(df)} LSTM records to {output_file}")
    return output_file


def _generate_default_template(output_file: Path, 
                              pump_serial: str, 
                              start_date: str, 
                              end_date: str,
                              validation_result: Any) -> Path:
    """
    Generate a default template CSV when synthetic data is detected
    
    Args:
        output_file: Intended output file path
        pump_serial: Pump serial number
        start_date: Start date string
        end_date: End date string
        validation_result: Validation result object
        
    Returns:
        Path to generated default template
    """
    # Create template filename
    template_file = output_file.with_name(f"SYNTHETIC_TEMPLATE_{output_file.name}")
    
    # Generate minimal realistic template data
    import pandas as pd
    import numpy as np
    
    # Create date range with 5-minute intervals
    date_range = pd.date_range(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        freq='5min'
    )
    
    # Generate realistic baseline template
    n_records = len(date_range)
    template_data = pd.DataFrame({
        'timestamp': date_range,
        'bg': np.random.normal(140, 25, n_records).clip(80, 200),  # Realistic BG range
        'delta_bg': np.random.normal(0, 3, n_records).clip(-20, 20),
        'basal_rate': np.random.normal(1.2, 0.2, n_records).clip(0.1, 3.0),
        'bolus_dose': np.zeros(n_records),  # Mostly zeros with occasional boluses
        'sin_time': np.sin(2 * np.pi * date_range.hour / 24),
        'cos_time': np.cos(2 * np.pi * date_range.hour / 24),
    })
    
    # Add occasional bolus doses (5% of records)
    bolus_indices = np.random.choice(n_records, size=int(0.05 * n_records), replace=False)
    bolus_values = np.random.exponential(2.5, len(bolus_indices)).clip(0.1, 15.0)
    template_data.iloc[bolus_indices, template_data.columns.get_loc('bolus_dose')] = bolus_values
    
    # Save template with warning header
    with open(template_file, 'w') as f:
        f.write(f"# ⚠️  DEFAULT TEMPLATE - SYNTHETIC DATA REJECTED\n")
        f.write(f"# Original data for pump {pump_serial} was rejected as synthetic/placeholder\n")
        f.write(f"# Validation confidence: {validation_result.confidence_score:.3f}\n")
        f.write(f"# Validation issues: {'; '.join(validation_result.issues)}\n")
        f.write(f"# Date Range: {start_date} to {end_date}\n")
        f.write(f"# Template Records: {len(template_data)}\n")
        f.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"# USE FOR STRUCTURE ONLY - DO NOT USE FOR TRAINING\n")
        f.write("\n")
    
    # Append template data
    template_data.to_csv(template_file, mode='a', index=False)
    
    logger.info(f"🔄 Generated default template: {template_file}")
    logger.info(f"   Template contains {len(template_data)} baseline records")
    logger.info(f"   ⚠️  DO NOT USE FOR TRAINING - Structure template only")
    
    return template_file


def save_structured_metadata(structure: Dict[str, Path],
                           metadata: Dict[str, Any],
                           filename: str) -> Path:
    """
    Save metadata using structured directory layout
    
    Args:
        structure: Directory structure mapping
        metadata: Metadata dictionary
        filename: Metadata filename
        
    Returns:
        Path to saved metadata file
    """
    from .structure_utils import get_metadata_file
    
    metadata_file = get_metadata_file(structure, filename)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to {metadata_file}")
    return metadata_file


def remove_invalid_csv_files(directory: Path, 
                             column_mapping: Optional[Dict[str, str]] = None,
                             invalid_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Delete CSV files where ALL BG values are NaN and all basal/bolus values are 0.0
    
    This post-processing cleanup step removes completely invalid placeholder CSV files that
    contain no meaningful physiological data, ensuring only valid training data
    remains in the dataset.
    
    Args:
        directory: Directory to scan for CSV files (recursive)
        column_mapping: Optional dict to map column names (e.g., {'bg': 'blood_glucose'})
        invalid_threshold: Fraction of invalid records to trigger removal (default: 1.0 = 100%)
        
    Returns:
        Dictionary with cleanup statistics:
            - invalid_csvs_removed: Number of files deleted
            - total_files_scanned: Total CSV files checked
            - bytes_freed: Disk space freed in bytes
            - removed_files: List of removed file paths with stats
            - last_cleanup: Timestamp of cleanup
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return {
            'invalid_csvs_removed': 0,
            'total_files_scanned': 0,
            'bytes_freed': 0,
            'removed_files': [],
            'last_cleanup': pd.Timestamp.now(tz='UTC').isoformat()
        }
    
    # Default column names (can be overridden)
    col_names = column_mapping or {
        'bg': 'bg',
        'basal_rate': 'basal_rate', 
        'bolus_dose': 'bolus_dose'
    }
    
    removed_files = []
    bytes_freed = 0
    total_scanned = 0
    
    logger.info(f"🧹 Starting post-processing cleanup in: {directory}")
    
    # Recursively find all CSV files
    csv_files = list(directory.rglob("*.csv"))
    
    for csv_file in csv_files:
        total_scanned += 1
        
        try:
            # Skip comment-only files or empty files
            if csv_file.stat().st_size < 100:  # Less than 100 bytes, likely empty
                continue
            
            # Read CSV, skipping comment lines
            df = pd.read_csv(csv_file, comment='#')
            
            # Check if required columns exist
            required_cols = set(col_names.values())
            available_cols = set(df.columns)
            
            if not required_cols.issubset(available_cols):
                # Try alternate column names (bolus vs bolus_dose)
                alt_mapping = {
                    'bg': 'bg',
                    'basal_rate': 'basal_rate',
                    'bolus_dose': 'bolus'  # Try 'bolus' if 'bolus_dose' not found
                }
                
                if set(alt_mapping.values()).issubset(available_cols):
                    col_names = alt_mapping
                else:
                    # logger.debug(f"Skipping {csv_file.name}: missing required columns")
                    continue
            
            # Extract columns
            bg_col = df[col_names['bg']]
            basal_col = df[col_names['basal_rate']]
            bolus_col = df[col_names['bolus_dose']]
            
            # Calculate invalid record ratio
            # Invalid = BG is NaN AND basal is 0 AND bolus is 0
            total_records = len(df)
            if total_records == 0:
                continue
            
            invalid_records = (
                bg_col.isna() & 
                (basal_col == 0.0) & 
                (bolus_col == 0.0)
            ).sum()
            
            invalid_ratio = invalid_records / total_records
            
            # Remove file if ≥ threshold (default 95%) of records are invalid
            is_invalid = invalid_ratio >= invalid_threshold
            
            if is_invalid:
                file_size = csv_file.stat().st_size
                bytes_freed += file_size
                removed_files.append(str(csv_file.relative_to(directory)))
                
                csv_file.unlink()
                logger.info(f"🗑️  Removed invalid CSV: {csv_file.name} ({invalid_ratio*100:.1f}% invalid, {file_size:,} bytes)")
                
        except Exception as e:
            logger.warning(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Summary statistics
    cleanup_stats = {
        'invalid_csvs_removed': len(removed_files),
        'total_files_scanned': total_scanned,
        'bytes_freed': bytes_freed,
        'removed_files': removed_files,
        'last_cleanup': pd.Timestamp.now(tz='UTC').isoformat()
    }
    
    if removed_files:
        logger.info(f"✅ Cleanup complete:")
        logger.info(f"   - Removed: {len(removed_files)} invalid CSV files")
        logger.info(f"   - Scanned: {total_scanned} total files")
        logger.info(f"   - Freed: {bytes_freed:,} bytes ({bytes_freed / 1024 / 1024:.2f} MB)")
    else:
        logger.info(f"✅ Cleanup complete: No invalid files found ({total_scanned} files scanned)")
    
    return cleanup_stats
