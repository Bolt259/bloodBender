"""
CSV Reader with Metadata Parsing

Reads bloodBath v2.0 CSV files and extracts header metadata.
"""

import pandas as pd
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class CsvReader:
    """Reads CSV files with metadata extraction"""
    
    def read_csv_with_metadata(self, filepath: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read CSV file and extract metadata from header
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        metadata = {}
        
        # Read header lines
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                
                # Parse metadata fields
                if ':' in line:
                    # Remove '# ' prefix and split on first ':'
                    clean_line = line.lstrip('#').strip()
                    if ':' in clean_line:
                        key, value = clean_line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        metadata[key] = value
        
        # Read CSV data (skip comment lines)
        df = pd.read_csv(filepath, comment='#')
        
        # Parse timestamp column if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        logger.debug(f"Read {len(df)} records from {filepath} with {len(metadata)} metadata fields")
        
        return df, metadata


def read_csv_with_metadata(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to read CSV with metadata
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    reader = CsvReader()
    return reader.read_csv_with_metadata(filepath)
