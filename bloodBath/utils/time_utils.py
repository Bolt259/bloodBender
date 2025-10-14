"""
Time utilities for timestamp handling and timezone conversion
"""

import arrow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Union


def tz_convert(timestamp: Union[str, datetime, pd.Timestamp], 
               to_tz: str = 'UTC') -> pd.Timestamp:
    """
    Convert timestamp to specified timezone
    
    Args:
        timestamp: Input timestamp
        to_tz: Target timezone
        
    Returns:
        Converted timestamp
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    elif isinstance(timestamp, datetime):
        timestamp = pd.Timestamp(timestamp)
    
    # If timestamp is naive, assume UTC
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    # Convert to target timezone
    return timestamp.tz_convert(to_tz)


def round_to_5min(timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    Round timestamp to nearest 5-minute interval
    
    Args:
        timestamp: Input timestamp
        
    Returns:
        Rounded timestamp
    """
    return timestamp.round('5min')


def generate_time_index(start_time: pd.Timestamp, 
                       end_time: pd.Timestamp, 
                       freq: str = '5min') -> pd.DatetimeIndex:
    """
    Generate regular time index between start and end times
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        freq: Frequency string (default: '5min')
        
    Returns:
        DatetimeIndex with regular intervals
    """
    start_rounded = start_time.floor(freq)
    end_rounded = end_time.ceil(freq)
    
    return pd.date_range(start=start_rounded, end=end_rounded, freq=freq)


def generate_date_chunks(start_date: str, 
                        end_date: str, 
                        chunk_days: int) -> List[Tuple[str, str]]:
    """
    Generate date chunks for API requests
    
    Args:
        start_date: Start date string (YYYY-MM-DD or ISO format)
        end_date: End date string (YYYY-MM-DD or ISO format)
        chunk_days: Number of days per chunk
        
    Returns:
        List of (start_date, end_date) tuples
    """
    start = arrow.get(start_date)
    end = arrow.get(end_date)
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current.shift(days=chunk_days), end)
        chunks.append((
            current.format('YYYY-MM-DD'),
            chunk_end.format('YYYY-MM-DD')
        ))
        current = chunk_end
    
    return chunks


def extract_timestamp_from_event(event) -> Optional[pd.Timestamp]:
    """
    Extract timestamp from a pump event using the proper Tandem epoch conversion
    
    Args:
        event: Pump event object
        
    Returns:
        Extracted timestamp or None if not found
    """
    import logging
    logger = logging.getLogger(__name__)
    
    timestamp = None
    
    # Priority 1: Use the event's timestamp property (already handles Tandem epoch conversion)
    if hasattr(event, 'timestamp') and event.timestamp is not None:
        ts = event.timestamp
        # Handle Arrow objects by converting to datetime first
        if hasattr(ts, 'datetime'):
            timestamp = pd.to_datetime(ts.datetime)
        else:
            timestamp = pd.to_datetime(ts)
        logger.debug(f"Extracted timestamp from event.timestamp: {timestamp}")
    
    # Priority 2: Use raw event's timestamp property (also handles Tandem epoch conversion)
    elif hasattr(event, 'raw') and hasattr(event.raw, 'timestamp') and event.raw.timestamp is not None:
        ts = event.raw.timestamp
        # Handle Arrow objects by converting to datetime first
        if hasattr(ts, 'datetime'):
            timestamp = pd.to_datetime(ts.datetime)
        else:
            timestamp = pd.to_datetime(ts)
        logger.debug(f"Extracted timestamp from event.raw.timestamp: {timestamp}")
    
    # Priority 3: Use eventTimestamp property if available
    elif hasattr(event, 'eventTimestamp') and event.eventTimestamp is not None:
        ts = event.eventTimestamp
        # Handle Arrow objects by converting to datetime first
        if hasattr(ts, 'datetime'):
            timestamp = pd.to_datetime(ts.datetime)
        else:
            timestamp = pd.to_datetime(ts)
        logger.debug(f"Extracted timestamp from event.eventTimestamp: {timestamp}")
    
    # Priority 4: Manual conversion using Tandem epoch (if raw timestamp exists)
    elif hasattr(event, 'raw') and hasattr(event.raw, 'timestampRaw') and event.raw.timestampRaw is not None:
        # Import here to avoid circular imports
        from eventparser.raw_event import TANDEM_EPOCH
        
        raw_timestamp = event.raw.timestampRaw
        try:
            # Convert from Tandem epoch (2008-01-01) to Unix timestamp
            unix_timestamp = TANDEM_EPOCH + raw_timestamp
            timestamp = pd.to_datetime(unix_timestamp, unit='s')
            logger.debug(f"Extracted timestamp from event.raw.timestampRaw with Tandem epoch: {timestamp}")
        except Exception as e:
            logger.warning(f"Could not parse timestampRaw {raw_timestamp}: {e}")
            timestamp = None
    
    # Priority 5: Try other timestamp fields as fallback
    elif hasattr(event, 'utc_timestamp') and event.utc_timestamp is not None:
        timestamp = pd.to_datetime(event.utc_timestamp)
        logger.debug(f"Extracted timestamp from event.utc_timestamp: {timestamp}")
    
    # Validate timestamp is reasonable (between 2008 and 2030)
    if timestamp is not None:
        if timestamp.year < 2008 or timestamp.year > 2030:
            logger.warning(f"Suspicious timestamp extracted: {timestamp}, setting to None")
            timestamp = None
        else:
            # Round to second precision to avoid microsecond duplicates
            timestamp = timestamp.round('s')
    
    return timestamp


def handle_duplicate_timestamps(df: pd.DataFrame, 
                              timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Handle duplicate timestamps in a DataFrame to prevent reindexing errors
    
    Args:
        df: DataFrame with potential duplicate timestamps
        timestamp_col: Name of the timestamp column
        
    Returns:
        DataFrame with duplicate timestamps handled
    """
    if df.empty or timestamp_col not in df.columns:
        return df
    
    # Count duplicates
    duplicates = df[timestamp_col].duplicated()
    if duplicates.any():
        # For duplicate timestamps, add microseconds to make them unique
        # This preserves the temporal order while making timestamps unique
        for i, is_dup in enumerate(duplicates):
            if is_dup:
                # Find how many duplicates we've seen for this timestamp
                current_ts = df.iloc[i][timestamp_col]
                prev_dups = df.iloc[:i][timestamp_col].eq(current_ts).sum()
                
                # Add microseconds to make it unique
                df.at[i, timestamp_col] = current_ts + pd.Timedelta(microseconds=prev_dups)
    
    return df


def add_time_of_day_features(time_index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add cyclical time-of-day features (sin/cos encoding)
    
    Args:
        time_index: DateTime index
        
    Returns:
        Tuple of (sin_time, cos_time) arrays
    """
    # Convert to hours from midnight
    hours = time_index.hour + time_index.minute / 60.0
    
    # Convert to cyclical features
    sin_time = np.sin(2 * np.pi * hours / 24)
    cos_time = np.cos(2 * np.pi * hours / 24)
    
    return sin_time, cos_time
