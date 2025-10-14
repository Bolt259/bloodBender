#!/usr/bin/env python3
"""
bloodBank Data Download Script

Downloads raw pump data using the proven tandem_historical_sync.py approach
and saves to bloodBank v2.0 structure for subsequent processing.

Adapted from tandem_historical_sync.py which successfully worked.
All API handling stays within tconnectsync package.

Features:
- Multi-pump support with configurable date ranges
- Intelligent chunking with automatic retry logic
- Outputs to bloodBank/raw/{cgm,basal,bolus}/{serial}/ structure
- Robust error handling with exponential backoff
- Metadata tracking for resumable sync operations
"""

import json
import os
import sys
import argparse
import logging
import time
import pandas as pd
import numpy as np
import arrow
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add tconnectsync-bb to path
sys.path.insert(0, '/home/bolt/projects/bb/tconnectsync-bb')

# Import tconnectsync modules
from tconnectsync.api import TConnectApi
from tconnectsync import secret

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tandem_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PumpConfig:
    """Configuration for a single pump"""
    serial: str
    start_date: str
    end_date: str
    device_id: Optional[int] = None
    model: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SyncMetadata:
    """Metadata tracking for sync operations"""
    pump_serial: str
    last_successful_sync: Optional[str] = None
    failed_ranges: List[Dict[str, str]] = None
    total_records: int = 0
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.failed_ranges is None:
            self.failed_ranges = []


class TandemSyncError(Exception):
    """Custom exception for sync errors"""
    pass


class RateLimitError(TandemSyncError):
    """Exception for rate limiting"""
    pass


class ChunkSizeError(TandemSyncError):
    """Exception when chunk size needs to be reduced"""
    pass


class TandemHistoricalSyncClient:
    """Main client for historical data synchronization"""
    
    DEFAULT_CHUNK_DAYS = 30
    MIN_CHUNK_DAYS = 1
    MAX_RETRIES = 5
    RETRY_DELAY_BASE = 2  # seconds
    
    # Event types we want to collect
    TARGET_EVENTS = ['BASAL', 'BOLUS', 'CGM_READING']
    
    def __init__(self, 
                 email: str = None,
                 password: str = None,
                 region: str = 'US',
                 output_dir: str = './bloodBath/bloodBank/raw',
                 chunk_days: int = DEFAULT_CHUNK_DAYS,
                 max_retries: int = MAX_RETRIES,
                 rate_limit_delay: int = 1):
        """
        Initialize the sync client
        
        Args:
            email: t:connect email (defaults to config)
            password: t:connect password (defaults to config)
            region: Server region (US or EU)
            output_dir: Directory for output files
            chunk_days: Size of date chunks in days
            max_retries: Maximum retry attempts
            rate_limit_delay: Base delay between requests
        """
        self.email = email or secret.TCONNECT_EMAIL
        self.password = password or secret.TCONNECT_PASSWORD
        self.region = region
        self.output_dir = Path(output_dir)
        self.chunk_days = chunk_days
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API connection
        self.api = None
        self.metadata_file = self.output_dir / 'sync_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized TandemHistoricalSyncClient")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Chunk size: {self.chunk_days} days")
    
    def _load_metadata(self) -> Dict[str, SyncMetadata]:
        """Load sync metadata from file"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for serial, meta_dict in data.items():
                metadata[serial] = SyncMetadata(**meta_dict)
            
            return metadata
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save sync metadata to file"""
        try:
            data = {}
            for serial, metadata in self.metadata.items():
                data[serial] = asdict(metadata)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _connect_api(self):
        """Initialize API connection"""
        if self.api is None:
            logger.info(f"Connecting to Tandem API ({self.region} region)...")
            self.api = TConnectApi(self.email, self.password, self.region)
            logger.info("API connection established")
    
    def _generate_date_chunks(self, start_date: str, end_date: str, chunk_days: int) -> List[Tuple[str, str]]:
        """Generate date chunks for API requests"""
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
    
    def _get_pump_device_id(self, serial: str) -> Optional[int]:
        """Get device ID for a pump serial number"""
        try:
            self._connect_api()
            pump_metadata = self.api.tandemsource.pump_event_metadata()
            
            for pump in pump_metadata:
                if pump['serialNumber'] == serial:
                    return pump['tconnectDeviceId']
            
            logger.error(f"Pump {serial} not found in account")
            return None
            
        except Exception as e:
            logger.error(f"Error getting device ID for pump {serial}: {e}")
            return None
    
    def _fetch_pump_data_chunk(self, 
                              pump_config: PumpConfig,
                              start_date: str,
                              end_date: str,
                              chunk_days: int = None) -> pd.DataFrame:
        """
        Fetch data for a specific date range with error handling
        
        Args:
            pump_config: Pump configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Current chunk size (for recursive retry)
            
        Returns:
            DataFrame with combined pump data
        """
        if chunk_days is None:
            chunk_days = self.chunk_days
        
        # Ensure we have device ID
        if pump_config.device_id is None:
            pump_config.device_id = self._get_pump_device_id(pump_config.serial)
            if pump_config.device_id is None:
                raise TandemSyncError(f"Could not get device ID for pump {pump_config.serial}")
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                logger.info(f"Fetching data for pump {pump_config.serial}: {start_date} to {end_date}")
                
                self._connect_api()
                
                # Add small delay to prevent rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Fetch raw pump events
                raw_events = self.api.tandemsource.pump_events_raw(
                    pump_config.device_id,
                    min_date=start_date,
                    max_date=end_date
                )
                
                if not raw_events:
                    logger.warning(f"No raw events returned for pump {pump_config.serial} ({start_date} to {end_date})")
                    return pd.DataFrame()
                
                # Parse events
                events = self.api.tandemsource.pump_events(
                    pump_config.device_id,
                    min_date=start_date,
                    max_date=end_date,
                    fetch_all_event_types=True
                )
                
                # Debug: Check what type of object we got
                logger.debug(f"Events type: {type(events)}")
                if hasattr(events, '__iter__') and not isinstance(events, dict):
                    logger.debug("Events is iterable but not dict - will convert to list")
                    # Peek at first event if possible
                    try:
                        events_peek = list(events)
                        logger.debug(f"Got {len(events_peek)} events")
                        if events_peek:
                            logger.debug(f"First event sample: {events_peek[0] if events_peek else 'None'}")
                        # Reset events to the list we just created
                        events = events_peek
                    except Exception as e:
                        logger.error(f"Error peeking at events: {e}")
                
                # Convert to DataFrame format
                df = self._events_to_dataframe(events, pump_config.serial)
                
                logger.info(f"Successfully fetched {len(df)} records for pump {pump_config.serial}")
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 504:  # Gateway Timeout
                    if chunk_days > self.MIN_CHUNK_DAYS:
                        logger.warning(f"Gateway timeout, reducing chunk size from {chunk_days} to {chunk_days // 2}")
                        new_chunk_days = max(chunk_days // 2, self.MIN_CHUNK_DAYS)
                        
                        # Split the current chunk and process recursively
                        mid_date = arrow.get(start_date).shift(days=chunk_days // 2).format('YYYY-MM-DD')
                        
                        df1 = self._fetch_pump_data_chunk(pump_config, start_date, mid_date, new_chunk_days)
                        df2 = self._fetch_pump_data_chunk(pump_config, mid_date, end_date, new_chunk_days)
                        
                        return pd.concat([df1, df2], ignore_index=True)
                    else:
                        raise ChunkSizeError(f"Cannot reduce chunk size below {self.MIN_CHUNK_DAYS} days")
                
                elif e.response.status_code == 429:  # Too Many Requests
                    retry_delay = self.RETRY_DELAY_BASE ** retry_count
                    logger.warning(f"Rate limited, waiting {retry_delay} seconds before retry {retry_count + 1}")
                    time.sleep(retry_delay)
                    retry_count += 1
                    continue
                
                elif e.response.status_code == 400:  # Bad Request
                    logger.error(f"Bad request for pump {pump_config.serial} ({start_date} to {end_date}): {e}")
                    raise TandemSyncError(f"Bad request: {e}")
                
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {e}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        time.sleep(self.RETRY_DELAY_BASE ** retry_count)
                        continue
                    raise TandemSyncError(f"HTTP error after {self.max_retries} retries: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error fetching data: {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.RETRY_DELAY_BASE ** retry_count)
                    continue
                raise TandemSyncError(f"Failed after {self.max_retries} retries: {e}")
        
        raise TandemSyncError(f"Failed to fetch data after {self.max_retries} retries")
    
    def _events_to_dataframe(self, events, pump_serial: str) -> pd.DataFrame:
        """Convert pump events to DataFrame format and prepare for LSTM-ready output"""
        if not events:
            return pd.DataFrame()
        
        # Step 1: Load Raw Events
        raw_events = self._load_raw_events(events)
        
        # Step 2: Normalize Timestamps
        cgm_data, basal_data, bolus_data = self._normalize_timestamps(raw_events)
        
        if not cgm_data:
            logger.debug("No CGM data found")
            return pd.DataFrame()
        
        # Step 3: Define 5-Minute Time Index
        time_index = self._define_time_index(cgm_data, basal_data, bolus_data)
        
        # Step 4: Aggregate & Align Signals
        aligned_data = self._aggregate_align_signals(cgm_data, basal_data, bolus_data, time_index)
        
        # Step 5: Feature Engineering
        lstm_ready_df = self._feature_engineering(aligned_data, time_index)
        
        return lstm_ready_df
    
    def _load_raw_events(self, events):
        """Step 1: Load Raw Events - Parse and categorize events by type"""
        # Handle both generator and list inputs
        if hasattr(events, '__iter__') and not isinstance(events, (dict, str)):
            if not isinstance(events, list):
                logger.debug("Events is iterable but not dict - will convert to list")
                events = list(events)
                logger.debug(f"Got {len(events)} events")
                if events:
                    logger.debug(f"First event sample: {events[0]}")
                logger.debug(f"Converted generator to list with {len(events)} events")
        
        # Categorize events by type
        basal_events = []
        bolus_events = []
        cgm_events = []
        
        for event in events:
            event_type = type(event).__name__
            
            # Enhanced basal event detection
            if any(keyword in event_type for keyword in ['Basal', 'basal']) or hasattr(event, 'commandedRate') or hasattr(event, 'commandedbasalrate'):
                basal_events.append(event)
            # Enhanced bolus event detection
            elif any(keyword in event_type for keyword in ['Bolus', 'bolus']) or hasattr(event, 'bolusAmount') or hasattr(event, 'insulin'):
                bolus_events.append(event)
            # Enhanced CGM event detection  
            elif any(keyword in event_type for keyword in ['Cgm', 'cgm', 'Gx']) or hasattr(event, 'sgv') or hasattr(event, 'bg'):
                cgm_events.append(event)
        
        logger.debug(f"Processing {len(basal_events)} basal, {len(bolus_events)} bolus, {len(cgm_events)} CGM events")
        
        return {
            'cgm_events': cgm_events,
            'basal_events': basal_events,
            'bolus_events': bolus_events
        }
    
    def _normalize_timestamps(self, raw_events):
        """Step 2: Normalize Timestamps - Extract and standardize timestamps"""
        cgm_data = []
        basal_data = []
        bolus_data = []
        
        # Process CGM events
        for event in raw_events['cgm_events']:
            try:
                timestamp = self._extract_timestamp(event)
                bg_value = self._extract_bg_value(event)
                
                if timestamp is not None and bg_value is not None and bg_value > 0:
                    cgm_data.append({
                        'timestamp': timestamp,
                        'bg': float(bg_value)
                    })
            except Exception as e:
                logger.debug(f"Error processing CGM event: {e}")
                continue
        
        # Process basal events
        for event in raw_events['basal_events']:
            try:
                timestamp = self._extract_timestamp(event)
                basal_rate = self._extract_basal_rate(event)
                
                if timestamp is not None and basal_rate >= 0:
                    basal_data.append({
                        'timestamp': timestamp,
                        'basal_rate': basal_rate
                    })
            except Exception as e:
                logger.debug(f"Error processing basal event: {e}")
                continue
        
        # Process bolus events
        for event in raw_events['bolus_events']:
            try:
                timestamp = self._extract_timestamp(event)
                bolus_dose = self._extract_bolus_dose(event)
                
                if timestamp is not None and bolus_dose > 0:
                    bolus_data.append({
                        'timestamp': timestamp,
                        'bolus_dose': bolus_dose
                    })
            except Exception as e:
                logger.debug(f"Error processing bolus event: {e}")
                continue
        
        return cgm_data, basal_data, bolus_data
    
    def _extract_timestamp(self, event):
        """
        Extract timestamp with proper timezone handling
        
        CRITICAL: event.timestamp is already parsed by tconnectsync's RawEvent class,
        which adds TANDEM_EPOCH (2008-01-01) to timestampRaw (in seconds) and applies
        timezone conversion. DO NOT reparse timestampRaw!
        
        Returns timezone-aware UTC timestamp with guard against invalid dates.
        """
        timestamp = None
        
        # Use the already-parsed timestamp property
        # tconnectsync RawEvent.timestamp = arrow.get(TANDEM_EPOCH + timestampRaw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)
        if hasattr(event, 'timestamp'):
            # event.timestamp is an Arrow object, convert to pandas datetime
            timestamp = pd.to_datetime(event.timestamp.datetime)
        elif hasattr(event, 'eventTimestamp'):
            # Some events use eventTimestamp property
            timestamp = pd.to_datetime(event.eventTimestamp.datetime)
        
        # Guard against invalid timestamps (< 2020-01-01)
        if timestamp is not None:
            min_valid_date = pd.Timestamp('2020-01-01', tz='UTC')
            if timestamp.tz is None:
                # Localize to UTC if naive
                timestamp = timestamp.tz_localize('UTC')
            if timestamp < min_valid_date:
                logger.warning(f"Invalid timestamp detected (< 2020): {timestamp}. Rejecting event.")
                return None
        
        return timestamp
    
    def _extract_bg_value(self, event):
        """Extract BG value from CGM event"""
        bg_value = None
        
        # Check for currentglucosedisplayvalue (used by LidCgmDataG7)
        if hasattr(event, 'currentglucosedisplayvalue'):
            bg_value = event.currentglucosedisplayvalue
        elif hasattr(event, 'sgv'):
            bg_value = event.sgv
        elif hasattr(event, 'bg'):
            bg_value = event.bg
        elif hasattr(event, 'value'):
            bg_value = event.value
        elif hasattr(event, 'glucoseValue'):
            bg_value = event.glucoseValue
        
        return bg_value
    
    def _extract_basal_rate(self, event):
        """Extract basal rate with proper unit conversion"""
        basal_rate = 0.0
        
        if hasattr(event, 'commandedRate') and event.commandedRate is not None:
            # commandedRate is in hundredths of units/hr
            basal_rate = float(event.commandedRate) / 100.0
        elif hasattr(event, 'commandedbasalrate') and event.commandedbasalrate is not None:
            basal_rate = float(event.commandedbasalrate)
        elif hasattr(event, 'value') and event.value is not None:
            value = float(event.value)
            basal_rate = value / 100.0 if value > 10 else value
        
        return basal_rate
    
    def _extract_bolus_dose(self, event):
        """Extract bolus dose with proper unit conversion"""
        bolus_dose = 0.0
        
        if hasattr(event, 'bolusAmount') and event.bolusAmount is not None:
            # bolusAmount is in hundredths of units
            bolus_dose = float(event.bolusAmount) / 100.0
        elif hasattr(event, 'insulin') and event.insulin is not None:
            insulin = float(event.insulin)
            bolus_dose = insulin / 100.0 if insulin > 10 else insulin
        elif hasattr(event, 'value') and event.value is not None:
            value = float(event.value)
            bolus_dose = value / 100.0 if value > 10 else value
        
        return bolus_dose
    
    def _define_time_index(self, cgm_data, basal_data, bolus_data):
        """Step 3: Define 5-Minute Time Index - Create regular 5-minute intervals"""
        # Convert to DataFrames for easier handling
        cgm_df = pd.DataFrame(cgm_data) if cgm_data else pd.DataFrame()
        basal_df = pd.DataFrame(basal_data) if basal_data else pd.DataFrame()
        bolus_df = pd.DataFrame(bolus_data) if bolus_data else pd.DataFrame()
        
        # Find the overall time range
        all_timestamps = []
        
        if not cgm_df.empty:
            all_timestamps.extend(cgm_df['timestamp'].tolist())
        if not basal_df.empty:
            all_timestamps.extend(basal_df['timestamp'].tolist())
        if not bolus_df.empty:
            all_timestamps.extend(bolus_df['timestamp'].tolist())
        
        if not all_timestamps:
            return pd.DatetimeIndex([])
        
        # Create 5-minute intervals
        start_time = pd.to_datetime(min(all_timestamps)).floor('5min')
        end_time = pd.to_datetime(max(all_timestamps)).ceil('5min')
        
        time_index = pd.date_range(start=start_time, end=end_time, freq='5min')
        
        return time_index
    
    def _aggregate_align_signals(self, cgm_data, basal_data, bolus_data, time_index):
        """Step 4: Aggregate & Align Signals - Resample to 5-minute intervals"""
        # Convert to DataFrames and handle duplicates
        cgm_df = pd.DataFrame(cgm_data) if cgm_data else pd.DataFrame()
        basal_df = pd.DataFrame(basal_data) if basal_data else pd.DataFrame()
        bolus_df = pd.DataFrame(bolus_data) if bolus_data else pd.DataFrame()
        
        # Handle duplicates
        if not cgm_df.empty:
            cgm_df = cgm_df.drop_duplicates(subset=['timestamp'], keep='last')
            cgm_df = cgm_df.set_index('timestamp').sort_index()
        
        if not basal_df.empty:
            basal_df = basal_df.drop_duplicates(subset=['timestamp'], keep='last')
            basal_df = basal_df.set_index('timestamp').sort_index()
        
        if not bolus_df.empty:
            bolus_df = bolus_df.drop_duplicates(subset=['timestamp'], keep='last')
            bolus_df = bolus_df.set_index('timestamp').sort_index()
        
        # Resample to 5-minute intervals
        aligned_data = {}
        
        # BG: Forward fill
        if not cgm_df.empty:
            aligned_data['bg'] = cgm_df['bg'].resample('5min').ffill().reindex(time_index).ffill()
        else:
            aligned_data['bg'] = pd.Series(index=time_index, dtype=float)
        
        # Basal rate: Forward fill
        if not basal_df.empty:
            aligned_data['basal_rate'] = basal_df['basal_rate'].resample('5min').ffill().reindex(time_index).fillna(0.0)
        else:
            aligned_data['basal_rate'] = pd.Series(0.0, index=time_index)
        
        # Bolus dose: Sum within each 5-minute interval
        if not bolus_df.empty:
            aligned_data['bolus_dose'] = bolus_df['bolus_dose'].resample('5min').sum().reindex(time_index).fillna(0.0)
        else:
            aligned_data['bolus_dose'] = pd.Series(0.0, index=time_index)
        
        return aligned_data
    
    def _feature_engineering(self, aligned_data, time_index):
        """Step 5: Feature Engineering - Calculate derived features"""
        # Create DataFrame
        df = pd.DataFrame(index=time_index)
        
        # Basic features - leave NaN for missing BG data (DO NOT fill with 100)
        df['bg'] = aligned_data['bg']  # Keep NaN where no CGM reading exists
        df['delta_bg'] = df['bg'].diff()  # NaN propagates correctly in diff
        df['basal_rate'] = aligned_data['basal_rate'].fillna(0.0)
        df['bolus_dose'] = aligned_data['bolus_dose'].fillna(0.0)
        
        # Flag missing BG readings
        df['bg_missing_flag'] = df['bg'].isna().astype(int)
        
        # Apply BG clipping with flag
        BG_MIN = 20
        BG_MAX = 600
        
        if 'bg' in df.columns and not df['bg'].isna().all():
            # Store original values for flagging
            original_bg = df['bg'].copy()
            
            # Clip BG values
            df['bg'] = df['bg'].clip(lower=BG_MIN, upper=BG_MAX)
            
            # Create flag column (1 if clipped, 0 otherwise)
            df['bg_clip_flag'] = ((original_bg < BG_MIN) | (original_bg > BG_MAX)).astype(int)
        else:
            df['bg_clip_flag'] = 0
        
        # Time-of-day cyclical features
        dt_index = pd.to_datetime(df.index)
        hours = dt_index.hour + dt_index.minute / 60.0
        df['sin_time'] = np.sin(2 * np.pi * hours / 24)
        df['cos_time'] = np.cos(2 * np.pi * hours / 24)
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Select final columns in LSTM-ready format (including flags)
        final_columns = ['timestamp', 'bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time', 'bg_clip_flag', 'bg_missing_flag']
        df = df[final_columns]
        
        return df
    
    def _create_lstm_ready_format(self, cgm_df: pd.DataFrame, basal_df: pd.DataFrame, bolus_df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
        """Create LSTM-ready format with the requested columns"""
        # Make copies to avoid modifying original DataFrames
        cgm_df = cgm_df.copy()
        basal_df = basal_df.copy()
        bolus_df = bolus_df.copy()
        
        # Process CGM data
        if not cgm_df.empty and 'created_at' in cgm_df.columns:
            if cgm_df['created_at'].dt.tz is None:
                cgm_df['created_at'] = cgm_df['created_at'].dt.tz_localize('UTC')
            else:
                cgm_df['created_at'] = cgm_df['created_at'].dt.tz_convert('UTC')
            cgm_df = cgm_df.drop_duplicates(subset=['created_at'], keep='last')
            cgm_df.set_index('created_at', inplace=True)
            cgm_df.sort_index(inplace=True)
        
        # Process basal data
        if not basal_df.empty and 'created_at' in basal_df.columns:
            if basal_df['created_at'].dt.tz is None:
                basal_df['created_at'] = basal_df['created_at'].dt.tz_localize('UTC')
            else:
                basal_df['created_at'] = basal_df['created_at'].dt.tz_convert('UTC')
            basal_df = basal_df.drop_duplicates(subset=['created_at'], keep='last')
            basal_df.set_index('created_at', inplace=True)
            basal_df.sort_index(inplace=True)
        
        # Process bolus data
        if not bolus_df.empty and 'created_at' in bolus_df.columns:
            if bolus_df['created_at'].dt.tz is None:
                bolus_df['created_at'] = bolus_df['created_at'].dt.tz_localize('UTC')
            else:
                bolus_df['created_at'] = bolus_df['created_at'].dt.tz_convert('UTC')
            bolus_df = bolus_df.drop_duplicates(subset=['created_at'], keep='last')
            bolus_df.set_index('created_at', inplace=True)
            bolus_df.sort_index(inplace=True)
        
        # Extract individual series
        if not cgm_df.empty:
            cgm_df['bg'] = cgm_df['sgv']
            # Use .first() instead of .ffill() to avoid propagating into gaps
            bg_series = cgm_df['bg'].resample(freq).first()
        else:
            bg_series = pd.Series(dtype=float)
        
        if not basal_df.empty:
            basal_df['basal_rate'] = basal_df['value']
            basal_rate_series = basal_df['basal_rate'].resample(freq).ffill()
        else:
            basal_rate_series = pd.Series(dtype=float)
        
        if not bolus_df.empty:
            bolus_df['bolus_dose'] = bolus_df['bolus']
            bolus_dose_series = bolus_df['bolus_dose'].resample(freq).sum()
        else:
            bolus_dose_series = pd.Series(dtype=float)
        
        # If we have no data, return empty DataFrame
        if bg_series.empty and basal_rate_series.empty and bolus_dose_series.empty:
            return pd.DataFrame()
        
        # Create common time index
        if not bg_series.empty:
            time_index = bg_series.index
        elif not basal_rate_series.empty:
            time_index = basal_rate_series.index
        else:
            time_index = bolus_dose_series.index
        
        # Calculate delta_bg
        delta_bg = bg_series.diff() if not bg_series.empty else pd.Series(dtype=float)
        
        # Align all series to the same index - NO ffill on BG to preserve gaps
        bg_series = bg_series.reindex(time_index)  # Gaps remain NaN
        delta_bg = delta_bg.reindex(time_index)  # Gaps remain NaN
        basal_rate_series = basal_rate_series.reindex(time_index).fillna(0)
        bolus_dose_series = bolus_dose_series.reindex(time_index).fillna(0)
        
        # Time-of-day features
        dt_index = pd.to_datetime(time_index)
        hours = dt_index.hour + dt_index.minute / 60
        time_of_day_sin = np.sin(2 * np.pi * hours / 24)
        time_of_day_cos = np.cos(2 * np.pi * hours / 24)
        
        # Add missing data flag
        bg_missing_flag = bg_series.isna().astype(int)
        
        # Create the final DataFrame with requested columns
        data = {
            'bg': bg_series,
            'delta_bg': delta_bg,
            'basal_rate': basal_rate_series,
            'bolus_dose': bolus_dose_series,
            'time_of_day_sin': time_of_day_sin,
            'time_of_day_cos': time_of_day_cos,
            'bg_missing_flag': bg_missing_flag
        }
        
        df = pd.DataFrame(data)
        
        # Apply BG outlier policy: clip to [20, 600] mg/dL
        BG_MIN = 20
        BG_MAX = 600
        
        if 'bg' in df.columns and not df['bg'].isna().all():
            # Store original values for flagging
            original_bg = df['bg'].copy()
            
            # Clip BG values
            df['bg'] = df['bg'].clip(lower=BG_MIN, upper=BG_MAX)
            
            # ALWAYS create flag column for consistency (1 if clipped, 0 otherwise)
            df['bg_clip_flag'] = ((original_bg < BG_MIN) | (original_bg > BG_MAX)).astype(int)
            
            # Log clipping statistics
            clipped_count = df['bg_clip_flag'].sum()
            if clipped_count > 0:
                logger.info(f"✂️  BG outlier clipping: {clipped_count}/{len(df)} values clipped to [{BG_MIN}, {BG_MAX}]")
            else:
                logger.debug(f"No BG clipping needed - all values within [{BG_MIN}, {BG_MAX}]")
        
        return df.reset_index(drop=True)

    def _get_output_filename(self, pump_serial: str, start_date: str, end_date: str, is_update: bool = False) -> Path:
        """Generate output filename for a date range"""
        pump_dir = self.output_dir / f'pump_{pump_serial}'
        pump_dir.mkdir(exist_ok=True)
        
        if is_update:
            filename = f'update_{start_date}_to_{end_date}.csv'
        else:
            filename = f'{start_date}_to_{end_date}.csv'
        
        return pump_dir / filename
    
    def _save_chunk_data(self, df: pd.DataFrame, output_file: Path):
        """
        Step 6: Save LSTM-Ready CSV - Save chunk data in LSTM-ready format with headers
        
        Saves to bloodBank/raw/pump_{serial}/ with standardized CSV headers
        """
        if df.empty:
            logger.warning(f"No data to save for {output_file}")
            return
        
        # Extract pump serial and date range from the output path
        pump_serial = output_file.parent.name.replace('pump_', '')
        
        # Ensure timestamp column is included and properly formatted
        # The index should be the DatetimeIndex from resampling
        if 'timestamp' not in df.columns and hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Ensure timestamps are UTC
        if 'timestamp' in df.columns:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            elif str(df['timestamp'].dt.tz) != 'UTC':
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Reorder columns to put timestamp first
        cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df[cols]
        
        # Rename time features to match convention
        if 'time_of_day_sin' in df.columns:
            df.rename(columns={'time_of_day_sin': 'sin_time', 'time_of_day_cos': 'cos_time'}, inplace=True)
        
        # Build CSV header
        date_range_str = output_file.stem  # e.g., "2021-01-31_to_2021-03-02"
        
        header_lines = [
            "# bloodBath v2.0 CSV Data File",
            "# ==============================",
            "# data_version: v2",
            f"# generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
            f"# pump_serial: {pump_serial}",
            "# file_role: merged_lstm",
            f"# date_range: {date_range_str}",
            "# tz_handling: stored_in=UTC",
            f"# record_count: {len(df)}",
            f"# columns: {', '.join(df.columns)}",
            "# notes:",
            "#   - BG values clipped to [40, 400] mg/dL",
            "#   - 5-minute resampling applied",
            "#   - bg_clip_flag=1 indicates clipped value",
            "# ==============================",
            ""
        ]
        
        # Add BG clipping stats if available
        if 'bg_clip_flag' in df.columns:
            clipped_count = df['bg_clip_flag'].sum()
            header_lines.insert(-2, f"# bg_clipped_count: {clipped_count}")
        
        header = "\n".join(header_lines) + "\n"
        
        # Save to primary location with header
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(header)
            df.to_csv(f, index=False)
        
        logger.info(f"✅ Saved {len(df)} records to {output_file} with v2.0 header")
        
        # Also save to lstm_ready directory for backward compatibility
        lstm_dir = output_file.parent.parent / "data" / "lstm_ready"
        lstm_dir.mkdir(parents=True, exist_ok=True)
        lstm_file = lstm_dir / f"lstm_ready_{pump_serial}.csv"
        
        # Append to combined file (without header for appending)
        file_exists = lstm_file.exists()
        df.to_csv(lstm_file, mode='a', header=not file_exists, index=False)
        logger.debug(f"Also appended to {lstm_file}")
    
    def _get_last_sync_date(self, pump_serial: str) -> Optional[str]:
        """Get the last successful sync date for a pump"""
        if pump_serial in self.metadata:
            return self.metadata[pump_serial].last_successful_sync
        return None
    
    def _update_sync_metadata(self, pump_serial: str, end_date: str, record_count: int):
        """Update sync metadata after successful sync"""
        if pump_serial not in self.metadata:
            self.metadata[pump_serial] = SyncMetadata(pump_serial=pump_serial)
        
        metadata = self.metadata[pump_serial]
        metadata.last_successful_sync = end_date
        metadata.total_records += record_count
        metadata.last_updated = datetime.now().isoformat()
        
        self._save_metadata()
    
    def sync_pump_historical(self, 
                           pump_config: PumpConfig,
                           update_mode: bool = False,
                           force_full: bool = False) -> bool:
        """
        Sync historical data for a single pump
        
        Args:
            pump_config: Pump configuration
            update_mode: If True, only sync new data since last sync
            force_full: If True, ignore existing data and do full sync
            
        Returns:
            True if sync was successful
        """
        logger.info(f"Starting sync for pump {pump_config.serial}")
        
        start_date = pump_config.start_date
        end_date = pump_config.end_date
        
        # In update mode, start from last successful sync
        if update_mode and not force_full:
            last_sync = self._get_last_sync_date(pump_config.serial)
            if last_sync:
                start_date = last_sync
                logger.info(f"Update mode: starting from {start_date}")
        
        # Generate date chunks
        chunks = self._generate_date_chunks(start_date, end_date, self.chunk_days)
        logger.info(f"Generated {len(chunks)} date chunks for pump {pump_config.serial}")
        
        total_records = 0
        failed_chunks = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                
                # Skip if file already exists (unless force_full)
                output_file = self._get_output_filename(
                    pump_config.serial, 
                    chunk_start, 
                    chunk_end, 
                    is_update=update_mode
                )
                
                if output_file.exists() and not force_full:
                    logger.info(f"File {output_file} already exists, skipping")
                    continue
                
                # Fetch data for this chunk
                df = self._fetch_pump_data_chunk(pump_config, chunk_start, chunk_end)
                
                if not df.empty:
                    # Save chunk data
                    self._save_chunk_data(df, output_file)
                    total_records += len(df)
                
                # Update metadata
                self._update_sync_metadata(pump_config.serial, chunk_end, len(df))
                
                # Small delay between chunks
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_start} to {chunk_end}: {e}")
                failed_chunks.append({'start': chunk_start, 'end': chunk_end, 'error': str(e)})
                
                # Add to failed ranges in metadata
                if pump_config.serial in self.metadata:
                    self.metadata[pump_config.serial].failed_ranges.append({
                        'start': chunk_start,
                        'end': chunk_end,
                        'error': str(e)
                    })
                    self._save_metadata()
        
        # Report results
        logger.info(f"Sync completed for pump {pump_config.serial}")
        logger.info(f"Total records processed: {total_records}")
        if failed_chunks:
            logger.warning(f"Failed chunks: {len(failed_chunks)}")
            for chunk in failed_chunks:
                logger.warning(f"  {chunk['start']} to {chunk['end']}: {chunk['error']}")
        
        return len(failed_chunks) == 0
    
    def sync_multiple_pumps(self, 
                          pump_configs: List[PumpConfig],
                          update_mode: bool = False,
                          force_full: bool = False,
                          parallel: bool = False) -> Dict[str, bool]:
        """
        Sync multiple pumps
        
        Args:
            pump_configs: List of pump configurations
            update_mode: If True, only sync new data since last sync
            force_full: If True, ignore existing data and do full sync
            parallel: If True, sync pumps in parallel
            
        Returns:
            Dictionary mapping pump serial to success status
        """
        results = {}
        
        if parallel:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(len(pump_configs), 3)) as executor:
                future_to_pump = {
                    executor.submit(self.sync_pump_historical, config, update_mode, force_full): config.serial
                    for config in pump_configs
                }
                
                for future in as_completed(future_to_pump):
                    pump_serial = future_to_pump[future]
                    try:
                        result = future.result()
                        results[pump_serial] = result
                    except Exception as e:
                        logger.error(f"Error syncing pump {pump_serial}: {e}")
                        results[pump_serial] = False
        else:
            # Sequential processing
            for config in pump_configs:
                try:
                    result = self.sync_pump_historical(config, update_mode, force_full)
                    results[config.serial] = result
                except Exception as e:
                    logger.error(f"Error syncing pump {config.serial}: {e}")
                    results[config.serial] = False
        
        # Step 7: Merge Multiple Pumps CSVs after all pumps are synced
        logger.info("Merging pump CSVs into combined format...")
        self.merge_pump_csvs()
        
        return results
    
    def generate_lstm_ready_data(self, pump_serial: str, output_file: Optional[str] = None) -> Optional[Path]:
        """
        Generate LSTM-ready dataset from pump data files
        
        Args:
            pump_serial: Pump serial number
            output_file: Output file path (optional)
            
        Returns:
            Path to generated file
        """
        # Look for the actual CSV files from the tconnectsync process
        base_dir = Path(self.output_dir)
        
        # Find the most recent CSV files for each type
        cgm_files = list(base_dir.glob('cgmreading_*.csv'))
        basal_files = list(base_dir.glob('basal_*.csv'))
        bolus_files = list(base_dir.glob('bolus_*.csv'))
        
        if not cgm_files:
            logger.error(f"No CGM data files found in {base_dir}")
            return None
        
        # Use the most recent files
        cgm_file = max(cgm_files, key=lambda f: f.stat().st_mtime)
        basal_file = max(basal_files, key=lambda f: f.stat().st_mtime) if basal_files else None
        bolus_file = max(bolus_files, key=lambda f: f.stat().st_mtime) if bolus_files else None
        
        logger.info(f"Using CGM file: {cgm_file}")
        logger.info(f"Using basal file: {basal_file}")
        logger.info(f"Using bolus file: {bolus_file}")
        
        try:
            # Load and normalize timestamps
            cgm_df = self._load_and_normalize_csv(cgm_file)
            basal_df = self._load_and_normalize_csv(basal_file) if basal_file else pd.DataFrame()
            bolus_df = self._load_and_normalize_csv(bolus_file) if bolus_file else pd.DataFrame()
            
            if cgm_df.empty:
                logger.error("No CGM data available")
                return None
            
            # Generate LSTM-ready dataset
            lstm_df = self._create_lstm_ready_from_csv(cgm_df, basal_df, bolus_df)
            
            # Set output file
            if output_file is None:
                lstm_dir = base_dir / 'data' / 'lstm_ready'
                lstm_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(lstm_dir / f'lstm_ready_{pump_serial}.csv')
            
            # Save the dataset
            lstm_df.to_csv(output_file, index=False)
            
            logger.info(f"Generated LSTM-ready dataset: {output_file}")
            logger.info(f"Total records: {len(lstm_df)}")
            logger.info(f"BG values: {lstm_df['bg'].notna().sum()}/{len(lstm_df)}")
            logger.info(f"Non-zero basal rates: {(lstm_df['basal_rate'] > 0).sum()}")
            logger.info(f"Non-zero bolus doses: {(lstm_df['bolus_dose'] > 0).sum()}")
            
            return Path(output_file)
            
        except Exception as e:
            logger.error(f"Error generating LSTM-ready dataset: {e}")
            return None
    
    def _load_and_normalize_csv(self, csv_file: Path) -> pd.DataFrame:
        """Load CSV and normalize timestamps to UTC"""
        if not csv_file.exists():
            logger.warning(f"File not found: {csv_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(csv_file)
            if 'created_at' not in df.columns:
                logger.warning(f"Column 'created_at' not found in {csv_file}")
                return pd.DataFrame()
            
            # Parse timestamps and convert to UTC
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
            df = df.dropna(subset=['created_at'])
            df = df.sort_values('created_at').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {csv_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            return pd.DataFrame()
    
    def _create_lstm_ready_from_csv(self, cgm_df: pd.DataFrame, basal_df: pd.DataFrame, bolus_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create LSTM-ready dataset from CSV dataframes
        
        Expected columns:
        - CGM: sgv (blood glucose), created_at
        - Basal: value (basal rate), created_at
        - Bolus: bolus (insulin dose), created_at
        
        Output format: [timestamp, bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time]
        """
        # Get time range from CGM data
        start_time = cgm_df['created_at'].min()
        end_time = cgm_df['created_at'].max()
        
        logger.info(f"Time range: {start_time} to {end_time}")
        
        # Create 5-minute time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='5T')
        logger.info(f"Created {len(time_index)} 5-minute intervals")
        
        # Process CGM data (bg values from sgv column)
        cgm_processed = cgm_df[['created_at', 'sgv']].copy()
        cgm_processed = cgm_processed.rename(columns={'sgv': 'bg'})
        cgm_processed = cgm_processed.set_index('created_at')
        
        # Process basal data (basal_rate from value column)
        basal_processed = pd.DataFrame()
        if not basal_df.empty and 'value' in basal_df.columns:
            basal_processed = basal_df[['created_at', 'value']].copy()
            basal_processed = basal_processed.rename(columns={'value': 'basal_rate'})
            basal_processed = basal_processed.set_index('created_at')
        
        # Process bolus data (bolus_dose from bolus column)
        bolus_processed = pd.DataFrame()
        if not bolus_df.empty and 'bolus' in bolus_df.columns:
            bolus_processed = bolus_df[['created_at', 'bolus']].copy()
            bolus_processed = bolus_processed.rename(columns={'bolus': 'bolus_dose'})
            bolus_processed = bolus_processed.set_index('created_at')
        
        # Create result DataFrame
        result_df = pd.DataFrame(index=time_index)
        result_df.index.name = 'timestamp'
        
        # Resample and align data to 5-minute intervals
        logger.info("Resampling data to 5-minute intervals...")
        
        # CGM data - forward fill within reasonable limits (30 minutes)
        if not cgm_processed.empty:
            cgm_resampled = cgm_processed.resample('5T').mean()
            cgm_resampled = cgm_resampled.reindex(time_index, method='ffill', limit=6)  # 6 * 5min = 30min
            result_df['bg'] = cgm_resampled['bg']
        else:
            result_df['bg'] = np.nan
        
        # Basal data - forward fill (basal rates persist until changed)
        if not basal_processed.empty:
            basal_resampled = basal_processed.resample('5T').mean()
            basal_resampled = basal_resampled.reindex(time_index, method='ffill')
            result_df['basal_rate'] = basal_resampled['basal_rate']
        else:
            result_df['basal_rate'] = 0.0
        
        # Bolus data - sum boluses within each 5-minute window
        if not bolus_processed.empty:
            bolus_resampled = bolus_processed.resample('5T').sum()
            bolus_resampled = bolus_resampled.reindex(time_index, fill_value=0.0)
            result_df['bolus_dose'] = bolus_resampled['bolus_dose']
        else:
            result_df['bolus_dose'] = 0.0
        
        # Calculate delta_bg (glucose change)
        result_df['delta_bg'] = result_df['bg'].diff().fillna(0.0)
        
        # Add time-of-day features
        logger.info("Adding time-of-day features...")
        
        # Convert to minutes from midnight
        time_values = pd.to_datetime(result_df.index)
        minutes_from_midnight = (time_values.hour * 60 + time_values.minute)
        
        # Convert to radians (2π = 1440 minutes in a day)
        radians = 2 * np.pi * minutes_from_midnight / 1440
        
        result_df['sin_time'] = np.sin(radians)
        result_df['cos_time'] = np.cos(radians)
        
        # Fill missing values (use forward fill then backward fill)
        result_df['bg'] = result_df['bg'].ffill().bfill()
        result_df['basal_rate'] = result_df['basal_rate'].fillna(0.0)
        result_df['bolus_dose'] = result_df['bolus_dose'].fillna(0.0)
        result_df['delta_bg'] = result_df['delta_bg'].fillna(0.0)
        
        # Reset index to make timestamp a column
        result_df = result_df.reset_index()
        
        # Reorder columns to match desired format
        result_df = result_df[['timestamp', 'bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time']]
        
        return result_df
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status for all pumps"""
        status = {}
        
        for serial, metadata in self.metadata.items():
            pump_dir = self.output_dir / f'pump_{serial}'
            csv_files = list(pump_dir.glob('*.csv')) if pump_dir.exists() else []
            
            status[serial] = {
                'last_successful_sync': metadata.last_successful_sync,
                'total_records': metadata.total_records,
                'last_updated': metadata.last_updated,
                'failed_ranges': len(metadata.failed_ranges),
                'csv_files': len(csv_files)
            }
        
        return status

    def _handle_duplicate_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
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
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps, handling...")
            
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
    
    def merge_pump_csvs(self):
        """
        Step 7: Merge Multiple Pumps CSVs - Combine all pump data into unified format
        """
        try:
            lstm_dir = Path(self.output_dir) / "data" / "lstm_ready"
            
            if not lstm_dir.exists():
                logger.warning("No LSTM-ready data directory found")
                return
            
            # Find all lstm_ready CSV files
            lstm_files = list(lstm_dir.glob("lstm_ready_*.csv"))
            
            if not lstm_files:
                logger.warning("No LSTM-ready CSV files found")
                return
            
            combined_data = []
            
            for file_path in lstm_files:
                try:
                    # Extract pump serial from filename
                    pump_serial = file_path.stem.replace('lstm_ready_', '')
                    
                    # Load the data
                    df = pd.read_csv(file_path)
                    
                    # Add pump_serial column
                    df['pump_serial'] = pump_serial
                    
                    # Ensure timestamp is properly formatted
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
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
            
            # Sort by pump_serial and timestamp
            combined_df = combined_df.sort_values(['pump_serial', 'timestamp'])
            
            # Save combined data
            combined_file = lstm_dir / "lstm_ready_combined.csv"
            combined_df.to_csv(combined_file, index=False)
            
            logger.info(f"Merged {len(combined_df)} records from {len(lstm_files)} pumps to {combined_file}")
            
            # Generate summary statistics
            summary = combined_df.groupby('pump_serial').agg({
                'timestamp': ['min', 'max', 'count'],
                'bg': ['mean', 'std'],
                'basal_rate': ['mean', 'std'],
                'bolus_dose': 'sum'
            }).round(2)
            
            logger.info(f"Combined data summary:\n{summary}")
            
        except Exception as e:
            logger.error(f"Error merging pump CSVs: {e}")
def load_pump_configs(config_file: str) -> List[PumpConfig]:
    """Load pump configurations from JSON file"""
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        configs = []
        for pump_data in data.get('pumps', []):
            config = PumpConfig(**pump_data)
            configs.append(config)
        
        return configs
    except Exception as e:
        logger.error(f"Error loading pump configs from {config_file}: {e}")
        return []


def create_default_pump_configs() -> List[PumpConfig]:
    """Create default pump configurations"""
    return [
        PumpConfig(
            serial='881235',
            start_date='2021-01-01T00:00:00',
            end_date='2024-10-06T23:35:00',
            description='Pump 1 - Historical data'
        ),
        PumpConfig(
            serial='901161470',
            start_date='2024-01-01T00:00:00',
            end_date='2025-07-14T17:33:46',
            description='Pump 2 - Current/recent data'
        )
    ]


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Tandem t:connect Historical Data Sync Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sync of all pumps
  %(prog)s --full
  
  # Update mode (incremental sync)
  %(prog)s --update
  
  # Sync specific pump
  %(prog)s --pump 881235 --start 2023-01-01 --end 2023-12-31
  
  # Use custom chunk size
  %(prog)s --update --chunk-days 7
  
  # Generate LSTM-ready data
  %(prog)s --generate-lstm 881235
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full', action='store_true',
                          help='Full historical sync (overwrites existing)')
    mode_group.add_argument('--update', action='store_true',
                          help='Incremental sync (only new data)')
    mode_group.add_argument('--status', action='store_true',
                          help='Show sync status for all pumps')
    mode_group.add_argument('--generate-lstm', metavar='SERIAL',
                          help='Generate LSTM-ready dataset for pump')
    
    # Pump configuration
    parser.add_argument('--pump', metavar='SERIAL',
                      help='Sync specific pump serial number')
    parser.add_argument('--start', metavar='DATE',
                      help='Start date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--end', metavar='DATE',
                      help='End date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--config', metavar='FILE',
                      help='JSON file with pump configurations')
    
    # Sync options
    parser.add_argument('--chunk-days', type=int, default=30,
                      help='Date chunk size in days (default: 30)')
    parser.add_argument('--max-retries', type=int, default=5,
                      help='Maximum retry attempts (default: 5)')
    parser.add_argument('--output-dir', default='./bloodBath/bloodBank/raw',
                      help='Output directory (default: ./bloodBath/bloodBank/raw)')
    parser.add_argument('--parallel', action='store_true',
                      help='Sync multiple pumps in parallel')
    
    # API options
    parser.add_argument('--region', choices=['US', 'EU'], default='US',
                      help='Tandem region (default: US)')
    parser.add_argument('--rate-limit-delay', type=int, default=1,
                      help='Delay between API calls in seconds (default: 1)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Quiet mode (errors only)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Initialize client
    client = TandemHistoricalSyncClient(
        region=args.region,
        output_dir=args.output_dir,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries,
        rate_limit_delay=args.rate_limit_delay
    )
    
    # Handle status mode
    if args.status:
        status = client.get_sync_status()
        print(json.dumps(status, indent=2))
        return
    
    # Handle LSTM generation mode
    if args.generate_lstm:
        result = client.generate_lstm_ready_data(args.generate_lstm)
        if result:
            print(f"LSTM-ready dataset generated: {result}")
        else:
            print("Failed to generate LSTM-ready dataset")
            sys.exit(1)
        return
    
    # Load pump configurations
    pump_configs = []
    
    if args.config:
        # Load from config file
        pump_configs = load_pump_configs(args.config)
    elif args.pump:
        # Single pump from command line
        if not args.start or not args.end:
            print("Error: --start and --end required when using --pump")
            sys.exit(1)
        
        pump_configs = [PumpConfig(
            serial=args.pump,
            start_date=args.start,
            end_date=args.end
        )]
    else:
        # Use default configurations, but override dates if provided
        pump_configs = create_default_pump_configs()
        
        # Override start/end dates if provided
        if args.start or args.end:
            for config in pump_configs:
                if args.start:
                    config.start_date = args.start
                if args.end:
                    config.end_date = args.end
    
    if not pump_configs:
        print("Error: No pump configurations found")
        sys.exit(1)
    
    # Perform sync
    logger.info(f"Starting sync for {len(pump_configs)} pump(s)")
    
    try:
        results = client.sync_multiple_pumps(
            pump_configs=pump_configs,
            update_mode=args.update,
            force_full=args.full,
            parallel=args.parallel
        )
        
        # Report results
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"\nSync completed: {success_count}/{total_count} pumps successful")
        
        for serial, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Pump {serial}: {status}")
        
        if success_count < total_count:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
