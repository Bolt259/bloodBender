"""
Production Data Harvest Manager for bloodBath

This module provides comprehensive data synchronization and harvest capabilities,
integrating the MonthlyLSTMGenerator functionality into the bloodBath framework.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import random

from ..core.client import TandemHistoricalSyncClient
from ..core.config import PumpConfig
from ..data.processors import UnifiedDataProcessor
from ..data.validators import LstmDataValidator
from ..utils.logging_utils import setup_logger


class HarvestManager:
    """
    Comprehensive production data harvest and synchronization manager
    
    Features:
    - Automatic data range detection for pumps
    - Incremental monthly file generation with overlap
    - Quality validation and outlier handling
    - Retry logic with authentication refresh
    - Comprehensive analytics and reporting
    """
    
    def __init__(self, 
                 output_dir: str = "/home/bolt/projects/bb/training_data",
                 chunk_days: int = 15,
                 enable_validation: bool = True):
        """
        Initialize harvest manager
        
        Args:
            output_dir: Base directory for training data
            chunk_days: API chunk size for data retrieval (15 days optimal)
            enable_validation: Whether to enable quality validation
        """
        self.output_dir = Path(output_dir)
        self.chunk_days = chunk_days
        self.enable_validation = enable_validation
        
        # Physiological limits for outlier handling
        self.BASAL_HARD_MAX = 8.0   # units/hour (extreme but possible)
        self.BASAL_HARD_MIN = 0.0
        self.BG_HARD_MAX = 401      # mg/dL
        self.BG_HARD_MIN = 39       # mg/dL
        
        # Setup logging
        log_file = self.output_dir / "logs" / f"harvest_manager_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('harvest_manager', log_file=log_file)
        
        # Initialize client and processors
        self.client = TandemHistoricalSyncClient()
        self.processor = UnifiedDataProcessor()
        self.validator = LstmDataValidator() if enable_validation else None
        
        # Retry configuration for API calls
        self.retry_config = {
            'max_retries': 5,
            'base_delay': 30,  # Start with 30 second delay
            'max_delay': 300,  # Max 5 minutes between retries
            'backoff_multiplier': 2.0,
            'jitter_range': 0.1  # ±10% random jitter
        }
        
        # Track API authentication state
        self.auth_failures = 0
        self.last_successful_auth = None
        
        # Track generation statistics
        self.stats = {
            'files_generated': 0,
            'total_records': 0,
            'quality_scores': [],
            'failed_months': [],
            'api_retries': 0,
            'auth_refreshes': 0
        }
    
    def detect_pump_data_range(self, pump_serial: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Detect actual data availability for a pump by testing different date ranges
        
        Args:
            pump_serial: Pump serial number
            
        Returns:
            Tuple of (earliest_date, latest_date) or (None, None) if detection fails
        """
        self.logger.info(f"🔍 Detecting data availability for pump {pump_serial}...")
        
        try:
            # Test connection with retry logic
            try:
                connection_ok = self._api_call_with_retry(self.client.test_connection)
                if not connection_ok:
                    self.logger.error("API connection failed after all retries")
                    return None, None
            except Exception as e:
                self.logger.error(f"API connection failed after all retries: {e}")
                return None, None
            
            # Test recent data (should always exist)
            recent_end = datetime.now()
            recent_start = recent_end - timedelta(days=30)
            
            self.logger.info(f"   Testing recent data: {recent_start.date()} to {recent_end.date()}")
            
            test_config = PumpConfig(
                serial=pump_serial,
                start_date=recent_start.strftime('%Y-%m-%d'),
                end_date=recent_end.strftime('%Y-%m-%d')
            )
            
            if not self.client.sync_pump_historical(test_config):
                self.logger.error(f"   No recent data found for pump {pump_serial}")
                return None, None
            
            # Find earliest available data by testing progressively older dates
            latest_date = recent_end
            earliest_date = None
            valid_years = []  # Track all years with valid data
            
            # Test years going back from current
            test_years = [2024, 2023, 2022, 2021, 2020, 2019]
            
            for year in test_years:
                test_start = datetime(year, 1, 1)
                test_end = datetime(year, 1, 31)  # Just test January of each year
                
                self.logger.info(f"   Testing {year}: {test_start.date()} to {test_end.date()}")
                
                if self._test_date_range(pump_serial, test_start, test_end):
                    # Found data in this year - add to valid years
                    valid_years.append(year)
                    self.logger.info(f"   ✅ Found data in {year}")
                else:
                    self.logger.info(f"   ❌ No data in {year}")
            
            # Set earliest_date to the OLDEST year that has valid data
            if valid_years:
                oldest_valid_year = min(valid_years)  # Get the earliest (smallest) year
                earliest_date = datetime(oldest_valid_year, 1, 1)
                self.logger.info(f"   📅 Valid data years: {valid_years}")
                self.logger.info(f"   📅 Using earliest year: {oldest_valid_year}")
            
            if earliest_date:
                self.logger.info(f"🎯 Pump {pump_serial} data range: {earliest_date.date()} to {latest_date.date()}")
                return earliest_date, latest_date
            else:
                self.logger.warning(f"Could not determine data range for pump {pump_serial}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Data range detection failed: {e}")
            return None, None
    
    def _test_date_range(self, pump_serial: str, start_date: datetime, end_date: datetime) -> bool:
        """Test if data exists in a specific date range with retry logic"""
        return self._api_call_with_retry(
            self._test_date_range_impl, 
            pump_serial, 
            start_date, 
            end_date
        )
    
    def _test_date_range_impl(self, pump_serial: str, start_date: datetime, end_date: datetime) -> bool:
        """Implementation of date range test"""
        test_config = PumpConfig(
            serial=pump_serial,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Check for actual data by looking at generated LSTM files
        sweetblood_dir = Path(self.client.output_dir)
        lstm_dir = sweetblood_dir / "lstm_pump_data"
        
        # Count existing files before sync
        existing_files = set(lstm_dir.glob(f"pump_{pump_serial}_*.csv"))
        
        # Perform the sync
        result = self.client.sync_pump_historical(test_config)
        if not result:
            return False
        
        # Check if new files were created with real data
        new_files = set(lstm_dir.glob(f"pump_{pump_serial}_*.csv")) - existing_files
        
        if not new_files:
            return False  # No new files created
            
        # Check the newest file for real data
        latest_file = max(new_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file, comment='#')
            
            # For date range detection, we're more lenient about data volume
            if len(df) == 0:  # No data at all
                return False
                
            if 'bg' in df.columns:
                bg_values = df['bg'].dropna()
                if len(bg_values) > 0:
                    # Check for COMPLETELY fake data patterns
                    unique_values = bg_values.nunique()
                    
                    # If there's any variety in BG values, consider it potentially real
                    if unique_values > 1:
                        # Check if we have at least some realistic BG values
                        realistic_bg = ((bg_values >= 50) & (bg_values <= 400)).sum()
                        if realistic_bg >= 5:  # At least 5 real BG values makes it worth including
                            return True
                    
                    # Only reject if ALL values are identical fake patterns
                    if unique_values <= 1 or bg_values.isna().all():
                        return False  # Completely fake or missing data
                        
            return True  # Default to accepting data during range detection
        except:
            return False
    
    def detect_existing_data(self, pump_serial: str) -> Dict[str, bool]:
        """
        Detect what monthly files already exist and are valid
        
        Args:
            pump_serial: Pump serial number
            
        Returns:
            Dict mapping month_labels to validity (True = good, False = needs regeneration)
        """
        self.logger.info(f"🔍 Checking existing monthly files for pump {pump_serial}...")
        
        monthly_dir = self.output_dir / "monthly_lstm" / f"pump_{pump_serial}"
        existing_files = {}
        
        if monthly_dir.exists():
            csv_files = list(monthly_dir.glob(f"pump_{pump_serial}_*.csv"))
            self.logger.info(f"   Found {len(csv_files)} existing files to validate")
            
            for csv_file in csv_files:
                try:
                    # Extract month label from filename: pump_881235_2022_01.csv -> 2022_01
                    filename_parts = csv_file.stem.split('_')
                    if len(filename_parts) >= 4:
                        month_label = f"{filename_parts[2]}_{filename_parts[3]}"
                        
                        # Validate existing file
                        is_valid = self._validate_existing_file(csv_file)
                        existing_files[month_label] = is_valid
                        
                        status = "✅ Valid" if is_valid else "❌ Invalid"
                        file_size = csv_file.stat().st_size / 1024  # KB
                        self.logger.info(f"   {month_label}: {status} ({file_size:.1f}KB)")
                        
                except Exception as e:
                    self.logger.warning(f"   Error parsing {csv_file.name}: {e}")
        else:
            self.logger.info(f"   No existing directory found at {monthly_dir}")
        
        valid_count = len([v for v in existing_files.values() if v])
        invalid_count = len([v for v in existing_files.values() if not v])
        self.logger.info(f"   📊 Summary: {valid_count} valid, {invalid_count} invalid files")
        
        return existing_files
    
    def _validate_existing_file(self, file_path: Path) -> bool:
        """
        Validate that existing file has good data
        
        Args:
            file_path: Path to CSV file to validate
            
        Returns:
            True if file is valid and doesn't need regeneration
        """
        try:
            # Basic file checks
            if not file_path.exists() or file_path.stat().st_size < 1000:  # Less than 1KB
                return False
            
            # Load and validate CSV data
            df = pd.read_csv(file_path, comment='#')
            
            # Must have reasonable amount of data
            if len(df) < 100:  
                return False
            
            # Check required columns
            required_cols = ['timestamp', 'bg', 'basal_rate']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Validate BG data quality
            if 'bg' in df.columns:
                bg_values = df['bg'].dropna()
                if len(bg_values) == 0:
                    return False
                
                # Check for fake data patterns
                unique_bg_count = bg_values.nunique()
                if unique_bg_count <= 1:  # All same value
                    return False
                    
                if bg_values.isna().all():  # All missing values
                    return False
                
                # Check for realistic BG range
                realistic_count = ((bg_values >= 50) & (bg_values <= 400)).sum()
                if realistic_count / len(bg_values) < 0.8:  # Less than 80% realistic
                    return False
            
            # Validate timestamp coverage (should span close to 30 days)
            if 'timestamp' in df.columns:
                try:
                    timestamps = pd.to_datetime(df['timestamp'])
                    time_span = (timestamps.max() - timestamps.min()).days
                    if time_span < 20:  # Should cover at least 20 days for a monthly file
                        return False
                except:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    def _api_call_with_retry(self, func, *args, **kwargs):
        """
        Execute API call with exponential backoff retry logic
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises exception after max retries
        """
        last_exception = None
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                # Attempt the API call
                result = func(*args, **kwargs)
                
                # Success - reset auth failure counter
                if self.auth_failures > 0:
                    self.logger.info(f"✅ API call recovered after {self.auth_failures} auth failures")
                    self.auth_failures = 0
                    self.last_successful_auth = datetime.now()
                
                return result
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if this looks like an authentication/token issue
                is_auth_error = any(keyword in error_str for keyword in [
                    'unauthorized', 'token', 'authentication', 'login', 
                    'expired', 'invalid', 'forbidden', '401', '403'
                ])
                
                if is_auth_error:
                    self.auth_failures += 1
                    self.logger.warning(f"🔐 Authentication error detected (attempt {attempt + 1}/{self.retry_config['max_retries']}): {e}")
                    
                    # Try to refresh authentication
                    if self._refresh_authentication():
                        self.logger.info("🔄 Authentication refreshed, retrying...")
                        continue
                else:
                    self.logger.warning(f"⚠️ API error (attempt {attempt + 1}/{self.retry_config['max_retries']}): {e}")
                
                # Don't retry on the last attempt
                if attempt == self.retry_config['max_retries'] - 1:
                    break
                
                # Calculate retry delay with exponential backoff and jitter
                delay = min(
                    self.retry_config['base_delay'] * (self.retry_config['backoff_multiplier'] ** attempt),
                    self.retry_config['max_delay']
                )
                
                # Add random jitter to avoid thundering herd
                jitter = delay * self.retry_config['jitter_range'] * (2 * random.random() - 1)
                actual_delay = delay + jitter
                
                self.stats['api_retries'] += 1
                self.logger.info(f"⏳ Waiting {actual_delay:.1f}s before retry {attempt + 2}/{self.retry_config['max_retries']}")
                time.sleep(actual_delay)
        
        # All retries exhausted
        if last_exception:
            self.logger.error(f"❌ API call failed after {self.retry_config['max_retries']} attempts: {last_exception}")
            raise last_exception
        else:
            error_msg = f"API call failed after {self.retry_config['max_retries']} attempts with no exception details"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _refresh_authentication(self) -> bool:
        """
        Attempt to refresh API authentication
        
        Returns:
            True if authentication was refreshed, False otherwise
        """
        try:
            self.logger.info("🔄 Attempting to refresh API authentication...")
            
            # Create a new client instance to force re-authentication
            old_client = self.client
            self.client = TandemHistoricalSyncClient()
            
            # Test the connection to ensure authentication worked
            if self.client.test_connection():
                self.logger.info("✅ Authentication refresh successful")
                self.stats['auth_refreshes'] += 1
                self.last_successful_auth = datetime.now()
                return True
            else:
                self.logger.warning("❌ Authentication refresh failed - connection test failed")
                self.client = old_client  # Restore old client
                return False
                
        except Exception as e:
            self.logger.warning(f"❌ Authentication refresh failed: {e}")
            return False
    
    def apply_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply physiological limits and outlier flagging
        
        Args:
            df: Raw LSTM DataFrame
            
        Returns:
            DataFrame with outlier handling applied
        """
        self.logger.info("🔧 Applying outlier handling and physiological limits")
        
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Track original values for flagging
        processed_df['bg_original'] = processed_df['bg']
        processed_df['basal_original'] = processed_df['basal_rate']
        
        # Apply hard physiological limits
        bg_clipped = np.clip(processed_df['bg'], self.BG_HARD_MIN, self.BG_HARD_MAX)
        basal_clipped = np.clip(processed_df['basal_rate'], self.BASAL_HARD_MIN, self.BASAL_HARD_MAX)
        
        # Create anomaly flags
        processed_df['bg_anomaly_flag'] = (processed_df['bg'] != bg_clipped).astype(int)
        processed_df['basal_anomaly_flag'] = (processed_df['basal_rate'] != basal_clipped).astype(int)
        
        # Apply clipping
        processed_df['bg'] = bg_clipped
        processed_df['basal_rate'] = basal_clipped
        
        # Statistical outlier detection (Z-score method)
        if len(processed_df) > 1:  # Need at least 2 points for std calculation
            basal_mean = processed_df['basal_rate'].mean()
            basal_std = processed_df['basal_rate'].std()
            if basal_std > 0:  # Avoid division by zero
                basal_z_scores = np.abs((processed_df['basal_rate'] - basal_mean) / basal_std)
                processed_df['basal_statistical_outlier'] = (basal_z_scores > 3).astype(int)
            else:
                processed_df['basal_statistical_outlier'] = 0
        else:
            processed_df['basal_statistical_outlier'] = 0
        
        # Log outlier statistics
        bg_anomalies = processed_df['bg_anomaly_flag'].sum()
        basal_anomalies = processed_df['basal_anomaly_flag'].sum()
        statistical_outliers = processed_df['basal_statistical_outlier'].sum()
        
        self.logger.info(f"   BG hard limit violations: {bg_anomalies}")
        self.logger.info(f"   Basal hard limit violations: {basal_anomalies}")
        self.logger.info(f"   Basal statistical outliers (Z>3): {statistical_outliers}")
        
        return processed_df
    
    def generate_monthly_date_ranges(self, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   overlap_days: int = 15) -> List[Tuple[datetime, datetime, str]]:
        """
        Generate monthly date ranges with overlap
        
        Args:
            start_date: Start of data collection
            end_date: End of data collection  
            overlap_days: Days of overlap between months
            
        Returns:
            List of (start_date, end_date, month_label) tuples
        """
        ranges = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate month end (30 days from start)
            month_end = min(current_start + timedelta(days=30), end_date)
            
            # Create month label
            month_label = current_start.strftime("%Y_%m")
            
            ranges.append((current_start, month_end, month_label))
            
            # Next month starts 15 days later (50% overlap)
            current_start = current_start + timedelta(days=30 - overlap_days)
        
        return ranges
    
    def sync_pump_data(self, 
                      pump_serial: str,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Main sync operation for a pump - detects ranges and generates missing data
        
        Args:
            pump_serial: Pump serial number
            start_date: Optional start date (auto-detected if None)
            end_date: Optional end date (auto-detected if None) 
            force_regenerate: If True, regenerate all files even if they exist
            
        Returns:
            Dictionary with sync results and statistics
        """
        self.logger.info(f"\n🚀 STARTING SYNC FOR PUMP {pump_serial}")
        self.logger.info("="*60)
        
        try:
            # Auto-detect date range if not provided
            if start_date is None or end_date is None:
                detected_start, detected_end = self.detect_pump_data_range(pump_serial)
                if detected_start is None or detected_end is None:
                    self.logger.error(f"Could not detect data range for pump {pump_serial}")
                    return {'success': False, 'error': 'Data range detection failed'}
                
                start_date = start_date or detected_start
                end_date = end_date or detected_end
            
            self.logger.info(f"📅 Sync range: {start_date.date()} to {end_date.date()}")
            
            # Check existing data unless forcing regeneration
            existing_data = {} if force_regenerate else self.detect_existing_data(pump_serial)
            
            # Generate monthly ranges
            monthly_ranges = self.generate_monthly_date_ranges(start_date, end_date)
            
            # Filter ranges to process
            ranges_to_process = []
            skipped_count = 0
            
            for start, end, month_label in monthly_ranges:
                if not force_regenerate and month_label in existing_data and existing_data[month_label]:
                    skipped_count += 1
                else:
                    ranges_to_process.append((start, end, month_label))
            
            self.logger.info(f"📊 Processing plan: {len(ranges_to_process)} new, {skipped_count} existing")
            
            # Initialize sync stats
            sync_stats = {
                'pump_serial': pump_serial,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_months': len(monthly_ranges),
                'skipped_files': skipped_count,
                'successful_files': 0,
                'failed_files': 0,
                'total_records': 0,
                'monthly_files': [],
                'success': True
            }
            
            # Process needed ranges
            for i, (month_start, month_end, month_label) in enumerate(ranges_to_process, 1):
                self.logger.info(f"\n📅 Processing {i}/{len(ranges_to_process)}: {month_label}")
                
                month_stats = self.generate_monthly_file(pump_serial, month_start, month_end, month_label)
                
                if month_stats:
                    sync_stats['successful_files'] += 1
                    sync_stats['total_records'] += month_stats['records']
                    sync_stats['monthly_files'].append(month_stats)
                else:
                    sync_stats['failed_files'] += 1
            
            # Final validation pass if enabled
            if self.enable_validation and self.validator:
                self.logger.info("\n🔍 Running validation on all monthly files...")
                self._validate_all_monthly_files(pump_serial, sync_stats)
            
            # Log final summary
            total_valid = sync_stats['skipped_files'] + sync_stats['successful_files']
            self.logger.info(f"\n✅ SYNC COMPLETE FOR PUMP {pump_serial}")
            self.logger.info(f"   Valid files: {total_valid}/{sync_stats['total_months']}")
            self.logger.info(f"   Records synced: {sync_stats['total_records']:,}")
            
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"❌ Sync failed for pump {pump_serial}: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_monthly_file(self, 
                            pump_serial: str,
                            start_date: datetime, 
                            end_date: datetime,
                            month_label: str) -> Optional[Dict]:
        """
        Generate single monthly LSTM file using the integrated pipeline
        
        Args:
            pump_serial: Pump serial number
            start_date: Month start date
            end_date: Month end date
            month_label: Month identifier (YYYY_MM)
            
        Returns:
            Dictionary with generation statistics or None if failed
        """
        self.logger.info(f"📅 Generating: pump_{pump_serial}_{month_label}")
        
        try:
            # Use TandemHistoricalSyncClient to pull raw data
            pump_config = PumpConfig(
                serial=pump_serial,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Sync with retry logic
            success = self._api_call_with_retry(self.client.sync_pump_historical, pump_config)
            if not success:
                self.logger.warning(f"   Data sync failed for {month_label}")
                return None
            
            # Find the generated files in the sweetBlood directory
            sweetblood_dir = Path(self.client.output_dir) 
            
            # Look for CGM, basal, and bolus CSV files
            data_sources = {
                'cgm': list(sweetblood_dir.glob(f"**/cgmreading_*.csv")),
                'basal': list(sweetblood_dir.glob(f"**/basal_*.csv")),
                'bolus': list(sweetblood_dir.glob(f"**/bolus_*.csv"))
            }
            
            # Use the most recent files
            source_files = {}
            for data_type, files in data_sources.items():
                if files:
                    source_files[data_type] = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"   Using {data_type}: {source_files[data_type].name}")
            
            if not source_files:
                self.logger.warning(f"   No source files found for {month_label}")
                return None
            
            # Process through the unified pipeline
            try:
                # Load and combine the source data
                combined_data = self._load_and_combine_sources(source_files)
                if combined_data is None or len(combined_data) == 0:
                    self.logger.warning(f"   No valid data after combining sources for {month_label}")
                    return None
                
                # Process through unified pipeline
                processed_data = self.processor.process_lstm_data(
                    combined_data,
                    pump_serial=pump_serial,
                    max_gap_hours=400,  # Lenient for monthly processing
                    min_segment_length=12
                )
                
                if processed_data is None or len(processed_data) == 0:
                    self.logger.warning(f"   Processing failed for {month_label}")
                    return None
                
                # Apply outlier handling
                processed_data = self.apply_outlier_handling(processed_data)
                
                # Validate if enabled
                if self.enable_validation and self.validator:
                    validation_result = self.validator.validate_dataframe(processed_data)
                    if not validation_result.is_valid:
                        self.logger.warning(f"   Validation failed for {month_label}: {validation_result.issues}")
                        return None
                
                # Save the monthly file
                output_file = (self.output_dir / "monthly_lstm" / f"pump_{pump_serial}" / 
                              f"pump_{pump_serial}_{month_label}.csv")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Add header with metadata
                header_lines = [
                    "# bloodBath Monthly LSTM Dataset",
                    f"# Pump: {pump_serial}",
                    f"# Month: {month_label}", 
                    f"# Date Range: {start_date.date()} to {end_date.date()}",
                    f"# Records: {len(processed_data)}",
                    f"# Generated: {datetime.now().isoformat()}",
                    f"# Pipeline: Unified Processing + Outlier Handling",
                    ""
                ]
                
                with open(output_file, 'w') as f:
                    f.write('\n'.join(header_lines))
                    processed_data.to_csv(f, index=False)
                
                self.logger.info(f"   ✅ Saved {len(processed_data)} records to {output_file}")
                
                # Return statistics
                return {
                    'month': month_label,
                    'records': len(processed_data),
                    'date_range': f"{start_date.date()} to {end_date.date()}",
                    'file_path': str(output_file),
                    'quality_score': 0.9  # Default good score for processed data
                }
                
            except Exception as e:
                self.logger.error(f"   Processing error for {month_label}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"   Generation failed for {month_label}: {e}")
            return None
    
    def _load_and_combine_sources(self, source_files: Dict[str, Path]) -> Optional[List[Dict]]:
        """Load and combine CGM, basal, and bolus data sources"""
        try:
            combined_data = []
            
            # Load each data type
            for data_type, file_path in source_files.items():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Convert to list of dicts for unified processor
                    records = df.to_dict('records')
                    
                    # Add data type identifier
                    for record in records:
                        record['data_type'] = data_type
                    
                    combined_data.extend(records)
                    self.logger.info(f"   Loaded {len(records)} {data_type} records")
                    
                except Exception as e:
                    self.logger.warning(f"   Failed to load {data_type} data: {e}")
            
            return combined_data if combined_data else None
            
        except Exception as e:
            self.logger.error(f"Data combination failed: {e}")
            return None
    
    def _validate_all_monthly_files(self, pump_serial: str, sync_stats: Dict):
        """Run validation on all monthly files for a pump"""
        try:
            monthly_dir = self.output_dir / "monthly_lstm" / f"pump_{pump_serial}"
            if not monthly_dir.exists():
                return
            
            csv_files = list(monthly_dir.glob(f"pump_{pump_serial}_*.csv"))
            validation_results = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, comment='#')
                    result = self.validator.validate_dataframe(df)
                    
                    validation_results.append({
                        'file': csv_file.name,
                        'valid': result.is_valid,
                        'confidence': result.confidence_score,
                        'issues': result.issues
                    })
                    
                    status = "✅" if result.is_valid else "❌"
                    self.logger.info(f"   {status} {csv_file.name}: {result.confidence_score:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"   Validation error for {csv_file.name}: {e}")
            
            # Add validation summary to stats
            sync_stats['validation_results'] = validation_results
            valid_files = len([r for r in validation_results if r['valid']])
            sync_stats['validation_summary'] = {
                'files_validated': len(validation_results),
                'files_passed': valid_files,
                'validation_rate': valid_files / len(validation_results) if validation_results else 0
            }
            
        except Exception as e:
            self.logger.error(f"Validation pass failed: {e}")
    
    def sync_all_pumps(self, pump_serials: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Sync data for multiple pumps
        
        Args:
            pump_serials: List of pump serial numbers (auto-detect if None)
            
        Returns:
            Comprehensive sync results for all pumps
        """
        if pump_serials is None:
            pump_serials = ["881235", "901161470"]  # Default pumps
        
        self.logger.info(f"\n🚀 STARTING MULTI-PUMP SYNC")
        self.logger.info(f"Pumps: {pump_serials}")
        self.logger.info("="*60)
        
        all_results = {
            'sync_timestamp': datetime.now().isoformat(),
            'pumps_processed': pump_serials,
            'pump_results': {},
            'overall_success': True,
            'total_files_generated': 0,
            'total_records': 0
        }
        
        for pump_serial in pump_serials:
            self.logger.info(f"\n📡 Syncing pump {pump_serial}...")
            
            pump_result = self.sync_pump_data(pump_serial)
            all_results['pump_results'][pump_serial] = pump_result
            
            if pump_result.get('success', False):
                all_results['total_files_generated'] += pump_result.get('successful_files', 0)
                all_results['total_records'] += pump_result.get('total_records', 0)
            else:
                all_results['overall_success'] = False
        
        # Save comprehensive analytics
        self._save_sync_analytics(all_results)
        
        return all_results
    
    def _save_sync_analytics(self, results: Dict[str, Any]):
        """Save comprehensive sync analytics"""
        try:
            analytics_dir = self.output_dir / "analytics"
            analytics_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analytics_file = analytics_dir / f"sync_analytics_{timestamp}.json"
            
            with open(analytics_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"📊 Analytics saved: {analytics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analytics: {e}")