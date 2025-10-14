"""
Main TandemHistoricalSyncClient for orchestrating pump data synchronization
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from .config import PumpConfig, SyncMetadata, load_metadata, save_metadata
from .exceptions import TandemSyncError, ChunkSizeError, RateLimitError
from ..api import TandemConnector, TandemDataFetcher
from ..data import EventExtractor, DataProcessor, DataValidator
from ..utils.file_utils import (
    save_lstm_ready_data, get_output_filename, get_lstm_output_path,
    load_csv_with_datetime, find_most_recent_files, merge_csv_files
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TandemHistoricalSyncClient:
    """
    Main client for historical data synchronization and processing
    """
    
    # Target event types we want to collect
    TARGET_EVENTS = ['BASAL', 'BOLUS', 'CGM_READING']
    
    def __init__(self, 
                 email: Optional[str] = None,
                 password: Optional[str] = None,
                 region: str = 'US',
                 output_dir: Optional[str] = None,  # Will use internal sweetBlood path
                 chunk_days: int = 30,
                 max_retries: int = 5,
                 rate_limit_delay: int = 1,
                 enable_data_validation: bool = True):
        """
        Initialize the sync client
        
        Args:
            email: t:connect email (defaults to environment variables)
            password: t:connect password (defaults to environment variables)
            region: Server region (US or EU)
            output_dir: Directory for output files
            chunk_days: Size of date chunks in days
            max_retries: Maximum retry attempts
            rate_limit_delay: Base delay between requests
            enable_data_validation: Enable CSV validation to detect synthetic data
        """
        # Setup directory structure using bloodBank v2.0 architecture
        from ..utils.structure_utils import setup_bloodbank_environment
        
        # Use bloodBank structure (output_dir can override for testing)
        self.structure = setup_bloodbank_environment(output_dir)
        self.output_dir = self.structure['base']
        
        # Store validation setting
        self.enable_data_validation = enable_data_validation
        
        # Get credentials from environment if not provided
        from .config import get_credentials
        creds = get_credentials(email, password, region)
        
        # Initialize components
        self.connector = TandemConnector(creds.email, creds.password, creds.region)
        self.fetcher = TandemDataFetcher(
            self.connector, 
            chunk_days=chunk_days, 
            max_retries=max_retries,
            rate_limit_delay=rate_limit_delay
        )
        self.extractor = EventExtractor()
        self.processor = DataProcessor()
        self.validator = DataValidator()
        
        # Metadata management using structured directory
        from ..utils.structure_utils import get_metadata_file
        self.metadata_file = get_metadata_file(self.structure, 'sync_metadata.json')
        self.metadata = load_metadata(self.metadata_file)
        
        logger.info(f"Initialized TandemHistoricalSyncClient")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Chunk size: {chunk_days} days")
    
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
        
        try:
            # Fetch raw events
            raw_events = self.fetcher.fetch_pump_data_range(
                pump_config, start_date, end_date
            )
            
            if not raw_events:
                logger.warning(f"No events fetched for pump {pump_config.serial}")
                return True  # Not an error, just no data
            
            # Process events into LSTM-ready format
            lstm_df = self._process_events_to_lstm(raw_events)
            
            if lstm_df.empty:
                logger.warning(f"No valid data after processing for pump {pump_config.serial}")
                return True
            
            # Save data using structured directories
            from ..utils.file_utils import save_structured_lstm_data, save_structured_metadata
            
            output_file = save_structured_lstm_data(
                self.structure,
                lstm_df,
                pump_config.serial,
                start_date,
                end_date,
                validate_data=self.enable_data_validation  # ✅ Use client setting
            )
            
            if output_file:
                # Save additional metadata
                metadata = {
                    'pump_serial': pump_config.serial,
                    'start_date': start_date,
                    'end_date': end_date,
                    'records_count': len(lstm_df),
                    'sync_timestamp': pd.Timestamp.now().isoformat(),
                    'update_mode': update_mode,
                    'force_full': force_full,
                    'output_file': str(output_file)
                }
                
                save_structured_metadata(
                    self.structure,
                    metadata,
                    f'sync_{pump_config.serial}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                
            # Update sync metadata
            self._update_sync_metadata(pump_config.serial, end_date, len(lstm_df))
            
            logger.info(f"Successfully synced pump {pump_config.serial}: {len(lstm_df)} records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync pump {pump_config.serial}: {e}")
            
            # Add to failed ranges in metadata
            if pump_config.serial not in self.metadata:
                self.metadata[pump_config.serial] = SyncMetadata(pump_serial=pump_config.serial)
            
            metadata = self.metadata[pump_config.serial]
            # Ensure failed_ranges is initialized as a list
            if metadata.failed_ranges is None:
                metadata.failed_ranges = []
            
            metadata.failed_ranges.append({
                'start': start_date,
                'end': end_date,
                'error': str(e)
            })
            save_metadata(self.metadata, self.metadata_file)
            
            return False
    
    def _process_events_to_lstm(self, raw_events: List[Any]) -> pd.DataFrame:
        """
        Process raw events into LSTM-ready format
        
        Args:
            raw_events: List of raw pump events
            
        Returns:
            DataFrame with LSTM-ready data
        """
        # Step 1: Extract and categorize events
        categorized_events = self.extractor.extract_events(raw_events)
        
        # Step 2: Normalize each event type
        cgm_data = self.extractor.normalize_cgm_events(categorized_events['cgm_events'])
        basal_data = self.extractor.normalize_basal_events(categorized_events['basal_events'])
        bolus_data = self.extractor.normalize_bolus_events(categorized_events['bolus_events'])
        
        # # Step 3: Validate data
        # cgm_data, _ = self.validator.validate_events(cgm_data)
        # basal_data, _ = self.validator.validate_events(basal_data)
        # bolus_data, _ = self.validator.validate_events(bolus_data)
        
        # Step 4: Deduplicate
        cgm_data = self.extractor.deduplicate_events(cgm_data)
        basal_data = self.extractor.deduplicate_events(basal_data)
        bolus_data = self.extractor.deduplicate_events(bolus_data)
        
        # Step 5: Save individual CSV files if data exists
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if cgm_data:
            cgm_df = pd.DataFrame(cgm_data)
            cgm_file = Path(self.output_dir) / f'cgmreading_{timestamp}.csv'
            cgm_df.to_csv(cgm_file, index=False)
            logger.info(f"Saved CGM data to {cgm_file}")
        
        if basal_data:
            basal_df = pd.DataFrame(basal_data)
            basal_file = Path(self.output_dir) / f'basal_{timestamp}.csv'
            basal_df.to_csv(basal_file, index=False)
            logger.info(f"Saved basal data to {basal_file}")
        
        if bolus_data:
            bolus_df = pd.DataFrame(bolus_data)
            bolus_file = Path(self.output_dir) / f'bolus_{timestamp}.csv'
            bolus_df.to_csv(bolus_file, index=False)
            logger.info(f"Saved bolus data to {bolus_file}")
        
        # Step 6: Create LSTM-ready dataset
        lstm_df = self.processor.create_lstm_ready_data(cgm_data, basal_data, bolus_data)
        
        return lstm_df
    
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
        
        # Merge pump CSVs after all pumps are synced
        if any(results.values()):
            logger.info("Merging pump CSVs into combined format...")
            self.merge_pump_csvs()
        
        return results
    
    def generate_lstm_ready_data(self, pump_serial: str, output_file: Optional[str] = None) -> Optional[Path]:
        """
        Generate LSTM-ready dataset from existing CSV files
        
        Args:
            pump_serial: Pump serial number
            output_file: Output file path (optional)
            
        Returns:
            Path to generated file or None if failed
        """
        try:
            # Find the most recent CSV files for each type
            cgm_files = find_most_recent_files(self.output_dir, 'cgmreading_*.csv')
            basal_files = find_most_recent_files(self.output_dir, 'basal_*.csv')
            bolus_files = find_most_recent_files(self.output_dir, 'bolus_*.csv')
            
            if not cgm_files:
                logger.error(f"No CGM data files found in {self.output_dir}")
                return None
            
            # Load the most recent files
            cgm_df = load_csv_with_datetime(cgm_files[0])
            basal_df = load_csv_with_datetime(basal_files[0]) if basal_files else pd.DataFrame()
            bolus_df = load_csv_with_datetime(bolus_files[0]) if bolus_files else pd.DataFrame()
            
            if cgm_df.empty:
                logger.error("No CGM data available")
                return None
            
            # Generate LSTM-ready dataset
            lstm_df = self.processor.create_lstm_ready_from_csv(cgm_df, basal_df, bolus_df)
            
            # Set output file
            if output_file is None:
                output_file = str(get_lstm_output_path(self.output_dir, pump_serial))
            
            # Save the dataset
            lstm_df.to_csv(output_file, index=False)
            
            logger.info(f"Generated LSTM-ready dataset: {output_file}")
            logger.info(f"Total records: {len(lstm_df)}")
            
            return Path(output_file)
            
        except Exception as e:
            logger.error(f"Error generating LSTM-ready dataset: {e}")
            return None
    
    def merge_pump_csvs(self):
        """
        Merge all pump LSTM-ready CSV files into a combined format
        """
        try:
            lstm_dir = self.structure['lstm_pump_data']
            
            if not lstm_dir.exists():
                logger.warning("No LSTM-ready data directory found")
                return
            
            # Find all pump CSV files
            pump_files = list(lstm_dir.glob("pump_*.csv"))
            
            if not pump_files:
                logger.warning("No pump CSV files found")
                return
            
            combined_data = []
            
            for file_path in pump_files:
                try:
                    # Extract pump serial from filename (pump_XXXXXX_timestamp.csv)
                    parts = file_path.stem.split('_')
                    if len(parts) >= 2:
                        pump_serial = parts[1]
                    else:
                        pump_serial = 'unknown'
                    
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
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            combined_file = lstm_dir / f"combined_{timestamp}.csv"
            combined_df.to_csv(combined_file, index=False)
            
            logger.info(f"Merged {len(combined_df)} records from {len(pump_files)} pumps to {combined_file}")
            
        except Exception as e:
            logger.error(f"Error merging pump CSVs: {e}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current sync status for all pumps
        
        Returns:
            Dictionary with sync status for each pump
        """
        status = {}
        
        for serial, metadata in self.metadata.items():
            # Count CSV files for this pump in lstm_pump_data directory
            lstm_data_dir = self.structure['lstm_pump_data']
            pump_files = list(lstm_data_dir.glob(f'pump_{serial}_*.csv')) if lstm_data_dir.exists() else []
            
            status[serial] = {
                'last_successful_sync': metadata.last_successful_sync,
                'total_records': metadata.total_records,
                'last_updated': metadata.last_updated,
                'failed_ranges': len(metadata.failed_ranges) if metadata.failed_ranges else 0,
                'csv_files': len(pump_files)
            }
        
        return status
    
    def _get_last_sync_date(self, pump_serial: str) -> Optional[str]:
        """
        Get the last successful sync date for a pump
        
        Args:
            pump_serial: Pump serial number
            
        Returns:
            Last sync date or None if not found
        """
        if pump_serial in self.metadata:
            return self.metadata[pump_serial].last_successful_sync
        return None
    
    def _update_sync_metadata(self, pump_serial: str, end_date: str, record_count: int):
        """
        Update sync metadata after successful sync
        
        Args:
            pump_serial: Pump serial number
            end_date: End date of sync
            record_count: Number of records processed
        """
        if pump_serial not in self.metadata:
            self.metadata[pump_serial] = SyncMetadata(pump_serial=pump_serial)
        
        metadata = self.metadata[pump_serial]
        metadata.last_successful_sync = end_date
        metadata.total_records += record_count
        metadata.last_updated = datetime.now().isoformat()
        
        # Save to structured metadata directory
        from ..utils.file_utils import save_structured_metadata
        save_structured_metadata(
            self.structure,
            {serial: metadata.__dict__ for serial, metadata in self.metadata.items()},
            'sync_metadata.json'
        )
    
    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive client statistics
        
        Returns:
            Dictionary with client stats
        """
        return {
            'output_directory': str(self.output_dir),
            'structure_base': str(self.structure['base']),
            'lstm_pump_data_directory': str(self.structure['lstm_pump_data']),
            'metadata_directory': str(self.structure['metadata']),
            'models_directory': str(self.structure['models']),
            'logs_directory': str(self.structure['logs']),
            'connector_info': self.connector.get_connection_info(),
            'fetcher_stats': self.fetcher.get_fetcher_stats(),
            'extraction_stats': self.extractor.get_extraction_stats(),
            'processing_stats': self.processor.get_processing_stats(),
            'validation_stats': self.validator.get_validation_stats(),
            'sync_status': self.get_sync_status()
        }
    
    def reset_all_stats(self):
        """Reset all component statistics"""
        self.extractor.reset_stats()
        self.processor.reset_stats()
        self.validator.reset_stats()
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection is working, False otherwise
        """
        return self.connector.test_connection()
    
    def disconnect(self):
        """Disconnect from API"""
        self.connector.disconnect()
