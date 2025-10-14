#!/usr/bin/env python3
"""
Comprehensive End-to-End Validation Framework for bloodBath System

This framework performs complete validation of data ingestion, organization, 
and preprocessing subsystems to ensure system readiness for 5-year scale
historical data processing and LSTM model integration.

Usage:
    python -m bloodBath.validation.test_framework --test-type full --years 5
    python -m bloodBath.validation.test_framework --test-type incremental --start-date 2019-01-01
"""

import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import sys

# Import bloodBath components
from ..core.client import TandemHistoricalSyncClient
from ..core.config import PumpConfig
from ..utils.logging_utils import get_logger
from ..utils.env_utils import get_credentials, get_env_config

logger = get_logger(__name__)


@dataclass
class ValidationMetrics:
    """Structured validation metrics for reporting"""
    test_start_time: datetime
    test_end_time: Optional[datetime] = None
    total_records_processed: int = 0
    api_batches_completed: int = 0
    api_batches_failed: int = 0
    avg_api_latency_seconds: float = 0.0
    preprocessing_duration_seconds: float = 0.0
    missing_timestamps_count: int = 0
    duplicate_timestamps_count: int = 0
    raw_data_size_mb: float = 0.0
    processed_data_size_mb: float = 0.0
    continuity_gaps_detected: int = 0
    physiological_range_violations: int = 0
    pipeline_completion_status: str = "IN_PROGRESS"
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


@dataclass 
class BatchMetadata:
    """Metadata for individual data retrieval batches"""
    batch_id: str
    pump_serial: str
    start_time: datetime
    end_time: datetime
    record_count: int = 0
    request_duration_seconds: float = 0.0
    data_size_bytes: int = 0
    completion_status: str = "PENDING"
    error_message: Optional[str] = None
    retry_count: int = 0
    
    
class ValidationTestFramework:
    """
    Comprehensive end-to-end validation framework for bloodBath system
    """
    
    # Expected physiological ranges for validation
    GLUCOSE_RANGE_MG_DL = (40, 600)  # Extreme but plausible CGM range
    BASAL_RANGE_U_HR = (0.0, 10.0)   # Typical basal rate range
    BOLUS_RANGE_U = (0.0, 50.0)      # Typical bolus dose range
    
    # Expected data frequency (5-minute intervals)
    EXPECTED_INTERVAL_MINUTES = 5
    MAX_GAP_MINUTES = 60  # Alert if gaps exceed 1 hour
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 chunk_days: int = 30,
                 max_retries: int = 5,
                 rate_limit_delay: int = 2):
        """
        Initialize validation framework
        
        Args:
            output_dir: Override default sweetBlood directory
            chunk_days: Days per API request chunk  
            max_retries: Max retries per failed chunk
            rate_limit_delay: Delay between API requests (seconds)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.chunk_days = chunk_days
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize client
        creds = get_credentials()
        self.client = TandemHistoricalSyncClient(
            email=creds.email,
            password=creds.password,
            region=creds.region,
            output_dir=str(self.output_dir) if self.output_dir else None,
            chunk_days=chunk_days,
            max_retries=max_retries,
            rate_limit_delay=rate_limit_delay
        )
        
        # Validation state
        self.metrics = ValidationMetrics(test_start_time=datetime.now(timezone.utc))
        self.batch_metadata: List[BatchMetadata] = []
        self.validation_report_path: Optional[Path] = None
        
        # Create validation output directory
        self.validation_output = self._setup_validation_directory()
        
    def _setup_validation_directory(self) -> Path:
        """Create validation output directory structure"""
        if self.output_dir:
            validation_dir = self.output_dir / "validation"
        else:
            # Use bloodBath's internal sweetBlood structure
            validation_dir = Path(__file__).parent.parent / "sweetBlood" / "validation"
            
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (validation_dir / "batch_metadata").mkdir(exist_ok=True)
        (validation_dir / "raw_data").mkdir(exist_ok=True)
        (validation_dir / "processed_data").mkdir(exist_ok=True)
        (validation_dir / "reports").mkdir(exist_ok=True)
        (validation_dir / "logs").mkdir(exist_ok=True)
        
        return validation_dir
        
    def test_api_connection(self) -> bool:
        """Test API connection and pump discovery"""
        logger.info("Testing API connection and pump discovery...")
        
        try:
            # Test basic connection
            if not self.client.test_connection():
                self.metrics.error_messages.append("Failed to establish API connection")
                return False
                
            # Test pump discovery  
            from ..utils.pump_info import analyze_pump_activity
            pump_info = analyze_pump_activity(self.client)
            
            if not pump_info:
                self.metrics.error_messages.append("No pumps discovered on account")
                return False
                
            logger.info(f"Successfully discovered {len(pump_info)} pump(s)")
            for serial, info in pump_info.items():
                logger.info(f"  Pump {serial}: {info.get('start_date', 'N/A')} to {info.get('end_date', 'N/A')}")
                
            return True
            
        except Exception as e:
            error_msg = f"API connection test failed: {str(e)}"
            logger.error(error_msg)
            self.metrics.error_messages.append(error_msg)
            return False
            
    def calculate_batch_schedule(self, 
                               pump_serial: str, 
                               start_date: datetime, 
                               end_date: datetime,
                               batch_frequency: str = "monthly") -> List[Tuple[datetime, datetime]]:
        """
        Calculate optimal batch schedule for incremental retrieval
        
        Args:
            pump_serial: Pump serial number
            start_date: Overall start date
            end_date: Overall end date  
            batch_frequency: 'monthly', 'quarterly', or 'custom'
            
        Returns:
            List of (start, end) datetime tuples for each batch
        """
        batches = []
        current_start = start_date
        
        if batch_frequency == "monthly":
            delta = timedelta(days=30)
        elif batch_frequency == "quarterly":
            delta = timedelta(days=90)
        else:  # custom based on chunk_days
            delta = timedelta(days=self.chunk_days)
            
        while current_start < end_date:
            current_end = min(current_start + delta, end_date)
            batches.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
            
        logger.info(f"Calculated {len(batches)} batches for pump {pump_serial} "
                   f"({batch_frequency} frequency)")
        return batches
        
    def execute_incremental_batch_retrieval(self,
                                          pump_configs: List[PumpConfig],
                                          batch_frequency: str = "monthly",
                                          resume_from_batch: Optional[str] = None) -> bool:
        """
        Execute incremental batch retrieval with resume capability
        
        Args:
            pump_configs: List of pump configurations to process
            batch_frequency: Batching strategy ('monthly', 'quarterly', 'custom')
            resume_from_batch: Batch ID to resume from (if resuming)
            
        Returns:
            True if all batches completed successfully
        """
        logger.info(f"Starting incremental batch retrieval for {len(pump_configs)} pumps")
        
        all_batches_successful = True
        
        for pump_config in pump_configs:
            logger.info(f"Processing pump {pump_config.serial}...")
            
            # Calculate batch schedule
            start_date = datetime.fromisoformat(pump_config.start_date.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(pump_config.end_date.replace('Z', '+00:00'))
            
            batch_schedule = self.calculate_batch_schedule(
                pump_config.serial, start_date, end_date, batch_frequency
            )
            
            # Process each batch
            for batch_idx, (batch_start, batch_end) in enumerate(batch_schedule):
                batch_id = f"{pump_config.serial}_{batch_start.strftime('%Y%m%d')}_{batch_end.strftime('%Y%m%d')}"
                
                # Skip if resuming and haven't reached resume point
                if resume_from_batch and batch_id != resume_from_batch:
                    if not any(b.batch_id == resume_from_batch for b in self.batch_metadata):
                        continue
                        
                # Create batch metadata
                batch_meta = BatchMetadata(
                    batch_id=batch_id,
                    pump_serial=pump_config.serial,
                    start_time=batch_start,
                    end_time=batch_end
                )
                
                # Execute batch retrieval
                batch_success = self._execute_single_batch(pump_config, batch_meta)
                
                if not batch_success:
                    all_batches_successful = False
                    
                # Save batch metadata
                self._save_batch_metadata(batch_meta)
                self.batch_metadata.append(batch_meta)
                
                # Add delay between batches to respect rate limits
                time.sleep(self.rate_limit_delay)
                
        return all_batches_successful
        
    def _execute_single_batch(self, 
                            pump_config: PumpConfig, 
                            batch_meta: BatchMetadata) -> bool:
        """Execute data retrieval for a single batch with error handling"""
        logger.info(f"Executing batch {batch_meta.batch_id}")
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Create temporary config for this batch
                batch_config = PumpConfig(
                    serial=pump_config.serial,
                    start_date=batch_meta.start_time.isoformat(),
                    end_date=batch_meta.end_time.isoformat()
                )
                
                # Sync data for this batch
                success = self.client.sync_pump_historical(batch_config)
                
                if success:
                    # Update batch metadata with success info
                    batch_meta.request_duration_seconds = time.time() - start_time
                    batch_meta.completion_status = "COMPLETED"
                    batch_meta.retry_count = retry_count
                    
                    # Get record count (approximate from client stats)
                    stats = self.client.get_client_stats()
                    extraction_stats = stats.get('extraction_stats', {})
                    batch_meta.record_count = extraction_stats.get('total_processed', 0)
                    
                    # Update overall metrics
                    self.metrics.api_batches_completed += 1
                    self.metrics.total_records_processed += batch_meta.record_count
                    
                    logger.info(f"✓ Batch {batch_meta.batch_id} completed successfully "
                              f"({batch_meta.record_count} records)")
                    return True
                    
                else:
                    retry_count += 1
                    logger.warning(f"Batch {batch_meta.batch_id} failed, retry {retry_count}/{self.max_retries}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    
            except Exception as e:
                retry_count += 1
                error_msg = f"Batch {batch_meta.batch_id} error: {str(e)}"
                logger.error(error_msg)
                
                if retry_count >= self.max_retries:
                    batch_meta.error_message = error_msg
                    batch_meta.completion_status = "FAILED"
                    batch_meta.retry_count = retry_count
                    self.metrics.api_batches_failed += 1
                    self.metrics.error_messages.append(error_msg)
                    return False
                    
                time.sleep(2 ** retry_count)  # Exponential backoff
                
        return False
        
    def _save_batch_metadata(self, batch_meta: BatchMetadata):
        """Save individual batch metadata to disk"""
        metadata_file = self.validation_output / "batch_metadata" / f"{batch_meta.batch_id}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(batch_meta), f, indent=2, default=str)
            
    def validate_data_continuity(self, pump_serial: str) -> Dict[str, Any]:
        """
        Validate data continuity and detect gaps in time series
        
        Args:
            pump_serial: Pump serial number to validate
            
        Returns:
            Dictionary with continuity validation results
        """
        logger.info(f"Validating data continuity for pump {pump_serial}")
        
        # Find all LSTM-ready data files for this pump
        lstm_data_dir = self.client.structure.get('lstm_pump_data', Path())
        csv_files = list(lstm_data_dir.glob(f'pump_{pump_serial}_*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found for pump {pump_serial}")
            return {"status": "no_data", "files_found": 0}
            
        # Load and combine all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    dataframes.append(df)
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                
        if not dataframes:
            return {"status": "load_error", "files_found": len(csv_files)}
            
        # Combine all data
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # Analyze continuity
        continuity_results = {
            "status": "analyzed",
            "files_found": len(csv_files),
            "total_records": len(combined_df),
            "date_range": {
                "start": combined_df['timestamp'].min().isoformat(),
                "end": combined_df['timestamp'].max().isoformat()
            }
        }
        
        # Check for expected 5-minute intervals
        if len(combined_df) > 1:
            time_diffs = combined_df['timestamp'].diff().dt.total_seconds() / 60  # Minutes
            expected_interval = self.EXPECTED_INTERVAL_MINUTES
            
            # Find gaps larger than expected
            large_gaps = time_diffs[time_diffs > self.MAX_GAP_MINUTES]
            continuity_results["large_gaps_count"] = len(large_gaps)
            continuity_results["max_gap_minutes"] = float(time_diffs.max()) if not time_diffs.empty else 0
            
            # Expected vs actual record count
            total_minutes = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).total_seconds() / 60
            expected_records = int(total_minutes / expected_interval)
            continuity_results["expected_records"] = expected_records
            continuity_results["actual_records"] = len(combined_df)
            continuity_results["completeness_ratio"] = len(combined_df) / expected_records if expected_records > 0 else 0
            
        # Update global metrics
        self.metrics.continuity_gaps_detected += continuity_results.get("large_gaps_count", 0)
        
        return continuity_results
        
    def validate_physiological_ranges(self, pump_serial: str) -> Dict[str, Any]:
        """
        Validate that glucose and insulin values are within physiological ranges
        
        Args:
            pump_serial: Pump serial number to validate
            
        Returns:
            Dictionary with range validation results
        """
        logger.info(f"Validating physiological ranges for pump {pump_serial}")
        
        # Load data files
        lstm_data_dir = self.client.structure.get('lstm_pump_data', Path())
        csv_files = list(lstm_data_dir.glob(f'pump_{pump_serial}_*.csv'))
        
        if not csv_files:
            return {"status": "no_data"}
            
        # Combine all data
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                
        if not dataframes:
            return {"status": "load_error"}
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Validate ranges
        range_results = {"status": "analyzed", "violations": {}}
        
        # Glucose validation
        if 'bg' in combined_df.columns:
            bg_values = combined_df['bg'].dropna()
            bg_violations = bg_values[
                (bg_values < self.GLUCOSE_RANGE_MG_DL[0]) | 
                (bg_values > self.GLUCOSE_RANGE_MG_DL[1])
            ]
            range_results["violations"]["glucose"] = {
                "count": len(bg_violations),
                "percentage": (len(bg_violations) / len(bg_values) * 100) if len(bg_values) > 0 else 0,
                "range_expected": self.GLUCOSE_RANGE_MG_DL,
                "range_actual": [float(bg_values.min()), float(bg_values.max())] if not bg_values.empty else [0, 0]
            }
            
        # Basal rate validation  
        if 'basal_rate' in combined_df.columns:
            basal_values = combined_df['basal_rate'].dropna()
            basal_violations = basal_values[
                (basal_values < self.BASAL_RANGE_U_HR[0]) | 
                (basal_values > self.BASAL_RANGE_U_HR[1])
            ]
            range_results["violations"]["basal"] = {
                "count": len(basal_violations),
                "percentage": (len(basal_violations) / len(basal_values) * 100) if len(basal_values) > 0 else 0,
                "range_expected": self.BASAL_RANGE_U_HR,
                "range_actual": [float(basal_values.min()), float(basal_values.max())] if not basal_values.empty else [0, 0]
            }
            
        # Bolus validation
        if 'bolus_dose' in combined_df.columns:
            bolus_values = combined_df['bolus_dose'].dropna()
            bolus_violations = bolus_values[
                (bolus_values < self.BOLUS_RANGE_U[0]) | 
                (bolus_values > self.BOLUS_RANGE_U[1])
            ]
            range_results["violations"]["bolus"] = {
                "count": len(bolus_violations),
                "percentage": (len(bolus_violations) / len(bolus_values) * 100) if len(bolus_values) > 0 else 0,
                "range_expected": self.BOLUS_RANGE_U,
                "range_actual": [float(bolus_values.min()), float(bolus_values.max())] if not bolus_values.empty else [0, 0]
            }
            
        # Update global metrics
        total_violations = sum(v["count"] for v in range_results["violations"].values())
        self.metrics.physiological_range_violations += total_violations
        
        return range_results
        
    def generate_validation_report(self) -> Path:
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        # Finalize metrics
        self.metrics.test_end_time = datetime.now(timezone.utc)
        test_duration = self.metrics.test_end_time - self.metrics.test_start_time
        
        # Calculate average API latency
        if self.batch_metadata:
            total_latency = sum(b.request_duration_seconds for b in self.batch_metadata)
            self.metrics.avg_api_latency_seconds = total_latency / len(self.batch_metadata)
            
        # Set completion status
        if self.metrics.api_batches_failed == 0 and self.metrics.api_batches_completed > 0:
            self.metrics.pipeline_completion_status = "SUCCESS"
        elif self.metrics.api_batches_completed > 0:
            self.metrics.pipeline_completion_status = "PARTIAL_SUCCESS"  
        else:
            self.metrics.pipeline_completion_status = "FAILURE"
            
        # Create comprehensive report
        report = {
            "validation_summary": {
                "test_duration_hours": test_duration.total_seconds() / 3600,
                "overall_status": self.metrics.pipeline_completion_status,
                "total_errors": len(self.metrics.error_messages)
            },
            "data_retrieval": {
                "total_records_processed": self.metrics.total_records_processed,
                "batches_completed": self.metrics.api_batches_completed,
                "batches_failed": self.metrics.api_batches_failed,
                "success_rate": (self.metrics.api_batches_completed / 
                                (self.metrics.api_batches_completed + self.metrics.api_batches_failed) * 100
                                if (self.metrics.api_batches_completed + self.metrics.api_batches_failed) > 0 else 0),
                "avg_api_latency_seconds": self.metrics.avg_api_latency_seconds
            },
            "data_quality": {
                "continuity_gaps_detected": self.metrics.continuity_gaps_detected,
                "physiological_range_violations": self.metrics.physiological_range_violations,
                "missing_timestamps": self.metrics.missing_timestamps_count,
                "duplicate_timestamps": self.metrics.duplicate_timestamps_count
            },
            "storage_metrics": {
                "raw_data_size_mb": self.metrics.raw_data_size_mb,
                "processed_data_size_mb": self.metrics.processed_data_size_mb
            },
            "error_summary": self.metrics.error_messages,
            "detailed_metrics": asdict(self.metrics),
            "batch_details": [asdict(batch) for batch in self.batch_metadata]
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.validation_output / "reports" / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Also save as CSV for easy analysis
        csv_file = report_file.with_suffix('.csv')
        
        # Create summary DataFrame
        summary_data = {
            "Metric": [
                "Test Duration (hours)",
                "Overall Status", 
                "Total Records Processed",
                "Batches Completed",
                "Batches Failed",
                "Success Rate (%)",
                "Avg API Latency (s)",
                "Continuity Gaps",
                "Range Violations",
                "Total Errors"
            ],
            "Value": [
                f"{test_duration.total_seconds() / 3600:.2f}",
                self.metrics.pipeline_completion_status,
                self.metrics.total_records_processed,
                self.metrics.api_batches_completed,
                self.metrics.api_batches_failed,
                f"{report['data_retrieval']['success_rate']:.1f}",
                f"{self.metrics.avg_api_latency_seconds:.2f}",
                self.metrics.continuity_gaps_detected,
                self.metrics.physiological_range_violations,
                len(self.metrics.error_messages)
            ]
        }
        
        pd.DataFrame(summary_data).to_csv(csv_file, index=False)
        
        logger.info(f"Validation report saved to: {report_file}")
        self.validation_report_path = report_file
        
        return report_file
        
    def run_full_validation(self, 
                          years: int = 5,
                          batch_frequency: str = "monthly",
                          target_pumps: Optional[List[str]] = None) -> bool:
        """
        Run complete end-to-end validation for specified time period
        
        Args:
            years: Number of years of historical data to validate
            batch_frequency: Batching strategy for retrieval
            target_pumps: Specific pump serials to test (None for all)
            
        Returns:
            True if validation passes all criteria
        """
        logger.info(f"Starting full {years}-year validation test")
        
        try:
            # 1. Test API connection and pump discovery
            if not self.test_api_connection():
                return False
                
            # 2. Get pump configurations
            from ..utils.pump_info import analyze_pump_activity
            pump_info = analyze_pump_activity(self.client)
            
            # Create pump configurations for testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            pump_configs = []
            for serial, info in pump_info.items():
                if target_pumps and serial not in target_pumps:
                    continue
                    
                # Use pump's actual date range if available, otherwise use calculated range
                actual_start = info.get('start_date', start_date.isoformat())
                actual_end = info.get('end_date', end_date.isoformat())
                
                pump_config = PumpConfig(
                    serial=serial,
                    start_date=max(start_date.isoformat(), actual_start),
                    end_date=min(end_date.isoformat(), actual_end)
                )
                pump_configs.append(pump_config)
                
            if not pump_configs:
                logger.error("No valid pump configurations found")
                return False
                
            logger.info(f"Testing {len(pump_configs)} pump configurations")
            
            # 3. Execute incremental batch retrieval
            retrieval_success = self.execute_incremental_batch_retrieval(
                pump_configs, batch_frequency
            )
            
            # 4. Validate data continuity and quality for each pump
            for pump_config in pump_configs:
                continuity_results = self.validate_data_continuity(pump_config.serial)
                logger.info(f"Pump {pump_config.serial} continuity: {continuity_results.get('status')}")
                
                range_results = self.validate_physiological_ranges(pump_config.serial)
                logger.info(f"Pump {pump_config.serial} range validation: {range_results.get('status')}")
                
            # 5. Generate comprehensive report
            self.generate_validation_report()
            
            # 6. Determine overall success
            success_criteria = [
                self.metrics.api_batches_completed > 0,
                self.metrics.api_batches_failed < (self.metrics.api_batches_completed * 0.1),  # <10% failure rate
                self.metrics.total_records_processed > 1000,  # Minimum data threshold
                len(self.metrics.error_messages) < 5  # Low error count
            ]
            
            overall_success = all(success_criteria)
            
            logger.info(f"Validation test completed. Overall success: {overall_success}")
            logger.info(f"Report available at: {self.validation_report_path}")
            
            return overall_success
            
        except Exception as e:
            error_msg = f"Validation test failed with exception: {str(e)}"
            logger.error(error_msg)
            self.metrics.error_messages.append(error_msg)
            self.metrics.pipeline_completion_status = "FAILURE"
            
            # Still generate report for debugging
            try:
                self.generate_validation_report()
            except:
                pass
                
            return False


def main():
    """Main CLI interface for validation framework"""
    parser = argparse.ArgumentParser(description="bloodBath Validation Test Framework")
    
    parser.add_argument('--test-type', 
                       choices=['full', 'connection', 'batch', 'continuity', 'ranges'],
                       default='full',
                       help='Type of validation test to run')
    
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of historical data to validate')
    
    parser.add_argument('--batch-frequency',
                       choices=['monthly', 'quarterly', 'custom'],
                       default='monthly',
                       help='Batch frequency for data retrieval')
    
    parser.add_argument('--target-pumps', nargs='+',
                       help='Specific pump serials to test')
    
    parser.add_argument('--output-dir',
                       help='Custom output directory for validation results')
    
    parser.add_argument('--chunk-days', type=int, default=30,
                       help='Days per API request chunk')
    
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retries per failed batch')
    
    parser.add_argument('--rate-limit-delay', type=int, default=2,
                       help='Delay between API requests (seconds)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create framework instance
    framework = ValidationTestFramework(
        output_dir=args.output_dir,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries,
        rate_limit_delay=args.rate_limit_delay
    )
    
    # Run appropriate test
    if args.test_type == 'full':
        success = framework.run_full_validation(
            years=args.years,
            batch_frequency=args.batch_frequency,
            target_pumps=args.target_pumps
        )
    elif args.test_type == 'connection':
        success = framework.test_api_connection()
    else:
        logger.error(f"Test type '{args.test_type}' not yet implemented")
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()