#!/usr/bin/env python3
"""
Enhanced batch data retrieval system for bloodBath

Provides incremental, resumable batch data retrieval with compressed storage,
versioning, and comprehensive metadata tracking for large-scale historical
data synchronization.

Features:
- Monthly/quarterly batch processing
- Automatic resume from interruptions  
- Compressed Parquet/Feather storage
- Batch verification and integrity checks
- Rate limiting and retry logic
- Comprehensive batch metadata tracking
"""

import json
import time
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
import hashlib
import gzip
import pickle

from ..core.config import PumpConfig
from ..api.fetcher import TandemDataFetcher
from ..core.exceptions import TandemSyncError
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BatchJob:
    """Represents a single batch job for data retrieval"""
    job_id: str
    pump_serial: str
    start_date: datetime
    end_date: datetime
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    record_count: int = 0
    raw_data_size_bytes: int = 0
    compressed_size_bytes: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BatchSchedule:
    """Represents a complete batch schedule for a pump"""
    pump_serial: str
    overall_start: datetime
    overall_end: datetime
    batch_frequency: str  # "monthly", "quarterly", "custom"
    chunk_size_days: int
    total_batches: int
    jobs: List[BatchJob]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class EnhancedBatchRetriever:
    """
    Enhanced batch retrieval system with resume capability and compressed storage
    """
    
    def __init__(self, 
                 base_fetcher: TandemDataFetcher,
                 storage_dir: Path,
                 compression_format: str = "parquet"):  # "parquet", "feather", "pickle_gz"
        """
        Initialize enhanced batch retriever
        
        Args:
            base_fetcher: Underlying TandemDataFetcher instance
            storage_dir: Base directory for batch storage
            compression_format: Storage format for raw data
        """
        self.fetcher = base_fetcher
        self.storage_dir = Path(storage_dir)
        self.compression_format = compression_format
        
        # Setup storage structure
        self._setup_storage_structure()
        
        # Batch tracking
        self.active_schedules: Dict[str, BatchSchedule] = {}
        self.job_history: List[BatchJob] = []
        
        # Load existing schedules and job history
        self._load_existing_state()
        
    def _setup_storage_structure(self):
        """Create storage directory structure"""
        directories = [
            "raw_batches",     # Raw batch data files
            "processed",       # Processed batch data
            "metadata",        # Batch metadata and schedules
            "checksums",       # Data integrity checksums
            "logs",           # Batch processing logs
            "temp"            # Temporary files during processing
        ]
        
        for dir_name in directories:
            (self.storage_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def _load_existing_state(self):
        """Load existing batch schedules and job history from disk"""
        metadata_dir = self.storage_dir / "metadata"
        
        # Load schedules
        schedule_files = metadata_dir.glob("schedule_*.json")
        for schedule_file in schedule_files:
            try:
                with open(schedule_file) as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    for job_data in data["jobs"]:
                        for date_field in ["created_at", "started_at", "completed_at"]:
                            if job_data.get(date_field):
                                job_data[date_field] = datetime.fromisoformat(job_data[date_field])
                    
                    # Reconstruct schedule
                    schedule = BatchSchedule(**data)
                    self.active_schedules[schedule.pump_serial] = schedule
                    
            except Exception as e:
                logger.warning(f"Error loading schedule from {schedule_file}: {e}")
                
        # Load job history
        history_file = metadata_dir / "job_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history_data = json.load(f)
                    for job_data in history_data:
                        # Convert datetime strings
                        for date_field in ["created_at", "started_at", "completed_at"]:
                            if job_data.get(date_field):
                                job_data[date_field] = datetime.fromisoformat(job_data[date_field])
                        self.job_history.append(BatchJob(**job_data))
            except Exception as e:
                logger.warning(f"Error loading job history: {e}")
                
    def create_batch_schedule(self,
                            pump_serial: str,
                            start_date: datetime,
                            end_date: datetime,
                            batch_frequency: str = "monthly") -> BatchSchedule:
        """
        Create a complete batch schedule for a pump
        
        Args:
            pump_serial: Pump serial number
            start_date: Overall start date for data retrieval
            end_date: Overall end date for data retrieval  
            batch_frequency: Batching strategy
            
        Returns:
            BatchSchedule with all jobs defined
        """
        logger.info(f"Creating batch schedule for pump {pump_serial}: "
                   f"{start_date} to {end_date} ({batch_frequency})")
        
        # Determine chunk size based on frequency
        if batch_frequency == "monthly":
            chunk_days = 30
        elif batch_frequency == "quarterly":
            chunk_days = 90
        else:  # custom - use fetcher's default
            chunk_days = self.fetcher.chunk_days
            
        # Generate batch jobs
        jobs = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            job_id = f"{pump_serial}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            
            job = BatchJob(
                job_id=job_id,
                pump_serial=pump_serial,
                start_date=current_start,
                end_date=current_end
            )
            
            jobs.append(job)
            current_start = current_end + timedelta(days=1)
            
        # Create schedule
        schedule = BatchSchedule(
            pump_serial=pump_serial,
            overall_start=start_date,
            overall_end=end_date,
            batch_frequency=batch_frequency,
            chunk_size_days=chunk_days,
            total_batches=len(jobs),
            jobs=jobs
        )
        
        # Store and activate schedule
        self.active_schedules[pump_serial] = schedule
        self._save_schedule(schedule)
        
        logger.info(f"Created schedule with {len(jobs)} batch jobs")
        return schedule
        
    def _save_schedule(self, schedule: BatchSchedule):
        """Save batch schedule to disk"""
        metadata_dir = self.storage_dir / "metadata"
        schedule_file = metadata_dir / f"schedule_{schedule.pump_serial}.json"
        
        # Convert to serializable format
        schedule_data = asdict(schedule)
        
        with open(schedule_file, 'w') as f:
            json.dump(schedule_data, f, indent=2, default=str)
            
    def execute_batch_schedule(self,
                             pump_serial: str,
                             resume: bool = True,
                             max_concurrent_jobs: int = 1) -> Dict[str, Any]:
        """
        Execute all jobs in a batch schedule
        
        Args:
            pump_serial: Pump serial number
            resume: Whether to resume from previous execution
            max_concurrent_jobs: Maximum concurrent batch jobs (for future threading)
            
        Returns:
            Execution summary dictionary
        """
        if pump_serial not in self.active_schedules:
            raise ValueError(f"No active schedule found for pump {pump_serial}")
            
        schedule = self.active_schedules[pump_serial]
        logger.info(f"Executing batch schedule for pump {pump_serial} "
                   f"({schedule.total_batches} jobs, resume={resume})")
        
        # Execution tracking
        execution_start = datetime.now()
        completed_jobs = 0
        failed_jobs = 0
        skipped_jobs = 0
        total_records = 0
        total_raw_bytes = 0
        
        # Process each job
        for job in schedule.jobs:
            # Skip completed jobs if resuming
            if resume and job.status == "COMPLETED":
                skipped_jobs += 1
                completed_jobs += 1  # Count as completed for summary
                total_records += job.record_count
                total_raw_bytes += job.raw_data_size_bytes
                logger.debug(f"Skipping completed job: {job.job_id}")
                continue
                
            # Execute job
            job_result = self._execute_batch_job(job)
            
            if job_result["success"]:
                completed_jobs += 1
                total_records += job.record_count
                total_raw_bytes += job.raw_data_size_bytes
            else:
                failed_jobs += 1
                
            # Update schedule
            self._save_schedule(schedule)
            
            # Add to job history
            self.job_history.append(job)
            
            # Rate limiting delay
            time.sleep(self.fetcher.rate_limit_delay)
            
        # Save updated job history
        self._save_job_history()
        
        # Execution summary
        execution_end = datetime.now()
        execution_time = (execution_end - execution_start).total_seconds()
        
        summary = {
            "pump_serial": pump_serial,
            "execution_start": execution_start.isoformat(),
            "execution_end": execution_end.isoformat(),
            "execution_time_seconds": execution_time,
            "total_jobs": schedule.total_batches,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs, 
            "skipped_jobs": skipped_jobs,
            "success_rate": (completed_jobs / schedule.total_batches * 100) if schedule.total_batches > 0 else 0,
            "total_records_retrieved": total_records,
            "total_raw_data_mb": total_raw_bytes / (1024 * 1024),
            "avg_throughput_records_per_hour": (total_records / (execution_time / 3600)) if execution_time > 0 else 0
        }
        
        logger.info(f"Batch execution completed: {completed_jobs}/{schedule.total_batches} jobs successful")
        return summary
        
    def _execute_batch_job(self, job: BatchJob) -> Dict[str, Any]:
        """Execute a single batch job with error handling and storage"""
        logger.info(f"Executing batch job: {job.job_id}")
        
        job.status = "IN_PROGRESS" 
        job.started_at = datetime.now()
        
        try:
            # Create pump config for this batch
            pump_config = PumpConfig(
                serial=job.pump_serial,
                start_date=job.start_date.isoformat(),
                end_date=job.end_date.isoformat()
            )
            
            # Fetch raw events using existing fetcher
            raw_events = self.fetcher.fetch_pump_events_for_date_range(
                pump_config,
                job.start_date.strftime('%Y-%m-%d'),
                job.end_date.strftime('%Y-%m-%d')
            )
            
            if not raw_events:
                logger.warning(f"No events returned for job {job.job_id}")
                job.status = "COMPLETED"  # Not an error, just no data
                job.completed_at = datetime.now()
                return {"success": True, "message": "No data in date range"}
                
            # Convert to structured format for storage
            structured_data = self._convert_events_to_structured_format(raw_events)
            
            # Store compressed batch data
            storage_result = self._store_batch_data(job, structured_data, raw_events)
            
            # Update job with results
            job.record_count = len(structured_data)
            job.raw_data_size_bytes = storage_result["raw_size_bytes"]
            job.compressed_size_bytes = storage_result["compressed_size_bytes"] 
            job.checksum = storage_result["checksum"]
            job.status = "COMPLETED"
            job.completed_at = datetime.now()
            
            logger.info(f"✓ Job {job.job_id} completed: {job.record_count} records, "
                       f"{job.raw_data_size_bytes / 1024:.1f}KB raw")
            
            return {
                "success": True,
                "records": job.record_count,
                "raw_size_bytes": job.raw_data_size_bytes,
                "compressed_size_bytes": job.compressed_size_bytes
            }
            
        except Exception as e:
            # Handle job failure
            job.retry_count += 1
            error_msg = f"Job {job.job_id} failed (attempt {job.retry_count}): {str(e)}"
            logger.error(error_msg)
            
            if job.retry_count >= self.fetcher.max_retries:
                job.status = "FAILED"
                job.error_message = error_msg
                job.completed_at = datetime.now()
            else:
                job.status = "PENDING"  # Will retry
                
            return {"success": False, "error": error_msg}
            
    def _convert_events_to_structured_format(self, raw_events: List[Any]) -> pd.DataFrame:
        """Convert raw API events to structured DataFrame format"""
        # This would use the existing EventExtractor logic
        from ..data.extractors import EventExtractor
        
        extractor = EventExtractor()
        
        # Extract and categorize events
        categorized = extractor.extract_events(raw_events)
        
        # Normalize each event type
        cgm_events = extractor.normalize_cgm_events(categorized.get('cgm_events', []))
        basal_events = extractor.normalize_basal_events(categorized.get('basal_events', []))
        bolus_events = extractor.normalize_bolus_events(categorized.get('bolus_events', []))
        
        # Combine into single DataFrame with event type column
        all_events = []
        
        for event in cgm_events:
            event['event_type'] = 'CGM'
            all_events.append(event)
            
        for event in basal_events:
            event['event_type'] = 'BASAL'
            all_events.append(event)
            
        for event in bolus_events:
            event['event_type'] = 'BOLUS'
            all_events.append(event)
            
        if not all_events:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_events)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
        return df
        
    def _store_batch_data(self, 
                         job: BatchJob, 
                         structured_data: pd.DataFrame,
                         raw_events: List[Any]) -> Dict[str, Any]:
        """Store batch data in compressed format with checksum"""
        batch_dir = self.storage_dir / "raw_batches" / job.pump_serial
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        if self.compression_format == "parquet":
            data_file = batch_dir / f"{job.job_id}.parquet"
            raw_file = batch_dir / f"{job.job_id}_raw.parquet.gz"
        elif self.compression_format == "feather":
            data_file = batch_dir / f"{job.job_id}.feather"
            raw_file = batch_dir / f"{job.job_id}_raw.feather.gz"
        else:  # pickle_gz
            data_file = batch_dir / f"{job.job_id}.pkl.gz"
            raw_file = batch_dir / f"{job.job_id}_raw.pkl.gz"
            
        # Store structured data
        if self.compression_format == "parquet":
            structured_data.to_parquet(data_file, compression='gzip')
        elif self.compression_format == "feather":
            structured_data.to_feather(data_file)
        else:  # pickle_gz
            with gzip.open(data_file, 'wb') as f:
                pickle.dump(structured_data, f)
                
        # Store raw events (compressed pickle for exact preservation)
        with gzip.open(raw_file, 'wb') as f:
            pickle.dump(raw_events, f)
            
        # Calculate sizes and checksum
        structured_size = data_file.stat().st_size
        raw_size = raw_file.stat().st_size
        
        # Calculate checksum of structured data
        checksum = hashlib.md5()
        if self.compression_format == "parquet":
            df_bytes = structured_data.to_parquet()
            checksum.update(df_bytes)
        else:
            checksum.update(data_file.read_bytes())
            
        return {
            "compressed_size_bytes": structured_size,
            "raw_size_bytes": raw_size,
            "checksum": checksum.hexdigest(),
            "structured_file": str(data_file),
            "raw_file": str(raw_file)
        }
        
    def _save_job_history(self):
        """Save job history to disk"""
        history_file = self.storage_dir / "metadata" / "job_history.json"
        
        history_data = [asdict(job) for job in self.job_history]
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
            
    def get_batch_summary(self, pump_serial: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of batch processing status"""
        if pump_serial:
            schedules = {pump_serial: self.active_schedules.get(pump_serial)}
        else:
            schedules = self.active_schedules
            
        summary = {
            "active_schedules": len([s for s in schedules.values() if s is not None]),
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "pending_jobs": 0,
            "total_records": 0,
            "total_storage_mb": 0
        }
        
        for pump_serial, schedule in schedules.items():
            if schedule is None:
                continue
                
            for job in schedule.jobs:
                summary["total_jobs"] += 1
                
                if job.status == "COMPLETED":
                    summary["completed_jobs"] += 1
                    summary["total_records"] += job.record_count
                    summary["total_storage_mb"] += job.compressed_size_bytes / (1024 * 1024)
                elif job.status == "FAILED":
                    summary["failed_jobs"] += 1
                else:
                    summary["pending_jobs"] += 1
                    
        return summary
        
    def load_batch_data(self, job_id: str) -> Optional[pd.DataFrame]:
        """Load stored batch data by job ID"""
        # Find the job
        target_job = None
        for schedule in self.active_schedules.values():
            for job in schedule.jobs:
                if job.job_id == job_id:
                    target_job = job
                    break
                    
        if target_job is None:
            logger.error(f"Job {job_id} not found")
            return None
            
        # Load data file
        batch_dir = self.storage_dir / "raw_batches" / target_job.pump_serial
        
        if self.compression_format == "parquet":
            data_file = batch_dir / f"{job_id}.parquet"
            if data_file.exists():
                return pd.read_parquet(data_file)
        elif self.compression_format == "feather":
            data_file = batch_dir / f"{job_id}.feather"
            if data_file.exists():
                return pd.read_feather(data_file)
        else:  # pickle_gz
            data_file = batch_dir / f"{job_id}.pkl.gz"
            if data_file.exists():
                with gzip.open(data_file, 'rb') as f:
                    return pickle.load(f)
                    
        logger.error(f"Batch data file not found for job {job_id}")
        return None
        
    def verify_batch_integrity(self, job_id: str) -> bool:
        """Verify integrity of stored batch data using checksums"""
        target_job = None
        for schedule in self.active_schedules.values():
            for job in schedule.jobs:
                if job.job_id == job_id:
                    target_job = job
                    break
                    
        if target_job is None or target_job.checksum is None:
            return False
            
        # Load and recalculate checksum
        batch_data = self.load_batch_data(job_id)
        if batch_data is None:
            return False
            
        # Calculate current checksum
        checksum = hashlib.md5()
        if self.compression_format == "parquet":
            df_bytes = batch_data.to_parquet()
            checksum.update(df_bytes)
        else:
            # For other formats, serialize the DataFrame
            import pickle
            checksum.update(pickle.dumps(batch_data))
            
        current_checksum = checksum.hexdigest()
        
        return current_checksum == target_job.checksum