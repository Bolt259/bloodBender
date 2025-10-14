#!/usr/bin/env python3
"""
Comprehensive Metrics and Logging System for bloodBath Validation

Provides structured JSON/CSV logging for all pipeline stages with performance
metrics, validation summaries, batch job tracking, and comprehensive reporting.

Features:
- Multi-format logging (JSON, CSV, structured logs)
- Performance metrics tracking across all pipeline stages
- Batch job progress monitoring and reporting
- Validation result aggregation and trend analysis
- Comprehensive dashboard-ready metrics export
- Real-time progress tracking and alerting
- Historical analysis and comparison capabilities
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, NamedTuple
from dataclasses import dataclass, field, asdict
import json
import csv
import time
from collections import defaultdict
import threading
from contextlib import contextmanager

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStageMetrics:
    """Metrics for a single pipeline stage"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    memory_peak_mb: Optional[float] = None
    cpu_time_seconds: Optional[float] = None
    status: str = "running"  # "running", "completed", "failed", "cancelled"
    error_message: Optional[str] = None
    stage_specific_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJobMetrics:
    """Comprehensive metrics for a batch job"""
    job_id: str
    pump_serial: str
    job_type: str  # "retrieval", "organization", "validation"
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Data volume metrics
    total_records: int = 0
    data_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Processing metrics
    throughput_records_per_second: float = 0.0
    throughput_mb_per_second: float = 0.0
    
    # Quality metrics
    data_completeness_pct: float = 0.0
    validation_score: float = 0.0
    critical_issues: int = 0
    warnings: int = 0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Stage breakdown
    stage_metrics: List[PipelineStageMetrics] = field(default_factory=list)
    
    # Status
    final_status: str = "running"  # "completed", "failed", "cancelled"
    error_summary: Optional[str] = None


@dataclass
class ValidationSessionMetrics:
    """Metrics for an entire validation session"""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Job summary
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    # Data summary
    total_pumps: int = 0
    total_records_processed: int = 0
    total_data_volume_gb: float = 0.0
    
    # Time range processed
    earliest_data_timestamp: Optional[datetime] = None
    latest_data_timestamp: Optional[datetime] = None
    
    # Overall quality metrics
    avg_validation_score: float = 0.0
    total_critical_issues: int = 0
    total_warnings: int = 0
    
    # Performance summary
    total_processing_time_hours: float = 0.0
    avg_throughput_records_per_hour: float = 0.0
    
    # Job breakdown
    batch_jobs: List[BatchJobMetrics] = field(default_factory=list)
    
    session_status: str = "running"


class MetricsCollector:
    """
    Collects and tracks metrics throughout the validation pipeline
    """
    
    def __init__(self, session_name: str, base_output_dir: Path):
        """
        Initialize metrics collector
        
        Args:
            session_name: Name for this validation session
            base_output_dir: Base directory for metrics output
        """
        self.session_name = session_name
        self.session_id = f"{session_name}_{int(time.time())}"
        self.base_output_dir = Path(base_output_dir)
        
        # Create metrics directory
        self.metrics_dir = self.base_output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session metrics
        self.session_metrics = ValidationSessionMetrics(
            session_id=self.session_id,
            session_name=session_name,
            start_time=datetime.now()
        )
        
        # Current tracking
        self.current_job: Optional[BatchJobMetrics] = None
        self.current_stage: Optional[PipelineStageMetrics] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup logging
        self._setup_structured_logging()
        
        logger.info(f"Metrics collector initialized for session: {self.session_id}")
        
    def _setup_structured_logging(self):
        """Setup structured logging for metrics"""
        
        # JSON log file for structured data
        json_log_file = self.metrics_dir / f"{self.session_id}_structured.jsonl"
        self.json_handler = logging.FileHandler(json_log_file)
        self.json_handler.setLevel(logging.INFO)
        
        # CSV log file for tabular metrics
        self.csv_log_file = self.metrics_dir / f"{self.session_id}_metrics.csv"
        
        # Real-time progress file
        self.progress_file = self.metrics_dir / f"{self.session_id}_progress.json"
        
    @contextmanager
    def batch_job(self, job_id: str, pump_serial: str, job_type: str):
        """
        Context manager for tracking a complete batch job
        
        Args:
            job_id: Unique identifier for the job
            pump_serial: Pump serial number being processed
            job_type: Type of job (retrieval, organization, validation)
        """
        job_metrics = BatchJobMetrics(
            job_id=job_id,
            pump_serial=pump_serial,
            job_type=job_type,
            start_time=datetime.now()
        )
        
        with self._lock:
            self.current_job = job_metrics
            self.session_metrics.total_jobs += 1
            
        try:
            logger.info(f"Starting {job_type} job {job_id} for pump {pump_serial}")
            
            yield job_metrics
            
            # Job completed successfully
            job_metrics.end_time = datetime.now()
            job_metrics.total_duration_seconds = (
                job_metrics.end_time - job_metrics.start_time
            ).total_seconds()
            job_metrics.final_status = "completed"
            
            with self._lock:
                self.session_metrics.completed_jobs += 1
                
            logger.info(f"Completed job {job_id} in {job_metrics.total_duration_seconds:.2f}s")
            
        except Exception as e:
            # Job failed
            job_metrics.end_time = datetime.now()
            job_metrics.total_duration_seconds = (
                job_metrics.end_time - job_metrics.start_time
            ).total_seconds()
            job_metrics.final_status = "failed"
            job_metrics.error_summary = str(e)
            
            with self._lock:
                self.session_metrics.failed_jobs += 1
                
            logger.error(f"Job {job_id} failed after {job_metrics.total_duration_seconds:.2f}s: {e}")
            raise
            
        finally:
            # Add job to session metrics
            with self._lock:
                self.session_metrics.batch_jobs.append(job_metrics)
                self.current_job = None
                
            # Log job completion
            self._log_job_metrics(job_metrics)
            self._update_progress_file()
            
    @contextmanager 
    def pipeline_stage(self, stage_name: str):
        """
        Context manager for tracking a pipeline stage within a job
        
        Args:
            stage_name: Name of the pipeline stage
        """
        if not self.current_job:
            logger.warning(f"Stage {stage_name} started outside of job context")
            return
            
        stage_metrics = PipelineStageMetrics(
            stage_name=stage_name,
            start_time=datetime.now()
        )
        
        with self._lock:
            self.current_stage = stage_metrics
            
        try:
            logger.info(f"Starting stage: {stage_name}")
            
            yield stage_metrics
            
            # Stage completed
            stage_metrics.end_time = datetime.now()
            stage_metrics.duration_seconds = (
                stage_metrics.end_time - stage_metrics.start_time
            ).total_seconds()
            stage_metrics.status = "completed"
            
            logger.info(f"Completed stage {stage_name} in {stage_metrics.duration_seconds:.2f}s")
            
        except Exception as e:
            # Stage failed
            stage_metrics.end_time = datetime.now()
            stage_metrics.duration_seconds = (
                stage_metrics.end_time - stage_metrics.start_time
            ).total_seconds()
            stage_metrics.status = "failed"
            stage_metrics.error_message = str(e)
            
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
            
        finally:
            # Add stage to current job
            if self.current_job:
                self.current_job.stage_metrics.append(stage_metrics)
                
            with self._lock:
                self.current_stage = None
                
    def record_data_metrics(self, 
                          records_processed: int,
                          data_size_mb: float,
                          compressed_size_mb: Optional[float] = None):
        """Record data volume metrics for current job"""
        if not self.current_job:
            return
            
        self.current_job.total_records += records_processed
        self.current_job.data_size_mb += data_size_mb
        
        if compressed_size_mb is not None:
            self.current_job.compressed_size_mb += compressed_size_mb
            if self.current_job.data_size_mb > 0:
                self.current_job.compression_ratio = (
                    self.current_job.compressed_size_mb / self.current_job.data_size_mb
                )
                
        # Update throughput if job duration available
        if self.current_job.start_time:
            elapsed_seconds = (datetime.now() - self.current_job.start_time).total_seconds()
            if elapsed_seconds > 0:
                self.current_job.throughput_records_per_second = (
                    self.current_job.total_records / elapsed_seconds
                )
                self.current_job.throughput_mb_per_second = (
                    self.current_job.data_size_mb / elapsed_seconds
                )
                
    def record_validation_metrics(self,
                                validation_score: float,
                                critical_issues: int,
                                warnings: int,
                                completeness_pct: float):
        """Record validation quality metrics for current job"""
        if not self.current_job:
            return
            
        self.current_job.validation_score = validation_score
        self.current_job.critical_issues = critical_issues
        self.current_job.warnings = warnings
        self.current_job.data_completeness_pct = completeness_pct
        
    def record_resource_metrics(self,
                              memory_mb: float,
                              cpu_percent: Optional[float] = None):
        """Record resource usage metrics"""
        if self.current_job:
            self.current_job.peak_memory_mb = max(
                self.current_job.peak_memory_mb, memory_mb
            )
            if cpu_percent is not None:
                # Simple moving average for CPU
                if self.current_job.avg_cpu_percent == 0:
                    self.current_job.avg_cpu_percent = cpu_percent
                else:
                    self.current_job.avg_cpu_percent = (
                        (self.current_job.avg_cpu_percent + cpu_percent) / 2
                    )
                    
        if self.current_stage:
            self.current_stage.memory_peak_mb = max(
                self.current_stage.memory_peak_mb or 0, memory_mb
            )
            
    def record_stage_metrics(self, **kwargs):
        """Record stage-specific metrics"""
        if self.current_stage:
            self.current_stage.stage_specific_metrics.update(kwargs)
            
    def finalize_session(self):
        """Finalize the validation session and generate final reports"""
        
        self.session_metrics.end_time = datetime.now()
        
        # Calculate session-level aggregates
        if self.session_metrics.batch_jobs:
            # Data aggregates
            self.session_metrics.total_records_processed = sum(
                job.total_records for job in self.session_metrics.batch_jobs
            )
            self.session_metrics.total_data_volume_gb = sum(
                job.data_size_mb for job in self.session_metrics.batch_jobs
            ) / 1024
            
            # Quality aggregates
            completed_jobs = [job for job in self.session_metrics.batch_jobs 
                            if job.final_status == "completed"]
            
            if completed_jobs:
                self.session_metrics.avg_validation_score = np.mean([
                    job.validation_score for job in completed_jobs
                ])
                
            self.session_metrics.total_critical_issues = sum(
                job.critical_issues for job in self.session_metrics.batch_jobs
            )
            self.session_metrics.total_warnings = sum(
                job.warnings for job in self.session_metrics.batch_jobs
            )
            
            # Time range
            timestamps = []
            for job in self.session_metrics.batch_jobs:
                timestamps.extend([job.start_time, job.end_time])
            timestamps = [t for t in timestamps if t is not None]
            
            if timestamps:
                self.session_metrics.earliest_data_timestamp = min(timestamps)
                self.session_metrics.latest_data_timestamp = max(timestamps)
                
            # Performance aggregates
            total_duration = (
                self.session_metrics.end_time - self.session_metrics.start_time
            ).total_seconds()
            self.session_metrics.total_processing_time_hours = total_duration / 3600
            
            if self.session_metrics.total_processing_time_hours > 0:
                self.session_metrics.avg_throughput_records_per_hour = (
                    self.session_metrics.total_records_processed / 
                    self.session_metrics.total_processing_time_hours
                )
                
            # Get unique pumps
            unique_pumps = set(job.pump_serial for job in self.session_metrics.batch_jobs)
            self.session_metrics.total_pumps = len(unique_pumps)
            
        # Determine final session status
        if self.session_metrics.failed_jobs == 0:
            self.session_metrics.session_status = "completed"
        elif self.session_metrics.completed_jobs > 0:
            self.session_metrics.session_status = "partial_success"
        else:
            self.session_metrics.session_status = "failed"
            
        # Generate final reports
        self._generate_final_reports()
        
        logger.info(f"Session {self.session_id} finalized: "
                   f"{self.session_metrics.completed_jobs}/{self.session_metrics.total_jobs} jobs completed")
                   
    def _log_job_metrics(self, job_metrics: BatchJobMetrics):
        """Log job metrics in structured format"""
        
        # JSON structured log
        json_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "job_completed",
            "session_id": self.session_id,
            **asdict(job_metrics)
        }
        
        # Convert datetime objects to ISO strings for JSON serialization
        def convert_datetimes(obj):
            if isinstance(obj, dict):
                return {k: convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetimes(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
            
        json_entry = convert_datetimes(json_entry)
        
        # Write to JSON log
        with open(self.metrics_dir / f"{self.session_id}_structured.jsonl", 'a') as f:
            f.write(json.dumps(json_entry) + '\n')
            
        # Write to CSV log
        self._append_to_csv_log(job_metrics)
        
    def _append_to_csv_log(self, job_metrics: BatchJobMetrics):
        """Append job metrics to CSV log"""
        
        csv_row = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'job_id': job_metrics.job_id,
            'pump_serial': job_metrics.pump_serial,
            'job_type': job_metrics.job_type,
            'duration_seconds': job_metrics.total_duration_seconds,
            'records_processed': job_metrics.total_records,
            'data_size_mb': job_metrics.data_size_mb,
            'validation_score': job_metrics.validation_score,
            'critical_issues': job_metrics.critical_issues,
            'warnings': job_metrics.warnings,
            'final_status': job_metrics.final_status,
            'throughput_records_per_sec': job_metrics.throughput_records_per_second,
            'peak_memory_mb': job_metrics.peak_memory_mb
        }
        
        file_exists = self.csv_log_file.exists()
        
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(csv_row)
            
    def _update_progress_file(self):
        """Update real-time progress file"""
        
        progress_data = {
            "session_id": self.session_id,
            "last_update": datetime.now().isoformat(),
            "total_jobs": self.session_metrics.total_jobs,
            "completed_jobs": self.session_metrics.completed_jobs,
            "failed_jobs": self.session_metrics.failed_jobs,
            "session_status": self.session_metrics.session_status,
            "current_job": {
                "job_id": self.current_job.job_id,
                "pump_serial": self.current_job.pump_serial,
                "job_type": self.current_job.job_type,
                "records_processed": self.current_job.total_records
            } if self.current_job else None
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def _generate_final_reports(self):
        """Generate comprehensive final reports"""
        
        # Session summary report
        summary_report = {
            "session_summary": asdict(self.session_metrics),
            "generated_at": datetime.now().isoformat()
        }
        
        # Convert datetimes for JSON serialization
        def convert_datetimes(obj):
            if isinstance(obj, dict):
                return {k: convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetimes(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
            
        summary_report = convert_datetimes(summary_report)
        
        # Save session summary
        summary_file = self.metrics_dir / f"{self.session_id}_session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
            
        # Generate performance analysis
        self._generate_performance_analysis()
        
        # Generate quality analysis
        self._generate_quality_analysis()
        
        logger.info(f"Final reports generated in {self.metrics_dir}")
        
    def _generate_performance_analysis(self):
        """Generate performance analysis report"""
        
        if not self.session_metrics.batch_jobs:
            return
            
        # Performance statistics
        durations = [job.total_duration_seconds for job in self.session_metrics.batch_jobs 
                    if job.total_duration_seconds is not None]
        throughputs = [job.throughput_records_per_second for job in self.session_metrics.batch_jobs
                      if job.throughput_records_per_second > 0]
        memory_usage = [job.peak_memory_mb for job in self.session_metrics.batch_jobs
                       if job.peak_memory_mb > 0]
        
        performance_stats = {}
        
        if durations:
            performance_stats["duration_statistics"] = {
                "mean_seconds": np.mean(durations),
                "median_seconds": np.median(durations),
                "std_seconds": np.std(durations),
                "min_seconds": np.min(durations),
                "max_seconds": np.max(durations)
            }
            
        if throughputs:
            performance_stats["throughput_statistics"] = {
                "mean_records_per_sec": np.mean(throughputs),
                "median_records_per_sec": np.median(throughputs),
                "std_records_per_sec": np.std(throughputs)
            }
            
        if memory_usage:
            performance_stats["memory_statistics"] = {
                "mean_peak_mb": np.mean(memory_usage),
                "max_peak_mb": np.max(memory_usage),
                "std_peak_mb": np.std(memory_usage)
            }
            
        # Performance by job type
        job_type_stats = defaultdict(list)
        for job in self.session_metrics.batch_jobs:
            if job.total_duration_seconds:
                job_type_stats[job.job_type].append(job.total_duration_seconds)
                
        performance_by_type = {}
        for job_type, durations in job_type_stats.items():
            performance_by_type[job_type] = {
                "job_count": len(durations),
                "mean_duration": np.mean(durations),
                "total_duration": sum(durations)
            }
            
        performance_report = {
            "performance_statistics": performance_stats,
            "performance_by_job_type": performance_by_type,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save performance report
        perf_file = self.metrics_dir / f"{self.session_id}_performance_analysis.json"
        with open(perf_file, 'w') as f:
            json.dump(performance_report, f, indent=2)
            
    def _generate_quality_analysis(self):
        """Generate data quality analysis report"""
        
        completed_jobs = [job for job in self.session_metrics.batch_jobs 
                         if job.final_status == "completed"]
        
        if not completed_jobs:
            return
            
        # Quality statistics
        validation_scores = [job.validation_score for job in completed_jobs 
                           if job.validation_score > 0]
        completeness_scores = [job.data_completeness_pct for job in completed_jobs 
                             if job.data_completeness_pct > 0]
        
        quality_stats = {}
        
        if validation_scores:
            quality_stats["validation_score_statistics"] = {
                "mean_score": np.mean(validation_scores),
                "median_score": np.median(validation_scores),
                "min_score": np.min(validation_scores),
                "max_score": np.max(validation_scores),
                "std_score": np.std(validation_scores)
            }
            
        if completeness_scores:
            quality_stats["completeness_statistics"] = {
                "mean_completeness": np.mean(completeness_scores),
                "median_completeness": np.median(completeness_scores),
                "min_completeness": np.min(completeness_scores),
                "max_completeness": np.max(completeness_scores)
            }
            
        # Issue summary
        total_critical = sum(job.critical_issues for job in completed_jobs)
        total_warnings = sum(job.warnings for job in completed_jobs)
        
        issue_summary = {
            "total_critical_issues": total_critical,
            "total_warnings": total_warnings,
            "jobs_with_critical_issues": sum(1 for job in completed_jobs if job.critical_issues > 0),
            "jobs_with_warnings": sum(1 for job in completed_jobs if job.warnings > 0),
            "clean_jobs": sum(1 for job in completed_jobs 
                            if job.critical_issues == 0 and job.warnings == 0)
        }
        
        quality_report = {
            "quality_statistics": quality_stats,
            "issue_summary": issue_summary,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save quality report
        quality_file = self.metrics_dir / f"{self.session_id}_quality_analysis.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
            
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress summary for real-time monitoring"""
        
        return {
            "session_id": self.session_id,
            "session_status": self.session_metrics.session_status,
            "total_jobs": self.session_metrics.total_jobs,
            "completed_jobs": self.session_metrics.completed_jobs,
            "failed_jobs": self.session_metrics.failed_jobs,
            "current_job": {
                "job_id": self.current_job.job_id,
                "pump_serial": self.current_job.pump_serial,
                "job_type": self.current_job.job_type,
                "records_processed": self.current_job.total_records,
                "duration_so_far": (datetime.now() - self.current_job.start_time).total_seconds()
            } if self.current_job else None,
            "current_stage": {
                "stage_name": self.current_stage.stage_name,
                "duration_so_far": (datetime.now() - self.current_stage.start_time).total_seconds()
            } if self.current_stage else None
        }
        
        
class ValidationLogger:
    """
    Specialized logger for validation operations with structured output
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize validation logger
        
        Args:
            metrics_collector: Associated metrics collector
        """
        self.metrics_collector = metrics_collector
        self.logger = get_logger(f"{__name__}.validation")
        
    def log_validation_start(self, pump_serial: str, data_range: str):
        """Log validation start"""
        self.logger.info(f"Starting validation for pump {pump_serial}, data range: {data_range}")
        
    def log_validation_result(self, result: Dict[str, Any]):
        """Log structured validation result"""
        
        summary = result.get("validation_summary", {})
        pump_serial = summary.get("pump_serial", "unknown")
        overall_status = summary.get("overall_status", "unknown")
        overall_score = summary.get("overall_score", 0)
        
        self.logger.info(f"Validation completed for pump {pump_serial}: "
                        f"status={overall_status}, score={overall_score:.1f}")
        
        # Record in metrics collector
        status_counts = result.get("status_summary", {})
        self.metrics_collector.record_validation_metrics(
            validation_score=overall_score,
            critical_issues=status_counts.get("FAIL", 0),
            warnings=status_counts.get("WARN", 0),
            completeness_pct=result.get("quality_metrics", {}).get("data_completeness", {}).get("cgm_completeness_pct", 0)
        )
        
    def log_processing_milestone(self, milestone: str, details: Dict[str, Any] = None):
        """Log processing milestone"""
        details_str = f" - {details}" if details else ""
        self.logger.info(f"Processing milestone: {milestone}{details_str}")
        
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        context_str = f" ({context})" if context else ""
        self.logger.error(f"Validation error{context_str}: {error}")
        
    def log_warning(self, message: str, details: Dict[str, Any] = None):
        """Log warning with optional details"""
        details_str = f" - {details}" if details else ""
        self.logger.warning(f"{message}{details_str}")