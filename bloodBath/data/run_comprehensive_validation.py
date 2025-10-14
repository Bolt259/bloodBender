#!/usr/bin/env python3
"""
Complete Validation Runner for bloodBath System

Demonstrates end-to-end usage of the comprehensive validation framework
for testing glucose data collection and processing before refactoring.

This script orchestrates:
1. Enhanced batch data retrieval with resume capability
2. Data organization and chronological merging
3. Comprehensive integrity validation
4. Structured metrics collection and reporting

Usage:
    python run_comprehensive_validation.py --config validation_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bloodBath.validation import (
    ValidationTestFramework,
    EnhancedBatchRetriever,
    EnhancedDataOrganizer,
    ComprehensiveValidator,
    MetricsCollector,
    ValidationLogger,
    ValidationConfig
)

from bloodBath.core.config import BloodBathConfig
from bloodBath.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ComprehensiveValidationRunner:
    """
    Orchestrates complete end-to-end validation of bloodBath system
    """
    
    def __init__(self, 
                 config_path: Path,
                 output_dir: Path,
                 session_name: str = "comprehensive_validation"):
        """
        Initialize validation runner
        
        Args:
            config_path: Path to validation configuration file
            output_dir: Output directory for results and metrics
            session_name: Name for this validation session
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self._load_configuration()
        
        # Initialize components
        self.metrics_collector = MetricsCollector(session_name, self.output_dir)
        self.validation_logger = ValidationLogger(self.metrics_collector)
        
        # Initialize framework components
        self.test_framework = ValidationTestFramework()
        self.batch_retriever = EnhancedBatchRetriever()
        self.data_organizer = EnhancedDataOrganizer()
        self.integrity_validator = ComprehensiveValidator(self.validation_config)
        
        logger.info(f"Validation runner initialized for session: {session_name}")
        
    def _load_configuration(self):
        """Load validation configuration"""
        
        try:
            with open(self.config_path) as f:
                config_data = json.load(f)
                
            # Extract validation parameters
            validation_params = config_data.get("validation", {})
            
            # Create validation config
            self.validation_config = ValidationConfig(
                cgm_normal_range=tuple(validation_params.get("cgm_normal_range", [70, 180])),
                cgm_extreme_range=tuple(validation_params.get("cgm_extreme_range", [40, 600])),
                basal_normal_range=tuple(validation_params.get("basal_normal_range", [0.5, 3.0])),
                max_acceptable_gap_minutes=validation_params.get("max_gap_minutes", 60),
                min_continuity_percentage=validation_params.get("min_continuity_pct", 85.0)
            )
            
            # Extract other parameters
            self.pump_serials = config_data.get("pump_serials", [])
            self.date_range = config_data.get("date_range", {})
            self.batch_mode = config_data.get("batch_mode", "monthly")
            self.validation_level = config_data.get("validation_level", "comprehensive")
            
            logger.info(f"Configuration loaded: {len(self.pump_serials)} pumps, "
                       f"validation level: {self.validation_level}")
                       
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete end-to-end validation
        
        Returns:
            Comprehensive validation results
        """
        
        logger.info("Starting comprehensive validation run")
        
        try:
            # Phase 1: Test API connectivity and system readiness
            self._run_system_readiness_tests()
            
            # Phase 2: Process each pump with batch retrieval and validation
            pump_results = {}
            for pump_serial in self.pump_serials:
                pump_results[pump_serial] = self._process_pump_validation(pump_serial)
                
            # Phase 3: Generate cross-pump analysis
            cross_pump_analysis = self._generate_cross_pump_analysis(pump_results)
            
            # Phase 4: Finalize session and generate reports
            self.metrics_collector.finalize_session()
            
            # Compile final results
            final_results = {
                "session_summary": {
                    "session_id": self.metrics_collector.session_id,
                    "session_name": self.session_name,
                    "validation_level": self.validation_level,
                    "pumps_processed": len(self.pump_serials),
                    "completion_time": datetime.now().isoformat()
                },
                "pump_results": pump_results,
                "cross_pump_analysis": cross_pump_analysis,
                "session_metrics": self.metrics_collector.session_metrics
            }
            
            # Save final results
            results_file = self.output_dir / f"{self.metrics_collector.session_id}_final_results.json"
            with open(results_file, 'w') as f:
                # Convert datetime objects for JSON serialization
                def convert_datetimes(obj):
                    if isinstance(obj, dict):
                        return {k: convert_datetimes(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_datetimes(item) for item in obj]
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                    
                json_results = convert_datetimes(final_results)
                json.dump(json_results, f, indent=2)
                
            logger.info(f"Comprehensive validation completed successfully")
            logger.info(f"Results saved to: {results_file}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            self.validation_logger.log_error(e, "comprehensive_validation")
            raise
            
    def _run_system_readiness_tests(self):
        """Run system readiness and connectivity tests"""
        
        logger.info("Running system readiness tests")
        
        # Test API connectivity
        api_test_result = self.test_framework.test_api_connectivity()
        
        if not api_test_result.get("success", False):
            raise Exception(f"API connectivity test failed: {api_test_result.get('error')}")
            
        # Test configuration validity
        config_test_result = self.test_framework.test_configuration_validity()
        
        if not config_test_result.get("success", False):
            raise Exception(f"Configuration test failed: {config_test_result.get('error')}")
            
        logger.info("System readiness tests passed")
        
    def _process_pump_validation(self, pump_serial: str) -> Dict[str, Any]:
        """
        Process complete validation for a single pump
        
        Args:
            pump_serial: Pump serial number to process
            
        Returns:
            Validation results for this pump
        """
        
        logger.info(f"Starting validation for pump {pump_serial}")
        
        # Create job context for this pump
        job_id = f"validation_{pump_serial}_{int(datetime.now().timestamp())}"
        
        with self.metrics_collector.batch_job(job_id, pump_serial, "full_validation") as job_metrics:
            
            # Stage 1: Batch Data Retrieval
            with self.metrics_collector.pipeline_stage("batch_retrieval") as stage:
                retrieval_results = self._run_batch_retrieval(pump_serial, stage)
                
            # Stage 2: Data Organization  
            with self.metrics_collector.pipeline_stage("data_organization") as stage:
                organization_results = self._run_data_organization(pump_serial, retrieval_results, stage)
                
            # Stage 3: Integrity Validation
            with self.metrics_collector.pipeline_stage("integrity_validation") as stage:
                validation_results = self._run_integrity_validation(pump_serial, organization_results, stage)
                
            # Compile pump results
            pump_results = {
                "pump_serial": pump_serial,
                "job_id": job_id,
                "retrieval_results": retrieval_results,
                "organization_results": organization_results,
                "validation_results": validation_results,
                "processing_summary": {
                    "total_records": job_metrics.total_records,
                    "data_size_mb": job_metrics.data_size_mb,
                    "validation_score": job_metrics.validation_score,
                    "processing_time_seconds": job_metrics.total_duration_seconds
                }
            }
            
            return pump_results
            
    def _run_batch_retrieval(self, pump_serial: str, stage_metrics) -> Dict[str, Any]:
        """Run batch data retrieval for a pump"""
        
        logger.info(f"Running batch retrieval for pump {pump_serial}")
        
        try:
            # Parse date range
            start_date = datetime.fromisoformat(self.date_range["start_date"])
            end_date = datetime.fromisoformat(self.date_range["end_date"])
            
            # Create batch schedule
            batch_schedule = self.batch_retriever.create_batch_schedule(
                pump_serial=pump_serial,
                start_date=start_date,
                end_date=end_date,
                batch_mode=self.batch_mode
            )
            
            stage_metrics.stage_specific_metrics["total_batches"] = len(batch_schedule.batch_jobs)
            
            # Execute batch schedule
            results = self.batch_retriever.execute_batch_schedule(batch_schedule)
            
            # Record metrics
            total_records = sum(r.get("records_retrieved", 0) for r in results.values())
            total_size_mb = sum(r.get("data_size_mb", 0) for r in results.values())
            
            self.metrics_collector.record_data_metrics(total_records, total_size_mb)
            
            stage_metrics.stage_specific_metrics.update({
                "batches_completed": len([r for r in results.values() if r.get("success", False)]),
                "total_records_retrieved": total_records,
                "total_size_mb": total_size_mb
            })
            
            return {
                "success": True,
                "batch_schedule": batch_schedule,
                "batch_results": results,
                "summary": {
                    "total_batches": len(batch_schedule.batch_jobs),
                    "successful_batches": len([r for r in results.values() if r.get("success", False)]),
                    "total_records": total_records,
                    "total_size_mb": total_size_mb
                }
            }
            
        except Exception as e:
            logger.error(f"Batch retrieval failed for pump {pump_serial}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _run_data_organization(self, pump_serial: str, retrieval_results: Dict[str, Any], stage_metrics) -> Dict[str, Any]:
        """Run data organization and merging"""
        
        if not retrieval_results.get("success", False):
            return {"success": False, "error": "Retrieval failed - skipping organization"}
            
        logger.info(f"Running data organization for pump {pump_serial}")
        
        try:
            batch_schedule = retrieval_results["batch_schedule"]
            
            # Merge chronological batches
            merged_df = self.data_organizer.merge_chronological_batches(
                batch_schedule=batch_schedule,
                pump_serial=pump_serial
            )
            
            stage_metrics.stage_specific_metrics["records_before_merge"] = len(merged_df)
            
            # Resample to 5-minute intervals
            resampled_df = self.data_organizer.resample_to_5min_intervals(merged_df)
            
            stage_metrics.stage_specific_metrics["records_after_resample"] = len(resampled_df)
            
            # Create LSTM-ready dataset
            lstm_df = self.data_organizer.create_lstm_ready_dataset(resampled_df)
            
            # Record data metrics
            data_size_mb = (lstm_df.memory_usage(deep=True).sum() / (1024 * 1024))
            self.metrics_collector.record_data_metrics(len(lstm_df), data_size_mb)
            
            return {
                "success": True,
                "merged_records": len(merged_df),
                "resampled_records": len(resampled_df),
                "lstm_records": len(lstm_df),
                "data_size_mb": data_size_mb,
                "lstm_dataset": lstm_df  # Pass to next stage
            }
            
        except Exception as e:
            logger.error(f"Data organization failed for pump {pump_serial}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _run_integrity_validation(self, pump_serial: str, organization_results: Dict[str, Any], stage_metrics) -> Dict[str, Any]:
        """Run comprehensive integrity validation"""
        
        if not organization_results.get("success", False):
            return {"success": False, "error": "Organization failed - skipping validation"}
            
        logger.info(f"Running integrity validation for pump {pump_serial}")
        
        try:
            lstm_dataset = organization_results["lstm_dataset"]
            
            # Run comprehensive validation
            validation_report = self.integrity_validator.validate_dataset(
                df=lstm_dataset,
                pump_serial=pump_serial,
                validation_level=self.validation_level
            )
            
            # Log validation results
            self.validation_logger.log_validation_result(validation_report)
            
            # Extract key metrics for stage
            validation_summary = validation_report.get("validation_summary", {})
            stage_metrics.stage_specific_metrics.update({
                "validation_score": validation_summary.get("overall_score", 0),
                "total_checks": validation_summary.get("total_checks", 0),
                "overall_status": validation_summary.get("overall_status", "unknown")
            })
            
            return {
                "success": True,
                "validation_report": validation_report
            }
            
        except Exception as e:
            logger.error(f"Integrity validation failed for pump {pump_serial}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _generate_cross_pump_analysis(self, pump_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-pump analysis and comparisons"""
        
        logger.info("Generating cross-pump analysis")
        
        successful_pumps = [
            pump_serial for pump_serial, results in pump_results.items()
            if results.get("validation_results", {}).get("success", False)
        ]
        
        if not successful_pumps:
            return {"error": "No successful validations for cross-pump analysis"}
            
        # Extract validation scores
        validation_scores = []
        completeness_scores = []
        
        for pump_serial in successful_pumps:
            validation_report = pump_results[pump_serial]["validation_results"]["validation_report"]
            
            validation_summary = validation_report.get("validation_summary", {})
            validation_scores.append(validation_summary.get("overall_score", 0))
            
            quality_metrics = validation_report.get("quality_metrics", {})
            completeness = quality_metrics.get("data_completeness", {})
            completeness_scores.append(completeness.get("cgm_completeness_pct", 0))
            
        # Calculate cross-pump statistics
        import numpy as np
        
        cross_pump_stats = {
            "total_pumps_analyzed": len(successful_pumps),
            "validation_score_statistics": {
                "mean": float(np.mean(validation_scores)),
                "median": float(np.median(validation_scores)),
                "std": float(np.std(validation_scores)),
                "min": float(np.min(validation_scores)),
                "max": float(np.max(validation_scores))
            },
            "completeness_statistics": {
                "mean": float(np.mean(completeness_scores)),
                "median": float(np.median(completeness_scores)),
                "std": float(np.std(completeness_scores)),
                "min": float(np.min(completeness_scores)),
                "max": float(np.max(completeness_scores))
            }
        }
        
        # Identify best and worst performing pumps
        pump_scores = list(zip(successful_pumps, validation_scores))
        pump_scores.sort(key=lambda x: x[1], reverse=True)
        
        cross_pump_stats["performance_ranking"] = {
            "best_pump": pump_scores[0][0] if pump_scores else None,
            "best_score": pump_scores[0][1] if pump_scores else None,
            "worst_pump": pump_scores[-1][0] if pump_scores else None,
            "worst_score": pump_scores[-1][1] if pump_scores else None
        }
        
        return cross_pump_stats


def create_sample_config():
    """Create a sample configuration file"""
    
    sample_config = {
        "validation": {
            "cgm_normal_range": [70, 180],
            "cgm_extreme_range": [40, 600],
            "basal_normal_range": [0.5, 3.0],
            "max_gap_minutes": 60,
            "min_continuity_pct": 85.0
        },
        "pump_serials": ["881235", "901161470"],
        "date_range": {
            "start_date": "2019-01-01T00:00:00",
            "end_date": "2024-01-01T00:00:00"
        },
        "batch_mode": "monthly",
        "validation_level": "comprehensive"
    }
    
    return sample_config


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Run comprehensive bloodBath validation")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--output-dir", type=Path, default="./validation_output", 
                       help="Output directory for results")
    parser.add_argument("--session-name", default="comprehensive_validation",
                       help="Session name for this validation run")
    parser.add_argument("--create-sample-config", action="store_true",
                       help="Create a sample configuration file")
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        sample_config = create_sample_config()
        config_file = Path("sample_validation_config.json")
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
            
        print(f"Sample configuration created: {config_file}")
        return
        
    if not args.config:
        print("Error: Configuration file required. Use --create-sample-config to generate one.")
        return
        
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return
        
    try:
        # Initialize and run validation
        runner = ComprehensiveValidationRunner(
            config_path=args.config,
            output_dir=args.output_dir,
            session_name=args.session_name
        )
        
        results = runner.run_complete_validation()
        
        print(f"\nValidation completed successfully!")
        print(f"Session ID: {runner.metrics_collector.session_id}")
        print(f"Results directory: {args.output_dir}")
        
        # Print summary
        session_summary = results["session_summary"]
        print(f"\nSummary:")
        print(f"  Pumps processed: {session_summary['pumps_processed']}")
        print(f"  Validation level: {session_summary['validation_level']}")
        
        if "cross_pump_analysis" in results:
            cross_analysis = results["cross_pump_analysis"]
            if "validation_score_statistics" in cross_analysis:
                score_stats = cross_analysis["validation_score_statistics"]
                print(f"  Average validation score: {score_stats['mean']:.1f}")
                print(f"  Score range: {score_stats['min']:.1f} - {score_stats['max']:.1f}")
                
    except Exception as e:
        logger.error(f"Validation run failed: {e}")
        print(f"Error: Validation failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()