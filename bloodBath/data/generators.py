"""
Data Generators for bloodBath

This module contains data generation utilities including monthly LSTM generators,
continuous sync generators, and other specialized data processing generators.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..utils.logging_utils import setup_logger
from .processors import UnifiedDataProcessor
from .validators import LstmDataValidator


class ContinuityChecker:
    """
    Data continuity analysis and quality checking
    
    Performs comprehensive analysis of LSTM-ready data to identify:
    - Time gaps larger than threshold 
    - Missing data periods
    - Unusual basal insulin patterns
    - Data quality issues
    """
    
    def __init__(self, max_gap_hours: float = 18.0):
        """
        Initialize continuity checker
        
        Args:
            max_gap_hours: Maximum acceptable gap in hours
        """
        self.max_gap_hours = max_gap_hours
        self.logger = logging.getLogger(__name__)
    
    def check_time_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for time gaps larger than specified threshold
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Dictionary with gap analysis results
        """
        self.logger.info(f"🔍 Checking time continuity (max gap: {self.max_gap_hours}h)")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Calculate time differences
        time_diffs = df['timestamp'].diff()
        
        # Find gaps larger than threshold
        max_gap = timedelta(hours=self.max_gap_hours)
        large_gaps = time_diffs[time_diffs > max_gap]
        
        gaps = []
        if len(large_gaps) > 0:
            for idx in large_gaps.index:
                gap_start = df.loc[idx-1, 'timestamp']
                gap_end = df.loc[idx, 'timestamp']
                gap_duration = time_diffs.loc[idx]
                gap_hours = gap_duration.total_seconds() / 3600
                
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hours': gap_hours,
                    'duration_str': str(gap_duration)
                })
        
        # Calculate overall statistics
        continuity_stats = {
            'total_records': len(df),
            'total_gaps_found': len(large_gaps),
            'median_interval': time_diffs.median(),
            'mean_interval': time_diffs.mean(),
            'max_gap': time_diffs.max(),
            'min_gap': time_diffs.min(),
            'gaps': gaps
        }
        
        self.logger.info(f"   Found {len(large_gaps)} gaps > {self.max_gap_hours}h")
        
        return continuity_stats
    
    def analyze_basal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze basal insulin patterns for anomalies
        
        Args:
            df: DataFrame with basal_rate column
            
        Returns:
            Dictionary with basal analysis results
        """
        if 'basal_rate' not in df.columns:
            return {'error': 'No basal_rate column found'}
        
        self.logger.info("💉 Analyzing basal insulin patterns")
        
        # Basic basal statistics
        basal_stats = df['basal_rate'].describe()
        
        # Check for unusual values
        high_values = df[df['basal_rate'] > 5.0]
        suspicious_values = df[df['basal_rate'].isin([21.0, 10.0])]
        zero_values = df[df['basal_rate'] == 0.0]
        
        # Check for rapid basal changes
        basal_changes = df['basal_rate'].diff().abs()
        large_changes = basal_changes[basal_changes > 2.0] if not basal_changes.empty else pd.Series(dtype=float)
        
        basal_analysis = {
            'basic_stats': basal_stats.to_dict(),
            'high_values_count': len(high_values),
            'suspicious_values_count': len(suspicious_values),
            'zero_values_count': len(zero_values),
            'large_changes_count': len(large_changes),
            'max_change': basal_changes.max() if len(basal_changes) > 0 else 0
        }
        
        self.logger.info(f"   Suspicious values: {len(suspicious_values)}")
        self.logger.info(f"   Large changes: {len(large_changes)}")
        
        return basal_analysis
    
    def analyze_glucose_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze blood glucose patterns and quality
        
        Args:
            df: DataFrame with bg column
            
        Returns:
            Dictionary with glucose analysis results
        """
        if 'bg' not in df.columns:
            return {'error': 'No bg column found'}
        
        self.logger.info("🩸 Analyzing glucose patterns")
        
        # Basic glucose statistics
        bg_stats = df['bg'].describe()
        
        # Check for out-of-range values
        low_bg = df[df['bg'] < 70]
        high_bg = df[df['bg'] > 300]
        very_high_bg = df[df['bg'] > 400]
        
        # Check glucose rate of change
        if 'delta_bg' in df.columns:
            glucose_roc = df['delta_bg'].abs()
            rapid_changes = glucose_roc[glucose_roc > 50]
            max_roc = glucose_roc.max() if len(rapid_changes) > 0 else 0
        else:
            rapid_changes = pd.Series(dtype=float)
            max_roc = 0

        glucose_analysis = {
            'basic_stats': bg_stats.to_dict(),
            'low_glucose_count': len(low_bg),
            'high_glucose_count': len(high_bg),
            'very_high_glucose_count': len(very_high_bg),
            'rapid_changes_count': len(rapid_changes),
            'max_rate_of_change': max_roc
        }
        
        self.logger.info(f"   High glucose (>300): {len(high_bg)} records")
        self.logger.info(f"   Low glucose (<70): {len(low_bg)} records")
        
        return glucose_analysis
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Comprehensive quality analysis
        """
        self.logger.info("📋 Generating comprehensive quality report")
        
        # Run all analyses
        continuity_results = self.check_time_continuity(df)
        basal_results = self.analyze_basal_patterns(df)
        glucose_results = self.analyze_glucose_patterns(df)
        
        # Overall assessment
        issues_found = []
        
        if continuity_results['total_gaps_found'] > 0:
            issues_found.append(f"{continuity_results['total_gaps_found']} time gaps >{self.max_gap_hours}h")
        
        if basal_results.get('suspicious_values_count', 0) > 0:
            issues_found.append(f"{basal_results['suspicious_values_count']} suspicious basal values")
        
        if glucose_results.get('very_high_glucose_count', 0) > 0:
            issues_found.append(f"{glucose_results['very_high_glucose_count']} glucose readings >400 mg/dL")
        
        # Data completeness
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            total_duration = df['timestamp'].max() - df['timestamp'].min()
            expected_records = total_duration.total_seconds() / 300  # 5-minute intervals
            completeness = len(df) / expected_records * 100 if expected_records > 0 else 0
        else:
            total_duration = timedelta(0)
            completeness = 0
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
                'duration_days': total_duration.days
            },
            'data_completeness_percent': completeness,
            'continuity_analysis': continuity_results,
            'basal_analysis': basal_results,
            'glucose_analysis': glucose_results,
            'issues_found': issues_found,
            'overall_quality': 'GOOD' if len(issues_found) == 0 else 'NEEDS_ATTENTION'
        }
        
        self.logger.info(f"Quality assessment: {summary['overall_quality']}")
        
        return summary


class LSTMDataGenerator:
    """
    LSTM-ready data generator with quality validation
    
    Combines raw pump data into LSTM-ready format with comprehensive
    quality checks and validation.
    """
    
    def __init__(self, 
                 output_dir: str = "/home/bolt/projects/bb/training_data",
                 enable_validation: bool = True):
        """
        Initialize LSTM data generator
        
        Args:
            output_dir: Output directory for generated files
            enable_validation: Enable quality validation
        """
        self.output_dir = Path(output_dir)
        self.enable_validation = enable_validation
        
        # Setup logging
        log_file = self.output_dir / "logs" / f"lstm_generator_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('lstm_generator', log_file=log_file)
        
        # Initialize processors
        self.processor = UnifiedDataProcessor()
        self.validator = LstmDataValidator() if enable_validation else None
        self.continuity_checker = ContinuityChecker()
        
        # Physiological limits
        self.BASAL_HARD_MAX = 8.0
        self.BASAL_HARD_MIN = 0.0
        self.BG_HARD_MAX = 600
        self.BG_HARD_MIN = 20
    
    def generate_lstm_dataset(self,
                            source_data: List[Dict],
                            pump_serial: str,
                            output_filename: str,
                            max_gap_hours: int = 18,
                            min_segment_length: int = 24) -> Optional[Dict[str, Any]]:
        """
        Generate LSTM-ready dataset from source data
        
        Args:
            source_data: List of data records 
            pump_serial: Pump serial number
            output_filename: Output CSV filename
            max_gap_hours: Maximum gap for processing
            min_segment_length: Minimum segment length
            
        Returns:
            Generation statistics or None if failed
        """
        self.logger.info(f"🔄 Generating LSTM dataset: {output_filename}")
        
        try:
            # Convert source data to DataFrame for processing
            if isinstance(source_data, list) and source_data:
                df = pd.DataFrame(source_data)
            else:
                self.logger.warning("No valid source data provided")
                return None
            
            # Basic data validation
            if len(df) == 0:
                self.logger.warning("Empty dataset provided")
                return None
            
            # Apply outlier handling
            processed_data = self._apply_outlier_handling(df)
            
            # Run quality analysis
            quality_report = self.continuity_checker.generate_quality_report(processed_data)
            
            # Validate if enabled
            validation_result = None
            validation_stats = None
            if self.enable_validation and self.validator:
                validated_df, validation_stats = self.validator.validate_dataframe(processed_data)
                validation_result = validation_stats
                # Use validation results to assess quality but don't block saving
            
            # Save the dataset
            output_file = self.output_dir / output_filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create header with metadata
            header_lines = [
                "# bloodBath LSTM Training Dataset",
                f"# Pump: {pump_serial}",
                f"# Generated: {datetime.now().isoformat()}",
                f"# Records: {len(processed_data)}",
                f"# Processing: Unified Pipeline + Outlier Handling",
                f"# Max Gap Hours: {max_gap_hours}",
                f"# Min Segment Length: {min_segment_length}",
            ]
            
            if validation_result:
                header_lines.extend([
                    f"# Validation: {validation_result.get('status', 'UNKNOWN')}",
                    f"# Validation Details: {validation_result.get('summary', 'N/A')}"
                ])
            
            if quality_report:
                header_lines.extend([
                    f"# Quality: {quality_report['overall_quality']}",
                    f"# Data Completeness: {quality_report['data_completeness_percent']:.1f}%",
                    f"# Issues: {len(quality_report['issues_found'])}"
                ])
            
            header_lines.append("")
            
            # Write file
            with open(output_file, 'w') as f:
                f.write('\n'.join(header_lines))
                processed_data.to_csv(f, index=False)
            
            self.logger.info(f"✅ Generated {len(processed_data)} records in {output_file}")
            
            # Return statistics
            stats = {
                'output_file': str(output_file),
                'records': len(processed_data),
                'pump_serial': pump_serial,
                'quality_report': quality_report,
                'validation_result': validation_result if validation_result else None,
                'processing_params': {
                    'max_gap_hours': max_gap_hours,
                    'min_segment_length': min_segment_length
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"LSTM generation failed: {e}")
            return None
    
    def _apply_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physiological limits and outlier flagging"""
        self.logger.info("🔧 Applying outlier handling")
        
        # Create copy
        processed_df = df.copy()
        
        # Track original values
        if 'bg' in processed_df.columns:
            processed_df['bg_original'] = processed_df['bg']
            
            # Apply BG limits
            bg_clipped = np.clip(processed_df['bg'], self.BG_HARD_MIN, self.BG_HARD_MAX)
            processed_df['bg_anomaly_flag'] = (processed_df['bg'] != bg_clipped).astype(int)
            processed_df['bg'] = bg_clipped
        
        if 'basal_rate' in processed_df.columns:
            processed_df['basal_original'] = processed_df['basal_rate']
            
            # Apply basal limits  
            basal_clipped = np.clip(processed_df['basal_rate'], self.BASAL_HARD_MIN, self.BASAL_HARD_MAX)
            processed_df['basal_anomaly_flag'] = (processed_df['basal_rate'] != basal_clipped).astype(int)
            processed_df['basal_rate'] = basal_clipped
            
            # Statistical outlier detection
            if len(processed_df) > 1:
                basal_mean = processed_df['basal_rate'].mean()
                basal_std = processed_df['basal_rate'].std()
                if basal_std > 0:
                    basal_z_scores = np.abs((processed_df['basal_rate'] - basal_mean) / basal_std)
                    processed_df['basal_statistical_outlier'] = (basal_z_scores > 3).astype(int)
                else:
                    processed_df['basal_statistical_outlier'] = 0
            else:
                processed_df['basal_statistical_outlier'] = 0
        
        # Log outlier statistics
        bg_anomalies = processed_df.get('bg_anomaly_flag', pd.Series(dtype=int)).sum()
        basal_anomalies = processed_df.get('basal_anomaly_flag', pd.Series(dtype=int)).sum()
        
        self.logger.info(f"   BG anomalies: {bg_anomalies}")
        self.logger.info(f"   Basal anomalies: {basal_anomalies}")
        
        return processed_df
    
    def batch_generate(self,
                      data_sources: List[Dict[str, Any]],
                      output_prefix: str = "lstm_batch") -> List[Dict[str, Any]]:
        """
        Generate multiple LSTM datasets from a batch of data sources
        
        Args:
            data_sources: List of source data configurations
            output_prefix: Prefix for output filenames
            
        Returns:
            List of generation statistics for each dataset
        """
        self.logger.info(f"🔄 Batch generating {len(data_sources)} LSTM datasets")
        
        results = []
        
        for i, source_config in enumerate(data_sources, 1):
            self.logger.info(f"\n📊 Processing batch {i}/{len(data_sources)}")
            
            try:
                # Extract configuration
                source_data = source_config['data']
                pump_serial = source_config['pump_serial']
                suffix = source_config.get('suffix', f'{i:03d}')
                
                output_filename = f"{output_prefix}_{pump_serial}_{suffix}.csv"
                
                # Generate dataset
                stats = self.generate_lstm_dataset(
                    source_data,
                    pump_serial,
                    output_filename,
                    max_gap_hours=source_config.get('max_gap_hours', 18),
                    min_segment_length=source_config.get('min_segment_length', 24)
                )
                
                if stats:
                    stats['batch_index'] = i
                    results.append(stats)
                else:
                    self.logger.warning(f"   Failed to generate dataset for batch {i}")
                    
            except Exception as e:
                self.logger.error(f"   Batch {i} failed: {e}")
        
        # Summary
        successful = len(results)
        failed = len(data_sources) - successful
        
        self.logger.info(f"\n📊 Batch generation complete:")
        self.logger.info(f"   Successful: {successful}")
        self.logger.info(f"   Failed: {failed}")
        
        return results