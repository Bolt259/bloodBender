#!/usr/bin/env python3
"""
Comprehensive Data Integrity and Quality Validation System

Provides extensive data quality checks, continuity validation, physiological
range verification, and anomaly detection for pump data processing pipeline.

Features:
- Multi-level data quality assessment
- Physiological range validation with context-aware limits
- Temporal continuity and gap analysis
- Statistical anomaly detection 
- Cross-signal consistency checks
- Data drift and sensor degradation detection
- Comprehensive reporting and alerting
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationResult(NamedTuple):
    """Result of a single validation check"""
    check_name: str
    status: str  # "PASS", "WARN", "FAIL"
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    message: str
    details: Dict[str, Any] = {}


@dataclass
class QualityMetrics:
    """Comprehensive data quality metrics"""
    # Completeness metrics
    total_records: int = 0
    cgm_completeness_pct: float = 0.0
    basal_completeness_pct: float = 0.0
    bolus_coverage_pct: float = 0.0
    
    # Continuity metrics  
    max_cgm_gap_minutes: float = 0.0
    avg_cgm_gap_minutes: float = 0.0
    total_cgm_gaps: int = 0
    continuity_score: float = 0.0
    
    # Range validation metrics
    cgm_range_violations: int = 0
    basal_range_violations: int = 0
    bolus_range_violations: int = 0
    
    # Statistical quality metrics
    cgm_mean: float = 0.0
    cgm_std: float = 0.0
    cgm_coefficient_of_variation: float = 0.0
    
    # Anomaly detection metrics
    statistical_anomalies: int = 0
    physiological_anomalies: int = 0
    sensor_drift_detected: bool = False
    
    # Consistency metrics
    insulin_glucose_correlation: float = 0.0
    temporal_consistency_score: float = 0.0


@dataclass
class ValidationConfig:
    """Configuration for validation checks"""
    # Physiological ranges (can be customized per patient)
    cgm_normal_range: Tuple[float, float] = (70, 180)
    cgm_extreme_range: Tuple[float, float] = (40, 600)
    basal_normal_range: Tuple[float, float] = (0.5, 3.0)
    basal_extreme_range: Tuple[float, float] = (0.0, 10.0)
    bolus_normal_range: Tuple[float, float] = (0.5, 15.0)
    bolus_extreme_range: Tuple[float, float] = (0.0, 50.0)
    
    # Continuity thresholds
    max_acceptable_gap_minutes: int = 60
    min_continuity_percentage: float = 85.0
    
    # Anomaly detection sensitivity
    anomaly_z_threshold: float = 3.0
    anomaly_iqr_multiplier: float = 2.5
    
    # Temporal validation
    min_data_hours: int = 24
    max_cgm_rate_change: float = 10.0  # mg/dL per minute
    
    # Correlation thresholds
    min_insulin_glucose_correlation: float = -0.1  # Expected negative correlation
    

class ComprehensiveValidator:
    """
    Comprehensive data integrity and quality validation system
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator with configuration
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or ValidationConfig()
        self.validation_results: List[ValidationResult] = []
        self.quality_metrics = QualityMetrics()
        
    def validate_dataset(self, 
                        df: pd.DataFrame,
                        pump_serial: str,
                        validation_level: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive validation of a dataset
        
        Args:
            df: DataFrame to validate
            pump_serial: Pump serial number for context
            validation_level: "basic", "standard", or "comprehensive"
            
        Returns:
            Validation report dictionary
        """
        logger.info(f"Starting {validation_level} validation for pump {pump_serial}")
        
        # Reset validation state
        self.validation_results = []
        self.quality_metrics = QualityMetrics()
        
        if df.empty:
            self._add_result("dataset_empty", "FAIL", "CRITICAL", 
                           "Dataset is empty - no data to validate")
            return self._generate_validation_report(pump_serial)
            
        # Basic validation checks (always performed)
        self._validate_schema(df)
        self._validate_completeness(df)
        self._validate_physiological_ranges(df)
        
        if validation_level in ["standard", "comprehensive"]:
            # Standard validation checks
            self._validate_continuity(df)
            self._validate_temporal_consistency(df)
            self._validate_basic_statistics(df)
            
        if validation_level == "comprehensive":
            # Comprehensive validation checks
            self._validate_statistical_anomalies(df)
            self._validate_physiological_plausibility(df)
            self._validate_cross_signal_consistency(df)
            self._validate_sensor_degradation(df)
            self._validate_insulin_glucose_dynamics(df)
            
        # Calculate overall quality metrics
        self._calculate_quality_metrics(df)
        
        logger.info(f"Validation completed: {len(self.validation_results)} checks performed")
        return self._generate_validation_report(pump_serial)
        
    def _validate_schema(self, df: pd.DataFrame):
        """Validate DataFrame schema and required columns"""
        required_columns = ['timestamp', 'pump_serial']
        data_columns = ['bg_mgdl', 'basal_rate_u_hr', 'bolus_units']
        
        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            self._add_result("schema_required_columns", "FAIL", "CRITICAL",
                           f"Missing required columns: {missing_required}")
            return
            
        # Check data columns (at least one should be present)
        present_data_columns = [col for col in data_columns if col in df.columns]
        if not present_data_columns:
            self._add_result("schema_data_columns", "FAIL", "CRITICAL",
                           f"No data columns present. Expected: {data_columns}")
            return
            
        # Validate timestamp column
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                self._add_result("schema_timestamp", "FAIL", "HIGH",
                               "Timestamp column cannot be converted to datetime")
                return
                
        self._add_result("schema_validation", "PASS", "LOW",
                        f"Schema valid with columns: {present_data_columns}")
                        
    def _validate_completeness(self, df: pd.DataFrame):
        """Validate data completeness and calculate completeness metrics"""
        total_records = len(df)
        self.quality_metrics.total_records = total_records
        
        # CGM completeness
        if 'bg_mgdl' in df.columns:
            cgm_valid = df['bg_mgdl'].notna().sum()
            cgm_completeness = (cgm_valid / total_records) * 100
            self.quality_metrics.cgm_completeness_pct = cgm_completeness
            
            if cgm_completeness < self.config.min_continuity_percentage:
                self._add_result("completeness_cgm", "FAIL", "HIGH",
                               f"CGM completeness {cgm_completeness:.1f}% below threshold "
                               f"{self.config.min_continuity_percentage}%")
            elif cgm_completeness < 95.0:
                self._add_result("completeness_cgm", "WARN", "MEDIUM",
                               f"CGM completeness {cgm_completeness:.1f}% could be improved")
            else:
                self._add_result("completeness_cgm", "PASS", "LOW",
                               f"CGM completeness excellent: {cgm_completeness:.1f}%")
                               
        # Basal completeness  
        if 'basal_rate_u_hr' in df.columns:
            basal_valid = df['basal_rate_u_hr'].notna().sum()
            basal_completeness = (basal_valid / total_records) * 100
            self.quality_metrics.basal_completeness_pct = basal_completeness
            
            if basal_completeness < 90.0:
                self._add_result("completeness_basal", "WARN", "MEDIUM",
                               f"Basal completeness {basal_completeness:.1f}% is low")
            else:
                self._add_result("completeness_basal", "PASS", "LOW",
                               f"Basal completeness: {basal_completeness:.1f}%")
                               
        # Bolus coverage (different metric - presence not completeness)
        if 'bolus_units' in df.columns:
            bolus_present = (df['bolus_units'] > 0).sum()
            bolus_coverage = (bolus_present / total_records) * 100
            self.quality_metrics.bolus_coverage_pct = bolus_coverage
            
            # Bolus coverage should be much lower (few intervals have boluses)
            if bolus_coverage > 20.0:
                self._add_result("completeness_bolus", "WARN", "MEDIUM",
                               f"Unusually high bolus coverage: {bolus_coverage:.1f}%")
            else:
                self._add_result("completeness_bolus", "PASS", "LOW",
                               f"Bolus coverage normal: {bolus_coverage:.1f}%")
                               
    def _validate_physiological_ranges(self, df: pd.DataFrame):
        """Validate values are within physiological ranges"""
        
        # CGM range validation
        if 'bg_mgdl' in df.columns:
            cgm_values = df['bg_mgdl'].dropna()
            
            # Extreme range violations (impossible values)
            extreme_violations = (
                (cgm_values < self.config.cgm_extreme_range[0]) |
                (cgm_values > self.config.cgm_extreme_range[1])
            ).sum()
            
            self.quality_metrics.cgm_range_violations = extreme_violations
            
            if extreme_violations > 0:
                violation_pct = (extreme_violations / len(cgm_values)) * 100
                self._add_result("range_cgm_extreme", "FAIL", "HIGH",
                               f"{extreme_violations} CGM values ({violation_pct:.2f}%) "
                               f"outside extreme range {self.config.cgm_extreme_range}")
                               
            # Normal range assessment (informational)
            normal_violations = (
                (cgm_values < self.config.cgm_normal_range[0]) |
                (cgm_values > self.config.cgm_normal_range[1])
            ).sum()
            
            if normal_violations > 0:
                normal_pct = (normal_violations / len(cgm_values)) * 100
                if normal_pct > 50.0:
                    self._add_result("range_cgm_normal", "WARN", "MEDIUM",
                                   f"{normal_pct:.1f}% of CGM values outside normal range "
                                   f"{self.config.cgm_normal_range} - check glycemic control")
                else:
                    self._add_result("range_cgm_normal", "PASS", "LOW",
                                   f"{normal_pct:.1f}% of CGM values outside normal range")
                                   
        # Basal rate validation
        if 'basal_rate_u_hr' in df.columns:
            basal_values = df['basal_rate_u_hr'].dropna()
            
            extreme_violations = (
                (basal_values < self.config.basal_extreme_range[0]) |
                (basal_values > self.config.basal_extreme_range[1])
            ).sum()
            
            self.quality_metrics.basal_range_violations = extreme_violations
            
            if extreme_violations > 0:
                self._add_result("range_basal", "FAIL", "HIGH",
                               f"{extreme_violations} basal rate values outside "
                               f"extreme range {self.config.basal_extreme_range}")
            else:
                self._add_result("range_basal", "PASS", "LOW",
                               "All basal rates within valid range")
                               
        # Bolus validation
        if 'bolus_units' in df.columns:
            bolus_values = df[df['bolus_units'] > 0]['bolus_units']  # Only check non-zero boluses
            
            if not bolus_values.empty:
                extreme_violations = (
                    (bolus_values < self.config.bolus_extreme_range[0]) |
                    (bolus_values > self.config.bolus_extreme_range[1])
                ).sum()
                
                self.quality_metrics.bolus_range_violations = extreme_violations
                
                if extreme_violations > 0:
                    self._add_result("range_bolus", "FAIL", "HIGH",
                                   f"{extreme_violations} bolus values outside "
                                   f"extreme range {self.config.bolus_extreme_range}")
                else:
                    self._add_result("range_bolus", "PASS", "LOW",
                                   "All bolus doses within valid range")
                                   
    def _validate_continuity(self, df: pd.DataFrame):
        """Validate temporal continuity and identify gaps"""
        if 'timestamp' not in df.columns:
            return
            
        timestamps = pd.to_datetime(df['timestamp']).sort_values()
        
        # Calculate time differences between consecutive readings
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # Minutes
        
        # Expected interval is 5 minutes
        expected_interval = 5.0
        
        # Find significant gaps (more than 2x expected interval)
        significant_gaps = time_diffs[time_diffs > (expected_interval * 2)]
        
        if not significant_gaps.empty:
            max_gap = significant_gaps.max()
            avg_gap = significant_gaps.mean()
            total_gaps = len(significant_gaps)
            
            self.quality_metrics.max_cgm_gap_minutes = max_gap
            self.quality_metrics.avg_cgm_gap_minutes = avg_gap
            self.quality_metrics.total_cgm_gaps = total_gaps
            
            if max_gap > self.config.max_acceptable_gap_minutes:
                self._add_result("continuity_gaps", "FAIL", "HIGH",
                               f"Maximum gap {max_gap:.1f} minutes exceeds threshold "
                               f"{self.config.max_acceptable_gap_minutes} minutes")
            elif total_gaps > 10:
                self._add_result("continuity_gaps", "WARN", "MEDIUM",
                               f"{total_gaps} significant gaps detected, "
                               f"max: {max_gap:.1f}min, avg: {avg_gap:.1f}min")
            else:
                self._add_result("continuity_gaps", "PASS", "LOW",
                               f"{total_gaps} minor gaps detected")
        else:
            self._add_result("continuity_gaps", "PASS", "LOW",
                           "No significant gaps in timestamp continuity")
                           
    def _validate_temporal_consistency(self, df: pd.DataFrame):
        """Validate temporal consistency and detect timestamp issues"""
        if 'timestamp' not in df.columns:
            return
            
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Check for duplicate timestamps
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            self._add_result("temporal_duplicates", "WARN", "MEDIUM",
                           f"{duplicates} duplicate timestamps detected")
                           
        # Check for timestamps in the future
        now = pd.Timestamp.now(tz='UTC')
        future_timestamps = (timestamps > now).sum()
        if future_timestamps > 0:
            self._add_result("temporal_future", "WARN", "HIGH",
                           f"{future_timestamps} timestamps in the future")
                           
        # Check for very old timestamps (likely 1970 epoch errors)
        epoch_1970 = pd.Timestamp('1970-01-01', tz='UTC')
        epoch_2000 = pd.Timestamp('2000-01-01', tz='UTC')
        old_timestamps = (timestamps < epoch_2000).sum()
        if old_timestamps > 0:
            self._add_result("temporal_epoch", "FAIL", "CRITICAL",
                           f"{old_timestamps} timestamps before year 2000 - "
                           "likely epoch conversion errors")
                           
        # Check data span
        if len(timestamps) > 1:
            data_span_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600
            if data_span_hours < self.config.min_data_hours:
                self._add_result("temporal_span", "WARN", "MEDIUM",
                               f"Data span {data_span_hours:.1f} hours is less than "
                               f"minimum {self.config.min_data_hours} hours")
            else:
                self._add_result("temporal_span", "PASS", "LOW",
                               f"Data span: {data_span_hours:.1f} hours")
                               
    def _validate_basic_statistics(self, df: pd.DataFrame):
        """Validate basic statistical properties"""
        
        # CGM statistics
        if 'bg_mgdl' in df.columns:
            cgm_values = df['bg_mgdl'].dropna()
            
            if not cgm_values.empty:
                cgm_mean = cgm_values.mean()
                cgm_std = cgm_values.std()
                cgm_cv = (cgm_std / cgm_mean) * 100 if cgm_mean > 0 else 0
                
                self.quality_metrics.cgm_mean = cgm_mean
                self.quality_metrics.cgm_std = cgm_std
                self.quality_metrics.cgm_coefficient_of_variation = cgm_cv
                
                # Check for reasonable statistics
                if cgm_mean < 80 or cgm_mean > 200:
                    self._add_result("stats_cgm_mean", "WARN", "MEDIUM",
                                   f"CGM mean {cgm_mean:.1f} mg/dL is unusual")
                                   
                if cgm_cv > 50:
                    self._add_result("stats_cgm_variability", "WARN", "MEDIUM",
                                   f"CGM coefficient of variation {cgm_cv:.1f}% is high")
                elif cgm_cv < 5:
                    self._add_result("stats_cgm_variability", "WARN", "MEDIUM",
                                   f"CGM coefficient of variation {cgm_cv:.1f}% is unusually low")
                else:
                    self._add_result("stats_cgm", "PASS", "LOW",
                                   f"CGM statistics normal: mean={cgm_mean:.1f}, CV={cgm_cv:.1f}%")
                                   
    def _validate_statistical_anomalies(self, df: pd.DataFrame):
        """Detect statistical anomalies using multiple methods"""
        
        # CGM anomaly detection
        if 'bg_mgdl' in df.columns:
            cgm_values = df['bg_mgdl'].dropna()
            
            if len(cgm_values) > 10:
                # Z-score based anomalies
                z_scores = np.abs(stats.zscore(cgm_values))
                z_anomalies = (z_scores > self.config.anomaly_z_threshold).sum()
                
                # IQR based anomalies
                q1 = cgm_values.quantile(0.25)
                q3 = cgm_values.quantile(0.75)
                iqr = q3 - q1
                iqr_lower = q1 - (self.config.anomaly_iqr_multiplier * iqr)
                iqr_upper = q3 + (self.config.anomaly_iqr_multiplier * iqr)
                iqr_anomalies = ((cgm_values < iqr_lower) | (cgm_values > iqr_upper)).sum()
                
                total_anomalies = max(z_anomalies, iqr_anomalies)
                self.quality_metrics.statistical_anomalies = total_anomalies
                
                anomaly_pct = (total_anomalies / len(cgm_values)) * 100
                
                if anomaly_pct > 5.0:
                    self._add_result("anomalies_statistical", "WARN", "MEDIUM",
                                   f"{total_anomalies} statistical anomalies ({anomaly_pct:.2f}%) detected")
                else:
                    self._add_result("anomalies_statistical", "PASS", "LOW",
                                   f"{total_anomalies} statistical anomalies ({anomaly_pct:.2f}%) - normal range")
                                   
    def _validate_physiological_plausibility(self, df: pd.DataFrame):
        """Validate physiological plausibility of glucose changes"""
        
        if 'bg_mgdl' in df.columns and 'timestamp' in df.columns:
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            cgm_values = df_sorted['bg_mgdl'].dropna()
            
            if len(cgm_values) > 1:
                # Calculate rate of change (mg/dL per minute)
                timestamps = pd.to_datetime(df_sorted.loc[cgm_values.index, 'timestamp'])
                time_diffs = timestamps.diff().dt.total_seconds() / 60  # Minutes
                glucose_diffs = cgm_values.diff()
                
                # Only calculate where time diff is reasonable (5-15 minutes)
                valid_intervals = (time_diffs >= 4) & (time_diffs <= 16)
                
                if valid_intervals.sum() > 0:
                    rates = (glucose_diffs / time_diffs)[valid_intervals]
                    extreme_rates = (np.abs(rates) > self.config.max_cgm_rate_change)
                    
                    extreme_count = extreme_rates.sum()
                    self.quality_metrics.physiological_anomalies = extreme_count
                    
                    if extreme_count > 0:
                        max_rate = np.abs(rates).max()
                        self._add_result("plausibility_glucose_rate", "WARN", "HIGH",
                                       f"{extreme_count} glucose changes exceed {self.config.max_cgm_rate_change} mg/dL/min "
                                       f"(max rate: {max_rate:.2f} mg/dL/min)")
                    else:
                        self._add_result("plausibility_glucose_rate", "PASS", "LOW",
                                       "All glucose rate changes within physiological limits")
                                       
    def _validate_cross_signal_consistency(self, df: pd.DataFrame):
        """Validate consistency between glucose, basal, and bolus signals"""
        
        # Check insulin-glucose correlation
        if 'bg_mgdl' in df.columns and 'bolus_units' in df.columns:
            # Calculate correlation with time-shifted insulin (insulin acts with delay)
            df_corr = df.copy()
            
            # Shift bolus forward by 30-60 minutes to account for action delay
            for shift_minutes in [30, 45, 60]:
                shift_periods = shift_minutes // 5  # Convert to 5-minute periods
                bolus_shifted = df_corr['bolus_units'].shift(-shift_periods)
                
                correlation = df_corr['bg_mgdl'].corr(bolus_shifted)
                
                if not np.isnan(correlation):
                    self.quality_metrics.insulin_glucose_correlation = correlation
                    break
                    
            if not np.isnan(self.quality_metrics.insulin_glucose_correlation):
                # Expected negative correlation (more insulin -> lower glucose eventually)
                if self.quality_metrics.insulin_glucose_correlation > 0.2:
                    self._add_result("consistency_insulin_glucose", "WARN", "MEDIUM",
                                   f"Unexpected positive insulin-glucose correlation: "
                                   f"{self.quality_metrics.insulin_glucose_correlation:.3f}")
                else:
                    self._add_result("consistency_insulin_glucose", "PASS", "LOW",
                                   f"Insulin-glucose correlation: {self.quality_metrics.insulin_glucose_correlation:.3f}")
                                   
    def _validate_sensor_degradation(self, df: pd.DataFrame):
        """Detect potential CGM sensor degradation patterns"""
        
        if 'bg_mgdl' in df.columns and 'timestamp' in df.columns:
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            cgm_values = df_sorted['bg_mgdl'].dropna()
            
            if len(cgm_values) > 144:  # At least 12 hours of data
                # Split into time windows and check for drift
                window_size = 144  # 12 hours in 5-minute intervals
                windows = []
                
                for i in range(0, len(cgm_values) - window_size, window_size):
                    window = cgm_values.iloc[i:i+window_size]
                    windows.append({
                        'mean': window.mean(),
                        'std': window.std(),
                        'range': window.max() - window.min()
                    })
                    
                if len(windows) >= 3:
                    # Check for systematic drift in mean values
                    means = [w['mean'] for w in windows]
                    stds = [w['std'] for w in windows]
                    
                    # Linear trend in means (drift detection)
                    x = np.arange(len(means))
                    slope, _, r_value, _, _ = stats.linregress(x, means)
                    
                    # Check for significant drift (>5 mg/dL per 12-hour window)
                    if abs(slope) > 5.0:
                        self.quality_metrics.sensor_drift_detected = True
                        self._add_result("sensor_drift", "WARN", "HIGH",
                                       f"Potential sensor drift detected: {slope:.2f} mg/dL per 12h window")
                                       
                    # Check for decreasing variability (sensor degradation)
                    std_slope, _, _, _, _ = stats.linregress(x, stds)
                    if std_slope < -2.0:  # Decreasing std by >2 mg/dL per window
                        self._add_result("sensor_degradation", "WARN", "MEDIUM",
                                       f"Potential sensor degradation: decreasing variability over time")
                    else:
                        self._add_result("sensor_quality", "PASS", "LOW",
                                       "No significant sensor drift or degradation detected")
                                       
    def _validate_insulin_glucose_dynamics(self, df: pd.DataFrame):
        """Validate expected insulin-glucose dynamics and pharmacokinetics"""
        
        if not all(col in df.columns for col in ['bg_mgdl', 'bolus_units', 'basal_rate_u_hr']):
            return
            
        # Check for reasonable insulin action patterns
        # Look at glucose response after significant boluses
        bolus_events = df[df['bolus_units'] >= 2.0]  # Significant boluses only
        
        if len(bolus_events) >= 5:
            glucose_responses = []
            
            for idx, bolus_event in bolus_events.iterrows():
                # Get glucose values 2-4 hours after bolus
                bolus_time = pd.to_datetime(bolus_event['timestamp'])
                response_window = df[
                    (pd.to_datetime(df['timestamp']) > bolus_time) &
                    (pd.to_datetime(df['timestamp']) <= bolus_time + timedelta(hours=4))
                ]
                
                if len(response_window) >= 12:  # At least 1 hour of post-bolus data
                    pre_bolus_bg = bolus_event['bg_mgdl']
                    min_post_bg = response_window['bg_mgdl'].min()
                    
                    if not np.isnan(pre_bolus_bg) and not np.isnan(min_post_bg):
                        glucose_drop = pre_bolus_bg - min_post_bg
                        glucose_responses.append(glucose_drop)
                        
            if glucose_responses:
                avg_response = np.mean(glucose_responses)
                
                # Expected some glucose reduction after bolus
                if avg_response < 5:
                    self._add_result("insulin_dynamics", "WARN", "MEDIUM",
                                   f"Minimal glucose response to boluses: avg {avg_response:.1f} mg/dL drop")
                elif avg_response > 100:
                    self._add_result("insulin_dynamics", "WARN", "HIGH",
                                   f"Excessive glucose response to boluses: avg {avg_response:.1f} mg/dL drop")
                else:
                    self._add_result("insulin_dynamics", "PASS", "LOW",
                                   f"Normal glucose response to boluses: avg {avg_response:.1f} mg/dL drop")
                                   
    def _calculate_quality_metrics(self, df: pd.DataFrame):
        """Calculate overall quality metrics and scores"""
        
        # Calculate continuity score (weighted combination)
        continuity_components = []
        
        if self.quality_metrics.cgm_completeness_pct > 0:
            continuity_components.append(self.quality_metrics.cgm_completeness_pct * 0.7)  # CGM most important
            
        if self.quality_metrics.basal_completeness_pct > 0:
            continuity_components.append(self.quality_metrics.basal_completeness_pct * 0.2)  # Basal important
            
        if self.quality_metrics.bolus_coverage_pct > 0:
            # Normalize bolus coverage (5% coverage is good, 20% is excellent)
            normalized_bolus = min(self.quality_metrics.bolus_coverage_pct * 5, 100)
            continuity_components.append(normalized_bolus * 0.1)
            
        if continuity_components:
            self.quality_metrics.continuity_score = sum(continuity_components) / len(continuity_components)
            
        # Calculate temporal consistency score
        consistency_penalties = 0
        for result in self.validation_results:
            if result.check_name.startswith('temporal_') and result.status in ['WARN', 'FAIL']:
                consistency_penalties += (1 if result.status == 'WARN' else 3)
                
        self.quality_metrics.temporal_consistency_score = max(0, 100 - (consistency_penalties * 10))
        
    def _add_result(self, check_name: str, status: str, severity: str, message: str, details: Dict[str, Any] = None):
        """Add a validation result"""
        result = ValidationResult(
            check_name=check_name,
            status=status,
            severity=severity,
            message=message,
            details=details or {}
        )
        self.validation_results.append(result)
        
    def _generate_validation_report(self, pump_serial: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Count results by status
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for result in self.validation_results:
            status_counts[result.status] += 1
            severity_counts[result.severity] += 1
            
        # Calculate overall score
        total_checks = len(self.validation_results)
        if total_checks > 0:
            pass_rate = (status_counts['PASS'] / total_checks) * 100
            fail_rate = (status_counts['FAIL'] / total_checks) * 100
            
            # Overall score (0-100)
            overall_score = pass_rate - (fail_rate * 2)  # Failures penalized more
            overall_score = max(0, min(100, overall_score))
        else:
            overall_score = 0
            
        # Determine overall status
        if status_counts['FAIL'] > 0 and severity_counts['CRITICAL'] > 0:
            overall_status = "CRITICAL_FAILURE"
        elif status_counts['FAIL'] > 0:
            overall_status = "FAILURE"
        elif status_counts['WARN'] > 0 and severity_counts['HIGH'] > 0:
            overall_status = "WARNING_HIGH"
        elif status_counts['WARN'] > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
            
        report = {
            "validation_summary": {
                "pump_serial": pump_serial,
                "overall_status": overall_status,
                "overall_score": overall_score,
                "total_checks": total_checks,
                "validation_timestamp": datetime.now().isoformat()
            },
            "status_summary": dict(status_counts),
            "severity_summary": dict(severity_counts),
            "quality_metrics": {
                "data_completeness": {
                    "total_records": self.quality_metrics.total_records,
                    "cgm_completeness_pct": self.quality_metrics.cgm_completeness_pct,
                    "basal_completeness_pct": self.quality_metrics.basal_completeness_pct,
                    "bolus_coverage_pct": self.quality_metrics.bolus_coverage_pct
                },
                "data_continuity": {
                    "max_gap_minutes": self.quality_metrics.max_cgm_gap_minutes,
                    "avg_gap_minutes": self.quality_metrics.avg_cgm_gap_minutes,
                    "total_gaps": self.quality_metrics.total_cgm_gaps,
                    "continuity_score": self.quality_metrics.continuity_score
                },
                "data_quality": {
                    "cgm_mean": self.quality_metrics.cgm_mean,
                    "cgm_std": self.quality_metrics.cgm_std,
                    "cgm_cv": self.quality_metrics.cgm_coefficient_of_variation,
                    "range_violations": self.quality_metrics.cgm_range_violations,
                    "statistical_anomalies": self.quality_metrics.statistical_anomalies,
                    "physiological_anomalies": self.quality_metrics.physiological_anomalies,
                    "sensor_drift_detected": self.quality_metrics.sensor_drift_detected
                },
                "signal_consistency": {
                    "insulin_glucose_correlation": self.quality_metrics.insulin_glucose_correlation,
                    "temporal_consistency_score": self.quality_metrics.temporal_consistency_score
                }
            },
            "detailed_results": [
                {
                    "check": result.check_name,
                    "status": result.status,
                    "severity": result.severity,
                    "message": result.message,
                    "details": result.details
                }
                for result in self.validation_results
            ]
        }
        
        return report