#!/usr/bin/env python3
"""
Enhanced Data Organization and Preprocessing Pipeline

Provides chronological merging, resampling to precise 5-minute intervals,
and multi-signal synchronization for LSTM-ready dataset preparation.

Features:
- Chronological merging of batch data with overlap detection
- Precise 5-minute resampling with configurable aggregation rules
- Multi-signal synchronization (CGM, basal, bolus) on unified timeline  
- Gap detection and intelligent interpolation strategies
- Data quality validation and anomaly detection
- Efficient processing of large multi-year datasets
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
from collections import defaultdict

from ..utils.logging_utils import get_logger
from .batch_retriever import EnhancedBatchRetriever

logger = get_logger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class ProcessingMetrics:
    """Metrics for data organization and preprocessing"""
    input_records: int = 0
    output_records: int = 0
    duplicates_removed: int = 0
    gaps_filled: int = 0
    interpolated_values: int = 0
    invalid_values_filtered: int = 0
    processing_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    resampling_ratio: float = 0.0
    continuity_score: float = 0.0


@dataclass
class ResamplingConfig:
    """Configuration for resampling operations"""
    target_frequency: str = "5min"  # Target resampling frequency
    cgm_aggregation: str = "mean"   # How to aggregate CGM values: mean, median, last
    basal_aggregation: str = "mean" # How to aggregate basal rates
    bolus_aggregation: str = "sum"  # How to aggregate bolus doses
    max_gap_minutes: int = 30       # Maximum gap to fill with forward-fill
    interpolation_method: str = "linear"  # Interpolation method for gaps
    timestamp_tolerance: str = "2.5min"   # Tolerance for timestamp alignment
    

class EnhancedDataOrganizer:
    """
    Enhanced data organization pipeline for multi-year pump data processing
    """
    
    # Physiological constraints for validation
    CGM_VALID_RANGE = (40, 600)      # mg/dL
    BASAL_VALID_RANGE = (0.0, 10.0)  # U/hr  
    BOLUS_VALID_RANGE = (0.0, 50.0)  # U
    
    def __init__(self, 
                 batch_retriever: EnhancedBatchRetriever,
                 resampling_config: Optional[ResamplingConfig] = None):
        """
        Initialize enhanced data organizer
        
        Args:
            batch_retriever: Enhanced batch retriever instance
            resampling_config: Configuration for resampling operations
        """
        self.batch_retriever = batch_retriever
        self.config = resampling_config or ResamplingConfig()
        
        # Processing state
        self.metrics = ProcessingMetrics()
        self.processing_cache: Dict[str, pd.DataFrame] = {}
        
    def merge_chronological_batches(self, 
                                  pump_serial: str,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Merge all batch data chronologically with overlap detection
        
        Args:
            pump_serial: Pump serial number
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Chronologically merged DataFrame
        """
        logger.info(f"Merging chronological batches for pump {pump_serial}")
        
        # Get schedule for this pump
        if pump_serial not in self.batch_retriever.active_schedules:
            raise ValueError(f"No active schedule found for pump {pump_serial}")
            
        schedule = self.batch_retriever.active_schedules[pump_serial]
        
        # Load completed batch data
        batch_dataframes = []
        total_input_records = 0
        
        for job in schedule.jobs:
            if job.status != "COMPLETED":
                logger.warning(f"Skipping incomplete job: {job.job_id}")
                continue
                
            # Apply date filtering
            if start_date and job.end_date < start_date:
                continue
            if end_date and job.start_date > end_date:
                continue
                
            # Load batch data
            batch_df = self.batch_retriever.load_batch_data(job.job_id)
            
            if batch_df is not None and not batch_df.empty:
                # Add batch metadata
                batch_df['batch_id'] = job.job_id
                batch_df['pump_serial'] = pump_serial
                
                batch_dataframes.append(batch_df)
                total_input_records += len(batch_df)
                logger.debug(f"Loaded batch {job.job_id}: {len(batch_df)} records")
                
        if not batch_dataframes:
            logger.warning(f"No batch data found for pump {pump_serial}")
            return pd.DataFrame()
            
        self.metrics.input_records = total_input_records
        
        # Combine all batches
        logger.info(f"Combining {len(batch_dataframes)} batches...")
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        
        # Ensure timestamp column is properly formatted
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], utc=True)
        else:
            raise ValueError("No timestamp column found in batch data")
            
        # Sort chronologically
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates (overlapping batches)
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['timestamp', 'event_type'], 
            keep='last'  # Keep latest version in case of overlaps
        ).reset_index(drop=True)
        
        duplicates_removed = initial_count - len(combined_df)
        self.metrics.duplicates_removed = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records from overlapping batches")
            
        # Apply date range filtering if specified
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]
            
        logger.info(f"Chronological merge complete: {len(combined_df)} records")
        return combined_df
        
    def resample_to_5min_intervals(self, 
                                 merged_df: pd.DataFrame,
                                 pump_serial: str) -> pd.DataFrame:
        """
        Resample multi-signal data to precise 5-minute intervals
        
        Args:
            merged_df: Chronologically merged DataFrame
            pump_serial: Pump serial number for context
            
        Returns:
            Resampled DataFrame with synchronized signals
        """
        logger.info(f"Resampling pump {pump_serial} data to {self.config.target_frequency} intervals")
        
        if merged_df.empty:
            return pd.DataFrame()
            
        # Separate by event type
        event_types = merged_df['event_type'].unique()
        logger.debug(f"Found event types: {list(event_types)}")
        
        # Create full time range at target frequency
        start_time = merged_df['timestamp'].min()
        end_time = merged_df['timestamp'].max()
        
        # Round to nearest 5-minute boundary for consistency
        start_time = start_time.floor(self.config.target_frequency)
        end_time = end_time.ceil(self.config.target_frequency)
        
        full_time_index = pd.date_range(
            start=start_time,
            end=end_time, 
            freq=self.config.target_frequency,
            tz='UTC'
        )
        
        logger.info(f"Target time range: {start_time} to {end_time} "
                   f"({len(full_time_index)} intervals)")
        
        # Initialize result DataFrame
        result_df = pd.DataFrame({'timestamp': full_time_index})
        result_df['pump_serial'] = pump_serial
        
        # Process each event type separately
        for event_type in event_types:
            type_df = merged_df[merged_df['event_type'] == event_type].copy()
            
            if event_type == 'CGM':
                resampled_cgm = self._resample_cgm_data(type_df, full_time_index)
                result_df = result_df.merge(resampled_cgm, on='timestamp', how='left')
                
            elif event_type == 'BASAL':
                resampled_basal = self._resample_basal_data(type_df, full_time_index)  
                result_df = result_df.merge(resampled_basal, on='timestamp', how='left')
                
            elif event_type == 'BOLUS':
                resampled_bolus = self._resample_bolus_data(type_df, full_time_index)
                result_df = result_df.merge(resampled_bolus, on='timestamp', how='left')
                
        # Fill gaps using intelligent strategies
        result_df = self._fill_gaps_intelligently(result_df)
        
        # Update metrics
        self.metrics.output_records = len(result_df)
        if self.metrics.input_records > 0:
            self.metrics.resampling_ratio = self.metrics.output_records / self.metrics.input_records
            
        logger.info(f"Resampling complete: {len(result_df)} synchronized intervals")
        return result_df
        
    def _resample_cgm_data(self, 
                          cgm_df: pd.DataFrame, 
                          time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample CGM data with appropriate aggregation"""
        if cgm_df.empty:
            return pd.DataFrame({'timestamp': time_index, 'bg_mgdl': np.nan})
            
        # Set timestamp as index for resampling
        cgm_df = cgm_df.set_index('timestamp')
        
        # Define aggregation rules based on config
        if self.config.cgm_aggregation == "mean":
            agg_func = 'mean'
        elif self.config.cgm_aggregation == "median":
            agg_func = 'median'  
        else:  # "last"
            agg_func = 'last'
            
        # Resample with appropriate aggregation
        if 'bg' in cgm_df.columns:
            resampled = cgm_df[['bg']].resample(self.config.target_frequency).agg(agg_func)
        elif 'value' in cgm_df.columns:
            resampled = cgm_df[['value']].resample(self.config.target_frequency).agg(agg_func)
            resampled = resampled.rename(columns={'value': 'bg'})
        else:
            logger.warning("No glucose value column found in CGM data")
            return pd.DataFrame({'timestamp': time_index, 'bg_mgdl': np.nan})
            
        # Reindex to full time range
        resampled = resampled.reindex(time_index)
        
        # Rename column to standard name
        if 'bg' in resampled.columns:
            resampled = resampled.rename(columns={'bg': 'bg_mgdl'})
            
        # Apply physiological range validation
        if 'bg_mgdl' in resampled.columns:
            invalid_mask = (
                (resampled['bg_mgdl'] < self.CGM_VALID_RANGE[0]) |
                (resampled['bg_mgdl'] > self.CGM_VALID_RANGE[1])
            )
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logger.warning(f"Filtering {invalid_count} CGM values outside valid range")
                resampled.loc[invalid_mask, 'bg_mgdl'] = np.nan
                self.metrics.invalid_values_filtered += invalid_count
                
        return resampled.reset_index()
        
    def _resample_basal_data(self, 
                           basal_df: pd.DataFrame, 
                           time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample basal rate data with appropriate aggregation"""
        if basal_df.empty:
            return pd.DataFrame({'timestamp': time_index, 'basal_rate_u_hr': np.nan})
            
        basal_df = basal_df.set_index('timestamp')
        
        # Basal rates should use forward-fill then aggregate
        basal_column = None
        for col in ['basal_rate', 'rate', 'value']:
            if col in basal_df.columns:
                basal_column = col
                break
                
        if basal_column is None:
            logger.warning("No basal rate column found in basal data")
            return pd.DataFrame({'timestamp': time_index, 'basal_rate_u_hr': np.nan})
            
        # Forward-fill to ensure continuous rates
        basal_filled = basal_df[[basal_column]].resample('1min').ffill()
        
        # Then aggregate to target frequency
        if self.config.basal_aggregation == "mean":
            resampled = basal_filled.resample(self.config.target_frequency).mean()
        else:  # "last" or other
            resampled = basal_filled.resample(self.config.target_frequency).last()
            
        # Reindex to full time range
        resampled = resampled.reindex(time_index)
        
        # Rename to standard column name
        resampled = resampled.rename(columns={basal_column: 'basal_rate_u_hr'})
        
        # Apply physiological range validation
        if 'basal_rate_u_hr' in resampled.columns:
            invalid_mask = (
                (resampled['basal_rate_u_hr'] < self.BASAL_VALID_RANGE[0]) |
                (resampled['basal_rate_u_hr'] > self.BASAL_VALID_RANGE[1])
            )
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logger.warning(f"Filtering {invalid_count} basal values outside valid range")
                resampled.loc[invalid_mask, 'basal_rate_u_hr'] = np.nan
                self.metrics.invalid_values_filtered += invalid_count
                
        return resampled.reset_index()
        
    def _resample_bolus_data(self, 
                           bolus_df: pd.DataFrame, 
                           time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample bolus dose data with summation aggregation"""
        if bolus_df.empty:
            return pd.DataFrame({'timestamp': time_index, 'bolus_units': 0.0})
            
        bolus_df = bolus_df.set_index('timestamp')
        
        # Find bolus dose column
        bolus_column = None
        for col in ['bolus_dose', 'dose', 'units', 'value']:
            if col in bolus_df.columns:
                bolus_column = col
                break
                
        if bolus_column is None:
            logger.warning("No bolus dose column found in bolus data")
            return pd.DataFrame({'timestamp': time_index, 'bolus_units': 0.0})
            
        # Sum boluses within each interval (multiple boluses can occur in 5 minutes)
        resampled = bolus_df[[bolus_column]].resample(self.config.target_frequency).sum()
        
        # Reindex to full time range and fill with 0 (no bolus = 0 units)
        resampled = resampled.reindex(time_index, fill_value=0.0)
        
        # Rename to standard column name
        resampled = resampled.rename(columns={bolus_column: 'bolus_units'})
        
        # Apply physiological range validation
        if 'bolus_units' in resampled.columns:
            invalid_mask = (
                (resampled['bolus_units'] < self.BOLUS_VALID_RANGE[0]) |
                (resampled['bolus_units'] > self.BOLUS_VALID_RANGE[1])
            )
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logger.warning(f"Filtering {invalid_count} bolus values outside valid range")
                resampled.loc[invalid_mask, 'bolus_units'] = np.nan
                self.metrics.invalid_values_filtered += invalid_count
                
        return resampled.reset_index()
        
    def _fill_gaps_intelligently(self, resampled_df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps using intelligent interpolation strategies"""
        logger.debug("Filling gaps with intelligent strategies...")
        
        # Count initial NaN values
        initial_nans = resampled_df.isnull().sum().sum()
        
        # Strategy 1: Forward-fill for short gaps (up to max_gap_minutes)
        max_fill_periods = self.config.max_gap_minutes // 5  # Convert to 5-minute periods
        
        for column in ['bg_mgdl', 'basal_rate_u_hr']:
            if column in resampled_df.columns:
                # Forward-fill with limit
                resampled_df[column] = resampled_df[column].fillna(
                    method='ffill', 
                    limit=max_fill_periods
                )
                
        # Strategy 2: Interpolation for longer gaps in CGM data
        if 'bg_mgdl' in resampled_df.columns and self.config.interpolation_method != "none":
            # Only interpolate gaps shorter than 2 hours (24 periods)
            resampled_df['bg_mgdl'] = resampled_df['bg_mgdl'].interpolate(
                method=self.config.interpolation_method,
                limit=24,
                limit_direction='both'
            )
            
        # Strategy 3: Basal rates - carry forward until next known rate
        if 'basal_rate_u_hr' in resampled_df.columns:
            # Basal rates should be forward-filled more aggressively
            resampled_df['basal_rate_u_hr'] = resampled_df['basal_rate_u_hr'].fillna(method='ffill')
            
        # Strategy 4: Bolus doses remain 0 (already handled in resampling)
        
        # Count final NaN values and calculate metrics
        final_nans = resampled_df.isnull().sum().sum()
        self.metrics.gaps_filled = initial_nans - final_nans
        
        # Add gap indicators for analysis
        resampled_df['cgm_gap_filled'] = (
            resampled_df['bg_mgdl'].notna() & 
            resampled_df['bg_mgdl'].shift(1).isna()
        ) if 'bg_mgdl' in resampled_df.columns else False
        
        return resampled_df
        
    def calculate_continuity_metrics(self, processed_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data continuity and quality metrics"""
        if processed_df.empty:
            return {}
            
        metrics = {}
        
        # CGM continuity
        if 'bg_mgdl' in processed_df.columns:
            cgm_valid = processed_df['bg_mgdl'].notna()
            metrics['cgm_continuity_pct'] = (cgm_valid.sum() / len(processed_df)) * 100
            
            # Gap analysis
            gaps = (~cgm_valid).astype(int)
            gap_lengths = []
            current_gap = 0
            
            for is_gap in gaps:
                if is_gap:
                    current_gap += 1
                else:
                    if current_gap > 0:
                        gap_lengths.append(current_gap)
                        current_gap = 0
                        
            if gap_lengths:
                metrics['cgm_max_gap_periods'] = max(gap_lengths)
                metrics['cgm_avg_gap_periods'] = np.mean(gap_lengths)
                metrics['cgm_total_gaps'] = len(gap_lengths)
            else:
                metrics['cgm_max_gap_periods'] = 0
                metrics['cgm_avg_gap_periods'] = 0
                metrics['cgm_total_gaps'] = 0
                
        # Basal continuity
        if 'basal_rate_u_hr' in processed_df.columns:
            basal_valid = processed_df['basal_rate_u_hr'].notna()
            metrics['basal_continuity_pct'] = (basal_valid.sum() / len(processed_df)) * 100
            
        # Bolus coverage (different metric - percentage of intervals with bolus)
        if 'bolus_units' in processed_df.columns:
            bolus_present = processed_df['bolus_units'] > 0
            metrics['bolus_coverage_pct'] = (bolus_present.sum() / len(processed_df)) * 100
            
        # Overall continuity score (weighted average)
        continuity_scores = []
        if 'cgm_continuity_pct' in metrics:
            continuity_scores.append(metrics['cgm_continuity_pct'] * 0.6)  # CGM most important
        if 'basal_continuity_pct' in metrics:
            continuity_scores.append(metrics['basal_continuity_pct'] * 0.3)  # Basal important
        if 'bolus_coverage_pct' in metrics:
            continuity_scores.append(metrics['bolus_coverage_pct'] * 0.1)  # Bolus coverage bonus
            
        if continuity_scores:
            metrics['overall_continuity_score'] = sum(continuity_scores) / len(continuity_scores)
            self.metrics.continuity_score = metrics['overall_continuity_score']
            
        return metrics
        
    def create_lstm_ready_dataset(self, 
                                pump_serial: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Create complete LSTM-ready dataset for a pump
        
        Args:
            pump_serial: Pump serial number
            start_date: Optional start date filter
            end_date: Optional end date filter
            save_path: Optional path to save the dataset
            
        Returns:
            LSTM-ready DataFrame with synchronized 5-minute intervals
        """
        logger.info(f"Creating LSTM-ready dataset for pump {pump_serial}")
        
        processing_start = datetime.now()
        
        try:
            # Step 1: Merge chronological batches
            merged_df = self.merge_chronological_batches(pump_serial, start_date, end_date)
            
            if merged_df.empty:
                logger.warning(f"No data available for pump {pump_serial}")
                return pd.DataFrame()
                
            # Step 2: Resample to 5-minute intervals
            resampled_df = self.resample_to_5min_intervals(merged_df, pump_serial)
            
            # Step 3: Add temporal features for LSTM
            lstm_ready_df = self._add_temporal_features(resampled_df)
            
            # Step 4: Calculate quality metrics
            continuity_metrics = self.calculate_continuity_metrics(lstm_ready_df)
            
            # Update processing metrics
            processing_end = datetime.now()
            self.metrics.processing_time_seconds = (processing_end - processing_start).total_seconds()
            
            # Log summary
            logger.info(f"LSTM dataset created: {len(lstm_ready_df)} records")
            logger.info(f"Processing time: {self.metrics.processing_time_seconds:.2f}s")
            logger.info(f"CGM continuity: {continuity_metrics.get('cgm_continuity_pct', 0):.1f}%")
            logger.info(f"Overall quality score: {continuity_metrics.get('overall_continuity_score', 0):.1f}")
            
            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save as compressed parquet for efficiency
                lstm_ready_df.to_parquet(save_path, compression='gzip')
                logger.info(f"Dataset saved to: {save_path}")
                
                # Save metadata alongside
                metadata = {
                    'pump_serial': pump_serial,
                    'processing_metrics': self.metrics.__dict__,
                    'continuity_metrics': continuity_metrics,
                    'created_at': datetime.now().isoformat(),
                    'record_count': len(lstm_ready_df),
                    'date_range': {
                        'start': lstm_ready_df['timestamp'].min().isoformat() if not lstm_ready_df.empty else None,
                        'end': lstm_ready_df['timestamp'].max().isoformat() if not lstm_ready_df.empty else None
                    }
                }
                
                metadata_path = save_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                    
            return lstm_ready_df
            
        except Exception as e:
            logger.error(f"Error creating LSTM dataset for pump {pump_serial}: {e}")
            raise
            
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for LSTM model training"""
        if df.empty or 'timestamp' not in df.columns:
            return df
            
        # Create copy for modifications
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for better LSTM performance
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Lag features for CGM (previous values)
        if 'bg_mgdl' in df.columns:
            df['bg_lag_1'] = df['bg_mgdl'].shift(1)   # 5 minutes ago
            df['bg_lag_2'] = df['bg_mgdl'].shift(2)   # 10 minutes ago
            df['bg_lag_6'] = df['bg_mgdl'].shift(6)   # 30 minutes ago
            df['bg_lag_12'] = df['bg_mgdl'].shift(12) # 60 minutes ago
            
            # Delta features (rate of change)
            df['bg_delta_5min'] = df['bg_mgdl'].diff(1)
            df['bg_delta_15min'] = df['bg_mgdl'].diff(3)
            df['bg_delta_30min'] = df['bg_mgdl'].diff(6)
            
        # Rolling statistics for basal (smoothed trends)
        if 'basal_rate_u_hr' in df.columns:
            df['basal_rate_mean_30min'] = df['basal_rate_u_hr'].rolling(6, min_periods=1).mean()
            df['basal_rate_mean_60min'] = df['basal_rate_u_hr'].rolling(12, min_periods=1).mean()
            
        # Cumulative bolus over time windows
        if 'bolus_units' in df.columns:
            df['bolus_cumsum_30min'] = df['bolus_units'].rolling(6, min_periods=1).sum()
            df['bolus_cumsum_60min'] = df['bolus_units'].rolling(12, min_periods=1).sum()
            df['bolus_cumsum_120min'] = df['bolus_units'].rolling(24, min_periods=1).sum()
            
        # Time since last bolus (useful feature)
        if 'bolus_units' in df.columns:
            bolus_times = df[df['bolus_units'] > 0].index
            time_since_bolus = []
            
            last_bolus_idx = -1
            for idx in df.index:
                # Find most recent bolus
                recent_boluses = bolus_times[bolus_times <= idx]
                if len(recent_boluses) > 0:
                    last_bolus_idx = recent_boluses[-1]
                    minutes_since = (idx - last_bolus_idx) * 5  # 5-minute intervals
                else:
                    minutes_since = 999  # No previous bolus
                    
                time_since_bolus.append(min(minutes_since, 999))  # Cap at 999 minutes
                
            df['minutes_since_bolus'] = time_since_bolus
            
        return df