"""
Unified multi-stream time-series resampling for LSTM training data preparation.

This module creates unified 5-minute master indices from BG, basal, and bolus streams,
handles stream alignment with different aggregation rules, and supports gap detection
and sequence segmentation for continuous training data preparation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import timedelta

from ..utils.time_utils import generate_time_index, extract_timestamp_from_event
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class UnifiedResampler:
    """
    Unified resampler that creates master 5-minute time indices from multiple data streams
    and handles stream alignment with proper aggregation rules for LSTM training.
    """
    
    def __init__(self, freq: str = '5min', edge_padding_minutes: int = 15):
        """
        Initialize the unified resampler
        
        Args:
            freq: Resampling frequency (default: '5min')
            edge_padding_minutes: Minutes to pad at start/end edges (default: 15)
        """
        self.freq = freq
        self.edge_padding = timedelta(minutes=edge_padding_minutes)
        self.resampling_stats = {
            'original_bg_count': 0,
            'original_basal_count': 0,
            'original_bolus_count': 0,
            'unified_intervals': 0,
            'time_span_hours': 0.0,
            'bg_coverage_pct': 0.0,
            'basal_coverage_pct': 0.0,
            'bolus_events': 0
        }
    
    def create_unified_master_index(self, 
                                  cgm_data: List[Dict[str, Any]], 
                                  basal_data: List[Dict[str, Any]], 
                                  bolus_data: List[Dict[str, Any]]) -> pd.DatetimeIndex:
        """
        Create unified 5-minute master time index from union of all stream timestamps.
        
        Args:
            cgm_data: List of normalized CGM events with 'timestamp' and 'bg' fields
            basal_data: List of normalized basal events with 'timestamp' and 'basal_rate' fields  
            bolus_data: List of normalized bolus events with 'timestamp' and 'bolus_dose' fields
            
        Returns:
            DatetimeIndex with uniform 5-minute intervals covering all data with edge padding
        """
        logger.info("Creating unified master time index from all data streams...")
        
        all_timestamps = []
        
        # Extract timestamps from CGM data
        cgm_timestamps = self._extract_timestamps_from_events(cgm_data, 'CGM')
        if cgm_timestamps:
            all_timestamps.extend(cgm_timestamps)
            self.resampling_stats['original_bg_count'] = len(cgm_timestamps)
        
        # Extract timestamps from basal data
        basal_timestamps = self._extract_timestamps_from_events(basal_data, 'Basal')
        if basal_timestamps:
            all_timestamps.extend(basal_timestamps)
            self.resampling_stats['original_basal_count'] = len(basal_timestamps)
        
        # Extract timestamps from bolus data
        bolus_timestamps = self._extract_timestamps_from_events(bolus_data, 'Bolus')
        if bolus_timestamps:
            all_timestamps.extend(bolus_timestamps)
            self.resampling_stats['original_bolus_count'] = len(bolus_timestamps)
        
        if not all_timestamps:
            logger.warning("No valid timestamps found in any data stream")
            return pd.DatetimeIndex([], dtype='datetime64[ns]')
        
        # Convert to pandas timestamps and validate
        valid_timestamps = []
        for ts in all_timestamps:
            try:
                pd_ts = pd.to_datetime(ts)
                # Validate timestamp is reasonable (2008-2030)
                if 2008 <= pd_ts.year <= 2030:
                    valid_timestamps.append(pd_ts)
                else:
                    logger.debug(f"Filtered out suspicious timestamp: {pd_ts}")
            except Exception as e:
                logger.debug(f"Could not parse timestamp {ts}: {e}")
        
        if not valid_timestamps:
            logger.warning("No valid timestamps after filtering")
            return pd.DatetimeIndex([], dtype='datetime64[ns]')
        
        # Calculate time range with edge padding
        start_time = min(valid_timestamps) - self.edge_padding
        end_time = max(valid_timestamps) + self.edge_padding
        
        # Generate unified master index
        master_index = generate_time_index(start_time, end_time, self.freq)
        
        # Update statistics
        self.resampling_stats['unified_intervals'] = len(master_index)
        if len(master_index) > 0:
            time_span = (master_index[-1] - master_index[0])
            self.resampling_stats['time_span_hours'] = time_span.total_seconds() / 3600
        
        logger.info(f"Created unified master index: {start_time} to {end_time} "
                   f"({len(master_index)} intervals, {self.resampling_stats['time_span_hours']:.1f} hours)")
        
        return master_index
    
    def align_streams_to_master_index(self, 
                                    cgm_data: List[Dict[str, Any]], 
                                    basal_data: List[Dict[str, Any]], 
                                    bolus_data: List[Dict[str, Any]], 
                                    master_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """
        Align all data streams to the unified master time index using appropriate aggregation rules.
        
        Stream alignment rules:
        - BG: Last observed value within each interval (no interpolation yet)
        - Basal: Forward-fill (stepwise rate persists until changed)
        - Bolus: Sum all events within each 5-minute interval
        
        Args:
            cgm_data: List of CGM events
            basal_data: List of basal events  
            bolus_data: List of bolus events
            master_index: Unified master time index
            
        Returns:
            Dictionary with aligned pandas Series for each stream
        """
        logger.info("Aligning data streams to master index...")
        
        aligned_streams = {}
        
        # Align BG stream - last observed value
        aligned_streams['bg'] = self._align_bg_stream(cgm_data, master_index)
        
        # Align basal stream - forward fill stepwise rates
        aligned_streams['basal_rate'] = self._align_basal_stream(basal_data, master_index)
        
        # Align bolus stream - sum events per interval
        aligned_streams['bolus_dose'] = self._align_bolus_stream(bolus_data, master_index)
        
        # Calculate coverage statistics
        self._calculate_coverage_stats(aligned_streams, master_index)
        
        logger.info(f"Stream alignment completed - BG: {self.resampling_stats['bg_coverage_pct']:.1f}% coverage, "
                   f"Basal: {self.resampling_stats['basal_coverage_pct']:.1f}% coverage, "
                   f"Bolus: {self.resampling_stats['bolus_events']} events")
        
        return aligned_streams
    
    def create_dataframes_from_events(self, 
                                    cgm_data: List[Dict[str, Any]], 
                                    basal_data: List[Dict[str, Any]], 
                                    bolus_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert event lists to DataFrames for easier processing.
        
        Args:
            cgm_data: List of CGM events
            basal_data: List of basal events
            bolus_data: List of bolus events
            
        Returns:
            Tuple of (cgm_df, basal_df, bolus_df)
        """
        # Convert CGM data
        cgm_df = self._events_to_dataframe(cgm_data, 'bg', 'CGM')
        
        # Convert basal data
        basal_df = self._events_to_dataframe(basal_data, 'basal_rate', 'Basal')
        
        # Convert bolus data  
        bolus_df = self._events_to_dataframe(bolus_data, 'bolus_dose', 'Bolus')
        
        return cgm_df, basal_df, bolus_df
    
    def _extract_timestamps_from_events(self, events: List[Dict[str, Any]], stream_name: str) -> List:
        """Extract timestamps from event list"""
        timestamps = []
        for event in events:
            ts = event.get('timestamp')
            if ts is not None:
                timestamps.append(ts)
        
        logger.debug(f"Extracted {len(timestamps)} timestamps from {stream_name} stream")
        return timestamps
    
    def _events_to_dataframe(self, events: List[Dict[str, Any]], value_col: str, stream_name: str) -> pd.DataFrame:
        """
        Convert events to DataFrame with timestamp index
        
        Args:
            events: List of event dictionaries
            value_col: Name of the value column to extract
            stream_name: Name of stream for logging
            
        Returns:
            DataFrame with timestamp index and value column
        """
        if not events:
            logger.debug(f"No events in {stream_name} stream")
            return pd.DataFrame()
        
        # Extract valid events with both timestamp and value
        valid_events = []
        for event in events:
            if event.get('timestamp') is not None and event.get(value_col) is not None:
                valid_events.append({
                    'timestamp': event['timestamp'],
                    value_col: event[value_col]
                })
        
        if not valid_events:
            logger.debug(f"No valid events with both timestamp and {value_col} in {stream_name} stream")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(valid_events)
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle duplicates by taking the last value
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Set timestamp as index
        df = df.set_index('timestamp').sort_index()
        
        logger.debug(f"Created {stream_name} DataFrame with {len(df)} records")
        return df
    
    def _align_bg_stream(self, cgm_data: List[Dict[str, Any]], master_index: pd.DatetimeIndex) -> pd.Series:
        """
        Align BG stream using last observed value within each interval.
        No interpolation is performed at this stage.
        """
        cgm_df = self._events_to_dataframe(cgm_data, 'bg', 'CGM')
        
        if cgm_df.empty:
            logger.debug("Empty CGM data, filling with NaN")
            return pd.Series(index=master_index, dtype=float, name='bg')
        
        # Resample to master frequency using last observed value
        bg_resampled = cgm_df['bg'].resample(self.freq).last()
        
        # Align to master index without forward filling (preserve gaps)
        bg_aligned = bg_resampled.reindex(master_index)
        bg_aligned.name = 'bg'
        
        return bg_aligned
    
    def _align_basal_stream(self, basal_data: List[Dict[str, Any]], master_index: pd.DatetimeIndex) -> pd.Series:
        """
        Align basal stream using forward-fill (stepwise rates persist until changed).
        """
        basal_df = self._events_to_dataframe(basal_data, 'basal_rate', 'Basal')
        
        if basal_df.empty:
            logger.debug("Empty basal data, filling with 0.0")
            return pd.Series(0.0, index=master_index, name='basal_rate')
        
        # Resample using forward fill method (rates persist)
        basal_resampled = basal_df['basal_rate'].resample(self.freq).last()
        
        # Align to master index with forward fill
        basal_aligned = basal_resampled.reindex(master_index, method='ffill')
        
        # Fill any remaining NaN values with 0.0
        basal_aligned = basal_aligned.fillna(0.0)
        basal_aligned.name = 'basal_rate'
        
        return basal_aligned
    
    def _align_bolus_stream(self, bolus_data: List[Dict[str, Any]], master_index: pd.DatetimeIndex) -> pd.Series:
        """
        Align bolus stream by summing all events within each 5-minute interval.
        """
        bolus_df = self._events_to_dataframe(bolus_data, 'bolus_dose', 'Bolus')
        
        if bolus_df.empty:
            logger.debug("Empty bolus data, filling with 0.0")
            return pd.Series(0.0, index=master_index, name='bolus_dose')
        
        # Sum bolus events within each interval
        bolus_resampled = bolus_df['bolus_dose'].resample(self.freq).sum()
        
        # Align to master index
        bolus_aligned = bolus_resampled.reindex(master_index, fill_value=0.0)
        bolus_aligned.name = 'bolus_dose'
        
        return bolus_aligned
    
    def _calculate_coverage_stats(self, aligned_streams: Dict[str, pd.Series], master_index: pd.DatetimeIndex):
        """Calculate coverage statistics for aligned streams"""
        total_intervals = len(master_index)
        
        if total_intervals > 0:
            # BG coverage (non-NaN values)
            bg_valid = aligned_streams['bg'].notna().sum()
            self.resampling_stats['bg_coverage_pct'] = (bg_valid / total_intervals) * 100
            
            # Basal coverage (non-zero values)  
            basal_nonzero = (aligned_streams['basal_rate'] > 0).sum()
            self.resampling_stats['basal_coverage_pct'] = (basal_nonzero / total_intervals) * 100
            
            # Bolus event count
            self.resampling_stats['bolus_events'] = (aligned_streams['bolus_dose'] > 0).sum()
    
    def get_resampling_stats(self) -> Dict[str, Any]:
        """Get resampling statistics"""
        return self.resampling_stats.copy()
    
    def reset_stats(self):
        """Reset resampling statistics"""
        self.resampling_stats = {
            'original_bg_count': 0,
            'original_basal_count': 0,
            'original_bolus_count': 0,
            'unified_intervals': 0,
            'time_span_hours': 0.0,
            'bg_coverage_pct': 0.0,
            'basal_coverage_pct': 0.0,
            'bolus_events': 0
        }


class SequenceSegmenter:
    """
    Handles gap detection and sequence segmentation for LSTM training preparation.
    Breaks data into continuous segments based on configurable gap thresholds.
    """
    
    def __init__(self, max_gap_hours: float = 15.0, min_segment_length: int = 12):
        """
        Initialize sequence segmenter
        
        Args:
            max_gap_hours: Maximum gap size before breaking into new sequence (default: 15 hours)
            min_segment_length: Minimum sequence length in intervals to keep (default: 12 = 1 hour)
        """
        self.max_gap_hours = max_gap_hours
        self.max_gap_intervals = int(max_gap_hours * 12)  # Convert hours to 5-min intervals
        self.min_segment_length = min_segment_length
        self.segmentation_stats = {
            'total_segments': 0,
            'avg_segment_length': 0.0,
            'max_segment_length': 0,
            'min_segment_length': 0,
            'total_gaps_detected': 0,
            'avg_gap_size_hours': 0.0
        }
    
    def detect_temporal_gaps(self, aligned_streams: Dict[str, pd.Series]) -> List[Tuple[int, int, float]]:
        """
        Detect temporal gaps >15 hours in the aligned data streams.
        
        Args:
            aligned_streams: Dictionary of aligned pandas Series
            
        Returns:
            List of (start_idx, end_idx, gap_hours) tuples for detected gaps
        """
        logger.info(f"Detecting temporal gaps > {self.max_gap_hours} hours...")
        
        # Use BG stream as primary reference for gap detection
        bg_series = aligned_streams.get('bg')
        if bg_series is None or bg_series.empty:
            logger.warning("No BG data available for gap detection")
            return []
        
        gaps = []
        time_index = bg_series.index
        
        # Calculate time differences between consecutive intervals
        time_diffs = pd.Series(time_index).diff()
        
        # Find gaps larger than threshold
        large_gaps = time_diffs > pd.Timedelta(hours=self.max_gap_hours)
        
        for i, is_gap in enumerate(large_gaps):
            if is_gap and i > 0:
                gap_start_idx = i - 1
                gap_end_idx = i
                gap_hours = time_diffs.iloc[i].total_seconds() / 3600
                
                gaps.append((gap_start_idx, gap_end_idx, gap_hours))
                logger.debug(f"Detected gap: {gap_hours:.1f} hours at index {gap_start_idx}-{gap_end_idx}")
        
        self.segmentation_stats['total_gaps_detected'] = len(gaps)
        if gaps:
            avg_gap_hours = sum(g[2] for g in gaps) / len(gaps)
            self.segmentation_stats['avg_gap_size_hours'] = avg_gap_hours
        
        logger.info(f"Detected {len(gaps)} temporal gaps > {self.max_gap_hours} hours")
        return gaps
    
    def segment_by_gaps(self, aligned_streams: Dict[str, pd.Series]) -> List[Dict[str, pd.Series]]:
        """
        Break aligned streams into continuous segments based on detected gaps.
        
        Args:
            aligned_streams: Dictionary of aligned pandas Series
            
        Returns:
            List of segment dictionaries, each containing aligned series for the segment
        """
        logger.info("Segmenting data streams by temporal gaps...")
        
        # Detect gaps
        gaps = self.detect_temporal_gaps(aligned_streams)
        
        segments = []
        
        # Get the full time index
        time_index = list(aligned_streams.values())[0].index
        
        if not gaps:
            # No gaps - return single segment
            if len(time_index) >= self.min_segment_length:
                segments.append(aligned_streams.copy())
                logger.info("No gaps detected - created single segment")
            else:
                logger.warning(f"Single segment too short ({len(time_index)} < {self.min_segment_length})")
        else:
            # Create segments between gaps
            segment_boundaries = [0]  # Start with beginning
            
            for gap_start, gap_end, gap_hours in gaps:
                segment_boundaries.extend([gap_start + 1, gap_end])
            
            segment_boundaries.append(len(time_index))  # End with final index
            
            # Create segments from boundaries
            for i in range(0, len(segment_boundaries) - 1, 2):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i + 1]
                
                segment_length = end_idx - start_idx
                
                if segment_length >= self.min_segment_length:
                    segment_data = {}
                    segment_time_index = time_index[start_idx:end_idx]
                    
                    for stream_name, series in aligned_streams.items():
                        segment_data[stream_name] = series.iloc[start_idx:end_idx].copy()
                    
                    segments.append(segment_data)
                    logger.debug(f"Created segment {len(segments)}: {segment_length} intervals "
                                f"({segment_time_index[0]} to {segment_time_index[-1]})")
                else:
                    logger.debug(f"Skipped short segment: {segment_length} intervals < {self.min_segment_length}")
        
        # Update statistics
        self._calculate_segmentation_stats(segments)
        
        logger.info(f"Created {len(segments)} continuous segments for LSTM training")
        return segments
    
    def _calculate_segmentation_stats(self, segments: List[Dict[str, pd.Series]]):
        """Calculate segmentation statistics"""
        if not segments:
            return
        
        segment_lengths = [len(list(seg.values())[0]) for seg in segments]
        
        self.segmentation_stats['total_segments'] = len(segments)
        self.segmentation_stats['avg_segment_length'] = sum(segment_lengths) / len(segment_lengths)
        self.segmentation_stats['max_segment_length'] = max(segment_lengths)
        self.segmentation_stats['min_segment_length'] = min(segment_lengths)
    
    def get_segmentation_stats(self) -> Dict[str, Any]:
        """Get segmentation statistics"""
        return self.segmentation_stats.copy()
    
    def reset_stats(self):
        """Reset segmentation statistics"""
        self.segmentation_stats = {
            'total_segments': 0,
            'avg_segment_length': 0.0,
            'max_segment_length': 0,
            'min_segment_length': 0,
            'total_gaps_detected': 0,
            'avg_gap_size_hours': 0.0
        }


class BgImputer:
    """
    Handles BG gap imputation with configurable gap size limits and masking.
    Imputes gaps ≤60 minutes using linear/PCHIP interpolation while masking larger gaps.
    """
    
    def __init__(self, max_impute_minutes: int = 60, min_bg: float = 20.0, max_bg: float = 600.0):
        """
        Initialize BG imputer
        
        Args:
            max_impute_minutes: Maximum gap size to impute in minutes (default: 60)
            min_bg: Minimum physiological BG value for clipping (default: 20 mg/dL)
            max_bg: Maximum physiological BG value for clipping (default: 600 mg/dL)
        """
        self.max_impute_intervals = max_impute_minutes // 5  # Convert to 5-min intervals
        self.min_bg = min_bg
        self.max_bg = max_bg
        self.imputation_stats = {
            'gaps_imputed': 0,
            'gaps_masked': 0,
            'values_imputed': 0,
            'values_clipped': 0,
            'avg_gap_size_imputed': 0.0,
            'avg_gap_size_masked': 0.0
        }
    
    def impute_bg_gaps(self, bg_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Impute BG gaps ≤60 minutes and mask larger gaps for LSTM training.
        
        Args:
            bg_series: BG series with NaN gaps
            
        Returns:
            Tuple of (imputed_bg_series, mask_bg_series, mask_label_series)
            - imputed_bg_series: BG values with short gaps imputed
            - mask_bg_series: Boolean mask indicating imputed/missing values
            - mask_label_series: Boolean mask for training labels (False where gaps > max_impute)
        """
        logger.info("Applying BG gap imputation and masking...")
        
        # Initialize output series
        imputed_bg = bg_series.copy()
        mask_bg = pd.Series(False, index=bg_series.index, name='mask_bg')  # True = missing/imputed
        mask_label = pd.Series(True, index=bg_series.index, name='mask_label')  # False = exclude from training
        
        # Find contiguous NaN gaps
        is_nan = bg_series.isna()
        gap_runs = self._find_contiguous_runs(is_nan)
        
        imputed_gaps = 0
        masked_gaps = 0
        total_imputed_values = 0
        
        for start_idx, end_idx in gap_runs:
            gap_length = end_idx - start_idx + 1
            gap_minutes = gap_length * 5
            
            if gap_length <= self.max_impute_intervals:
                # Try to impute short gaps
                if self._impute_gap(imputed_bg, start_idx, end_idx):
                    mask_bg.iloc[start_idx:end_idx + 1] = True  # Mark as imputed
                    imputed_gaps += 1
                    total_imputed_values += gap_length
                    logger.debug(f"Imputed gap: {gap_minutes} min ({gap_length} intervals)")
                else:
                    # Couldn't impute - mask from training
                    mask_label.iloc[start_idx:end_idx + 1] = False
                    mask_bg.iloc[start_idx:end_idx + 1] = True
                    masked_gaps += 1
                    logger.debug(f"Masked un-imputable gap: {gap_minutes} min ({gap_length} intervals)")
            else:
                # Gap too large - mask from training
                mask_label.iloc[start_idx:end_idx + 1] = False
                mask_bg.iloc[start_idx:end_idx + 1] = True
                masked_gaps += 1
                logger.debug(f"Masked large gap: {gap_minutes} min ({gap_length} intervals)")
        
        # Clip imputed values to physiological range
        clipped_values = self._clip_bg_values(imputed_bg)
        
        # Update statistics
        self.imputation_stats.update({
            'gaps_imputed': imputed_gaps,
            'gaps_masked': masked_gaps,
            'values_imputed': total_imputed_values,
            'values_clipped': clipped_values
        })
        
        logger.info(f"BG imputation completed: {imputed_gaps} gaps imputed, "
                   f"{masked_gaps} gaps masked, {total_imputed_values} values imputed")
        
        return imputed_bg, mask_bg, mask_label
    
    def _find_contiguous_runs(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Find contiguous runs of True values in a boolean mask"""
        runs = []
        in_run = False
        start_idx = None
        
        for i, value in enumerate(mask):
            if value and not in_run:
                # Start of a new run
                in_run = True
                start_idx = i
            elif not value and in_run:
                # End of current run
                in_run = False
                runs.append((start_idx, i - 1))
        
        # Handle run that extends to the end
        if in_run:
            runs.append((start_idx, len(mask) - 1))
            
        return runs
    
    def _impute_gap(self, bg_series: pd.Series, start_idx: int, end_idx: int) -> bool:
        """
        Impute a single BG gap using PCHIP or linear interpolation
        
        Args:
            bg_series: BG series to modify in-place
            start_idx: Start index of gap
            end_idx: End index of gap
            
        Returns:
            True if imputation successful, False otherwise
        """
        # Check for valid bounding values
        has_left_bound = start_idx > 0 and pd.notna(bg_series.iloc[start_idx - 1])
        has_right_bound = end_idx < len(bg_series) - 1 and pd.notna(bg_series.iloc[end_idx + 1])
        
        if not (has_left_bound and has_right_bound):
            return False
        
        gap_length = end_idx - start_idx + 1
        
        try:
            # Try PCHIP interpolation first
            from scipy.interpolate import PchipInterpolator
            
            left_idx = start_idx - 1
            right_idx = end_idx + 1
            
            x_known = np.array([left_idx, right_idx])
            y_known = np.array([bg_series.iloc[left_idx], bg_series.iloc[right_idx]])
            x_interp = np.arange(start_idx, end_idx + 1)
            
            interpolator = PchipInterpolator(x_known, y_known)
            interpolated_values = interpolator(x_interp)
            
            # Apply interpolated values
            bg_series.iloc[start_idx:end_idx + 1] = interpolated_values
            return True
            
        except ImportError:
            # Fallback to linear interpolation
            left_val = bg_series.iloc[start_idx - 1]
            right_val = bg_series.iloc[end_idx + 1]
            
            interpolated_values = np.linspace(left_val, right_val, gap_length + 2)[1:-1]
            bg_series.iloc[start_idx:end_idx + 1] = interpolated_values
            return True
            
        except Exception as e:
            logger.debug(f"Interpolation failed for gap {start_idx}-{end_idx}: {e}")
            return False
    
    def _clip_bg_values(self, bg_series: pd.Series) -> int:
        """Clip BG values to physiological range"""
        valid_mask = bg_series.notna()
        clipped_count = 0
        
        # Count and clip values outside range
        too_low = (bg_series < self.min_bg) & valid_mask
        too_high = (bg_series > self.max_bg) & valid_mask
        
        if too_low.any():
            clip_count_low = too_low.sum()
            bg_series.loc[too_low] = self.min_bg
            clipped_count += clip_count_low
            logger.debug(f"Clipped {clip_count_low} BG values to minimum {self.min_bg}")
        
        if too_high.any():
            clip_count_high = too_high.sum()
            bg_series.loc[too_high] = self.max_bg
            clipped_count += clip_count_high
            logger.debug(f"Clipped {clip_count_high} BG values to maximum {self.max_bg}")
        
        return clipped_count
    
    def get_imputation_stats(self) -> Dict[str, Any]:
        """Get imputation statistics"""
        return self.imputation_stats.copy()
    
    def reset_stats(self):
        """Reset imputation statistics"""
        self.imputation_stats = {
            'gaps_imputed': 0,
            'gaps_masked': 0,
            'values_imputed': 0,
            'values_clipped': 0,
            'avg_gap_size_imputed': 0.0,
            'avg_gap_size_masked': 0.0
        }


class FeatureEngineer:
    """
    Handles feature engineering for LSTM training including derived features,
    normalization, and time-based features.
    """
    
    def __init__(self, normalization_method: str = 'z-score'):
        """
        Initialize feature engineer
        
        Args:
            normalization_method: 'z-score' or 'min-max' normalization (default: 'z-score')
        """
        self.normalization_method = normalization_method
        self.feature_stats = {}
        
    def add_derived_features(self, aligned_streams: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Add derived features for LSTM training.
        
        Features added:
        - time_since_last_bolus: Minutes since last bolus >0
        - basal_delta: Change in basal rate from previous interval
        - bg_slope_15min: BG slope over 15 minutes (3 intervals)
        - bg_slope_30min: BG slope over 30 minutes (6 intervals)
        - time_of_day_sin/cos: Cyclical time-of-day features
        
        Args:
            aligned_streams: Dictionary of aligned series
            
        Returns:
            Dictionary with additional derived features
        """
        logger.info("Adding derived features for LSTM training...")
        
        enhanced_streams = aligned_streams.copy()
        
        # Time since last bolus
        enhanced_streams['time_since_last_bolus'] = self._calculate_time_since_last_bolus(
            aligned_streams['bolus_dose']
        )
        
        # Basal delta
        enhanced_streams['basal_delta'] = aligned_streams['basal_rate'].diff().fillna(0.0)
        
        # BG slopes (if BG data exists)
        if 'bg' in aligned_streams and aligned_streams['bg'].notna().any():
            enhanced_streams['bg_slope_15min'] = self._calculate_bg_slope(
                aligned_streams['bg'], window_intervals=3
            )
            enhanced_streams['bg_slope_30min'] = self._calculate_bg_slope(
                aligned_streams['bg'], window_intervals=6
            )
        else:
            # Fill with zeros if no BG data
            time_index = aligned_streams['bolus_dose'].index
            enhanced_streams['bg_slope_15min'] = pd.Series(0.0, index=time_index)
            enhanced_streams['bg_slope_30min'] = pd.Series(0.0, index=time_index)
        
        # Time-of-day cyclical features
        time_index = pd.to_datetime(aligned_streams['bolus_dose'].index)
        sin_time, cos_time = self._add_time_of_day_features(time_index)
        enhanced_streams['sin_time'] = sin_time
        enhanced_streams['cos_time'] = cos_time
        
        logger.info(f"Added {len(enhanced_streams) - len(aligned_streams)} derived features")
        return enhanced_streams
    
    def normalize_features(self, features_dict: Dict[str, pd.Series], 
                          exclude_features: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Normalize features using specified method (per-patient normalization).
        
        Args:
            features_dict: Dictionary of feature series
            exclude_features: List of features to exclude from normalization
            
        Returns:
            Dictionary with normalized features
        """
        if exclude_features is None:
            exclude_features = ['sin_time', 'cos_time', 'mask_bg', 'mask_label']
        
        logger.info(f"Normalizing features using {self.normalization_method} method...")
        
        normalized_features = features_dict.copy()
        
        for feature_name, series in features_dict.items():
            if feature_name in exclude_features:
                continue
            
            if series.notna().sum() == 0:
                logger.debug(f"Skipping normalization of {feature_name} (all NaN)")
                continue
            
            if self.normalization_method == 'z-score':
                normalized_series, stats = self._z_score_normalize(series)
            elif self.normalization_method == 'min-max':
                normalized_series, stats = self._min_max_normalize(series)
            else:
                logger.warning(f"Unknown normalization method: {self.normalization_method}")
                normalized_series = series
                stats = {}
            
            normalized_features[feature_name] = normalized_series
            self.feature_stats[feature_name] = stats
        
        logger.info(f"Normalized {len(self.feature_stats)} features")
        return normalized_features
    
    def _calculate_time_since_last_bolus(self, bolus_series: pd.Series) -> pd.Series:
        """Calculate minutes since last bolus event"""
        bolus_events = bolus_series > 0
        time_since_bolus = pd.Series(0, index=bolus_series.index, dtype=float)
        
        last_bolus_idx = None
        for i, (timestamp, has_bolus) in enumerate(bolus_events.items()):
            if has_bolus:
                last_bolus_idx = i
                time_since_bolus.iloc[i] = 0.0
            elif last_bolus_idx is not None:
                intervals_since = i - last_bolus_idx
                time_since_bolus.iloc[i] = intervals_since * 5.0  # Convert to minutes
            else:
                time_since_bolus.iloc[i] = np.inf  # No previous bolus
        
        # Cap at reasonable maximum (24 hours = 1440 minutes)
        time_since_bolus = time_since_bolus.clip(upper=1440.0)
        
        return time_since_bolus
    
    def _calculate_bg_slope(self, bg_series: pd.Series, window_intervals: int) -> pd.Series:
        """Calculate BG slope over specified window"""
        slope_series = pd.Series(0.0, index=bg_series.index)
        
        for i in range(window_intervals, len(bg_series)):
            window_start = i - window_intervals
            window_end = i
            
            # Check if we have valid BG values at both ends
            if (pd.notna(bg_series.iloc[window_start]) and 
                pd.notna(bg_series.iloc[window_end])):
                
                bg_change = bg_series.iloc[window_end] - bg_series.iloc[window_start]
                time_change = window_intervals * 5  # minutes
                slope = bg_change / time_change  # mg/dL per minute
                slope_series.iloc[i] = slope
        
        return slope_series
    
    def _add_time_of_day_features(self, time_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
        """Add cyclical time-of-day features"""
        # Convert to minutes from midnight
        minutes_from_midnight = (time_index.hour * 60 + time_index.minute)
        
        # Convert to cyclical features (0-2π range)
        radians = 2 * np.pi * minutes_from_midnight / 1440  # 1440 = minutes per day
        
        sin_time = pd.Series(np.sin(radians), index=time_index, name='sin_time')
        cos_time = pd.Series(np.cos(radians), index=time_index, name='cos_time')
        
        return sin_time, cos_time
    
    def _z_score_normalize(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Apply z-score normalization"""
        valid_values = series.dropna()
        
        if len(valid_values) == 0:
            return series, {}
        
        mean_val = valid_values.mean()
        std_val = valid_values.std()
        
        if std_val == 0:
            # Constant values - return zeros
            normalized = pd.Series(0.0, index=series.index)
        else:
            normalized = (series - mean_val) / std_val
        
        stats = {
            'method': 'z-score',
            'mean': mean_val,
            'std': std_val,
            'min': valid_values.min(),
            'max': valid_values.max()
        }
        
        return normalized, stats
    
    def _min_max_normalize(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Apply min-max normalization to [0, 1] range"""
        valid_values = series.dropna()
        
        if len(valid_values) == 0:
            return series, {}
        
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val == min_val:
            # Constant values - return zeros
            normalized = pd.Series(0.0, index=series.index)
        else:
            normalized = (series - min_val) / (max_val - min_val)
        
        stats = {
            'method': 'min-max',
            'min': min_val,
            'max': max_val,
            'mean': valid_values.mean(),
            'std': valid_values.std()
        }
        
        return normalized, stats
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get feature normalization statistics"""
        return self.feature_stats.copy()
    
    def reset_stats(self):
        """Reset feature statistics"""
        self.feature_stats = {}