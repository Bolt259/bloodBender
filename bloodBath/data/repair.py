"""
Retroactive synthetic 100 BG value detection and repair module.

This module identifies and corrects synthetic 100 mg/dL values introduced by 
earlier system defaults while preserving legitimate 100 mg/dL readings.
Operates directly on existing CSV archives without refetching data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
# from ..utils.time_utils import TimeUtils  # Not needed for current implementation

logger = logging.getLogger(__name__)


@dataclass
class Synthetic100Segment:
    """Represents a detected synthetic 100 BG segment"""
    start_idx: int
    end_idx: int
    length: int
    variance: float
    mean_bg: float
    has_insulin_activity: bool
    transition_gradient: float
    is_synthetic: bool
    confidence_score: float


class Synthetic100Detector:
    """
    Detects synthetic 100 BG value segments using multiple validation criteria
    """
    
    def __init__(self, 
                 min_segment_length: int = 6,  # 30 minutes (6 bins of 5 min)
                 variance_threshold: float = 1.0,
                 gradient_threshold: float = 3.0,  # 15 mg/dL per 5 min
                 insulin_window_minutes: int = 30):
        self.min_segment_length = min_segment_length
        self.variance_threshold = variance_threshold
        self.gradient_threshold = gradient_threshold
        self.insulin_window_bins = insulin_window_minutes // 5  # Convert to 5-min bins
        
    def detect_segments(self, df: pd.DataFrame) -> List[Synthetic100Segment]:
        """
        Detect all candidate 100 BG segments in the dataframe
        
        Args:
            df: DataFrame with datetime, bg, basal, bolus columns
            
        Returns:
            List of Synthetic100Segment objects
        """
        if df.empty or 'bg' not in df.columns:
            return []
        
        segments = []
        bg_series = df['bg'].fillna(-999)  # Temporary fill to identify runs
        
        # Find consecutive 100 value runs
        is_100 = (bg_series == 100.0)
        run_starts = []
        run_lengths = []
        
        i = 0
        while i < len(is_100):
            if is_100.iloc[i]:
                start = i
                while i < len(is_100) and is_100.iloc[i]:
                    i += 1
                length = i - start
                if length >= 2:  # At least 2 consecutive 100s
                    run_starts.append(start)
                    run_lengths.append(length)
            else:
                i += 1
        
        # Analyze each run
        for start_idx, length in zip(run_starts, run_lengths):
            end_idx = start_idx + length - 1
            segment = self._analyze_segment(df, start_idx, end_idx, length)
            segments.append(segment)
            
        logger.debug(f"Detected {len(segments)} candidate 100-BG segments")
        return segments
    
    def _analyze_segment(self, df: pd.DataFrame, start_idx: int, end_idx: int, length: int) -> Synthetic100Segment:
        """
        Analyze a single 100-BG segment for synthetic characteristics
        """
        # Basic segment info
        segment_data = df.iloc[start_idx:end_idx+1]
        bg_variance = segment_data['bg'].var() if len(segment_data) > 1 else 0.0
        mean_bg = segment_data['bg'].mean()
        
        # Check for insulin activity during segment
        has_insulin_activity = self._check_insulin_activity(df, start_idx, end_idx)
        
        # Check transition gradient
        transition_gradient = self._check_transition_gradient(df, start_idx, end_idx)
        
        # Determine if synthetic based on criteria
        is_synthetic = self._classify_as_synthetic(
            length, bg_variance, has_insulin_activity, transition_gradient
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            length, bg_variance, has_insulin_activity, transition_gradient
        )
        
        return Synthetic100Segment(
            start_idx=start_idx,
            end_idx=end_idx,
            length=length,
            variance=bg_variance,
            mean_bg=mean_bg,
            has_insulin_activity=has_insulin_activity,
            transition_gradient=transition_gradient,
            is_synthetic=is_synthetic,
            confidence_score=confidence_score
        )
    
    def _check_insulin_activity(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Check for insulin activity (bolus or significant basal changes) around segment
        """
        # Expand window around segment
        window_start = max(0, start_idx - self.insulin_window_bins)
        window_end = min(len(df), end_idx + self.insulin_window_bins + 1)
        window_data = df.iloc[window_start:window_end]
        
        # Check for bolus activity
        if 'bolus' in df.columns:
            bolus_activity = window_data['bolus'].sum() > 0
            if bolus_activity:
                return True
        
        # Check for significant basal changes
        if 'basal' in df.columns:
            basal_series = window_data['basal'].dropna()
            if len(basal_series) > 1:
                basal_changes = basal_series.diff().abs()
                significant_basal_change = (basal_changes > 0.05).any()
                if significant_basal_change:
                    return True
        
        return False
    
    def _check_transition_gradient(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """
        Check BG gradient before and after the 100-segment
        """
        # Get BG before segment (look back up to 6 bins)
        bg_before = None
        for i in range(max(0, start_idx - 6), start_idx):
            if not pd.isna(df.iloc[i]['bg']) and df.iloc[i]['bg'] != 100.0:
                bg_before = df.iloc[i]['bg']
                break
        
        # Get BG after segment (look forward up to 6 bins)  
        bg_after = None
        for i in range(end_idx + 1, min(len(df), end_idx + 7)):
            if not pd.isna(df.iloc[i]['bg']) and df.iloc[i]['bg'] != 100.0:
                bg_after = df.iloc[i]['bg']
                break
        
        # Calculate average gradient per 5-min bin
        if bg_before is not None and bg_after is not None:
            total_bins = (end_idx - start_idx + 1) + 2  # segment + transition bins
            gradient = abs(bg_after - bg_before) / total_bins
            return gradient
        
        return 0.0
    
    def _classify_as_synthetic(self, length: int, variance: float, 
                              has_insulin_activity: bool, transition_gradient: float) -> bool:
        """
        Classify segment as synthetic based on multiple criteria
        """
        # Criterion 1: Long segment with low variance
        long_flat_segment = (length >= self.min_segment_length and variance < self.variance_threshold)
        
        # Criterion 2: No insulin activity during flat BG (suspicious)
        no_insulin_flat = (length >= 4 and variance < self.variance_threshold and not has_insulin_activity)
        
        # Criterion 3: Unrealistic transition gradient
        unrealistic_gradient = transition_gradient > self.gradient_threshold
        
        # Classify as synthetic if any strong criterion is met
        is_synthetic = long_flat_segment or no_insulin_flat or unrealistic_gradient
        
        return is_synthetic
    
    def _calculate_confidence(self, length: int, variance: float,
                             has_insulin_activity: bool, transition_gradient: float) -> float:
        """
        Calculate confidence score for synthetic classification (0-1)
        """
        confidence = 0.0
        
        # Length contribution (longer = more suspicious)
        if length >= self.min_segment_length:
            confidence += min(0.4, length / 20.0)  # Cap at 0.4
        
        # Variance contribution (lower = more suspicious)
        if variance < self.variance_threshold:
            confidence += 0.3 * (1 - variance / self.variance_threshold)
        
        # Insulin activity contribution
        if not has_insulin_activity and length >= 4:
            confidence += 0.2
        
        # Gradient contribution
        if transition_gradient > self.gradient_threshold:
            confidence += 0.1 * min(1.0, transition_gradient / (2 * self.gradient_threshold))
        
        return min(1.0, confidence)


class Synthetic100Repairer:
    """
    Repairs synthetic 100 BG segments while preserving legitimate readings
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
    def repair_dataframe(self, df: pd.DataFrame, detector: Synthetic100Detector) -> Tuple[pd.DataFrame, Dict]:
        """
        Repair synthetic 100 values in dataframe
        
        Args:
            df: Input dataframe
            detector: Synthetic100Detector instance
            
        Returns:
            Tuple of (repaired_dataframe, repair_statistics)
        """
        if df.empty:
            return df.copy(), {}
        
        # Detect segments
        segments = detector.detect_segments(df)
        
        # Initialize repair columns if not present
        result_df = df.copy()
        repair_columns = [
            'bg_flagged_fake_100', 'bg_100_run_len', 'bg_replaced_from_fake', 'bg_100_real'
        ]
        for col in repair_columns:
            if col not in result_df.columns:
                result_df[col] = False if 'flagged' in col or 'replaced' in col or 'real' in col else 0
        
        # Apply repairs
        synthetic_segments = [s for s in segments if s.is_synthetic and s.confidence_score >= self.confidence_threshold]
        legitimate_segments = [s for s in segments if not s.is_synthetic or s.confidence_score < self.confidence_threshold]
        
        # Mark synthetic segments and replace with NaN
        total_synthetic_values = 0
        for segment in synthetic_segments:
            start_idx, end_idx = segment.start_idx, segment.end_idx
            
            # Flag as synthetic - use index-based assignment
            idx_range = result_df.index[start_idx:end_idx+1]
            result_df.loc[idx_range, 'bg_flagged_fake_100'] = True
            result_df.loc[idx_range, 'bg_100_run_len'] = segment.length
            result_df.loc[idx_range, 'bg_replaced_from_fake'] = True
            
            # Replace with NaN for imputation
            result_df.loc[idx_range, 'bg'] = np.nan
            total_synthetic_values += segment.length
            
            logger.info(f"Replaced synthetic 100-segment: indices {start_idx}-{end_idx} "
                       f"(length={segment.length}, confidence={segment.confidence_score:.3f})")
        
        # Mark legitimate 100s
        total_legitimate_100s = 0
        for segment in legitimate_segments:
            if segment.length <= 2:  # Short runs are likely legitimate
                start_idx, end_idx = segment.start_idx, segment.end_idx
                idx_range = result_df.index[start_idx:end_idx+1]
                result_df.loc[idx_range, 'bg_100_real'] = True
                total_legitimate_100s += segment.length
                
                logger.debug(f"Preserved legitimate 100-segment: indices {start_idx}-{end_idx} "
                           f"(length={segment.length}, confidence={segment.confidence_score:.3f})")
        
        # Compile statistics
        total_100_values = (df['bg'] == 100.0).sum()
        stats = {
            'total_records': len(df),
            'total_100_values': int(total_100_values),
            'synthetic_100_detected': int(total_synthetic_values),
            'legitimate_100_preserved': int(total_legitimate_100s),
            'synthetic_100_percentage': float(total_synthetic_values / max(1, total_100_values) * 100),
            'segments_analyzed': len(segments),
            'segments_flagged_synthetic': len(synthetic_segments),
            'segments_preserved_legitimate': len(legitimate_segments),
            'confidence_threshold_used': self.confidence_threshold
        }
        
        logger.info(f"Repair completed: {total_synthetic_values}/{total_100_values} "
                   f"100-values flagged as synthetic ({stats['synthetic_100_percentage']:.1f}%)")
        
        return result_df, stats
    
    def apply_gap_aware_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply gap-aware imputation to repaired dataframe
        """
        from .processors import DataProcessor
        
        processor = DataProcessor()
        
        # Ensure required imputation columns exist
        imputation_columns = ['bg_was_imputed', 'bg_impute_run_len', 'bg_hard_gap', 'bg_clip_flag']
        for col in imputation_columns:
            if col not in df.columns:
                if 'run_len' in col:
                    df[col] = 0
                else:
                    df[col] = False
        
        # Apply gap-aware imputation
        result_df = processor._apply_gap_aware_imputation(df)
        
        logger.info("Applied gap-aware imputation to repaired BG values")
        return result_df


def repair_csv_file(csv_path: str, output_path: str = None, 
                   backup_original: bool = True) -> Dict:
    """
    Repair synthetic 100 values in a CSV file
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path for output CSV (defaults to _cleaned.csv suffix)
        backup_original: Whether to backup original file
        
    Returns:
        Dictionary with repair statistics
    """
    import os
    import shutil
    
    logger.info(f"Starting repair of CSV file: {csv_path}")
    
    # Generate output path if not provided
    if output_path is None:
        base_path = csv_path.rsplit('.', 1)[0]
        output_path = f"{base_path}_cleaned.csv"
    
    # Backup original if requested
    if backup_original:
        backup_path = f"{csv_path}.backup"
        if not os.path.exists(backup_path):
            shutil.copy2(csv_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
    
    try:
        # Load CSV with comment handling for header lines
        df = pd.read_csv(csv_path, comment='#')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Initialize detector and repairer
        detector = Synthetic100Detector()
        repairer = Synthetic100Repairer()
        
        # Repair synthetic 100s
        repaired_df, stats = repairer.repair_dataframe(df, detector)
        
        # Apply gap-aware imputation
        final_df = repairer.apply_gap_aware_imputation(repaired_df)
        
        # Save repaired CSV
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved repaired CSV: {output_path}")
        
        # Add file paths to stats
        stats.update({
            'input_file': csv_path,
            'output_file': output_path,
            'backup_file': f"{csv_path}.backup" if backup_original else None
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error repairing CSV {csv_path}: {str(e)}")
        raise


def batch_repair_csvs(directory_path: str, pattern: str = "*.csv", 
                     suffix: str = "_cleaned") -> List[Dict]:
    """
    Repair all CSV files matching pattern in directory
    
    Args:
        directory_path: Directory containing CSV files
        pattern: File pattern to match
        suffix: Suffix for output files
        
    Returns:
        List of repair statistics for each file
    """
    import glob
    import os
    
    csv_files = glob.glob(os.path.join(directory_path, pattern))
    results = []
    
    logger.info(f"Found {len(csv_files)} CSV files to repair in {directory_path}")
    
    for csv_file in csv_files:
        if suffix not in csv_file:  # Skip already cleaned files
            try:
                base_name = os.path.splitext(csv_file)[0]
                output_file = f"{base_name}{suffix}.csv"
                
                stats = repair_csv_file(csv_file, output_file)
                results.append(stats)
                
                logger.info(f"Completed repair: {os.path.basename(csv_file)} -> "
                           f"{os.path.basename(output_file)}")
                           
            except Exception as e:
                logger.error(f"Failed to repair {csv_file}: {str(e)}")
                results.append({
                    'input_file': csv_file,
                    'error': str(e),
                    'success': False
                })
    
    # Summary statistics
    successful_repairs = [r for r in results if 'error' not in r]
    total_synthetic = sum(r.get('synthetic_100_detected', 0) for r in successful_repairs)
    total_100s = sum(r.get('total_100_values', 0) for r in successful_repairs)
    
    logger.info(f"Batch repair completed: {len(successful_repairs)}/{len(results)} files processed")
    logger.info(f"Total synthetic 100s detected and repaired: {total_synthetic}/{total_100s}")
    
    return results