"""
Unified multi-stream data processing for LSTM training data preparation.

Enhanced to use the new unified resampling system with proper sequence segmentation,
gap detection, feature engineering, and mask generation for LSTM training.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from ..utils.time_utils import generate_time_index, add_time_of_day_features, handle_duplicate_timestamps
from ..core.exceptions import DataValidationError
from .resampler import UnifiedResampler, SequenceSegmenter, BgImputer, FeatureEngineer

logger = logging.getLogger(__name__)


class UnifiedDataProcessor:
    """
    Unified data processor that creates LSTM-ready sequences with proper segmentation,
    gap handling, feature engineering, and masking for training.
    """
    
    def __init__(self, 
                 freq: str = '5min',
                 max_gap_hours: float = 15.0,
                 max_impute_minutes: int = 60,
                 min_segment_length: int = 12,
                 normalization_method: str = 'z-score'):
        """
        Initialize the unified data processor
        
        Args:
            freq: Resampling frequency (default: '5min')
            max_gap_hours: Maximum gap before breaking sequences (default: 15 hours)
            max_impute_minutes: Maximum gap size to impute (default: 60 minutes)
            min_segment_length: Minimum sequence length to keep (default: 12 intervals = 1 hour)
            normalization_method: Feature normalization method (default: 'z-score')
        """
        self.freq = freq
        self.max_gap_hours = max_gap_hours
        self.max_impute_minutes = max_impute_minutes
        self.min_segment_length = min_segment_length
        self.normalization_method = normalization_method
        
        # Initialize components
        self.resampler = UnifiedResampler(freq=freq)
        self.segmenter = SequenceSegmenter(max_gap_hours=max_gap_hours, min_segment_length=min_segment_length)
        self.imputer = BgImputer(max_impute_minutes=max_impute_minutes)
        self.feature_engineer = FeatureEngineer(normalization_method=normalization_method)
        
        # Processing statistics
        self.processing_stats = {
            'total_intervals': 0,
            'total_segments': 0,
            'bg_coverage': 0.0,
            'basal_coverage': 0.0,
            'bolus_events': 0,
            'gaps_imputed': 0,
            'gaps_masked': 0,
            'features_engineered': 0
        }
    
    def create_unified_lstm_sequences(self, 
                                    cgm_data: List[Dict[str, Any]],
                                    basal_data: List[Dict[str, Any]],
                                    bolus_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create unified LSTM-ready sequences with proper segmentation and masking.
        
        This method implements the complete pipeline:
        1. Create unified 5-minute master time index
        2. Align all streams with proper aggregation rules
        3. Detect temporal gaps and segment into continuous blocks
        4. Apply BG imputation and masking
        5. Engineer derived features
        6. Normalize features per-patient
        7. Output segmented sequences ready for LSTM training
        
        Args:
            cgm_data: List of normalized CGM events with 'timestamp' and 'bg' fields
            basal_data: List of normalized basal events with 'timestamp' and 'basal_rate' fields
            bolus_data: List of normalized bolus events with 'timestamp' and 'bolus_dose' fields
            
        Returns:
            List of sequence dictionaries, each containing:
            - 'data': DataFrame with aligned features and masks
            - 'metadata': Sequence metadata (start_time, end_time, length, etc.)
        """
        logger.info("Creating unified LSTM-ready sequences...")
        
        # Step 1: Create unified master time index
        master_index = self.resampler.create_unified_master_index(cgm_data, basal_data, bolus_data)
        
        if master_index.empty:
            logger.warning("No valid time range found, returning empty sequence list")
            return []
        
        # Step 2: Align all streams to master index
        aligned_streams = self.resampler.align_streams_to_master_index(
            cgm_data, basal_data, bolus_data, master_index
        )
        
        # Step 3: Detect and repair synthetic 100 values before segmentation
        aligned_streams = self._repair_synthetic_100_in_streams(aligned_streams)
        
        # Step 4: Segment by temporal gaps
        segments = self.segmenter.segment_by_gaps(aligned_streams)
        
        if not segments:
            logger.warning("No valid segments created after gap detection")
            return []
        
        # Step 5: Process each segment individually
        lstm_sequences = []
        
        for i, segment_data in enumerate(segments):
            logger.debug(f"Processing segment {i+1}/{len(segments)}")
            
            # Apply BG imputation and masking
            processed_segment = self._process_segment(segment_data)
            
            if processed_segment is not None:
                lstm_sequences.append(processed_segment)
        
        # Update statistics
        self._calculate_processing_stats(lstm_sequences)
        
        logger.info(f"Created {len(lstm_sequences)} LSTM-ready sequences from {len(segments)} segments")
        return lstm_sequences
    
    def _repair_synthetic_100_in_streams(self, aligned_streams: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Detect and repair synthetic 100 BG values in aligned streams
        
        Args:
            aligned_streams: Dictionary of aligned pandas Series
            
        Returns:
            Dictionary with repaired BG stream
        """
        if 'bg' not in aligned_streams:
            return aligned_streams
        
        # Convert BG series to DataFrame for repair
        bg_df = pd.DataFrame({'bg': aligned_streams['bg']})
        
        # Apply synthetic 100 repair
        repaired_df = self._repair_synthetic_100_values(bg_df)
        
        # Update the BG stream
        aligned_streams['bg'] = repaired_df['bg']
        
        return aligned_streams
    
    def _process_segment(self, segment_data: Dict[str, pd.Series]) -> Optional[Dict[str, Any]]:
        """
        Process a single segment for LSTM training
        
        Args:
            segment_data: Dictionary with aligned series for the segment
            
        Returns:
            Dictionary with processed segment data and metadata, or None if segment is invalid
        """
        # Extract time information
        time_index = list(segment_data.values())[0].index
        segment_length = len(time_index)
        
        if segment_length < self.min_segment_length:
            logger.debug(f"Skipping short segment: {segment_length} < {self.min_segment_length}")
            return None
        
        # Apply BG imputation and masking
        if 'bg' in segment_data:
            bg_imputed, mask_bg, mask_label = self.imputer.impute_bg_gaps(segment_data['bg'])
            segment_data['bg'] = bg_imputed
            segment_data['mask_bg'] = mask_bg
            segment_data['mask_label'] = mask_label
        else:
            # No BG data - create empty masks
            segment_data['mask_bg'] = pd.Series(True, index=time_index)  # All missing
            segment_data['mask_label'] = pd.Series(False, index=time_index)  # Exclude from training
        
        # Add derived features
        enhanced_data = self.feature_engineer.add_derived_features(segment_data)
        
        # Normalize features
        normalized_data = self.feature_engineer.normalize_features(enhanced_data)
        
        # Convert to DataFrame
        segment_df = pd.DataFrame(normalized_data)
        segment_df['timestamp'] = time_index
        
        # Reorder columns for LSTM training
        lstm_columns = self._get_lstm_column_order(segment_df.columns)
        segment_df = segment_df[lstm_columns]
        
        # Create metadata
        metadata = {
            'start_time': time_index[0],
            'end_time': time_index[-1], 
            'length': segment_length,
            'duration_hours': segment_length * 5 / 60,  # Convert intervals to hours
            'bg_coverage': segment_data['bg'].notna().sum() / segment_length if 'bg' in segment_data else 0.0,
            'gaps_imputed': self.imputer.imputation_stats.get('gaps_imputed', 0),
            'gaps_masked': self.imputer.imputation_stats.get('gaps_masked', 0),
            'feature_stats': self.feature_engineer.get_feature_stats()
        }
        
        return {
            'data': segment_df,
            'metadata': metadata
        }
    
    def _get_lstm_column_order(self, columns) -> List[str]:
        """
        Define the standard column order for LSTM training
        
        Args:
            columns: Available columns
            
        Returns:
            List of column names in desired order
        """
        # Standard LSTM column order
        standard_order = [
            'timestamp',
            'bg',
            'basal_rate', 
            'bolus_dose',
            'basal_delta',
            'time_since_last_bolus',
            'bg_slope_15min',
            'bg_slope_30min', 
            'sin_time',
            'cos_time',
            'mask_bg',
            'mask_label'
        ]
        
        # Include only columns that exist
        ordered_columns = [col for col in standard_order if col in columns]
        
        # Add any remaining columns not in standard order
        remaining_columns = [col for col in columns if col not in ordered_columns]
        ordered_columns.extend(remaining_columns)
        
        return ordered_columns
    
    def _calculate_processing_stats(self, lstm_sequences: List[Dict[str, Any]]):
        """
        Calculate processing statistics from completed sequences
        
        Args:
            lstm_sequences: List of processed LSTM sequences
        """
        if not lstm_sequences:
            return
        
        total_intervals = sum(seq['metadata']['length'] for seq in lstm_sequences)
        total_segments = len(lstm_sequences)
        
        # Calculate coverage statistics
        bg_coverage_sum = sum(seq['metadata']['bg_coverage'] * seq['metadata']['length'] for seq in lstm_sequences)
        avg_bg_coverage = bg_coverage_sum / total_intervals if total_intervals > 0 else 0.0
        
        # Count bolus events and basal coverage
        bolus_events = 0
        basal_coverage = 0
        
        for seq in lstm_sequences:
            df = seq['data']
            if 'bolus_dose' in df.columns:
                bolus_events += (df['bolus_dose'] > 0).sum()
            if 'basal_rate' in df.columns:
                basal_coverage += (df['basal_rate'] > 0).sum()
        
        # Update statistics
        self.processing_stats.update({
            'total_intervals': total_intervals,
            'total_segments': total_segments,
            'bg_coverage': avg_bg_coverage,
            'basal_coverage': basal_coverage / total_intervals if total_intervals > 0 else 0.0,
            'bolus_events': bolus_events,
            'gaps_imputed': sum(seq['metadata']['gaps_imputed'] for seq in lstm_sequences),
            'gaps_masked': sum(seq['metadata']['gaps_masked'] for seq in lstm_sequences),
            'features_engineered': len(self.feature_engineer.get_feature_stats())
        })
    
    def save_lstm_sequences(self, 
                           lstm_sequences: List[Dict[str, Any]], 
                           output_dir: Path, 
                           pump_id: str, 
                           month: str) -> List[Path]:
        """
        Save LSTM sequences as structured CSV files with standardized format and metadata
        
        CSV Format: timestamp,bg,basal,bolus,features,mask_bg,mask_label
        where 'features' is a JSON-encoded string containing all derived features
        
        Args:
            lstm_sequences: List of LSTM sequences to save
            output_dir: Output directory path
            pump_id: Pump identifier
            month: Month identifier (YYYY_MM format)
            
        Returns:
            List of created file paths
        """
        if not lstm_sequences:
            logger.warning("No LSTM sequences to save")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, sequence in enumerate(lstm_sequences):
            # Create filename
            sequence_id = f"{pump_id}_{month}_seq_{i:03d}"
            csv_file = output_dir / f"{sequence_id}.csv"
            metadata_file = output_dir / f"{sequence_id}_metadata.json"
            
            # Transform data to required CSV format
            df = sequence['data']
            lstm_ready_df = self._format_lstm_csv(df)
            
            # Add header comment with sequence info
            metadata = sequence['metadata']
            header_lines = [
                "# bloodBath LSTM Sequence Dataset",
                f"# Pump ID: {pump_id}",
                f"# Month: {month}",
                f"# Sequence: {i+1}/{len(lstm_sequences)}",
                f"# Start Time: {metadata['start_time']}",
                f"# End Time: {metadata['end_time']}",
                f"# Length: {metadata['length']} intervals ({metadata['duration_hours']:.1f} hours)",
                f"# BG Coverage: {metadata['bg_coverage']:.1%}",
                f"# Gaps Imputed: {metadata['gaps_imputed']}",
                f"# Gaps Masked: {metadata['gaps_masked']}",
                f"# Format: timestamp,bg,basal,bolus,features,mask_bg,mask_label",
                f"# Features JSON Keys: {', '.join(self._get_feature_column_names(df))}",
                f"# Generated: {datetime.now().isoformat()}"
            ]
            
            with open(csv_file, 'w') as f:
                f.write('\n'.join(header_lines) + '\n')
                lstm_ready_df.to_csv(f, index=False)
            
            # Save comprehensive metadata JSON
            metadata_json = self._create_comprehensive_metadata(
                sequence, pump_id, month, i, len(lstm_sequences), lstm_ready_df
            )
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_json, f, indent=2)
            
            saved_files.extend([csv_file, metadata_file])
            logger.debug(f"Saved sequence {i+1}: {csv_file}")
        
        logger.info(f"Saved {len(lstm_sequences)} LSTM sequences to {output_dir}")
        return saved_files
    
    def _format_lstm_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform processed DataFrame to LSTM-ready CSV format.
        
        Required columns: timestamp,bg,basal,bolus,features,mask_bg,mask_label
        
        Args:
            df: Processed DataFrame with all features
            
        Returns:
            DataFrame in LSTM-ready format
        """
        # Define core data columns
        core_columns = ['timestamp', 'bg', 'basal_rate', 'bolus_dose', 'mask_bg', 'mask_label']
        
        # Identify feature columns (exclude core columns and masks)
        exclude_cols = set(['timestamp', 'bg', 'basal_rate', 'bolus_dose', 'mask_bg', 'mask_label'])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create LSTM-ready DataFrame
        lstm_df = pd.DataFrame()
        
        # Add required columns in exact order
        lstm_df['timestamp'] = df['timestamp']
        lstm_df['bg'] = df['bg']
        lstm_df['basal'] = df['basal_rate']  # Rename to 'basal' for consistency
        lstm_df['bolus'] = df['bolus_dose']   # Rename to 'bolus' for consistency
        
        # Encode features as JSON strings
        features_list = []
        for idx in df.index:
            feature_dict = {}
            for col in feature_cols:
                value = df.at[idx, col]
                # Convert numpy types to native Python types for JSON serialization
                if pd.isna(value):
                    feature_dict[col] = None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    feature_dict[col] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    feature_dict[col] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    feature_dict[col] = bool(value)
                else:
                    feature_dict[col] = value
            
            features_list.append(json.dumps(feature_dict, separators=(',', ':')))
        
        lstm_df['features'] = features_list
        lstm_df['mask_bg'] = df['mask_bg']
        lstm_df['mask_label'] = df['mask_label']
        
        return lstm_df
    
    def _get_feature_column_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names for documentation"""
        exclude_cols = {'timestamp', 'bg', 'basal_rate', 'bolus_dose', 'mask_bg', 'mask_label'}
        return [col for col in df.columns if col not in exclude_cols]
    
    def _create_comprehensive_metadata(self, 
                                     sequence: Dict[str, Any], 
                                     pump_id: str, 
                                     month: str, 
                                     sequence_idx: int, 
                                     total_sequences: int,
                                     lstm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create comprehensive metadata JSON for LSTM sequence
        
        Args:
            sequence: Original sequence dictionary
            pump_id: Pump identifier
            month: Month identifier
            sequence_idx: Zero-based sequence index
            total_sequences: Total number of sequences
            lstm_df: LSTM-ready DataFrame
            
        Returns:
            Comprehensive metadata dictionary
        """
        metadata = sequence['metadata']
        
        # Calculate additional statistics from LSTM-ready data
        lstm_stats = self._calculate_lstm_stats(lstm_df)
        
        metadata_json = {
            'sequence_info': {
                'pump_id': pump_id,
                'month': month,
                'sequence_number': sequence_idx + 1,
                'total_sequences': total_sequences,
                'sequence_id': f"{pump_id}_{month}_seq_{sequence_idx:03d}"
            },
            'temporal_info': {
                'start_time': metadata['start_time'].isoformat(),
                'end_time': metadata['end_time'].isoformat(),
                'length_intervals': metadata['length'],
                'duration_hours': metadata['duration_hours'],
                'frequency': self.freq,
                'interval_minutes': 5
            },
            'data_quality': {
                'bg_coverage': metadata['bg_coverage'],
                'gaps_imputed': metadata['gaps_imputed'],
                'gaps_masked': metadata['gaps_masked'],
                'lstm_quality_score': lstm_stats['quality_score'],
                'bg_completeness': lstm_stats['bg_completeness'],
                'basal_completeness': lstm_stats['basal_completeness'],
                'bolus_activity': lstm_stats['bolus_activity']
            },
            'lstm_format': {
                'columns': ['timestamp', 'bg', 'basal', 'bolus', 'features', 'mask_bg', 'mask_label'],
                'feature_keys': self._get_feature_column_names(sequence['data']),
                'total_features': len(self._get_feature_column_names(sequence['data'])),
                'shape': [len(lstm_df), 7],  # rows x columns
                'tensor_shape': [len(lstm_df), len(self._get_feature_column_names(sequence['data'])) + 3]  # T x (F + bg + basal + bolus)
            },
            'value_ranges': {
                'bg_min': float(lstm_df['bg'].min()) if lstm_df['bg'].notna().any() else None,
                'bg_max': float(lstm_df['bg'].max()) if lstm_df['bg'].notna().any() else None,
                'basal_min': float(lstm_df['basal'].min()) if lstm_df['basal'].notna().any() else None,
                'basal_max': float(lstm_df['basal'].max()) if lstm_df['basal'].notna().any() else None,
                'bolus_total': float(lstm_df['bolus'].sum()) if lstm_df['bolus'].notna().any() else 0.0
            },
            'processing_config': {
                'max_gap_hours': self.max_gap_hours,
                'max_impute_minutes': self.max_impute_minutes,
                'min_segment_length': self.min_segment_length,
                'normalization_method': self.normalization_method
            },
            'feature_stats': metadata['feature_stats'],
            'validation_ready': {
                'uniform_spacing': True,  # Guaranteed by processing
                'required_columns': all(col in lstm_df.columns for col in 
                                      ['timestamp', 'bg', 'basal', 'bolus', 'features', 'mask_bg', 'mask_label']),
                'lstm_tensor_ready': lstm_stats['tensor_ready'],
                'training_recommendation': lstm_stats['training_recommendation']
            },
            'generated_at': datetime.now().isoformat(),
            'bloodbath_version': '2.0.0-lstm',
            'format_version': '1.0'
        }
        
        return metadata_json
    
    def _calculate_lstm_stats(self, lstm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate LSTM-specific quality statistics
        
        Args:
            lstm_df: LSTM-ready DataFrame
            
        Returns:
            Dictionary of LSTM statistics
        """
        total_intervals = len(lstm_df)
        
        # BG completeness (non-masked BG values)
        bg_available = (~lstm_df['mask_bg']).sum() if 'mask_bg' in lstm_df.columns else 0
        bg_completeness = bg_available / total_intervals if total_intervals > 0 else 0.0
        
        # Basal completeness (non-null basal values) 
        basal_available = lstm_df['basal'].notna().sum() if 'basal' in lstm_df.columns else 0
        basal_completeness = basal_available / total_intervals if total_intervals > 0 else 0.0
        
        # Bolus activity (percentage of intervals with bolus > 0)
        bolus_active = (lstm_df['bolus'] > 0).sum() if 'bolus' in lstm_df.columns else 0
        bolus_activity = bolus_active / total_intervals if total_intervals > 0 else 0.0
        
        # Overall quality score
        quality_score = (bg_completeness * 0.6 + basal_completeness * 0.3 + 
                        min(bolus_activity * 5, 0.1) * 1.0)  # Bolus activity capped at 10%
        
        # Training readiness assessment
        tensor_ready = (total_intervals >= 12 and bg_completeness > 0.3 and basal_completeness > 0.8)
        
        if quality_score > 0.8 and bg_completeness > 0.7:
            training_recommendation = "EXCELLENT"
        elif quality_score > 0.6 and bg_completeness > 0.5:
            training_recommendation = "GOOD"
        elif quality_score > 0.4 and bg_completeness > 0.3:
            training_recommendation = "FAIR"
        else:
            training_recommendation = "POOR"
        
        return {
            'quality_score': round(quality_score, 3),
            'bg_completeness': round(bg_completeness, 3),
            'basal_completeness': round(basal_completeness, 3),
            'bolus_activity': round(bolus_activity, 3),
            'tensor_ready': tensor_ready,
            'training_recommendation': training_recommendation
        }
    
    def process_pump_data_for_lstm(self, 
                                  pump_data_dir: Path, 
                                  output_dir: Path, 
                                  pump_id: str,
                                  target_months: Optional[List[str]] = None,
                                  validate_output: bool = True) -> Dict[str, Any]:
        """
        Process complete pump dataset for LSTM training with validation
        
        Args:
            pump_data_dir: Directory containing pump CSV files (basal, bolus, cgm)
            output_dir: Output directory for LSTM sequences
            pump_id: Pump identifier
            target_months: List of months to process (YYYY_MM format), or None for all
            validate_output: Whether to validate LSTM sequences before saving
            
        Returns:
            Processing summary with statistics and file paths
        """
        from .validators import LstmDataValidator
        
        pump_data_dir = Path(pump_data_dir)
        output_dir = Path(output_dir)
        
        logger.info(f"Processing pump {pump_id} data from {pump_data_dir}")
        
        # Discover available data files
        data_files = self._discover_pump_data_files(pump_data_dir, pump_id, target_months)
        
        if not data_files:
            logger.warning(f"No data files found for pump {pump_id}")
            return {'status': 'no_data', 'pump_id': pump_id, 'files_processed': 0}
        
        # Initialize validator if requested
        validator = LstmDataValidator() if validate_output else None
        
        processing_summary = {
            'pump_id': pump_id,
            'months_processed': [],
            'total_sequences': 0,
            'valid_sequences': 0,
            'saved_files': [],
            'validation_reports': {},
            'processing_errors': []
        }
        
        # Process each month
        for month, files_dict in data_files.items():
            logger.info(f"Processing month {month} for pump {pump_id}")
            
            try:
                # Create LSTM sequences for this month
                lstm_sequences = self.create_lstm_sequences_from_files(
                    files_dict['basal'], 
                    files_dict['bolus'],
                    files_dict['cgm']
                )
                
                if not lstm_sequences:
                    logger.warning(f"No valid sequences created for {pump_id} month {month}")
                    continue
                
                # Validate sequences if requested
                valid_sequences = lstm_sequences
                validation_report = None
                
                if validator:
                    valid_sequences, validation_report = validator.validate_lstm_sequences(lstm_sequences)
                    processing_summary['validation_reports'][month] = validation_report
                    logger.info(f"Validation: {len(valid_sequences)}/{len(lstm_sequences)} sequences passed")
                
                # Save sequences
                if valid_sequences:
                    month_output_dir = output_dir / pump_id / month
                    saved_files = self.save_lstm_sequences(
                        valid_sequences, month_output_dir, pump_id, month
                    )
                    processing_summary['saved_files'].extend(saved_files)
                
                # Update summary
                processing_summary['months_processed'].append(month)
                processing_summary['total_sequences'] += len(lstm_sequences)
                processing_summary['valid_sequences'] += len(valid_sequences)
                
            except Exception as e:
                error_msg = f"Error processing {pump_id} month {month}: {str(e)}"
                logger.error(error_msg)
                processing_summary['processing_errors'].append(error_msg)
        
        # Generate final summary
        processing_summary.update({
            'status': 'completed' if processing_summary['months_processed'] else 'failed',
            'success_rate': (processing_summary['valid_sequences'] / 
                           max(processing_summary['total_sequences'], 1)),
            'months_total': len(data_files),
            'months_successful': len(processing_summary['months_processed']),
            'generated_at': datetime.now().isoformat()
        })
        
        logger.info(f"Completed pump {pump_id}: {processing_summary['valid_sequences']} valid sequences from {processing_summary['total_sequences']} total")
        return processing_summary
    
    def create_lstm_sequences_from_files(self, 
                                       basal_file: Path, 
                                       bolus_file: Path, 
                                       cgm_file: Path) -> List[Dict[str, Any]]:
        """
        Create LSTM sequences from CSV files (wrapper for create_unified_lstm_sequences)
        
        Args:
            basal_file: Path to basal CSV file with 'timestamp' and 'basal_rate' columns
            bolus_file: Path to bolus CSV file with 'timestamp' and 'bolus_dose' columns
            cgm_file: Path to CGM CSV file with 'timestamp' and 'bg' columns
            
        Returns:
            List of LSTM sequence dictionaries
        """
        logger.info(f"Loading data from files: {basal_file.name}, {bolus_file.name}, {cgm_file.name}")
        
        # Load and convert CSV files to required format
        cgm_data = self._load_csv_to_dict_list(cgm_file, ['timestamp', 'bg'])
        basal_data = self._load_csv_to_dict_list(basal_file, ['timestamp', 'basal_rate'])
        bolus_data = self._load_csv_to_dict_list(bolus_file, ['timestamp', 'bolus_dose'])
        
        # Create LSTM sequences using the loaded data
        return self.create_unified_lstm_sequences(cgm_data, basal_data, bolus_data)
    
    def _load_csv_to_dict_list(self, file_path: Path, required_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Load CSV file and convert to list of dictionaries format with column mapping
        
        Args:
            file_path: Path to CSV file
            required_columns: List of required column names
            
        Returns:
            List of dictionaries with timestamp and data columns
        """
        try:
            df = pd.read_csv(file_path)
            
            # Define column mappings for different file types
            column_mappings = {
                'cgm': {
                    'created_at': 'timestamp',
                    'sgv': 'bg'
                },
                'basal': {
                    'created_at': 'timestamp', 
                    'value': 'basal_rate'
                },
                'bolus': {
                    'created_at': 'timestamp',
                    'bolus': 'bolus_dose'
                }
            }
            
            # Determine file type based on filename or columns
            file_type = None
            if 'cgm' in file_path.name.lower() or 'sgv' in df.columns:
                file_type = 'cgm'
            elif 'basal' in file_path.name.lower() or 'value' in df.columns:
                file_type = 'basal'  
            elif 'bolus' in file_path.name.lower() or 'bolus' in df.columns:
                file_type = 'bolus'
            
            # Apply column mapping if available
            if file_type and file_type in column_mappings:
                mapping = column_mappings[file_type]
                df = df.rename(columns=mapping)
                logger.info(f"Applied {file_type} column mapping for {file_path.name}")
            
            # Check required columns after mapping
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                # Show available columns for debugging
                available_cols = list(df.columns)
                raise DataValidationError(
                    f"Missing columns in {file_path.name}: {missing_columns}. "
                    f"Available columns: {available_cols}"
                )
            
            # Convert to list of dictionaries
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in required_columns:
                    if col == 'timestamp':
                        # Parse timestamp and convert to UTC to avoid timezone issues
                        timestamp = pd.to_datetime(row[col])
                        if timestamp.tz is not None:
                            timestamp = timestamp.tz_convert('UTC').tz_localize(None)
                        record[col] = timestamp
                    else:
                        value = row[col]
                        if pd.notna(value):
                            record[col] = float(value)
                        else:
                            record[col] = 0.0 if col != 'bg' else None  # BG can be None for gaps
                records.append(record)
            
            logger.info(f"Loaded {len(records)} records from {file_path.name}")
            return records
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise DataValidationError(f"Failed to load {file_path}: {e}")
    
    def _discover_pump_data_files(self, 
                                pump_data_dir: Path, 
                                pump_id: str,
                                target_months: Optional[List[str]] = None) -> Dict[str, Dict[str, Path]]:
        """
        Discover and organize pump data files by month
        
        Args:
            pump_data_dir: Directory containing pump CSV files
            pump_id: Pump identifier for file filtering
            target_months: List of target months (YYYY_MM), or None for all
            
        Returns:
            Dictionary organized as {month: {stream_type: file_path}}
        """
        data_files = {}
        
        # Look for files matching patterns: basal_YYYYMMDD_*.csv, bolus_YYYYMMDD_*.csv, cgmreading_YYYYMMDD_*.csv
        for file_path in pump_data_dir.glob("*.csv"):
            filename = file_path.name
            
            # Extract date and type from filename
            month_match = None
            stream_type = None
            
            if filename.startswith('basal_') and 'basal' not in filename[6:12]:
                stream_type = 'basal'
                month_match = filename[6:14]  # YYYYMMDD
            elif filename.startswith('bolus_'):
                stream_type = 'bolus' 
                month_match = filename[6:14]
            elif filename.startswith('cgmreading_'):
                stream_type = 'cgm'
                month_match = filename[11:19]
            
            if month_match and len(month_match) == 8:
                # Convert YYYYMMDD to YYYY_MM
                year_month = f"{month_match[:4]}_{month_match[4:6]}"
                
                # Filter by target months if specified
                if target_months and year_month not in target_months:
                    continue
                
                # Initialize month entry
                if year_month not in data_files:
                    data_files[year_month] = {}
                
                data_files[year_month][stream_type] = file_path
        
        # Filter out incomplete months (need all three stream types)
        complete_months = {}
        for month, files_dict in data_files.items():
            if all(stream in files_dict for stream in ['basal', 'bolus', 'cgm']):
                complete_months[month] = files_dict
            else:
                missing = [stream for stream in ['basal', 'bolus', 'cgm'] if stream not in files_dict]
                logger.warning(f"Skipping incomplete month {month}: missing {missing}")
        
        logger.info(f"Discovered {len(complete_months)} complete months for pump {pump_id}")
        return complete_months
    
    def _repair_synthetic_100_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and repair synthetic 100 BG values before gap-aware imputation
        
        Args:
            df: DataFrame with potential synthetic 100 values
            
        Returns:
            DataFrame with synthetic 100s replaced with NaN
        """
        try:
            from .repair import Synthetic100Detector, Synthetic100Repairer
            
            if df.empty or 'bg' not in df.columns:
                return df
            
            # Check if we have any 100 values to analyze
            bg_100_count = (df['bg'] == 100.0).sum()
            if bg_100_count == 0:
                logger.debug("No BG=100 values found, skipping synthetic repair")
                return df
            
            logger.info(f"Analyzing {bg_100_count} BG=100 values for synthetic patterns...")
            
            # Initialize detector and repairer
            detector = Synthetic100Detector()
            repairer = Synthetic100Repairer()
            
            # Detect and repair synthetic segments
            repaired_df, stats = repairer.repair_dataframe(df, detector)
            
            # Log repair statistics
            if stats.get('synthetic_100_detected', 0) > 0:
                logger.info(f"Synthetic 100 repair: {stats['synthetic_100_detected']}/{stats['total_100_values']} "
                           f"values flagged as synthetic ({stats['synthetic_100_percentage']:.1f}%)")
            else:
                logger.info("No synthetic 100 patterns detected - all 100 values preserved")
            
            return repaired_df
        
        except ImportError:
            logger.warning("Synthetic 100 repair module not available, skipping repair")
            return df
        except Exception as e:
            logger.warning(f"Error in synthetic 100 repair: {e}, skipping repair")
            return df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics
        
        Returns:
            Dictionary with processing statistics from all components
        """
        stats = self.processing_stats.copy()
        
        # Add component statistics
        stats['resampler'] = self.resampler.get_resampling_stats()
        stats['segmenter'] = self.segmenter.get_segmentation_stats()
        stats['imputer'] = self.imputer.get_imputation_stats()
        stats['feature_engineer'] = self.feature_engineer.get_feature_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset all processing statistics"""
        self.processing_stats = {
            'total_intervals': 0,
            'total_segments': 0,
            'bg_coverage': 0.0,
            'basal_coverage': 0.0,
            'bolus_events': 0,
            'gaps_imputed': 0,
            'gaps_masked': 0,
            'features_engineered': 0
        }
        
        # Reset component statistics
        self.resampler.reset_stats()
        self.segmenter.reset_stats()
        self.imputer.reset_stats()
        self.feature_engineer.reset_stats()


# Keep the original DataProcessor class for backward compatibility
class DataProcessor(UnifiedDataProcessor):
    """
    Legacy DataProcessor class that wraps UnifiedDataProcessor for backward compatibility.
    Maintains the original interface while using the new unified processing system.
    """
    
    def __init__(self, freq: str = '5min'):
        """
        Initialize with legacy interface
        
        Args:
            freq: Resampling frequency (default: '5min')
        """
        super().__init__(freq=freq)
        
        # Legacy stats format
        self.stats = {
            'total_intervals': 0,
            'bg_coverage': 0.0,
            'basal_coverage': 0.0,
            'bolus_events': 0
        }
    
    def create_lstm_ready_data(self, 
                              cgm_data: List[Dict[str, Any]],
                              basal_data: List[Dict[str, Any]],
                              bolus_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create LSTM-ready dataset (legacy interface)
        
        Returns the first sequence as a DataFrame for backward compatibility
        """
        sequences = self.create_unified_lstm_sequences(cgm_data, basal_data, bolus_data)
        
        if not sequences:
            logger.warning("No sequences created, returning empty DataFrame")
            return pd.DataFrame()
        
        # Return the first sequence as DataFrame
        first_sequence = sequences[0]['data']
        
        # Update legacy stats
        self.stats = {
            'total_intervals': len(first_sequence),
            'bg_coverage': first_sequence['bg'].notna().sum() / len(first_sequence) if 'bg' in first_sequence.columns else 0.0,
            'basal_coverage': (first_sequence['basal_rate'] > 0).sum() / len(first_sequence) if 'basal_rate' in first_sequence.columns else 0.0,
            'bolus_events': (first_sequence['bolus_dose'] > 0).sum() if 'bolus_dose' in first_sequence.columns else 0
        }
        
        return first_sequence
    
    def _create_dataframe(self, data: List[Dict[str, Any]], value_col: str) -> pd.DataFrame:
        """
        Create DataFrame from normalized event data
        
        Args:
            data: List of normalized events
            value_col: Name of the value column
            
        Returns:
            DataFrame with timestamp index
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Filter out events with None timestamps
        df = df[df['timestamp'].notna()]
        
        if df.empty:
            logger.warning(f"No valid timestamps found in {value_col} data")
            return pd.DataFrame()
        
        # Handle duplicate timestamps
        df = handle_duplicate_timestamps(df, 'timestamp')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # Keep only the value column
        if value_col in df.columns:
            df = df[[value_col]]
        
        return df
    
    def _create_time_index(self, cgm_df: pd.DataFrame, basal_df: pd.DataFrame, bolus_df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Create regular time index from all data sources
        
        Args:
            cgm_df: CGM DataFrame
            basal_df: Basal DataFrame
            bolus_df: Bolus DataFrame
            
        Returns:
            DatetimeIndex with regular intervals
        """
        all_timestamps = []
        
        # Collect all timestamps
        for df in [cgm_df, basal_df, bolus_df]:
            if not df.empty and df.index.notna().any():
                # Only include valid timestamps (not None or NaT)
                valid_timestamps = df.index.dropna()
                if not valid_timestamps.empty:
                    all_timestamps.extend(valid_timestamps.tolist())
        
        if not all_timestamps:
            logger.warning("No valid timestamps found in any data source")
            return pd.DatetimeIndex([], dtype='datetime64[ns]')
        
        # Create time index
        start_time = pd.to_datetime(min(all_timestamps))
        end_time = pd.to_datetime(max(all_timestamps))
        
        # Validate timestamp range
        if start_time.year < 2008 or end_time.year > 2030:
            logger.warning(f"Suspicious timestamp range: {start_time} to {end_time}")
            return pd.DatetimeIndex([], dtype='datetime64[ns]')
        
        time_index = generate_time_index(start_time, end_time, self.freq)
        
        logger.debug(f"Created time index: {start_time} to {end_time} ({len(time_index)} intervals)")
        return time_index
    
    def _aggregate_and_align(self, 
                           cgm_df: pd.DataFrame, 
                           basal_df: pd.DataFrame, 
                           bolus_df: pd.DataFrame,
                           time_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """
        Resample and align data to regular intervals
        
        Args:
            cgm_df: CGM DataFrame
            basal_df: Basal DataFrame
            bolus_df: Bolus DataFrame
            time_index: Target time index
            
        Returns:
            Dictionary with aligned series
        """
        aligned_data = {}
        
        # BG: Forward fill with reasonable limits (30 minutes = 6 intervals)
        if not cgm_df.empty:
            bg_resampled = cgm_df['bg'].resample(self.freq).mean()
            aligned_data['bg'] = bg_resampled.reindex(time_index, method='ffill', limit=6)
        else:
            aligned_data['bg'] = pd.Series(index=time_index, dtype=float)
        
        # Basal rate: Forward fill (rates persist until changed)
        if not basal_df.empty:
            basal_resampled = basal_df['basal_rate'].resample(self.freq).mean()
            aligned_data['basal_rate'] = basal_resampled.reindex(time_index, method='ffill').fillna(0.0)
        else:
            aligned_data['basal_rate'] = pd.Series(0.0, index=time_index)
        
        # Bolus dose: Sum within each interval
        if not bolus_df.empty:
            bolus_resampled = bolus_df['bolus_dose'].resample(self.freq).sum()
            aligned_data['bolus_dose'] = bolus_resampled.reindex(time_index, fill_value=0.0)
        else:
            aligned_data['bolus_dose'] = pd.Series(0.0, index=time_index)
        
        logger.debug("Completed data aggregation and alignment")
        return aligned_data
    
    def _add_features(self, aligned_data: Dict[str, pd.Series], time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Add derived features to create final LSTM-ready dataset
        
        Args:
            aligned_data: Dictionary with aligned series
            time_index: Time index
            
        Returns:
            DataFrame with all features
        """
        # Create result DataFrame
        df = pd.DataFrame(index=time_index)
        
        # Basic features - NO artificial BG defaults
        df['bg'] = aligned_data['bg']  # Keep NaN for missing BG data
        df['basal_rate'] = aligned_data['basal_rate'].fillna(0.0)
        df['bolus_dose'] = aligned_data['bolus_dose'].fillna(0.0)
        
        # Detect and repair synthetic 100 values before imputation
        df = self._repair_synthetic_100_values(df)
        
        # Apply gap-aware BG imputation
        df = self._apply_gap_aware_imputation(df)
        
        # Calculate delta_bg (glucose change) after imputation
        df['delta_bg'] = df['bg'].diff().fillna(0.0)
        
        # Add time-of-day features
        sin_time, cos_time = add_time_of_day_features(time_index)
        df['sin_time'] = sin_time
        df['cos_time'] = cos_time
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Select final columns in desired order (including imputation flags)
        final_columns = ['timestamp', 'bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time']
        
        # Include imputation flags if they exist
        flag_columns = ['bg_was_imputed', 'bg_impute_run_len', 'bg_hard_gap', 'bg_clip_flag']
        for col in flag_columns:
            if col in df.columns:
                final_columns.append(col)
        
        df = df[final_columns]
        
        logger.debug("Added features and finalized dataset structure")
        return df
    
    def _apply_gap_aware_imputation(self, df: pd.DataFrame, max_gap_bins: int = 12) -> pd.DataFrame:
        """
        Apply gap-aware BG imputation with proper flagging
        
        Args:
            df: DataFrame with BG data containing NaN values
            max_gap_bins: Maximum gap size to interpolate (default: 12 bins = 60 minutes)
            
        Returns:
            DataFrame with imputed BG values and quality flags
        """
        logger.debug("Applying gap-aware BG imputation...")
        
        # Initialize imputation flags
        df['bg_was_imputed'] = False
        df['bg_impute_run_len'] = 0
        df['bg_hard_gap'] = False
        df['bg_clip_flag'] = False
        
        # Track statistics before imputation
        missing_before = df['bg'].isna().sum()
        total_records = len(df)
        
        if missing_before > 0:
            logger.info(f"Missing BG before imputation: {missing_before}/{total_records} ({missing_before/total_records*100:.1f}%)")
            
            # Find contiguous NaN runs
            is_nan = df['bg'].isna()
            nan_runs = self._find_contiguous_runs(is_nan)
            
            for start_idx, end_idx in nan_runs:
                gap_length = end_idx - start_idx + 1
                
                if gap_length <= max_gap_bins:
                    # Try to interpolate short gaps
                    self._interpolate_gap(df, start_idx, end_idx, gap_length)
                else:
                    # Mark as hard gap (too long to interpolate)
                    idx_range = df.index[start_idx:end_idx+1]
                    df.loc[idx_range, 'bg_hard_gap'] = True
                    logger.debug(f"Hard gap marked: {gap_length} bins at index {start_idx}-{end_idx}")
        
        # Apply edge-limited forward/backward fill (≤3 bins)
        self._apply_edge_fill(df, max_edge_bins=3)
        
        # Clip BG values to physiological range and flag clipping
        self._clip_bg_values(df, min_bg=20, max_bg=600)
        
        # Final statistics
        missing_after = df['bg'].isna().sum()
        imputed_count = df['bg_was_imputed'].sum()
        
        logger.info(f"BG imputation complete: {imputed_count} values imputed, {missing_after} still missing")
        
        return df
    
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
    
    def _interpolate_gap(self, df: pd.DataFrame, start_idx: int, end_idx: int, gap_length: int):
        """Interpolate BG values in a gap using PCHIP or linear interpolation"""
        # Check if we have valid bounding values
        has_left_bound = start_idx > 0 and pd.notna(df.iloc[start_idx - 1]['bg'])
        has_right_bound = end_idx < len(df) - 1 and pd.notna(df.iloc[end_idx + 1]['bg'])
        
        if has_left_bound and has_right_bound:
            try:
                # Use PCHIP interpolation if scipy is available
                from scipy.interpolate import PchipInterpolator
                
                # Get bounding indices and values
                left_idx = start_idx - 1
                right_idx = end_idx + 1
                
                x_known = np.array([left_idx, right_idx])
                y_known = np.array([df.iloc[left_idx]['bg'], df.iloc[right_idx]['bg']])
                x_interp = np.arange(start_idx, end_idx + 1)
                
                # PCHIP interpolation
                interpolator = PchipInterpolator(x_known, y_known)
                interpolated_values = interpolator(x_interp)
                
                logger.debug(f"PCHIP interpolation for gap of {gap_length} bins")
                
            except ImportError:
                # Fallback to linear interpolation
                left_val = df.iloc[start_idx - 1]['bg']
                right_val = df.iloc[end_idx + 1]['bg']
                
                interpolated_values = np.linspace(left_val, right_val, gap_length + 2)[1:-1]
                logger.debug(f"Linear interpolation for gap of {gap_length} bins (scipy not available)")
            
            # Apply interpolated values using index-based assignment
            idx_range = df.index[start_idx:end_idx+1]
            df.loc[idx_range, 'bg'] = interpolated_values
            df.loc[idx_range, 'bg_was_imputed'] = True
            df.loc[idx_range, 'bg_impute_run_len'] = gap_length
            
        else:
            logger.debug(f"Cannot interpolate gap at {start_idx}-{end_idx}: missing bounding values")
    
    def _repair_synthetic_100_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and repair synthetic 100 BG values before gap-aware imputation
        
        Args:
            df: DataFrame with potential synthetic 100 values
            
        Returns:
            DataFrame with synthetic 100s replaced with NaN
        """
        from .repair import Synthetic100Detector, Synthetic100Repairer
        
        if df.empty or 'bg' not in df.columns:
            return df
        
        # Check if we have any 100 values to analyze
        bg_100_count = (df['bg'] == 100.0).sum()
        if bg_100_count == 0:
            logger.debug("No BG=100 values found, skipping synthetic repair")
            return df
        
        logger.info(f"Analyzing {bg_100_count} BG=100 values for synthetic patterns...")
        
        # Initialize detector and repairer
        detector = Synthetic100Detector()
        repairer = Synthetic100Repairer()
        
        # Detect and repair synthetic segments
        repaired_df, stats = repairer.repair_dataframe(df, detector)
        
        # Log repair statistics
        if stats.get('synthetic_100_detected', 0) > 0:
            logger.info(f"Synthetic 100 repair: {stats['synthetic_100_detected']}/{stats['total_100_values']} "
                       f"values flagged as synthetic ({stats['synthetic_100_percentage']:.1f}%)")
        else:
            logger.info("No synthetic 100 patterns detected - all 100 values preserved")
        
        return repaired_df
    
    def _apply_edge_fill(self, df: pd.DataFrame, max_edge_bins: int = 3):
        """Apply limited forward/backward fill at edges"""
        # Forward fill from beginning (limit to max_edge_bins)
        first_valid_idx = df['bg'].first_valid_index()
        if first_valid_idx is not None:
            # Convert to integer position for comparison
            first_valid_pos = df.index.get_loc(first_valid_idx)
            if isinstance(first_valid_pos, int) and first_valid_pos <= max_edge_bins and first_valid_pos > 0:
                fill_mask = df.index < first_valid_idx
                fill_count = fill_mask.sum()
                if fill_count > 0:
                    # Use iloc with integer position to get value
                    first_valid_value = df.iloc[first_valid_pos]['bg']
                    df.loc[fill_mask, 'bg'] = first_valid_value
                    df.loc[fill_mask, 'bg_was_imputed'] = True
                    df.loc[fill_mask, 'bg_impute_run_len'] = fill_count
                    logger.debug(f"Forward filled {fill_count} edge values")
        
        # Backward fill from end (limit to max_edge_bins)
        last_valid_idx = df['bg'].last_valid_index()
        if last_valid_idx is not None:
            # Convert to integer position for comparison
            last_valid_pos = df.index.get_loc(last_valid_idx)
            if isinstance(last_valid_pos, int):
                end_gap = len(df) - 1 - last_valid_pos
                if end_gap > 0 and end_gap <= max_edge_bins:
                    fill_mask = df.index > last_valid_idx
                    fill_count = fill_mask.sum()
                    if fill_count > 0:
                        # Use iloc with integer position to get value
                        last_valid_value = df.iloc[last_valid_pos]['bg']
                        df.loc[fill_mask, 'bg'] = last_valid_value
                        df.loc[fill_mask, 'bg_was_imputed'] = True
                        df.loc[fill_mask, 'bg_impute_run_len'] = fill_count
                        logger.debug(f"Backward filled {fill_count} edge values")
    
    def _clip_bg_values(self, df: pd.DataFrame, min_bg: float = 20, max_bg: float = 600):
        """Clip BG values to physiological range and flag clipping"""
        bg_mask = df['bg'].notna()
        
        # Identify values that need clipping
        too_low = (df['bg'] < min_bg) & bg_mask
        too_high = (df['bg'] > max_bg) & bg_mask
        
        clipping_applied = too_low | too_high
        
        if clipping_applied.any():
            # Apply clipping
            df.loc[too_low, 'bg'] = min_bg
            df.loc[too_high, 'bg'] = max_bg
            df.loc[clipping_applied, 'bg_clip_flag'] = True
            
            clip_count = clipping_applied.sum()
            logger.info(f"Clipped {clip_count} BG values to range [{min_bg}, {max_bg}]")
    
    def _calculate_stats(self, df: pd.DataFrame):
        """
        Calculate processing statistics
        
        Args:
            df: Final LSTM-ready DataFrame
        """
        self.stats['total_intervals'] = len(df)
        
        if len(df) > 0:
            self.stats['bg_coverage'] = df['bg'].notna().sum() / len(df)
            self.stats['basal_coverage'] = (df['basal_rate'] > 0).sum() / len(df)
            self.stats['bolus_events'] = (df['bolus_dose'] > 0).sum()
            
            # Add imputation statistics
            if 'bg_was_imputed' in df.columns:
                self.stats['bg_imputed_count'] = df['bg_was_imputed'].sum()
                self.stats['bg_hard_gaps'] = df['bg_hard_gap'].sum()
                self.stats['bg_clipped_count'] = df['bg_clip_flag'].sum()
        else:
            self.stats['bg_coverage'] = 0.0
            self.stats['basal_coverage'] = 0.0
            self.stats['bolus_events'] = 0
            self.stats['bg_imputed_count'] = 0
            self.stats['bg_hard_gaps'] = 0
            self.stats['bg_clipped_count'] = 0
    
    def create_lstm_ready_from_csv(self, 
                                  cgm_df: pd.DataFrame, 
                                  basal_df: pd.DataFrame, 
                                  bolus_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create LSTM-ready dataset from CSV dataframes
        
        Args:
            cgm_df: CGM DataFrame with 'created_at' and 'sgv' columns
            basal_df: Basal DataFrame with 'created_at' and 'value' columns
            bolus_df: Bolus DataFrame with 'created_at' and 'bolus' columns
            
        Returns:
            DataFrame with LSTM-ready format
        """
        logger.info("Creating LSTM-ready dataset from CSV data...")
        
        # Get time range from CGM data
        if cgm_df.empty:
            logger.error("No CGM data available")
            return pd.DataFrame()
        
        start_time = cgm_df['created_at'].min()
        end_time = cgm_df['created_at'].max()
        
        logger.info(f"Time range: {start_time} to {end_time}")
        
        # Create time index
        time_index = generate_time_index(start_time, end_time, self.freq)
        logger.info(f"Created {len(time_index)} {self.freq} intervals")
        
        # Process each data type
        cgm_processed = self._process_csv_data(cgm_df, 'created_at', 'sgv', 'bg')
        basal_processed = self._process_csv_data(basal_df, 'created_at', 'value', 'basal_rate')
        bolus_processed = self._process_csv_data(bolus_df, 'created_at', 'bolus', 'bolus_dose')
        
        # Create result DataFrame
        result_df = pd.DataFrame(index=time_index)
        result_df.index.name = 'timestamp'
        
        # Resample and align data
        logger.info("Resampling data to regular intervals...")
        
        # BG: Forward fill within reasonable limits (30 minutes)
        if not cgm_processed.empty:
            bg_resampled = cgm_processed['bg'].resample(self.freq).mean()
            result_df['bg'] = bg_resampled.reindex(time_index, method='ffill', limit=6)
        else:
            result_df['bg'] = np.nan
        
        # Basal rate: Forward fill (persist until changed)
        if not basal_processed.empty:
            basal_resampled = basal_processed['basal_rate'].resample(self.freq).mean()
            result_df['basal_rate'] = basal_resampled.reindex(time_index, method='ffill')
        else:
            result_df['basal_rate'] = 0.0
        
        # Bolus dose: Sum within each interval
        if not bolus_processed.empty:
            bolus_resampled = bolus_processed['bolus_dose'].resample(self.freq).sum()
            result_df['bolus_dose'] = bolus_resampled.reindex(time_index, fill_value=0.0)
        else:
            result_df['bolus_dose'] = 0.0
        
        # Calculate delta_bg
        result_df['delta_bg'] = result_df['bg'].diff().fillna(0.0)
        
        # Add time-of-day features
        logger.info("Adding time-of-day features...")
        
        time_values = pd.to_datetime(result_df.index)
        minutes_from_midnight = (time_values.hour * 60 + time_values.minute)
        radians = 2 * np.pi * minutes_from_midnight / 1440
        
        result_df['sin_time'] = np.sin(radians)
        result_df['cos_time'] = np.cos(radians)
        
        # Apply gap-aware BG imputation (no artificial defaults)
        result_df = self._apply_gap_aware_imputation(result_df)
        
        # Fill non-BG missing values
        result_df['basal_rate'] = result_df['basal_rate'].fillna(0.0)
        result_df['bolus_dose'] = result_df['bolus_dose'].fillna(0.0)
        
        # Recalculate delta_bg after imputation
        result_df['delta_bg'] = result_df['bg'].diff().fillna(0.0)
        
        # Reset index and reorder columns
        result_df = result_df.reset_index()
        result_df = result_df[['timestamp', 'bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time']]
        
        logger.info(f"Created LSTM-ready dataset with {len(result_df)} intervals")
        return result_df
    
    def _process_csv_data(self, df: pd.DataFrame, time_col: str, value_col: str, output_col: str) -> pd.DataFrame:
        """
        Process CSV data for resampling
        
        Args:
            df: Input DataFrame
            time_col: Timestamp column name
            value_col: Value column name
            output_col: Output column name
            
        Returns:
            Processed DataFrame with timestamp index
        """
        if df.empty or value_col not in df.columns:
            return pd.DataFrame()
        
        processed = df[[time_col, value_col]].copy()
        processed = processed.rename(columns={value_col: output_col})
        processed = processed.set_index(time_col)
        processed = processed.sort_index()
        
        return processed
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_intervals': 0,
            'bg_coverage': 0.0,
            'basal_coverage': 0.0,
            'bolus_events': 0
        }
