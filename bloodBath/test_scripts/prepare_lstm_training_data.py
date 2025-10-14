#!/usr/bin/env python3
"""
LSTM Training Data Preparation Script

Converts raw bloodBank CSV files to LSTM-ready training datasets with:
- Sequence segmentation at temporal gaps
- Feature engineering (temporal features, slopes, etc.)
- Per-patient normalization (z-score)
- Train/validate/test splits (70/15/15 chronological)
- Proper masking for missing BG values

Usage:
    python prepare_lstm_training_data.py --pump-serial 881235
    python prepare_lstm_training_data.py --pump-serial all
    python prepare_lstm_training_data.py --pump-serial 881235 --output-dir custom/path
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# Add bloodBath to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bloodBath.data.processors import UnifiedDataProcessor
from bloodBath.utils.logging_utils import setup_logger

logger = setup_logger(name='LSTM_Prep', level=logging.INFO)


class LSTMDatasetPreparer:
    """
    Prepare LSTM training datasets from raw bloodBank CSV files
    """
    
    def __init__(self, 
                 bloodbank_root: Path,
                 output_dir: Path,
                 train_ratio: float = 0.70,
                 validate_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        """
        Initialize LSTM dataset preparer
        
        Args:
            bloodbank_root: Path to bloodBank/raw/ directory
            output_dir: Path to save LSTM-ready datasets
            train_ratio: Training set ratio (default: 0.70)
            validate_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
        """
        self.bloodbank_root = Path(bloodbank_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.test_ratio = test_ratio
        
        # Initialize processor with spec parameters
        self.processor = UnifiedDataProcessor(
            freq='5min',
            max_gap_hours=15.0,
            max_impute_minutes=60,
            min_segment_length=12,  # 1 hour minimum
            normalization_method='z-score'
        )
        
        # Statistics
        self.stats = {
            'total_files_processed': 0,
            'total_sequences_created': 0,
            'total_records': 0,
            'train_sequences': 0,
            'validate_sequences': 0,
            'test_sequences': 0,
            'pump_stats': {}
        }
    
    def prepare_pump_data(self, pump_serial: str) -> Dict[str, Any]:
        """
        Prepare LSTM training data for a specific pump
        
        Args:
            pump_serial: Pump serial number
            
        Returns:
            Dictionary with preparation statistics
        """
        logger.info(f"🔄 Preparing LSTM data for pump {pump_serial}")
        
        # Find all CSV files for this pump
        pump_dir = self.bloodbank_root / f"pump_{pump_serial}"
        if not pump_dir.exists():
            logger.error(f"❌ Pump directory not found: {pump_dir}")
            return {'success': False, 'error': 'Directory not found'}
        
        csv_files = sorted(pump_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"❌ No CSV files found in {pump_dir}")
            return {'success': False, 'error': 'No CSV files'}
        
        logger.info(f"📁 Found {len(csv_files)} CSV files for pump {pump_serial}")
        
        # Load and concatenate all data chronologically
        all_sequences = []
        
        for csv_file in csv_files:
            logger.info(f"📄 Processing: {csv_file.name}")
            
            try:
                # Load CSV file
                df = pd.read_csv(csv_file, comment='#')
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                
                # Prepare data streams (convert to list of dicts for processor)
                cgm_data = []
                basal_data = []
                bolus_data = []
                
                for idx, row in df.iterrows():
                    ts = row['timestamp']
                    
                    # CGM data (may be NaN)
                    if pd.notna(row['bg']):
                        cgm_data.append({
                            'timestamp': ts,
                            'bg': float(row['bg'])
                        })
                    
                    # Basal data
                    if pd.notna(row['basal_rate']):
                        basal_data.append({
                            'timestamp': ts,
                            'basal_rate': float(row['basal_rate'])
                        })
                    
                    # Bolus data
                    if pd.notna(row['bolus_dose']) and row['bolus_dose'] > 0:
                        bolus_data.append({
                            'timestamp': ts,
                            'bolus_dose': float(row['bolus_dose'])
                        })
                
                # Process with UnifiedDataProcessor
                sequences = self.processor.create_unified_lstm_sequences(
                    cgm_data=cgm_data,
                    basal_data=basal_data,
                    bolus_data=bolus_data
                )
                
                all_sequences.extend(sequences)
                logger.info(f"  ✅ Created {len(sequences)} sequences from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing {csv_file.name}: {e}")
                continue
        
        if not all_sequences:
            logger.error(f"❌ No sequences created for pump {pump_serial}")
            return {'success': False, 'error': 'No sequences created'}
        
        logger.info(f"✅ Total sequences created: {len(all_sequences)}")
        
        # Split into train/validate/test chronologically
        train_seqs, val_seqs, test_seqs = self._split_sequences_chronologically(
            all_sequences
        )
        
        logger.info(f"📊 Split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
        
        # Save datasets
        pump_output_dir = self.output_dir / f"pump_{pump_serial}"
        pump_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._save_dataset(train_seqs, pump_output_dir / "train", pump_serial)
        self._save_dataset(val_seqs, pump_output_dir / "validate", pump_serial)
        self._save_dataset(test_seqs, pump_output_dir / "test", pump_serial)
        
        # Update statistics
        self.stats['total_files_processed'] += len(csv_files)
        self.stats['total_sequences_created'] += len(all_sequences)
        self.stats['train_sequences'] += len(train_seqs)
        self.stats['validate_sequences'] += len(val_seqs)
        self.stats['test_sequences'] += len(test_seqs)
        self.stats['pump_stats'][pump_serial] = {
            'files': len(csv_files),
            'sequences': len(all_sequences),
            'train': len(train_seqs),
            'validate': len(val_seqs),
            'test': len(test_seqs)
        }
        
        return {
            'success': True,
            'pump_serial': pump_serial,
            'files_processed': len(csv_files),
            'total_sequences': len(all_sequences),
            'train_sequences': len(train_seqs),
            'validate_sequences': len(val_seqs),
            'test_sequences': len(test_seqs)
        }
    
    def _split_sequences_chronologically(self, 
                                        sequences: List[Dict[str, Any]]
                                        ) -> Tuple[List, List, List]:
        """
        Split sequences into train/validate/test chronologically
        
        Args:
            sequences: List of sequence dictionaries
            
        Returns:
            Tuple of (train_sequences, validate_sequences, test_sequences)
        """
        # Sort by start time
        sorted_seqs = sorted(sequences, key=lambda s: s['metadata']['start_time'])
        
        total = len(sorted_seqs)
        train_end = int(total * self.train_ratio)
        val_end = int(total * (self.train_ratio + self.validate_ratio))
        
        train_seqs = sorted_seqs[:train_end]
        val_seqs = sorted_seqs[train_end:val_end]
        test_seqs = sorted_seqs[val_end:]
        
        logger.info(f"📅 Chronological split:")
        if train_seqs:
            logger.info(f"  Train: {train_seqs[0]['metadata']['start_time']} to {train_seqs[-1]['metadata']['end_time']}")
        if val_seqs:
            logger.info(f"  Val:   {val_seqs[0]['metadata']['start_time']} to {val_seqs[-1]['metadata']['end_time']}")
        if test_seqs:
            logger.info(f"  Test:  {test_seqs[0]['metadata']['start_time']} to {test_seqs[-1]['metadata']['end_time']}")
        
        return train_seqs, val_seqs, test_seqs
    
    def _save_dataset(self, sequences: List[Dict[str, Any]], 
                     output_dir: Path, 
                     pump_serial: str):
        """
        Save dataset sequences to directory
        
        Args:
            sequences: List of sequence dictionaries
            output_dir: Output directory path
            pump_serial: Pump serial number
        """
        if not sequences:
            logger.warning(f"⚠️  No sequences to save for {output_dir.name}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual sequence files
        for i, seq in enumerate(sequences):
            seq_df = seq['data']
            metadata = seq['metadata']
            
            # Create filename
            start_date = metadata['start_time'].strftime('%Y%m%d_%H%M%S')
            filename = f"sequence_{i:04d}_{start_date}.csv"
            filepath = output_dir / filename
            
            # Add comprehensive header with metadata
            with open(filepath, 'w') as f:
                f.write(f"# bloodBender LSTM Sequence Data\n")
                f.write(f"# ================================\n")
                f.write(f"# pump_serial: {pump_serial}\n")
                f.write(f"# sequence_id: {i}\n")
                f.write(f"# start_time: {metadata['start_time']}\n")
                f.write(f"# end_time: {metadata['end_time']}\n")
                f.write(f"# length: {metadata['length']} intervals\n")
                f.write(f"# duration_hours: {metadata['duration_hours']:.2f}\n")
                f.write(f"# bg_coverage: {metadata['bg_coverage']:.2%}\n")
                f.write(f"# gaps_imputed: {metadata.get('gaps_imputed', 0)}\n")
                f.write(f"# gaps_masked: {metadata.get('gaps_masked', 0)}\n")
                f.write(f"# normalization: per-patient z-score\n")
                f.write(f"# ================================\n")
            
            # Append data
            seq_df.to_csv(filepath, mode='a', index=False)
            
            self.stats['total_records'] += len(seq_df)
        
        logger.info(f"  ✅ Saved {len(sequences)} sequences to {output_dir}")
        
        # Save summary metadata
        summary = {
            'pump_serial': pump_serial,
            'dataset_type': output_dir.name,
            'num_sequences': len(sequences),
            'total_intervals': sum(s['metadata']['length'] for s in sequences),
            'time_range': {
                'start': str(sequences[0]['metadata']['start_time']),
                'end': str(sequences[-1]['metadata']['end_time'])
            },
            'avg_bg_coverage': np.mean([s['metadata']['bg_coverage'] for s in sequences]),
            'sequences': [
                {
                    'sequence_id': i,
                    'start_time': str(s['metadata']['start_time']),
                    'end_time': str(s['metadata']['end_time']),
                    'length': s['metadata']['length'],
                    'duration_hours': s['metadata']['duration_hours'],
                    'bg_coverage': s['metadata']['bg_coverage']
                }
                for i, s in enumerate(sequences)
            ]
        }
        
        import json
        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def prepare_all_pumps(self) -> Dict[str, Any]:
        """
        Prepare LSTM data for all pumps found in bloodBank
        
        Returns:
            Dictionary with overall statistics
        """
        logger.info("🚀 Preparing LSTM data for all pumps")
        
        # Find all pump directories
        pump_dirs = [d for d in self.bloodbank_root.iterdir() 
                    if d.is_dir() and d.name.startswith('pump_')]
        
        if not pump_dirs:
            logger.error("❌ No pump directories found")
            return {'success': False, 'error': 'No pump directories'}
        
        results = []
        for pump_dir in pump_dirs:
            pump_serial = pump_dir.name.replace('pump_', '')
            result = self.prepare_pump_data(pump_serial)
            results.append(result)
        
        return {
            'success': True,
            'pumps_processed': len(results),
            'results': results,
            'overall_stats': self.stats
        }
    
    def print_summary(self):
        """Print preparation summary"""
        print("\n" + "="*70)
        print("🎯 LSTM Training Data Preparation Complete")
        print("="*70)
        print(f"\n📊 Overall Statistics:")
        print(f"  Files processed:    {self.stats['total_files_processed']}")
        print(f"  Sequences created:  {self.stats['total_sequences_created']}")
        print(f"  Total records:      {self.stats['total_records']:,}")
        print(f"\n📈 Dataset Split:")
        print(f"  Training:    {self.stats['train_sequences']} sequences ({self.train_ratio:.0%})")
        print(f"  Validation:  {self.stats['validate_sequences']} sequences ({self.validate_ratio:.0%})")
        print(f"  Test:        {self.stats['test_sequences']} sequences ({self.test_ratio:.0%})")
        
        if self.stats['pump_stats']:
            print(f"\n🔧 Per-Pump Statistics:")
            for pump_serial, pump_stats in self.stats['pump_stats'].items():
                print(f"\n  Pump {pump_serial}:")
                print(f"    Files:      {pump_stats['files']}")
                print(f"    Sequences:  {pump_stats['sequences']}")
                print(f"    Train:      {pump_stats['train']}")
                print(f"    Validate:   {pump_stats['validate']}")
                print(f"    Test:       {pump_stats['test']}")
        
        print(f"\n📁 Output Directory: {self.output_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LSTM training data from bloodBank CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data for specific pump
  python prepare_lstm_training_data.py --pump-serial 881235

  # Prepare data for all pumps
  python prepare_lstm_training_data.py --pump-serial all

  # Custom output directory
  python prepare_lstm_training_data.py --pump-serial 881235 --output-dir /path/to/output

  # Custom split ratios
  python prepare_lstm_training_data.py --pump-serial all --train-ratio 0.8 --val-ratio 0.1
        """
    )
    
    parser.add_argument(
        '--pump-serial',
        type=str,
        required=True,
        help='Pump serial number or "all" for all pumps'
    )
    
    parser.add_argument(
        '--bloodbank-root',
        type=Path,
        default=Path('bloodBath/bloodBank/raw'),
        help='Path to bloodBank/raw directory (default: bloodBath/bloodBank/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('bloodBath/bloodBank/lstm_pump_data'),
        help='Output directory for LSTM data (default: bloodBath/bloodBank/lstm_pump_data)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio (default: 0.70)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"❌ Split ratios must sum to 1.0 (got {total_ratio})")
        return 1
    
    # Create preparer
    preparer = LSTMDatasetPreparer(
        bloodbank_root=args.bloodbank_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validate_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Prepare data
    if args.pump_serial.lower() == 'all':
        result = preparer.prepare_all_pumps()
    else:
        result = preparer.prepare_pump_data(args.pump_serial)
    
    if not result['success']:
        logger.error(f"❌ Preparation failed: {result.get('error', 'Unknown error')}")
        return 1
    
    # Print summary
    preparer.print_summary()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
