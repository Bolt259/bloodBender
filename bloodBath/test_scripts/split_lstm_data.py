#!/usr/bin/env python3
"""
LSTM Data Splitter - Split raw bloodBank CSV files into train/validate/test sets

The raw CSV files are already in LSTM-ready format with:
- 5-minute resampling
- Feature engineering (temporal features, slopes)
- Proper BG handling (NaN for missing, clipping flags)
- v2.0 format with metadata headers

This script simply:
1. Loads all CSV files chronologically
2. Segments at large gaps (>15 hours)  
3. Splits into train (70%), validate (15%), test (15%)
4. Saves to organized directories

Usage:
    python split_lstm_data.py --pump-serial 881235
    python split_lstm_data.py --pump-serial all
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LSTM_Split')


class LSTMDataSplitter:
    """Split bloodBank CSV files into train/validate/test sets"""
    
    def __init__(self,
                 bloodbank_root: Path,
                 output_dir: Path,
                 max_gap_hours: float = 15.0,
                 train_ratio: float = 0.70,
                 validate_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        self.bloodbank_root = Path(bloodbank_root)
        self.output_dir = Path(output_dir)
        self.max_gap_hours = max_gap_hours
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.test_ratio = test_ratio
        
        self.stats = {
            'total_files': 0,
            'total_sequences': 0,
            'total_records': 0,
            'train_sequences': 0,
            'validate_sequences': 0,
            'test_sequences': 0
        }
    
    def split_pump_data(self, pump_serial: str) -> Dict[str, Any]:
        """Split data for a specific pump"""
        logger.info(f"🔄 Splitting LSTM data for pump {pump_serial}")
        
        pump_dir = self.bloodbank_root / f"pump_{pump_serial}"
        if not pump_dir.exists():
            logger.error(f"❌ Pump directory not found: {pump_dir}")
            return {'success': False, 'error': 'Directory not found'}
        
        csv_files = sorted(pump_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"❌ No CSV files found")
            return {'success': False, 'error': 'No CSV files'}
        
        logger.info(f"📁 Found {len(csv_files)} CSV files")
        
        # Load all data chronologically
        all_data_frames = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, comment='#')
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                all_data_frames.append(df)
                logger.info(f"  ✅ Loaded {csv_file.name}: {len(df)} records")
            except Exception as e:
                logger.error(f"  ❌ Error loading {csv_file.name}: {e}")
                continue
        
        if not all_data_frames:
            logger.error("❌ No data loaded")
            return {'success': False, 'error': 'No data loaded'}
        
        # Concatenate all data
        full_df = pd.concat(all_data_frames, ignore_index=True)
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"✅ Combined: {len(full_df):,} total records")
        
        # Segment at large gaps
        sequences = self._segment_by_gaps(full_df)
        logger.info(f"📊 Created {len(sequences)} sequences from gaps")
        
        # Split chronologically
        train_seqs, val_seqs, test_seqs = self._split_chronologically(sequences)
        logger.info(f"📈 Split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
        
        # Save datasets
        pump_output = self.output_dir / f"pump_{pump_serial}"
        self._save_dataset(train_seqs, pump_output / "train", pump_serial, "train")
        self._save_dataset(val_seqs, pump_output / "validate", pump_serial, "validate")
        self._save_dataset(test_seqs, pump_output / "test", pump_serial, "test")
        
        # Update stats
        self.stats['total_files'] += len(csv_files)
        self.stats['total_sequences'] += len(sequences)
        self.stats['total_records'] += len(full_df)
        self.stats['train_sequences'] += len(train_seqs)
        self.stats['validate_sequences'] += len(val_seqs)
        self.stats['test_sequences'] += len(test_seqs)
        
        return {
            'success': True,
            'pump_serial': pump_serial,
            'files': len(csv_files),
            'sequences': len(sequences),
            'records': len(full_df)
        }
    
    def _segment_by_gaps(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Segment DataFrame at temporal gaps > max_gap_hours"""
        segments = []
        current_segment_start = 0
        
        for i in range(1, len(df)):
            gap = (df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp']).total_seconds() / 3600
            
            if gap > self.max_gap_hours:
                # Save current segment
                segment = df.iloc[current_segment_start:i].copy().reset_index(drop=True)
                if len(segment) >= 12:  # Min 1 hour
                    segments.append(segment)
                current_segment_start = i
        
        # Add final segment
        final_segment = df.iloc[current_segment_start:].copy().reset_index(drop=True)
        if len(final_segment) >= 12:
            segments.append(final_segment)
        
        return segments
    
    def _split_chronologically(self, sequences: List[pd.DataFrame]) -> Tuple[List, List, List]:
        """Split sequences chronologically"""
        total = len(sequences)
        train_end = int(total * self.train_ratio)
        val_end = int(total * (self.train_ratio + self.validate_ratio))
        
        train = sequences[:train_end]
        val = sequences[train_end:val_end]
        test = sequences[val_end:]
        
        if train:
            logger.info(f"  Train: {train[0]['timestamp'].iloc[0]} to {train[-1]['timestamp'].iloc[-1]}")
        if val:
            logger.info(f"  Val:   {val[0]['timestamp'].iloc[0]} to {val[-1]['timestamp'].iloc[-1]}")
        if test:
            logger.info(f"  Test:  {test[0]['timestamp'].iloc[0]} to {test[-1]['timestamp'].iloc[-1]}")
        
        return train, val, test
    
    def _save_dataset(self, sequences: List[pd.DataFrame], output_dir: Path, 
                     pump_serial: str, dataset_type: str):
        """Save dataset sequences"""
        if not sequences:
            logger.warning(f"⚠️  No sequences for {dataset_type}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, seq_df in enumerate(sequences):
            start_time = seq_df['timestamp'].iloc[0]
            filename = f"sequence_{i:04d}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = output_dir / filename
            
            # Write with header
            with open(filepath, 'w') as f:
                f.write(f"# bloodBender LSTM Sequence\n")
                f.write(f"# pump_serial: {pump_serial}\n")
                f.write(f"# dataset: {dataset_type}\n")
                f.write(f"# sequence_id: {i}\n")
                f.write(f"# start_time: {seq_df['timestamp'].iloc[0]}\n")
                f.write(f"# end_time: {seq_df['timestamp'].iloc[-1]}\n")
                f.write(f"# length: {len(seq_df)} intervals\n")
                f.write(f"# duration_hours: {len(seq_df) * 5 / 60:.2f}\n")
                bg_coverage = seq_df['bg'].notna().sum() / len(seq_df)
                f.write(f"# bg_coverage: {bg_coverage:.2%}\n")
            
            seq_df.to_csv(filepath, mode='a', index=False)
        
        logger.info(f"  ✅ Saved {len(sequences)} sequences to {output_dir.name}")
        
        # Save summary
        summary = {
            'pump_serial': pump_serial,
            'dataset_type': dataset_type,
            'num_sequences': len(sequences),
            'total_records': sum(len(s) for s in sequences),
            'time_range': {
                'start': str(sequences[0]['timestamp'].iloc[0]),
                'end': str(sequences[-1]['timestamp'].iloc[-1])
            }
        }
        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def split_all_pumps(self):
        """Split data for all pumps"""
        logger.info("🚀 Splitting LSTM data for all pumps")
        
        pump_dirs = [d for d in self.bloodbank_root.iterdir() 
                    if d.is_dir() and d.name.startswith('pump_')]
        
        for pump_dir in pump_dirs:
            pump_serial = pump_dir.name.replace('pump_', '')
            self.split_pump_data(pump_serial)
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("🎯 LSTM Data Split Complete")
        print("="*70)
        print(f"\n📊 Statistics:")
        print(f"  Files processed:    {self.stats['total_files']}")
        print(f"  Sequences created:  {self.stats['total_sequences']}")
        print(f"  Total records:      {self.stats['total_records']:,}")
        print(f"\n📈 Dataset Split:")
        print(f"  Training:    {self.stats['train_sequences']} sequences ({self.train_ratio:.0%})")
        print(f"  Validation:  {self.stats['validate_sequences']} sequences ({self.validate_ratio:.0%})")
        print(f"  Test:        {self.stats['test_sequences']} sequences ({self.test_ratio:.0%})")
        print(f"\n📁 Output: {self.output_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Split LSTM training data')
    parser.add_argument('--pump-serial', required=True, help='Pump serial or "all"')
    parser.add_argument('--bloodbank-root', type=Path, 
                       default=Path('bloodBath/bloodBank/raw'))
    parser.add_argument('--output-dir', type=Path,
                       default=Path('bloodBath/bloodBank/lstm_pump_data'))
    parser.add_argument('--max-gap-hours', type=float, default=15.0)
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    
    args = parser.parse_args()
    
    splitter = LSTMDataSplitter(
        bloodbank_root=args.bloodbank_root,
        output_dir=args.output_dir,
        max_gap_hours=args.max_gap_hours,
        train_ratio=args.train_ratio,
        validate_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    if args.pump_serial.lower() == 'all':
        splitter.split_all_pumps()
    else:
        result = splitter.split_pump_data(args.pump_serial)
        if result['success']:
            splitter.print_summary()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
