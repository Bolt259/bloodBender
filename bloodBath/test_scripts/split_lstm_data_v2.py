#!/usr/bin/env python3
"""
LSTM Data Splitter v2 - Chronological Time-Based Split

This script splits bloodBank CSV data chronologically by TIME (not sequences).
Since the data is continuous with no gaps, we split 70% train, 15% validate, 15% test
based on timestamps to preserve temporal ordering.

Usage:
    python split_lstm_data_v2.py --pump-serial <serial>
    python split_lstm_data_v2.py --pump-serial all
"""

import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
import logging

# Add bloodBath to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bloodBath.utils.logging_utils import setup_logger

def split_pump_data(pump_serial: str, base_dir: Path, max_gap_hours: float = 15.0):
    """
    Split pump data chronologically by time into train/validate/test sets.
    
    Args:
        pump_serial: Pump serial number
        base_dir: Base bloodBank directory
        max_gap_hours: Report gaps larger than this (for information only)
    """
    logger = logging.getLogger("LSTM_Split")
    
    logger.info(f"🔄 Splitting LSTM data for pump {pump_serial}")
    
    # Paths
    raw_dir = base_dir / "raw" / f"pump_{pump_serial}"
    output_dir = base_dir / "lstm_pump_data" / f"pump_{pump_serial}"
    
    if not raw_dir.exists():
        logger.warning(f"⚠️  Raw directory not found: {raw_dir}")
        return
    
    # Load all CSV files
    csv_files = sorted(raw_dir.glob("*.csv"))
    logger.info(f"📁 Found {len(csv_files)} CSV files")
    
    if not csv_files:
        logger.warning("⚠️  No CSV files found!")
        return
    
    # Load and concatenate all files chronologically
    all_data_frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            all_data_frames.append(df)
            logger.info(f"  ✅ Loaded {csv_file.name}: {len(df)} records")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {csv_file.name}: {e}")
    
    if not all_data_frames:
        logger.error("❌ No data loaded!")
        return
    
    # Combine and sort by timestamp
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"✅ Combined: {len(combined_df):,} total records")
    
    # Check for gaps (informational only)
    gap_count = 0
    for i in range(1, len(combined_df)):
        gap_hours = (combined_df.iloc[i]['timestamp'] - combined_df.iloc[i-1]['timestamp']).total_seconds() / 3600
        if gap_hours > max_gap_hours:
            gap_count += 1
            logger.info(f"  🔍 Gap {gap_count}: {gap_hours:.1f} hours at {combined_df.iloc[i]['timestamp']}")
    
    if gap_count == 0:
        logger.info(f"✅ No gaps > {max_gap_hours} hours - continuous data!")
    else:
        logger.info(f"📊 Found {gap_count} gaps > {max_gap_hours} hours")
    
    # Chronological split by TIME (70% train, 15% val, 15% test)
    total_records = len(combined_df)
    train_end_idx = int(total_records * 0.70)
    val_end_idx = int(total_records * 0.85)
    
    train_df = combined_df.iloc[:train_end_idx].copy()
    val_df = combined_df.iloc[train_end_idx:val_end_idx].copy()
    test_df = combined_df.iloc[val_end_idx:].copy()
    
    logger.info(f"📈 Chronological split:")
    logger.info(f"  Train: {len(train_df):,} records ({len(train_df)/total_records*100:.1f}%)")
    logger.info(f"    Range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"  Val:   {len(val_df):,} records ({len(val_df)/total_records*100:.1f}%)")
    logger.info(f"    Range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    logger.info(f"  Test:  {len(test_df):,} records ({len(test_df)/total_records*100:.1f}%)")
    logger.info(f"    Range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Create output directories
    for subset_name in ['train', 'validate', 'test']:
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    datasets = {
        'train': train_df,
        'validate': val_df,
        'test': test_df
    }
    
    summary = {
        'pump_serial': pump_serial,
        'created_at': timestamp_str,
        'total_records': total_records,
        'splits': {}
    }
    
    for subset_name, df in datasets.items():
        if len(df) == 0:
            logger.warning(f"⚠️  Empty {subset_name} dataset!")
            continue
        
        # Save CSV with metadata header
        output_file = output_dir / subset_name / f"lstm_{subset_name}_{timestamp_str}.csv"
        
        with open(output_file, 'w') as f:
            f.write(f"# bloodBath LSTM {subset_name.upper()} Dataset\n")
            f.write(f"# Pump Serial: {pump_serial}\n")
            f.write(f"# Created: {timestamp_str}\n")
            f.write(f"# Records: {len(df):,}\n")
            f.write(f"# Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"# Columns: {', '.join(df.columns)}\n")
            f.write("#\n")
            
            df.to_csv(f, index=False)
        
        logger.info(f"  ✅ Saved {len(df):,} records to {subset_name}/")
        
        summary['splits'][subset_name] = {
            'records': len(df),
            'percentage': round(len(df) / total_records * 100, 2),
            'time_start': df['timestamp'].min().isoformat(),
            'time_end': df['timestamp'].max().isoformat(),
            'file': str(output_file.relative_to(base_dir))
        }
    
    # Save summary JSON
    summary_file = output_dir / f"dataset_summary_{timestamp_str}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📝 Saved summary to {summary_file.relative_to(base_dir)}")

def main():
    parser = argparse.ArgumentParser(description="Split LSTM data chronologically")
    parser.add_argument('--pump-serial', required=True, 
                       help='Pump serial number or "all" for all pumps')
    parser.add_argument('--base-dir', type=Path,
                       default=Path(__file__).parent.parent / "bloodBank",
                       help='Base bloodBank directory')
    parser.add_argument('--max-gap-hours', type=float, default=15.0,
                       help='Report gaps larger than this (hours)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(
        name="LSTM_Split",
        log_file=Path(__file__).parent.parent / "logs" / "lstm_split.log",
        level=logging.INFO
    )
    
    # Find pumps to process
    if args.pump_serial == 'all':
        raw_dir = args.base_dir / "raw"
        if not raw_dir.exists():
            logger.error(f"❌ Raw directory not found: {raw_dir}")
            sys.exit(1)
        
        pump_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('pump_')]
        pump_serials = [d.name.replace('pump_', '') for d in pump_dirs]
        
        logger.info(f"🚀 Splitting LSTM data for {len(pump_serials)} pumps")
    else:
        pump_serials = [args.pump_serial]
    
    # Process each pump
    total_files = 0
    total_records = 0
    
    for pump_serial in pump_serials:
        split_pump_data(pump_serial, args.base_dir, args.max_gap_hours)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎯 LSTM Data Split Complete")
    logger.info("="*70)
    logger.info(f"📊 Processed {len(pump_serials)} pump(s)")
    logger.info(f"📁 Output: {args.base_dir / 'lstm_pump_data'}")
    logger.info("="*70)

if __name__ == '__main__':
    main()
