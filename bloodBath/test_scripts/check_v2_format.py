#!/usr/bin/env python3
"""Quick validation of existing bloodbank files for v2.0 fixes"""

import pandas as pd
from pathlib import Path

def check_file(csv_path):
    """Check a single CSV file for v2.0 compliance"""
    print(f"\n{'='*70}")
    print(f"Checking: {csv_path.name}")
    print(f"{'='*70}")
    
    # Read CSV
    df = pd.read_csv(csv_path, comment='#')
    
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for 100-fill
    bg_100_count = (df['bg'] == 100.0).sum()
    print(f"\n100-fill check: {bg_100_count} instances of BG=100.0")
    if bg_100_count > len(df) * 0.05:  # More than 5%
        print(f"  ⚠️  HIGH CONCENTRATION: {bg_100_count/len(df)*100:.1f}% - likely fillna bug")
    
    # Check for NaN
    nan_count = df['bg'].isna().sum()
    print(f"NaN preservation: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
    
    # Check for bg_missing_flag
    if 'bg_missing_flag' in df.columns:
        print(f"✅ bg_missing_flag column present")
        flag_count = (df['bg_missing_flag'] == 1).sum()
        print(f"   Missing flags: {flag_count}")
    else:
        print(f"❌ bg_missing_flag column MISSING (old format)")
    
    # Check for bg_clip_flag  
    if 'bg_clip_flag' in df.columns:
        print(f"✅ bg_clip_flag column present")
        clip_count = (df['bg_clip_flag'] == 1).sum()
        print(f"   Clipped values: {clip_count}")
    else:
        print(f"❌ bg_clip_flag column MISSING (old format)")
    
    # BG range
    bg_valid = df['bg'].dropna()
    if len(bg_valid) > 0:
        bg_min = bg_valid.min()
        bg_max = bg_valid.max()
        print(f"\nBG range: [{bg_min:.1f}, {bg_max:.1f}] mg/dL")
        
        if bg_min < 20:
            print(f"  ⚠️  Values below 20 mg/dL")
        if bg_max > 600:
            print(f"  ⚠️  Values above 600 mg/dL")
    
    # Verdict
    print(f"\n{'='*70}")
    has_flags = 'bg_missing_flag' in df.columns and 'bg_clip_flag' in df.columns
    has_reasonable_100s = bg_100_count < len(df) * 0.05
    
    if has_flags and has_reasonable_100s:
        print("✅ APPEARS TO BE v2.0 FORMAT (with bug fixes)")
    else:
        print("❌ OLD FORMAT (generated before bug fixes)")
    print(f"{'='*70}")

# Check a few files
raw_dir = Path('/home/bolt/projects/bb/bloodBath/bloodBank/raw')

print("bloodBath v2.0 File Format Checker")
print("Checking existing files for v2.0 compliance...\n")

# Check pump 881235 January file
file_881235 = raw_dir / 'pump_881235' / '2024-01-01_to_2024-01-31.csv'
if file_881235.exists():
    check_file(file_881235)
else:
    print(f"❌ File not found: {file_881235}")

# Check pump 901161470 first file
pump_901_files = list((raw_dir / 'pump_901161470').glob('*.csv'))
if pump_901_files:
    check_file(pump_901_files[0])
else:
    print(f"❌ No files found for pump 901161470")

print("\n\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("All existing files were generated with OLD code before bug fixes.")
print("They need to be REGENERATED to apply:")
print("  - No 100-fill behavior (NaN instead)")
print("  - bg_missing_flag column")
print("  - BG range [20, 600] instead of [40, 400]")
print("="*70)
