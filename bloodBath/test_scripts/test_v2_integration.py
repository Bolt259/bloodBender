#!/usr/bin/env python3
"""
Simple integration test for bloodBath v2.0 bug fixes

Runs bloodbank_download.py for January 2024 and validates output.
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path

def run_test():
    """Run integration test"""
    
    print("=" * 70)
    print("bloodBath v2.0 Bug Fix Integration Test")
    print("=" * 70)
    
    # Test output directory
    test_dir = Path('/home/bolt/projects/bb/test_fixed_v2')
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Test directory: {test_dir}")
    print(f"📥 Downloading January 2024 for pump 881235...")
    
    # Run bloodbank_download.py
    cmd = [
        '/home/bolt/projects/bb/bloodBath-env/bin/python',
        '/home/bolt/projects/bb/bloodbank_download.py',
        '--full',  # Required flag
        '--pump', '881235',
        '--start', '2024-01-01',
        '--end', '2024-01-31',
        '--output-dir', str(test_dir)
    ]
    
    print(f"\n🚀 Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Download failed with exit code {result.returncode}")
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        return False
    
    print(f"\n✅ Download completed successfully")
    
    # Find generated CSV file
    csv_files = list(test_dir.glob('pump_881235/*.csv'))
    
    if not csv_files:
        print("❌ ERROR: No CSV file generated")
        return False
    
    csv_file = csv_files[0]
    print(f"\n📄 Generated: {csv_file.name}")
    print(f"   Size: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Validate CSV
    print("\n" + "=" * 70)
    print("VALIDATION TESTS")
    print("=" * 70)
    
    df = pd.read_csv(csv_file, comment='#')
    
    print(f"\nRecords: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    all_pass = True
    
    # Test 1: Check for new columns
    print("\n[Test 1] Check for v2.0 columns...")
    required_cols = ['bg_missing_flag', 'bg_clip_flag']
    for col in required_cols:
        if col in df.columns:
            print(f"   ✅ {col} present")
        else:
            print(f"   ❌ {col} MISSING")
            all_pass = False
    
    # Test 2: No 100-fill artifacts
    print("\n[Test 2] Check for 100-fill artifacts...")
    bg_100_count = (df['bg'] == 100.0).sum()
    bg_100_percent = bg_100_count / len(df) * 100
    
    if bg_100_count == 0:
        print(f"   ✅ No 100.0 values found")
    elif bg_100_percent < 5:
        print(f"   ✅ Only {bg_100_count} ({bg_100_percent:.1f}%) - likely real readings")
    else:
        print(f"   ❌ FAIL: {bg_100_count} ({bg_100_percent:.1f}%) - fillna bug present")
        all_pass = False
    
    # Test 3: NaN preservation
    print("\n[Test 3] Check NaN preservation...")
    nan_count = df['bg'].isna().sum()
    nan_percent = nan_count / len(df) * 100
    
    if nan_count > 0:
        print(f"   ✅ {nan_count} NaN values ({nan_percent:.1f}%)")
        
        # Check flag matches
        if 'bg_missing_flag' in df.columns:
            matches = (df['bg'].isna() == (df['bg_missing_flag'] == 1)).all()
            if matches:
                print(f"   ✅ bg_missing_flag matches NaN perfectly")
            else:
                mismatch = (df['bg'].isna() != (df['bg_missing_flag'] == 1)).sum()
                print(f"   ❌ {mismatch} mismatches between NaN and flag")
                all_pass = False
    else:
        print(f"   ⚠️  No NaN values (unusual, may indicate perfect data or ffill bug)")
    
    # Test 4: BG range
    print("\n[Test 4] Check BG range [20, 600]...")
    bg_valid = df['bg'].dropna()
    
    if len(bg_valid) > 0:
        bg_min = bg_valid.min()
        bg_max = bg_valid.max()
        print(f"   BG range: [{bg_min:.1f}, {bg_max:.1f}]")
        
        violations_low = (bg_valid < 20).sum()
        violations_high = (bg_valid > 600).sum()
        
        if violations_low > 0:
            print(f"   ❌ {violations_low} values below 20 mg/dL")
            all_pass = False
        else:
            print(f"   ✅ No values below 20")
        
        if violations_high > 0:
            print(f"   ❌ {violations_high} values above 600 mg/dL")
            all_pass = False
        else:
            print(f"   ✅ No values above 600")
    
    # Test 5: Clipping stats
    print("\n[Test 5] Check clipping...")
    if 'bg_clip_flag' in df.columns:
        clip_count = (df['bg_clip_flag'] == 1).sum()
        clip_percent = clip_count / len(df) * 100
        print(f"   Clipped: {clip_count} ({clip_percent:.2f}%)")
        
        if clip_count > 0:
            clipped_vals = df.loc[df['bg_clip_flag'] == 1, 'bg']
            at_bounds = ((clipped_vals == 20) | (clipped_vals == 600)).all()
            if at_bounds:
                print(f"   ✅ All clipped values at boundaries")
            else:
                print(f"   ⚠️  Some clipped values not at [20, 600]")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    bg_valid = df['bg'].dropna()
    print(f"Total records:     {len(df)}")
    print(f"Missing BG:        {df['bg'].isna().sum()} ({df['bg'].isna().sum()/len(df)*100:.1f}%)")
    print(f"Valid BG:          {len(bg_valid)}")
    print(f"BG range:          [{bg_valid.min():.1f}, {bg_valid.max():.1f}]")
    print(f"Mean BG:           {bg_valid.mean():.1f} ± {bg_valid.std():.1f} mg/dL")
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    return all_pass

if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)
