#!/usr/bin/env python3
"""
Quick test of bloodBath v2.0 bug fixes

Tests:
1. No 100-fill behavior (BG=100 should not appear from fillna)
2. NaN preserved in missing data
3. bg_missing_flag matches NaN locations
4. BG range [20, 600]
5. bg_clip_flag correct
"""

import sys
import pandas as pd
from pathlib import Path

# Add bloodbank_download to path (it's in project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bloodbank_download import BloodbankDownloader

def test_single_month():
    """Test January 2024 for pump 881235"""
    
    print("=" * 70)
    print("bloodBath v2.0 Bug Fix Integration Test")
    print("=" * 70)
    
    # Create test output directory
    test_dir = Path('/home/bolt/projects/bb/test_fixed_v2')
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {test_dir}")
    
    # Initialize downloader
    downloader = BloodbankDownloader(
        output_dir=str(test_dir),
        email='your-email@example.com',  # Will use cached credentials
        password='dummy'  # Not used if cached
    )
    
    print("\n📥 Downloading January 2024 for pump 881235...")
    
    try:
        # Download one month
        downloader.download_pump_data(
            pump_serial='881235',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        print("\n✅ Download completed")
        
        # Find generated file
        csv_files = list(test_dir.glob('pump_881235/*.csv'))
        
        if not csv_files:
            print("❌ ERROR: No CSV file generated")
            return False
        
        csv_file = csv_files[0]
        print(f"\n📄 Generated file: {csv_file.name}")
        print(f"   Size: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Read CSV (skip comment lines)
        print("\n🔍 Running validation tests...")
        df = pd.read_csv(csv_file, comment='#')
        
        print(f"   Records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Test 1: No 100-fill artifacts
        print("\n[Test 1] Checking for 100-fill artifacts...")
        bg_100_count = (df['bg'] == 100.0).sum()
        if bg_100_count == 0:
            print("   ✅ PASS: No 100.0 values found")
        else:
            print(f"   ⚠️  WARNING: Found {bg_100_count} instances of BG=100.0")
            # Check if these coincide with missing flag
            bg_100_missing = ((df['bg'] == 100.0) & (df['bg_missing_flag'] == 1)).sum()
            if bg_100_missing > 0:
                print(f"   ❌ FAIL: {bg_100_missing} instances of 100.0 with missing flag")
                return False
            else:
                print("   ✅ PASS: 100.0 values are real CGM readings (missing_flag=0)")
        
        # Test 2: NaN present
        print("\n[Test 2] Checking for NaN preservation...")
        nan_count = df['bg'].isna().sum()
        if nan_count > 0:
            print(f"   ✅ PASS: {nan_count} NaN values preserved ({nan_count/len(df)*100:.1f}%)")
        else:
            print("   ⚠️  WARNING: No NaN values (may indicate perfect CGM data or ffill bug)")
        
        # Test 3: bg_missing_flag matches NaN
        print("\n[Test 3] Validating bg_missing_flag...")
        if 'bg_missing_flag' not in df.columns:
            print("   ❌ FAIL: bg_missing_flag column missing")
            return False
        
        flag_matches = (df['bg'].isna() == (df['bg_missing_flag'] == 1)).all()
        if flag_matches:
            print("   ✅ PASS: bg_missing_flag perfectly matches NaN locations")
        else:
            mismatch_count = (df['bg'].isna() != (df['bg_missing_flag'] == 1)).sum()
            print(f"   ❌ FAIL: {mismatch_count} mismatches between NaN and missing_flag")
            return False
        
        # Test 4: BG range [20, 600]
        print("\n[Test 4] Checking BG range [20, 600]...")
        bg_valid = df['bg'].dropna()
        bg_min = bg_valid.min()
        bg_max = bg_valid.max()
        
        print(f"   BG range: [{bg_min:.1f}, {bg_max:.1f}]")
        
        if bg_min < 20:
            print(f"   ❌ FAIL: BG values below 20 mg/dL: {(bg_valid < 20).sum()}")
            return False
        elif bg_min < 40:
            print(f"   ✅ PASS: BG min {bg_min:.1f} in [20, 40) range (hypoglycemia)")
        else:
            print(f"   ✅ PASS: BG min {bg_min:.1f} >= 40")
        
        if bg_max > 600:
            print(f"   ❌ FAIL: BG values above 600 mg/dL: {(bg_valid > 600).sum()}")
            return False
        elif bg_max > 400:
            print(f"   ✅ PASS: BG max {bg_max:.1f} in (400, 600] range (severe hyperglycemia)")
        else:
            print(f"   ✅ PASS: BG max {bg_max:.1f} <= 400")
        
        # Test 5: bg_clip_flag consistency
        print("\n[Test 5] Validating bg_clip_flag...")
        if 'bg_clip_flag' not in df.columns:
            print("   ❌ FAIL: bg_clip_flag column missing")
            return False
        
        clip_count = (df['bg_clip_flag'] == 1).sum()
        print(f"   Clipped values: {clip_count} ({clip_count/len(df)*100:.2f}%)")
        
        if clip_count > 0:
            # Check that clipped values are at boundaries
            clipped_values = df.loc[df['bg_clip_flag'] == 1, 'bg']
            at_boundaries = ((clipped_values == 20) | (clipped_values == 600)).all()
            
            if at_boundaries:
                print(f"   ✅ PASS: All clipped values at boundaries [20, 600]")
            else:
                print(f"   ⚠️  WARNING: Some clipped values not at boundaries")
                print(f"      Clipped BG range: [{clipped_values.min():.1f}, {clipped_values.max():.1f}]")
        else:
            print("   ✅ PASS: No clipping needed (all values within range)")
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total records:     {len(df)}")
        print(f"Missing BG:        {nan_count} ({nan_count/len(df)*100:.1f}%)")
        print(f"Clipped BG:        {clip_count} ({clip_count/len(df)*100:.2f}%)")
        print(f"BG range:          [{bg_min:.1f}, {bg_max:.1f}] mg/dL")
        print(f"Mean BG:           {bg_valid.mean():.1f} mg/dL")
        print(f"Std BG:            {bg_valid.std():.1f} mg/dL")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_single_month()
    sys.exit(0 if success else 1)
