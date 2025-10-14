#!/usr/bin/env python3
"""
Archive old bloodBank data and prepare for regeneration with v2.0 bug fixes.

This script:
1. Creates an archive directory with timestamp
2. Moves all existing pump_* directories to archive
3. Cleans up any partial/corrupt data
4. Prepares bloodBank/raw/ for fresh regeneration
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

def archive_old_data():
    """Archive existing bloodBank data before regeneration"""
    
    print("=" * 70)
    print("bloodBath v2.0 Data Archive & Cleanup")
    print("=" * 70)
    
    # Paths
    bloodbank_dir = Path('/home/bolt/projects/bb/bloodBath/bloodBank')
    raw_dir = bloodbank_dir / 'raw'
    archives_dir = bloodbank_dir / 'archives'
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"{timestamp}_pre_v2.0_bug_fixes"
    archive_path = archives_dir / archive_name
    
    print(f"\n📦 Archive directory: {archive_path}")
    
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pump directories
    pump_dirs = list(raw_dir.glob('pump_*'))
    
    if not pump_dirs:
        print("\n✅ No existing pump directories to archive")
        return True
    
    print(f"\n📂 Found {len(pump_dirs)} pump directories:")
    for pump_dir in pump_dirs:
        csv_count = len(list(pump_dir.glob('*.csv')))
        size_mb = sum(f.stat().st_size for f in pump_dir.glob('*.csv')) / 1024 / 1024
        print(f"   • {pump_dir.name}: {csv_count} files ({size_mb:.1f} MB)")
    
    # Move each pump directory to archive
    print(f"\n📤 Moving to archive...")
    
    for pump_dir in pump_dirs:
        try:
            dest = archive_path / pump_dir.name
            shutil.move(str(pump_dir), str(dest))
            print(f"   ✅ {pump_dir.name} → {dest.name}")
        except Exception as e:
            print(f"   ❌ Failed to move {pump_dir.name}: {e}")
            return False
    
    # Check for other directories/files to clean
    print(f"\n🧹 Checking for other files to clean...")
    
    other_items = [item for item in raw_dir.iterdir() if item.is_dir() or item.is_file()]
    other_items = [item for item in other_items if not item.name.startswith('.')]
    
    if other_items:
        print(f"   Found {len(other_items)} other items:")
        for item in other_items:
            if item.is_dir():
                csv_count = len(list(item.rglob('*.csv')))
                print(f"   • {item.name}/ ({csv_count} CSVs)")
            else:
                print(f"   • {item.name}")
        
        response = input("\n   Archive these too? (y/N): ").strip().lower()
        if response == 'y':
            for item in other_items:
                try:
                    dest = archive_path / item.name
                    if item.is_dir():
                        shutil.move(str(item), str(dest))
                    else:
                        shutil.copy2(str(item), str(dest))
                    print(f"   ✅ {item.name}")
                except Exception as e:
                    print(f"   ❌ {item.name}: {e}")
    else:
        print(f"   ✅ No other items found")
    
    # Create README in archive
    readme_path = archive_path / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(f"bloodBath Data Archive\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Archived: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reason: Pre-v2.0 bug fixes regeneration\n\n")
        f.write(f"Issues with this data:\n")
        f.write(f"  - BG 100-fill bug (missing data filled with 100.0)\n")
        f.write(f"  - Forward-fill bug (gaps filled with last value)\n")
        f.write(f"  - Missing bg_missing_flag column\n")
        f.write(f"  - Missing bg_clip_flag column\n")
        f.write(f"  - Old BG range [40, 400] instead of [20, 600]\n\n")
        f.write(f"v2.0 fixes applied in regeneration:\n")
        f.write(f"  ✅ NaN for missing BG (not 100)\n")
        f.write(f"  ✅ bg_missing_flag column added\n")
        f.write(f"  ✅ bg_clip_flag column added\n")
        f.write(f"  ✅ BG range updated to [20, 600] mg/dL\n")
        f.write(f"  ✅ No forward-fill into gaps\n\n")
        f.write(f"Archived contents:\n")
        for pump_dir in pump_dirs:
            f.write(f"  - {pump_dir.name}/\n")
    
    print(f"\n📝 Created README: {readme_path.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ARCHIVE SUMMARY")
    print("=" * 70)
    print(f"Archive location: {archive_path}")
    print(f"Pump directories: {len(pump_dirs)}")
    
    total_csvs = sum(len(list((archive_path / d.name).glob('*.csv'))) for d in pump_dirs)
    total_size = sum(
        sum(f.stat().st_size for f in (archive_path / d.name).glob('*.csv'))
        for d in pump_dirs
    ) / 1024 / 1024
    
    print(f"Total CSV files:  {total_csvs}")
    print(f"Total size:       {total_size:.1f} MB")
    print(f"\n✅ Archive complete - raw/ directory ready for regeneration")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    success = archive_old_data()
    sys.exit(0 if success else 1)
