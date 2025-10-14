#!/usr/bin/env python3
"""
Test script for CSV post-processing cleanup

This script tests the automated cleanup of invalid CSV files containing
only NaN BG values and zero basal/bolus values.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bloodBath.utils.file_utils import remove_invalid_csv_files
from bloodBath.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_cleanup_test(target_dir: Path, dry_run: bool = False, threshold: float = 1.0):
    """
    Run cleanup test on specified directory
    
    Args:
        target_dir: Directory to clean
        dry_run: If True, only report what would be deleted
        threshold: Invalid ratio threshold (1.0 = 100% invalid)
    """
    print("=" * 80)
    print("bloodBath CSV Post-Processing Cleanup Test")
    print("=" * 80)
    print(f"Target directory: {target_dir}")
    print(f"Dry run mode: {dry_run}")
    print(f"Invalid threshold: {threshold * 100:.0f}% (removes files with ≥{threshold * 100:.0f}% invalid records)")
    print("")
    
    if not target_dir.exists():
        print(f"ERROR: Directory not found: {target_dir}")
        return
    
    # Count files before cleanup
    csv_files_before = list(target_dir.rglob("*.csv"))
    print(f"📊 Found {len(csv_files_before)} CSV files before cleanup")
    print("")
    
    if dry_run:
        print("🔍 DRY RUN MODE - Simulating cleanup (no files will be deleted)")
        print("")
        
        # Manually check files
        import pandas as pd
        import warnings
        warnings.filterwarnings('ignore')
        
        invalid_files = []
        
        for csv_file in csv_files_before:
            try:
                df = pd.read_csv(csv_file, comment='#')
                
                # Check for required columns
                if 'bg' in df.columns and 'basal_rate' in df.columns:
                    bolus_col = 'bolus_dose' if 'bolus_dose' in df.columns else 'bolus'
                    
                    if bolus_col in df.columns:
                        # Calculate invalid ratio
                        total_records = len(df)
                        if total_records > 0:
                            invalid_records = (
                                df['bg'].isna() & 
                                (df['basal_rate'] == 0.0) & 
                                (df[bolus_col] == 0.0)
                            ).sum()
                            invalid_ratio = invalid_records / total_records
                            
                            # Remove if ≥threshold invalid
                            if invalid_ratio >= threshold:
                                file_size = csv_file.stat().st_size
                                invalid_files.append({
                                    'path': str(csv_file.relative_to(target_dir)),
                                    'size': file_size,
                                    'records': len(df),
                                    'invalid_ratio': invalid_ratio
                                })
                                print(f"   Would remove: {csv_file.name} ({invalid_ratio*100:.1f}% invalid, {file_size:,} bytes, {len(df)} records)")
            
            except Exception as e:
                pass  # Silently skip problematic files
        
        print("")
        print(f"📊 Summary (DRY RUN):")
        print(f"   - Would remove: {len(invalid_files)} files")
        print(f"   - Total size: {sum(f['size'] for f in invalid_files):,} bytes")
        print("")
        
        return {
            'dry_run': True,
            'would_remove': len(invalid_files),
            'files': invalid_files
        }
    
    # Run actual cleanup
    print("🧹 Running actual cleanup...")
    print("")
    
    cleanup_stats = remove_invalid_csv_files(target_dir, invalid_threshold=threshold)
    
    # Count files after cleanup
    csv_files_after = list(target_dir.rglob("*.csv"))
    
    print("")
    print("=" * 80)
    print("📊 Cleanup Results")
    print("=" * 80)
    print(f"Files before:  {len(csv_files_before)}")
    print(f"Files after:   {len(csv_files_after)}")
    print(f"Files removed: {cleanup_stats['invalid_csvs_removed']}")
    print(f"Bytes freed:   {cleanup_stats['bytes_freed']:,} ({cleanup_stats['bytes_freed'] / 1024 / 1024:.2f} MB)")
    print("")
    
    if cleanup_stats['removed_files']:
        print("Removed files:")
        for f in cleanup_stats['removed_files']:
            print(f"  - {f}")
    
    print("")
    print("=" * 80)
    
    # Save cleanup metadata
    metadata_file = target_dir / 'cleanup_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(cleanup_stats, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_file}")
    
    return cleanup_stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test CSV post-processing cleanup on bloodBank data"
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='bloodBath/bloodBank/raw',
        help='Directory to clean (default: bloodBath/bloodBank/raw)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate cleanup without deleting files'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.999,
        help='Invalid ratio threshold (1.0 = 100%% invalid, 0.999 = 99.9%% invalid, 0.95 = 95%% invalid)'
    )
    
    args = parser.parse_args()
    
    # Resolve target directory
    base_dir = Path(__file__).parent.parent.parent
    target_dir = base_dir / args.directory
    
    # Run cleanup test
    results = run_cleanup_test(target_dir, dry_run=args.dry_run, threshold=args.threshold)
    
    if results:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
