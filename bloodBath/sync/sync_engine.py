"""
Sync Engine for bloodBath - Lightweight synchronization interface

Provides a simplified interface for common sync operations, 
wrapping the HarvestManager functionality.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from .harvest_manager import HarvestManager


class SyncEngine:
    """
    Simplified synchronization interface for bloodBath
    """
    
    def __init__(self, output_dir: str = "/home/bolt/projects/bb/training_data"):
        """
        Initialize sync engine
        
        Args:
            output_dir: Directory for training data output
        """
        self.output_dir = Path(output_dir)
        self.harvest_manager = HarvestManager(str(output_dir))
    
    def sync(self, 
             pump_serial: Optional[str] = None,
             force_refresh: bool = False,
             enable_validation: bool = True) -> Dict[str, Any]:
        """
        Perform smart sync operation
        
        Args:
            pump_serial: Specific pump to sync (None = all pumps)
            force_refresh: Regenerate all files even if they exist
            enable_validation: Run validation on synced data
            
        Returns:
            Sync results dictionary
        """
        print("🚀 bloodBath Smart Sync")
        print("="*50)
        
        # Update validation setting
        self.harvest_manager.enable_validation = enable_validation
        if enable_validation:
            from ..data.validators import LstmDataValidator
            self.harvest_manager.validator = LstmDataValidator()
        else:
            self.harvest_manager.validator = None
        
        if pump_serial:
            # Sync specific pump
            print(f"📡 Syncing pump {pump_serial}")
            return self.harvest_manager.sync_pump_data(
                pump_serial, 
                force_regenerate=force_refresh
            )
        else:
            # Sync all pumps
            print("📡 Syncing all pumps")
            return self.harvest_manager.sync_all_pumps()
    
    def quick_validate(self, pump_serial: Optional[str] = None) -> Dict[str, Any]:
        """
        Quick validation of existing data without syncing new data
        
        Args:
            pump_serial: Specific pump to validate (None = all pumps)
            
        Returns:
            Validation results
        """
        print("🔍 bloodBath Quick Validation")
        print("="*50)
        
        validation_results = {}
        
        if pump_serial:
            pumps = [pump_serial]
        else:
            pumps = ["881235", "901161470"]
        
        for pump in pumps:
            print(f"\n📊 Validating pump {pump}...")
            
            # Check existing files
            existing_data = self.harvest_manager.detect_existing_data(pump)
            
            # Run validation if files exist
            if existing_data:
                if self.harvest_manager.validator is None:
                    from ..data.validators import LstmDataValidator
                    self.harvest_manager.validator = LstmDataValidator()
                
                # Create a dummy sync stats dict for validation
                sync_stats = {'pump_serial': pump}
                self.harvest_manager._validate_all_monthly_files(pump, sync_stats)
                
                validation_results[pump] = {
                    'existing_files': existing_data,
                    'validation_results': sync_stats.get('validation_results', []),
                    'validation_summary': sync_stats.get('validation_summary', {})
                }
            else:
                validation_results[pump] = {
                    'existing_files': {},
                    'validation_results': [],
                    'validation_summary': {'files_validated': 0, 'files_passed': 0}
                }
                print(f"   No existing files found for pump {pump}")
        
        return validation_results
    
    def status(self) -> Dict[str, Any]:
        """
        Get current status of data availability and sync state
        
        Returns:
            Status information for all pumps
        """
        print("📊 bloodBath Status Report")
        print("="*50)
        
        status_info = {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'pumps': {}
        }
        
        # Check each pump
        for pump_serial in ["881235", "901161470"]:
            print(f"\n🔍 Checking pump {pump_serial}...")
            
            # Get data range
            try:
                start_date, end_date = self.harvest_manager.detect_pump_data_range(pump_serial)
                data_range = {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'available': start_date is not None and end_date is not None
                }
            except:
                data_range = {'available': False}
            
            # Get existing files
            existing_files = self.harvest_manager.detect_existing_data(pump_serial)
            
            status_info['pumps'][pump_serial] = {
                'data_range': data_range,
                'existing_files': existing_files,
                'file_count': len(existing_files),
                'valid_files': len([f for f in existing_files.values() if f]),
                'invalid_files': len([f for f in existing_files.values() if not f])
            }
            
            print(f"   Data range: {data_range}")
            print(f"   Files: {len(existing_files)} total, "
                  f"{status_info['pumps'][pump_serial]['valid_files']} valid")
        
        return status_info
    
    def cleanup(self, pump_serial: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up invalid or corrupted data files
        
        Args:
            pump_serial: Specific pump to clean (None = all pumps)
            dry_run: If True, only report what would be cleaned
            
        Returns:
            Cleanup results
        """
        print(f"🧹 bloodBath Cleanup {'(DRY RUN)' if dry_run else '(LIVE)'}")
        print("="*50)
        
        cleanup_results = {
            'dry_run': dry_run,
            'files_to_remove': [],
            'files_removed': [],
            'space_freed': 0
        }
        
        if pump_serial:
            pumps = [pump_serial]
        else:
            pumps = ["881235", "901161470"]
        
        for pump in pumps:
            print(f"\n🧹 Cleaning pump {pump}...")
            
            existing_files = self.harvest_manager.detect_existing_data(pump)
            monthly_dir = self.output_dir / "monthly_lstm" / f"pump_{pump}"
            
            if monthly_dir.exists():
                for month_label, is_valid in existing_files.items():
                    if not is_valid:
                        # Find the file path
                        file_path = monthly_dir / f"pump_{pump}_{month_label}.csv"
                        
                        if file_path.exists():
                            file_size = file_path.stat().st_size
                            cleanup_results['files_to_remove'].append({
                                'path': str(file_path),
                                'size': file_size,
                                'reason': 'invalid_data'
                            })
                            
                            if not dry_run:
                                try:
                                    file_path.unlink()
                                    cleanup_results['files_removed'].append(str(file_path))
                                    cleanup_results['space_freed'] += file_size
                                    print(f"   🗑️ Removed {file_path.name} ({file_size/1024:.1f}KB)")
                                except Exception as e:
                                    print(f"   ❌ Failed to remove {file_path.name}: {e}")
                            else:
                                print(f"   🗑️ Would remove {file_path.name} ({file_size/1024:.1f}KB)")
        
        total_files = len(cleanup_results['files_to_remove'])
        total_space = sum(f['size'] for f in cleanup_results['files_to_remove']) / (1024 * 1024)  # MB
        
        print(f"\n📊 Cleanup Summary:")
        print(f"   Files to clean: {total_files}")
        print(f"   Space to free: {total_space:.2f}MB")
        
        if not dry_run:
            print(f"   Files removed: {len(cleanup_results['files_removed'])}")
            print(f"   Space freed: {cleanup_results['space_freed']/(1024*1024):.2f}MB")
        
        return cleanup_results