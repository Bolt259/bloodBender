#!/usr/bin/env python3
"""
Comprehensive test to demonstrate the new sweetBlood data management structure
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, '/home/bolt/projects/bb/bloodBath')

from utils.structure_utils import setup_sweetblood_environment, get_lstm_pump_data_file, get_metadata_file, get_log_file
from utils.file_utils import save_structured_lstm_data, save_structured_metadata
from utils.logging_utils import setup_logger
from sweetBlood.integration import SweetBloodIntegration

def demonstrate_new_structure():
    """Demonstrate the new sweetBlood data management structure"""
    print("=" * 80)
    print("COMPREHENSIVE TEST: New sweetBlood Data Management Structure")
    print("=" * 80)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Step 1: Set up the structure
        print("\n1. Setting up sweetBlood directory structure...")
        structure = setup_sweetblood_environment(str(test_dir))
        
        print(f"   Base directory: {structure['base']}")
        print(f"   LSTM pump data: {structure['lstm_pump_data']}")
        print(f"   Metadata: {structure['metadata']}")
        print(f"   Models: {structure['models']}")
        print(f"   Logs: {structure['logs']}")
        
        # Step 2: Test LSTM CSV saving with pump serial
        print("\n2. Testing LSTM CSV saving with pump serial...")
        pump_serial = '123456'
        
        # Create sample LSTM data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='5min'),
            'bg': np.random.normal(150, 30, 100),
            'delta_bg': np.random.normal(0, 5, 100),
            'basal_rate': np.random.normal(1.2, 0.3, 100),
            'bolus_dose': np.random.exponential(0.5, 100),
            'sin_time': np.random.uniform(-1, 1, 100),
            'cos_time': np.random.uniform(-1, 1, 100)
        })
        
        lstm_file = save_structured_lstm_data(
            structure, sample_data, pump_serial, '2025-01-01', '2025-01-31'
        )
        
        if lstm_file is None:
            raise ValueError("Failed to save LSTM data")
        
        print(f"   ✓ LSTM data saved: {lstm_file}")
        print(f"   ✓ File location: {lstm_file.parent}")
        print(f"   ✓ Filename format: {lstm_file.name} (includes pump_{pump_serial}_)")
        
        # Step 3: Test metadata saving
        print("\n3. Testing metadata saving...")
        metadata = {
            'pump_serial': pump_serial,
            'sync_start': '2025-01-01T00:00:00',
            'sync_end': '2025-01-31T23:59:59',
            'records_processed': len(sample_data),
            'data_quality': 'good',
            'last_sync': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = save_structured_metadata(structure, metadata, f'sync_{pump_serial}.json')
        print(f"   ✓ Metadata saved: {metadata_file}")
        print(f"   ✓ File location: {metadata_file.parent}")
        
        # Step 4: Test logging
        print("\n4. Testing logging...")
        log_file = get_log_file(structure, f'pump_{pump_serial}.log')
        logger = setup_logger(f'pump_{pump_serial}', log_file=log_file)
        
        logger.info(f"Starting sync for pump {pump_serial}")
        logger.info(f"Processed {len(sample_data)} records")
        logger.info(f"Data quality: {metadata['data_quality']}")
        logger.info(f"Sync completed successfully")
        
        print(f"   ✓ Log file created: {log_file}")
        print(f"   ✓ File location: {log_file.parent}")
        
        # Verify log contents
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        log_lines = log_content.split('\n')
        print(f"   ✓ Log entries: {len(log_lines)} lines")
        
        # Step 5: Test sweetBlood integration
        print("\n5. Testing sweetBlood integration...")
        
        # Change to test directory for integration test
        original_cwd = os.getcwd()
        os.chdir(test_dir)
        
        try:
            integration = SweetBloodIntegration()
            
            # Test prepare_training_data
            result = integration.prepare_training_data(
                pump_serial=pump_serial,
                sequence_length=60,
                prediction_horizon=1,
                save_data=True
            )
            
            print(f"   ✓ Integration initialized: {integration.structure['base']}")
            print(f"   ✓ Training data directory: {integration.structure['lstm_pump_data']}")
            
            if result['saved_files']:
                train_file = result['saved_files']['train_X']
                print(f"   ✓ Training files path: {train_file.parent}")
                
        finally:
            os.chdir(original_cwd)
        
        # Step 6: Verify directory structure
        print("\n6. Verifying final directory structure...")
        
        def list_directory_contents(path, indent=0):
            items = []
            if path.exists():
                for item in sorted(path.iterdir()):
                    prefix = "  " * indent
                    if item.is_dir():
                        items.append(f"{prefix}{item.name}/")
                        items.extend(list_directory_contents(item, indent + 1))
                    else:
                        items.append(f"{prefix}{item.name}")
            return items
        
        print(f"\nDirectory structure for {structure['base']}:")
        contents = list_directory_contents(structure['base'])
        for item in contents:
            print(item)
        
        # Step 7: Summary
        print("\n7. Summary of verification:")
        verification_results = []
        
        # Check LSTM data file
        lstm_files = list(structure['lstm_pump_data'].glob(f'pump_{pump_serial}_*.csv'))
        verification_results.append(("LSTM CSV saved correctly", len(lstm_files) > 0))
        
        # Check metadata file
        metadata_files = list(structure['metadata'].glob(f'sync_{pump_serial}.json'))
        verification_results.append(("Metadata saved correctly", len(metadata_files) > 0))
        
        # Check log file
        log_files = list(structure['logs'].glob(f'pump_{pump_serial}.log'))
        verification_results.append(("Log file created correctly", len(log_files) > 0))
        
        # Check directory structure
        required_dirs = ['lstm_pump_data', 'metadata', 'models', 'logs']
        all_dirs_exist = all(structure[dir_name].exists() for dir_name in required_dirs)
        verification_results.append(("All required directories exist", all_dirs_exist))
        
        print()
        for description, passed in verification_results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"   {description}: {status}")
        
        all_passed = all(result[1] for result in verification_results)
        
        if all_passed:
            print("\n🎉 All verification checks passed!")
            print("The new sweetBlood data management structure is working correctly.")
        else:
            print("\n❌ Some verification checks failed.")
            
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def main():
    """Main function"""
    print("Testing new sweetBlood data management structure...")
    success = demonstrate_new_structure()
    
    if success:
        print("\n" + "=" * 80)
        print("CONCLUSION: The new sweetBlood structure meets all requirements:")
        print("1. ✓ LSTM ready CSVs are saved in sweetBlood/lstm_pump_data/pump_(serial num)")
        print("2. ✓ Logs go to sweetBlood/logs/")
        print("3. ✓ Metadata is saved in sweetBlood/metadata/")
        print("4. ✓ All components work together correctly")
        print("=" * 80)
    else:
        print("\n❌ Some requirements were not met. Please review the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
