#!/usr/bin/env python3
"""
Test script to verify the new sweetBlood data management directory structure works correctly
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

def test_lstm_csv_saving():
    """Test that LSTM ready CSVs are saved in sweetBlood/lstm_pump_data/pump_(serial num)"""
    print("=" * 60)
    print("TEST 1: LSTM CSV Saving")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Set up structure
        structure = setup_sweetblood_environment(str(test_dir))
        
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
        
        # Test pump serial
        pump_serial = '123456'
        
        # Save the data
        saved_file = save_structured_lstm_data(
            structure, sample_data, pump_serial, '2025-01-01', '2025-01-31'
        )
        
        # Verify the file path
        if saved_file is None:
            raise ValueError("save_structured_lstm_data returned None")
            
        expected_dir = structure['lstm_pump_data']
        assert saved_file.parent == expected_dir, f"Expected {expected_dir}, got {saved_file.parent}"
        assert saved_file.name.startswith(f"pump_{pump_serial}_"), f"Expected filename to start with pump_{pump_serial}_, got {saved_file.name}"
        
        print(f"✓ LSTM data saved correctly: {saved_file}")
        
        # Test with get_lstm_pump_data_file utility function
        utility_file = get_lstm_pump_data_file(structure, pump_serial)
        assert utility_file.parent == expected_dir, f"Utility function returns wrong directory"
        assert utility_file.name.startswith(f"pump_{pump_serial}_"), f"Utility function returns wrong filename format"
        
        print(f"✓ Utility function works correctly: {utility_file}")
        
        # Verify file contents
        df_loaded = pd.read_csv(saved_file, comment='#')
        assert len(df_loaded) == 100, f"Expected 100 records, got {len(df_loaded)}"
        assert list(df_loaded.columns) == ['timestamp', 'bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time']
        
        print(f"✓ File contents verified: {len(df_loaded)} records with correct columns")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_logging_directory():
    """Test that logs go to sweetBlood/logs/"""
    print("=" * 60)
    print("TEST 2: Logging Directory")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Set up structure
        structure = setup_sweetblood_environment(str(test_dir))
        
        # Test log file path
        log_file = get_log_file(structure, 'test.log')
        expected_dir = structure['logs']
        assert log_file.parent == expected_dir, f"Expected {expected_dir}, got {log_file.parent}"
        
        print(f"✓ Log file path correct: {log_file}")
        
        # Test logger setup
        logger = setup_logger('test_logger', log_file=log_file)
        logger.info("Test log message")
        
        # Verify log file exists and has content
        assert log_file.exists(), "Log file was not created"
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert "Test log message" in content, "Log message not found in file"
        
        print(f"✓ Logger works correctly: {len(content)} bytes written")
        
        # Test basic logging setup without CLI imports
        print(f"✓ Basic logging uses correct directory: {structure['logs']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_metadata_directory():
    """Test that metadata is saved in sweetBlood/metadata/"""
    print("=" * 60)
    print("TEST 3: Metadata Directory")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Set up structure
        structure = setup_sweetblood_environment(str(test_dir))
        
        # Test metadata file path
        metadata_file = get_metadata_file(structure, 'test_metadata.json')
        expected_dir = structure['metadata']
        assert metadata_file.parent == expected_dir, f"Expected {expected_dir}, got {metadata_file.parent}"
        
        print(f"✓ Metadata file path correct: {metadata_file}")
        
        # Test saving metadata
        test_metadata = {
            'pump_serial': '123456',
            'sync_date': '2025-01-15',
            'records_processed': 1000,
            'last_sync': '2025-01-15T10:30:00'
        }
        
        saved_file = save_structured_metadata(structure, test_metadata, 'test_metadata.json')
        
        # Verify the file path
        assert saved_file.parent == expected_dir, f"Expected {expected_dir}, got {saved_file.parent}"
        assert saved_file.name == 'test_metadata.json', f"Expected test_metadata.json, got {saved_file.name}"
        
        print(f"✓ Metadata saved correctly: {saved_file}")
        
        # Verify file contents
        import json
        with open(saved_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata == test_metadata, "Metadata content mismatch"
        
        print(f"✓ Metadata content verified: {len(loaded_metadata)} items")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_client_integration():
    """Test that the structure utilities work correctly"""
    print("=" * 60)
    print("TEST 4: Structure Integration")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Set up structure
        structure = setup_sweetblood_environment(str(test_dir))
        
        # Check that the structure was created correctly
        assert structure['base'] == test_dir
        assert structure['lstm_pump_data'] == test_dir / 'lstm_pump_data'
        assert structure['metadata'] == test_dir / 'metadata'
        assert structure['logs'] == test_dir / 'logs'
        assert structure['models'] == test_dir / 'models'
        
        print(f"✓ Structure mapping correct: {structure}")
        
        # Check that directories exist
        for name, path in structure.items():
            if name != 'base':
                assert path.exists(), f"Directory {name} was not created: {path}"
                
        print(f"✓ All directories created successfully")
        
        # Test integration with utility functions
        pump_serial = '654321'
        
        # Test LSTM file path
        lstm_file = get_lstm_pump_data_file(structure, pump_serial)
        assert lstm_file.parent == structure['lstm_pump_data']
        assert lstm_file.name.startswith(f"pump_{pump_serial}_")
        
        # Test metadata file path
        metadata_file = get_metadata_file(structure, 'sync_metadata.json')
        assert metadata_file.parent == structure['metadata']
        assert metadata_file.name == 'sync_metadata.json'
        
        # Test log file path
        log_file = get_log_file(structure, 'test.log')
        assert log_file.parent == structure['logs']
        assert log_file.name == 'test.log'
        
        print(f"✓ All utility functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def main():
    """Run all tests"""
    print("Testing new sweetBlood data management directory structure...")
    print()
    
    tests = [
        ("LSTM CSV Saving", test_lstm_csv_saving),
        ("Logging Directory", test_logging_directory),
        ("Metadata Directory", test_metadata_directory),
        ("Client Integration", test_client_integration)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running {name}...")
        result = test_func()
        results.append((name, result))
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The new sweetBlood directory structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
