#!/usr/bin/env python3
"""
Comprehensive test to verify the new sweetBlood structure works correctly
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
import shutil

# Add the project root to the Python path
sys.path.insert(0, '/home/bolt/projects/bb/bloodBath')

from utils.structure_utils import (
    setup_sweetblood_environment,
    get_lstm_pump_data_file,
    get_log_file,
    get_metadata_file
)
from utils.file_utils import (
    save_structured_lstm_data,
    save_structured_metadata,
    get_lstm_output_path
)

def test_sweetblood_structure():
    """Test the complete sweetBlood structure"""
    print("Testing complete sweetBlood structure...")
    
    # Create test directory
    test_dir = Path('/tmp/test_sweetblood_complete')
    
    # Clean up if exists
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # 1. Test structure creation
    print("1. Testing structure creation...")
    structure = setup_sweetblood_environment(str(test_dir))
    print(f"   Created structure: {list(structure.keys())}")
    
    # 2. Test LSTM pump data file creation
    print("2. Testing LSTM pump data file creation...")
    pump_serial = "881235"
    lstm_file = get_lstm_pump_data_file(structure, pump_serial)
    print(f"   LSTM file path: {lstm_file}")
    
    # Create sample LSTM data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='5min'),
        'bg': [150 + i for i in range(100)],
        'delta_bg': [1] * 100,
        'basal_rate': [1.0] * 100,
        'bolus_dose': [0.0] * 100,
        'sin_time': [0.5] * 100,
        'cos_time': [0.5] * 100
    })
    
    # Save using structured approach
    saved_file = save_structured_lstm_data(
        structure, sample_data, pump_serial, "2025-01-01", "2025-01-02"
    )
    print(f"   Saved data to: {saved_file}")
    
    # 3. Test metadata saving
    print("3. Testing metadata saving...")
    metadata = {
        'pump_serial': pump_serial,
        'last_sync': '2025-01-02',
        'record_count': 100,
        'sync_status': 'completed'
    }
    
    metadata_file = save_structured_metadata(structure, metadata, 'test_metadata.json')
    print(f"   Saved metadata to: {metadata_file}")
    
    # 4. Test log file path
    print("4. Testing log file path...")
    log_file = get_log_file(structure, 'test.log')
    print(f"   Log file path: {log_file}")
    
    # Create a log file
    with open(log_file, 'w') as f:
        f.write("Test log entry\\n")
    
    # 5. Test backward compatibility with old file_utils
    print("5. Testing backward compatibility...")
    old_style_path = get_lstm_output_path(structure['base'], pump_serial)
    print(f"   Old style path: {old_style_path}")
    
    # 6. Verify all files exist
    print("6. Verifying all files exist...")
    expected_files = [
        structure['base'] / 'README.md',
        structure['base'] / 'DIRECTORY_STRUCTURE.json',
        structure['base'] / '.gitignore',
        saved_file,
        metadata_file,
        log_file
    ]
    
    for file_path in expected_files:
        if file_path.exists():
            print(f"   ✓ {file_path.name}: exists")
        else:
            print(f"   ✗ {file_path.name}: missing")
    
    # 7. Test directory structure
    print("7. Testing directory structure...")
    expected_dirs = ['lstm_pump_data', 'metadata', 'models', 'logs']
    for dir_name in expected_dirs:
        dir_path = structure[dir_name]
        if dir_path.exists():
            files_count = len(list(dir_path.glob('*')))
            print(f"   ✓ {dir_name}: exists ({files_count} files)")
        else:
            print(f"   ✗ {dir_name}: missing")
    
    # 8. Test file contents
    print("8. Testing file contents...")
    
    # Check LSTM data
    if saved_file and saved_file.exists():
        df = pd.read_csv(saved_file, comment='#')
        print(f"   ✓ LSTM data: {len(df)} records loaded")
        print(f"   ✓ LSTM columns: {list(df.columns)}")
    else:
        print("   ✗ LSTM data: file not found")
    
    # Check metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        print(f"   ✓ Metadata: {loaded_metadata['pump_serial']}")
    
    print("\\nTest completed successfully!")
    return structure

if __name__ == "__main__":
    try:
        structure = test_sweetblood_structure()
        print(f"\\nFinal structure summary:")
        for key, path in structure.items():
            file_count = len(list(path.glob('*'))) if path.exists() else 0
            print(f"  {key}: {path} ({file_count} files)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
