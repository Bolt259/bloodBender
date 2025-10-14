#!/usr/bin/env python3
"""
Test script to verify CLI logging works with the structured approach
"""

import sys
import os
from pathlib import Path
import shutil

# Add the project root to the Python path
sys.path.insert(0, '/home/bolt/projects/bb/bloodBath')

from cli.main import setup_cli_logging
from utils.structure_utils import setup_sweetblood_environment

def test_cli_logging():
    """Test CLI logging with structured directories"""
    print("Testing CLI logging with structured directories...")
    
    # Create test directory
    test_dir = Path('/tmp/test_cli_logging')
    
    # Clean up if exists
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # 1. Test without structure (backward compatibility)
    print("1. Testing without structure (backward compatibility)...")
    setup_cli_logging(verbose=True, structure=None)
    
    # 2. Test with structure
    print("2. Testing with structure...")
    structure = setup_sweetblood_environment(str(test_dir))
    setup_cli_logging(verbose=True, structure=structure)
    
    # Check if log file exists in the right place
    log_file = structure['logs'] / 'bloodBath.log'
    if log_file.exists():
        print(f"   ✓ Log file created: {log_file}")
        
        # Check log content
        with open(log_file, 'r') as f:
            content = f.read()
        print(f"   ✓ Log file size: {len(content)} bytes")
    else:
        print(f"   ✗ Log file not found: {log_file}")
    
    # 3. Test logging functionality
    print("3. Testing logging functionality...")
    import logging
    logger = logging.getLogger('bloodBath')
    
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Check log content again
    with open(log_file, 'r') as f:
        content = f.read()
    
    if "Test info message" in content:
        print("   ✓ Info message logged")
    else:
        print("   ✗ Info message not found in log")
    
    if "Test warning message" in content:
        print("   ✓ Warning message logged")
    else:
        print("   ✗ Warning message not found in log")
    
    if "Test error message" in content:
        print("   ✓ Error message logged")
    else:
        print("   ✗ Error message not found in log")
    
    print("\\nCLI logging test completed!")
    return structure

if __name__ == "__main__":
    try:
        structure = test_cli_logging()
        print(f"\\nLogs directory: {structure['logs']}")
        print(f"Files in logs directory: {list(structure['logs'].glob('*'))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
