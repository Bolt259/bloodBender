#!/usr/bin/env python3
"""
Test script to verify the simplified sweetBlood structure works correctly
"""

import sys
import os
sys.path.insert(0, '/home/bolt/projects/bb/bloodBath')

from pathlib import Path
from utils.structure_utils import setup_sweetblood_environment, create_sweetblood_structure

def test_simplified_structure():
    """Test the simplified sweetBlood structure"""
    print("Testing simplified sweetBlood structure...")
    
    # Create test directory
    test_dir = Path('/tmp/test_sweetblood')
    
    # Clean up if exists
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    # Test structure creation
    structure = create_sweetblood_structure(test_dir)
    
    print(f"Created structure: {structure}")
    
    # Verify expected directories exist
    expected_dirs = ['base', 'lstm_pump_data', 'metadata', 'models', 'logs']
    
    for dir_name in expected_dirs:
        if dir_name in structure:
            path = structure[dir_name]
            if path.exists():
                print(f"✓ {dir_name}: {path}")
            else:
                print(f"✗ {dir_name}: {path} (not found)")
        else:
            print(f"✗ {dir_name}: missing from structure")
    
    # Test setup_sweetblood_environment
    print("\nTesting setup_sweetblood_environment...")
    structure2 = setup_sweetblood_environment(str(test_dir))
    
    # Check for generated files
    expected_files = ['README.md', 'DIRECTORY_STRUCTURE.json', '.gitignore']
    
    for file_name in expected_files:
        file_path = structure2['base'] / file_name
        if file_path.exists():
            print(f"✓ {file_name}: {file_path}")
        else:
            print(f"✗ {file_name}: {file_path} (not found)")
    
    # Test that old directories don't exist
    old_dirs = ['data', 'lstm', 'config', 'temp']
    for old_dir in old_dirs:
        old_path = test_dir / old_dir
        if old_path.exists():
            print(f"✗ Old directory still exists: {old_path}")
        else:
            print(f"✓ Old directory correctly removed: {old_dir}")
    
    print("\nStructure test completed!")
    return structure2

if __name__ == "__main__":
    try:
        structure = test_simplified_structure()
        print(f"\nFinal structure: {structure}")
        
        # Check README content
        readme_path = structure['base'] / 'README.md'
        if readme_path.exists():
            print(f"\nREADME content preview:")
            with open(readme_path, 'r') as f:
                content = f.read()
                print(content[:500] + "..." if len(content) > 500 else content)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
