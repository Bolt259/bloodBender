#!/usr/bin/env python3
"""
Test script to verify sweetBlood integration works with the new directory structure
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, '/home/bolt/projects/bb/bloodBath')

from sweetBlood.integration import SweetBloodIntegration

def test_sweetblood_integration():
    """Test that sweetBlood integration works with the new structure"""
    print("Testing sweetBlood integration with new directory structure...")
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        # Create sweetBlood directory in test location
        sweetblood_dir = test_dir / 'sweetBlood'
        sweetblood_dir.mkdir()
        
        # Change to the test directory temporarily
        original_cwd = os.getcwd()
        
        try:
            os.chdir(test_dir)
            
            # Initialize the integration
            integration = SweetBloodIntegration()
            
            # Check that the structure was created
            assert integration.structure['base'].exists()
            assert integration.structure['lstm_pump_data'].exists()
            assert integration.structure['metadata'].exists()
            assert integration.structure['models'].exists()
            assert integration.structure['logs'].exists()
            
            print(f"✓ Directory structure created: {integration.structure}")
            
            # Test prepare_training_data
            result = integration.prepare_training_data(
                pump_serial='123456',
                sequence_length=60,
                prediction_horizon=1,
                save_data=True
            )
            
            # Check result structure
            assert 'metadata' in result
            assert 'saved_files' in result
            assert result['metadata']['train_samples'] == 0
            assert result['metadata']['val_samples'] == 0
            assert result['metadata']['test_samples'] == 0
            
            # Check that saved_files uses the correct directory
            if result['saved_files']:
                train_file = result['saved_files']['train_X']
                assert train_file.parent == integration.structure['lstm_pump_data']
                print(f"✓ Training files will be saved to: {train_file.parent}")
            
            # Test get_training_data_info
            info = integration.get_training_data_info()
            assert 'training_data_dir' in info
            assert info['training_data_dir'] == str(integration.structure['lstm_pump_data'])
            
            print(f"✓ Training data directory: {info['training_data_dir']}")
            
            print("✓ sweetBlood integration test passed!")
            return True
            
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_sweetblood_integration()
    sys.exit(0 if success else 1)
