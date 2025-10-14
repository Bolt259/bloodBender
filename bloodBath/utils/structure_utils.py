"""
Directory structure utilities for bloodBath
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def create_sweetblood_structure(base_dir: Path) -> Dict[str, Path]:
    """
    Create the simplified sweetBlood directory structure
    
    Args:
        base_dir: Base directory path (e.g., ./sweetBlood)
        
    Returns:
        Dict mapping structure names to paths
    """
    structure = {
        'base': base_dir,
        'lstm_pump_data': base_dir / 'lstm_pump_data',
        'metadata': base_dir / 'metadata',
        'models': base_dir / 'models',
        'logs': base_dir / 'logs'
    }
    
    # Create all directories
    for name, path in structure.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
    
    return structure


def create_project_structure_info(structure: Dict[str, Path]) -> Dict[str, Any]:
    """
    Create a README/info file explaining the directory structure
    
    Args:
        structure: Directory structure mapping
        
    Returns:
        Dict with structure information
    """
    info = {
        "bloodBath_directory_structure": {
            "description": "Simplified directory structure for bloodBath pump data synchronization",
            "created_by": "bloodBath package",
            "directories": {
                "lstm_pump_data/": {
                    "description": "LSTM-ready pump data organized by serial number and date ranges"
                },
                "metadata/": {
                    "description": "Sync metadata and configuration files"
                },
                "models/": {
                    "description": "Trained machine learning models and checkpoints"
                },
                "logs/": {
                    "description": "Log files for debugging and monitoring"
                }
            }
        }
    }
    
    return info


def save_structure_info(structure: Dict[str, Path], info: Dict[str, Any]) -> Path:
    """
    Save directory structure information to a JSON file
    
    Args:
        structure: Directory structure mapping
        info: Structure information
        
    Returns:
        Path to the saved info file
    """
    info_file = structure['base'] / 'DIRECTORY_STRUCTURE.json'
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    logger.info(f"Saved directory structure info to: {info_file}")
    return info_file


def create_readme_file(structure: Dict[str, Path]) -> Path:
    """
    Create a README file explaining the directory structure
    
    Args:
        structure: Directory structure mapping
        
    Returns:
        Path to the created README file
    """
    readme_content = """# bloodBath Data Directory Structure

This directory contains all data, metadata, and models for the bloodBath pump synchronization system.

## Directory Structure

### 📁 lstm_pump_data/
Contains LSTM-ready pump data organized by pump serial number and date ranges:
- **pump_XXXXXX_YYYYMMDD_HHMMSS.csv**: Individual pump LSTM-ready datasets
- **combined_YYYYMMDD_HHMMSS.csv**: Combined multi-pump datasets
- Files include: timestamp, bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time columns

### 📁 metadata/
Sync metadata and configuration tracking:
- **sync_metadata.json**: Tracks sync status for each pump
- **pump_configs.json**: Pump configuration settings
- **data_quality_reports.json**: Data validation reports

### 📁 models/
Machine learning models and checkpoints:
- **trained_models/**: Final trained models
- **checkpoints/**: Model training checkpoints  
- **configs/**: Model configuration files
- **metrics/**: Training metrics and performance data

### 📁 logs/
Log files for debugging and monitoring:
- **bloodBath.log**: Main application log
- **sync_YYYYMMDD.log**: Daily sync operation logs
- **error_YYYYMMDD.log**: Error logs by date

## Usage

This directory structure is automatically created by the bloodBath package when you run:

```bash
python -m bloodBath sync --pump-serial YOUR_PUMP_SERIAL
```

The package will:
1. Create the directory structure if it doesn't exist
2. Store LSTM-ready pump data in organized files
3. Track metadata about sync operations
4. Save models and training logs
5. Provide a clean structure for model training and storage

## Environment Variables

The base directory can be configured with the `BLOODBATH_OUTPUT_DIR` environment variable:

```bash
BLOODBATH_OUTPUT_DIR=./my_custom_directory
```

Default: `./sweetBlood`
"""
    
    readme_file = structure['base'] / 'README.md'
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created README file at: {readme_file}")
    return readme_file


def get_pump_data_directory(structure: Dict[str, Path], pump_serial: str) -> Path:
    """
    Get or create pump-specific data directory
    
    Args:
        structure: Directory structure mapping
        pump_serial: Pump serial number
        
    Returns:
        Path to pump data directory (returns lstm_pump_data for simplified structure)
    """
    return structure['lstm_pump_data']


def get_metadata_file(structure: Dict[str, Path], filename: str) -> Path:
    """
    Get path to metadata file
    
    Args:
        structure: Directory structure mapping
        filename: Metadata filename
        
    Returns:
        Path to metadata file
    """
    return structure['metadata'] / filename


def get_lstm_data_directory(structure: Dict[str, Path], data_type: str = 'lstm_pump_data') -> Path:
    """
    Get LSTM data directory
    
    Args:
        structure: Directory structure mapping
        data_type: Type of data (defaults to 'lstm_pump_data')
        
    Returns:
        Path to LSTM data directory
    """
    return structure['lstm_pump_data']


def get_model_directory(structure: Dict[str, Path], model_type: str = 'models') -> Path:
    """
    Get model directory
    
    Args:
        structure: Directory structure mapping
        model_type: Type of model directory (defaults to 'models')
        
    Returns:
        Path to model directory
    """
    return structure['models']


def get_logs_directory(structure: Dict[str, Path]) -> Path:
    """
    Get logs directory
    
    Args:
        structure: Directory structure mapping
        
    Returns:
        Path to logs directory
    """
    return structure['logs']


def get_log_file(structure: Dict[str, Path], filename: str) -> Path:
    """
    Get path to log file
    
    Args:
        structure: Directory structure mapping
        filename: Log filename
        
    Returns:
        Path to log file
    """
    return structure['logs'] / filename


def get_lstm_pump_data_file(structure: Dict[str, Path], pump_serial: str, timestamp: Optional[str] = None) -> Path:
    """
    Get path to LSTM pump data file
    
    Args:
        structure: Directory structure mapping
        pump_serial: Pump serial number
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        Path to LSTM pump data file
    """
    import pandas as pd
    
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"pump_{pump_serial}_{timestamp}.csv"
    return structure['lstm_pump_data'] / filename


def setup_bloodbank_environment(output_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Set up the bloodBank environment using the new bloodBath architecture
    
    Args:
        output_dir: Override output directory (defaults to bloodBank structure)
        
    Returns:
        Dict mapping structure names to paths compatible with bloodBank v2.0
    """
    # Import bloodBank configuration
    from ..core.config import DATA_PATHS, BLOODBANK_ROOT
    
    if output_dir:
        # If custom output dir specified, adapt structure
        base_dir = Path(str(output_dir)).resolve()
        logger.info(f"Setting up custom bloodBank environment at: {base_dir}")
        
        structure = {
            'base': base_dir,
            'lstm_pump_data': base_dir / 'lstm_pump_data',
            'metadata': base_dir / 'metadata', 
            'models': base_dir / 'models',
            'logs': base_dir / 'logs'
        }
        
        # Create directories
        for path in structure.values():
            path.mkdir(parents=True, exist_ok=True)
            
    else:
        # Use standard bloodBank structure
        logger.info(f"Setting up bloodBank environment at: {BLOODBANK_ROOT}")
        
        # Ensure bloodBank structure exists
        for category, paths in DATA_PATHS.items():
            if isinstance(paths, dict):
                for path in paths.values():
                    path.mkdir(parents=True, exist_ok=True)
        
        # Map bloodBank paths to client-compatible structure
        structure = {
            'base': BLOODBANK_ROOT,
            'lstm_pump_data': DATA_PATHS['raw']['lstm'],  # Raw LSTM data location
            'metadata': DATA_PATHS['raw']['metadata'],     # Metadata location
            'models': BLOODBANK_ROOT / 'models',           # Models (create if needed)
            'logs': DATA_PATHS['archives']['logs']         # Logs location
        }
        
        # Ensure additional directories exist
        (BLOODBANK_ROOT / 'models').mkdir(parents=True, exist_ok=True)
    
    # Create bloodBank-compatible documentation
    info = create_bloodbank_structure_info(structure)
    save_structure_info(structure, info)
    create_bloodbank_readme_file(structure)
    
    logger.info(f"bloodBank environment ready: {structure['base']}")
    return structure


def create_bloodbank_structure_info(structure: Dict[str, Path]) -> Dict[str, Any]:
    """
    Create bloodBank v2.0 structure information
    
    Args:
        structure: Directory structure mapping
        
    Returns:
        Dict with bloodBank structure information
    """
    info = {
        "bloodBath_bloodBank_structure": {
            "description": "bloodBank v2.0 directory structure for unified pump data management",
            "schema_version": "2.0",
            "created_by": "bloodBath package",
            "directories": {
                "raw/": {
                    "description": "Raw pump data organized by type and serial number",
                    "subdirectories": {
                        "cgm/": "CGM readings organized by pump serial",
                        "basal/": "Basal rate data organized by pump serial", 
                        "bolus/": "Bolus/meal data organized by pump serial",
                        "lstm/": "Raw LSTM processing workspace",
                        "metadata/": "Sync and validation metadata"
                    }
                },
                "merged/": {
                    "description": "Processed LSTM sequences ready for ML training",
                    "subdirectories": {
                        "train/": "Training dataset (70%)",
                        "validate/": "Validation dataset (15%)",
                        "test/": "Test dataset (15%)",
                        "full/": "Complete merged sequences"
                    }
                },
                "archives/": {
                    "description": "Legacy data and operational logs",
                    "subdirectories": {
                        "legacy/": "Legacy sweetBlood data",
                        "logs/": "System and sync operation logs"
                    }
                },
                "models/": {
                    "description": "Trained models and checkpoints"
                }
            }
        }
    }
    
    return info


def create_bloodbank_readme_file(structure: Dict[str, Path]) -> Path:
    """
    Create bloodBank v2.0 README file
    
    Args:
        structure: Directory structure mapping
        
    Returns:
        Path to created README file
    """
    readme_content = """# bloodBath bloodBank v2.0 Data Architecture

This directory implements the unified bloodBank v2.0 data architecture for comprehensive pump data management and ML training.

## Architecture Overview

### 📁 raw/ - Raw Data Organization
**Purpose**: Raw pump data organized by type and serial number
- `cgm/{serial}/`: CGM readings for each pump
- `basal/{serial}/`: Basal rate data for each pump  
- `bolus/{serial}/`: Bolus/meal data for each pump
- `lstm/`: Raw LSTM processing workspace
- `metadata/`: Sync metadata and validation reports

### 📁 merged/ - LSTM-Ready Sequences
**Purpose**: Processed sequences ready for ML training
- `train/` (70%): Training dataset with chronological split
- `validate/` (15%): Validation dataset for hyperparameter tuning
- `test/` (15%): Test dataset for final model evaluation
- `full/`: Complete processed sequences before splitting

### 📁 archives/ - Legacy and Logs
**Purpose**: Legacy data preservation and operational tracking
- `legacy/`: Migrated sweetBlood data for compatibility
- `logs/`: System logs, sync reports, and error tracking

### 📁 models/ - ML Models
**Purpose**: Trained models and training artifacts
- Trained LSTM models for glucose prediction
- Model checkpoints and configurations
- Training metrics and evaluation reports

## Data Flow

```
Raw Pump Data → Validation → LSTM Processing → Train/Val/Test Splits → Model Training
     ↓              ↓             ↓                    ↓                    ↓
  raw/{type}    metadata/     merged/full/        merged/{split}/       models/
```

## Schema v2.0 Features

✅ **Unified Data Architecture**: Single source of truth for all pump data
✅ **Type-Based Organization**: Separate directories for CGM, basal, bolus data  
✅ **Serial-Based Partitioning**: Each pump's data isolated for parallel processing
✅ **ML-Ready Splits**: Chronological train/validate/test splits maintained
✅ **Comprehensive Metadata**: Full tracking of sync status, validation results
✅ **Legacy Compatibility**: Seamless migration from sweetBlood structure
✅ **Scalable Design**: Support for multiple pumps and years of data

## Usage

Initialize and sync data:
```bash
# Full resynchronization (4+ years)
python full_bloodbank_resync.py

# Incremental updates
python -m bloodBath sync --pump-serial 881235 --update-mode

# Validation and reports
python -m bloodBath validate --pump-serial all
```

Access processed data:
```python
from bloodBath.core.config import DATA_PATHS

# Training data
train_dir = DATA_PATHS['merged']['train']

# Raw CGM data for pump 881235
cgm_dir = DATA_PATHS['raw']['cgm'] / '881235'

# Metadata and reports
metadata_dir = DATA_PATHS['raw']['metadata']
```

## Configuration

bloodBank root configured in `bloodBath/core/config.py`:
```python
BLOODBANK_ROOT = Path(__file__).parent.parent / "bloodBank"
```

Override with environment variable:
```bash
export BLOODBATH_BLOODBANK_ROOT=/path/to/custom/bloodbank
```

---
*Generated by bloodBath v2.0 - Schema v2.0*
"""
    
    readme_file = structure['base'] / 'BLOODBANK_README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created bloodBank README at: {readme_file}")
    return readme_file


def setup_sweetblood_environment(output_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    DEPRECATED: Legacy sweetBlood environment setup
    
    Use setup_bloodbank_environment() for new bloodBank v2.0 architecture.
    This function maintained for backward compatibility only.
    
    Args:
        output_dir: Override output directory (uses env var or default)
        
    Returns:
        Dict mapping structure names to paths
    """
    logger.warning("setup_sweetblood_environment() is DEPRECATED. Use setup_bloodbank_environment() instead.")
    
    # Get output directory from environment or use default
    if not output_dir:
        from .env_utils import get_env_config
        env_config = get_env_config()
        output_dir = env_config.get('output_dir', './sweetBlood')
    
    base_dir = Path(str(output_dir)).resolve()
    
    logger.info(f"Setting up LEGACY sweetBlood environment at: {base_dir}")
    
    # Create directory structure
    structure = create_sweetblood_structure(base_dir)
    
    # Create documentation
    info = create_project_structure_info(structure)
    save_structure_info(structure, info)
    create_readme_file(structure)
    
    # Create a .gitignore file to exclude temporary files
    gitignore_content = """# bloodBath temporary files
*.tmp
*.temp
*.log

# Large data files
*.parquet
*.h5
*.hdf5
*.pkl
*.pickle

# Model checkpoints (keep only final models)
models/checkpoints/
*.ckpt

# OS generated files
.DS_Store
Thumbs.db

# Python cache
__pycache__/
*.pyc
*.pyo
"""
    
    gitignore_file = structure['base'] / '.gitignore'
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    
    logger.info(f"Created .gitignore file at: {gitignore_file}")
    
    return structure
