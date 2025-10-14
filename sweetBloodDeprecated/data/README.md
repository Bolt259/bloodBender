# bloodBath Data Directory Structure

This directory contains all data, metadata, and models for the bloodBath pump synchronization system.

## Directory Structure

### 📁 data/
Contains raw and processed pump data:
- **pump_data/**: Individual pump data files organized by serial number
- **raw_events/**: Raw pump events before processing
- **processed/**: Processed and validated pump data
- **exports/**: Exported data files for external use

### 📁 metadata/
Sync metadata and configuration tracking:
- **sync_metadata.json**: Tracks sync status for each pump
- **pump_configs.json**: Pump configuration settings
- **data_quality_reports.json**: Data validation reports

### 📁 lstm/
LSTM model training data and features:
- **training_data/**: Training datasets for LSTM models
- **validation_data/**: Validation datasets
- **test_data/**: Test datasets
- **features/**: Feature engineering outputs

### 📁 models/
Machine learning models and checkpoints:
- **checkpoints/**: Model training checkpoints
- **trained_models/**: Final trained models
- **configs/**: Model configuration files
- **metrics/**: Training metrics and performance data

### 📁 logs/
Log files for debugging and monitoring

### 📁 config/
Configuration files and settings

### 📁 temp/
Temporary files during processing

## Usage

This directory structure is automatically created by the bloodBath package when you run:

```bash
python -m bloodBath sync --pump-serial YOUR_PUMP_SERIAL
```

The package will:
1. Create the directory structure if it doesn't exist
2. Store pump data in organized folders
3. Track metadata about sync operations
4. Generate LSTM-ready datasets for machine learning
5. Provide a clean structure for model training and storage

## Environment Variables

The base directory can be configured with the `BLOODBATH_OUTPUT_DIR` environment variable:

```bash
BLOODBATH_OUTPUT_DIR=./my_custom_directory
```

Default: `./sweetBlood`
