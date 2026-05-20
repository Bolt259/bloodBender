# bloodBath Data Directory Structure

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
