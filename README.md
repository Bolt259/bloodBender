# bloodBender 🩸

**Advanced Tandem Insulin Pump Data Synchronization & Processing System**

A comprehensive Python package for downloading, processing, and preparing Tandem t:connect diabetes pump data for machine learning applications.

## 📋 Overview

bloodBender (package name: `bloodBath`) is a modular system designed to:

- 🔄 Synchronize historical pump data from Tandem t:connect
- 📊 Process CGM readings, basal rates, and bolus data
- 🧹 Clean and validate diabetes data with strict quality controls
- 📈 Generate LSTM-ready datasets for predictive modeling
- 🔍 Maintain data integrity with comprehensive validation

## 🏗️ Repository Structure

```
bloodBender/
├── bloodBath/              # Main Python package
│   ├── api/               # T:connect API integration
│   ├── cli/               # Command-line interface
│   ├── core/              # Core client and configuration
│   ├── data/              # Data processing and validation
│   ├── io/                # CSV reading and writing
│   ├── sync/              # Synchronization engines
│   ├── utils/             # Utility functions
│   ├── validation/        # Comprehensive validation framework
│   ├── spec/              # Design specifications
│   └── bloodBank/         # Data storage (v2.0 architecture)
│
├── bareMetalBender/       # C++ IVP solver - embedded glucose dynamics engine
│
├── bloodBath-env/         # Python virtual environment
│
├── bloodbank_download.py  # Main download script (v2.0)
│
└── *.log                  # Operation logs
```

## ✨ Features

### Data Processing (v2.0)

- **Smart Gap Handling**: Preserves NaN for missing BG values (no artificial fills)
- **Extended BG Range**: Supports 20-600 mg/dL (expanded from 40-400)
- **Metadata Flags**: `bg_missing_flag` and `bg_clip_flag` for transparency
- **5-Minute Resampling**: Standardized time intervals for ML training
- **Multi-Pump Support**: Handles multiple pump serials with date range validation

### Data Quality

- **CSV Post-Processing**: Automatically removes files with 100% invalid data
- **Comprehensive Validation**: Timestamp verification, range checks, schema compliance
- **v2.0 CSV Format**: Comment headers with metadata and processing information
- **Archival System**: Preserves old data before regeneration

### Architecture

- **Modular Design**: Clean separation of concerns (API, processing, validation, I/O)
- **bloodBank v2.0**: Unified data storage with organized directory structure
- **Test Framework**: Comprehensive testing in `bloodBath/test_scripts/`
- **Design Specification**: 685-line technical spec documenting all constants and workflows

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nickweiss425/bloodBender.git
cd bloodBender

# Activate the environment
source bloodBath-env/bin/activate

# Set up credentials (create .env file)
cp .env.example .env
# Edit .env with your t:connect credentials
```

### Basic Usage

```bash
# Download pump data (using v2.0 fixes)
python bloodbank_download.py \
  --pump-serial YOUR_SERIAL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir bloodBath/bloodBank/raw/

# Using the bloodBath package
python -m bloodBath sync --pump-serial YOUR_SERIAL

# Check status
python -m bloodBath status

# Generate LSTM-ready data
python -m bloodBath unified-lstm --pump-serial YOUR_SERIAL
```

## 📊 Data Format (v2.0)

CSV files include comprehensive metadata headers:

```csv
# bloodBath v2.0 CSV Data File
# Pump Serial: 881235
# Date Range: 2021-10-22 to 2022-10-22
# Total Records: 105120
# BG Range: [20, 600] mg/dL
# Processing: 5-minute resampling, NaN preservation

time,bg,basal,bolus,bg_missing_flag,bg_clip_flag
2021-10-22 00:00:00+00:00,120.0,0.85,0.0,False,False
2021-10-22 00:05:00+00:00,NaN,0.85,0.0,True,False
...
```

## 🧪 Testing

```bash
# Run v2.0 integration tests
python bloodBath/test_scripts/test_v2_integration.py

# Validate CSV format
python bloodBath/test_scripts/check_v2_format.py

# CSV cleanup test
python bloodBath/test_scripts/test_csv_cleanup.py --dry-run
```

## 📚 Documentation

- **Design Specification**: `bloodBath/spec/bloodBath_Design_Specification_v2.0.md`
- **Implementation Report**: `bloodBath/v2.0_Complete_Report.md`
- **bloodBank Architecture**: `bloodBath/bloodBank/BLOODBANK_README.md`
- **API Reference**: `API_REFERENCE.md`
- **LSTM Processing**: `LSTM_PROCESSING_GUIDE.md`

## 🔧 Configuration

Environment variables (`.env`):

```bash
# T:connect Credentials
TCONNECT_EMAIL=your@email.com
TCONNECT_PASSWORD=your_password
TCONNECT_REGION=US

# System Configuration
PUMP_SERIAL_NUMBER=123456
TIMEZONE_NAME=America/Los_Angeles
BLOODBATH_OUTPUT_DIR=./bloodBath/bloodBank
BLOODBATH_LOG_LEVEL=INFO
```

## 📈 Data Coverage

### Current Pumps

- **Pump 881235**: 2021-10-22 to 2024-10-06 (45 files, ~1.2M records)
- **Pump 901161470**: 2024-10-07 to 2025-10-11 (23 files, ~883K records)

### Data Quality (Post-Cleanup)

- 54 files with valid data (17 invalid files removed)
- 96.48 MB of invalid data cleaned
- 100% valid data preservation

## 🛠️ Development

### Recent Updates (v2.0)

- ✅ Fixed BG stitching bug (no more 100-fills)
- ✅ Extended BG range to 20-600 mg/dL
- ✅ Added `bg_missing_flag` and `bg_clip_flag` columns
- ✅ Implemented CSV post-processing cleanup
- ✅ Fixed pkg_resources deprecation warnings
- ✅ Comprehensive design specification created
- ✅ Unified Python environment (bloodBath-env)

### Testing

All tests located in `bloodBath/test_scripts/` per design specification:

- `test_v2_integration.py` - Integration testing
- `test_bug_fixes.py` - Regression testing
- `test_csv_cleanup.py` - Data quality testing
- `check_v2_format.py` - Format validation

## 🤝 Contributing

This repository is part of a senior project for diabetes prediction research. The system is designed to be modular and extensible.

### Key Components

- **bloodBath Package**: Main processing system
- **bareMetalBender**: C++ IVP solver - low-level glucose dynamics engine
- **bloodbank_download.py**: Standalone download script with v2.0 fixes

## 📝 License

Part of academic research project. See individual component licenses.

## 🔗 Related Projects

- [tconnectsync](https://github.com/jwoglom/tconnectsync) - Base T:connect API client (modified)
- Tandem Diabetes Care - t:connect platform

## 🙏 Acknowledgments

- Original tconnectsync package by jwoglom
- Tandem t:connect API
- Python diabetes data community

---

**Version**: 2.0  
**Last Updated**: October 2025  
**Status**: Active Development

For detailed technical information, see `bloodBath/spec/bloodBath_Design_Specification_v2.0.md`
