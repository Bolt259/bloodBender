# bloodBath Training Data Directory Structure

## Overview

This directory contains all LSTM training data generated from the bloodBath system, organized for efficient machine learning workflows.

## Directory Layout

```
training_data/
├── monthly_lstm/                    # Monthly LSTM-ready CSV files
│   ├── pump_881235/                 # Historical pump data (2021-2024)
│   │   ├── pump_881235_2021_01.csv  # January 2021 data
│   │   ├── pump_881235_2021_02.csv  # February 2021 data (15-day overlap with Jan)
│   │   └── ...                      # All monthly files (~46 files total)
│   └── pump_901161470/              # Active pump data (2024-2025)
│       ├── pump_901161470_2024_01.csv
│       ├── pump_901161470_2024_02.csv
│       └── ...                      # All monthly files (~15 files total)
├── raw_data/                        # Raw API response data (backup)
│   ├── pump_881235/                 # Raw CGM/pump data by date range
│   └── pump_901161470/
├── analytics/                       # Quality reports and manifests
│   ├── generation_summary_YYYYMMDD_HHMMSS.json
│   ├── training_manifest.json       # Master file listing for ML training
│   ├── quality_reports/
│   └── continuity_analysis/
└── logs/                           # Generation and processing logs
    ├── monthly_lstm_generation_YYYYMMDD_HHMMSS.log
    └── quality_validation_YYYYMMDD_HHMMSS.log
```

## File Naming Convention

### Monthly LSTM Files

- Format: `pump_{serial}_{YYYY}_{MM}.csv`
- Examples:
  - `pump_881235_2023_06.csv` (Pump 881235, June 2023)
  - `pump_901161470_2024_12.csv` (Pump 901161470, December 2024)

### Data Overlap Strategy

- Each monthly file contains 30 days of data
- 15-day overlap between consecutive months (50% overlap)
- Ensures temporal continuity for LSTM training sequences

## CSV File Structure

Each monthly LSTM CSV contains:

```csv
# bloodBath Monthly LSTM Training Dataset
# Pump Serial: 881235
# Month: 2023_06
# Date Range: 2023-06-01 to 2023-06-30
# Records: 8640
# Generated: 2025-10-10T15:30:45.123456
# Outlier Handling: Enabled (BG: 20-600, Basal: 0.0-8.0)
# Columns: timestamp, bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time, bg_anomaly_flag, basal_anomaly_flag, basal_statistical_outlier

timestamp,bg,delta_bg,basal_rate,bolus_dose,sin_time,cos_time,bg_anomaly_flag,basal_anomaly_flag,basal_statistical_outlier
2023-06-01 00:00:00-07:00,142.0,0.0,2.1,0.0,-1.0,0.0,0,0,0
...
```

## Data Features

### Core Features

- `timestamp`: UTC timestamp with timezone
- `bg`: Blood glucose (mg/dL, clipped 20-600)
- `delta_bg`: Glucose rate of change (mg/dL per 5min)
- `basal_rate`: Basal insulin rate (units/hour, clipped 0-8)
- `bolus_dose`: Bolus insulin dose (units)
- `sin_time`, `cos_time`: Cyclical time encoding

### Quality Features

- `bg_anomaly_flag`: 1 if BG was clipped to physiological limits
- `basal_anomaly_flag`: 1 if basal was clipped to physiological limits
- `basal_statistical_outlier`: 1 if basal is statistical outlier (Z>3)

## Quality Metrics

Each file includes quality validation:

- **Continuity Score**: Percentage of expected 5-minute intervals present
- **Gap Analysis**: Detection of gaps >18 hours
- **Outlier Statistics**: Count of physiological and statistical outliers
- **Overall Quality Score**: 0-1 scale combining all metrics

## Usage for LSTM Training

### Recommended Training Setup

1. **Context Window**: 288 timesteps (24 hours of 5-min data)
2. **Prediction Horizon**: 12-72 timesteps (1-6 hours ahead)
3. **Batch Size**: Process multiple monthly files together
4. **Train/Validation Split**: Use chronological split (early months for training)

### Loading Data

```python
import pandas as pd

# Load single monthly file
df = pd.read_csv('pump_881235_2023_06.csv', comment='#')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load multiple months for training
monthly_files = list(Path('monthly_lstm/pump_881235').glob('*.csv'))
dfs = [pd.read_csv(f, comment='#') for f in monthly_files]
combined_df = pd.concat(dfs, ignore_index=True)
```

## Data Availability

### Expected Coverage

- **Pump 881235**: ~46 monthly files (Jan 2021 - Oct 2024)
- **Pump 901161470**: ~15 monthly files (Jan 2024 - Oct 2025)
- **Total Records**: ~500,000+ glucose/insulin measurements
- **Total Duration**: ~5 years of continuous glucose monitoring data

### Quality Expectations

- **Temporal Completeness**: >95% of expected 5-minute intervals
- **Data Continuity**: <5% of months with gaps >18 hours
- **Outlier Rate**: <2% of records flagged as physiological anomalies

## Maintenance

### Regular Updates

- New monthly files generated automatically each month
- Quality reports updated with each generation
- Manifest updated to include new files

### Quality Monitoring

- Continuous validation of data completeness
- Outlier detection and flagging
- Cross-pump consistency validation

## Contact

For questions about the training data structure or quality issues, refer to the generation logs in `logs/` directory.
