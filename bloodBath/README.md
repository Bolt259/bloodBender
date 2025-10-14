# bloodBath: Modular Tandem Pump Synchronization

A clean, testable, modular Python package for synchronizing historical data from Tandem insulin pumps and generating LSTM-ready datasets for machine learning applications.

## Features

- **Multi-pump support**: Sync data from multiple Tandem pumps
- **Modular architecture**: Clean separation of concerns with dedicated modules
- **LSTM-ready output**: Generate 5-minute aggregated datasets optimized for LSTM training
- **Robust error handling**: Comprehensive error handling with retry logic
- **Flexible CLI**: Command-line interface with multiple sync options
- **Data validation**: Built-in data quality checks and validation
- **Incremental sync**: Update mode for syncing only new data

## Installation

```bash
# Install dependencies
pip install pandas numpy arrow tconnectsync pathlib

# The package is ready to use from the bloodBath directory
```

## Quick Start

### Using the Library

```python
from bloodBath import TandemHistoricalSyncClient, PumpConfig

# Create client
client = TandemHistoricalSyncClient(
    email='your@email.com',
    password='your_password',
    output_dir='./pump_data'
)

# Configure pump
config = PumpConfig(
    serial='123456',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Sync data
success = client.sync_pump_historical(config)

# Generate LSTM dataset
if success:
    lstm_file = client.generate_lstm_ready_data('123456')
    print(f"LSTM dataset saved to: {lstm_file}")
```

### Using the CLI

```bash
# Test connection
python -m bloodBath test --email your@email.com --password your_password

# Sync a single pump
python -m bloodBath sync --pump-serial 123456 --start-date 2024-01-01

# Sync multiple pumps from config file
python -m bloodBath sync --config pump_configs.json

# Update mode (only sync new data)
python -m bloodBath sync --pump-serial 123456 --update

# Generate LSTM dataset
python -m bloodBath lstm --pump-serial 123456

# Check sync status
python -m bloodBath status

# Create default configuration
python -m bloodBath create-config --output pumps.json
```

## Package Structure

```
bloodBath/
├── __init__.py          # Main package exports
├── __main__.py          # Module entry point
├── core/                # Core synchronization logic
│   ├── client.py        # Main TandemHistoricalSyncClient
│   ├── config.py        # Configuration management
│   └── exceptions.py    # Custom exceptions
├── api/                 # API connection and data fetching
│   ├── connector.py     # API authentication and connection
│   └── fetcher.py       # Data fetching with retry logic
├── data/                # Data processing and validation
│   ├── extractors.py    # Event extraction and normalization
│   ├── processors.py    # Data processing and LSTM preparation
│   └── validators.py    # Data validation and quality checks
├── utils/               # Utility functions
│   ├── file_utils.py    # File I/O operations
│   ├── time_utils.py    # Timestamp handling
│   └── logging_utils.py # Logging configuration
└── cli/                 # Command-line interface
    └── main.py          # CLI implementation
```

## Data Flow

1. **Authentication**: Connect to Tandem API with credentials
2. **Data Fetching**: Retrieve pump events in date-chunked requests
3. **Event Extraction**: Categorize and normalize raw pump events
4. **Data Validation**: Validate event integrity and quality
5. **Processing**: Create 5-minute aggregated LSTM-ready datasets
6. **Output**: Save data in structured CSV format

## Configuration

### Pump Configuration File Format

```json
{
  "pumps": [
    {
      "serial": "123456",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    {
      "serial": "789012",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    }
  ]
}
```

### Environment Variables

```bash
# t:connect credentials (optional, can be passed directly)
export TANDEM_EMAIL="your@email.com"
export TANDEM_PASSWORD="your_password"
export TANDEM_REGION="US"  # or "EU"
```

## Output Format

### LSTM-Ready Dataset

The package generates 5-minute aggregated datasets with the following columns:

- `timestamp`: ISO timestamp for each 5-minute interval
- `bg`: Blood glucose value (mg/dL)
- `basal_rate`: Basal insulin rate (units/hour)
- `bolus_dose`: Bolus insulin dose (units)
- `hour`: Hour of day (0-23)
- `minute`: Minute of hour (0-59)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Weekend flag (boolean)

### File Structure

```
sweetBlood/
├── sync_metadata.json           # Sync metadata and status
├── pump_123456/                # Per-pump data directories
│   ├── cgmreading_*.csv        # CGM readings
│   ├── basal_*.csv             # Basal events
│   └── bolus_*.csv             # Bolus events
└── data/
    └── lstm_ready/
        ├── lstm_ready_123456.csv    # LSTM dataset per pump
        └── lstm_ready_combined.csv  # Combined multi-pump dataset
```

## Error Handling

The package includes comprehensive error handling:

- **Rate limiting**: Automatic retry with exponential backoff
- **Chunk size errors**: Automatic reduction of date chunk size
- **Authentication errors**: Clear error messages and retry logic
- **Data validation**: Quality checks with detailed error reporting
- **Network errors**: Robust connection handling with retries

## Development

### Running Tests

```bash
# Run package tests (when test suite is available)
python -m pytest tests/

# Test specific components
python -c "from bloodBath import TandemHistoricalSyncClient; print('Import successful')"
```

### Logging

The package uses structured logging with configurable levels:

```python
from bloodBath.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(
    name='bloodBath',
    level=logging.INFO,
    log_file='bloodBath.log'
)
```

## Migration from Original

The original `tandem_historical_sync.py` (1422 lines) has been refactored into this modular package. Key improvements:

- **Modularity**: Separated concerns into focused modules
- **Testability**: Clean interfaces and dependency injection
- **Maintainability**: Clear code organization and documentation
- **Extensibility**: Easy to add new pump types or data sources
- **Error handling**: Comprehensive error management
- **Performance**: Optimized data processing and memory usage

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include error handling and logging
4. Test new features thoroughly
5. Update documentation as needed

## License

This package is built on top of the `tconnectsync` library and follows the same licensing terms.
