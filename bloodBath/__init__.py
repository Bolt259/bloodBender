"""
bloodBath: Clean, testable, modular Tandem pump synchronization

A Python package for synchronizing historical data from Tandem insulin pumps
and generating LSTM-ready datasets for machine learning applications.

Usage:
    from bloodBath import TandemHistoricalSyncClient, PumpConfig
    
    # Create client with credentials from environment (.env file)
    client = TandemHistoricalSyncClient(
        output_dir='./pump_data'
    )
    
    # Or specify credentials explicitly
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
    client.sync_pump_historical(config)
    
    # Generate LSTM dataset
    client.generate_lstm_ready_data('123456')
"""

from .core import (
    TandemHistoricalSyncClient,
    PumpConfig,
    SyncMetadata,
    CredentialsConfig,
    load_pump_configs,
    create_default_pump_configs,
    get_credentials,
    load_credentials_from_env,
    TandemSyncError,
    RateLimitError,
    ChunkSizeError,
    BadRequestError,
    AuthenticationError,
    DataValidationError,
    APIConnectionError
)

from .utils import (
    get_env_config,
    get_timezone_name,
    get_pump_serial_number,
    get_cache_credentials_setting,
    validate_credentials,
    create_env_template,
    get_env_file_locations
)

from .api import TandemConnector, TandemDataFetcher
from .data import EventExtractor, DataProcessor, DataValidator
from .cli import main as cli_main

__version__ = '1.0.0'

__all__ = [
    # Core components
    'TandemHistoricalSyncClient',
    'PumpConfig',
    'SyncMetadata',
    'CredentialsConfig',
    'load_pump_configs',
    'create_default_pump_configs',
    'get_credentials',
    'load_credentials_from_env',
    
    # Environment utilities
    'get_env_config',
    'get_timezone_name',
    'get_pump_serial_number',
    'get_cache_credentials_setting',
    'validate_credentials',
    'create_env_template',
    'get_env_file_locations',
    
    # API components
    'TandemConnector',
    'TandemDataFetcher',
    
    # Data processing
    'EventExtractor',
    'DataProcessor',
    'DataValidator',
    
    # CLI
    'cli_main',
    
    # Exceptions
    'TandemSyncError',
    'RateLimitError',
    'ChunkSizeError',
    'BadRequestError',
    'AuthenticationError',
    'DataValidationError',
    'APIConnectionError'
]