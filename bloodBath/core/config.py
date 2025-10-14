"""
Configuration classes for pump configs and sync metadata
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# bloodBath Data Architecture - Schema v2.0
BLOODBANK_ROOT = Path(__file__).parent.parent / "bloodBank"
DATA_PATHS = {
    'raw': {
        'cgm': BLOODBANK_ROOT / "raw" / "cgm",
        'basal': BLOODBANK_ROOT / "raw" / "basal",
        'bolus': BLOODBANK_ROOT / "raw" / "bolus",
        'lstm': BLOODBANK_ROOT / "raw" / "lstm",
        'metadata': BLOODBANK_ROOT / "raw" / "metadata"
    },
    'merged': {
        'train': BLOODBANK_ROOT / "merged" / "train",
        'validate': BLOODBANK_ROOT / "merged" / "validate",
        'test': BLOODBANK_ROOT / "merged" / "test"
    },
    'archives': {
        'legacy': BLOODBANK_ROOT / "archives" / "legacy",
        'logs': BLOODBANK_ROOT / "archives" / "logs"
    }
}


@dataclass
class CredentialsConfig:
    """Configuration for authentication credentials"""
    email: Optional[str] = None
    password: Optional[str] = None
    region: str = 'US'
    timezone_name: Optional[str] = None
    cache_credentials: bool = True
    
    def is_valid(self) -> bool:
        """Check if credentials are valid"""
        return bool(self.email and self.password)


@dataclass
class PumpConfig:
    """Configuration for a single pump"""
    serial: str
    start_date: str
    end_date: str
    device_id: Optional[int] = None
    model: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SyncMetadata:
    """Metadata tracking for sync operations"""
    pump_serial: str
    last_successful_sync: Optional[str] = None
    failed_ranges: Optional[List[Dict[str, str]]] = None
    total_records: int = 0
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.failed_ranges is None:
            self.failed_ranges = []


def load_credentials_from_env() -> CredentialsConfig:
    """
    Load credentials from environment variables
    
    Returns:
        CredentialsConfig: Configuration with credentials from environment
    """
    from ..utils.env_utils import get_env_config
    
    env_config = get_env_config()
    
    return CredentialsConfig(
        email=env_config.get('email'),
        password=env_config.get('password'),
        region=env_config.get('region', 'US'),
        timezone_name=env_config.get('timezone_name'),
        cache_credentials=env_config.get('cache_credentials', True)
    )


def get_credentials(email: Optional[str] = None, password: Optional[str] = None, 
                   region: Optional[str] = None, timezone_name: Optional[str] = None,
                   cache_credentials: Optional[bool] = None) -> CredentialsConfig:
    """
    Get credentials from parameters or environment variables
    
    Args:
        email: Email address (if not provided, gets from environment)
        password: Password (if not provided, gets from environment)
        region: Region (if not provided, gets from environment or defaults to 'US')
        timezone_name: Timezone name (if not provided, gets from environment)
        cache_credentials: Whether to cache credentials (if not provided, gets from environment)
    
    Returns:
        CredentialsConfig: Configuration with credentials
    """
    # Load from environment first
    env_creds = load_credentials_from_env()
    
    # Override with provided parameters
    return CredentialsConfig(
        email=email or env_creds.email,
        password=password or env_creds.password,
        region=region or env_creds.region,
        timezone_name=timezone_name or env_creds.timezone_name,
        cache_credentials=cache_credentials if cache_credentials is not None else env_creds.cache_credentials
    )


def get_default_pump_serial() -> Optional[str]:
    """
    Get default pump serial from environment variables
    
    Returns:
        str: Default pump serial number or None if not found
    """
    from ..utils.env_utils import get_pump_serial_number
    return get_pump_serial_number()


def load_pump_configs(config_file: str) -> List[PumpConfig]:
    """Load pump configurations from JSON file"""
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        configs = []
        for pump_data in data.get('pumps', []):
            config = PumpConfig(**pump_data)
            configs.append(config)
        
        return configs
    except Exception as e:
        logger.error(f"Error loading pump configs from {config_file}: {e}")
        return []


def create_default_pump_configs() -> List[PumpConfig]:
    """Create default pump configurations"""
    return [
        PumpConfig(
            serial='881235',
            start_date='2021-01-01T00:00:00',
            end_date='2024-10-06T23:35:00',
            description='Pump 1 - Historical data'
        ),
        PumpConfig(
            serial='901161470',
            start_date='2024-01-01T00:00:00',
            end_date='2025-07-14T17:33:46',
            description='Pump 2 - Current/recent data'
        )
    ]


def save_metadata(metadata: Dict[str, SyncMetadata], metadata_file: Path):
    """Save sync metadata to file"""
    try:
        data = {}
        for serial, metadata_obj in metadata.items():
            data[serial] = asdict(metadata_obj)
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Could not save metadata: {e}")


def load_metadata(metadata_file: Path) -> Dict[str, SyncMetadata]:
    """Load sync metadata from file"""
    if not metadata_file.exists():
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        metadata = {}
        for serial, meta_dict in data.items():
            metadata[serial] = SyncMetadata(**meta_dict)
        
        return metadata
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
        return {}
