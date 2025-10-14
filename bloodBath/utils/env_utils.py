"""
Environment variable utilities for bloodBath
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


def load_env_file(env_file: Optional[Path] = None) -> bool:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Optional path to .env file. If not provided, searches in:
                 - Current directory
                 - Project root directory
                 - User home directory
    
    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    search_paths = []
    
    if env_file:
        search_paths.append(env_file)
    else:
        # Search in multiple locations
        current_dir = Path.cwd()
        project_root = Path(__file__).parent.parent.parent  # Go up from utils/env_utils.py
        home_dir = Path.home()
        
        search_paths.extend([
            current_dir / '.env',
            project_root / '.env',
            home_dir / '.env',
            home_dir / '.config' / 'bloodBath' / '.env'
        ])
    
    for path in search_paths:
        if path.exists():
            logger.debug(f"Loading .env file from: {path}")
            load_dotenv(path)
            return True
    
    logger.debug(f"No .env file found in search paths: {[str(p) for p in search_paths]}")
    return False


def get_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Get email and password from environment variables
    
    Returns:
        tuple: (email, password) or (None, None) if not found
    """
    # Try to load .env file first
    load_env_file()
    
    # Get credentials from environment
    email = os.getenv('TCONNECT_EMAIL') or os.getenv('BLOODBATH_EMAIL')
    password = os.getenv('TCONNECT_PASSWORD') or os.getenv('BLOODBATH_PASSWORD')
    
    return email, password


def get_env_config() -> dict:
    """
    Get all bloodBath-related environment variables
    
    Returns:
        dict: Configuration dictionary with environment variables
    """
    # Try to load .env file first
    load_env_file()
    
    config = {}
    
    # Credentials
    email, password = get_credentials()
    if email:
        config['email'] = email
    if password:
        config['password'] = password
    
    # Optional configuration
    region = os.getenv('TCONNECT_REGION') or os.getenv('BLOODBATH_REGION')
    if region:
        config['region'] = region
    
    # Timezone configuration
    timezone_name = os.getenv('TIMEZONE_NAME') or os.getenv('BLOODBATH_TIMEZONE')
    if timezone_name:
        config['timezone_name'] = timezone_name
    
    # Pump serial number
    pump_serial = os.getenv('PUMP_SERIAL_NUMBER') or os.getenv('BLOODBATH_PUMP_SERIAL')
    if pump_serial:
        config['pump_serial_number'] = pump_serial
    
    # Cache credentials setting
    cache_credentials = os.getenv('CACHE_CREDENTIALS') or os.getenv('BLOODBATH_CACHE_CREDENTIALS')
    if cache_credentials:
        config['cache_credentials'] = cache_credentials.lower() in ('true', '1', 'yes', 'on')
    
    # Output directory
    output_dir = os.getenv('BLOODBATH_OUTPUT_DIR')
    if output_dir:
        config['output_dir'] = output_dir
    
    # Log level
    log_level = os.getenv('BLOODBATH_LOG_LEVEL')
    if log_level:
        config['log_level'] = log_level
    
    return config


def get_timezone_name() -> Optional[str]:
    """
    Get timezone name from environment variables
    
    Returns:
        str: Timezone name or None if not found
    """
    load_env_file()
    return os.getenv('TIMEZONE_NAME') or os.getenv('BLOODBATH_TIMEZONE')


def get_pump_serial_number() -> Optional[str]:
    """
    Get pump serial number from environment variables
    
    Returns:
        str: Pump serial number or None if not found
    """
    load_env_file()
    return os.getenv('PUMP_SERIAL_NUMBER') or os.getenv('BLOODBATH_PUMP_SERIAL')


def get_cache_credentials_setting() -> bool:
    """
    Get cache credentials setting from environment variables
    
    Returns:
        bool: True if credentials should be cached, False otherwise
    """
    load_env_file()
    cache_setting = os.getenv('CACHE_CREDENTIALS') or os.getenv('BLOODBATH_CACHE_CREDENTIALS')
    if cache_setting:
        return cache_setting.lower() in ('true', '1', 'yes', 'on')
    return True  # Default to True


def validate_credentials(email: Optional[str] = None, password: Optional[str] = None) -> bool:
    """
    Validate that credentials are available
    
    Args:
        email: Email to validate (if not provided, gets from environment)
        password: Password to validate (if not provided, gets from environment)
    
    Returns:
        bool: True if both email and password are available
    """
    if not email or not password:
        env_email, env_password = get_credentials()
        email = email or env_email
        password = password or env_password
    
    return bool(email and password)


def create_env_template(output_path: Optional[Path] = None) -> Path:
    """
    Create a template .env file
    
    Args:
        output_path: Path where to create the .env file (default: current directory)
    
    Returns:
        Path: Path to the created .env file
    """
    if not output_path:
        output_path = Path.cwd() / '.env'
    
    template_content = """# bloodBath Configuration
# Copy this file to .env and fill in your credentials

# Tandem t:connect Credentials (Required)
TCONNECT_EMAIL=your@email.com
TCONNECT_PASSWORD=your_password

# Regional and Timezone Configuration
TCONNECT_REGION=US
TIMEZONE_NAME=America/Los_Angeles

# Pump Configuration
PUMP_SERIAL_NUMBER=881235

# Authentication Settings
CACHE_CREDENTIALS=true

# Optional Configuration
BLOODBATH_OUTPUT_DIR=./data
BLOODBATH_LOG_LEVEL=INFO

# Alternative variable names (if you prefer)
# BLOODBATH_EMAIL=your@email.com
# BLOODBATH_PASSWORD=your_password
# BLOODBATH_REGION=US
# BLOODBATH_TIMEZONE=America/Los_Angeles
# BLOODBATH_PUMP_SERIAL=881235
# BLOODBATH_CACHE_CREDENTIALS=true
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    logger.info(f"Created .env template at: {output_path}")
    return output_path


def get_env_file_locations() -> list[Path]:
    """
    Get list of locations where .env files are searched
    
    Returns:
        list[Path]: List of search paths
    """
    current_dir = Path.cwd()
    project_root = Path(__file__).parent.parent.parent
    home_dir = Path.home()
    
    return [
        current_dir / '.env',
        project_root / '.env',
        home_dir / '.env',
        home_dir / '.config' / 'bloodBath' / '.env'
    ]
