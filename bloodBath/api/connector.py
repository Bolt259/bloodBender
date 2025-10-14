"""
API connector for Tandem t:connect authentication and session management
"""

import time
import logging
from typing import Optional

from .real_tandemsource import TandemSourceApi
from ..core.exceptions import AuthenticationError, APIConnectionError

logger = logging.getLogger(__name__)

# For backward compatibility with tconnectsync
class TConnectApi:
    """Wrapper to maintain compatibility with tconnectsync"""
    def __init__(self, email: str, password: str, region: str = 'US'):
        self.email = email
        self.password = password
        self.region = region
        self._tandemsource = None
    
    @property
    def tandemsource(self) -> TandemSourceApi:
        if self._tandemsource and not self._tandemsource.needs_relogin():
            return self._tandemsource
        
        self._tandemsource = TandemSourceApi(self.email, self.password, self.region)
        return self._tandemsource

class TandemConnector:
    """
    Handles authentication and API session management for Tandem t:connect
    """
    
    def __init__(self, 
                 email: Optional[str] = None,
                 password: Optional[str] = None,
                 region: str = 'US'):
        """
        Initialize the connector
        
        Args:
            email: t:connect email
            password: t:connect password
            region: Server region (US or EU)
        """
        # For now, require explicit credentials
        if not email or not password:
            raise AuthenticationError("Email and password are required")
            
        self.email = email
        self.password = password
        self.region = region
        self.api = None
        self._last_connection_time = None
    
    def connect(self) -> TConnectApi:
        """
        Establish API connection and authenticate
        
        Returns:
            Authenticated TConnectApi instance
            
        Raises:
            AuthenticationError: If authentication fails
            APIConnectionError: If connection fails
        """
        if self.api is None:
            logger.info(f"Connecting to Tandem API ({self.region} region)...")
            
            try:
                self.api = TConnectApi(self.email, self.password, self.region)
                self._last_connection_time = time.time()
                logger.info("API connection established successfully")
                
            except Exception as e:
                logger.error(f"Failed to connect to API: {e}")
                raise APIConnectionError(f"Failed to connect to API: {e}")
        
        return self.api
    
    def get_api(self) -> TConnectApi:
        """
        Get the current API instance, connecting if necessary
        
        Returns:
            TConnectApi instance
        """
        if self.api is None:
            return self.connect()
        return self.api
    
    def is_connected(self) -> bool:
        """
        Check if API is connected
        
        Returns:
            True if connected, False otherwise
        """
        return self.api is not None
    
    def disconnect(self):
        """
        Disconnect from API
        """
        if self.api is not None:
            logger.info("Disconnecting from API")
            self.api = None
            self._last_connection_time = None
    
    def get_pump_device_id(self, serial: str) -> Optional[int]:
        """
        Get device ID for a pump serial number
        
        Args:
            serial: Pump serial number
            
        Returns:
            Device ID or None if not found
        """
        try:
            api = self.get_api()
            pump_metadata = api.tandemsource.pump_event_metadata()
            
            for pump in pump_metadata:
                if pump.get('serialNumber') == serial:
                    device_id = pump.get('tconnectDeviceId')
                    logger.debug(f"Found device ID {device_id} for pump {serial}")
                    return device_id
            
            logger.error(f"Pump {serial} not found in account")
            return None
            
        except Exception as e:
            logger.error(f"Error getting device ID for pump {serial}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            api = self.get_api()
            # Try to get pump metadata as a connection test
            pump_metadata = api.tandemsource.pump_event_metadata()
            logger.info(f"Connection test successful - found {len(pump_metadata)} pumps")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        Get connection information
        
        Returns:
            Dictionary with connection details
        """
        return {
            'email': self.email,
            'region': self.region,
            'connected': self.is_connected(),
            'last_connection_time': self._last_connection_time
        }
