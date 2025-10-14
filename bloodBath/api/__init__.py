"""
API module for Tandem t:connect integration
"""

from .connector import TandemConnector
from .fetcher import TandemDataFetcher
from .tandemsource import TandemSourceApi
from .common import ApiException, ApiLoginException

# Main API class for compatibility
class TConnectApi:
    """Main API wrapper for Tandem t:connect"""
    
    def __init__(self, email: str, password: str, region: str = 'US'):
        """Initialize TConnectApi"""
        self.email = email
        self.password = password
        self.region = region
        self._tandemsource = None
    
    @property
    def tandemsource(self) -> TandemSourceApi:
        """Get TandemSourceApi instance"""
        if self._tandemsource and not self._tandemsource.needs_relogin():
            return self._tandemsource
        
        self._tandemsource = TandemSourceApi(self.email, self.password, self.region)
        return self._tandemsource

__all__ = [
    'TandemConnector',
    'TandemDataFetcher', 
    'TConnectApi',
    'TandemSourceApi',
    'ApiException',
    'ApiLoginException'
]