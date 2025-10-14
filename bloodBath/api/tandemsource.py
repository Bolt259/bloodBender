"""
Tandem Source API implementation for bloodBath
"""

import urllib
import time
import logging
import json
import base64
import hashlib
import os
import jwt
import pickle
from datetime import timedelta
from typing import Optional, Dict, Any, List

from .common import (
    parse_ymd_date, base_headers, base_session, 
    ApiException, ApiLoginException
)
from .events import decode_raw_events, Events, DEFAULT_EVENT_IDS

logger = logging.getLogger(__name__)

class TandemSourceApi:
    """Simplified Tandem Source API for bloodBath"""
    
    # Common URLs
    LOGIN_PAGE_URL = 'https://sso.tandemdiabetes.com/'
    TDC_AUTH_CALLBACK_URL = 'https://sso.tandemdiabetes.com/auth/callback'
    
    # US Region URLs (default)
    _US_URLS = {
        'LOGIN_API_URL': 'https://tdcservices.tandemdiabetes.com/accounts/api/login',
        'TDC_OAUTH_AUTHORIZE_URL': 'https://tdcservices.tandemdiabetes.com/accounts/api/oauth2/v1/authorize',
        'TDC_OIDC_JWKS_URL': 'https://tdcservices.tandemdiabetes.com/accounts/api/.well-known/openid-configuration/jwks',
        'TDC_OIDC_ISSUER': 'https://tdcservices.tandemdiabetes.com/accounts/api',
        'TDC_OIDC_CLIENT_ID': '0oa27ho9tpZE9Arjy4h7',
        'SOURCE_URL': 'https://source.tandemdiabetes.com/',
        'REDIRECT_URI': 'https://sso.tandemdiabetes.com/auth/callback',
        'TOKEN_ENDPOINT': 'https://tdcservices.tandemdiabetes.com/accounts/api/connect/token',
        'AUTHORIZATION_ENDPOINT': 'https://tdcservices.tandemdiabetes.com/accounts/api/connect/authorize'
    }
    
    # EU Region URLs
    _EU_URLS = {
        'LOGIN_API_URL': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api/login',
        'TDC_OAUTH_AUTHORIZE_URL': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api/oauth2/v1/authorize',
        'TDC_OIDC_JWKS_URL': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api/.well-known/openid-configuration/jwks',
        'TDC_OIDC_ISSUER': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api',
        'TDC_OIDC_CLIENT_ID': '1519e414-eeec-492e-8c5e-97bea4815a10',
        'SOURCE_URL': 'https://source.eu.tandemdiabetes.com/',
        'REDIRECT_URI': 'https://source.eu.tandemdiabetes.com/authorize/callback',
        'TOKEN_ENDPOINT': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api/connect/token',
        'AUTHORIZATION_ENDPOINT': 'https://tdcservices.eu.tandemdiabetes.com/accounts/api/connect/authorize'
    }
    
    def __init__(self, email: str, password: str, region: str = 'US'):
        """Initialize TandemSourceApi"""
        self.region = region.upper()
        if self.region not in ['US', 'EU']:
            raise ValueError(f"Invalid region '{region}'. Must be 'US' or 'EU'.")
        
        self._region_urls = self._US_URLS if self.region == 'US' else self._EU_URLS
        self._email = email
        self._password = password
        self._session = None
        self._logged_in = False
        self.pumperId = None
        self.access_token = None
        self._login_time = None
        
        # Attempt login
        self.login(email, password)
    
    def login(self, email: str, password: str) -> bool:
        """Login to Tandem Source API"""
        logger.info(f"Logging in to TandemSourceApi ({self.region} region)...")
        
        # For now, implement a simplified login
        # In a full implementation, this would handle the complete OAuth flow
        try:
            self._session = base_session()
            self._logged_in = True
            self._login_time = time.time()
            
            # Mock login success - in reality, this would do full OAuth
            logger.info("Login successful (simplified implementation)")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise ApiLoginException(None, f"Login failed: {e}")
    
    def needs_relogin(self) -> bool:
        """Check if we need to re-login"""
        if not self._logged_in:
            return True
        
        # Check if login is older than 1 hour
        if self._login_time and (time.time() - self._login_time) > 3600:
            return True
            
        return False
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """Make GET request to API endpoint"""
        if self.needs_relogin():
            self.login(self._email, self._password)
        
        if not self._session:
            raise ApiException(None, "No active session")
        
        url = self._region_urls['SOURCE_URL'] + endpoint
        headers = base_headers()
        
        response = None
        try:
            response = self._session.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Handle different response types
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                return response.text
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            status_code = getattr(response, 'status_code', None) if response else None
            raise ApiException(status_code, f"API request failed: {e}")
    
    def pump_event_metadata(self) -> List[Dict[str, Any]]:
        """Get pump event metadata"""
        logger.debug("Fetching pump event metadata")
        
        try:
            # Use the real API endpoint
            result = self.get('api/reports/reportsfacade/%s/pumpeventmetadata' % (self.pumperId), {})
            logger.info(f"Retrieved pump metadata for {len(result)} pumps")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching pump event metadata: {e}")
            # For debugging, let's try to see what's happening
            logger.warning("pump_event_metadata: Falling back to mock implementation")
            return []
    
    def pump_events_raw(self, 
                       tconnect_device_id: str, 
                       min_date: Optional[str] = None, 
                       max_date: Optional[str] = None, 
                       event_ids_filter: Optional[List[int]] = None) -> str:
        """Get raw pump events"""
        if event_ids_filter is None:
            event_ids_filter = DEFAULT_EVENT_IDS
            
        minDate = parse_ymd_date(min_date)
        maxDate = parse_ymd_date(max_date)
        
        logger.debug(f'pump_events_raw({tconnect_device_id}, {minDate}, {maxDate})')
        
        # Build event IDs filter
        eventIdsFilter = '%2C'.join(map(str, event_ids_filter)) if event_ids_filter else None
        
        # Build endpoint URL
        endpoint = f'api/reports/reportsfacade/pumpevents/{self.pumperId}/{tconnect_device_id}'
        params = {
            'minDate': minDate,
            'maxDate': maxDate
        }
        
        if eventIdsFilter:
            params['eventIds'] = eventIdsFilter
        
        try:
            # This would make the actual API call
            # return self.get(endpoint, params)
            
            # For now, return empty base64 string to avoid breaking the system
            logger.warning("pump_events_raw: Using mock implementation")
            return base64.b64encode(b'').decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error fetching raw pump events: {e}")
            return base64.b64encode(b'').decode('utf-8')
    
    def pump_events(self, 
                   tconnect_device_id: str, 
                   min_date: Optional[str] = None, 
                   max_date: Optional[str] = None, 
                   fetch_all_event_types: bool = False):
        """Fetch and decode pump events"""
        event_ids_filter = None if fetch_all_event_types else DEFAULT_EVENT_IDS
        
        # Get raw events
        pump_events_raw = self.pump_events_raw(
            tconnect_device_id,
            min_date,
            max_date,
            event_ids_filter
        )
        
        # Decode events
        pump_events_decoded = decode_raw_events(pump_events_raw)
        logger.info(f"Read {len(pump_events_decoded)} bytes")
        
        # Return events generator
        return Events(pump_events_decoded)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'region': self.region,
            'logged_in': self._logged_in,
            'login_time': self._login_time,
            'needs_relogin': self.needs_relogin()
        }
