"""
Real TandemSourceApi implementation copied from tconnectsync
"""

import urllib.parse
import arrow
import time
import logging
import json
import base64
import hashlib
import os
import jwt
import pickle
from datetime import timedelta
from pathlib import Path

from jwt.algorithms import RSAAlgorithm
import requests

from .common import parse_ymd_date, base_headers, base_session, ApiException, ApiLoginException
from ..eventparser.generic import Events

logger = logging.getLogger(__name__)

# Cache settings
CACHE_CREDENTIALS = True
CACHE_CREDENTIALS_PATH = Path.home() / '.tconnectsync' / 'cache.pkl'

class TandemSourceApi:
    # Common URLs that are shared between regions
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

    def __init__(self, email, password, region='US'):
        self.region = region.upper()
        if self.region not in ['US', 'EU']:
            raise ValueError(f"Invalid region '{region}'. Must be 'US' or 'EU'.")
        
        self._region_urls = self._US_URLS if self.region == 'US' else self._EU_URLS
        
        # Initialize attributes
        self.jwtData = None
        self.pumperId = None
        self.accountId = None
        self.idToken = None
        self.accessToken = None
        self.accessTokenExpiresAt = None
        self.loginSession = None
        
        self.login(email, password)
        self._email = email
        self._password = password

    @property
    def LOGIN_API_URL(self):
        return self._region_urls['LOGIN_API_URL']
    
    @property
    def TDC_OAUTH_AUTHORIZE_URL(self):
        return self._region_urls['TDC_OAUTH_AUTHORIZE_URL']
    
    @property
    def TDC_OIDC_JWKS_URL(self):
        return self._region_urls['TDC_OIDC_JWKS_URL']
    
    @property
    def TDC_OIDC_ISSUER(self):
        return self._region_urls['TDC_OIDC_ISSUER']
    
    @property
    def TDC_OIDC_CLIENT_ID(self):
        return self._region_urls['TDC_OIDC_CLIENT_ID']
    
    @property
    def SOURCE_URL(self):
        return self._region_urls['SOURCE_URL']

    def login(self, email, password):
        logger.info(f"Logging in to TandemSourceApi ({self.region} region)...")
        if self.try_load_cached_creds(email):
            logger.info("Successfully used cached credentials")
            return True

        with base_session() as s:
            initial = s.get(self.LOGIN_PAGE_URL, headers=base_headers())

            data = {
                "username": email,
                "password": password
            }

            req = s.post(self.LOGIN_API_URL, json=data, headers={'Referer': self.LOGIN_PAGE_URL, **base_headers()}, allow_redirects=False)

            logger.debug("1. made POST to LOGIN_API")
            # {"redirectUrl":"/","status":"SUCCESS"}
            if req.status_code != 200:
                raise ApiException(req.status_code, 'Error sending POST to login_api_url: %s' % req.text)

            req_json = req.json()
            login_ok = req_json.get('status', '') == 'SUCCESS'

            if not login_ok:
                raise ApiException(req.status_code, 'Error parsing login_api_url: %s' % json.dumps(req_json))

            logger.debug("2. starting OIDC")

            # oidc
            client_id = self.TDC_OIDC_CLIENT_ID
            redirect_uri = self._region_urls['REDIRECT_URI']
            scope = 'openid profile email'

            token_endpoint = self._region_urls['TOKEN_ENDPOINT']

            def generate_code_verifier():
                """Generates a high-entropy code verifier."""
                code_verifier = base64.urlsafe_b64encode(os.urandom(64)).decode('utf-8').rstrip('=')
                return code_verifier

            def generate_code_challenge(verifier):
                """Generates a code challenge from the code verifier."""
                sha256_digest = hashlib.sha256(verifier.encode('utf-8')).digest()
                code_challenge = base64.urlsafe_b64encode(sha256_digest).decode('utf-8').rstrip('=')
                return code_challenge

            code_verifier = generate_code_verifier()
            code_challenge = generate_code_challenge(code_verifier)

            authorization_endpoint = self._region_urls['AUTHORIZATION_ENDPOINT']

            oidc_step1_params = {
                'client_id': client_id,
                'response_type': 'code',
                'scope': scope,
                'redirect_uri': redirect_uri,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256',
            }

            logger.debug("3. calling oidc_step1 with %s" % json.dumps(oidc_step1_params))
            oidc_step1 = s.get(
                authorization_endpoint + '?' + urllib.parse.urlencode(oidc_step1_params),
                headers={'Referer': self.LOGIN_PAGE_URL, **base_headers()},
                allow_redirects=True
            )

            if oidc_step1.status_code // 100 != 2:
                raise ApiException(oidc_step1.status_code, 'Got unexpected status code for oidc step1: %s' % oidc_step1.text)

            oidc_step1_loc = oidc_step1.url
            oidc_step1_query = urllib.parse.parse_qs(urllib.parse.urlparse(oidc_step1_loc).query)
            if 'code' not in oidc_step1_query:
                raise ApiException(oidc_step1.status_code, 'No code for oidc step1 ReturnUrl (%s): %s' % (oidc_step1_loc, json.dumps(oidc_step1_query)))

            oidc_step1_callback_code = oidc_step1_query['code'][0]

            oidc_step2_token_data = {
                'grant_type': 'authorization_code',
                'client_id': client_id,
                'code': oidc_step1_callback_code,
                'redirect_uri': redirect_uri,
                'code_verifier': code_verifier,
            }

            logger.debug("4. calling oidc_step2 with %s" % json.dumps(oidc_step2_token_data))

            oidc_step2 = s.post(token_endpoint, data=oidc_step2_token_data, headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                **base_headers()
            })

            if oidc_step2.status_code//100 != 2:
                raise ApiException(oidc_step1.status_code, 'Got unexpected status code for oidc step2: %s' % oidc_step1.text)

            oidc_json = oidc_step2.json()
            logger.debug("5. parsing oidc_step2 json response: %s" % json.dumps(oidc_json))

            if not 'access_token' in oidc_json:
                raise ApiException(oidc_step1.status_code, 'Missing access_token in oidc_step2 json: %s' % json.dumps(oidc_json))

            if not 'id_token' in oidc_json:
                raise ApiException(oidc_step1.status_code, 'Missing id_token in oidc_step2 json: %s' % json.dumps(oidc_json))

            self.loginSession = s
            self.idToken = oidc_json['id_token']
            self.extract_jwt()

            self.accessToken = oidc_json['access_token']
            self.accessTokenExpiresAt = arrow.get(arrow.get().int_timestamp + oidc_json['expires_in'])

            self.cache_creds(email)

            return True

    def extract_jwt(self):
        logger.debug("6. extracting JWT from %s" % self.idToken)
        id_token = self.idToken

        jwks_response = self.loginSession.get(self.TDC_OIDC_JWKS_URL)
        jwks = jwks_response.json()
        public_keys = {}
        for jwk in jwks['keys']:
            kid = jwk['kid']
            public_keys[kid] = RSAAlgorithm.from_jwk(json.dumps(jwk))

        # Get the key ID (kid) from the headers of the ID Token
        unverified_header = jwt.get_unverified_header(id_token)
        kid = unverified_header['kid']

        key = public_keys.get(kid)
        if not key:
            raise ApiException(0, 'Public key not found for JWT: %s' % kid)

        audience = self.TDC_OIDC_CLIENT_ID
        issuer = self.TDC_OIDC_ISSUER

        # Decode and verify the ID Token
        # Add leeway to handle small time differences between client and server
        id_token_claims = jwt.decode(
            id_token,
            key=key,
            algorithms=['RS256'],
            audience=audience,
            issuer=issuer,
            leeway=timedelta(seconds=300)  # Allow 5 minutes tolerance
        )

        logger.info("Decoded JWT: %s" % json.dumps(id_token_claims))

        self.jwtData = id_token_claims
        self.pumperId = id_token_claims.get('pumperId')
        self.accountId = id_token_claims.get('accountId')

    def try_load_cached_creds(self, email):
        if not CACHE_CREDENTIALS:
            return False

        if not os.path.exists(CACHE_CREDENTIALS_PATH):
            logger.info("No cached credentials exist")
            return False

        _saved_blob = {}
        try:
            with open(CACHE_CREDENTIALS_PATH, 'rb') as f:
                _saved_blob = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load cached credentials at {CACHE_CREDENTIALS_PATH}: {e}")
            return False

        if not _saved_blob:
            logger.warning(f"Could not load cached credentials at {CACHE_CREDENTIALS_PATH}: empty dict")
            return False

        if _saved_blob.get('cache_creds_version') != 1.0:
            logger.warning(f"Unexpected cache_creds_version at {CACHE_CREDENTIALS_PATH}: {_saved_blob['cache_creds_version']}, expected 1.0")
            return False

        if _saved_blob.get('cache_creds_email') != email:
            logger.warning(f"Cached credentials are for a different email ({_saved_blob['cache_creds_email']} in cache, but using {email}), skipping")
            return False

        # Check if cached region matches current region
        cached_region = _saved_blob.get('cache_creds_region', 'US')  # Default to US for backward compatibility
        if cached_region != self.region:
            logger.warning(f"Cached credentials are for a different region ({cached_region} in cache, but using {self.region}), skipping")
            return False

        at_expiry = _saved_blob['accessTokenExpiresAt']
        if arrow.get().int_timestamp >= arrow.get(at_expiry).int_timestamp:
            logger.info(f"Cached credentials have expired ({_saved_blob['accessTokenExpiresAt']}), skipping")
            return False

        self.jwtData = _saved_blob['jwtData']
        self.pumperId = _saved_blob['pumperId']
        self.accountId = _saved_blob['accountId']
        self.idToken = _saved_blob['idToken']
        self.accessToken = _saved_blob['accessToken']
        self.accessTokenExpiresAt = _saved_blob['accessTokenExpiresAt']
        self.loginSession = _saved_blob['loginSession']

        return True

    def cache_creds(self, email):
        if not CACHE_CREDENTIALS:
            return

        # Ensure cache directory exists
        CACHE_CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

        _saved_blob = {
            'cache_creds_version': 1.0,
            'cache_creds_email': email,
            'cache_creds_region': self.region,
            'jwtData': self.jwtData,
            'pumperId': self.pumperId,
            'accountId': self.accountId,
            'idToken': self.idToken,
            'accessToken': self.accessToken,
            'accessTokenExpiresAt': self.accessTokenExpiresAt,
            'loginSession': self.loginSession
        }

        try:
            with open(CACHE_CREDENTIALS_PATH, 'wb') as f:
                pickle.dump(_saved_blob, f)
            logger.info(f"Cached credentials to {CACHE_CREDENTIALS_PATH}")
        except Exception as e:
            logger.warning(f"Could not cache credentials to {CACHE_CREDENTIALS_PATH}: {e}")

    def needs_relogin(self):
        if not self.accessToken or not self.accessTokenExpiresAt:
            return True
        
        # Check if token expires in next 5 minutes
        return arrow.get().int_timestamp >= (arrow.get(self.accessTokenExpiresAt).int_timestamp - 300)

    def api_headers(self):
        """
        Headers for authenticated data requests to the Tandem Source API.

        The WAF enforces same-origin: Origin/Referer must match SOURCE_URL
        (source.tandemdiabetes.com / source.eu.tandemdiabetes.com), otherwise it
        returns HTTP 403 ("The request is blocked"). Mirrors upstream v3.0.0
        api_headers().
        """
        if not self.accessToken:
            raise ApiException(0, 'No access token provided')
        return {
            'Authorization': f'Bearer {self.accessToken}',
            'Origin': self.SOURCE_URL.rstrip('/'),
            'Referer': self.SOURCE_URL,
            **base_headers()
        }

    def get(self, endpoint, query, tries=0):
        """
        Make a GET request to the API with proper (WAF-compliant) authentication.

        Args:
            endpoint: API endpoint path (relative to SOURCE_URL)
            query: Query parameters (sent as request body, matching upstream)
            tries: Number of retry attempts

        Returns:
            Response JSON data
        """
        if self.needs_relogin():
            logger.info("Access token expired, re-logging in...")
            self.login(self._email, self._password)

        try:
            url = f"{self.SOURCE_URL}{endpoint}"
            with base_session() as s:
                response = s.get(url, headers=self.api_headers(), data=query)

                if response.status_code != 200:
                    raise ApiException(response.status_code, f'Error getting {endpoint}: {response.text}')

                return response.json()

        except Exception as e:
            logger.error(f"Error making GET request to {endpoint}: {e}")
            raise

    def pump_event_metadata(self):
        """
        Get the list of pumps on the account.

        Uses the new BFF pumper endpoint (api/reports/bff/pumper/{pumperId}) and
        maps each pump to the {serialNumber, tconnectDeviceId} shape our consumers
        depend on. tconnectDeviceId is mapped from the BFF assignmentId (the UUID
        device id used by the pump-logs endpoint). Best-effort optional fields
        (modelNumber, softwareVersion, date/upload info) are passed through where
        the BFF provides them.

        Returns:
            List of pump metadata dictionaries
        """
        if self.needs_relogin():
            logger.info("Access token expired, re-logging in...")
            self.login(self._email, self._password)

        try:
            pumper = self.get_pumper()
        except Exception as e:
            logger.error(f"Error getting pump metadata: {e}")
            raise

        pumps = []
        for pump in (pumper or {}).get('pumps', []) or []:
            available = pump.get('availableDataRange') or {}
            pumps.append({
                'serialNumber': pump.get('serialNumber'),
                # tconnectDeviceId <- BFF assignmentId (UUID device id)
                'tconnectDeviceId': pump.get('assignmentId'),
                'modelNumber': pump.get('modelNumber'),
                'modelName': pump.get('modelName'),
                'softwareVersion': pump.get('softwareVersion'),
                'minDateWithEvents': available.get('start'),
                'maxDateWithEvents': available.get('end') or pump.get('maxDateOfEvents'),
                'lastUpload': pump.get('lastUploadDate'),
                # Preserve the raw BFF pump for any downstream consumer.
                'raw': pump,
            })
        return pumps

    def get_pumper(self):
        """
        Returns the pumper's profile plus the list of pumps on the account
        (BffPumper.pumps) from the new BFF endpoint. Replaces the old reportsfacade
        pump-event-metadata endpoint: pumps[].assignmentId is the UUID device id
        used by the pump-logs endpoint.
        """
        return self.get(f'api/reports/bff/pumper/{self.pumperId}', {})

    # Matches the Tandem Source web app's getLogIDList() as observed in the live
    # GET api/reports/bff/pump-logs request. Includes FSL3 ids 477/480/486.
    DEFAULT_EVENT_IDS = [229,5,28,4,26,99,279,3,16,59,21,55,20,280,64,65,66,61,33,371,171,369,460,172,370,461,372,480,399,256,213,406,477,394,212,404,214,405,486,447,313,60,14,6,90,230,140,12,11,53,13,63,203,307,191]

    # The pump-logs endpoint caps each request at roughly four weeks, so a
    # longer range is paged in windows no larger than this.
    PUMP_LOGS_WINDOW_DAYS = 28

    def get_pump_logs(self, device_id, min_date=None, max_date=None, event_ids_filter=DEFAULT_EVENT_IDS):
        """
        Fetch pre-decoded pump events for a single date window from the BFF
        endpoint GET api/reports/bff/pump-logs/{device_id}. device_id is the UUID
        assignmentId (from pump_event_metadata()/get_pumper()). Returns
        {events, clockChanges}.

        The server caps the window at ~4 weeks; callers needing a longer range
        must page by date window (see pump_events).

        Note: the server currently ignores eventIds and returns every event in the
        window regardless of the filter, so effective filtering happens
        client-side via EventClass dispatch. We still send eventIds to mirror the
        web app and stay forward-compatible.
        """
        minDate = parse_ymd_date(min_date)
        maxDate = parse_ymd_date(max_date)
        logger.debug(f'get_pump_logs({device_id}, {minDate}, {maxDate})')

        query = urllib.parse.urlencode({
            'pumperId': self.pumperId,
            'startDate': f'{minDate}T00:00:00Z',
            'endDate': f'{maxDate}T23:59:59Z',
            'eventIds': ','.join(map(str, event_ids_filter)) if event_ids_filter else '',
        })
        return self.get(f'api/reports/bff/pump-logs/{device_id}?{query}', {})

    @classmethod
    def _pump_log_windows(cls, min_date, max_date):
        """
        Split the (min_date, max_date) range into inclusive date windows no
        larger than PUMP_LOGS_WINDOW_DAYS. A None bound defaults to today (via
        parse_ymd_date), so an unset range yields a single one-day window.
        """
        start = arrow.get(parse_ymd_date(min_date))
        end = arrow.get(parse_ymd_date(max_date))
        if end < start:
            start, end = end, start

        windows = []
        cur = start
        while cur <= end:
            win_end = min(cur.shift(days=cls.PUMP_LOGS_WINDOW_DAYS - 1), end)
            windows.append((cur.format('YYYY-MM-DD'), win_end.format('YYYY-MM-DD')))
            cur = win_end.shift(days=1)
        return windows

    def pump_events_raw(self, tconnect_device_id, min_date=None, max_date=None, event_ids_filter=None):
        """
        Return the raw (undecoded) pump-logs events list for the given device and
        date range. Thin wrapper over the new BFF pump-logs endpoint; used by the
        fetcher purely as a non-empty gate before calling pump_events(). Pages the
        range in <=28-day windows so a truthy list is returned whenever any events
        exist in the requested range.
        """
        if self.needs_relogin():
            logger.info("Access token expired, re-logging in...")
            self.login(self._email, self._password)

        if event_ids_filter is None:
            event_ids_filter = self.DEFAULT_EVENT_IDS

        events = []
        for window_start, window_end in self._pump_log_windows(min_date, max_date):
            resp = self.get_pump_logs(tconnect_device_id, window_start, window_end, event_ids_filter)
            events.extend((resp or {}).get('events') or [])
            if events:
                # Enough to satisfy the non-empty gate; avoid extra requests.
                break
        return events

    def pump_events(self, tconnect_device_id, min_date=None, max_date=None, fetch_all_event_types=False):
        """
        Fetch and parse pump events from the pump-logs endpoint.
        Default of fetch_all_event_types=False filters to the same event ids used
        in the Tandem Source backend. If fetch_all_event_types=True, then all event
        types from the history log are returned.
        tconnect_device_id is the UUID assignmentId from pump_event_metadata().
        """
        event_ids_filter = None if fetch_all_event_types else self.DEFAULT_EVENT_IDS

        # Page across date windows, deduplicating events that appear in more than
        # one window by their (sequenceGroup, sequenceNumber) identity.
        seen = set()
        events = []
        clock_change_count = 0
        for window_start, window_end in self._pump_log_windows(min_date, max_date):
            resp = self.get_pump_logs(tconnect_device_id, window_start, window_end, event_ids_filter)
            clock_change_count += len((resp or {}).get('clockChanges') or [])
            for event in (resp or {}).get('events') or []:
                key = (event.get('sequenceGroup'), event.get('sequenceNumber'))
                if key in seen:
                    continue
                seen.add(key)
                events.append(event)

        logger.info(f"Read {len(events)} events ({clock_change_count} clock changes skipped)")
        return Events(events)

    def pumper_info(self):
        """
        Get information about the user and available pumps
        
        Returns:
            Dictionary with pumper information
        """
        endpoint = f'api/pumpers/pumpers/{self.pumperId}'
        return self.get(endpoint, {})
