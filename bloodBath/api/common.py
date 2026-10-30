"""
Common API utilities and base session management
"""

import datetime
from typing import List, Tuple
import requests
import random
import arrow

def parse_date(date):
    if type(date) == str:
        return date
    return (date or datetime.datetime.now()).strftime('%m-%d-%Y')

def parse_ymd_date(date):
    if type(date) == str:
        date = arrow.get(date)
    return (date or datetime.datetime.now()).strftime('%Y-%m-%d')

def parsed_date_to_arrow(date):
    return arrow.get(datetime.datetime.strptime(date, '%m-%d-%Y'))

# Current (2025-2026) Chrome / Firefox desktop user agents. The Tandem Source
# WAF is more forgiving of a recent desktop-browser UA; keep this list small and
# up to date rather than exhaustive.
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0',
]

def base_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def base_session():
    s = requests.Session()
    s.headers.update(base_headers())
    return s

class ApiException(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class ApiLoginException(ApiException):
    pass
