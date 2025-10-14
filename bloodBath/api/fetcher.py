"""
Data fetcher with chunked request logic and retry handling
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from ..core.config import PumpConfig
from ..core.exceptions import (
    TandemSyncError, RateLimitError, ChunkSizeError, 
    BadRequestError, APIConnectionError
)
from ..utils.time_utils import generate_date_chunks
from .connector import TandemConnector

logger = logging.getLogger(__name__)


class TandemDataFetcher:
    """
    Handles chunked data fetching with retry logic and error handling
    """
    
    DEFAULT_CHUNK_DAYS = 30
    MIN_CHUNK_DAYS = 1
    MAX_RETRIES = 5
    RETRY_DELAY_BASE = 2  # seconds
    
    def __init__(self, 
                 connector: TandemConnector,
                 chunk_days: int = DEFAULT_CHUNK_DAYS,
                 max_retries: int = MAX_RETRIES,
                 rate_limit_delay: int = 1):
        """
        Initialize the data fetcher
        
        Args:
            connector: TandemConnector instance
            chunk_days: Size of date chunks in days
            max_retries: Maximum retry attempts
            rate_limit_delay: Base delay between requests
        """
        self.connector = connector
        self.chunk_days = chunk_days
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
    
    def fetch_pump_data_range(self, 
                             pump_config: PumpConfig,
                             start_date: str,
                             end_date: str) -> List[Any]:
        """
        Fetch data for a specific date range, handling chunking automatically
        
        Args:
            pump_config: Pump configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of pump events
        """
        # Generate date chunks
        chunks = generate_date_chunks(start_date, end_date, self.chunk_days)
        logger.info(f"Fetching data for pump {pump_config.serial} in {len(chunks)} chunks")
        
        all_events = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            try:
                events = self._fetch_pump_data_chunk(
                    pump_config, chunk_start, chunk_end
                )
                all_events.extend(events)
                
                # Small delay between chunks
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk {chunk_start} to {chunk_end}: {e}")
                # Continue with other chunks instead of failing completely
                continue
        
        logger.info(f"Total events fetched for pump {pump_config.serial}: {len(all_events)}")
        return all_events
    
    def _fetch_pump_data_chunk(self, 
                              pump_config: PumpConfig,
                              start_date: str,
                              end_date: str,
                              chunk_days: Optional[int] = None) -> List[Any]:
        """
        Fetch data for a specific date range with error handling and retry logic
        
        Args:
            pump_config: Pump configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Current chunk size (for recursive retry)
            
        Returns:
            List of pump events
        """
        if chunk_days is None:
            chunk_days = self.chunk_days
        
        # Ensure we have device ID
        if pump_config.device_id is None:
            pump_config.device_id = self.connector.get_pump_device_id(pump_config.serial)
            if pump_config.device_id is None:
                raise TandemSyncError(f"Could not get device ID for pump {pump_config.serial}")
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                logger.debug(f"Fetching data for pump {pump_config.serial}: {start_date} to {end_date}")
                
                api = self.connector.get_api()
                
                # Add small delay to prevent rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Fetch raw pump events
                raw_events = api.tandemsource.pump_events_raw(
                    pump_config.device_id,
                    min_date=start_date,
                    max_date=end_date
                )
                
                if not raw_events:
                    logger.warning(f"No raw events returned for pump {pump_config.serial} ({start_date} to {end_date})")
                    return []
                
                # Parse events
                events = api.tandemsource.pump_events(
                    pump_config.device_id,
                    min_date=start_date,
                    max_date=end_date,
                    fetch_all_event_types=True
                )
                
                # Convert to list if it's a generator
                if hasattr(events, '__iter__') and not isinstance(events, (dict, str)):
                    if not isinstance(events, list):
                        events = list(events)
                
                # Ensure we return a list
                if not isinstance(events, list):
                    events = [events] if events else []
                
                logger.debug(f"Successfully fetched {len(events)} events for pump {pump_config.serial}")
                return events
                
            except requests.exceptions.HTTPError as e:
                retry_count += 1
                
                if e.response.status_code == 504:  # Gateway Timeout
                    if chunk_days > self.MIN_CHUNK_DAYS:
                        logger.warning(f"Gateway timeout, reducing chunk size from {chunk_days} to {chunk_days // 2}")
                        new_chunk_days = max(chunk_days // 2, self.MIN_CHUNK_DAYS)
                        
                        # Split the current chunk and process recursively
                        return self._split_and_fetch_chunk(
                            pump_config, start_date, end_date, new_chunk_days
                        )
                    else:
                        raise ChunkSizeError(f"Cannot reduce chunk size below {self.MIN_CHUNK_DAYS} days")
                
                elif e.response.status_code == 429:  # Too Many Requests
                    retry_delay = self.RETRY_DELAY_BASE ** retry_count
                    logger.warning(f"Rate limited, waiting {retry_delay} seconds before retry {retry_count}")
                    time.sleep(retry_delay)
                    continue
                
                elif e.response.status_code == 400:  # Bad Request
                    logger.error(f"Bad request for pump {pump_config.serial} ({start_date} to {end_date}): {e}")
                    raise BadRequestError(f"Bad request: {e}")
                
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {e}")
                    if retry_count < self.max_retries:
                        time.sleep(self.RETRY_DELAY_BASE ** retry_count)
                        continue
                    raise TandemSyncError(f"HTTP error after {self.max_retries} retries: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error fetching data: {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.RETRY_DELAY_BASE ** retry_count)
                    continue
                raise TandemSyncError(f"Failed after {self.max_retries} retries: {e}")
        
        raise TandemSyncError(f"Failed to fetch data after {self.max_retries} retries")
    
    def _split_and_fetch_chunk(self, 
                              pump_config: PumpConfig,
                              start_date: str,
                              end_date: str,
                              new_chunk_days: int) -> List[Any]:
        """
        Split a chunk in half and fetch both parts
        
        Args:
            pump_config: Pump configuration
            start_date: Start date
            end_date: End date
            new_chunk_days: New chunk size
            
        Returns:
            Combined list of events from both chunks
        """
        import arrow
        
        # Calculate midpoint
        start_arrow = arrow.get(start_date)
        end_arrow = arrow.get(end_date)
        total_days = (end_arrow - start_arrow).days
        mid_date = start_arrow.shift(days=total_days // 2).format('YYYY-MM-DD')
        
        logger.info(f"Splitting chunk: {start_date} to {end_date} -> {start_date} to {mid_date} and {mid_date} to {end_date}")
        
        # Fetch both parts
        events1 = self._fetch_pump_data_chunk(pump_config, start_date, mid_date, new_chunk_days)
        events2 = self._fetch_pump_data_chunk(pump_config, mid_date, end_date, new_chunk_days)
        
        # Combine results
        all_events = events1 + events2
        logger.info(f"Split chunk results: {len(events1)} + {len(events2)} = {len(all_events)} events")
        
        return all_events
    
    def fetch_pump_events_for_date_range(self,
                                        pump_config: PumpConfig,
                                        start_date: str,
                                        end_date: str,
                                        event_types: Optional[List[str]] = None) -> List[Any]:
        """
        Fetch pump events for a specific date range with optional event type filtering
        
        Args:
            pump_config: Pump configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            event_types: List of event types to filter (optional)
            
        Returns:
            List of filtered pump events
        """
        events = self.fetch_pump_data_range(pump_config, start_date, end_date)
        
        # Filter by event types if specified
        if event_types:
            filtered_events = []
            for event in events:
                event_type = type(event).__name__
                if any(event_type_filter in event_type for event_type_filter in event_types):
                    filtered_events.append(event)
            
            logger.info(f"Filtered {len(events)} events to {len(filtered_events)} events of types: {event_types}")
            return filtered_events
        
        return events
    
    def get_fetcher_stats(self) -> Dict[str, Any]:
        """
        Get fetcher statistics
        
        Returns:
            Dictionary with fetcher configuration and stats
        """
        return {
            'chunk_days': self.chunk_days,
            'max_retries': self.max_retries,
            'rate_limit_delay': self.rate_limit_delay,
            'connector_info': self.connector.get_connection_info()
        }
