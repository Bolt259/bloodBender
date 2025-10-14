#!/usr/bin/env python3
"""
Pump Information Retrieval Utility

This utility analyzes the active usage periods for each pump on your Tandem account.
It shows when each pump was first used, last used, and the total duration of activity.
Adapted for bloodBath package.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import arrow
except ImportError:
    arrow = None

logger = logging.getLogger(__name__)


def analyze_pump_activity(api_client, region='US') -> Dict[str, Any]:
    """
    Analyze pump activity periods for all pumps on the account.
    
    Args:
        api_client: TandemHistoricalSyncClient instance
        region: API region (US or EU)
        
    Returns:
        Dictionary with pump serial numbers as keys and activity info as values.
    """
    logger.info(f"Connecting to Tandem Source API ({region} region)...")
    
    # Get pump metadata from the API
    try:
        api = api_client.connector.get_api()
        pump_metadata = api.tandemsource.pump_event_metadata()
    except Exception as e:
        logger.error(f"Failed to get pump metadata: {e}")
        return {}
    
    if not pump_metadata:
        logger.warning("No pumps found on your account.")
        return {}
    
    logger.info(f"Found {len(pump_metadata)} pump(s) on your account.")
    
    pumps_info = {}
    
    for pump in pump_metadata:
        serial = pump['serialNumber']
        model = pump.get('modelNumber', 'Unknown')
        min_date = pump.get('minDateWithEvents')
        max_date = pump.get('maxDateWithEvents')
        last_upload = pump.get('lastUpload')
        software_version = pump.get('softwareVersion', 'Unknown')
        
        # Parse dates
        try:
            start_date = _parse_date(min_date) if min_date else None
            end_date = _parse_date(max_date) if max_date else None
            
            # Handle lastUpload which can be a dict or string
            if isinstance(last_upload, dict) and 'lastUploadedAt' in last_upload:
                upload_date = _parse_date(last_upload['lastUploadedAt'])
            elif isinstance(last_upload, str):
                upload_date = _parse_date(last_upload)
            else:
                upload_date = None
                
        except Exception as e:
            logger.warning(f"Could not parse dates for pump {serial}: {e}")
            start_date = end_date = upload_date = None
        
        # Calculate activity duration
        duration_days = None
        if start_date and end_date:
            duration_days = (end_date - start_date).days + 1
        
        # Determine current status
        status = "Unknown"
        if end_date:
            days_since_last_event = (datetime.now() - end_date).days
            if days_since_last_event <= 1:
                status = "Active (current)"
            elif days_since_last_event <= 7:
                status = "Recently active"
            elif days_since_last_event <= 30:
                status = "Inactive (recent)"
            else:
                status = "Inactive (old)"
        
        pumps_info[serial] = {
            'model': model,
            'software_version': software_version,
            'first_event': start_date,
            'last_event': end_date,
            'last_upload': upload_date,
            'duration_days': duration_days,
            'status': status,
            'tconnect_device_id': pump.get('tconnectDeviceId'),
            'raw_pump_data': pump
        }
    
    return pumps_info


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if not date_str:
        return None
    
    if arrow:
        try:
            # Parse with arrow and convert to naive datetime
            return arrow.get(date_str).naive
        except Exception:
            pass
    
    # Fallback to standard datetime parsing
    try:
        # Try ISO format first
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            # Convert to naive datetime (remove timezone)
            return dt.replace(tzinfo=None)
        else:
            return datetime.fromisoformat(date_str)
    except Exception:
        pass
    
    # Try common formats
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def get_pump_active_date_range(api_client, pump_serial: str) -> Optional[Dict[str, Any]]:
    """
    Get the active date range for a specific pump.
    
    Args:
        api_client: TandemHistoricalSyncClient instance
        pump_serial: Pump serial number
        
    Returns:
        Dictionary with start_date, end_date, and other pump info, or None if not found
    """
    pumps_info = analyze_pump_activity(api_client)
    
    if pump_serial not in pumps_info:
        logger.warning(f"Pump {pump_serial} not found in account")
        return None
    
    pump_info = pumps_info[pump_serial]
    
    return {
        'pump_serial': pump_serial,
        'start_date': pump_info['first_event'],
        'end_date': pump_info['last_event'],
        'duration_days': pump_info['duration_days'],
        'status': pump_info['status'],
        'model': pump_info['model'],
        'software_version': pump_info['software_version']
    }


def print_pump_summary(pumps_info: Dict[str, Any]) -> None:
    """Print a formatted summary of pump activity."""
    
    print("=" * 80)
    print("📊 PUMP ACTIVITY SUMMARY")
    print("=" * 80)
    
    # Sort pumps by last event date (most recent first)
    sorted_pumps = sorted(
        pumps_info.items(), 
        key=lambda x: x[1]['last_event'] or datetime(1900, 1, 1), 
        reverse=True
    )
    
    for serial, info in sorted_pumps:
        print(f"\n🔧 Pump Serial: {serial}")
        print(f"   Model: {info['model']}")
        print(f"   Software: {info['software_version']}")
        print(f"   Status: {info['status']}")
        print(f"   tConnect Device ID: {info['tconnect_device_id']}")
        
        if info['first_event']:
            print(f"   📅 First Event: {info['first_event'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   📅 First Event: Unknown")
            
        if info['last_event']:
            print(f"   📅 Last Event:  {info['last_event'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   📅 Last Event:  Unknown")
            
        if info['last_upload']:
            print(f"   📤 Last Upload: {info['last_upload'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   📤 Last Upload: Unknown")
            
        if info['duration_days']:
            print(f"   ⏱️  Active Duration: {info['duration_days']} days ({info['duration_days']/365.25:.1f} years)")
        
        print(f"   {'-' * 60}")


def get_optimal_sync_range(api_client, pump_serial: str, max_days: Optional[int] = None) -> Optional[Dict[str, str]]:
    """
    Get the optimal date range for syncing a specific pump.
    
    Args:
        api_client: TandemHistoricalSyncClient instance
        pump_serial: Pump serial number
        max_days: Maximum number of days to sync (default: None for full range)
        
    Returns:
        Dictionary with start_date and end_date as strings, or None if not found
    """
    pump_info = get_pump_active_date_range(api_client, pump_serial)
    
    if not pump_info or not pump_info['start_date'] or not pump_info['end_date']:
        logger.warning(f"No active date range found for pump {pump_serial}")
        return None
    
    start_date = pump_info['start_date']
    end_date = pump_info['end_date']
    
    # Only limit the range if max_days is specified and the range is too long
    if max_days and pump_info['duration_days'] and pump_info['duration_days'] > max_days:
        # Use the most recent data
        start_date = end_date - timedelta(days=max_days)
        logger.info(f"Limiting sync range to {max_days} days for pump {pump_serial}")
    else:
        logger.info(f"Using full data range for pump {pump_serial}: {pump_info['duration_days']} days")
    
    return {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'duration_days': (end_date - start_date).days + 1,
        'status': pump_info['status']
    }
