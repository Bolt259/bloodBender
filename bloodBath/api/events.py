"""
Event parsing utilities for Tandem pump events
"""

import struct
import base64
import logging
from dataclasses import dataclass
from typing import Iterator, List, Any, Optional
from itertools import islice

logger = logging.getLogger(__name__)

# Event length in bytes
EVENT_LEN = 64

@dataclass
class RawEvent:
    """Raw event data structure"""
    id: int
    timestamp: int
    data: bytes
    
    @classmethod
    def build(cls, data: bytes) -> 'RawEvent':
        """Build a raw event from bytes"""
        if len(data) < EVENT_LEN:
            data = data + b'\x00' * (EVENT_LEN - len(data))
        
        # Extract event ID and timestamp from the first 8 bytes
        # This is a simplified version - actual parsing may be more complex
        try:
            event_id = struct.unpack('<I', data[0:4])[0]
            timestamp = struct.unpack('<I', data[4:8])[0]
            return cls(id=event_id, timestamp=timestamp, data=data)
        except struct.error:
            logger.warning(f"Error parsing raw event data: {data.hex()}")
            return cls(id=0, timestamp=0, data=data)

def decode_raw_events(raw_data: str) -> bytes:
    """Decode base64 encoded raw events"""
    try:
        return base64.b64decode(raw_data)
    except Exception as e:
        logger.error(f"Error decoding raw events: {e}")
        return b''

def batched(iterable, n):
    """Batch an iterable into chunks of size n"""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            break
        yield batch

def Event(data: bytes) -> RawEvent:
    """Create an event from raw bytes"""
    return RawEvent.build(data)

def Events(data: bytes) -> Iterator[RawEvent]:
    """Create an iterator of events from raw bytes"""
    for i in range(0, len(data), EVENT_LEN):
        chunk = data[i:i + EVENT_LEN]
        if len(chunk) == EVENT_LEN:
            yield Event(chunk)

# Event type mappings - simplified version
# In the full implementation, these would map to specific event classes
EVENT_TYPES = {
    1: 'ALARM',
    2: 'ALERT',
    3: 'BASAL_RATE_CHANGE',
    4: 'BOLUS_DELIVERY',
    5: 'CGM_READING',
    6: 'REMINDER',
    # Add more as needed
}

def get_event_type(event_id: int) -> str:
    """Get event type name from ID"""
    return EVENT_TYPES.get(event_id, 'UNKNOWN')

# Default event IDs used by Tandem Source
DEFAULT_EVENT_IDS = [
    229, 5, 28, 4, 26, 99, 279, 3, 16, 59, 21, 55, 20, 280, 64, 65, 66, 61, 33, 371, 
    171, 369, 460, 172, 370, 461, 372, 399, 256, 213, 406, 394, 212, 404, 214, 405, 
    447, 313, 60, 14, 6, 90, 230, 140, 12, 11, 53, 13, 63, 203, 307, 191
]
