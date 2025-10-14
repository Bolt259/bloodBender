#!/usr/bin/env python3
"""
Test script to verify timestamp extraction fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.time_utils import extract_timestamp_from_event
from eventparser.raw_event import TANDEM_EPOCH
import pandas as pd
import struct
import arrow
from secret import TIMEZONE_NAME

# Create a mock event with a timestampRaw value
class MockEvent:
    def __init__(self, timestamp_raw):
        self.raw = MockRawEvent(timestamp_raw)
        
class MockRawEvent:
    def __init__(self, timestamp_raw):
        self.timestampRaw = timestamp_raw
        
    @property
    def timestamp(self):
        import arrow
        from secret import TIMEZONE_NAME
        return arrow.get(TANDEM_EPOCH + self.timestampRaw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

# Test with a reasonable timestamp value
# Let's use a timestamp from 2024 (16 years after Tandem epoch)
# 16 years * 365.25 days/year * 24 hours/day * 3600 seconds/hour ≈ 505,000,000 seconds
test_timestamp_raw = 505000000

print(f"TANDEM_EPOCH: {TANDEM_EPOCH}")
print(f"Test timestampRaw: {test_timestamp_raw}")
print(f"Unix timestamp: {TANDEM_EPOCH + test_timestamp_raw}")

# Create mock event
mock_event = MockEvent(test_timestamp_raw)

# Test extraction
extracted_timestamp = extract_timestamp_from_event(mock_event)
print(f"Extracted timestamp: {extracted_timestamp}")

# Verify it's reasonable
if extracted_timestamp:
    print(f"Year: {extracted_timestamp.year}")
    print(f"Date: {extracted_timestamp.date()}")
    print(f"Is reasonable: {2008 <= extracted_timestamp.year <= 2030}")
else:
    print("Failed to extract timestamp")
