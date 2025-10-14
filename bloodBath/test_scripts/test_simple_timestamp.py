#!/usr/bin/env python3
"""
Simple test to verify the timestamp extraction fix works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules directly to avoid package import issues
import pandas as pd
import arrow
from secret import TIMEZONE_NAME
from eventparser.raw_event import TANDEM_EPOCH

# Import the time utils function directly
from utils.time_utils import extract_timestamp_from_event

print("=== Simple Timestamp Extraction Test ===\n")

# Mock event classes
class MockCgmEvent:
    def __init__(self, timestamp_raw, bg_value):
        self.currentglucosedisplayvalue = bg_value
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

class MockBasalEvent:
    def __init__(self, timestamp_raw, basal_rate):
        self.commandedRate = basal_rate * 100  # Raw API returns in hundredths
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

# Create test events with reasonable timestamps (2024)
test_timestamp_raw = 505000000  # Jan 1, 2024

cgm_event = MockCgmEvent(test_timestamp_raw, 120.0)
basal_event = MockBasalEvent(test_timestamp_raw + 300, 0.8)

print("Testing timestamp extraction...")
cgm_timestamp = extract_timestamp_from_event(cgm_event)
basal_timestamp = extract_timestamp_from_event(basal_event)

print(f"CGM event timestamp: {cgm_timestamp}")
print(f"Basal event timestamp: {basal_timestamp}")

# Validate timestamps
cgm_valid = cgm_timestamp is not None and 2008 <= cgm_timestamp.year <= 2030
basal_valid = basal_timestamp is not None and 2008 <= basal_timestamp.year <= 2030

print(f"CGM timestamp valid: {'✓' if cgm_valid else '✗'}")
print(f"Basal timestamp valid: {'✓' if basal_valid else '✗'}")

# Show the improvement - what old timestamps would have looked like
print("\nComparing with old broken timestamps:")
print(f"Old (Unix epoch): {pd.to_datetime(test_timestamp_raw, unit='s')}")
print(f"New (Tandem epoch): {cgm_timestamp}")

print("\n=== Test completed ===")
