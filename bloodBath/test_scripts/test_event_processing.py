#!/usr/bin/env python3
"""
Test script to verify the timestamp extraction fix works with event processing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.extractors import EventExtractor
from eventparser.raw_event import TANDEM_EPOCH
import pandas as pd
import arrow
from secret import TIMEZONE_NAME

print("=== Event Processing Timestamp Fix Test ===\n")

# Mock event classes that simulate real pump events
class MockCgmEvent:
    def __init__(self, timestamp_raw, bg_value):
        self.currentglucosedisplayvalue = bg_value
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

class MockBasalEvent:
    def __init__(self, timestamp_raw, basal_rate):
        self.commandedRate = basal_rate * 100  # Raw API returns in hundredths
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

class MockBolusEvent:
    def __init__(self, timestamp_raw, bolus_dose):
        self.bolusAmount = bolus_dose * 100  # Raw API returns in hundredths
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

# Create test events with reasonable timestamps (2024)
test_timestamp_raw = 505000000  # Jan 1, 2024

# Create mock events
cgm_events = [
    MockCgmEvent(test_timestamp_raw, 120.0),
    MockCgmEvent(test_timestamp_raw + 300, 115.0),  # 5 minutes later
    MockCgmEvent(test_timestamp_raw + 600, 110.0),  # 10 minutes later
]

basal_events = [
    MockBasalEvent(test_timestamp_raw, 0.8),
    MockBasalEvent(test_timestamp_raw + 1800, 1.0),  # 30 minutes later
]

bolus_events = [
    MockBolusEvent(test_timestamp_raw + 900, 3.5),  # 15 minutes later
]

# Test event extraction
extractor = EventExtractor()

print("Testing CGM event extraction...")
normalized_cgm = extractor.normalize_cgm_events(cgm_events)
print(f"Extracted {len(normalized_cgm)} CGM events")
for event in normalized_cgm:
    print(f"  - {event['timestamp']}: {event['bg']} mg/dL")

print("\nTesting basal event extraction...")
normalized_basal = extractor.normalize_basal_events(basal_events)
print(f"Extracted {len(normalized_basal)} basal events")
for event in normalized_basal:
    print(f"  - {event['timestamp']}: {event['basal_rate']} units/hour")

print("\nTesting bolus event extraction...")
normalized_bolus = extractor.normalize_bolus_events(bolus_events)
print(f"Extracted {len(normalized_bolus)} bolus events")
for event in normalized_bolus:
    print(f"  - {event['timestamp']}: {event['bolus_dose']} units")

# Check that all timestamps are reasonable
print("\nTimestamp validation:")
all_events = normalized_cgm + normalized_basal + normalized_bolus
for event in all_events:
    ts = event['timestamp']
    valid = 2008 <= ts.year <= 2030
    print(f"  - {ts}: {'✓' if valid else '✗'} (Year: {ts.year})")

print("\n=== Test completed ===")
