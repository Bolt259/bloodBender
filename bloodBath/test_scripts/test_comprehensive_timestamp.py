#!/usr/bin/env python3
"""
Comprehensive test for timestamp extraction from various event types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.time_utils import extract_timestamp_from_event
from eventparser.raw_event import TANDEM_EPOCH
import pandas as pd
import arrow
from secret import TIMEZONE_NAME

print("=== Comprehensive Timestamp Extraction Test ===\n")

# Test 1: Event with timestamp property (Arrow object)
class MockEventWithTimestamp:
    def __init__(self, timestamp_raw):
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

print("Test 1: Event with Arrow timestamp property")
event1 = MockEventWithTimestamp(505000000)
ts1 = extract_timestamp_from_event(event1)
print(f"Result: {ts1}")
print(f"Valid: {ts1 is not None and 2008 <= ts1.year <= 2030}\n")

# Test 2: Event with raw object containing timestamp
class MockEventWithRawTimestamp:
    def __init__(self, timestamp_raw):
        self.raw = MockRawEvent(timestamp_raw)

class MockRawEvent:
    def __init__(self, timestamp_raw):
        self.timestampRaw = timestamp_raw
        self.timestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

print("Test 2: Event with raw.timestamp property")
event2 = MockEventWithRawTimestamp(505000000)
ts2 = extract_timestamp_from_event(event2)
print(f"Result: {ts2}")
print(f"Valid: {ts2 is not None and 2008 <= ts2.year <= 2030}\n")

# Test 3: Event with eventTimestamp property
class MockEventWithEventTimestamp:
    def __init__(self, timestamp_raw):
        self.eventTimestamp = arrow.get(TANDEM_EPOCH + timestamp_raw, tzinfo='UTC').replace(tzinfo=TIMEZONE_NAME)

print("Test 3: Event with eventTimestamp property")
event3 = MockEventWithEventTimestamp(505000000)
ts3 = extract_timestamp_from_event(event3)
print(f"Result: {ts3}")
print(f"Valid: {ts3 is not None and 2008 <= ts3.year <= 2030}\n")

# Test 4: Event with raw timestampRaw property (fallback manual conversion)
class MockEventWithRawTimestampRaw:
    def __init__(self, timestamp_raw):
        self.raw = MockRawEventWithTimestampRaw(timestamp_raw)

class MockRawEventWithTimestampRaw:
    def __init__(self, timestamp_raw):
        self.timestampRaw = timestamp_raw

print("Test 4: Event with raw.timestampRaw property (manual conversion)")
event4 = MockEventWithRawTimestampRaw(505000000)
ts4 = extract_timestamp_from_event(event4)
print(f"Result: {ts4}")
print(f"Valid: {ts4 is not None and 2008 <= ts4.year <= 2030}\n")

# Test 5: Event with invalid timestamp (should return None)
class MockEventWithInvalidTimestamp:
    def __init__(self):
        self.timestamp = arrow.get(946684800)  # Year 2000 (before 2008)

print("Test 5: Event with invalid timestamp (year 2000)")
event5 = MockEventWithInvalidTimestamp()
ts5 = extract_timestamp_from_event(event5)
print(f"Result: {ts5}")
print(f"Valid: {ts5 is None}\n")

# Test 6: Event with no timestamp (should return None)
class MockEventWithNoTimestamp:
    def __init__(self):
        pass

print("Test 6: Event with no timestamp")
event6 = MockEventWithNoTimestamp()
ts6 = extract_timestamp_from_event(event6)
print(f"Result: {ts6}")
print(f"Valid: {ts6 is None}\n")

print("=== All tests completed ===")
