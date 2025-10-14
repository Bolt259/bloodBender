# bloodBath System Design Specification v2.0

**Document Version**: 2.0  
**Last Updated**: 2025-10-13  
**Status**: ACTIVE - Single Source of Truth  
**Purpose**: Comprehensive technical design specification for bloodBath data processing system

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Constants](#design-constants)
3. [Data Schemas](#data-schemas)
4. [Processing Pipeline](#processing-pipeline)
5. [Validation Rules](#validation-rules)
6. [File Organization](#file-organization)
7. [Integration Points](#integration-points)
8. [Error Handling](#error-handling)
9. [Change Log](#change-log)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        bloodBath System                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  API Layer (tconnectsync)                                        │
│  - TConnectApi: Authentication & event fetching                 │
│  - RawEvent parsing: TANDEM_EPOCH + timestampRaw                │
│  - Timezone handling: UTC storage, configurable display         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Data Extraction (bloodBath/data/extractors.py)                 │
│  - Extract CGM readings (bg values)                             │
│  - Extract basal rates (units/hour)                             │
│  - Extract bolus doses (units)                                  │
│  - Timestamp validation (reject < MIN_VALID_DATE)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  5-Minute Resampling (bloodbank_download.py)                    │
│  - Resample CGM: ffill within reasonable gaps                   │
│  - Resample Basal: ffill (continuous delivery)                  │
│  - Resample Bolus: sum (discrete events)                        │
│  - Handle missing data: NaN (NOT 100)                           │
│  - Add bg_missing_flag for NaN bins                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Data Processing (bloodBath/data/processors.py)                 │
│  - Clip BG to [BG_MIN, BG_MAX]                                  │
│  - Add bg_clip_flag for outliers                                │
│  - Calculate delta_bg (first diff)                              │
│  - Add temporal features (sin_time, cos_time)                   │
│  - Gap-aware imputation (≤ MAX_IMPUTE_MINUTES)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Validation (bloodBath/data/validators.py)                      │
│  - Timestamp continuity                                         │
│  - BG range compliance                                          │
│  - Synthetic pattern detection                                  │
│  - Quality scoring                                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CSV Writing (bloodBath/io/csv_writer.py)                       │
│  - v2.0 header with metadata                                    │
│  - Standardized naming                                          │
│  - UTC timestamps                                               │
│  - Clipping & missing data stats                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Storage (bloodBath/bloodBank/)                                 │
│  - raw/: Individual pump CSV files                              │
│  - merged/: Combined datasets                                   │
│  - archives/: Historical backups                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LSTM Preparation                                               │
│  - Sequence segmentation at gaps > MAX_GAP_HOURS                │
│  - Normalization (per-patient z-score)                          │
│  - Train/validate/test splits (70/15/15)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Dependencies

```
bloodBath/
├── api/
│   ├── connector.py          # API authentication
│   └── fetcher.py            # Event retrieval
├── core/
│   ├── client.py             # Main orchestration
│   └── config.py             # **DESIGN CONSTANTS** ⭐
├── data/
│   ├── extractors.py         # Event parsing
│   ├── processors.py         # Resampling, imputation
│   ├── validators.py         # Quality checks
│   └── repair.py             # Synthetic detection
├── io/
│   ├── csv_writer.py         # Standardized output
│   └── csv_reader.py         # Metadata parsing
└── utils/
    ├── time_utils.py         # Timestamp handling
    └── logging_utils.py      # Logging setup
```

---

## 2. Design Constants

### 2.1 Core Constants (SINGLE SOURCE OF TRUTH)

**File**: `bloodBath/core/config.py`

```python
# ========================================================================
# bloodBath Design Constants v2.0
# ========================================================================

# Blood Glucose Limits (mg/dL)
BG_MIN = 20    # Minimum physiologically plausible BG
BG_MAX = 600   # Maximum physiologically plausible BG
# Rationale: Covers hypoglycemia (20-40) and severe hyperglycemia (>400)
# Values outside this range are clipped and flagged

# Temporal Parameters
RESAMPLE_FREQ = '5min'          # Standard resampling frequency
MAX_GAP_HOURS = 15.0            # Maximum gap before sequence break
MAX_IMPUTE_MINUTES = 60         # Maximum imputation window (12 bins)
MIN_SEQUENCE_LENGTH = 12        # Minimum viable LSTM sequence (1 hour)

# Tandem Pump Constants
TANDEM_EPOCH = 1199145600       # 2008-01-01 00:00:00 UTC
MIN_VALID_DATE = '2020-01-01'   # Reject timestamps before this date

# Pump-Specific Valid Date Ranges
PUMP_881235_START = '2021-10-22T20:35:00+00:00'   # First valid data point
PUMP_881235_END = '2024-10-06T23:30:00+00:00'     # Last valid data point (converted to UTC from -07:00)
PUMP_901161470_START = '2024-10-07T00:30:00+00:00'  # First valid data point (converted to UTC from -07:00)
PUMP_901161470_END = '2025-10-11T16:35:00+00:00'  # Last valid data point (converted to UTC from -07:00)
# Note: Pump 881235 ends when pump 901161470 begins (patient switched pumps)

# Data Quality Thresholds
MAX_BG_MISSING_PERCENT = 30.0   # Max % missing BG before flagging
MAX_SYNTHETIC_PERCENT = 10.0    # Max % synthetic 100s before flagging
MIN_VALID_DAYS = 7              # Minimum days of data per file

# CSV Format
CSV_VERSION = 'v2.0'
CSV_COMMENT_PREFIX = '#'
TIMESTAMP_FORMAT = 'ISO8601'    # YYYY-MM-DDTHH:MM:SS+00:00

# Normalization
NORMALIZATION_METHOD = 'per_patient_zscore'  # Alternative: 'global_zscore'
CLIP_ZSCORE_RANGE = (-5, 5)                  # Clip normalized values

# Train/Validate/Test Split
TRAIN_RATIO = 0.70
VALIDATE_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_METHOD = 'chronological'  # Keep temporal ordering
```

### 2.2 Derived Constants

```python
# Calculated from core constants
BINS_PER_HOUR = 60 // 5                    # 12 bins
MAX_IMPUTE_BINS = MAX_IMPUTE_MINUTES // 5  # 12 bins
MAX_GAP_BINS = int(MAX_GAP_HOURS * BINS_PER_HOUR)  # 180 bins
```

---

## 3. Data Schemas

### 3.1 Raw API Event (from tconnectsync)

```python
class RawEvent:
    """Event from Tandem pump via tconnectsync API"""

    # Required fields
    timestamp: Arrow              # Timezone-aware datetime
    eventId: int                  # Event type identifier
    seqNum: int                   # Sequence number

    # Optional fields (depends on event type)
    currentglucosedisplayvalue: float  # CGM reading (mg/dL)
    commandedRate: int            # Basal rate (hundredths of units/hr)
    bolusAmount: int              # Bolus dose (hundredths of units)
```

### 3.2 CSV v2.0 Format

**Header Block** (lines starting with `#`):

```
# bloodBath v2.0 CSV Data File
# ==============================
# data_version: v2
# generated_at_utc: 2025-10-13T19:15:30+00:00
# pump_serial: 881235
# file_role: merged_lstm
# date_range: 2024-01-01 to 2024-01-31
# tz_handling: stored_in=UTC
# record_count: 8935
# columns: timestamp, bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time, bg_clip_flag, bg_missing_flag
# bg_range_policy: clipped to [20, 600] mg/dL
# bg_clipped_count: 15
# bg_missing_count: 42
# notes:
#   - Missing BG bins stored as NaN (not 100)
#   - bg_clip_flag=1 indicates value clipped to range
#   - bg_missing_flag=1 indicates no CGM reading available
#   - 5-minute resampling applied
# ==============================
```

**Data Columns**:

```csv
timestamp,bg,delta_bg,basal_rate,bolus_dose,sin_time,cos_time,bg_clip_flag,bg_missing_flag
2024-01-01T08:00:00+00:00,145.0,2.0,0.75,0.0,0.342,0.940,0,0
2024-01-01T08:05:00+00:00,NaN,NaN,0.75,0.0,0.364,0.931,0,1
2024-01-01T08:10:00+00:00,142.0,-3.0,0.75,0.0,0.386,0.922,0,0
```

### 3.3 LSTM Sequence Format

```python
class LSTMSequence:
    """Single training sequence for LSTM model"""

    # Temporal data (T timesteps x F features)
    timestamps: np.ndarray      # (T,) datetime64[ns]
    bg: np.ndarray              # (T,) float32, normalized
    delta_bg: np.ndarray        # (T,) float32, normalized
    basal_rate: np.ndarray      # (T,) float32, normalized
    bolus_dose: np.ndarray      # (T,) float32, normalized
    sin_time: np.ndarray        # (T,) float32, [-1, 1]
    cos_time: np.ndarray        # (T,) float32, [-1, 1]

    # Flags (not normalized, for masking/loss)
    bg_clip_flag: np.ndarray    # (T,) int, {0, 1}
    bg_missing_flag: np.ndarray # (T,) int, {0, 1}

    # Metadata
    pump_serial: str
    sequence_id: int
    start_time: datetime
    end_time: datetime
    duration_hours: float
    gap_count: int
```

---

## 4. Processing Pipeline

### 4.1 Data Flow Steps

**Step 1: API Fetch**

```python
def fetch_pump_events(pump_serial, start_date, end_date):
    """
    Fetch raw events from Tandem API

    Input: date range
    Output: List[RawEvent]

    Processing:
    1. Authenticate with TConnectApi
    2. Request events in 30-day chunks
    3. Parse timestamps: TANDEM_EPOCH + timestampRaw (seconds)
    4. Validate: timestamp >= MIN_VALID_DATE
    5. Convert to UTC timezone-aware
    """
```

**Step 2: Extract & Separate Streams**

```python
def extract_event_streams(events):
    """
    Separate events by type

    Input: List[RawEvent]
    Output: (cgm_data, basal_data, bolus_data)

    Processing:
    - CGM: Extract bg from currentglucosedisplayvalue
    - Basal: Extract rate from commandedRate / 100.0
    - Bolus: Extract dose from bolusAmount / 100.0
    - Create DataFrames with 'timestamp' and 'value' columns
    """
```

**Step 3: 5-Minute Resampling**

```python
def resample_to_5min(cgm_df, basal_df, bolus_df):
    """
    Resample all streams to 5-minute bins

    CGM Strategy:
    - resample('5min').ffill() within reasonable gaps
    - Leave NaN for gaps > MAX_IMPUTE_MINUTES
    - DO NOT fill with 100

    Basal Strategy:
    - resample('5min').ffill()  # Continuous delivery
    - Fill missing with 0.0 (pump suspended)

    Bolus Strategy:
    - resample('5min').sum()    # Discrete events
    - Fill missing with 0.0 (no bolus)

    Missing Data Handling:
    - If ALL three streams are empty: Return empty DataFrame
    - If only BG is missing: Use basal or bolus time index, keep BG as NaN
    - If only basal/bolus missing: Use BG time index, fill basal/bolus with 0.0
    - Preserves pump treatment data even when CGM is disconnected/off
    - Ensures complete treatment history for insulin dosing analysis

    Output: Aligned time series with common 5-min index
    """
```

**Step 4: Merge & Feature Engineering**

```python
def create_lstm_features(bg_series, basal_series, bolus_series):
    """
    Create LSTM-ready feature DataFrame

    Features:
    1. bg: From CGM, may contain NaN
    2. delta_bg: bg.diff(), NaN where bg is NaN
    3. basal_rate: From basal_series, fillna(0)
    4. bolus_dose: From bolus_series, fillna(0)
    5. sin_time: sin(2π * hour_of_day / 24)
    6. cos_time: cos(2π * hour_of_day / 24)
    7. bg_clip_flag: 1 if bg was clipped to [BG_MIN, BG_MAX]
    8. bg_missing_flag: 1 if bg is NaN
    """
```

**Step 5: Apply BG Policy**

```python
def apply_bg_policy(df):
    """
    Enforce BG range and flagging

    For each row where bg is not NaN:
    1. Store original_bg = bg
    2. Clip: bg = clip(bg, BG_MIN, BG_MAX)
    3. Flag: bg_clip_flag = (original_bg < BG_MIN OR original_bg > BG_MAX) ? 1 : 0

    For each row where bg is NaN:
    1. bg_missing_flag = 1
    2. bg_clip_flag = 0 (no clipping on missing data)
    """
```

**Step 6: Gap Detection & Sequence Segmentation**

```python
def segment_sequences(df):
    """
    Break data into sequences at large gaps

    Algorithm:
    1. Calculate time diffs: df['timestamp'].diff()
    2. Find gaps: time_diff > MAX_GAP_HOURS
    3. Split DataFrame at gap boundaries
    4. Filter: Keep only sequences with len >= MIN_SEQUENCE_LENGTH

    Output: List[DataFrame], one per sequence
    """
```

**Step 7: Normalization**

```python
def normalize_sequence(df, method='per_patient_zscore'):
    """
    Normalize features for LSTM training

    Per-patient Z-score (default):
    - Calculate μ, σ for each feature across ALL patient data
    - normalized = (value - μ) / σ
    - Clip to CLIP_ZSCORE_RANGE to handle outliers

    Features to normalize:
    - bg
    - delta_bg
    - basal_rate
    - bolus_dose

    Features NOT normalized:
    - sin_time, cos_time (already [-1, 1])
    - bg_clip_flag, bg_missing_flag (binary flags)
    """
```

---

## 5. Validation Rules

### 5.1 Per-File Validation

**Timestamp Validation**:

- ✅ All timestamps >= MIN_VALID_DATE
- ✅ No 1970 dates
- ✅ Monotonically increasing
- ✅ Consistent 5-minute spacing (with acceptable gaps)

**BG Validation**:

- ✅ All non-NaN values in [BG_MIN, BG_MAX]
- ✅ Missing BG < MAX_BG_MISSING_PERCENT
- ✅ No exact 100.0 values (legacy synthetic indicator)
- ✅ bg_clip_flag matches actual clipping
- ✅ bg_missing_flag matches NaN locations

**Data Quality**:

- ✅ At least MIN_VALID_DAYS of data
- ✅ CGM coverage > 70%
- ✅ No duplicate timestamps
- ✅ Basal rates in [0, 8] units/hr
- ✅ Bolus doses in [0, 25] units

### 5.2 Synthetic Pattern Detection

**100-Fill Detection** (should NOT occur):

```python
def detect_synthetic_100s(df):
    """
    Flag files with suspicious 100.0 patterns

    Indicators:
    - More than 5% exact 100.0 values
    - Long runs of consecutive 100.0 (>10 bins)
    - 100.0 values during pump suspension (basal=0)

    Action: REJECT file, regenerate with corrected pipeline
    """
```

### 5.3 Edge Cases & Special Scenarios

**Scenario 1: CGM Disconnected, Pump Active**

- Situation: Patient has basal/bolus data but no BG readings
- Handling: Keep all basal/bolus data, set bg=NaN, bg_missing_flag=1
- Rationale: Insulin treatment history is valuable for model training
- Example: Overnight CGM sensor expiration during active insulin delivery

**Scenario 2: Pump Suspended, CGM Active**

- Situation: Patient has BG readings but basal_rate=0, no boluses
- Handling: Keep all BG data, set basal=0, bolus=0
- Rationale: Natural pump-off periods during exercise, showers, etc.
- Example: Patient removes pump for swimming

**Scenario 3: All Data Streams Empty**

- Situation: No BG, basal, or bolus data in time range
- Handling: Return empty DataFrame, skip file generation
- Rationale: No actionable data to train on
- Example: Pump not in use during date range

**Scenario 4: Partial Data Coverage**

- Situation: Some 5-min bins have data, others completely empty
- Handling: Generate file with NaN/0 values, rely on MAX_GAP_HOURS segmentation
- Rationale: Preserve all available data, let sequence segmentation handle large gaps
- Example: Patient hospitalization with intermittent pump use

---

## 6. File Organization

### 6.1 Directory Structure

```
bloodBath/
├── bloodBank/              # Data storage (v2.0)
│   ├── raw/               # Individual pump monthly files
│   │   ├── pump_881235/
│   │   │   ├── 2021-01-31_to_2021-03-02.csv
│   │   │   ├── 2021-03-02_to_2021-04-01.csv
│   │   │   └── ...
│   │   └── pump_901161470/
│   │       ├── 2024-01-01_to_2024-01-31.csv
│   │       └── ...
│   ├── merged/            # Combined multi-pump datasets
│   │   ├── combined_2021_2024.csv
│   │   └── metadata.json
│   └── archives/          # Historical backups
│       └── 2025-10-13_pre_bg_fix/
├── test_scripts/          # ⭐ ALL test scripts MUST go here
│   ├── test_v2_integration.py
│   ├── archive_and_cleanup.py
│   └── ...
├── spec/                  # Design specifications and documentation
├── logs/                  # Processing logs
├── validation/            # Validation reports
└── config/               # Configuration files
```

### 6.2 Naming Conventions

**Raw CSV Files**:

```
Format: pump_{serial}_{YYYY-MM-DD}_to_{YYYY-MM-DD}.csv
Example: pump_881235_2024-01-01_to_2024-01-31.csv
```

**Combined Files**:

```
Format: combined_{start_YYYY}_{end_YYYY}.csv
Example: combined_2021_2024.csv
```

**Archive Directories**:

```
Format: {YYYY-MM-DD}_{description}/
Example: 2025-10-13_pre_bg_fix/
```

---

## 7. Integration Points

### 7.1 tconnectsync API

**Dependency**: `tconnectsync` Python package  
**Key Classes**:

- `TConnectApi`: Authentication and event fetching
- `RawEvent`: Event parsing with TANDEM_EPOCH handling

**Integration Rules**:

- ALWAYS use `event.timestamp.datetime` (already parsed)
- NEVER reparse `event.raw.timestampRaw` with pandas
- Respect API rate limits (30-second backoff on 429)
- Cache credentials in `~/.config/tconnectsync/`

### 7.2 LSTM Model Expectations

**Input Shape**: `(batch, timesteps, features)`

- timesteps: Variable length sequences
- features: 8 (bg, delta_bg, basal, bolus, sin_time, cos_time, clip_flag, missing_flag)

**Normalization**: Per-patient z-score expected
**Masking**: Use `bg_missing_flag` for attention masking

---

## 8. Error Handling

### 8.1 Missing Data Policy

**BG Missing**:

- Store as `NaN` (NOT 100)
- Set `bg_missing_flag = 1`
- Allow forward-fill up to MAX_IMPUTE_MINUTES
- Beyond that, leave as NaN

**Basal Missing**:

- Default to 0.0 (pump suspended)
- Log warning if gap > 4 hours

**Bolus Missing**:

- Default to 0.0 (no bolus given)
- No warnings (normal behavior)

### 8.2 Outlier Handling

**BG Outliers**:

- Clip to [BG_MIN, BG_MAX]
- Set `bg_clip_flag = 1`
- Log count and percentage

**Basal Outliers**:

- Warn if > 8.0 units/hr
- Clip to [0, 8] for safety

**Bolus Outliers**:

- Warn if > 25 units
- Log but don't clip (may be valid)

---

## 9. Change Log

### v2.0 (2025-10-13)

- **Updated BG range**: 40-400 → 20-600 mg/dL
- **Fixed BG fill bug**: 100 placeholder → NaN
- **Added columns**: `bg_clip_flag`, `bg_missing_flag`
- **Fixed timestamp bug**: Corrected 1970 date issue
- **Implemented CSV v2.0**: Standardized headers with metadata

### v1.0 (2024-10-XX)

- Initial specification
- Basic pipeline: fetch → resample → merge
- CSV v1.0 format (no headers)

---

## 10. Usage Guidelines

### 10.1 For Developers

**When adding new features**:

1. Check constants in `config.py` first
2. Follow schema definitions exactly
3. Add validation tests
4. Update this spec document

**When debugging**:

1. Check if constants match across modules
2. Verify CSV headers have correct metadata
3. Look for 100-fill patterns (should not exist)
4. Confirm timestamps are UTC

### 10.2 For Data Scientists

**When training models**:

1. Use `bg_missing_flag` for masking
2. Expect NaN in BG column (handle appropriately)
3. Normalization stats saved in metadata
4. Train/validate/test splits are chronological

**When validating results**:

1. Check for 1970 dates (should be zero)
2. Verify BG in [20, 600] range
3. Confirm no 100-fill artifacts
4. Validate 5-minute spacing

---

## Appendix A: Quick Reference

**File Paths**:

- Constants: `bloodBath/core/config.py`
- Processing: `bloodBath/data/processors.py`
- Validation: `bloodBath/data/validators.py`
- CSV I/O: `bloodBath/io/`
- **Test Scripts**: `bloodBath/test_scripts/` ⭐ (MANDATORY LOCATION FOR ALL TESTS)

**Key Constants**:

- `BG_MIN = 20`
- `BG_MAX = 600`
- `RESAMPLE_FREQ = '5min'`
- `MAX_GAP_HOURS = 15.0`

**Data Locations**:

- Raw: `bloodBath/bloodBank/raw/pump_{serial}/`
- Merged: `bloodBath/bloodBank/merged/`
- Logs: `bloodBath/logs/`
- **Tests**: `bloodBath/test_scripts/` ⭐ (ALL TEST SCRIPTS MUST GO HERE)

---

**END OF SPECIFICATION**
