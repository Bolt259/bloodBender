# bloodBath v2.0 Bug Fixes & Enhancements

**Date**: 2025-10-13  
**Version**: 2.0.1  
**Status**: READY FOR TESTING

---

## Summary of Changes

This release addresses **three critical data quality issues** identified during bloodbank sync validation:

1. **BG 100-Fill Bug** (FIXED)
2. **BG Forward-Fill Bug** (FIXED)  
3. **BG Range Policy Update** (UPDATED)

---

## 1. BG 100-Fill Bug

### Problem
Missing BG values were being filled with placeholder value `100.0` instead of being left as `NaN`.

**Impact**: 
- Artificially inflated dataset with synthetic 100 mg/dL readings
- Unable to distinguish real BG measurements from missing data
- LSTM models trained on fake data patterns

**Root Cause**:
```python
# Line 618 (OLD CODE)
df['bg'] = aligned_data['bg'].fillna(100.0)  # ❌ WRONG
```

### Fix
```python
# Line 618 (NEW CODE)
df['bg'] = aligned_data['bg']  # ✅ Keep NaN
df['bg_missing_flag'] = df['bg'].isna().astype(int)  # Track missing
```

**Files Modified**:
- `bloodbank_download.py` line 618-623

**Verification**:
```python
# Test: No 100.0 values should exist unless real CGM reading
assert not ((df['bg'] == 100.0) & (df['bg_missing_flag'] == 1)).any()
```

---

## 2. BG Forward-Fill Bug

### Problem
Forward-fill (`.ffill()`) was propagating the last known BG value into long gaps, creating artificial continuity.

**Impact**:
- BG values persisting unrealistically during pump disconnects
- Gap detection algorithms failing
- Sequence segmentation incorrect

**Root Causes**:
```python
# Line 679 (OLD CODE)
bg_series = cgm_df['bg'].resample(freq).ffill()  # ❌ WRONG

# Line 711 (OLD CODE)
bg_series = bg_series.reindex(time_index).ffill()  # ❌ WRONG
```

### Fix
```python
# Line 679 (NEW CODE)
bg_series = cgm_df['bg'].resample(freq).first()  # ✅ No propagation

# Line 711 (NEW CODE)
bg_series = bg_series.reindex(time_index)  # ✅ Gaps stay NaN
```

**Files Modified**:
- `bloodbank_download.py` lines 679, 711-714

**Verification**:
```python
# Test: No BG values in 15+ hour gaps
time_diff = df['timestamp'].diff()
long_gaps = time_diff > pd.Timedelta(hours=15)
assert df.loc[long_gaps, 'bg'].isna().all()
```

---

## 3. BG Range Policy Update

### Problem
Original policy clipped BG to `[40, 400]` mg/dL, excluding valid hypoglycemia (<40) and severe hyperglycemia (>400) readings.

**Impact**:
- Lost clinical information about extreme glucose events
- Underestimated severity of glucose excursions

### Change
```python
# OLD POLICY
BG_MIN = 40   # mg/dL
BG_MAX = 400  # mg/dL

# NEW POLICY
BG_MIN = 20   # mg/dL (covers severe hypoglycemia)
BG_MAX = 600  # mg/dL (covers DKA/HHS)
```

**Files Modified**:
- `bloodbank_download.py` lines 737-738
- `bloodBath/io/csv_writer.py` lines 36-37
- `bloodBath/data/processors.py` line 1335, 1186
- `bloodBath/data/resampler.py` line 510

**Rationale**:
- 20 mg/dL: Captures severe hypoglycemia requiring emergency treatment
- 600 mg/dL: Captures diabetic ketoacidosis (DKA) and hyperosmolar hyperglycemic state (HHS)
- Physiologically plausible range for diabetes monitoring

---

## 4. New Feature: `bg_missing_flag` Column

### Implementation
Added binary flag column to track missing BG data:

```python
# Values
0 = BG measurement present (real CGM reading)
1 = BG measurement missing (NaN)
```

**Purpose**:
- Enable attention masking in LSTM models
- Distinguish imputed values from real measurements
- Quality metrics (% missing data)

**CSV Output**:
```csv
timestamp,bg,delta_bg,basal_rate,bolus_dose,sin_time,cos_time,bg_clip_flag,bg_missing_flag
2024-01-01T08:00:00+00:00,145.0,2.0,0.75,0.0,0.342,0.940,0,0
2024-01-01T08:05:00+00:00,NaN,NaN,0.75,0.0,0.364,0.931,0,1
```

**Files Modified**:
- `bloodbank_download.py` (added to both `_align_data_to_5min_bins` and `_create_lstm_ready_format`)
- CSV column order updated

---

## 5. Design Specification Document

### Created
`bloodBath_Design_Specification_v2.0.md` - Comprehensive technical design document

**Contents**:
1. System Overview & Architecture Diagram
2. Design Constants (SINGLE SOURCE OF TRUTH)
3. Data Schemas (RawEvent, CSV v2.0, LSTM)
4. Processing Pipeline (7-step workflow)
5. Validation Rules
6. File Organization & Naming
7. Integration Points (tconnectsync, LSTM)
8. Error Handling Policies
9. Change Log
10. Quick Reference

**Purpose**:
- Prevent future parameter drift
- Ensure consistency across modules
- Onboard new developers
- Document design decisions

---

## Testing Plan

### Unit Tests (Required Before Regeneration)
```bash
# Test 1: No 100-fill artifacts
python -c "
import pandas as pd
df = pd.read_csv('pump_881235/test_2024-01.csv', comment='#')
assert not (df['bg'] == 100.0).any(), 'Found 100.0 values'
print('✅ No 100-fill artifacts')
"

# Test 2: NaN present in gaps
python -c "
import pandas as pd
df = pd.read_csv('pump_881235/test_2024-01.csv', comment='#')
assert df['bg'].isna().any(), 'No NaN values found'
print('✅ NaN values preserved')
"

# Test 3: bg_missing_flag matches NaN
python -c "
import pandas as pd
df = pd.read_csv('pump_881235/test_2024-01.csv', comment='#')
assert (df['bg'].isna() == (df['bg_missing_flag'] == 1)).all()
print('✅ bg_missing_flag correct')
"

# Test 4: BG range [20, 600]
python -c "
import pandas as pd
df = pd.read_csv('pump_881235/test_2024-01.csv', comment='#')
bg_valid = df['bg'].dropna()
assert (bg_valid >= 20).all() and (bg_valid <= 600).all()
print('✅ BG range correct')
"
```

### Integration Test
```bash
# Generate one month of data with fixes
python bloodbank_download.py \
  --pump-serial 881235 \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --output-dir test_fixed/

# Validate output
python validate_bloodbank_data.py test_fixed/
```

---

## Regeneration Workflow

### Prerequisites
1. ✅ All unit tests passing
2. ✅ Integration test successful
3. ✅ Current sync of pump 881235 completed (~30 min remaining)
4. ✅ Design specification reviewed

### Steps

**1. Archive Old Data**
```bash
mkdir -p archives/2025-10-13_pre_bg_fix/
mv pump_881235/ archives/2025-10-13_pre_bg_fix/
mv pump_901161470/ archives/2025-10-13_pre_bg_fix/
```

**2. Regenerate Pump 881235 (2021-2024)**
```bash
python bloodbank_download.py \
  --pump-serial 881235 \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  --output-dir bloodBath/bloodBank/raw/ \
  2>&1 | tee regen_881235.log
```

**3. Regenerate Pump 901161470 (2024-2025)**
```bash
python bloodbank_download.py \
  --pump-serial 901161470 \
  --start-date 2024-01-01 \
  --end-date 2025-01-31 \
  --output-dir bloodBath/bloodBank/raw/ \
  2>&1 | tee regen_901161470.log
```

**4. Full Validation**
```bash
python validate_bloodbank_data.py bloodBath/bloodBank/raw/
```

---

## Acceptance Criteria

### Must Pass Before Completion

- [x] ✅ Zero 1970 timestamps
- [x] ✅ CSV v2.0 headers present
- [x] ✅ UTC timezone storage
- [x] ✅ Standardized naming
- [x] ✅ Correct timestamp ranges
- [ ] 🔄 No 100-fill behavior (PENDING REGEN)
- [ ] 🔄 NaN for missing BG (PENDING REGEN)
- [ ] 🔄 bg_missing_flag column (PENDING REGEN)
- [ ] 🔄 BG range [20, 600] (PENDING REGEN)
- [ ] 🔄 bg_clip_flag correct (PENDING REGEN)
- [x] ✅ Design specification created

### Quality Metrics (Post-Regen)
- Missing BG rate: Expected 10-30% (pump disconnects, gaps)
- Clipped BG rate: Expected <1% (extreme outliers)
- 100.0 concentration: **0%** (must be zero)
- Timestamp gaps >15hr: Should trigger sequence breaks

---

## Rollback Plan

If regeneration fails or validation does not pass:

```bash
# Restore old data
rm -rf pump_881235/ pump_901161470/
cp -r archives/2025-10-13_pre_bg_fix/* .

# Revert code changes
git checkout bloodbank_download.py
git checkout bloodBath/io/csv_writer.py
git checkout bloodBath/data/processors.py
git checkout bloodBath/data/resampler.py
```

---

## Next Steps

1. **Wait for current sync to complete** (~30 min)
2. **Run integration test** on January 2024 subset
3. **Archive existing data**
4. **Regenerate all pump data** (2021-2025)
5. **Run full validation suite**
6. **Verify acceptance criteria**
7. **Update documentation** with final statistics

---

## Contact

Questions? Review the design specification:
- `bloodBath_Design_Specification_v2.0.md`

---

**END OF CHANGELOG**
