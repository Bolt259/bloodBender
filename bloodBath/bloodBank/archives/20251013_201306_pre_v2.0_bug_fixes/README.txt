bloodBath Data Archive
==================================================

Archived: 2025-10-13 20:13:38
Reason: Pre-v2.0 bug fixes regeneration

Issues with this data:
  - BG 100-fill bug (missing data filled with 100.0)
  - Forward-fill bug (gaps filled with last value)
  - Missing bg_missing_flag column
  - Missing bg_clip_flag column
  - Old BG range [40, 400] instead of [20, 600]

v2.0 fixes applied in regeneration:
  ✅ NaN for missing BG (not 100)
  ✅ bg_missing_flag column added
  ✅ bg_clip_flag column added
  ✅ BG range updated to [20, 600] mg/dL
  ✅ No forward-fill into gaps

Archived contents:
  - pump_881235/
  - pump_901161470/
