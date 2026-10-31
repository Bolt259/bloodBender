# bloodBender Unified Ticket Tracker

Single source of truth across all subprojects: **bloodBath** (data sync), **bloodTwin** (ML),
**infra** (Nix/services), **bareMetalBender** (embedded future).

Status legend: `open` `in-progress` `blocked` `done` `deferred`
Priority: P0 (blocking) → P1 (high) → P2 (normal) → P3 (low/someday)

---

## Epics

Three cross-cutting initiatives. Tickets can belong to more than one.

### E1 — Data Integrity
*Make sure what goes into the model is correct and complete.*
BB-01, BB-02, BB-03, BT-02

### E2 — Model Quality
*Make the model better: accuracy, calibration, dose-response.*
BT-06, BT-07, and future retrain tickets. Also depends on E1 (garbage in → garbage out).

### E3 — Clinical UX
*Show predictions usefully; alert on risk before it happens.*
BT-04 (multi-horizon trajectory fan), IN-01 (hypoglycemia push notification)

---

## P0 — Blocking / Critical

| ID    | Sub     | Status | Title                                         | Deps   |
|-------|---------|--------|-----------------------------------------------|--------|
| BB-01 | infra   | open   | Push main to origin (82 commits unprotected)  | —      |
| BT-01 | twin    | done   | ~~Validate dose-conditioned fan vs physics fan~~ | —   |

### BB-01: Push main to origin
82 local commits never pushed. Single point of failure — NVMe failure = all bb3 work lost.
Requires reviewing what to push (`.env` must stay local, model artifacts OK, glucose CSVs gitignored).

### BT-01: Validate dose-conditioning ✅ RESOLVED 2026-10-31
**Findings:**
1. Critical bug found and fixed: `build_feature_frame` in `bloodBath/core/patient_state.py`
   assigned `sin_time`/`cos_time` Series with DatetimeIndex to an integer-indexed DataFrame →
   pandas 2.x alignment produces all NaN → ALL inference was running without time-of-day
   features since day one. Fix: use `.values` in the assignment (line 463). Commit separately.
2. Dose-conditioned fan tested after fix: wrong direction (+1 mg/dL per unit instead of −24).
   Root cause: the model sees both `hypothetical_dose` AND `bolus_dose` in the feature vector;
   the 30% masking-to-zero approach doesn't teach the model the correct dose→BG direction.
   Physics fan is correct and is now the default for `fan()`.
3. `ScenarioSimulator.fan()` reverted to always use physics fan. `dose_conditioned_fan()` is
   available for research but not production. See BT-06 for the proper redesign.
**Impact:** sin/cos fix materially improves ALL inference (0U 60min prediction shifted from
113.4 → 143.7 on test window — the time-of-day context was missing everywhere before).

---

## P1 — High Value

| ID    | Sub     | Status   | Title                                              | Deps   |
|-------|---------|----------|----------------------------------------------------|--------|
| BT-02 | twin    | open     | 41-day gap audit (pump 1539514 inactive period)    | —      |
| BT-03 | twin    | open     | CLAUDE.md — project context for AI sessions        | BB-01  |
| BB-02 | bath    | open     | Tandem API systematic field audit                  | —      |
| IN-01 | infra   | open     | Phone notification on hypoglycemia risk            | —      |

### BT-02: 41-day gap audit
Pump 1539514 has an inactive period in 2024-2025 (validate/test split lands there). Verify
the 41-day gap is handled: mask_label=False for those windows → excluded from training.
Also check: does the IOB physiological cap (`b205c913`) interact correctly with the gap?
**Action:** `grep -n "mask_label" bloodBath/data/resampler.py` around lines 563-580, run
`nix develop .#default -- python -c "import ...dataset; check window counts per pump"`.

### BT-03: CLAUDE.md
Create `/home/lkshply/dev/bloodBender/CLAUDE.md` so future AI sessions (and Claude Code)
auto-load project context without needing to mine transcripts. Should cover: branch strategy,
nix shell commands, `.env` location, pump serials, artifact directories, active service URLs.

### BB-02: Tandem API field audit
Architecture redesign plan calls for systematic verification of every extractor field against
real BFF JSON. `casing bug` caught in 2026 silently dropped 93% of BG. Follow-up: audit
all fields in `bloodBath/data/extractors.py` against a captured BFF response sample.

### IN-01: Phone notification on hypoglycemia risk
Infrastructure is ready per nix service design. Trigger: failure_mode_clf predicts
`site_failure` or `exercise_induced_sensitivity` with ≥85% precision, or predicted BG
trajectory dips below 70 mg/dL within 30 min. Delivery: ntfy.sh or similar push service
(no third-party insulin-pump commands — display/alert only). Tagged as deferred in MVP.

---

## P2 — Normal

| ID    | Sub     | Status   | Title                                                   | Deps      |
|-------|---------|----------|---------------------------------------------------------|-----------|
| BT-04 | twin    | open     | Multi-horizon control display (trajectory fan UI)       | BT-01     |
| BB-03 | bath    | blocked  | new_site_lag labels (patient must log site changes)     | patient   |
| BB-04 | bath    | open     | Retire deprecated sync/ + dead API files                | —         |
| IN-02 | infra   | open     | Close stale cleanup/deprecated-modules branch           | BB-01     |
| BT-05 | twin    | open     | arch/redesign branch (Hampel deeper, uv2nix, Rust plan) | BB-01     |

### BT-04: Multi-horizon control display
`feat/multi-horizon-control` not started. Depends on BT-01 confirming dose-conditioned fan
actually works. UI: show fan of 5 BG trajectories (0/0.5/1.0/1.5/2.0 U) with conformal
intervals, each trajectory its own color, reasoning line = expected BG delta per scenario.
No hard gates — patient picks their path.

### BB-03: new_site_lag labels
Feature is wired and flowing (pump 881235 has 139 events). Patient must log site changes
going forward via Tandem CIQ or manual event so future retrains see real site-change labels.
Failure mode classifier (`failure_mode_clf.pkl`) currently has zero `new_site_lag` examples.

### BB-04: Retire deprecated code
`bloodBath/sync/sync_engine.py`, `simple_sync.py` — deprecated.
`bloodBath/api/tandemsource.py`, `events.py`, `common_updated.py` — dead.
See cleanup_deprecated.sh which exists but may be stale.

### BT-05: arch/redesign branch
Hampel filter is in main (`42256133`) — but the design calls for it to also run identically
in the embedded patient_state assembly (bareMetalBender). Other redesign items: uv2nix
deeper integration, unified patient_state schema as single source of truth across
bloodBath/bloodTwin/bareMetalBender, Rust planning for embedded port.

---

## P3 — Low / Someday

| ID    | Sub     | Status   | Title                                                   | Deps      |
|-------|---------|----------|---------------------------------------------------------|-----------|
| BM-01 | metal   | open     | bareMetalBender Rust/Tokio design doc                   | BT-01     |
| BB-05 | bath    | deferred | Dexcom Clarity API cross-check for retroactive corr.    | —         |
| BT-06 | twin    | open     | Dose-conditioned LSTM: proper training regime (Option C first) | —  |
| IN-03 | infra   | deferred | Re-login churn fix in real_tandemsource                 | —         |
| BT-07 | twin    | deferred | sklearn version skew — pin or store as numpy arrays     | done?     |

### BM-01: bareMetalBender Rust/Tokio
Embedded Rust port: async control loop (Tokio), same patient_state schema, runs model.onnx
via onnxruntime-rs or tract. pumpX2 BLE approach as reference for pump comms.
Far future — Python MVP must be validated first.

### BB-05: Dexcom Clarity API
Secondary cross-check for retroactive CGM corrections. Requires separate Dexcom credentials.
Current mitigation: 3h re-pull window (`12472575`) + Hampel filter. Low priority.

### BT-07: sklearn version skew
Scaler now loads as numpy arrays (`1aa372ed`) — may already be done. Verify and close.

---

## Done (recent, for reference)

| ID    | Title                                              | Commit      |
|-------|----------------------------------------------------|-------------|
| —     | Conformal coverage 74.9% → 90% target              | `6304790b`  |
| —     | v1.3: site_change + dose-masking 30%               | `540c7482`  |
| —     | Dashboard: Jinja2, charts.py, hover, cursor        | `e1ddcb60`+ |
| —     | Nix services: 15-min sync + 5-min inference cron  | `9064de00`  |
| —     | Retroactive CGM 3h re-pull window (#6)             | `12472575`  |
| —     | IOB physiological cap + extended bolus dedup       | `b205c913`  |
| —     | site_change extractor fix (line 298 missing arg)   | `8b5e6d33`  |
| —     | Monthly recal + weekly ISF estimation Nix apps     | `0b0b802c`  |
| —     | 160 tests passing (post-train synthesis)           | `12472575`  |
</content>
</invoke>