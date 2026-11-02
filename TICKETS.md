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
| BB-01 | infra   | done   | ~~Push main to origin (84 commits pushed)~~   | —      |
| BT-01 | twin    | done   | ~~Validate dose-conditioned fan vs physics fan~~ | —   |

### BB-01: Push main to origin ✅ RESOLVED 2026-10-31
84 commits pushed to `git@github.com:Bolt259/bloodBender.git`. `.env` confirmed not tracked;
`.env.example` has only placeholder credentials; `.envrc` has no secrets. All clear.

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
| BT-02 | twin    | done     | ~~41-day gap audit (pump 1539514 inactive period)~~ | —     |
| BT-03 | twin    | done     | ~~CLAUDE.md — project context for AI sessions~~    | BB-01  |
| BB-02 | bath    | open     | Tandem API systematic field audit                  | —      |
| IN-01 | infra   | done     | ~~Phone notification on hypoglycemia risk~~        | —      |

### BT-02: 41-day gap audit ✅ RESOLVED 2026-10-31

**Findings:**
1. Pump 1539514 (Mobi) data starts 2025-11-11, ends 2026-02-09. No gap within the active
   period — the pump simply wasn't in use before Nov 2025. No mid-active-period gap to exclude.
2. `_create_sequences` (dataset.py) groups by `subject_id` (pump_serial): windows cannot
   span across pumps. Each pump is isolated.
3. BG NaN check (`sequence[target].isna().any()`) in `_create_sequences` would exclude any
   window where training labels are missing, regardless of mask_label.
4. SequenceSegmenter (max_gap_hours=15.0) in processors.py correctly breaks genuine gaps
   into separate segment CSVs. `_load_data` formerly loaded only `sorted(csv_files)[-1]`,
   silently dropping all earlier segments if a pump had multiple. **Fixed:** all segment
   CSVs per pump per split are now loaded and concatenated (commit with BT-02 closure).
5. `mask_label` is a model INPUT feature (passed as-is), not a window filter. Conformal
   calibration DOES respect mask_label (calibrate_conformal.py line 105). Training loss
   does not gate on mask_label directly, but BG NaN exclusion in step 3 handles the same
   cases (large gaps → NaN BG → excluded).

**Action taken:** Fixed `_load_data` to load all segment CSVs, not just the last one.

### BT-03: CLAUDE.md ✅ RESOLVED 2026-10-31
Created and pushed. Covers: nix shells + apps, pump serials, artifact paths, feature schema
v1.3, known gotchas (pandas index alignment, dose-conditioning direction), branch strategy,
systemd timers, open work pointers. Commit `ed0ca247`.

### BB-02: Tandem API field audit
Architecture redesign plan calls for systematic verification of every extractor field against
real BFF JSON. `casing bug` caught in 2026 silently dropped 93% of BG. Follow-up: audit
all fields in `bloodBath/data/extractors.py` against a captured BFF response sample.

### IN-01: Phone notification on hypoglycemia risk ✅ RESOLVED 2026-11-01
`bloodTwin/inference/notify.py` — stdlib-only, no extra deps. Two triggers:
1. Hypo risk: baseline (0 U) predicted BG < 70 mg/dL within 30 min (6 steps)
2. Failure mode: top state is `site_failure` or `exercise_induced_sensitivity` with ≥85% confidence
15-min per-trigger cooldown (`.notify_last_{trigger}.ts` stamp files in LOG_DIR).
Wired into `bloodbender-inference-cron` (packages.nix) — runs after successful `mv` of
inference output. Opt-in: only runs when `NTFY_TOPIC` env var is set.
Set `NTFY_TOPIC=bloodbender-<yourname>` in `.env`, subscribe in ntfy app on phone.
12 tests added (`bloodTwin/tests/test_notify.py`). Total: 172 passing.

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