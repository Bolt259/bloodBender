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

| ID    | Sub     | Status | Title                                                        | Deps   |
|-------|---------|--------|--------------------------------------------------------------|--------|
| BT-10 | twin    | done   | ~~Residual decoder retrain — fix 60-min global-mean collapse~~ | —    |
| BB-01 | infra   | done   | ~~Push main to origin (84 commits pushed)~~                  | —      |
| BT-01 | twin    | done   | ~~Validate dose-conditioned fan vs physics fan~~             | —      |

### BT-10: Residual decoder retrain — fix 60-min global-mean collapse
**Root cause (Opus audit 2026-11-02):** The LSTM decoder outputs **absolute BG**, causing the
60-min head to collapse to a global attractor (~94–113 mg/dL) regardless of current BG.
Confirmed: `corr(current_bg, pred@60min) = −0.019` across 39 archives; output std compressed
from 34.5 → 9.0 (ratio 0.26). Model predicts a fixed endpoint, not a trajectory.
On elevated-BG/low-IOB/no-bolus windows: **98% predict decline, mean −24.9 mg/dL despite
zero insulin driver**. Production MAE: ~33@30min / ~45@60min vs test claims 9.87/15.47.

**Fix:** Switch decoder output from absolute BG to **Δ (residuals relative to current BG)**:
- In training: target = `future_bg - current_bg` (a signed delta per horizon step)
- At inference: add predicted delta back to `current_bg` to get absolute prediction
- This structurally prevents global-mean regression since the model now learns zero-mean deltas

**Implementation:** `bloodTwin/models/lstm.py` decoder (L86-92); training target construction
in `bloodTwin/data/dataset.py`; inference output reconstruction in `bloodTwin/inference/conformal.py`.
Config: new `bloodTwin/configs/lstm_v1_5.yaml`. Full retrain required.

**Secondary improvement (same retrain):** Oversample/upweight windows where elevated BG
persists without correction inside the horizon — currently only ~23% of elevated-window
horizons, creating class imbalance in what the model learns "high BG does."

### BT-11: Recalibrate conformal on production residuals
**Context update (2026-11-02):** BT-10 is done — v1.5 residual decoder deployed. The
original "quick mitigation while BT-10 is in flight" rationale is moot. However,
production intervals are still too narrow: current q_hat ±31@30min / ±52@60min vs
production p90_abs 50.7@30min / 81.2@60min (from 43 archives, mixed v1.4+v1.5).

**Why not recalibrate now:** Only 43 production samples, ~half from v1.4 (global-mean
attractor). The 90th percentile from 43 samples is the 39th largest value — far too
noisy. Recalibrating would inflate intervals using v1.4's inflated errors.

**When:** After 7 days of v1.5-only archives (~2016 samples). Run `nix run
.#bloodbender-accuracy` to check p90_abs@60min — if it's still > q_hat@60min
significantly (>10 mg/dL gap), recalibrate using `nix run .#bloodbender-calibrate`
(val-split only is fine since 14,565 windows >> production sample count).

**Watch:** `bloodBath/logs/accuracy_summary.json` — run weekly after v1.5 roll-out.

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
| BT-11 | twin    | deferred | Recalibrate conformal on production residuals (wait for 7d v1.5 data) | —   |
| BB-06 | bath    | done     | ~~Retrain v1.4 after basal/bolus extractor fixes~~ | BB-02  |
| BT-02 | twin    | done     | ~~41-day gap audit (pump 1539514 inactive period)~~ | —     |
| BT-03 | twin    | done     | ~~CLAUDE.md — project context for AI sessions~~    | BB-01  |
| BB-02 | bath    | done     | ~~Tandem API systematic field audit~~              | —      |
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

### BB-02: Tandem API field audit ✅ RESOLVED 2026-11-01

Full audit of `extractors.py` attribute probes against `eventparser/events.py` dataclasses.
Event flow: raw BFF JSON → `generic.py Event()` → typed dataclass → `EventExtractor`.

**Bugs found and fixed (commit with BB-02 closure):**

1. **Basal rate 10× inflation** — `LidBasalDelivery.commandedRate` is milliunits/hr;
   extractor divided by 100 instead of 1000. Every basal reading was 10× too high.
2. **Basal rate = 0 for LidBasalRateChange/LidNewDay** — real field is `commandedBasalRate`
   (camelCase, U/hr directly); extractor probed dead lowercase `commandedbasalrate`. These
   events produced `basal_rate = 0` rather than being excluded.
3. **LidBolusCompleted/LidBolexCompleted silently dropped** — real field is `insulinDelivered`
   (units). Extractor only probed `deliveredTotal` (LidBolusDelivery). The implicit `None`
   return caused `bolus_dose > 0` to throw TypeError, swallowed by except. Added
   `insulinDelivered` as fallback probe.
4. **LidBgReadingTaken (fingerstick) classified as CGM** — `hasattr(event, 'bg')` matches
   both fingerstick events and is extracted via the `bg` branch. Low severity; fingerstick
   BG is physiologically valid data. Left as-is, noted for awareness.

**Dead probes (harmless):** `currentglucosedisplayvalue`, `sgv`, `value`, `glucoseValue`,
`commandedbasalrate`, `bolusAmount`, `insulin` — none match any real dataclass field.

**⚠ DEPLOYMENT WARNING:** Bugs 1–3 change the values produced by the extractor.
The current v1.3 artifacts (scaler.pkl, model.onnx, conformal_summary.json) were trained and
calibrated on the pre-fix values. DO NOT restart bloodbender-sync or bloodbender-inference-cron
with this fix active until a full retrain + recalibration is complete — see BB-06.

**Bolus double-counting note:** LidBolusDelivery (mid-delivery) and LidBolusCompleted
(end-of-bolus) may fire for the same physical bolus with different timestamps; they both
now extract a dose. The resampler's bolus aggregation (max per 5-min bin) limits the impact
but the exact behavior is unverified — investigate during BB-06 retrain data prep.

### BB-06: Retrain v1.4 ✅ RESOLVED 2026-11-01

**What changed:** Extractor fixes (BB-02) corrected basal 10× inflation and bolus unit errors.
v1.3 artifacts were trained on pre-fix values; retraining was required before deploying.

**Completed steps:**
1. Stopped `bloodbender-sync` and `bloodbender-inference-cron` timers
2. Applied extractor fix to `bloodBender-data-sync/bloodBath/data/extractors.py`
3. Ran full sync — merged CSVs regenerated with correct basal (mean 0.47 U/hr vs 4.7 U/hr pre-fix)
4. Trained v1.4 on RTX 2070 SUPER: `nix develop .#cuda`, config `lstm_v1_4.yaml`
5. Ran conformal calibration: `BLOODTWIN_ARTIFACTS=...v1_4 nix run .#bloodbender-calibrate`
6. Added `Environment=BLOODTWIN_ARTIFACTS=bloodTwin/artifacts/bloodTwin_unified_lstm_v1_4`
   to `~/.config/systemd/user/bloodbender-inference.service`; copied artifacts to data-sync
7. Restarted both timers; verified inference output (basal_rate confirmed ~0.47 U/hr mean)

**v1.4 metrics:**
- test_mae_30min: 9.87 mg/dL  (v1.3: 9.88)
- test_mae_60min: 15.47 mg/dL (v1.3: 15.27)
- test_rmse: 23.04 mg/dL      (v1.3: 22.65)
- meal_acc: 81.1%              (v1.3: 78.7%)
- conformal: 90% coverage, ±31@30min ±52@60min (v1.3: ±31/±51 — nearly identical)

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
| UI-01 | twin    | done     | ~~Hide static dose labels from right side of dashboard chart~~ | —     |
| UI-02 | twin    | done     | ~~Dashboard polish: diag cursor, tooltip swap, grid visibility~~ | —       |
| UI-03 | twin    | done     | ~~CI bands on hover only; bolus hover tooltip; bolus on diag chart~~ | —   |
| BT-04 | twin    | done     | ~~Multi-horizon control display (trajectory fan UI)~~   | BT-01     |
| BB-03 | bath    | blocked  | new_site_lag labels (patient must log site changes)     | patient   |
| BB-04 | bath    | done     | ~~Retire deprecated sync/ + dead API files~~            | —         |
| IN-02 | infra   | done     | ~~Close stale cleanup/deprecated-modules branch~~       | BB-01     |
| BT-05 | twin    | done     | ~~arch/redesign: schema.py, fix stale FEATURE_NAMES, Hampel contract~~ | BB-01 |

### UI-02: Dashboard polish
Three small UX issues in `bloodTwin/dashboard/charts.py` + `templates/index.html.j2`:
1. **Model accuracy chart has no cursor** — add vertical cursor line + hover tooltip
   (matching the history and fan charts).
2. **History tooltip text order is wrong** — bold white text shows time, secondary shows
   BG value. Should be swapped: BG value bold, time secondary.
3. **Gridlines invisible in target bands** — current `stroke="#161b22"` is near-black,
   essentially invisible against the green (`#14532d`) and purple (`#2d1b4e`) bands.
   Raise to `stroke="#30363d"` across all three charts.

### UI-01: Hide static dose labels from right side of chart
Dose/scenario values currently show as permanent labels to the right of the graph.
Since hover already shows these values, the static labels are redundant clutter.
**Fix:** In `bloodTwin/dashboard/charts.py`, remove or suppress the right-side
annotation/text for each fan scenario line. Keep hover tooltip behavior as-is.
**Priority:** P2. Small change, isolated to charts.py.

### BT-04: Multi-horizon control display ✅ RESOLVED 2026-11-01
5 key doses (0/0.5/1.0/1.5/2.0 U) rendered with distinct colors (blue/green/amber/orange/red),
each with a shaded 90% CI polygon. Non-key doses (0.25/0.75/…) rendered as thin dim background
lines. Fan table filtered to 5 key rows with color swatches and `delta_reasoning` text
(e.g. "vs 0U: −0.6@30, −2.3@60 mg/dL"). Legend updated to show 5 named dose colors.
Hover tooltip on key dose points now shows reasoning (first clause of delta_reasoning).

### BB-03: new_site_lag labels
Feature is wired and flowing (pump 881235 has 139 events). Patient must log site changes
going forward via Tandem CIQ or manual event so future retrains see real site-change labels.
Failure mode classifier (`failure_mode_clf.pkl`) currently has zero `new_site_lag` examples.

### BB-04: Retire deprecated code ✅ RESOLVED 2026-11-01
The originally cited files (`sync/`, `api/tandemsource.py`, etc.) were already gone.
What remained was the sweetBlood compatibility layer:
- Removed 3 no-op stubs (`add_sweetblood_args`, `handle_sweetblood_commands`,
  `SweetBloodIntegration`) from `cli/main.py` and all call sites
- `cmd_lstm` replaced with a one-liner exit (was broken: stub returned `{status: deprecated}`,
  then code tried `result['metadata']` — KeyError on every call)
- `cmd_status` stripped of sweetblood output; `TandemHistoricalSyncClient` now uses its
  default `setup_bloodbank_environment` instead of the deprecated shim
- `setup_sweetblood_environment` and `create_sweetblood_structure` removed from
  `utils/structure_utils.py`; re-exports removed from `utils/__init__.py`
- `cleanup_deprecated.sh` deleted (stale, checked for wrong branch)
- Legacy docs deleted: `bloodBath/spec/`, `SWEETBLOOD_STRUCTURE_VERIFICATION.md`,
  `v2.0_Complete_Report.md`, `v2.0_Implementation_Summary.md`
- 139 tests still passing.

### BT-05: arch/redesign ✅ RESOLVED 2026-11-01
**Canonical schema module** (`bloodTwin/schema.py`) created as single source of truth for:
- `FEATURE_NAMES` (15 features, index-ordered) — was hardcoded in 2+ places with stale
  11-feature version in `conformal.py` (missing `iob`, `bg_acceleration`,
  `hypothetical_dose`, `site_change`)
- `FEAT_IDX` — dict for O(1) index lookup
- `LOOKBACK=288`, `HORIZON=12`, `CGM_STEP_MINUTES=5` — model time constants
- `HAMPEL_*` constants — embedded port contract for `BgHampelFilter`

`conformal.py` updated to import from `schema.py` (stale local definition removed).
`scripts/audit_baseline_meanreversion.py` updated to import from `schema.py`.
`CLAUDE.md` updated with Hampel filter algorithm contract for bareMetalBender Rust port.
35 bloodTwin tests passing.

---

## P2 — Normal

| ID    | Sub     | Status   | Title                                                   | Deps      |
|-------|---------|----------|---------------------------------------------------------|-----------|
| BT-09 | twin    | done     | ~~Production accuracy tracker + fix archive gap~~       | —         |

### BT-09: Production retrospective accuracy tracker
**Root cause of "model looks worse than expected"**: production MAE is ~33@30min / ~45@60min
(vs test-set claims 9.87 / 15.47), with a systematic **−21 / −36 mg/dL negative bias** —
the model predicts much more BG drop than actually occurs. Computed from 28 archived
inferences by matching prediction timestamps to actual BG in the merged CSV.

**Contributing factors to investigate:**
1. Training data distribution: v1.4 was trained on 6+ years of historical data where the
   patient was actively managing BG. Every "high BG with no recent bolus" window is
   followed by a correction — the model may have learned this as a BG trajectory feature
   rather than an IOB feature (similar to the BT-08 concern, but less severe).
2. IOB over-counting: the `iob: 7.74` high IOB readings suggest real IOB values may be
   inflated (investigate if IOB cap is applied correctly in production merged CSVs).
3. Archiving is dashboard-gated: `_maybe_archive()` only runs inside `render_page()`,
   so production archives are sparse and biased toward times the dashboard was open.

**✅ RESOLVED 2026-11-02:**
1. `bloodTwin/inference/archive.py` — extracted archive logic; ARCHIVE_MAX=2016 (7d).
   `bloodbender-inference-cron` (packages.nix) now calls it after every successful
   inference. No longer dashboard-gated.
2. `scripts/accuracy_tracker.py` — walks archive, matches 0U baseline predictions to
   actual BG (±2.5 min tolerance), writes `bloodBath/logs/accuracy_summary.json`.
   `nix run .#bloodbender-accuracy` app added.
3. Dashboard shows prod accuracy row below diagnostic chart (MAE + bias, color-coded).
   Run manually or wire to a daily cron via `nix run .#bloodbender-accuracy`.

**Current production accuracy (v1.5, 43 inferences, mixed v1.4+v1.5):**
- @30min: MAE=28.6 mg/dL, bias=−15.9 (model predicts too low)
- @60min: MAE=43.2 mg/dL, bias=−31.8
- Negative bias confirms BT-08 finding: model still predicts decline more than observed.
  As v1.5 production data accumulates (7d window), numbers will reflect v1.5 only.

---

## P3 — Low / Someday

| ID    | Sub     | Status   | Title                                                   | Deps      |
|-------|---------|----------|---------------------------------------------------------|-----------|
| BM-01 | metal   | open     | bareMetalBender Rust/Tokio design doc                   | BT-01     |
| BB-05 | bath    | deferred | Dexcom Clarity API cross-check for retroactive corr.    | —         |
| BT-06 | twin    | done     | ~~Dose-conditioned LSTM: Option C IOB injection at inference~~  | —  |
| BT-08 | twin    | done     | ~~Baseline (0 U) prediction mean-reversion bias audit (v1.5)~~ | —     |
| IN-03 | infra   | deferred | Re-login churn fix in real_tandemsource                 | —         |
| BT-07 | twin    | done     | ~~sklearn version skew — scaler stored as numpy arrays~~ | —        |

### BT-06: Dose-conditioned LSTM: Option C ✅ RESOLVED 2026-11-01
`iob_conditioned_fan()` added to `ScenarioSimulator`. For each candidate dose `d`,
copies the raw lookback window and adds `d` to `iob[now]`. The LSTM uses its
learned IOB→BG response (trained on 6+ years of real bolus history) to produce
a genuinely dose-conditioned trajectory — no retraining needed.

Why better than `dose_conditioned_fan()` (which used `hypothetical_dose`):
the model trained on 30%-masked `hypothetical_dose` = noisy copy of `bolus_dose`,
producing +1 mg/dL/U instead of −24. IOB is a real causal feature the model trusts.

`fan()` now dispatches to `iob_conditioned_fan()` first, falls back to physics
`run()` on error. Option B (proper counterfactual training regime with causal
masking of `hypothetical_dose`) remains the long-term target for a future retrain.

### BM-01: bareMetalBender Rust/Tokio
Embedded Rust port: async control loop (Tokio), same patient_state schema, runs model.onnx
via onnxruntime-rs or tract. pumpX2 BLE approach as reference for pump comms.
Far future — Python MVP must be validated first.

### BB-05: Dexcom Clarity API
Secondary cross-check for retroactive CGM corrections. Requires separate Dexcom credentials.
Current mitigation: 3h re-pull window (`12472575`) + Hampel filter. Low priority.

### BT-08: Baseline (0 U) prediction mean-reversion bias audit
**Hypothesis:** The 0 U scenario may show BG trending toward the patient's typical range
even when there is no physiological mechanism to explain it. The LSTM was trained on
observational data where the patient was actively managing BG — so every time BG drifted
high or low, a correction followed shortly after. The model may have learned a spurious
`(high BG) → BG comes down` pattern that conflates the patient's future correction bolus
with the natural trajectory, making the baseline prediction optimistically self-correcting.

**Why it matters:** If 0 U baseline silently predicts mean-reversion, the physics fan
scenarios all show dose effects as deltas relative to an already-optimistic baseline.
The fan looks smaller than reality — underdosing becomes the "safe" apparent choice.

**Important caveat:** A downward BG trend in the 0 U baseline may simply reflect
active IOB already on board — the LSTM sees `iob` as a feature and correctly anticipates
that circulating insulin will bring BG down. This is NOT a bug. The audit must control
for IOB: only flag windows where current IOB ≈ 0 AND BG is trending down in the baseline.
If the model shows mean-reversion only when IOB > 0, the behavior is correct.

**How to check:**
1. Pull windows where current BG is ≥ 180 mg/dL, current IOB ≈ 0 (< 0.1 U), AND no
   recent bolus in last 90 min. Run baseline (0 U) prediction. If LSTM predicts BG
   declining toward ~140 without any insulin input and negligible IOB, the bias is present.
2. Compare predicted 60-min BG vs actual 60-min BG for these windows across the test set.
   A systematic over-prediction of descent = mean-reversion artifact.
3. Repeat for low BG windows (≤ 80) — if model predicts BG rising without food/glucagon
   input, same artifact in the other direction.

**Possible root causes:**
- Observational data confounding: patient always corrected high BG → model learned "high
  BG precedes a bolus → BG falls" without isolating the insulin effect.
- `mask_label=False` windows excluded but correction windows are over-represented in
  the training distribution (patient is active → fewer genuinely uncorrected highs).
- Related to BT-06: dose conditioning is broken, so the model may be smuggling the
  dose effect into the baseline via the BG trajectory signal.

**Action:** Write `scripts/audit_baseline_meanreversion.py` — load test CSVs, filter
elevated/low windows with no recent bolus, run ConformalPredictor on each, compare
predicted vs actual 60-min BG. Plot predicted_bg_60min vs actual_bg_60min scatter.

**✅ RESOLVED 2026-11-02 (v1.5 re-audit):**
- Elevated BG (≥180), IOB≈0, no recent bolus (1164 windows):
  77% predict decline @60min, mean delta −5.0 mg/dL (v1.4: 98%, −24.9 mg/dL)
- Low BG (≤80), IOB≈0, no recent bolus (2988 windows):
  5% predict decline @60min, mean delta +22.6 mg/dL — correct recovery direction
- Result: **5× bias reduction** vs v1.4. Residual −5 mg/dL mean not clinically alarming
  but still over-predicts BG descent. Watch via BT-09 accuracy_summary.json over time.

**⚠ REOPENED 2026-11-02:** Prior closure was based on a faulty run — script defaulted to
v1.3 artifacts (now fixed to v1.4) and capped at 500 windows (now 2000), and had a
`df[tuple]` bug (now `df[list(FEATURE_NAMES)]`). Opus audit of v1.4 confirms the bias IS
present: 98% of elevated-BG/low-IOB/no-bolus windows predict decline at 60min, mean −24.9
mg/dL, with `corr(current_bg, pred@60min) = −0.019` (essentially no correlation).
Root cause and fix tracked in BT-10 (residual decoder retrain).

### BT-07: sklearn version skew ✅ RESOLVED 2026-11-01
`scaler.pkl` stores `{'scaler': RobustScaler, 'feature_names': [...], 'center_': ndarray(15,), 'scale_': ndarray(15,)}`.
`conformal.py` uses `center_`/`scale_` numpy arrays directly — no sklearn version dependency at inference.

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

---

## Design Review (Opus)

*Generated 2026-11-02 by a full read-through of the codebase. These are opinionated,
forward-looking design tickets — not bug reports against the current working system.
Ordered roughly P0 → P3. Read the "Problem" of DR-01 first; it is the thesis the rest
hang off of.*

### DR-01: The 3× production-vs-test MAE gap is a causal-confounding problem, not a model-capacity problem
**Problem:** Test MAE is 9.7/15.3 @30/60; production MAE is 28.6/43.2 with bias −15.9/−31.8
(confirmed in `accuracy_summary.json`, n=43). This gap is NOT because the LSTM is too small
or under-trained. It is because **test and production are not the same distribution**, and the
difference is causal:

1. **Observational confounding (the core issue).** The model is trained on 6+ years where the
   patient *actively managed* BG. Every elevated-BG window is followed by a correction bolus,
   so the data teaches "high BG → BG falls" as a *trajectory* property of BG itself, independent
   of insulin. `dataset.py` tries to patch this with `correction_step_mask` (masks loss from the
   first in-horizon correction onward), but this only removes the *post-correction* steps from
   the gradient — it does not remove the *selection bias* that uncorrected highs are rare in the
   data (per BB-08 notes, only ~23% of elevated-window horizons stay uncorrected). The model
   still sees a training set dominated by "high → comes down."
2. **The test MAE is computed on the same confounded distribution**, so it looks great: on test
   data, high BG really is usually followed by a correction, so predicting decline is "correct."
   Test MAE is therefore not a trustworthy proxy for production usefulness. The 0U baseline — the
   single most-used output (it feeds `accuracy_tracker`, hypo notify, and the fan anchor) — is
   precisely the counterfactual the training data cannot teach, because a true 0U future is
   almost never observed.
3. **Residual decoder (v1.5) treated a symptom.** It fixed the global-mean *attractor* (output
   no longer collapses to ~100) but not the *direction bias* — the model still predicts more
   descent than happens (bias −32 @60). Deltas are still learned from confounded data.

**Sketch:** Reframe the target. The model should predict **BG conditioned on the actual future
insulin/carb inputs**, then the fan asks "what if those inputs were X instead." Concretely:
- Make future planned insulin an *input teacher-forced along the horizon*, not something the
  model has to implicitly guess. I.e. feed the model the known future `bolus_dose`/`basal_rate`
  for the horizon steps (seq2seq decoder with future-known covariates, like TFT / N-HiTS with
  future inputs). Then "0U" is a real, in-distribution query: set future bolus = 0.
- Evaluate on a **counterfactual-honest test metric**: MAE restricted to windows with *no*
  in-horizon correction (the uncorrected-high / uncorrected-low tails). Track this separately
  from the headline MAE. This is the number that actually predicts production behavior.
- See DR-02 (causal training regime) and DR-06 (future-covariate decoder) for the two halves.
**Effort:** XL
**Priority:** P0 (this is the root cause of the single most important production problem)

### DR-02: Build a counterfactual/causal training regime (the deferred "Option B")
**Problem:** Dose-response is currently a hack stack: Option C (IOB injection at `iob[now]`) in
`iob_conditioned_fan`, with a physics `_dose_bg_delta` fallback. IOB injection abuses a feature:
it adds raw units to `iob` and trusts the model's learned `iob→BG` response. But that response
was *also* learned on confounded data (boluses were given *because* BG was high/rising), so the
"causal" IOB coefficient is contaminated by reverse causation. The fan can show the wrong
magnitude and, in edge cases, wrong sign.

**Sketch:** Three complementary moves, cheapest first:
1. **Reweighting / balanced sampling.** Up-weight the rare uncorrected-high and uncorrected-low
   horizons so the loss is not dominated by "patient corrected." `dataset.py` already computes
   `corrected_in_horizon` per window — use it to build an inverse-propensity sample weight
   (WeightedRandomSampler) so ~50% of each batch is uncorrected-horizon windows. Cheap, no new
   data, directly attacks the confounding. Do this in the *next* retrain regardless.
2. **Future-covariate teacher forcing (see DR-06).** Feed known future insulin into the decoder.
   Training then learns `f(state, future_insulin) → BG`. Dose-response becomes native: vary the
   future-insulin input and read the trajectory. Retires both `iob_conditioned_fan` and the
   physics fan.
3. **Explicit causal structure.** Decompose BG prediction into a learned *endogenous drift*
   (carb absorption, basal mismatch, dawn) plus a *physics insulin-effect term*
   (−dose·ISF·action(t)) whose ISF is the only thing fit, with the insulin term's gradient
   *detached* from the confounded correlation. This is the UVA/Padova-style gray-box approach
   and is the most robust but highest effort.
Ship (1) now; plan (2); keep (3) as the north star.
**Effort:** L (for 1+2), XL (with 3)
**Priority:** P0

### DR-03: The scenario fan's "tapered anchor" silently defeats the residual decoder and corrupts conformal coverage
**Problem:** `scenarios.py` (both `run()` and `iob_conditioned_fan()`) does:
`anchor_offset = current_bg - baseline.predicted_bg[0]`, then adds `anchor_offset * linspace(1,0)`
to the whole trajectory *and to the conformal bounds*. This is layered on top of the residual
decoder which already reconstructs `predicted = delta + current_bg`. Two problems:
1. **Double-anchoring.** v1.5 already grounds `predicted_bg[0]` near `current_bg` (it's
   `current_bg + delta[0]`, and `delta[0]` is small). The extra taper then shifts again by the
   leftover, injecting a `current_bg`-proportional offset back into the trajectory — exactly the
   level-dependence the residual decoder was designed to remove. The displayed curve is no longer
   the model's calibrated output.
2. **Conformal invalidation.** `q_hat` was calibrated against *unanchored* reconstructed
   predictions (`calibrate_conformal.py` does NOT apply any taper). Shifting `lower`/`upper` by
   `anchor_offset*taper` moves the interval off the distribution it was calibrated on, so the
   90% coverage guarantee no longer holds for the displayed bands. This is part of why
   production p90_abs (51/81) >> q_hat (31/52): the bands are both too narrow AND shifted.

**Sketch:**
- Remove the tapered anchor entirely. With a residual decoder the trajectory is already anchored
  at `current_bg` by construction; if `predicted_bg[0]` drifts, that is a *model* signal, not
  something the display should paper over.
- If an anchor is still wanted for UX, apply it *before* calibration too: calibrate `q_hat`
  against the anchored predictions so the bands match what is shown. Pick one convention and use
  it in both `scenarios.py` and `calibrate_conformal.py`.
- Add a test that asserts fan `predicted_bg`/`lower`/`upper` equal `conformal.predict` output for
  the 0U scenario (no silent post-processing divergence).
**Effort:** M
**Priority:** P0 (actively corrupts the headline number and the safety intervals)

### DR-04: Conformal intervals are globally too narrow in production — move to locally-adaptive, state-conditioned calibration
**Problem:** Fixed per-horizon `q_hat` assumes exchangeability between the val split and
production. It is violated: production p90_abs is 51/81 vs calibrated q_hat 31/52 — the model
errs ~60% more in production than on val. The existing `volatility_scale` (ratio of recent BG
std to median cal std, clamped 0.5–3.0) is a crude knob and clearly insufficient. Root causes:
the val split is older/cleaner than live data, and error is heteroscedastic in ways a single
scalar can't capture (post-meal, high-slope, and low-IOB-high-BG windows are far less certain).

**Sketch:** Replace fixed-q conformal with **Mondrian / conditional conformal**:
- Bin calibration residuals by a small set of state buckets: {BG band (low/target/high)} ×
  {|slope| high/low} × {IOB present/absent}. Store a `q_hat[bucket, horizon]` table. At inference,
  pick the bucket from the current window. This captures that an uncorrected-high window has a
  much wider legitimate interval than a flat-in-range window.
- Add **online conformal (ACI — adaptive conformal inference)**: maintain a running coverage
  error and nudge the effective quantile level each step so empirical coverage tracks 90% in
  production regardless of distribution drift. `ConformalCalibrator.update()` already keeps a
  sliding buffer — extend it to adjust α, fed by `accuracy_tracker`'s realized residuals.
- Calibrate on a mix of val + recent production residuals once ≥7 days of v1.5 data exist
  (this is BT-11, still deferred — it should be unblocked now that only v1.5 data accrues).
**Effort:** L
**Priority:** P1

### DR-05: `calibrate_conformal.py` has three correctness/staleness bugs that make recalibration unsafe to run today
**Problem:** The one tool meant to keep intervals honest is subtly broken:
1. **Only reads the last segment.** `load_val_csv` does `pd.read_csv(csvs[-1])` — the exact bug
   BT-02 fixed in `dataset.py` (`_load_data` now concatenates all segment CSVs). A pump whose val
   split was gap-split into multiple `lstm_validate_*.csv` files is silently calibrated on only
   its final segment. Fix: concat all segments, mirroring `dataset.py`.
2. **Default artifacts dir is `v1_3`.** `ARTIFACTS_DIR` constant and both nix apps
   (`bloodbender-calibrate`, `bloodbender-fit-isf` in `packages.nix`) default to
   `bloodTwin_unified_lstm_v1_3`. The monthly cron, if it fires without `BLOODTWIN_ARTIFACTS`
   set, will recalibrate the *wrong, retired* model's artifacts. Point defaults at the active
   artifact (ideally resolve via a `current` symlink — see DR-11).
3. **Calibration path ≠ inference path.** `collect_residuals` builds its own scaling/ffill
   (`pd.DataFrame(window).ffill().fillna(0.0)`) and residual reconstruction, duplicating logic in
   `conformal.ConformalPredictor._run`. These have already drifted (the fan anchor is in one and
   not the other — DR-03). Calibrate by calling the *same* `ConformalPredictor` code path used at
   inference, so q_hat is guaranteed to match served predictions.
**Effort:** M
**Priority:** P1 (blocks DR-04 and BT-11; silently miscalibrates)

### DR-06: Replace the encoder-only LSTM with a seq2seq model that consumes known future covariates
**Problem:** The current model is an encoder-only LSTM: 288-step history → one Linear decoder →
12 outputs at once (`lstm.py` decoder is `Linear(hidden, horizon)`). It has no notion of the
future — it cannot be *told* "no insulin for the next hour" vs "1U now." That's why dose-response
has to be faked post-hoc (physics shift / IOB injection). It also emits all 12 horizons from a
single hidden vector, which is what let the 60-min head collapse in v1.4.

**Sketch:** Move to a decoder that ingests **future-known covariates** per horizon step:
- Candidate architectures, in order of effort/payoff:
  - **Temporal Fusion Transformer (TFT)** — purpose-built for mixed static/past/future covariates
    with quantile output (gives calibrated intervals natively, could reduce reliance on conformal).
  - **N-HiTS / N-BEATSx** — strong, cheap multi-horizon forecasters that accept future exogenous
    inputs; good fit for a single-patient dataset where a transformer may overfit.
  - **Minimal change:** keep the LSTM encoder, add an **autoregressive GRU decoder** that takes
    `[hidden, future_bolus[h], future_basal[h], sin/cos[h]]` at each step. Smallest diff, gets
    future-covariate conditioning without a framework swap.
- Future covariates available at inference: planned/known basal schedule, time-of-day (sin/cos are
  deterministic into the future), and the *candidate dose* (the thing the fan varies).
- This unifies DR-02: the fan becomes "run the decoder with future_bolus = candidate" — one code
  path, correctly-signed, no physics fallback, no IOB abuse.
**Effort:** XL
**Priority:** P1

### DR-07: The feature set is missing the strongest available predictors; several current features are weak or redundant
**Problem:** 15 features, but the ones that matter most for a T1D 60-min forecast are absent, and
some present ones are low-value:
- **Missing — carbohydrates / announced meals.** There is no carb/meal-announcement feature at
  all. Post-meal excursions are the single largest BG driver and the hardest to predict; the model
  is left to infer them from `bg_acceleration` after the rise has started. If the pump/CIQ logs
  carb entries, this is the highest-value feature to add. Even a binary "meal announced in last N
  min" helps enormously.
- **Missing — basal-as-delivered over the horizon** (needed for DR-06) and **temp-basal /
  suspend state** (Control-IQ auto-adjusts basal; the model is blind to it).
- **Missing — exercise / heart-rate / sleep** signal (if any wearable is available) — these drive
  the exercise-sensitivity failure mode the ISF multipliers currently hardcode.
- **Weak/redundant:** `basal_delta` (diff of a mostly-constant rate), `time_since_last_bolus`
  (capped, largely subsumed by `iob`), and keeping both `bg_slope_15min` and `bg_slope_30min` plus
  `bg_acceleration` — three correlated derivatives of one signal. An ablation would likely show
  2 of these can go.
- **`mask_label` as a model INPUT is a design smell** (see DR-08).
**Sketch:** (a) Add carbs/meal-announcement if any source exists — top priority. (b) Run a proper
feature-importance ablation (permutation importance on the val set) and drop features that don't
earn their place; fewer features = less overfit on a single-patient dataset. (c) Add Control-IQ
mode/temp-basal state.
**Effort:** M (ablation + Control-IQ) / L (carbs, depends on data availability)
**Priority:** P1

### DR-08: `mask_label` should be a loss gate, not a model input feature
**Problem:** `mask_label` (True = row valid for training) is fed into the model as input feature
index 12 (confirmed in `schema.py` FEATURE_NAMES and the v1.5 config). This is backwards: a
data-quality flag describing *label reliability* is not a physiological input and gives the model
nothing causal to learn — at best noise, at worst a leak (the model can learn that
`mask_label=False` regions have weird BG and shift its predictions there). BT-02 already noted
"mask_label is a model INPUT feature, not a loss gate." The loss already masks on observed BG
(`target_mask`), so `mask_label` on the *input* is doing double or wrong duty.
**Sketch:** Remove `mask_label` from FEATURE_NAMES (14-feature schema). Keep using it where it
belongs: as a *window-level filter* in `_create_sequences` (drop windows whose horizon labels are
unreliable) and in `calibrate_conformal` (already respected there). Requires a retrain (input
width 15→14). Fold into the next retrain alongside DR-02/DR-07. Same argument applies to
double-checking `mask_bg` — it is legitimately useful as an input (tells the model "this BG is
imputed"), so keep that one.
**Effort:** S (code) + retrain
**Priority:** P2

### DR-09: Production ML monitoring is a single JSON snapshot — build drift detection and alerting
**Problem:** `accuracy_summary.json` is a good start but it is a point-in-time overwrite: no time
series, no per-state breakdown, no alerting. There is no way to see MAE/bias *trending* over days,
no detection of feature/input drift, and the bias (−32@60) that would alarm any ML on-call is
only visible if someone opens the dashboard and reads a number.
**Sketch:**
- **Append, don't overwrite.** Write a dated row (JSONL or SQLite) each run so MAE/bias/coverage
  become time series. Plot rolling 7-day trends on the dashboard.
- **Per-state slices.** Break accuracy down by BG band and corrected/uncorrected-horizon so the
  confounding (DR-01) is visible: the uncorrected-high MAE is the number to watch.
- **Coverage monitor.** For every archived prediction, once the actual arrives, record
  interval-hit/miss; track realized coverage vs target 90%. Feed this into DR-04's ACI loop.
- **Input drift.** Log per-feature summary stats (mean/std/NaN-rate) of served windows vs the
  training distribution; alert on PSI/KL drift. This would have caught the basal 10× bug and the
  sin/cos all-NaN bug *from monitoring*, not from a manual audit.
- **Alert thresholds via the existing ntfy path** (`notify.py`): fire if 7-day bias exceeds a
  bound or coverage drops below ~80%.
**Effort:** L
**Priority:** P2

### DR-10: Archiving/accuracy is single-pump and 0U-only — widen the ground-truth loop
**Problem:** `accuracy_tracker` and `archive` only track the **0U baseline** for pump
`1539514`. The fan's whole value proposition — dose-response — is never validated against reality,
and the other two pumps are invisible. Also `_maybe_archive` was dashboard-gated historically;
it's now in the cron (good), but the matcher silently drops any prediction whose +60 actual isn't
observed yet with no backfill pass.
**Sketch:** (a) Add a backfill mode that re-walks the archive as new BG arrives so late-arriving
actuals get matched (production API lag is 15–40 min). (b) When the patient *does* take a bolus
shortly after an inference, retrospectively compare the fan scenario whose dose ≈ the actual bolus
to the realized trajectory — this is free dose-response validation. (c) Parameterize by pump.
**Effort:** M
**Priority:** P2

### DR-11: Kill the two-directory deployment architecture — it is the project's biggest operational risk
**Problem:** Source lives in `/bloodBender`; production runs from `/bloodBender-data-sync` with its
own (older) git checkout, and every code change requires *manually patching* data-sync and
restarting services (CLAUDE.md §"Two-Directory Architecture"). This has already caused real
incidents: the basal 10× bug and the sin/cos NaN bug shipped partly because source and runtime
diverged, and the retrain runbooks (BB-06) are full of "copy artifacts to data-sync" steps that
are trivially forgettable. A single-patient medical-adjacent system should not have a
copy-paste-to-prod step.
**Sketch:**
- **One repo, one code path.** Make the systemd services run the Nix-built apps (which already
  bake the code into the store via `writeShellApplication`) and set `BLOODBENDER_ROOT` to a *data*
  directory that contains only runtime data (CSVs, logs, artifacts) — never source. `packages.nix`
  already separates `BLOODBENDER_ROOT` (data) from the baked code; the data-sync *code* checkout is
  the redundant part. Delete it; point `BLOODBENDER_ROOT` at a pure-data dir.
- **Artifacts via a `current` symlink.** Replace the hardcoded `v1_3`/`v1_5` paths scattered across
  `service.py`, `packages.nix`, `audit_*.py`, and the systemd `Environment=` line with one
  `artifacts/current → bloodTwin_unified_lstm_v1_5` symlink. One place to flip on deploy;
  recalibration and inference always agree.
- **Deploy = `nix run .#install-services`** after a flake rebuild; no manual patching. Add a smoke
  test to the deploy app that runs one inference and asserts the output schema before enabling
  timers.
**Effort:** L
**Priority:** P2 (not blocking today, but the highest-leverage fix for recurring incidents)

### DR-12: Schema is defined in two places with divergent versions and field counts — collapse to one
**Problem:** There are two "canonical" schemas:
- `bloodTwin/schema.py`: 15 features, `SCHEMA_VERSION` implied v1.4/v1.5, includes
  `hypothetical_dose` + `site_change`.
- `bloodBath/core/patient_state.py`: `SCHEMA` registry = **13 fields**, `SCHEMA_VERSION = "1.1"`,
  and does **not** include `hypothetical_dose` or `site_change`. `build_feature_frame` returns only
  `ALL_COLUMN_NAMES` (13 cols). The two trailing features are bolted on downstream by
  `ensure_passthrough_features` / `dataset.py`'s `_load_data`.
So the "single source of truth" (CLAUDE.md says import `FEATURE_NAMES` from `schema.py`) disagrees
with the "canonical derivation" (`patient_state.build_feature_frame`) on both the feature list and
the version string. The config comments also say "15 inputs, up from 14" then "schema v1.4" then
"schema v1.2" in adjacent lines — the versioning is incoherent.
**Sketch:** Make `patient_state.SCHEMA` the *only* definition; have `bloodTwin/schema.py` import
`FEATURE_NAMES` from it (or generate it) rather than redeclaring. Add `hypothetical_dose` and
`site_change` as real `PatientStateField` entries so `build_feature_frame` emits all 15 columns
directly and `ensure_passthrough_features` can be deleted. Set one `SCHEMA_VERSION` and assert it
matches `model_metadata.json` at inference load (fail fast on mismatch). Add a test that
`build_feature_frame(df).columns == FEATURE_NAMES`.
**Effort:** M
**Priority:** P2

### DR-13: Retrain is a 20-minute manual GPU ritual with no reproducibility guarantee — make it a pipeline
**Problem:** Retraining is "manual (`nix develop .#cuda`, run the script)". Given how often
retrains are forced by data/feature fixes (BB-06, BT-10, and every ticket here that says "fold
into next retrain"), a hand-run process is a liability: no pinned data snapshot, no guaranteed
seed capture in the artifact, no automatic conformal recalibration after, no automatic metric
regression gate. `test_results.yaml` is written but nothing compares it to the previous model.
**Sketch:** A single `nix run .#retrain -- --config lstm_vX.yaml` that: (1) snapshots the merged
CSVs' content hash into the artifact dir; (2) trains; (3) exports ONNX + metadata; (4) runs
`calibrate_conformal`; (5) runs the counterfactual-honest eval (DR-01) and the mean-reversion
audit; (6) **gates**: refuses to promote the `current` symlink if uncorrected-high MAE or
production-proxy bias regressed vs the incumbent; (7) prints a diff of key metrics. Keep training
on the GPU shell but make the orchestration one reproducible command.
**Effort:** L
**Priority:** P2

### DR-14: Dashboard is missing the clinical essentials a T1D patient needs at a glance
**Problem:** The dashboard shows current BG, 30/60 predictions, a fan table, and model internals
(ISF, ECD, failure modes). From a T1D clinical-UX standpoint it is missing the things that matter
most moment-to-moment, and over-exposes model guts that could mislead (the fan suggests doses the
disclaimer says are display-only, but the framing invites action).
**Sketch:**
- **Trend arrow + rate** (↑↑ / ↑ / → / ↓ / ↓↓ with mg/dL/min) — the single most-used CGM UI
  element; derivable from `bg_slope_15min`, currently not surfaced.
- **Time-in-range** for the last 24h (the clinical metric patients actually track) and a hypo/hyper
  event count.
- **Explicit, honest uncertainty framing.** Given DR-03/DR-04 the intervals are known too narrow in
  production — until fixed, the UI should say so, not imply false precision. Show the production MAE
  band (±28/±43) alongside the conformal band so the patient sees real-world spread.
- **Hypo lead-time, prominent.** The one genuinely actionable, safety-relevant output (notify.py
  hypo trigger) should be the biggest element when active, not a row of model internals.
- **Staleness honesty.** 90-min stale threshold is long for a safety display; show data age
  prominently and de-emphasize predictions when the last real CGM point is old.
**Effort:** M
**Priority:** P3

### DR-15: `effective_isf` / failure-mode ISF multipliers are unvalidated hardcoded constants in a safety path
**Problem:** `ISF_MULTIPLIERS` (site_failure 0.05, exercise 2.2, new_site_lag 0.5, dawn 0.65) are
literature/engineering estimates with explicit "tunable per patient" comments, and they feed the
ECD (estimated correction dose) the fan displays. `new_site_lag` has *zero* labeled examples
(BB-03 blocked on the patient logging site changes), `site_change` only flows from one of three
pumps, and the whole failure-mode classifier runs at inference with residual features zero-filled
(`service.py _classify_state` sets residual_30/60 = NaN → zero). So a safety-relevant dose number
is partly driven by a classifier running on partially-missing inputs multiplied by unvalidated
constants.
**Sketch:** (a) Make the failure-mode contribution *advisory only* — show the state and its effect
on ISF as text, but do not let an unvalidated multiplier move the displayed ECD until there is per-
patient validation. (b) Gate `site_failure`/`new_site_lag` suggestions behind patient confirmation
(the code already special-cases site_failure in `best_scenario`; extend the honesty to the ISF
math). (c) Treat the zero-filled residual features at inference as a correctness bug — either train
the classifier to work without them or don't feed a half-blind classifier into a dose number.
**Effort:** M
**Priority:** P3

### DR-16: Remove dead/duplicated code paths that invite the next divergence bug
**Problem:** Several parallel implementations of the same logic coexist, and divergence between
them has already caused bugs:
- `FeatureEngineer._calculate_time_since_last_bolus` / `_calculate_bg_slope` /
  `_add_time_of_day_features` in `resampler.py` are dead re-implementations of the canonical
  `patient_state.derive_*` functions (the class already imports and uses the canonical ones). They
  can silently rot out of sync with the real feature math.
- `dose_conditioned_fan()` is retained "for research" but is known-wrong (BT-01: +1 mg/dL/U).
  Keeping a wrong-signed dose-response function one `fan()` dispatch away from production is a
  footgun.
- `ScenarioFan` declares `baseline`, `model_predicted_peak`, `current_bg`, `model_trigger`,
  `meal_probability`, `failure_top_state/prob` **twice** (dataclass fields duplicated at L239-245
  and L252-258) — harmless today but a clear copy-paste artifact.
**Sketch:** Delete the dead `_calculate_*`/`_add_*` methods; delete `dose_conditioned_fan` (its
lesson is captured in BT-06); de-duplicate the `ScenarioFan` fields. Each deletion removes a path
that can diverge from the one true implementation.
**Effort:** S
**Priority:** P3
</content>
</invoke>