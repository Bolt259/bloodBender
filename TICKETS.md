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
| BT-10 | twin    | open   | **Residual decoder retrain** — fix 60-min global-mean collapse | —    |
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
**Quick mitigation while BT-10 retrain is in flight.** Production error is ~3× test
(q_hat@60min=51.7 vs actual production 60-min error ~45 mg/dL mean with −36 bias).
The current conformal intervals do NOT cover reality at 90% for the systematically biased
60-min predictions.

**Fix:** Run `nix run .#bloodbender-calibrate` against a set of **production archived
predictions vs actual BG outcomes** instead of the val-split CSVs. This will widen
q_hat to reflect real-world uncertainty until BT-10 fixes the underlying bias.
Requires BT-09 archiving to be fixed first (dashboardgated archive → cron-gated).

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
| BT-11 | twin    | open     | Recalibrate conformal on production residuals      | —      |
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
| BT-09 | twin    | open     | Production accuracy tracker + fix archive gap           | —         |

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

**Required work:**
1. Move `_maybe_archive()` from `server.py:render_page()` into the 5-min inference cron
   (`nix/packages.nix bloodbender-inference-cron`) so every inference run is captured.
2. Add a scheduled accuracy script (like `bloodbender-fit-isf`) that walks the archive,
   matches predicted timestamps to actual BG in merged CSVs, computes rolling signed error
   + MAE per horizon, writes `bloodBath/logs/accuracy_summary.json`.
3. Surface prod MAE + bias on the dashboard next to the ghost chart.
4. Investigate the systematic negative bias — likely a training distribution issue; may
   require new data augmentation or a targeted retrain.

---

## P3 — Low / Someday

| ID    | Sub     | Status   | Title                                                   | Deps      |
|-------|---------|----------|---------------------------------------------------------|-----------|
| BM-01 | metal   | open     | bareMetalBender Rust/Tokio design doc                   | BT-01     |
| BB-05 | bath    | deferred | Dexcom Clarity API cross-check for retroactive corr.    | —         |
| BT-06 | twin    | done     | ~~Dose-conditioned LSTM: Option C IOB injection at inference~~  | —  |
| BT-08 | twin    | open     | Baseline (0 U) prediction mean-reversion bias audit — REOPENED | —      |
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
</content>
</invoke>