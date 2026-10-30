# bloodBender Inference Service — Design Specification

**Status:** Design approved, implementation pending  
**Last updated:** 2026-10-27  
**Branch target:** `feat/multi-horizon-control` (builds on `arch/align-observed-bg-targets`)

---

## Core philosophy

The service is a **trajectory simulator, not a dosing controller.** It shows the patient a family of predicted BG curves — what the model expects to happen if they deliver 0U, 0.5U, 1U, 1.5U, etc. right now — and lets the patient pick their path. The service never issues a command. It offers probabilistic situational awareness.

All logic is driven by **rates of change and trajectory shape**, not absolute BG thresholds. A patient at 70 mg/dL and rising fast after a prolonged low is in a fundamentally different situation than a patient at 70 mg/dL and falling — the trajectory is what matters, not the value.

---

## What the service does

### Input
- Last 288 rows (24h × 5-min) of the canonical patient state window (11 features, schema v1.0)
- Patient config (see below) — small set of values, some hardcoded, some derived from historical data

### Outputs (per reading cycle, every 5 min)

1. **Trajectory fan** — predicted BG curves for N dose scenarios (e.g. 0, 0.5, 1.0, 1.5, 2.0 U) over the 60-min prediction horizon, each with conformal prediction intervals
2. **Event state probabilities** — calibrated confidence in what is currently driving BG dynamics (updated every reading)
3. **Suggested scenario** — which dose scenario the model considers most likely to achieve stable trajectory, with reasoning expressed as projected BG change rather than a rule

---

## Trajectory fan

Run the ONNX model for each dose scenario:
- For dose `d`: append a bolus event of `d` units to the feature window and re-derive features, then predict
- Scenario 0U is the "do nothing" baseline
- Confidence intervals via per-horizon adaptive conformal prediction (calibrated separately for each of the 12 horizon steps)

Display to patient:
- All curves on the same chart, color-coded by dose
- Shaded confidence band per curve (90% coverage interval)
- Curves update live every 5 min as new CGM readings arrive

**No hard gates on target_low / target_high / min_bg_for_correction.** Those values can exist as display overlays (colored bands the patient sets), but they do NOT gate or modify the model's output. The model speaks in trajectories; the patient reads the chart.

---

## Event state probabilities

Updated every new CGM reading. Expressed as calibrated probabilities summing to (or near) 1 across mutually exclusive events. Each event has an associated expected trajectory signature.

| State | Key signatures | Detection signal |
|---|---|---|
| **Post-meal rise** | Rising `bg_acceleration` (2nd deriv), recent bolus absent or small, meal-aux-head output | Aux classifier confidence |
| **Dawn phenomenon** | 4–8am window, BG rising without insulin/meal input, reproducible across days | Time-locked LSTM residual |
| **Site/infusion failure** | Large persistent positive residual (BG not responding to delivered insulin), IOB delivered but no BG drop | Persistent over-prediction |
| **Exercise-induced sensitivity** | Unusual negative residual (BG dropping faster than IOB explains), precedes or follows patterns of recent lows, no meal | LSTM under-prediction in context |
| **Rebound hyperglycemia** | Following prolonged low, BG now rising steeply — this is the key case: model predicts continued rise even from ~70 mg/dL | Positive slope + recent low history |
| **Stable / nominal** | Residuals near zero, BG within expected band | Default |

The **rebound case** is an explicit design target: after a prolonged low the body often overcorrects (glucose tabs, liver glycogenolysis). The model, having seen this pattern in 6+ years of the patient's data, should learn to predict the subsequent rise and suggest a small dose even when current BG is near the low threshold — driven purely by the trajectory shape, not by whether BG is "above target."

**Display:** probability bars next to the trajectory fan. Updated per-reading. Low confidence events should be shown subdued; only surface the top 2-3.

---

## Suggested scenario

The service nominates the dose scenario whose predicted median trajectory most closely tracks toward the patient's personal stable range — but expresses the reasoning as:

> "At +30 min: +18 mg/dL without dose, +4 mg/dL at 1.0 U. At +60 min: +32 mg/dL without dose, +8 mg/dL at 1.0 U."

Not as: "Correct with 1.0 U because BG > target." The delta language is what matters. Patient decides.

---

## Patient config

### Hardcoded (patient sets once, rarely changes)
| Parameter | Description |
|---|---|
| `DIA` | Duration of insulin action (h). Used for IOB decay curve. Default 5h (Tandem default), adjust per patient experience. |
| `stable_range_low` | Display band lower bound (mg/dL). Not a logic gate. |
| `stable_range_high` | Display band upper bound (mg/dL). Not a logic gate. |
| `dose_step` | Increment between scenario doses (U). Default 0.5. |
| `max_scenario_dose` | Largest dose scenario to simulate (U). Display-only cap, not a safety gate. |

### Derived from historical data (computed, not set)
| Parameter | How derived |
|---|---|
| `ISF` (insulin sensitivity factor) | Fit from historical bolus events: median BG drop per unit over the 2–4h post-bolus window, excluding meal windows (bolus > typical correction threshold). Rolling 30-day window so it reflects current sensitivity. |
| `typical_bolus_profile` | Empirical distribution of bolus sizes, used to calibrate dose scenarios to realistic values for this patient. |
| `dawn_onset_window` | Learned per-patient from reproducible nocturnal BG rises — not hardcoded 4–8am. |

Derived values are estimates, not ground truth. The model does not require them to make predictions — they inform scenario spacing and ISF-based expected-delta reasoning.

---

## Feature roadmap (ordered)

### Schema v1.0 → v1.1 (next retrain)
Adds to the 11-feature vector:
- `iob` — explicit insulin on board computed as bi-exponential decay over `DIA` window from all recent boluses. Population PK parameters are fine (LSTM corrects residual implicitly from BG response history). Keep `time_since_last_bolus` as a companion.
- `bg_acceleration` — 2nd derivative: `(bg_slope_15min − bg_slope_15min.shift(3)) / 15.0`. The single strongest meal-onset cue. Captures the "BG is speeding up" signal that raw slope misses at onset.

**On 3rd derivative (BG jerk):** Not recommended at this stage. CGM noise (~9% MARD at 5-min resolution) means each finite difference amplifies noise. 1st derivative is usable, 2nd requires care, 3rd would be dominated by sensor artifact even after Hampel filtering. Revisit if a denoised CGM stream becomes available (Kalman-smoothed).

**Auxiliary meal-rise classification head:** multi-task output on the shared LSTM encoder. Outputs P(meal_event) alongside the BG predictions. This is the "growing confidence" signal as BG accelerates — trained on labeled windows where a bolus was given ~15–30 min after the detected onset (indirect supervision).

### Schema v1.2 (future)
- `hampel_flag` — binary flag for rows where the Hampel filter rejected a CGM spike. Causal filter only. Must be applied identically in `resampler.py` and the live window assembler.
- Potential: COB if meal announcements are added (out of scope for now)

---

## Conformal prediction (first thing to build)

Wraps the existing ONNX model with zero retraining. Use adaptive split conformal:

1. Split the calibration set (val split) into horizon-specific residual sets
2. For each of the 12 horizon steps, compute nonconformity scores on calibration examples
3. At inference: report (prediction ± q̂_α) for the chosen coverage level (e.g. 90%)
4. Volatility-adaptive: scale q̂ by recent BG variability so intervals widen during turbulent periods

Interval widths to expect at 90% coverage given current MAE:
- 30 min: ~±28–35 mg/dL
- 60 min: ~±45–55 mg/dL

These are honest. Tell the patient honestly that 60-min predictions are uncertain.

---

## What this service is NOT

- Not a closed-loop controller. Never automatically delivers insulin.
- Not a replacement for clinical judgment or physician-approved pump settings.
- Not a safety interlock. The hard safety layer belongs in the embedded pump firmware, not here.
- Displayed probabilities are model estimates, not clinical diagnoses.

---

## Implementation sequence

1. **Conformal prediction** — wrap existing ONNX model, zero retrain, enables confidence UI
2. **IOB + bg_acceleration features** — retrain schema v1.1, update ONNX export
3. **Auxiliary meal-rise head** — multi-task retrain alongside v1.1
4. **ISF derivation from historical data** — data analysis pass, no model change
5. **Failure-mode residual classifier** — GBM on top of conformal residuals, calibrated
6. **Trajectory fan UI** — wire together all of the above into a display layer
7. **Embedded port** — once Python MVP is validated, port to Rust/C++ on-device
