# Dose-conditioned LSTM

Status: architecture scaffolding (branch `arch/redesign`). No training has been
run yet — this document specifies the design the schema and inference code have
been wired for.

## 1. Why the physics fan is insufficient

The MVP (`lstm.yaml`, 13 features, schema v1.1) never receives a *candidate*
dose as input. The LSTM is trained only on sequences where the dose is whatever
the patient actually did, and at inference the model is run exactly once — on
the unmodified lookback window — producing a single 0 U trajectory.

`ScenarioSimulator.run()` then fakes a dose response on top of that single
prediction:

```
delta[h] = -dose_u × ISF × (1 − iob_remaining(h × 5 min))
```

This is a **physics approximation**, not a learned response:

* ISF is a static (or GP-smoothed) scalar; the real relationship between a bolus
  and future BG is non-linear and state-dependent (current BG, IOB, carbs on
  board, time of day, recent exercise).
* The IOB kernel is a fixed bi-linear Fiasp curve (t_peak = 55 min, DIA =
  300 min). It encodes *insulin pharmacokinetics in general*, not *this
  patient's* observed glucose-lowering dynamics.
* The fan is a set of parallel shifts of one baseline curve, so every dose
  produces the same trajectory shape — the model cannot say "2 U behaves
  differently from 4 U beyond a linear scaling" because it never saw dose vary.

In short: the model has never learned what a dose *does*. The fan is the
engineer's physics guess bolted onto a dose-blind forecast.

## 2. The dose-conditioned design

Add a 14th input feature, `hypothetical_dose` (schema v1.2,
`lstm_dose_conditioned.yaml`). The model now consumes the dose as part of its
input sequence and learns the actual dose → BG mapping end to end.

### Training / inference asymmetry

This is the core idea and the easiest thing to get wrong.

* **Training:** `hypothetical_dose[t] = bolus_dose[t]` for every step `t`
  (0.0 when no bolus). It is a literal copy of the actual-bolus column
  (`dataset.py::_load_data`). The model learns, over 6+ years of the patient's
  history, how a bolus at a given state propagates into the next 60 minutes of
  BG — including patient-specific carb ratios, time-of-day insulin sensitivity,
  and post-meal dynamics.

* **Inference:** the historical lookback already happened and must not be
  rewritten — doing so would take the sequence out of the training
  distribution. We run the model **once per candidate dose**, and for each run
  override **only the current (last lookback) step** with the candidate dose.

## 3. What `hypothetical_dose` is at each lookback step during inference

**Decision: set only the current step; leave all prior steps at their actual
historical `bolus_dose` values.**

For a candidate dose `d` and a lookback of length `L = 288`:

```
hypothetical_dose[t] = bolus_dose[t]   for t = 0 .. L-2   (history, unchanged)
hypothetical_dose[L-1] = d             (the "take d U now" counterfactual)
```

### The options considered

* **Option B — set the ENTIRE lookback to `d`.** Rejected: nonsensical. The past
  already happened; claiming the patient dosed `d` at every one of the last 288
  steps is both physically false and wildly out of distribution. It would also
  double-count with `iob`, which already reflects the real dosing history.

* **Option A — override only the current step; prior steps keep their actual
  historical values.**

* **Option C — override only the current step; prior steps already contain the
  real `bolus_dose` from the patient's history (which is what the model trained
  on anyway).**

A and C describe the **same** operation. That is the correct approach: at train
time `hypothetical_dose == bolus_dose` throughout every window, so at inference
the only step allowed to diverge from that invariant is the one representing
"now". Every earlier step must equal the real bolus so the sequence stays in the
distribution the model learned from.

Concretely: `hypothetical_dose[t] = actual bolus` for `t < now`, and
`hypothetical_dose[now] = candidate_dose`.

Because the feature is a passthrough (no RobustScaling — see
`dataset.py::_PASSTHROUGH_FEATURES`), the candidate dose is injected in native
units (U) directly, no scaler round-trip required.

## 4. Expected improvement

* **Learned, patient-specific dose response.** The model replaces the static
  `ISF × IOB` physics with the patient's own glucose dynamics: effective carb
  ratio, how fast *their* Fiasp actually pulls BG down, and how that varies with
  starting BG and IOB.
* **Time-of-day sensitivity.** Dawn resistance, overnight sensitivity, and
  post-exercise amplification can be learned from data rather than hand-coded as
  `ISF_MULTIPLIERS`.
* **Non-linear dose scaling.** Larger doses need not produce a mere linear
  rescaling of one baseline trajectory — the model can express saturation,
  stacking, and different trajectory *shapes* per dose.

## 5. Backward compatibility

* `lstm.yaml` (13-feature MVP) is untouched. `_load_data` only materialises
  `hypothetical_dose` when it is present in the requested feature list, so
  existing artifacts/scalers load unchanged.
* `ScenarioSimulator` keeps the physics fan (`run()`) as the fallback used for
  13-feature MVP artifacts. The new `dose_conditioned_fan()` is selected only
  when a 14-feature (`hypothetical_dose`-aware) model is loaded; otherwise the
  system degrades gracefully to the physics approximation.

## 6. Open items before training

* Export must widen the ONNX input to 14 features and the scaler must be refit
  with `hypothetical_dose` as a passthrough column (center 0, scale 1).
* Conformal calibration should be regenerated against the 14-feature model.
* Consider whether the correction-step masking in `dataset.py::__getitem__`
  interacts with dose conditioning (the model now *sees* the correction as
  input, so a future revision may relax the horizon mask once the dose-response
  is learned — out of scope for this scaffolding).
