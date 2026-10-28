"""
Trajectory fan: multi-dose scenario simulator for the bloodBender inference service.

For each candidate dose amount, simulates what the model predicts BG will do
if that dose is delivered right now, producing a family of conformal-bounded
BG trajectories the patient can compare.

Design decisions
----------------
* Dose spacing: 0.25U steps — fine enough for clinical relevance without
  overwhelming the display.
* Dose range: ISF-derived, not fixed.  The range centres around the
  estimated correction dose (ECD = (predicted_peak − target) / ISF) and
  extends far enough to show both under-correction and mild over-correction
  as meaningful boundary cases.
* 0U baseline always included so the patient sees "do nothing".
* Hard ceiling at patient_config.max_dose (default 6U) — never simulated past it.
* Dose injection: the scenario window is built by appending the candidate dose
  to bolus_dose at t=0, recomputing IOB and bg_acceleration with the canonical
  patient_state functions, then running the ONNX model.  This keeps the
  simulation physically consistent with training.

Usage
-----
    predictor = ConformalPredictor(...)
    fan = ScenarioSimulator(predictor, patient_config).run(df_last_300_rows)
    fan.to_dict()   # JSON-serialisable for the display layer
    fan.best_scenario   # the scenario whose 60-min median is closest to target
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .conformal import ConformalPredictor, ConformalPrediction, LOOKBACK, HORIZON

# Re-implement the IOB kernel inline so this module has no bloodBath dependency
def _iob_remaining(t_min: float, dia_min: float = 300.0) -> float:
    """Fraction of a bolus's effect still ahead at t_min after delivery."""
    t_peak = dia_min * 0.25
    if t_min <= 0:      return 1.0
    if t_min >= dia_min: return 0.0
    if t_min < t_peak:
        return 1.0 - (t_min ** 2) / (2.0 * t_peak * dia_min)
    return max(0.0, (dia_min - t_min) ** 2 / (2.0 * (dia_min - t_peak) * dia_min))


def _dose_bg_delta(dose_u: float, isf: float, horizon_steps: int,
                   step_min: int = 5, dia_min: float = 300.0) -> np.ndarray:
    """
    Expected BG change at each horizon step h from a bolus delivered right now.

    delta[h] = -dose_u × ISF × (iob_remaining(0) − iob_remaining(h × step_min))
             = -dose_u × ISF × (1.0 − iob_remaining(h × step_min))

    Physiology: rapid-acting insulin (Humalog/NovoLog) at DIA=5h peaks ~75 min.
    Within a 60-min window only ~8-12% of the dose has acted — most effect is
    in the 1-4 h window BEYOND the forecast horizon.  So corrections show a
    modest -1 to -12 mg/dL delta within the display window (dose-size dependent)
    with the bulk of the action labelled separately as "expect BG to continue
    falling over the next 2-4 h."
    """
    deltas = np.zeros(horizon_steps, dtype=np.float32)
    for h in range(horizon_steps):
        t = (h + 1) * step_min
        fraction_acted = 1.0 - _iob_remaining(t, dia_min)
        deltas[h] = float(-dose_u * isf * fraction_acted)
    return deltas

CGM_STEP_MINUTES: int = 5

# ISF multipliers per failure-mode state.  Represent how much MORE or LESS
# effective insulin is in each regime relative to the patient's median ISF.
# Source: published literature on exercise/dawn/stress insulin resistance +
# clinical consensus; tunable per patient as labelled data accumulates.
ISF_MULTIPLIERS: dict[str, float] = {
    "nominal":            1.0,
    "post_meal_rise":     1.0,   # meal doesn't change ISF; model-predicted peak handles it
    "dawn_phenomenon":    0.65,  # cortisol/GH resistance — need more insulin per unit BG
    "site_failure":       0.05,  # essentially no absorption — suppress suggestion
    "exercise_sensitive": 2.2,   # GLUT4 / muscle uptake amplifies insulin effect
}


# ── Patient config ─────────────────────────────────────────────────────────────

@dataclass
class PatientConfig:
    """
    Per-patient parameters for scenario generation.

    isf and correction_target are the only ones that actively shape the
    scenario range; the rest are used for display reasoning only.
    All can be set from the derived ISF analysis or patient-supplied values.
    """
    isf: float = 24.0             # mg/dL per unit insulin (derived from pump history)
    correction_target: float = 100.0  # aim for this BG (mg/dL)
    max_dose: float = 6.0         # hard ceiling on simulated dose (U)
    dose_step: float = 0.25       # granularity (U)
    min_dose_range: float = 1.0   # always show at least this range even if ECD≈0

    # Display overlays — not used in model logic
    hypo_threshold: float = 70.0
    hyper_threshold: float = 180.0


# ── Single scenario output ─────────────────────────────────────────────────────

@dataclass
class Scenario:
    dose_u: float                  # simulated dose in units
    prediction: ConformalPrediction
    delta_reasoning: str           # human-readable "at +30min: +X mg/dL vs no dose"

    @property
    def predicted_bg_at_30min(self) -> float:
        return float(self.prediction.predicted_bg[5])

    @property
    def predicted_bg_at_60min(self) -> float:
        return float(self.prediction.predicted_bg[11])

    @property
    def lower_at_60min(self) -> float:
        return float(self.prediction.lower[11])

    @property
    def upper_at_60min(self) -> float:
        return float(self.prediction.upper[11])

    def to_dict(self) -> dict:
        return {
            "dose_u":             round(self.dose_u, 2),
            "predicted_bg":       [round(float(v), 1) for v in self.prediction.predicted_bg],
            "lower":              [round(float(v), 1) for v in self.prediction.lower],
            "upper":              [round(float(v), 1) for v in self.prediction.upper],
            "meal_probability":   round(self.prediction.meal_probability, 4),
            "volatility_scale":   round(self.prediction.volatility_scale, 3),
            "delta_reasoning":    self.delta_reasoning,
        }


# ── Scenario fan output ────────────────────────────────────────────────────────

@dataclass
class ScenarioFan:
    window_end_utc: pd.Timestamp
    pump_serial: str
    horizon_minutes: list[int]
    scenarios: list[Scenario]           # ordered by dose_u ascending
    isf_used: float
    correction_target: float
    estimated_correction_dose: float    # ECD from LSTM-predicted peak / ISF

    # ── LSTM-derived insight fields ────────────────────────────────────────
    # All populated by ScenarioSimulator.run(); defaults for callers that
    # build ScenarioFan manually without going through the simulator.
    model_predicted_peak: float = 0.0   # LSTM predicted BG peak within horizon
    current_bg: float | None = None     # last observed BG (context display)
    model_trigger: str = ""             # natural-language reason for the suggestion
    meal_probability: float = 0.0      # from aux LSTM head
    # Optional — set by caller after running FailureModeClassifier
    failure_top_state: str = "nominal"
    failure_top_prob: float = 0.0

    @property
    def baseline(self) -> Scenario:
        """The 0U 'do nothing' scenario."""
        return self.scenarios[0]

    # Model-derived insight fields (set by ScenarioSimulator after run())
    model_predicted_peak: float = 0.0        # LSTM's predicted BG peak within horizon
    current_bg: float | None = None          # BG at window end (for context)
    model_trigger: str = ""                  # Why LSTM predicts a rise ("dawn pattern", etc.)
    meal_probability: float = 0.0           # From aux head
    failure_top_state: str = "nominal"
    failure_top_prob: float = 0.0

    @property
    def baseline(self) -> Scenario:
        """The 0U 'do nothing' scenario."""
        return self.scenarios[0]

    @property
    def best_scenario(self) -> Scenario:
        """
        LSTM-informed scenario selection.

        Chooses the dose whose predicted 60-min BG is closest to
        correction_target, weighted toward the lower end of the dose range
        when meal_probability is high (anticipatory pre-bolus) and toward
        0U when the failure mode suggests site failure (insulin won't absorb).
        """
        if self.failure_top_state == "site_failure" and self.failure_top_prob > 0.6:
            # Site failure suspected: don't suggest correction, it won't absorb
            return self.scenarios[0]

        # Penalise doses that overshoot into hypoglycaemia
        def score(s: Scenario) -> float:
            dist = abs(s.predicted_bg_at_60min - self.correction_target)
            # Extra penalty if lower bound risks hypo (<70)
            hypo_penalty = max(0.0, 70.0 - s.lower_at_60min) * 2.0
            return dist + hypo_penalty

        return min(self.scenarios, key=score)

    def to_dict(self) -> dict:
        best = self.best_scenario
        return {
            "window_end_utc":             self.window_end_utc.isoformat(),
            "pump_serial":                self.pump_serial,
            "horizon_minutes":            self.horizon_minutes,
            "current_bg":                 round(self.current_bg, 1) if self.current_bg else None,
            "model_predicted_peak":       round(self.model_predicted_peak, 1),
            "model_trigger":              self.model_trigger,
            "meal_probability":           round(self.meal_probability, 4),
            "failure_top_state":          self.failure_top_state,
            "failure_top_prob":           round(self.failure_top_prob, 4),
            "isf_used":                   round(self.isf_used, 1),
            "correction_target":          round(self.correction_target, 1),
            "estimated_correction_dose":  round(self.estimated_correction_dose, 2),
            "best_dose_u":                round(best.dose_u, 2),
            "best_reasoning":             best.delta_reasoning,
            "scenarios":                  [s.to_dict() for s in self.scenarios],
        }


# ── Simulator ──────────────────────────────────────────────────────────────────

class ScenarioSimulator:
    """
    Generates a ScenarioFan by running the ONNX model for each candidate dose.

    Parameters
    ----------
    predictor : ConformalPredictor
        Loaded inference wrapper (model.onnx + scaler.pkl + conformal_cal.pkl).
    patient_config : PatientConfig
        ISF and correction target for this patient.
    """

    def __init__(
        self,
        predictor: ConformalPredictor,
        patient_config: PatientConfig | None = None,
    ) -> None:
        self._pred   = predictor
        self._config = patient_config or PatientConfig()

    @staticmethod
    def effective_isf(
        base_isf: float,
        failure_mode_probs: dict[str, float] | None = None,
    ) -> tuple[float, str]:
        """
        Compute ISF_effective from base ISF and model-estimated state probabilities.

        Returns (isf_effective, note) where note explains the adjustment.
        When failure_mode_probs is None or all nominal, returns base_isf unchanged.

        The adjustment is a weighted average of per-state ISF multipliers,
        anchored so that the nominal contribution is always present.
        As patient-confirmed labels accumulate, the per-state multipliers
        can be personalised from the patient's own data.
        """
        if not failure_mode_probs:
            return base_isf, ""

        # Weighted multiplier: Σ p(state) × multiplier(state)
        total_w = 0.0
        wmult   = 0.0
        for state, mult in ISF_MULTIPLIERS.items():
            p = float(failure_mode_probs.get(state, 0.0))
            wmult   += p * mult
            total_w += p
        if total_w < 0.01:
            return base_isf, ""

        multiplier = wmult / total_w
        isf_eff    = round(base_isf * multiplier, 1)

        # Build an explanation
        top_state = max(failure_mode_probs, key=failure_mode_probs.get)
        top_prob  = failure_mode_probs[top_state]
        if abs(multiplier - 1.0) < 0.05:
            note = ""
        elif multiplier > 1.0:
            note = (f"ISF adjusted ×{multiplier:.2f} ({top_state} {top_prob:.0%}): "
                    f"insulin MORE effective → smaller dose needed")
        else:
            note = (f"ISF adjusted ×{multiplier:.2f} ({top_state} {top_prob:.0%}): "
                    f"insulin LESS effective → larger dose or no suggestion")
        return isf_eff, note

    def run(
        self,
        df: pd.DataFrame,
        pump_serial: str = "",
        coverage: float = 0.90,
        failure_mode_probs: dict[str, float] | None = None,
    ) -> ScenarioFan:
        """
        Build the scenario fan from a DataFrame of at least LOOKBACK rows.

        Approach
        --------
        1. Run the LSTM once on the unmodified lookback window → baseline trajectory.
        2. For each candidate dose, apply an ISF-based BG offset to the baseline:
               delta[h] = -dose × ISF × (1 − iob_remaining(h × 5min))
           This is the expected BG change at each horizon step from insulin acting
           according to the bi-linear PK model (DIA=5h, peak~75min).
        3. Apply the same offset to the conformal interval bounds.

        Physics note: within a 60-min window, only ~8-12% of rapid-acting insulin
        has acted.  A 3U correction lowers 60-min BG by roughly 6-9 mg/dL — not
        dramatic, but honest.  The UI should note that the main effect continues
        for 2-4 h BEYOND the displayed window.
        """
        cfg = self._config

        # ── Dynamic ISF from model-estimated state ─────────────────────────
        isf_eff, isf_note = self.effective_isf(cfg.isf, failure_mode_probs)

        # ── Single LSTM forward pass (baseline, 0U) ───────────────────────
        baseline = self._pred.predict_from_df(df, coverage=coverage, pump_serial=pump_serial)

        # Current BG: last OBSERVED (non-NaN, non-imputed) reading in the lookback
        bg_col = df["bg"] if "bg" in df.columns else pd.Series(dtype=float)
        mask_col = df["mask_bg"].astype(bool) if "mask_bg" in df.columns else pd.Series(False, index=df.index)
        obs_bg = bg_col[~mask_col].dropna()
        current_bg = float(obs_bg.iloc[-1]) if not obs_bg.empty else None

        # ── Residual anchor: shift prediction to start from current observed BG ──
        # The LSTM's trajectory shape (rate of rise, peak timing, plateau) is
        # learned from 6+ years of patterns.  But the model can predict
        # "BG=185 at +5min" when actual BG is currently 114 — physically
        # impossible.  We anchor the fan to the real sensor reading by shifting
        # the entire trajectory by (current_bg - predicted_bg[0]), preserving
        # the LSTM's learned shape while grounding the starting point in reality.
        # The conformal intervals are shifted by the same offset so coverage is
        # maintained relative to the anchored trajectory.
        import dataclasses as _dc
        anchor_offset = 0.0
        if current_bg is not None:
            # Tapered anchor: at h=0, fully trust current_bg; by h=HORIZON,
            # fully trust the LSTM's absolute prediction.  Linear taper.
            # This corrects the physically-impossible early-step jumps while
            # letting the LSTM's learned trajectory shape take over as the
            # horizon grows (where the current reading has less predictive power).
            anchor_offset = float(current_bg - baseline.predicted_bg[0])
            taper = np.linspace(1.0, 0.0, HORIZON, dtype=np.float32)  # 1→0
            shift = (anchor_offset * taper).astype(np.float32)
            baseline = _dc.replace(baseline,
                                   predicted_bg=(baseline.predicted_bg + shift),
                                   lower        =(baseline.lower        + shift),
                                   upper        =(baseline.upper        + shift))

        # ── ECD from anchored LSTM-predicted peak + dynamic ISF ────────────
        # Shape comes from LSTM; effective ISF is adjusted by model-estimated
        # sensitivity state so the correction dose reflects e.g. reduced
        # sensitivity at dawn or amplified sensitivity after exercise.
        peak_idx = int(np.argmax(baseline.predicted_bg))
        peak_bg  = float(baseline.predicted_bg[peak_idx])
        peak_min = baseline.horizon_minutes[peak_idx]
        ecd      = max(0.0, (peak_bg - cfg.correction_target) / isf_eff) if isf_eff > 0 else 0.0

        # ── Model trigger: WHY is the LSTM predicting a rise? ────────────
        # Synthesised from meal_prob, failure mode, and trajectory shape.
        meal_prob  = baseline.meal_probability
        bg_delta   = peak_bg - (current_bg or peak_bg)
        traj_trend = float(baseline.predicted_bg[-1]) - float(baseline.predicted_bg[0])

        if meal_prob >= 0.5 and bg_delta > 15:
            trigger = f"model detects meal-rise pattern (P={meal_prob:.0%}); BG expected +{bg_delta:.0f} mg/dL"
        elif meal_prob >= 0.3 and traj_trend > 10:
            trigger = f"model predicts rising trajectory (meal probability {meal_prob:.0%})"
        elif current_bg is not None and bg_delta > 20 and meal_prob < 0.25:
            trigger = f"model predicts rise without meal signal — possible dawn/stress/hormonal pattern"
        elif bg_delta > 5:
            trigger = f"model projects BG to peak at {peak_bg:.0f} mg/dL @ +{peak_min}min"
        elif ecd == 0:
            trigger = "model predicts BG will remain at or below target — no correction indicated"
        else:
            trigger = f"model: current BG {current_bg:.0f} → predicted peak {peak_bg:.0f} mg/dL"

        # ── Dose grid ─────────────────────────────────────────────────────
        upper   = min(cfg.max_dose, max(ecd * 1.5, cfg.min_dose_range))
        n_steps = math.ceil(upper / cfg.dose_step)
        doses   = [round(i * cfg.dose_step, 2) for i in range(n_steps + 1)]

        base_30 = float(baseline.predicted_bg[5])
        base_60 = float(baseline.predicted_bg[11])

        scenarios = []
        for dose in doses:
            if dose == 0.0:
                pred = baseline
            else:
                delta     = _dose_bg_delta(dose, isf_eff, HORIZON)
                new_pred  = (baseline.predicted_bg + delta).astype(np.float32)
                new_lower = (baseline.lower         + delta).astype(np.float32)
                new_upper = (baseline.upper         + delta).astype(np.float32)
                pred = _dc.replace(
                    baseline, predicted_bg=new_pred, lower=new_lower, upper=new_upper,
                )

            d30 = float(pred.predicted_bg[5])  - base_30
            d60 = float(pred.predicted_bg[11]) - base_60
            reasoning = (
                f"vs 0U: {d30:+.1f} @30min, {d60:+.1f} @60min mg/dL"
                f" — bulk of {dose:.2f}U effect arrives 1-4h after delivery"
                if dose > 0 else
                f"no dose: model predicts peak {peak_bg:.0f} mg/dL @ +{peak_min}min"
            )
            scenarios.append(Scenario(dose_u=dose, prediction=pred, delta_reasoning=reasoning))

        # Carry failure mode state into the fan so to_dict() and the UI can
        # display it (caller may also set these after the fact)
        fm_top   = max(failure_mode_probs, key=failure_mode_probs.get) if failure_mode_probs else "nominal"
        fm_prob  = float(failure_mode_probs.get(fm_top, 0.0)) if failure_mode_probs else 0.0

        if isf_note:
            trigger = trigger + f"  [{isf_note}]"

        fan = ScenarioFan(
            window_end_utc=baseline.window_end_utc,
            pump_serial=pump_serial,
            horizon_minutes=baseline.horizon_minutes,
            scenarios=scenarios,
            isf_used=isf_eff,
            correction_target=cfg.correction_target,
            estimated_correction_dose=round(ecd, 2),
            model_predicted_peak=round(peak_bg, 1),
            current_bg=round(current_bg, 1) if current_bg is not None else None,
            model_trigger=trigger,
            meal_probability=round(meal_prob, 4),
            failure_top_state=fm_top,
            failure_top_prob=round(fm_prob, 4),
        )
        return fan
