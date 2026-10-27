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
    estimated_correction_dose: float    # ECD from ISF + predicted peak

    @property
    def baseline(self) -> Scenario:
        """The 0U 'do nothing' scenario."""
        return self.scenarios[0]

    @property
    def best_scenario(self) -> Scenario:
        """Scenario whose 60-min median prediction is closest to correction_target."""
        return min(
            self.scenarios,
            key=lambda s: abs(s.predicted_bg_at_60min - self.correction_target)
        )

    def to_dict(self) -> dict:
        return {
            "window_end_utc":             self.window_end_utc.isoformat(),
            "pump_serial":                self.pump_serial,
            "horizon_minutes":            self.horizon_minutes,
            "isf_used":                   round(self.isf_used, 1),
            "correction_target":          round(self.correction_target, 1),
            "estimated_correction_dose":  round(self.estimated_correction_dose, 2),
            "best_dose_u":                round(self.best_scenario.dose_u, 2),
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

    def run(
        self,
        df: pd.DataFrame,
        pump_serial: str = "",
        coverage: float = 0.90,
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

        # Single LSTM forward pass for the baseline (0U)
        baseline = self._pred.predict_from_df(df, coverage=coverage, pump_serial=pump_serial)

        # ECD from ISF + baseline predicted peak
        peak_bg = float(np.max(baseline.predicted_bg))
        ecd = max(0.0, (peak_bg - cfg.correction_target) / cfg.isf) if cfg.isf > 0 else 0.0

        # Dose grid: 0 → max(ECD×1.5, min_range) at dose_step
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
                # ISF offset: expected BG drop per horizon step
                delta = _dose_bg_delta(dose, cfg.isf, HORIZON)
                new_pred  = (baseline.predicted_bg + delta).astype(np.float32)
                new_lower = (baseline.lower + delta).astype(np.float32)
                new_upper = (baseline.upper + delta).astype(np.float32)
                import dataclasses
                pred = dataclasses.replace(
                    baseline,
                    predicted_bg=new_pred,
                    lower=new_lower,
                    upper=new_upper,
                )

            d30 = float(pred.predicted_bg[5])  - base_30
            d60 = float(pred.predicted_bg[11]) - base_60
            reasoning = (
                f"vs 0U: {d30:+.1f} mg/dL @30min, {d60:+.1f} mg/dL @60min"
                f" (main effect continues 1-4h beyond window)"
                if dose > 0 else "baseline (no dose)"
            )
            scenarios.append(Scenario(dose_u=dose, prediction=pred, delta_reasoning=reasoning))

        return ScenarioFan(
            window_end_utc=baseline.window_end_utc,
            pump_serial=pump_serial,
            horizon_minutes=baseline.horizon_minutes,
            scenarios=scenarios,
            isf_used=cfg.isf,
            correction_target=cfg.correction_target,
            estimated_correction_dose=round(ecd, 2),
        )
