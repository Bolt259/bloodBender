"""
ML-based local / dynamic ISF (insulin sensitivity factor).

Combines a hard-derived GLOBAL baseline (the patient's median ISF from the
correction-bolus analysis) with a small LEARNED LOCAL model that predicts how
much the patient's *current-state* ISF deviates from that baseline.

Why this design
---------------
The static baseline (24 mg/dL/U for pump 1539514) is a single median over
hundreds of correction episodes.  But ISF is not constant: it drifts with
time-of-day (dawn resistance ↓ sensitivity), recent BG trend, and insulin
on board.  The failure-mode multipliers that used to encode this
(``ISF_MULTIPLIERS`` in scenarios.py: dawn ×0.65, exercise ×2.2, …) were
hand-set constants.  This module replaces the *learnable* part of that — the
normal day-to-day sensitivity variation — with multipliers estimated from the
patient's own correction history.

Model class — why a Gaussian Process
-------------------------------------
This is a genuinely small-data problem: pump 1539514 yields ~200 clean
correction episodes, ~1000 across all of the patient's pumps.  At that scale:

* Gradient-boosted trees / deep nets overfit and give no honest uncertainty.
* A Gaussian Process (sklearn.gaussian_process) is purpose-built for N≈10²–10³,
  gives a calibrated *predictive variance*, and is fast at inference for this N
  (one K* matvec against the stored training set — well under 5 ms).

The GP regresses the LOG-RATIO target  y = log(ISF_observed / ISF_baseline)
on the standardised state features, so a prediction mu maps to a multiplicative
deviation  exp(mu)  — naturally positive, symmetric in log-space, and
interpretable ("×1.3 → 30% more sensitive right now").

Combining with the baseline — shrink-to-prior
----------------------------------------------
The GP posterior at a query point is N(mu, sigma²), where sigma² decomposes
into irreducible biological noise (the WhiteKernel) and epistemic uncertainty
(distance from training support).  We do NOT trust mu blindly.  Two gates
shrink the correction toward 0 (i.e. toward the baseline):

    w_local = signal_var / (signal_var + epistemic_var)   per-query: OOD → 0
    w_global = blend_strength ∈ [0,1]                     from CV: no edge → 0
    w = w_local * w_global

``blend_strength`` is set at fit time from the out-of-fold CV improvement, so a
model that fails to beat the baseline in cross-validation collapses to the
baseline everywhere.  The effective ISF is then

    ISF_eff = clip( baseline * exp(clip(w * mu, ±1)),  ISF_MIN,  ISF_MAX )

so a confident, CV-validated prediction uses the learned multiplier, while an
out-of-distribution query or a weak model collapses to the baseline — and the
result is always a physically plausible ISF.

Interface
---------
    model = ISFModel(baseline_isf=24.0)
    model.fit(pump_data_df)            # df = raw merged lstm_*.csv rows
    isf = model.predict(features_dict) # -> float, effective ISF (mg/dL/U)
    model.save(path); ISFModel.load(path)

``predict`` accepts a dict of the current patient state (same field names as
the schema) and is tolerant of missing keys (they default to 0 / neutral).
Any failure inside ``predict`` falls back to the baseline — this component must
never break the inference path.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Physical guard rails ────────────────────────────────────────────────────
# ISF for a T1D adult on rapid-acting insulin is realistically ~5–100 mg/dL/U.
# Anything outside this is a data artefact, not physiology.
ISF_MIN: float = 5.0
ISF_MAX: float = 100.0

# Cap the learned log-multiplier so a single confident-but-extreme prediction
# cannot swing ISF more than ~2.7× in either direction before clipping.
MAX_ABS_LOG_RATIO: float = 1.0

# ── Correction-episode extraction parameters ────────────────────────────────
# Mirrors bloodTwin/analytics/derive_isf.find_correction_boluses so the TARGET
# used to train this model is exactly the quantity the baseline median was
# computed from — only the aggregation differs (per-episode regression here vs
# a single median there).
MIN_BOLUS_U: float = 0.5        # ignore micro-boluses / noise
MAX_BOLUS_U: float = 10.0       # above this it's a meal bolus, not a pure correction
MIN_BG_AT_BOLUS: float = 140.0  # only episodes where BG was elevated
NADIR_WINDOW_MIN: int = 90      # look for BG nadir within 90 min of the bolus
MEAL_EXCLUSION_MIN: int = 90    # no other bolus within ±90 min (isolation)
MIN_BG_DROP: float = 10.0       # BG must actually fall > this for a valid estimate
_STEP_MIN: int = 5              # minutes per row

# ── Feature set fed to the GP ────────────────────────────────────────────────
# Chosen from the schema v1.1 columns the literature links to ISF variation:
#   sin_time, cos_time  – circadian / dawn phenomenon (primary driver)
#   bg_at_bolus         – higher starting BG weakly tracks lower apparent ISF
#   bg_slope_15min      – recent trend (falling already → care re: exercise)
#   bg_slope_30min      – slower trend component
#   iob                 – residual insulin confounds the apparent drop
#   bg_acceleration     – curvature of the recent trajectory
# Kept deliberately small (7 features) so the GP is well-conditioned at N≈200.
FEATURE_ORDER: tuple[str, ...] = (
    "sin_time",
    "cos_time",
    "bg_at_bolus",
    "bg_slope_15min",
    "bg_slope_30min",
    "iob",
    "bg_acceleration",
)

# Map schema column names → the feature name used here.  bg_at_bolus is the
# schema's "bg"; everything else shares its name.
_SCHEMA_TO_FEATURE: dict[str, str] = {
    "bg": "bg_at_bolus",
}


# ── Episode extraction ───────────────────────────────────────────────────────

def extract_correction_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract isolated correction-bolus episodes and their per-episode ISF plus
    the state features at delivery time.

    Parameters
    ----------
    df : DataFrame
        Raw merged rows (schema v1.1 columns) with a ``timestamp`` column and
        at least: bg, bolus_dose, bg_slope_15min, bg_slope_30min, sin_time,
        cos_time, iob, bg_acceleration.  Must be time-sorted / 5-min spaced.

    Returns
    -------
    DataFrame with one row per valid episode:
        timestamp, bolus_dose, bg_at_bolus, nadir_bg, bg_drop, isf_estimate,
        and one column per FEATURE_ORDER entry (state at the bolus row).

    Confounders filtered (same as the baseline median extractor):
      * concurrent meal        → isolation window (no other bolus within ±90 min)
      * site failure / no drop  → require BG drop > MIN_BG_DROP
      * trivial boluses / meals → dose in [MIN_BOLUS_U, MAX_BOLUS_U]
      * normoglycaemic starts   → require BG ≥ MIN_BG_AT_BOLUS
    Exercise episodes are NOT separately detected here; they surface as
    high-ISF outliers and are absorbed by the GP (and clipped at ISF_MAX).
    """
    required = {"bg", "bolus_dose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"extract_correction_episodes: missing columns {missing}")

    d = df.reset_index(drop=True)
    nadir_steps = NADIR_WINDOW_MIN // _STEP_MIN
    meal_excl_steps = MEAL_EXCLUSION_MIN // _STEP_MIN

    bolus_mask = (d["bolus_dose"] >= MIN_BOLUS_U) & (d["bolus_dose"] <= MAX_BOLUS_U)
    bolus_idx = d.index[bolus_mask].tolist()

    records: list[dict] = []
    for i in bolus_idx:
        dose = float(d.at[i, "bolus_dose"])
        bg = d.at[i, "bg"]
        if pd.isna(bg) or bg < MIN_BG_AT_BOLUS:
            continue

        # Isolation: no other qualifying bolus within ±meal_excl_steps
        lo = max(0, i - meal_excl_steps)
        hi = min(len(d) - 1, i + meal_excl_steps)
        nearby = d.loc[lo:hi, "bolus_dose"].drop(index=i, errors="ignore")
        if (nearby >= MIN_BOLUS_U).any():
            continue

        # Nadir within the forward window (observed BG only)
        future = d.loc[i + 1 : i + nadir_steps, "bg"].dropna()
        if len(future) < 3:
            continue
        nadir_bg = float(future.min())
        bg_drop = float(bg) - nadir_bg
        if bg_drop <= MIN_BG_DROP:
            continue

        isf_est = bg_drop / dose
        rec = {
            "timestamp": d.at[i, "timestamp"] if "timestamp" in d.columns else i,
            "bolus_dose": dose,
            "bg_at_bolus": float(bg),
            "nadir_bg": nadir_bg,
            "bg_drop": bg_drop,
            "isf_estimate": isf_est,
        }
        # State features at the bolus row
        for feat in FEATURE_ORDER:
            if feat == "bg_at_bolus":
                continue  # already set
            rec[feat] = float(d.at[i, feat]) if feat in d.columns and not pd.isna(d.at[i, feat]) else 0.0
        records.append(rec)

    events = pd.DataFrame(records)
    log.info("extract_correction_episodes: %d valid episodes", len(events))
    return events


# ── The model ────────────────────────────────────────────────────────────────

@dataclass
class ISFModel:
    """
    Two-component ISF: hard-derived global baseline × learned local multiplier.

    Fields persisted in the pkl:
      baseline_isf   : the global median ISF (prior / anchor)
      _gp            : fitted GaussianProcessRegressor (None if fit failed / skipped)
      _feat_mean     : feature standardisation centre [n_feat]
      _feat_std      : feature standardisation scale  [n_feat]
      _prior_var     : GP marginal variance sigma0² (for shrink weight)
      n_train        : number of training episodes
      fit_metadata   : dict of diagnostics (CV error, residual std, …)
    """

    baseline_isf: float = 24.0
    _gp: object = field(default=None, repr=False)
    _feat_mean: np.ndarray | None = field(default=None, repr=False)
    _feat_std: np.ndarray | None = field(default=None, repr=False)
    _signal_var: float = 1.0          # ConstantKernel value (explained variance)
    _noise_var: float = 1.0           # WhiteKernel value (irreducible biological noise)
    _blend_strength: float = 1.0      # global [0,1] confidence from CV (0 → pure baseline)
    n_train: int = 0
    fit_metadata: dict = field(default_factory=dict)

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, pump_data_df: pd.DataFrame) -> "ISFModel":
        """
        Fit the local GP on correction episodes extracted from the pump data.

        If too few episodes are available (<MIN_FIT_EPISODES), the GP is left
        unfitted and ``predict`` returns the baseline for every query — a safe
        degenerate case that still satisfies the interface.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        MIN_FIT_EPISODES = 30

        events = extract_correction_episodes(pump_data_df)
        if len(events) < MIN_FIT_EPISODES:
            log.warning(
                "ISFModel.fit: only %d episodes (<%d) — model will return baseline",
                len(events), MIN_FIT_EPISODES,
            )
            self._gp = None
            self.n_train = len(events)
            self.fit_metadata = {"status": "insufficient_data", "n_episodes": len(events)}
            return self

        # If the caller didn't pin a baseline, derive it as the episode median
        # (this is exactly the hard-derived global number).
        if not self.baseline_isf or self.baseline_isf <= 0:
            self.baseline_isf = float(events["isf_estimate"].median())

        X = events[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
        # Target: log-ratio of observed ISF to baseline, clipped to sane range
        ratio = events["isf_estimate"].to_numpy(dtype=np.float64) / self.baseline_isf
        ratio = np.clip(ratio, math.exp(-MAX_ABS_LOG_RATIO * 1.5),
                        math.exp(MAX_ABS_LOG_RATIO * 1.5))
        y = np.log(ratio)

        # Standardise features (GP length scales assume comparable scales)
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0)
        self._feat_std[self._feat_std == 0] = 1.0
        Xs = (X - self._feat_mean) / self._feat_std

        # Kernel: constant * RBF (smooth state dependence) + white noise
        # (episode-to-episode biological variability).  Bounds keep the
        # optimiser away from degenerate fits on small N.
        y_var = float(np.var(y)) or 1e-3
        kernel = (
            ConstantKernel(y_var, (1e-3, 1e2))
            * RBF(length_scale=np.ones(Xs.shape[1]),
                  length_scale_bounds=(1e-1, 1e3))
            + WhiteKernel(noise_level=y_var, noise_level_bounds=(1e-3, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,          # y already ~zero-mean in log space
            n_restarts_optimizer=4,
            alpha=1e-6,
            random_state=0,
        )
        gp.fit(Xs, y)
        self._gp = gp
        self.n_train = len(events)

        # Decompose the learned kernel into explained-signal variance
        # (ConstantKernel, the k1 branch) and irreducible noise variance
        # (WhiteKernel, the k2 branch).  The RBF contributes 1.0 on its own
        # diagonal, so kernel_(x,x) = signal_var + noise_var.
        full_diag = float(gp.kernel_(Xs[:1])[0, 0])
        try:
            self._noise_var = float(gp.kernel_.k2.noise_level)
        except Exception:
            self._noise_var = max(1e-6, 0.5 * full_diag)
        self._signal_var = max(1e-9, full_diag - self._noise_var)

        # Diagnostics: 5-fold CV MAE (model vs baseline) in ISF space.
        self.fit_metadata = self._diagnostics(Xs, y, events)

        # Global blend strength from honest CV: if the model does not beat the
        # baseline out-of-fold, pull predictions back toward baseline.  Maps a
        # CV MAE improvement of 0% → 0 (pure baseline) and ≥20% → 1 (full model),
        # clamped.  This makes the component self-regulating on small data.
        impr = float(self.fit_metadata.get("cv_improvement_pct", 0.0))
        self._blend_strength = float(np.clip(impr / 20.0, 0.0, 1.0))
        self.fit_metadata["blend_strength"] = round(self._blend_strength, 3)
        log.info("ISFModel.fit: n=%d  baseline=%.1f  %s",
                 self.n_train, self.baseline_isf, self.fit_metadata)
        return self

    def _diagnostics(self, Xs: np.ndarray, y: np.ndarray, events: pd.DataFrame) -> dict:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.model_selection import KFold

        meta: dict = {"status": "ok", "n_episodes": int(len(events)),
                      "baseline_isf": round(self.baseline_isf, 2)}
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            errs_model, errs_base = [], []
            for tr, te in kf.split(Xs):
                g = GaussianProcessRegressor(kernel=self._gp.kernel_, alpha=1e-6,
                                             optimizer=None, normalize_y=False)
                g.fit(Xs[tr], y[tr])
                pred = g.predict(Xs[te])
                # ISF-space abs error: model vs baseline-only (pred=0)
                isf_true = self.baseline_isf * np.exp(y[te])
                isf_model = self.baseline_isf * np.exp(pred)
                isf_base = np.full_like(isf_true, self.baseline_isf)
                errs_model.append(np.abs(isf_model - isf_true))
                errs_base.append(np.abs(isf_base - isf_true))
            em = np.concatenate(errs_model)
            eb = np.concatenate(errs_base)
            meta["cv_mae_model"] = round(float(em.mean()), 3)
            meta["cv_mae_baseline"] = round(float(eb.mean()), 3)
            meta["cv_improvement_pct"] = round(100.0 * (eb.mean() - em.mean()) / eb.mean(), 2)
        except Exception as e:  # diagnostics must never break fit
            meta["cv_error"] = str(e)
        return meta

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, features: Mapping[str, float]) -> float:
        """
        Effective ISF (mg/dL/U) for the given current patient state.

        Robust by construction: any missing feature defaults to 0, and any
        internal failure returns the baseline.  Shrinks toward the baseline
        when the GP is uncertain at the query point.
        """
        try:
            if self._gp is None or self._feat_mean is None:
                return float(self.baseline_isf)

            x = np.array(
                [float(self._lookup(features, feat)) for feat in FEATURE_ORDER],
                dtype=np.float64,
            )[None, :]
            xs = (x - self._feat_mean) / self._feat_std

            mu, std = self._gp.predict(xs, return_std=True)
            mu = float(mu[0])
            # Epistemic variance only: sklearn's return_std includes the
            # WhiteKernel noise, which is irreducible biological scatter, not
            # query uncertainty.  Subtract it so the shrink weight reflects how
            # far the query is from training support, not the noise floor.
            epistemic_var = max(0.0, float(std[0]) ** 2 - self._noise_var)

            # Local shrink-to-prior: confident (epistemic→0) uses full mu, OOD
            # (epistemic→signal_var) collapses toward baseline.
            denom = self._signal_var + epistemic_var
            w_local = self._signal_var / denom if denom > 0 else 0.0

            # Global blend strength from CV gates the whole correction.
            w = w_local * self._blend_strength
            log_mult = float(np.clip(w * mu, -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO))

            isf_eff = self.baseline_isf * math.exp(log_mult)
            return float(np.clip(isf_eff, ISF_MIN, ISF_MAX))
        except Exception as e:  # never break the inference path
            log.warning("ISFModel.predict failed (%s) — returning baseline", e)
            return float(self.baseline_isf)

    def predict_multiplier(self, features: Mapping[str, float]) -> float:
        """ISF multiplier relative to baseline (predict / baseline). For display."""
        base = self.baseline_isf if self.baseline_isf else 1.0
        return self.predict(features) / base

    @staticmethod
    def _lookup(features: Mapping[str, float], feat: str) -> float:
        """Resolve a FEATURE_ORDER name from a state dict, honouring the schema alias."""
        if feat in features:
            return features[feat]
        # Reverse the schema alias (bg_at_bolus ← bg)
        for schema_name, alias in _SCHEMA_TO_FEATURE.items():
            if alias == feat and schema_name in features:
                return features[schema_name]
        return 0.0

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "ISFModel":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ISFModel, got {type(obj)}")
        return obj
