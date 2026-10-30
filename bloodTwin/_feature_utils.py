"""Lightweight feature-frame utilities shared across the inference path.

Kept in a separate module so unit tests can import it without pulling in
onnxruntime (which is only available in the .#inference Nix shell).
"""

from __future__ import annotations

import pandas as pd


def ensure_passthrough_features(
    feature_df: pd.DataFrame,
    feature_names: list[str],
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Synthesize schema features that patient_state.build_feature_frame never emits.

    build_feature_frame (bloodBath) predates v1.2 and v1.3 features.  Source of
    truth for all synthesised values is the API-synced merged CSV (raw_df) or a
    computed derivative of it — never a hardcoded constant except as a safe
    fallback when the column is genuinely absent from the source data.

    raw_df — the original window DataFrame passed to build_feature_frame.
             Needed to copy columns that build_feature_frame silently drops
             (e.g. site_change).  Optional: callers that don't pass it fall
             back to 0.0 defaults.

    Returns a new DataFrame if any column was added; the input is never mutated.
    """
    # ── hypothetical_dose (v1.2) ──────────────────────────────────────────────
    # Train-time synthetic: equals bolus_dose at every step.
    if "hypothetical_dose" in feature_names and "hypothetical_dose" not in feature_df.columns:
        feature_df = feature_df.copy()
        src = feature_df if "bolus_dose" in feature_df.columns else raw_df
        if src is not None and "bolus_dose" in src.columns:
            vals = src["bolus_dose"].fillna(0.0).values[-len(feature_df):]
            feature_df["hypothetical_dose"] = vals if len(vals) == len(feature_df) else 0.0
        else:
            feature_df["hypothetical_dose"] = 0.0

    # ── site_change (v1.3) ────────────────────────────────────────────────────
    # Binary flag from LID_CANNULA_FILLED events, present in merged CSVs but
    # not computed by build_feature_frame.  Copy from raw_df; default 0.0 when
    # the window predates site_change extraction (old merged CSVs).
    if "site_change" in feature_names and "site_change" not in feature_df.columns:
        if feature_df is not feature_df:  # ensure we have a copy
            feature_df = feature_df.copy()
        else:
            feature_df = feature_df.copy()
        if raw_df is not None and "site_change" in raw_df.columns:
            vals = raw_df["site_change"].fillna(0.0).values[-len(feature_df):]
            feature_df["site_change"] = vals if len(vals) == len(feature_df) else 0.0
        else:
            feature_df["site_change"] = 0.0

    return feature_df
