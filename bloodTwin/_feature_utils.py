"""Lightweight feature-frame utilities shared across the inference path.

Kept in a separate module so unit tests can import it without pulling in
onnxruntime (which is only available in the .#inference Nix shell).
"""

from __future__ import annotations

import pandas as pd


def ensure_passthrough_features(feature_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Synthesize v1.2+ schema features that patient_state.build_feature_frame never emits.

    hypothetical_dose is a train-time synthetic equal to bolus_dose at every
    step.  build_feature_frame is part of the bloodBath sync stack and predates
    v1.2, so it will never add this column.  Source of truth is always
    bolus_dose from the API-synced merged CSV — never a hardcoded constant.

    Returns a new DataFrame if any column was added; the input is never mutated.
    """
    if "hypothetical_dose" in feature_names and "hypothetical_dose" not in feature_df.columns:
        feature_df = feature_df.copy()
        if "bolus_dose" in feature_df.columns:
            feature_df["hypothetical_dose"] = feature_df["bolus_dose"].fillna(0.0)
        else:
            feature_df["hypothetical_dose"] = 0.0
    return feature_df
