#!/usr/bin/env bash
# Run Blood Bender LSTM inference on latest data and emit prediction + decision-support correction estimate
#
# Prereqs:
# - Env: /home/bolt/projects/bb/bloodBath-env
# - Repo root: /home/bolt/projects/bb
# - Artifacts: TorchScript model (.ts) + scaler.pkl under artifacts/{pump_id}/
# - Data: monthly LSTM CSVs under either bloodBath/bloodBank/merged or training_data_legacy/monthly_lstm

set -euo pipefail

# 0) Activate environment and set project paths
source /home/bolt/projects/bb/bloodBath-env/bin/activate
cd /home/bolt/projects/bb
export PYTHONPATH=/home/bolt/projects/bb

# 1) Select pump and locate latest inputs/artifacts
PUMP_ID="${PUMP_ID:-901161470}"   # override with: PUMP_ID=881235 ./run_lstm_inference.sh
ART_ROOT_CANDIDATES=(
  "bloodBath/sweetBlood/artifacts"
  "bloodTwin/artifacts"
  "artifacts"
  "bloodBath/artifacts"
)
DATA_ROOT_CANDIDATES=(
  "bloodBath/bloodBank/merged"
  "bloodBath/training_data"
  "training_data_legacy/monthly_lstm"
)

# Resolve artifacts dir and files (prefer TorchScript)
# Allow explicit overrides via env: ART_DIR, TS_PATH, SCALER_PATH
if [[ -n "${ART_DIR:-}" && -d "${ART_DIR}" ]]; then
  : # use provided ART_DIR
else
  ART_DIR=""
  for d in "${ART_ROOT_CANDIDATES[@]}"; do
    [[ -d "$d/$PUMP_ID" ]] && ART_DIR="$d/$PUMP_ID" && break
  done
fi
if [[ -z "${ART_DIR}" ]]; then
  echo "ERR: No artifacts directory found. Set ART_DIR or use a known location. Searched with pump ${PUMP_ID} under: ${ART_ROOT_CANDIDATES[*]}" >&2
  exit 2
fi

# Only discover files if not explicitly set in env
if [[ -z "${TS_PATH:-}" ]]; then
  TS_PATH="$(ls -1t "${ART_DIR}"/*.ts 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${ONNX_PATH:-}" ]]; then
  ONNX_PATH="$(ls -1t "${ART_DIR}"/*.onnx 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${CKPT_PATH:-}" ]]; then
  CKPT_PATH="$(ls -1t "${ART_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${SCALER_PATH:-}" ]]; then
  SCALER_PATH="$(ls -1t "${ART_DIR}"/scaler*.pkl 2>/dev/null | head -n1 || true)"
fi

# Require TorchScript for this runner (ONNX not used in current Python block)
if [[ -z "${TS_PATH}" ]]; then
  echo "ERR: TorchScript model (.ts) not found. Provide TS_PATH or place a .ts file under ${ART_DIR}" >&2
  exit 3
fi
if [[ -z "${SCALER_PATH}" ]]; then
  echo "ERR: No scaler*.pkl found. Provide SCALER_PATH or place it under ${ART_DIR}" >&2
  exit 4
fi

# Resolve latest monthly CSV (pick the first root that has CSVs for this pump)
# Allow explicit override via CSV_PATH
if [[ -n "${CSV_PATH:-}" && -f "${CSV_PATH}" ]]; then
  LATEST_CSV="${CSV_PATH}"
  DATA_DIR="$(dirname "${CSV_PATH}")"
else
  DATA_DIR=""
  LATEST_CSV=""
  for d in "${DATA_ROOT_CANDIDATES[@]}"; do
    if [[ -d "$d/pump_${PUMP_ID}" ]]; then
      cand="$(ls -1t "$d/pump_${PUMP_ID}"/*.csv 2>/dev/null | head -n1 || true)"
      if [[ -n "$cand" ]]; then
        DATA_DIR="$d"
        LATEST_CSV="$cand"
        break
      fi
    fi
  done
  if [[ -z "${DATA_DIR}" || -z "${LATEST_CSV}" ]]; then
    echo "ERR: No CSVs found for pump ${PUMP_ID}. Searched roots: ${DATA_ROOT_CANDIDATES[*]}" >&2
    echo "Provide CSV_PATH to specify an explicit file." >&2
    exit 6
  fi
fi

# Export for Python subprocess
export PUMP_ID ART_DIR TS_PATH ONNX_PATH CKPT_PATH SCALER_PATH LATEST_CSV
export LOOKBACK="${LOOKBACK:-288}"
export HORIZON="${HORIZON:-12}"
export FEATURE_NAMES="${FEATURE_NAMES:-bg,delta_bg,basal_rate_clipped,basal_anomaly_flag,sin_time,cos_time,bg_trend_30min,insulin_on_board}"

echo "Using:"
echo "  Pump:       ${PUMP_ID}"
echo "  Model dir:  ${ART_DIR}"
echo "  Model ts:   ${TS_PATH:-none}  (onnx: ${ONNX_PATH:-none}, ckpt: ${CKPT_PATH:-none})"
echo "  Scaler:     ${SCALER_PATH}"
echo "  Latest CSV: ${LATEST_CSV}"

# 2) Run inference on the last lookback window of the latest CSV.
#    Assumptions:
#    - lookback=288 (24h @5min), horizon=12 (60min)
#    - scaler.pkl stores feature_names used during training
#    - CSV contains those features (e.g., bg, delta_bg, basal_rate_clipped, basal_anomaly_flag, sin_time, cos_time, bg_trend_30min, insulin_on_board)
#    - Missing feature values are forward-filled, remaining NaNs -> 0 for non-BG features; BG NaNs excluded from target
#    - Outputs a JSON with predictions and a correction estimate (decision-support only; not for clinical use)

python - <<'PY'
import os, json, pickle, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

import torch

PUMP_ID = os.environ.get("PUMP_ID", "901161470")
ART_DIR = Path(os.environ.get("ART_DIR", "")) or None
TS_PATH = os.environ.get("TS_PATH") or ""
ONNX_PATH = os.environ.get("ONNX_PATH") or ""
CKPT_PATH = os.environ.get("CKPT_PATH") or ""
SCALER_PATH = os.environ.get("SCALER_PATH") or ""
LATEST_CSV = os.environ.get("LATEST_CSV") or ""
LOOKBACK = int(os.environ.get("LOOKBACK", "288"))
HORIZON = int(os.environ.get("HORIZON", "12"))

# Load scaler to get robust parameters
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Handle both dict-style and sklearn RobustScaler objects
if hasattr(scaler, "get"):  # dict-like
    feature_names = scaler.get("feature_names") or scaler.get("features") or []
    center = np.asarray(scaler.get("center_"))
    scale = np.asarray(scaler.get("scale_"))
else:  # sklearn object
    center = np.asarray(scaler.center_)
    scale = np.asarray(scaler.scale_)
    # Get feature names from env (comma-separated)
    feature_names = os.environ.get("FEATURE_NAMES", "").split(",")
    feature_names = [f.strip() for f in feature_names if f.strip()]

if not feature_names or center.size == 0 or scale.size == 0:
    raise RuntimeError("Scaler missing required data. Provide FEATURE_NAMES env var (comma-separated list).")
if len(feature_names) != center.size:
    raise RuntimeError(f"Feature count mismatch: {len(feature_names)} names vs {center.size} scaler dimensions")

# Load latest CSV
df = pd.read_csv(LATEST_CSV, comment="#")
# Normalize time index
time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
df = df.sort_values(time_col).reset_index(drop=True)

# Engineer missing features with sensible defaults
# Map basal_rate -> basal_rate_clipped if needed
if "basal_rate" in df.columns and "basal_rate_clipped" not in df.columns:
    df["basal_rate_clipped"] = df["basal_rate"].clip(0, 25)  # typical max basal ~25U/hr
# Default flags
if "basal_anomaly_flag" not in df.columns:
    df["basal_anomaly_flag"] = 0
# Compute bg_trend_30min (BG change over last 30 min = 6 rows @5min)
if "bg_trend_30min" not in df.columns:
    df["bg_trend_30min"] = df["bg"].diff(periods=6).fillna(0.0)
# Simple insulin_on_board estimate (demo only; real IOB requires pharmacokinetics model)
if "insulin_on_board" not in df.columns:
    if "bolus_dose" in df.columns:
        # Exponential decay with ~4hr half-life (DIA ~6hr): decay constant k ≈ 0.173/hr = 0.0144/5min
        df["insulin_on_board"] = df["bolus_dose"].rolling(window=72, min_periods=1).apply(
            lambda x: sum(x.iloc[i] * np.exp(-0.0144 * (len(x) - i)) for i in range(len(x))), raw=False
        )
    else:
        df["insulin_on_board"] = 0.0

# Select and sanitize features
missing_feats = [c for c in feature_names if c not in df.columns]
if missing_feats:
    raise RuntimeError(f"CSV missing required features after engineering: {missing_feats}. Available: {df.columns.tolist()}")

feat_df = df[feature_names].copy()
# Forward-fill reasonable signals; replace residual NaNs with 0 for non-BG features
feat_df = feat_df.ffill().fillna(0.0)

# Build last lookback window
if len(feat_df) < LOOKBACK:
    raise RuntimeError(f"Not enough rows ({len(feat_df)}) for lookback={LOOKBACK}")
X_win = feat_df.iloc[-LOOKBACK:].to_numpy(dtype=np.float32)

# Robust scale (ensure float32 throughout)
X_scaled = (X_win - center.astype(np.float32)) / np.where(scale == 0, 1.0, scale.astype(np.float32))
X = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0)  # [1, T, F]

# Use CPU to avoid device mismatch issues in exported models
device = "cpu"

# Load model (prefer TorchScript)
model = None
if TS_PATH:
    model = torch.jit.load(TS_PATH, map_location=device)
else:
    # Optional: load Lightning checkpoint (requires model class in PYTHONPATH). Skipped here for simplicity.
    raise RuntimeError("TorchScript model not found; please export .ts")

model.eval()

with torch.inference_mode():
    y_hat = model(X.to(device))  # expected [1, H]

y_hat = y_hat.detach().float().cpu().numpy().reshape(-1).tolist()

# Correction estimate (Decision Support ONLY; not medical advice)
# You MUST supply personalized parameters: ISF (mg/dL per unit insulin), IOB (active insulin), target BG.
# This section is for research/demo. DO NOT use for real-world dosing.
TARGET_BG = float(os.environ.get("TARGET_BG", "110"))   # mg/dL
ISF = float(os.environ.get("ISF", "40"))                # mg/dL per unit
IOB = float(os.environ.get("IOB", "0.0"))               # units
MAX_CORR = float(os.environ.get("MAX_CORR", "5.0"))     # units cap
pred_h60 = y_hat[min(len(y_hat)-1, 11)]                   # ~60min horizon

delta = pred_h60 - TARGET_BG
raw_correction_u = max(0.0, delta / ISF)                  # only correct if above target
suggested_u = max(0.0, raw_correction_u - IOB)
suggested_u = min(suggested_u, MAX_CORR)

out = {
  "pump_id": PUMP_ID,
  "csv_path": LATEST_CSV,
  "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
  "lookback": LOOKBACK,
  "horizon": HORIZON,
  "feature_names": feature_names,
  "pred_bg_series_mgdl": y_hat,
  "decision_support": {
    "target_bg": TARGET_BG,
    "pred_bg_at_60min": pred_h60,
    "delta_from_target": delta,
    "isf_mgdl_per_unit": ISF,
    "iob_units": IOB,
    "max_correction_cap_units": MAX_CORR,
    "suggested_correction_units": suggested_u,
    "disclaimer": "For research/demo only. Not medical advice. Requires clinician-approved settings and safety layers."
  }
}
print(json.dumps(out, indent=2))
# Also write to analytics log
analytics = Path("analytics/lstm_metrics"); analytics.mkdir(parents=True, exist_ok=True)
out_path = analytics / f"infer_{PUMP_ID}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote: {out_path}")
PY
