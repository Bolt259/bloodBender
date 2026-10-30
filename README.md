# bloodBender

A data pipeline and machine learning system for blood glucose prediction from Tandem insulin pump data. The end goal is a continuously self-training model that produces control values for closed-loop insulin delivery, targeted at both cloud/local training and embedded inference.

The repository is the `bloodBender` project; the core Python package is named `bloodBath`.

## Modules

| Module | Language | Role |
| --- | --- | --- |
| `bloodBath` | Python | Pull, clean, validate, and resample Tandem t:connect data into ML-ready datasets |
| `bloodTwin` | Python / PyTorch | LSTM model that forecasts blood glucose 60 minutes ahead |
| `bareMetalBender` | C++ | Glucose-insulin dynamics solver for physics-based validation and the embedded inference path |

Data flows in one direction: `bloodBath` produces datasets, `bloodTwin` trains on them, and `bareMetalBender` provides a physics model used both to validate predictions and as the foundation for on-device control.

```
Tandem API -> bloodBath -> bloodBank (CSV) -> bloodTwin (LSTM) -> exported model
                                  \-> bareMetalBender (dynamics solver / embedded target)
```

## bloodBath

Fetches pump events from the Tandem t:connect API, extracts CGM/basal/bolus streams, resamples to a 5-minute grid, validates, and writes versioned CSV with metadata headers.

Key behavior:

- Preserves `NaN` for missing BG instead of synthetic fills, flagged via `bg_missing_flag`.
- Clips BG to `[20, 600]` mg/dL, flagged via `bg_clip_flag`.
- Adds `delta_bg` and `sin_time` / `cos_time` temporal features.
- Splits chronologically (70/15/15 train/val/test) to avoid leakage.
- Handles multiple pump serials across device transitions.

Layout:

```
bloodBath/
  api/         t:connect authentication and event fetching
  core/        client, config (all constants live in config.py), exceptions
  data/        extractors, processors, validators, repair
  io/          CSV reader/writer (metadata headers)
  validation/  integrity checks and test framework
  utils/       env, time, structure, logging helpers
  cli/         command-line interface
  bloodBank/   on-disk data store (raw / merged / lstm_pump_data / metadata)
  spec/        design specification (single source of truth for constants)
```

CLI:

```bash
python -m bloodBath status
python -m bloodBath sync --pump-serial 881235 --start-date 2024-01-01
python -m bloodBath validate --pump-serial all
```

## bloodTwin

PyTorch Lightning LSTM for 60-minute glucose forecasting.

- Input: 8 features over a 288-step (24h) lookback window.
- Output: 12-step (60 min) horizon.
- Architecture: 2-layer LSTM, 128 hidden units, dropout 0.2, feedforward decoder.
- Loss: MAE. Optimizer: Adam. Mixed precision (16-bit).
- Targets: MAE < 15 mg/dL at 30 min, < 20 mg/dL at 60 min, RMSE < 25 mg/dL.
- Exports: TorchScript (`.ts`), ONNX (`.onnx`), and a fitted scaler (`.pkl`).

```bash
python bloodTwin/pipelines/train_lstm.py --config bloodTwin/configs/lstm.yaml
tensorboard --logdir bloodTwin/analytics/tensorboard_logs
```

## bareMetalBender

A C++ initial-value-problem solver for glucose-insulin dynamics. It serves two purposes: a physics-based check on learned predictions, and the deployment path for embedded, real-time inference where a Python runtime is not viable.

```bash
cd bareMetalBender && make && ./ivp
python plot_data.py
```

## Environment

The project uses a Nix flake for a reproducible toolchain (Python, PyTorch/CUDA, and the C++ build environment).

```bash
nix develop            # full environment
nix develop .#python   # Python only, no CUDA
nix develop .#cpp      # C++ build tools only
```

With direnv, `direnv allow` activates the environment automatically on entry.

Credentials and runtime configuration live in `.env` (copy from `.env.example`):

```bash
TCONNECT_EMAIL=
TCONNECT_PASSWORD=
TCONNECT_REGION=US
PUMP_SERIAL_NUMBER=
TIMEZONE_NAME=America/Los_Angeles
BLOODBATH_OUTPUT_DIR=./bloodBath/bloodBank
```

## Data format

CSV files carry a comment header followed by a 5-minute time series:

```csv
# bloodBath v2.0 CSV Data File
# Pump Serial: 881235
# Date Range: 2021-10-22 to 2022-10-22
# BG Range: [20, 600] mg/dL
time,bg,basal,bolus,bg_missing_flag,bg_clip_flag
2021-10-22 00:00:00+00:00,120.0,0.85,0.0,False,False
2021-10-22 00:05:00+00:00,NaN,0.85,0.0,True,False
```

## Testing

```bash
python -m pytest bloodBath/test_scripts/
python bloodTwin/smoke_test.py
cd bareMetalBender && make clean && make
```

## Documentation

- `bloodBath/spec/bloodBath_Design_Specification_v2.0.md` — technical specification and constants
- `CODEBASE_CONTEXT.md` — full system overview
- `bloodTwin/README.md`, `bareMetalBender/README.md` — module detail

## Status

Senior project for diabetes prediction research. The data pipeline and a unified LSTM are functional; the continuously self-training control model and embedded deployment are in progress.

Based on `tconnectsync` by jwoglom (modified).
