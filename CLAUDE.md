# bloodBender — AI Session Context

Auto-loaded by Claude Code. Read this before anything else.

## Two-Directory Architecture

The project lives in **two separate directories**:

| Path | Role |
|------|------|
| `/home/lkshply/dev/bloodBender` | Source repo (this repo). Code changes go here. |
| `/home/lkshply/dev/bloodBender-data-sync` | Runtime data directory. Systemd services run from here, with `BLOODBENDER_ROOT=/home/lkshply/dev/bloodBender-data-sync`. Raw sync data and merged LSTM CSVs live here. |

The `bloodBender-data-sync` dir is a separate git repo (older snapshot of the code). The running nix services (`bloodbender-sync`, `bloodbender-inference-cron`) use `PYTHONPATH=/home/lkshply/dev/bloodBender-data-sync`, so they execute source code from data-sync, not from this repo. To push a source change into production: patch `bloodBender-data-sync` manually and restart services via `nix run .#install-services`.

**Important shell hygiene**: The devshell hook sets `PYTHONPATH=$PWD`. Always run nix shell commands from `/home/lkshply/dev/bloodBender` (never from data-sync), or tests and imports will load data-sync source instead of this repo.

`bloodBender/bloodBath/bloodBank/merged` is a symlink → `bloodBender-data-sync/bloodBath/bloodBank/merged`. This lets training scripts find LSTM CSVs at their expected relative path.

## What This Project Is

Personal diabetes management system. Single patient (lkshply). **No dosing commands — display and alert only.**

- **bloodBath** — data sync subsystem: polls Tandem t:connect API v3.0.0, extracts features, resamples to 5-min LSTM windows
- **bloodTwin** — ML subsystem: LSTM training, ONNX inference, conformal prediction, web dashboard
- **bareMetalBender** — future embedded Rust/C++ port (not started)

## Critical Security Rules

- `.env` lives at `/home/lkshply/dev/bloodBender/.env` — contains real Tandem credentials. **Never push, never log, never expose outside the machine.**
- `.env.example` must only contain placeholder credentials. Always verify before committing.
- The system never sends commands to the pump. Inference output is display/alert only.

## Pump Hardware (all same patient)

| Serial    | Model         | Notes                                      |
|-----------|---------------|--------------------------------------------|
| 881235    | t:slim X2     | Primary, has 139 site_change events        |
| 901161470 | t:slim X2     | Secondary                                  |
| 1539514   | Tandem Mobi   | Had ~41-day inactive gap in 2024-2025      |

Default sync pulls all three. Feature `site_change` only populated from 881235 so far.

## Nix Shells

```bash
nix develop .#default    # bloodBath + bloodTwin dev (no GPU torch)
nix develop .#inference  # ONNX inference + matplotlib (CPU only, no bloodBath)
nix develop .#cuda       # GPU training (RTX 2070 SUPER, 15 GiB RAM)
nix develop .#cpp        # bareMetalBender C++ work
```

**Important:** `.#inference` does NOT have `arrow` or other bloodBath deps. If you need to import bloodBath from an inference script, use `.#default`.

## Nix Apps (one-shot commands)

```bash
nix run .#sync-data               # manual sync (all pumps, delta update)
nix run .#normal-full-sync        # full re-sync from scratch
nix run .#run-inference           # single inference pass → latest.json
nix run .#bloodbender-calibrate   # conformal recalibration (monthly cron)
nix run .#bloodbender-fit-isf     # ISF re-estimation (weekly cron)
nix run .#install-services        # install systemd user timers
```

Retrain is manual (`nix develop .#cuda`, then run the training script directly) — it takes 20+ min and requires the GPU shell.

## Active Systemd Timers

```
bloodbender-sync.timer        15-min sync
bloodbender-inference.timer    5-min inference → latest.json
bloodbender-calibrate.timer   monthly conformal recalibration
bloodbender-fit-isf.timer     weekly ISF re-estimation
```

Manage with: `systemctl --user {status,start,stop} bloodbender-<name>.timer`

## Active Model Artifact

```
bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3/
  model.onnx              ONNX LSTM (15 features, v1.3)
  scaler.pkl              RobustScaler as numpy arrays (center_, scale_)
  conformal_summary.json  90% coverage, ±31@30min ±51@60min (14,075 cal windows)
  test_results.yaml       MAE 30min=9.88, 60min=15.27, RMSE=22.65, meal_acc=78.7%
```

Controlled by env var `BLOODTWIN_ARTIFACTS` (defaults to the path above).

## Feature Schema v1.3 (15 features, order matters)

```
bg, basal_rate, bolus_dose, basal_delta, time_since_last_bolus,
bg_slope_15min, bg_slope_30min, sin_time, cos_time, iob,
bg_acceleration, mask_bg, mask_label, hypothetical_dose, site_change
```

Canonical derivation: `bloodBath/core/patient_state.py → build_feature_frame()`

## Key Source Files

| Path | Purpose |
|------|---------|
| `bloodBath/core/patient_state.py` | Feature derivation + `build_feature_frame()` |
| `bloodBath/data/resampler.py` | 5-min LSTM window assembly |
| `bloodBath/data/extractors.py` | Pump event extractors (site_change fix at line 298) |
| `bloodBath/core/config.py` | Pump serial registry |
| `bloodTwin/inference/conformal.py` | ConformalPredictor (run → scale → ONNX → q_hat bands) |
| `bloodTwin/inference/scenarios.py` | ScenarioSimulator — `fan()` dispatches physics fan |
| `bloodTwin/inference/service.py` | Inference entry point; `ARTIFACTS_DIR` config |
| `bloodTwin/dashboard/server.py` | Flask dashboard at http://127.0.0.1:8765 |
| `bloodTwin/dashboard/charts.py` | matplotlib chart rendering |

## Known Gotchas

**pandas 2.x index alignment** — `derive_time_of_day()` returns a DatetimeIndex Series. Any DataFrame with an integer index (like `build_feature_frame`'s `out`) must use `.values` when assigning the result, or ALL rows become NaN. This bug was silently present in all inference from day one; fixed 2026-10-31 (commit `cc941a6e`).

**Training vs inference index mismatch** — the resampler (`resampler.py`) uses DataFrames with DatetimeIndex so the assignment worked fine there. Only inference (`patient_state.py`) was affected.

**Dose-conditioned fan** — `ScenarioSimulator.dose_conditioned_fan()` exists but produces wrong direction (+1 mg/dL/U instead of −24). `fan()` is wired to the physics fan (correct). See TICKETS.md BT-06 for redesign plan.

**Conformal calibration** — was 74.9% before recalibration in bb3 session. Now at 90% target. If you see low coverage, rerun `nix run .#bloodbender-calibrate`.

## Branch Strategy

```
main (always deployable)
 └─ feat/<name>     new features
 └─ fix/<name>      bug fixes
 └─ arch/<name>     architectural changes (larger scope)
 └─ refactor/<name> refactors
```

`cleanup/deprecated-modules-and-nix-migration` is a stale branch (behind main, safe to delete). `feat/multi-horizon-control` and `refactor/retire-sweetblood-cli` are planned but not started.

## Open Work

See `TICKETS.md` for the full prioritized list. Top items:

- **BT-02** (P1) — audit 41-day gap in pump 1539514 training windows
- **BB-02** (P1) — Tandem API field audit against real BFF JSON
- **IN-01** (P1) — hypoglycemia push notification (ntfy.sh)
- **BT-06** (P3) — dose-conditioned LSTM redesign (Option C: IOB injection, no retrain)

## Tests

```bash
nix develop .#default --command pytest bloodBath/tests/ -q      # 137 tests
nix develop .#default --command pytest bloodTwin/tests/ -q      # 23 tests
```

160 total, all passing as of commit `cc941a6e`.
