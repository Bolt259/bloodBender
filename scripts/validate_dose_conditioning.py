"""
BT-01 validation: verify dose-conditioned fan (v1.3) produces distinct trajectory shapes
vs the physics fan. If dose-masking is working, different doses should diverge in shape,
not just parallel-shift vertically.

Run from repo root in .#inference shell:
    nix develop .#inference --command python scripts/validate_dose_conditioning.py

Exit codes: 0 = OK, 1 = shapes are parallel shifts (dose-masking not working).
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
ARTIFACT_V13 = REPO / "bloodTwin/artifacts/bloodTwin_unified_lstm_v1_3"
DATA_FILE = (
    Path("/home/lkshply/dev/bloodBender-data-sync/bloodBath/bloodBank/merged")
    / "pump_901161470/test/lstm_test_901161470_20261029_102241.csv"
)
LOOKBACK = 288  # 24h at 5-min resolution
HORIZON = 12   # 60-min forecast

sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")


def load_window(csv_path: Path, window_rows: int = LOOKBACK + 60) -> pd.DataFrame:
    """Load the last `window_rows` rows that have real (non-imputed) BG values."""
    df = pd.read_csv(csv_path, comment="#", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # find a contiguous run of real BG readings at the END of the file
    real = df[df["mask_label"] == True].copy()  # noqa: E712
    if len(real) < window_rows:
        print(f"[warn] only {len(real)} real rows; using all of them")
        return real
    tail = real.iloc[-window_rows:].copy()
    return tail


def shape_divergence(traj0: np.ndarray, traj1: np.ndarray) -> float:
    """
    Measure how much traj1 diverges in SHAPE from traj0 beyond a simple vertical shift.
    A pure parallel shift has divergence ≈ 0; learned dose response has divergence > 0.
    """
    shift = traj1[0] - traj0[0]          # shift at first horizon step
    shifted0 = traj0 + shift             # traj0 translated to match traj1's start
    residual = traj1 - shifted0          # what remains after removing the shift
    return float(np.std(residual))       # shape divergence = std of residual deviation


def main():
    from bloodTwin.inference.conformal import ConformalPredictor
    from bloodTwin.inference.scenarios import ScenarioSimulator, PatientConfig

    print("Loading v1.3 model...")
    predictor = ConformalPredictor(
        onnx_path=str(ARTIFACT_V13 / "model.onnx"),
        scaler_path=str(ARTIFACT_V13 / "scaler.pkl"),
        cal_path=str(ARTIFACT_V13 / "conformal_cal.pkl"),
    )
    print(f"  features ({len(predictor.feature_names)}): {predictor.feature_names}")
    supports = predictor.feature_names and "hypothetical_dose" in predictor.feature_names
    print(f"  supports dose conditioning: {supports}")
    if not supports:
        print("FAIL: model lacks hypothetical_dose feature — v1.3 artifact wrong?")
        sys.exit(1)

    cfg = PatientConfig(isf=24.0, correction_target=110, dose_step=0.5, max_dose=4.0)
    simulator = ScenarioSimulator(predictor, cfg)

    print(f"\nLoading data window from:\n  {DATA_FILE}")
    df = load_window(DATA_FILE)
    print(f"  {len(df)} rows, BG range: {df['bg'].min():.0f}–{df['bg'].max():.0f} mg/dL")

    # --- Dose-conditioned fan ---
    print("\nRunning dose-conditioned fan (0–3U at 0.5U steps)...")
    fan = simulator.dose_conditioned_fan(df, pump_serial="901161470")
    doses = [s.dose_u for s in fan.scenarios]
    trajs = {s.dose_u: s.prediction.predicted_bg for s in fan.scenarios}
    print(f"  doses computed: {doses}")

    # Print trajectory table
    horizons = fan.horizon_minutes
    print(f"\n{'Dose':>6}  {'30min':>8}  {'60min':>8}  {'30min Δ':>9}  {'60min Δ':>9}")
    base30 = float(trajs[0.0][5])
    base60 = float(trajs[0.0][11])
    for dose in doses:
        t = trajs[dose]
        d30 = float(t[5]) - base30
        d60 = float(t[11]) - base60
        print(f"{dose:>6.1f}  {float(t[5]):>8.1f}  {float(t[11]):>8.1f}  {d30:>+9.1f}  {d60:>+9.1f}")

    # --- Shape divergence analysis ---
    print("\n--- Shape divergence vs pure vertical shift ---")
    print(f"{'Dose':>6}  {'StdDev(residual)':>18}  {'Verdict':>30}")
    physics_ok = True
    for dose in doses[1:]:
        div = shape_divergence(trajs[0.0], trajs[dose])
        # A pure physics fan shifts by ISF*IOB ≈ constant per step → std residual ≈ 0
        # A learned fan has time-varying response → std residual > 1 mg/dL
        verdict = "LEARNED (diverges in shape)" if div > 1.0 else "PHYSICS-LIKE (parallel shift)"
        if div <= 1.0:
            physics_ok = False
        print(f"{dose:>6.1f}  {div:>18.3f}  {verdict}")

    # --- Confirm dose ordering is monotone (higher dose → lower BG at 60min) ---
    print("\n--- Dose ordering check (higher dose → lower BG at 60min?) ---")
    bg60 = [float(trajs[d][11]) for d in sorted(doses)]
    is_monotone = all(bg60[i] >= bg60[i+1] for i in range(len(bg60)-1))
    print(f"  60-min BG by dose: {[f'{v:.1f}' for v in bg60]}")
    print(f"  Monotone (higher dose → lower BG): {is_monotone}")

    # --- Physics fan comparison ---
    print("\nRunning physics fan (v1.1 fallback mode) on same window...")
    # Force physics mode by temporarily monkey-patching
    orig = simulator.supports_dose_conditioning.fget
    fan_phys = simulator.run(df, pump_serial="901161470")
    trajs_phys = {s.dose_u: s.prediction.predicted_bg for s in fan_phys.scenarios}
    phys_doses = [s.dose_u for s in fan_phys.scenarios]
    print(f"  physics doses: {phys_doses}")
    if phys_doses and len(phys_doses) > 1:
        div_phys = shape_divergence(trajs_phys[phys_doses[0]], trajs_phys[phys_doses[1]])
        print(f"  physics 1st-pair shape divergence: {div_phys:.3f} (expected ≈ 0)")

    # --- Summary ---
    print("\n" + "="*60)
    if not physics_ok:
        print("RESULT: FAIL — dose-conditioned trajectories are parallel shifts.")
        print("  → 30% dose-masking may be insufficient; consider increasing to 50%.")
        print("  → Or check that dose_conditioned_fan is actually using the v1.3 ONNX.")
        sys.exit(1)
    elif not is_monotone:
        print("RESULT: WARN — trajectories diverge in shape but dose ordering is wrong.")
        print("  → Model learned dose response but direction is inverted? Check bolus sign.")
    else:
        print("RESULT: PASS — dose-conditioned fan shows learned, non-parallel trajectories")
        print("  and correct dose ordering (higher dose → lower BG). BT-01 validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
