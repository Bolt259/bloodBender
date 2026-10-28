"""
Trajectory fan UI — multi-dose BG prediction display.

Two modes:
  --live          Use the most recent 300 rows (future unknown)
  --retrospective Use a window N steps back from the end of the test split
                  and overlay actual BG readings so the prediction can be
                  evaluated against what actually happened.

Output: bloodTwin/analytics/plots/scenario_fan_<pump>_<ts>.png

Usage (inference nix shell, from repo root):
    python bloodTwin/analytics/plot_scenario_fan.py --retrospective --offset 200
    python bloodTwin/analytics/plot_scenario_fan.py --live
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from bloodTwin.inference.conformal import ConformalPredictor, LOOKBACK, HORIZON
from bloodTwin.inference.scenarios import ScenarioSimulator, PatientConfig, ScenarioFan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART          = PROJECT_ROOT / "bloodTwin/artifacts/bloodTwin_unified_lstm"
MERGED_DIR   = PROJECT_ROOT / "bloodBath/bloodBank/merged"
PLOTS_DIR    = PROJECT_ROOT / "bloodTwin/analytics/plots"

HYPO_LINE  = 70.0
HYPER_LINE = 180.0


# ── colour map: light → dark as dose increases ───────────────────────────────
def _dose_color(dose: float, max_dose: float) -> str:
    ratio = dose / max_dose if max_dose > 0 else 0
    # 0U = blue, max = red, interpolated
    r = int(92  + ratio * (220 - 92))
    g = int(122 + ratio * (50  - 122))
    b = int(224 + ratio * (50  - 224))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_isf(pump_id: str) -> float:
    p = PROJECT_ROOT / "bloodTwin/analytics/isf/isf_combined.json"
    if p.exists():
        with open(p) as f:
            return json.load(f).get("current_isf_30d", 24.0)
    return 24.0


def load_test_df(pump_id: str) -> pd.DataFrame:
    d = MERGED_DIR / f"pump_{pump_id}" / "test"
    csv = sorted(d.glob("lstm_test_*.csv"))[-1]
    df = pd.read_csv(csv, comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_window(df: pd.DataFrame, offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Return (window_df, actuals_df).

    window_df : last 300 rows ending at (end - offset) steps
    actuals_df: next HORIZON rows after window end (None if at tail / future unknown)
    """
    end_idx = len(df) - offset
    if end_idx < LOOKBACK + 12:
        raise ValueError(f"offset {offset} leaves fewer than {LOOKBACK+12} rows")

    window   = df.iloc[max(0, end_idx - 300) : end_idx].copy()
    future   = df.iloc[end_idx : end_idx + HORIZON].copy() if end_idx + HORIZON <= len(df) else None
    return window, future


def build_fan(window: pd.DataFrame, pump_id: str, isf: float) -> ScenarioFan:
    predictor = ConformalPredictor(ART/"model.onnx", ART/"scaler.pkl", ART/"conformal_cal.pkl")
    config    = PatientConfig(isf=isf, correction_target=100.0, max_dose=6.0, dose_step=0.25)
    return ScenarioSimulator(predictor, config).run(window, pump_serial=pump_id, coverage=0.90)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_fan(
    fan: ScenarioFan,
    actuals: pd.DataFrame | None,
    pump_id: str,
    retrospective: bool,
) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    # Layout: main trajectory fan (left 75%) + state panel (right 25%)
    ax_main = fig.add_axes([0.06, 0.15, 0.67, 0.72])
    ax_prob = fig.add_axes([0.76, 0.15, 0.20, 0.72])

    # ── Build time axis ───────────────────────────────────────────────────
    t0  = pd.Timestamp(fan.window_end_utc)
    t_x = [t0 + pd.Timedelta(minutes=m) for m in fan.horizon_minutes]

    # ── Safety bands ──────────────────────────────────────────────────────
    ax_main.axhspan(HYPO_LINE,  HYPER_LINE, alpha=0.06, color="green")
    ax_main.axhline(HYPO_LINE,  color="orange", lw=0.8, ls="--", alpha=0.7)
    ax_main.axhline(HYPER_LINE, color="orange", lw=0.8, ls="--", alpha=0.7)
    ax_main.axhline(fan.correction_target, color="#999", lw=0.6, ls=":", alpha=0.5,
                    label=f"Target {fan.correction_target:.0f}")

    # ── Current BG marker ─────────────────────────────────────────────────
    if fan.current_bg is not None:
        ax_main.scatter([t0], [fan.current_bg], s=80, color="#222", zorder=6,
                        label=f"Current BG {fan.current_bg:.0f} mg/dL")

    # ── Scenario curves ───────────────────────────────────────────────────
    max_dose = fan.scenarios[-1].dose_u
    best_dose = fan.best_scenario.dose_u

    # Draw all curves (thin), then overdraw 0U and best thicker
    for s in fan.scenarios:
        col   = _dose_color(s.dose_u, max_dose)
        lw    = 2.2 if s.dose_u in (0.0, best_dose) else 0.7
        alpha = 0.9 if s.dose_u in (0.0, best_dose) else 0.45
        zord  = 4   if s.dose_u in (0.0, best_dose) else 2
        bg    = s.prediction.predicted_bg

        ax_main.plot(t_x, bg, color=col, lw=lw, alpha=alpha, zorder=zord)

        # Shaded CI only for 0U and best
        if s.dose_u == 0.0:
            ax_main.fill_between(t_x, s.prediction.lower, s.prediction.upper,
                                 alpha=0.12, color=col, zorder=1)
            ax_main.annotate("0U (no dose)", xy=(t_x[-1], float(bg[-1])),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=7, color=col, va="center")
        if s.dose_u == best_dose and best_dose > 0:
            ax_main.fill_between(t_x, s.prediction.lower, s.prediction.upper,
                                 alpha=0.12, color=col, zorder=1)
            ax_main.annotate(f"{best_dose:.2f}U (suggested)",
                             xy=(t_x[-1], float(bg[-1])),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=7, color=col, va="center")

    # ── Actuals overlay (retrospective) ───────────────────────────────────
    if actuals is not None and not actuals.empty:
        obs = actuals[actuals["bg"].notna() & ~actuals["mask_bg"].astype(bool)]
        if not obs.empty:
            # Flag rows where a correction bolus occurred — these actuals are
            # post-intervention and will look lower than a true 0U trajectory
            has_bolus = (obs["bolus_dose"].fillna(0) > 0.5) if "bolus_dose" in obs.columns \
                        else pd.Series(False, index=obs.index)
            corrected = obs[has_bolus]
            clean     = obs[~has_bolus]

            ax_main.scatter(clean["timestamp"],     clean["bg"],
                            s=30, color="#111", zorder=7, marker="o",
                            label="Actual BG (no correction)")
            ax_main.plot(obs["timestamp"], obs["bg"], color="#111",
                         lw=1.5, zorder=6, alpha=0.8)
            if not corrected.empty:
                ax_main.scatter(corrected["timestamp"], corrected["bg"],
                                s=60, color="red", zorder=8, marker="^",
                                label="Actual BG (correction bolus given ↑)")
                ax_main.annotate(
                    "⚠ patient corrected — actual BG lower than true 0U trajectory",
                    xy=(0.01, 0.02), xycoords="axes fraction",
                    fontsize=7, color="red", alpha=0.8,
                )

    # ── Colour legend (dose gradient bar) ────────────────────────────────
    cmap_patches = [
        mpatches.Patch(color=_dose_color(d, max_dose), alpha=0.8,
                       label=f"{d:.2f}U" if d in (0.0, best_dose, max_dose / 2, max_dose) else "")
        for d in [s.dose_u for s in fan.scenarios]
    ]
    # Just show corner doses
    legend_patches = [
        mpatches.Patch(color=_dose_color(0.0,      max_dose), label=f"0U (no dose)"),
        mpatches.Patch(color=_dose_color(best_dose, max_dose), label=f"{best_dose:.2f}U (suggested)"),
        mpatches.Patch(color=_dose_color(max_dose,  max_dose), label=f"{max_dose:.2f}U (max)"),
        mpatches.Patch(color="green", alpha=0.3, label="Target range (70–180)"),
    ]
    if actuals is not None:
        legend_patches.append(
            mpatches.Patch(color="#111", label="Actual BG" + (" ✓" if not actuals.empty else ""))
        )
    ax_main.legend(handles=legend_patches, fontsize=8, loc="upper left")

    # ── Formatting ────────────────────────────────────────────────────────
    import matplotlib.dates as mdates
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_main.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 61, 10)))
    plt.setp(ax_main.xaxis.get_majorticklabels(), fontsize=8)
    ax_main.set_xlabel("Time (UTC)", fontsize=9)
    ax_main.set_ylabel("Blood Glucose (mg/dL)", fontsize=9)
    ax_main.set_ylim(40, 360)
    ax_main.grid(axis="y", alpha=0.25)

    mode_str = "retrospective" if retrospective else "live"
    title_lines = [
        f"bloodTwin trajectory fan  —  pump {pump_id}  [{mode_str}]  "
        f"window end {t0.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Current BG: {fan.current_bg or '?'} mg/dL  |  "
        f"Model predicts peak: {fan.model_predicted_peak:.0f} mg/dL  |  "
        f"ECD: {fan.estimated_correction_dose:.2f}U  |  "
        f"Suggested: {best_dose:.2f}U  |  ISF: {fan.isf_used:.1f} mg/dL/U",
        f"Model insight: {fan.model_trigger}",
    ]
    ax_main.set_title("\n".join(title_lines), fontsize=8.5, loc="left", pad=6)

    # ── State probability panel ───────────────────────────────────────────
    states = ["post_meal_rise", "dawn_phenomenon", "site_failure",
              "exercise_sensitive", "nominal"]
    state_labels = ["Post-meal rise", "Dawn phenomenon", "Site failure",
                    "Exercise sensitive", "Nominal"]
    state_colors = ["#e09a5c", "#5c7ae0", "#e05c5c", "#5ce0a8", "#aaa"]
    # Use meal_probability as a proxy for post_meal; others unknown without classifier
    probs = [fan.meal_probability, 0.0, 0.0, 0.0,
             max(0.0, 1.0 - fan.meal_probability)]
    # If failure_top_state is set by caller, reflect it
    if fan.failure_top_state != "nominal" and fan.failure_top_prob > 0.1:
        idx = states.index(fan.failure_top_state) if fan.failure_top_state in states else -1
        if idx >= 0:
            probs[idx] = fan.failure_top_prob
            probs[-1]  = max(0.0, 1.0 - sum(probs[:-1]))

    ax_prob.barh(state_labels[::-1], probs[::-1],
                 color=state_colors[::-1], alpha=0.75, edgecolor="none")
    for i, (prob, label) in enumerate(zip(probs[::-1], state_labels[::-1])):
        ax_prob.text(min(prob + 0.02, 0.95), i, f"{prob:.0%}",
                     va="center", fontsize=8)
    ax_prob.set_xlim(0, 1.1)
    ax_prob.set_xlabel("Probability", fontsize=8)
    ax_prob.set_title("Patient state\n(model-estimated)", fontsize=8.5)
    ax_prob.axvline(0.5, color="#ccc", lw=0.7, ls="--")
    ax_prob.grid(axis="x", alpha=0.2)

    # ── Dose colour gradient bar at bottom ───────────────────────────────
    ax_grad = fig.add_axes([0.06, 0.05, 0.67, 0.025])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_grad.imshow(grad, aspect="auto", cmap=plt.cm.coolwarm_r)
    ax_grad.set_xticks([0, 64, 128, 192, 255])
    ax_grad.set_xticklabels(
        [f"0U", f"{max_dose*0.25:.2f}U", f"{max_dose*0.5:.2f}U",
         f"{max_dose*0.75:.2f}U", f"{max_dose:.2f}U"], fontsize=7
    )
    ax_grad.set_yticks([])
    ax_grad.set_title("Dose scenarios", fontsize=7, pad=2)

    ts_str = t0.strftime("%Y%m%d_%H%M")
    out = PLOTS_DIR / f"scenario_fan_{pump_id}_{ts_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ── Retrospective summary ─────────────────────────────────────────────────────

def retrospective_summary(fan: ScenarioFan, actuals: pd.DataFrame | None) -> None:
    if actuals is None or actuals.empty:
        print("No actuals available to verify (window is at or near the end of data).")
        return
    obs = actuals[actuals["bg"].notna() & ~actuals["mask_bg"].astype(bool)]
    if obs.empty:
        print("All future rows are imputed or missing — cannot verify.")
        return

    baseline_preds = fan.baseline.prediction.predicted_bg
    actual_bg      = obs["bg"].values
    actual_ts      = obs["timestamp"].values
    n = min(len(actual_bg), HORIZON)

    print(f"\nRetrospective verification  (window end: {fan.window_end_utc})")
    print(f"Current BG at prediction time: {fan.current_bg or '?'} mg/dL")
    print(f"Model trigger: \"{fan.model_trigger}\"")
    print()
    bolus_col = obs.get("bolus_dose", pd.Series(0.0, index=obs.index)) \
                   if hasattr(obs, "get") else pd.Series(0.0)

    print(f"  {'min':>5}  {'predicted':>10}  {'actual':>8}  {'error':>8}  {'in 90% CI':>10}  note")
    for i in range(n):
        pred  = float(baseline_preds[i])
        act   = float(actual_bg[i])
        err   = act - pred
        lo    = float(fan.baseline.prediction.lower[i])
        hi    = float(fan.baseline.prediction.upper[i])
        in_ci = "✓" if lo <= act <= hi else "✗"
        ts    = pd.Timestamp(actual_ts[i]).strftime("%H:%M")
        # Flag if a correction bolus occurred at or before this step
        note = ""
        if i < len(obs) and "bolus_dose" in obs.columns:
            step_bolus = float(obs["bolus_dose"].iloc[i]) if i < len(obs) else 0.0
            if step_bolus > 0.5:
                note = f"  ← patient gave {step_bolus:.1f}U (actual reflects correction)"
        print(f"  {fan.horizon_minutes[i]:>5}  {pred:>10.1f}  {act:>8.1f}  {err:>+8.1f}  {in_ci:>10}  @{ts}{note}")

    actual_peak = float(actual_bg[:n].max())
    pred_peak   = float(max(baseline_preds[:n]))
    print(f"\n  Actual peak:    {actual_peak:.1f} mg/dL")
    print(f"  Predicted peak: {pred_peak:.1f} mg/dL")
    print(f"  Peak error:     {actual_peak - pred_peak:+.1f} mg/dL")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pump",          default="1539514")
    ap.add_argument("--live",          action="store_true",
                    help="Use the most recent window (default)")
    ap.add_argument("--retrospective", action="store_true",
                    help="Look back into test split and verify against actuals")
    ap.add_argument("--offset",        type=int, default=144,
                    help="Steps back from end for retrospective (default 144 = 12h)")
    args = ap.parse_args()

    isf = load_isf(args.pump)
    print(f"ISF loaded: {isf:.1f} mg/dL/U")

    df = load_test_df(args.pump)
    print(f"Test split: {len(df)} rows  [{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}]")

    if args.retrospective:
        offset = args.offset
        print(f"Retrospective mode: looking back {offset} steps ({offset*5} min)")
    else:
        offset = 0
        print("Live mode: using most recent window")

    window, actuals = get_window(df, offset=offset)
    print(f"Window: {window['timestamp'].iloc[0]} → {window['timestamp'].iloc[-1]}")
    if actuals is not None:
        obs = actuals[actuals["bg"].notna()]
        print(f"Actuals available: {len(obs)} observed BG readings over next 60 min")
    else:
        print("No actuals available (future data)")

    print("Building scenario fan...")
    fan = build_fan(window, args.pump, isf)
    print(f"Fan: {len(fan.scenarios)} scenarios, 0–{fan.scenarios[-1].dose_u}U")
    print(f"Model trigger: \"{fan.model_trigger}\"")
    print(f"ECD: {fan.estimated_correction_dose:.2f}U  |  Best: {fan.best_scenario.dose_u:.2f}U")

    if args.retrospective and actuals is not None:
        retrospective_summary(fan, actuals)

    out = plot_fan(fan, actuals, args.pump, retrospective=args.retrospective)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
