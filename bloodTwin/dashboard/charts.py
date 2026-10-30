"""
SVG chart generators — pure static SVG, no inline <script>.
All interactivity lives in templates/index.html.j2.

  render_fan_svg(fan)                    — 60-min dose scenario fan
  render_history_svg(df)                 — 24h BG history (scrollable)
  render_diagnostic_svg(df, archive)     — past prediction ghost curves

Target bands:
  Personal 75–130  green   (#14532d / #22c55e)
  Clinical 70–180  purple  (#2d1b4e / #a855f7)
"""
from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd

CLINICAL_LO, CLINICAL_HI = 70.0, 180.0
PERSONAL_LO,  PERSONAL_HI  = 75.0, 130.0

_PERSONAL_FILL  = "#14532d";  _PERSONAL_LABEL = "#22c55e"
_CLINICAL_FILL  = "#2d1b4e";  _CLINICAL_LABEL = "#a855f7"


def _axis_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    step = (hi - lo) / n
    return [lo + step * k for k in range(n + 1)]


# ── Fan chart ─────────────────────────────────────────────────────────────────

def render_fan_svg(fan: list[dict[str, Any]]) -> str:
    if not fan:
        return "<p>No scenarios.</p>"

    W, H = 960, 520
    ml, mr, mt, mb = 58, 90, 28, 48
    pw, ph = W - ml - mr, H - mt - mb

    all_vals: list[float] = []
    for s in fan:
        for seq in (s.get("predicted_bg"), s.get("lower"), s.get("upper")):
            if seq:
                all_vals.extend(float(v) for v in seq)
    if not all_vals:
        return "<p>No prediction values.</p>"

    ymin = min(min(all_vals), CLINICAL_LO)
    ymax = max(max(all_vals), CLINICAL_HI)
    pad  = (ymax - ymin) * 0.06 or 1.0
    ymin -= pad; ymax += pad

    nsteps = max(len(s.get("predicted_bg") or []) for s in fan)
    nsteps = max(nsteps, 1)

    def x(i: int) -> float:
        return ml + pw * i / max(nsteps - 1, 1)
    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin))

    step_xs = json.dumps([round(x(i), 1) for i in range(nsteps)])

    parts: list[str] = [
        f'<svg id="bb-fan" viewBox="0 0 {W} {H}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'data-steps=\'{step_xs}\' data-mt="{mt}" data-ph="{ph}" '
        f'style="background:#0d1117;border:1px solid #21262d;border-radius:10px;'
        f'cursor:crosshair;display:block">'
    ]

    # Gridlines
    for gv in _axis_ticks(ymin, ymax):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ml-8}" y="{gy+4:.1f}" fill="#6e7681" font-size="11" '
                     f'text-anchor="end">{gv:.0f}</text>')
    for i in range(0, nsteps, 2):
        gx = x(i)
        parts.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt+ph}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{H-mb+20}" fill="#6e7681" font-size="11" '
                     f'text-anchor="middle">{i*5}m</text>')

    # Target bands
    def _band(lo, hi, fill, opacity, lbl_col, label):
        yhi, ylo = y(hi), y(lo)
        lx = ml + pw + 6
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{fill}" opacity="{opacity}"/>')
        parts.append(f'<text x="{lx}" y="{yhi-2:.1f}" fill="{lbl_col}" '
                     f'font-size="9" opacity="0.8">{label}</text>')
        parts.append(f'<text x="{lx}" y="{yhi+10:.1f}" fill="{lbl_col}" '
                     f'font-size="10">{hi:.0f}</text>')
        parts.append(f'<text x="{lx}" y="{ylo+10:.1f}" fill="{lbl_col}" '
                     f'font-size="10">{lo:.0f}</text>')

    _band(CLINICAL_LO, CLINICAL_HI, _CLINICAL_FILL, "0.45", _CLINICAL_LABEL, "clinical")
    _band(PERSONAL_LO,  PERSONAL_HI,  _PERSONAL_FILL,  "0.5",  _PERSONAL_LABEL,  "target")

    sorted_fan = sorted(fan, key=lambda s: float((s.get("predicted_bg") or [0])[-1]), reverse=True)

    label_ys: list[float] = []
    def _nudge_y(raw: float) -> float:
        cand = raw
        for prev in sorted(label_ys):
            if abs(cand - prev) < 13:
                cand = prev + 13
        label_ys.append(cand)
        return cand

    for s in sorted_fan:
        preds  = s.get("predicted_bg") or []
        lo_seq = s.get("lower") or []
        up_seq = s.get("upper") or []
        if not preds:
            continue
        dose     = float(s.get("dose_u", 0.0))
        is_nom   = abs(dose) < 1e-9
        dose_lbl = "0 U" if is_nom else f"{dose:.2g} U"
        stroke   = "#58a6ff" if is_nom else "#4a8596"
        nom_str  = "1" if is_nom else "0"
        width    = "2.5" if is_nom else "1.3"
        opacity  = "1"   if is_nom else "0.7"
        lbl_col  = "#58a6ff" if is_nom else "#79c0d4"
        dot_r    = "3.5" if is_nom else "2.5"

        parts.append(f'<g class="bb-scenario" data-scenario="{html.escape(dose_lbl)}" '
                     f'data-stroke="{stroke}" data-nom="{nom_str}" '
                     f'data-width="{width}" data-opacity="{opacity}">')

        pts = " ".join(f"{x(j):.1f},{y(float(v)):.1f}" for j, v in enumerate(preds))
        parts.append(f'<polyline class="bb-line" points="{pts}" fill="none" '
                     f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')

        # Visible dots (pointer-events=none so they don't block hit targets)
        for j, v in enumerate(preds):
            parts.append(f'<circle cx="{x(j):.1f}" cy="{y(float(v)):.1f}" r="{dot_r}" '
                         f'fill="{stroke}" opacity="{opacity}" pointer-events="none"/>')

        # Transparent hit targets with all data needed for JS tooltip
        for j, v in enumerate(preds):
            lo_v = float(lo_seq[j]) if j < len(lo_seq) else None
            up_v = float(up_seq[j]) if j < len(up_seq) else None
            ci   = f"{lo_v:.0f}–{up_v:.0f}" if lo_v is not None else ""
            cx, cy = x(j), y(float(v))
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent" '
                f'class="bb-pt" '
                f'data-dose="{html.escape(dose_lbl)}" '
                f'data-mins="{j*5}" '
                f'data-bg="{float(v):.0f}" '
                f'data-ci="{html.escape(ci)}" '
                f'data-sx="{cx:.1f}" data-sy="{cy:.1f}"/>'
            )

        lx_lbl = x(len(preds) - 1) + 8
        ly = _nudge_y(y(float(preds[-1])) + 4)
        parts.append(f'<text class="bb-end-label" x="{lx_lbl:.1f}" y="{ly:.1f}" '
                     f'fill="{lbl_col}" font-size="11" '
                     f'font-weight="{"600" if is_nom else "400"}">'
                     f'{html.escape(dose_lbl)}</text>')
        parts.append('</g>')

    # Static overlay elements — shown/hidden by JS in template
    parts.append(f'<line id="bb-fan-cursor" x1="0" y1="{mt}" x2="0" y2="{mt+ph}" '
                 f'stroke="#e6edf3" stroke-width="1" opacity="0.35" '
                 f'stroke-dasharray="4,3" visibility="hidden" pointer-events="none"/>')
    parts.append(
        '<g id="bb-fan-tip" visibility="hidden" pointer-events="none">'
        '<rect id="bb-fan-tip-bg" rx="5" ry="5" fill="#161b22" stroke="#30363d" stroke-width="1"/>'
        '<text id="bb-fan-t1" fill="#e6edf3" font-size="12" font-weight="600"/>'
        '<text id="bb-fan-t2" fill="#8b949e" font-size="11"/>'
        '<text id="bb-fan-t3" fill="#58a6ff" font-size="10"/>'
        '</g>'
    )
    parts.append('</svg>')
    return "".join(parts)


# ── BG History chart ──────────────────────────────────────────────────────────

def render_history_svg(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No history data.</p>"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True).iloc[-288:].reset_index(drop=True)
    n = len(df)
    if n == 0:
        return "<p>No history data.</p>"

    PX_PER_STEP = 8
    W  = max(n * PX_PER_STEP + 120, 800)
    H  = 260
    ml, mr, mt, mb = 50, 20, 20, 40
    pw, ph = W - ml - mr, H - mt - mb

    bg_vals = df["bg"].ffill().fillna(CLINICAL_HI).tolist()
    mask    = df.get("mask_bg", pd.Series([False]*n)).fillna(False).tolist()
    bolus   = df.get("bolus_dose", pd.Series([0.0]*n)).fillna(0.0).tolist()

    ymin = min(min(bg_vals), CLINICAL_LO)
    ymax = max(max(bg_vals), CLINICAL_HI)
    pad  = (ymax - ymin) * 0.08 or 5.0
    ymin -= pad; ymax += pad

    def x(i: int) -> float:
        return ml + pw * i / max(n - 1, 1)
    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin))

    ts_local = [pd.Timestamp(t).to_pydatetime().astimezone().strftime("%H:%M")
                for t in df["timestamp"].tolist()]
    xs_js  = json.dumps([round(x(i), 1) for i in range(n)])
    bg_js  = json.dumps([round(float(v), 1) for v in bg_vals])
    ts_js  = json.dumps(ts_local)

    parts: list[str] = [
        f'<svg id="bb-hist" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'data-xs=\'{xs_js}\' data-bgs=\'{bg_js}\' data-ts=\'{ts_js}\' '
        f'data-ml="{ml}" data-mt="{mt}" data-ph="{ph}" data-W="{W}" '
        f'style="background:#0d1117;display:block;cursor:crosshair">'
    ]

    # Gridlines
    for gv in _axis_ticks(ymin, ymax, n=5):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{gy+4:.1f}" fill="#6e7681" font-size="10" '
                     f'text-anchor="end">{gv:.0f}</text>')

    def _band(lo, hi, fill, opacity):
        yhi, ylo = y(hi), y(lo)
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{fill}" opacity="{opacity}"/>')
    _band(CLINICAL_LO, CLINICAL_HI, _CLINICAL_FILL, "0.45")
    _band(PERSONAL_LO,  PERSONAL_HI,  _PERSONAL_FILL,  "0.5")

    # Time axis labels
    label_step = max(1, n // 12)
    for i in range(0, n, label_step):
        parts.append(f'<text x="{x(i):.1f}" y="{H-mb+18}" fill="#6e7681" '
                     f'font-size="10" text-anchor="middle">{ts_local[i]}</text>')
        parts.append(f'<line x1="{x(i):.1f}" y1="{mt+ph}" x2="{x(i):.1f}" y2="{mt+ph+4}" '
                     f'stroke="#6e7681" stroke-width="1"/>')

    # BG lines
    full_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bg_vals))
    parts.append(f'<polyline points="{full_pts}" fill="none" '
                 f'stroke="#2d4a6a" stroke-width="1.5" stroke-dasharray="3,3"/>')
    obs_pts = " ".join(f"{x(i):.1f},{y(v):.1f}"
                       for i, v in enumerate(bg_vals) if not mask[i])
    if obs_pts:
        parts.append(f'<polyline points="{obs_pts}" fill="none" '
                     f'stroke="#58a6ff" stroke-width="2"/>')
    for i, (v, m) in enumerate(zip(bg_vals, mask)):
        if m:
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" '
                         f'fill="#4a6fa5" opacity="0.5"/>')

    # Bolus markers
    max_bolus = max((b for b in bolus if b > 0), default=1.0)
    for i, b in enumerate(bolus):
        if b > 0.1:
            tick_h = 6 + 16 * (b / max_bolus)
            bx = x(i)
            parts.append(f'<line x1="{bx:.1f}" y1="{y(CLINICAL_LO):.1f}" '
                         f'x2="{bx:.1f}" y2="{y(CLINICAL_LO)+tick_h:.1f}" '
                         f'stroke="#d97706" stroke-width="2" opacity="0.8"/>')
            parts.append(
                f'<circle cx="{bx:.1f}" cy="{y(CLINICAL_LO)+tick_h:.1f}" r="3" '
                f'fill="#d97706" opacity="0.8">'
                f'<title>{b:.2f} U @ {ts_local[i]}</title></circle>'
            )

    # "Now" marker
    parts.append(f'<line x1="{x(n-1):.1f}" y1="{mt}" x2="{x(n-1):.1f}" y2="{mt+ph}" '
                 f'stroke="#f0883e" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>')
    parts.append(f'<text x="{x(n-1)-4:.1f}" y="{mt+12}" fill="#f0883e" '
                 f'font-size="10" text-anchor="end">now</text>')

    # Overlay rect for mouse events + static overlays
    parts.append(f'<rect id="bb-hist-overlay" x="{ml}" y="{mt}" '
                 f'width="{pw}" height="{ph}" fill="transparent" cursor="crosshair"/>')
    parts.append(f'<line id="bb-hist-cursor" x1="0" y1="{mt}" x2="0" y2="{mt+ph}" '
                 f'stroke="#e6edf3" stroke-width="1" opacity="0.35" '
                 f'stroke-dasharray="4,3" visibility="hidden" pointer-events="none"/>')
    parts.append(
        '<g id="bb-hist-tip" visibility="hidden" pointer-events="none">'
        '<rect id="bb-hist-tip-bg" rx="4" ry="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>'
        '<text id="bb-hist-t1" fill="#e6edf3" font-size="11" font-weight="600"/>'
        '<text id="bb-hist-t2" fill="#8b949e" font-size="10"/>'
        '</g>'
    )
    parts.append('</svg>')
    return "".join(parts)


# ── Diagnostic chart: past prediction ghost curves ────────────────────────────

def render_diagnostic_svg(df: pd.DataFrame, archive: list[dict]) -> str:
    """Show past nominal (0U) predictions as ghost curves over actual BG.

    Each ghost curve starts at its inference timestamp and covers 60 min
    forward, overlaid on the actual BG line so model accuracy is visible.
    """
    if df is None or df.empty:
        return "<p>No history data for diagnostics.</p>"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True).iloc[-288:].reset_index(drop=True)
    n = len(df)
    if n == 0:
        return "<p>No data.</p>"

    PX_PER_STEP = 8
    W  = max(n * PX_PER_STEP + 120, 800)
    H  = 200
    ml, mr, mt, mb = 50, 20, 16, 36
    pw, ph = W - ml - mr, H - mt - mb

    bg_vals = df["bg"].ffill().fillna(CLINICAL_HI).tolist()
    ymin = min(min(bg_vals), CLINICAL_LO)
    ymax = max(max(bg_vals), CLINICAL_HI)
    pad  = (ymax - ymin) * 0.08 or 5.0
    ymin -= pad; ymax += pad

    t0_utc = df["timestamp"].iloc[0]
    t1_utc = df["timestamp"].iloc[-1]
    t_span = (t1_utc - t0_utc).total_seconds()

    def time_to_x(t_utc: pd.Timestamp) -> float:
        dt = (t_utc - t0_utc).total_seconds()
        return ml + pw * max(0.0, min(1.0, dt / t_span)) if t_span > 0 else ml

    def x(i: int) -> float:
        return ml + pw * i / max(n - 1, 1)
    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin))

    ts_local = [pd.Timestamp(t).to_pydatetime().astimezone().strftime("%H:%M")
                for t in df["timestamp"].tolist()]

    parts: list[str] = [
        f'<svg id="bb-diag" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="background:#0d1117;display:block">'
    ]

    # Minimal gridlines
    for gv in _axis_ticks(ymin, ymax, n=4):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{gy+4:.1f}" fill="#6e7681" font-size="10" '
                     f'text-anchor="end">{gv:.0f}</text>')

    def _band(lo, hi, fill, opacity):
        yhi, ylo = y(hi), y(lo)
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{fill}" opacity="{opacity}"/>')
    _band(CLINICAL_LO, CLINICAL_HI, _CLINICAL_FILL, "0.35")
    _band(PERSONAL_LO,  PERSONAL_HI,  _PERSONAL_FILL,  "0.4")

    # Time axis labels
    label_step = max(1, n // 10)
    for i in range(0, n, label_step):
        parts.append(f'<text x="{x(i):.1f}" y="{H-mb+16}" fill="#6e7681" '
                     f'font-size="10" text-anchor="middle">{ts_local[i]}</text>')

    # Ghost prediction curves from archive (faint gray)
    for entry in archive:
        inf_ts_raw = entry.get("timestamp")
        if not inf_ts_raw:
            continue
        try:
            inf_ts = pd.Timestamp(inf_ts_raw, tz="UTC") if "+" not in str(inf_ts_raw) and "Z" not in str(inf_ts_raw) else pd.Timestamp(inf_ts_raw)
            if inf_ts.tzinfo is None:
                inf_ts = inf_ts.tz_localize("UTC")
            else:
                inf_ts = inf_ts.tz_convert("UTC")
        except Exception:
            continue

        # Find nominal (0U) scenario
        fan = entry.get("fan") or []
        nominal = next((s for s in fan if abs(float(s.get("dose_u", 1))) < 1e-9), None)
        if not nominal:
            continue
        preds = nominal.get("predicted_bg") or []
        if not preds:
            continue

        # Build curve — each step is 5 min after inf_ts
        ghost_pts = []
        for step, v in enumerate(preds):
            t = inf_ts + pd.Timedelta(minutes=(step + 1) * 5)
            gx = time_to_x(t)
            gy = y(float(v))
            # Skip points outside the chart's time range
            if gx < ml or gx > ml + pw:
                continue
            ghost_pts.append(f"{gx:.1f},{gy:.1f}")

        if len(ghost_pts) > 1:
            parts.append(f'<polyline points="{" ".join(ghost_pts)}" fill="none" '
                         f'stroke="#6e7681" stroke-width="1.2" opacity="0.25" '
                         f'stroke-dasharray="3,2"/>')

        # Dot at start of prediction (the "now" of that inference)
        sx = time_to_x(inf_ts)
        if ml <= sx <= ml + pw:
            # Find actual BG at that time for the start dot
            parts.append(f'<circle cx="{sx:.1f}" cy="{y(float(preds[0])):.1f}" '
                         f'r="2" fill="#6e7681" opacity="0.35"/>')

    # Actual BG line on top (so it reads clearly over ghost curves)
    full_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bg_vals))
    parts.append(f'<polyline points="{full_pts}" fill="none" '
                 f'stroke="#2d4a6a" stroke-width="1.5" stroke-dasharray="3,3"/>')
    obs_pts = " ".join(f"{x(i):.1f},{y(v):.1f}"
                       for i, v in enumerate(bg_vals)
                       if not df.get("mask_bg", pd.Series([False]*n)).fillna(False).iloc[i])
    if obs_pts:
        parts.append(f'<polyline points="{obs_pts}" fill="none" '
                     f'stroke="#58a6ff" stroke-width="2"/>')

    # "Now" line
    parts.append(f'<line x1="{x(n-1):.1f}" y1="{mt}" x2="{x(n-1):.1f}" y2="{mt+ph}" '
                 f'stroke="#f0883e" stroke-width="1" stroke-dasharray="4,2" opacity="0.6"/>')

    parts.append(f'<text x="{ml}" y="{mt+12}" fill="#6e7681" font-size="10">'
                 f'{len(archive)} past predictions shown</text>')

    parts.append('</svg>')
    return "".join(parts)
