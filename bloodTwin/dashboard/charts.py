"""
SVG chart generators for the bloodBender dashboard.

Two charts:
  render_fan_svg(fan)          — 60-min dose scenario fan
  render_history_svg(df)       — 24h BG history (scrollable, wide SVG)

Both return raw SVG strings (no surrounding HTML).
Target bands:
  Clinical:  70–180 mg/dL  (green,  #1b3a2a)
  Personal:  75–130 mg/dL  (amber,  #78350f)
"""
from __future__ import annotations

import html
from typing import Any

import pandas as pd

# ── Shared constants ──────────────────────────────────────────────────────────
CLINICAL_LO, CLINICAL_HI = 70.0, 180.0
PERSONAL_LO,  PERSONAL_HI  = 75.0, 130.0


def _axis_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    step = (hi - lo) / n
    return [lo + step * k for k in range(n + 1)]


# ── Fan chart ─────────────────────────────────────────────────────────────────

def render_fan_svg(fan: list[dict[str, Any]]) -> str:
    """Scenario fan: one polyline per dose, hover tooltips, dual target bands."""
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
    ymin -= pad
    ymax += pad

    nsteps = max(len(s.get("predicted_bg") or []) for s in fan)
    nsteps = max(nsteps, 1)

    def x(i: int) -> float:
        return ml + pw * i / max(nsteps - 1, 1)

    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin))

    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" '
        f'style="background:#0d1117;border:1px solid #21262d;border-radius:10px">'
    ]

    # ── Gridlines ────────────────────────────────────────────────────────────
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

    # ── Target bands ─────────────────────────────────────────────────────────
    def band(lo, hi, color, opacity, label_color):
        yhi, ylo = y(hi), y(lo)
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{color}" opacity="{opacity}"/>')
        parts.append(f'<text x="{ml+pw+6}" y="{yhi+4:.1f}" fill="{label_color}" '
                     f'font-size="10">{hi:.0f}</text>')
        parts.append(f'<text x="{ml+pw+6}" y="{ylo+4:.1f}" fill="{label_color}" '
                     f'font-size="10">{lo:.0f}</text>')

    band(CLINICAL_LO, CLINICAL_HI, "#1b3a2a", "0.5",  "#3a8a5a")
    band(PERSONAL_LO,  PERSONAL_HI,  "#78350f", "0.35", "#d97706")

    # ── Legend chips (right margin) ───────────────────────────────────────────
    lx = ml + pw + 6
    parts.append(f'<text x="{lx}" y="{y(PERSONAL_HI)-14:.1f}" '
                 f'fill="#d97706" font-size="9">target</text>')
    parts.append(f'<text x="{lx}" y="{y(CLINICAL_HI)-14:.1f}" '
                 f'fill="#3a8a5a" font-size="9">clinical</text>')

    # ── Scenarios ─────────────────────────────────────────────────────────────
    nominal_idx = next(
        (i for i, s in enumerate(fan) if abs(float(s.get("dose_u", 1))) < 1e-9), 0
    )

    def _final_bg(s: dict) -> float:
        p = s.get("predicted_bg") or []
        return float(p[-1]) if p else 0.0

    sorted_fan = sorted(fan, key=_final_bg, reverse=True)

    label_ys: list[float] = []

    def _nudge_y(raw: float) -> float:
        MIN_GAP = 13
        cand = raw
        for prev in sorted(label_ys):
            if abs(cand - prev) < MIN_GAP:
                cand = prev + MIN_GAP
        label_ys.append(cand)
        return cand

    for s in sorted_fan:
        preds = s.get("predicted_bg") or []
        lo_seq = s.get("lower") or []
        up_seq = s.get("upper") or []
        if not preds:
            continue
        dose      = float(s.get("dose_u", 0.0))
        is_nom    = abs(dose) < 1e-9
        dose_lbl  = "0 U" if is_nom else f"{dose:.2g} U"
        stroke    = "#58a6ff" if is_nom else "#4a8596"
        width     = "2.5"    if is_nom else "1.3"
        opacity   = "1"      if is_nom else "0.7"
        lbl_col   = "#58a6ff" if is_nom else "#79c0d4"
        dot_r     = "3.5"    if is_nom else "2.5"

        pts = " ".join(f"{x(j):.1f},{y(float(v)):.1f}" for j, v in enumerate(preds))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                     f'stroke-width="{width}" opacity="{opacity}"/>')

        # Visible dots
        for j, v in enumerate(preds):
            parts.append(f'<circle cx="{x(j):.1f}" cy="{y(float(v)):.1f}" r="{dot_r}" '
                         f'fill="{stroke}" opacity="{opacity}" pointer-events="none"/>')

        # Hover hit-targets
        for j, v in enumerate(preds):
            lo_v  = float(lo_seq[j]) if j < len(lo_seq) else None
            up_v  = float(up_seq[j]) if j < len(up_seq) else None
            ci    = f"{lo_v:.0f}–{up_v:.0f}" if lo_v is not None else ""
            cx, cy = x(j), y(float(v))
            tip   = f"{dose_lbl}|+{j*5}m|{float(v):.0f}|{ci}"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent" '
                f'class="bb-pt" cursor="crosshair" '
                f'data-tip="{html.escape(tip)}" data-sx="{cx:.1f}" data-sy="{cy:.1f}"/>'
            )

        # End label
        lx_lbl = x(len(preds) - 1) + 8
        raw_ly = y(float(preds[-1])) + 4
        ly = _nudge_y(raw_ly)
        fw = "600" if is_nom else "400"
        parts.append(f'<text x="{lx_lbl:.1f}" y="{ly:.1f}" fill="{lbl_col}" '
                     f'font-size="11" font-weight="{fw}">{html.escape(dose_lbl)}</text>')

    # ── Tooltip overlay ───────────────────────────────────────────────────────
    parts.append(
        '<g id="bb-tip" style="display:none" pointer-events="none">'
        '<rect id="bb-tip-bg" rx="5" ry="5" fill="#161b22" stroke="#30363d" stroke-width="1"/>'
        '<text id="bb-t1" fill="#e6edf3" font-size="12" font-weight="600"/>'
        '<text id="bb-t2" fill="#8b949e" font-size="11"/>'
        '<text id="bb-t3" fill="#58a6ff" font-size="10"/>'
        '</g>'
    )
    parts.append(
        '<script type="text/javascript">'
        '(function(){'
        'var tip=document.getElementById("bb-tip"),'
        'bg=document.getElementById("bb-tip-bg"),'
        't1=document.getElementById("bb-t1"),'
        't2=document.getElementById("bb-t2"),'
        't3=document.getElementById("bb-t3");'
        'function show(e){'
        'var d=e.target.getAttribute("data-tip").split("|"),'
        'sx=parseFloat(e.target.getAttribute("data-sx")),'
        'sy=parseFloat(e.target.getAttribute("data-sy")),'
        'dose=d[0],mins=d[1],bgv=d[2],ci=d[3],'
        'W=158,H=ci?52:36,p=7,'
        'tx=sx+14,ty=sy-H-6;'
        'if(tx+W>870){tx=sx-W-8;}'
        'if(ty<10){ty=sy+10;}'
        'bg.setAttribute("x",tx-p);bg.setAttribute("y",ty-16);'
        'bg.setAttribute("width",W);bg.setAttribute("height",H+8);'
        't1.setAttribute("x",tx);t1.setAttribute("y",ty);t1.textContent=dose;'
        't2.setAttribute("x",tx);t2.setAttribute("y",ty+16);'
        't2.textContent=mins+" → "+bgv+" mg/dL";'
        't3.setAttribute("x",tx);t3.setAttribute("y",ty+30);'
        't3.textContent=ci?"90% CI ["+ci+"]":"";'
        'tip.style.display="";}'
        'function hide(){tip.style.display="none";}'
        'document.querySelectorAll(".bb-pt").forEach(function(el){'
        'el.addEventListener("mouseover",show);'
        'el.addEventListener("mouseout",hide);});'
        '})();'
        '</script>'
    )
    parts.append('</svg>')
    return "".join(parts)


# ── BG History chart ──────────────────────────────────────────────────────────

def render_history_svg(df: pd.DataFrame) -> str:
    """24h BG sparkline — wide scrollable SVG, bolus markers, dual target bands."""
    if df is None or df.empty:
        return "<p>No history data.</p>"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Take last 288 rows (24h @ 5 min)
    df = df.iloc[-288:].reset_index(drop=True)
    n  = len(df)
    if n == 0:
        return "<p>No history data.</p>"

    # Dimensions — wide enough to see every 5-min point clearly
    PX_PER_STEP = 8
    W  = max(n * PX_PER_STEP + 120, 800)
    H  = 260
    ml, mr, mt, mb = 50, 20, 20, 40
    pw, ph = W - ml - mr, H - mt - mb

    bg_vals = df["bg"].ffill().fillna(CLINICAL_HI).tolist()
    mask    = df.get("mask_bg", pd.Series([False] * n)).fillna(False).tolist()
    bolus   = df.get("bolus_dose", pd.Series([0.0] * n)).fillna(0.0).tolist()

    ymin = min(min(bg_vals), CLINICAL_LO)
    ymax = max(max(bg_vals), CLINICAL_HI)
    pad  = (ymax - ymin) * 0.08 or 5.0
    ymin -= pad
    ymax += pad

    def x(i: int) -> float:
        return ml + pw * i / max(n - 1, 1)

    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin))

    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="background:#0d1117;display:block">'
    ]

    # Gridlines
    for gv in _axis_ticks(ymin, ymax, n=5):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{gy+4:.1f}" fill="#6e7681" font-size="10" '
                     f'text-anchor="end">{gv:.0f}</text>')

    # Target bands
    def band(lo, hi, color, opacity):
        yhi, ylo = y(hi), y(lo)
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{color}" opacity="{opacity}"/>')

    band(CLINICAL_LO, CLINICAL_HI, "#1b3a2a", "0.5")
    band(PERSONAL_LO,  PERSONAL_HI,  "#78350f", "0.35")

    # Time axis labels every ~2 hours (24 steps)
    ts_list = df["timestamp"].tolist()
    label_step = max(1, n // 12)
    for i in range(0, n, label_step):
        ts = pd.Timestamp(ts_list[i]).to_pydatetime().astimezone()
        parts.append(f'<text x="{x(i):.1f}" y="{H-mb+18}" fill="#6e7681" '
                     f'font-size="10" text-anchor="middle">{ts.strftime("%H:%M")}</text>')
        parts.append(f'<line x1="{x(i):.1f}" y1="{mt+ph}" x2="{x(i):.1f}" y2="{mt+ph+4}" '
                     f'stroke="#6e7681" stroke-width="1"/>')

    # BG line — split into observed vs imputed segments
    obs_pts: list[str] = []
    imp_pts: list[str] = []
    for i, (v, m) in enumerate(zip(bg_vals, mask)):
        pt = f"{x(i):.1f},{y(v):.1f}"
        (imp_pts if m else obs_pts).append(pt)

    if obs_pts:
        pts = " ".join(f"{x(i):.1f},{y(bg_vals[i]):.1f}"
                       for i in range(n) if not mask[i])
        # Draw full line first (faint), then overlay observed segments
        full_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bg_vals))
        parts.append(f'<polyline points="{full_pts}" fill="none" '
                     f'stroke="#2d4a6a" stroke-width="1.5" stroke-dasharray="3,3"/>')

    # Observed BG line (solid)
    solid_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bg_vals)
                         if not mask[i])
    if solid_pts:
        parts.append(f'<polyline points="{solid_pts}" fill="none" '
                     f'stroke="#58a6ff" stroke-width="2"/>')

    # Imputed dots (dimmer)
    for i, (v, m) in enumerate(zip(bg_vals, mask)):
        if m:
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" '
                         f'fill="#4a6fa5" opacity="0.5"/>')

    # Bolus markers — vertical ticks at bottom
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
                f'fill="#d97706" opacity="0.8" cursor="crosshair">'
                f'<title>{b:.2f} U bolus @ '
                f'{pd.Timestamp(ts_list[i]).to_pydatetime().astimezone().strftime("%H:%M")}</title>'
                f'</circle>'
            )

    # "Now" marker at rightmost point
    parts.append(f'<line x1="{x(n-1):.1f}" y1="{mt}" x2="{x(n-1):.1f}" y2="{mt+ph}" '
                 f'stroke="#f0883e" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>')
    parts.append(f'<text x="{x(n-1)-4:.1f}" y="{mt+12}" fill="#f0883e" '
                 f'font-size="10" text-anchor="end">now</text>')

    parts.append('</svg>')
    return "".join(parts)
