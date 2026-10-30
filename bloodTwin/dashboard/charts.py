"""
SVG chart generators for the bloodBender dashboard.

  render_fan_svg(fan)        — 60-min dose scenario fan
  render_history_svg(df)     — 24h BG history (scrollable)

Target bands:
  Personal:  75–130 mg/dL  (green  — your goal)
  Clinical:  70–180 mg/dL  (purple — clinical safe range)
"""
from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd

CLINICAL_LO, CLINICAL_HI = 70.0, 180.0
PERSONAL_LO,  PERSONAL_HI  = 75.0, 130.0

# Band colours
_PERSONAL_FILL  = "#14532d";  _PERSONAL_LABEL  = "#22c55e"   # green
_CLINICAL_FILL  = "#2d1b4e";  _CLINICAL_LABEL  = "#a855f7"   # purple


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
        f'preserveAspectRatio="xMidYMid meet" data-steps=\'{step_xs}\' '
        f'data-ml="{ml}" data-mr="{mr}" data-mt="{mt}" data-mb="{mb}" '
        f'style="background:#0d1117;border:1px solid #21262d;border-radius:10px;cursor:crosshair">'
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

    # Target bands (personal=green, clinical=purple)
    def _band(lo, hi, fill, opacity, label_col, label):
        yhi, ylo = y(hi), y(lo)
        lx = ml + pw + 6
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{fill}" opacity="{opacity}"/>')
        parts.append(f'<text x="{lx}" y="{yhi-2:.1f}" fill="{label_col}" '
                     f'font-size="9" opacity="0.8">{label}</text>')
        parts.append(f'<text x="{lx}" y="{yhi+10:.1f}" fill="{label_col}" '
                     f'font-size="10">{hi:.0f}</text>')
        parts.append(f'<text x="{lx}" y="{ylo+10:.1f}" fill="{label_col}" '
                     f'font-size="10">{lo:.0f}</text>')

    _band(CLINICAL_LO, CLINICAL_HI, _CLINICAL_FILL, "0.45", _CLINICAL_LABEL, "clinical")
    _band(PERSONAL_LO,  PERSONAL_HI,  _PERSONAL_FILL,  "0.5",  _PERSONAL_LABEL,  "target")

    # Scenarios — each wrapped in a <g data-scenario="..."> for highlight
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
        width    = "2.5"    if is_nom else "1.3"
        opacity  = "1"      if is_nom else "0.7"
        lbl_col  = "#58a6ff" if is_nom else "#79c0d4"
        dot_r    = "3.5"    if is_nom else "2.5"

        parts.append(f'<g class="bb-scenario" data-scenario="{html.escape(dose_lbl)}"'
                     f' data-stroke="{stroke}" data-nom="{int(is_nom)}">')

        pts = " ".join(f"{x(j):.1f},{y(float(v)):.1f}" for j, v in enumerate(preds))
        parts.append(f'<polyline class="bb-line" points="{pts}" fill="none" '
                     f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')

        for j, v in enumerate(preds):
            parts.append(f'<circle cx="{x(j):.1f}" cy="{y(float(v)):.1f}" r="{dot_r}" '
                         f'fill="{stroke}" opacity="{opacity}" pointer-events="none"/>')

        for j, v in enumerate(preds):
            lo_v = float(lo_seq[j]) if j < len(lo_seq) else None
            up_v = float(up_seq[j]) if j < len(up_seq) else None
            ci   = f"{lo_v:.0f}–{up_v:.0f}" if lo_v is not None else ""
            cx, cy = x(j), y(float(v))
            tip  = f"{dose_lbl}|+{j*5}m|{float(v):.0f}|{ci}"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent" '
                f'class="bb-pt" cursor="crosshair" '
                f'data-tip="{html.escape(tip)}" '
                f'data-sx="{cx:.1f}" data-sy="{cy:.1f}" '
                f'data-scenario="{html.escape(dose_lbl)}"/>'
            )

        lx_lbl = x(len(preds) - 1) + 8
        ly = _nudge_y(y(float(preds[-1])) + 4)
        parts.append(f'<text class="bb-end-label" x="{lx_lbl:.1f}" y="{ly:.1f}" '
                     f'fill="{lbl_col}" font-size="11" '
                     f'font-weight="{"600" if is_nom else "400"}">'
                     f'{html.escape(dose_lbl)}</text>')

        parts.append('</g>')

    # Cursor line (hidden until hover)
    parts.append(f'<line id="bb-fan-cursor" x1="0" y1="{mt}" x2="0" y2="{mt+ph}" '
                 f'stroke="#e6edf3" stroke-width="1" opacity="0.4" '
                 f'stroke-dasharray="4,3" style="display:none" pointer-events="none"/>')

    # Tooltip overlay
    parts.append(
        '<g id="bb-tip" style="display:none" pointer-events="none">'
        '<rect id="bb-tip-bg" rx="5" ry="5" fill="#161b22" stroke="#30363d" stroke-width="1"/>'
        '<text id="bb-t1" fill="#e6edf3" font-size="12" font-weight="600"/>'
        '<text id="bb-t2" fill="#8b949e" font-size="11"/>'
        '<text id="bb-t3" fill="#58a6ff" font-size="10"/>'
        '</g>'
    )

    # JS: cursor tracking + scenario highlighting + tooltips
    parts.append('''<script type="text/javascript">
(function(){
  var svg=document.getElementById("bb-fan");
  var cursor=document.getElementById("bb-fan-cursor");
  var tip=document.getElementById("bb-tip");
  var tipBg=document.getElementById("bb-tip-bg");
  var t1=document.getElementById("bb-t1");
  var t2=document.getElementById("bb-t2");
  var t3=document.getElementById("bb-t3");
  var steps=JSON.parse(svg.getAttribute("data-steps"));
  var mt=parseFloat(svg.getAttribute("data-mt"));
  var ph=''' + str(ph) + ''';

  function svgX(e){
    var pt=svg.createSVGPoint();
    pt.x=e.clientX; pt.y=e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse()).x;
  }
  function nearestStep(sx){
    var best=0,bd=1e9;
    steps.forEach(function(sx2,i){var d=Math.abs(sx2-sx);if(d<bd){bd=d;best=i;}});
    return best;
  }
  function highlightScenario(dose){
    document.querySelectorAll(".bb-scenario").forEach(function(g){
      var match=(dose===null||g.getAttribute("data-scenario")===dose);
      var lines=g.querySelectorAll(".bb-line");
      lines.forEach(function(l){
        l.style.opacity=match?"1":"0.2";
        l.style.strokeWidth=match?(g.getAttribute("data-nom")==="1"?"3":"2"):(g.getAttribute("data-nom")==="1"?"2.5":"1.3");
      });
    });
  }
  function resetHighlight(){
    document.querySelectorAll(".bb-scenario").forEach(function(g){
      var lines=g.querySelectorAll(".bb-line");
      var nom=g.getAttribute("data-nom")==="1";
      lines.forEach(function(l){
        l.style.opacity=nom?"1":"0.7";
        l.style.strokeWidth=nom?"2.5":"1.3";
      });
    });
  }

  svg.addEventListener("mousemove",function(e){
    var sx=svgX(e);
    var step=nearestStep(sx);
    var cx=steps[step];
    cursor.setAttribute("x1",cx); cursor.setAttribute("x2",cx);
    cursor.style.display="";
  });
  svg.addEventListener("mouseleave",function(){
    cursor.style.display="none";
    resetHighlight();
    tip.style.display="none";
  });

  document.querySelectorAll(".bb-pt").forEach(function(el){
    el.addEventListener("mouseover",function(e){
      var d=e.target.getAttribute("data-tip").split("|");
      var dose=d[0],mins=d[1],bgv=d[2],ci=d[3];
      var sx=parseFloat(e.target.getAttribute("data-sx"));
      var sy=parseFloat(e.target.getAttribute("data-sy"));
      var sc=e.target.getAttribute("data-scenario");
      highlightScenario(sc);
      cursor.setAttribute("x1",sx); cursor.setAttribute("x2",sx);
      cursor.style.display="";
      var W=158,H=ci?52:36,p=7,tx=sx+14,ty=sy-H-6;
      if(tx+W>870){tx=sx-W-8;} if(ty<10){ty=sy+10;}
      tipBg.setAttribute("x",tx-p); tipBg.setAttribute("y",ty-16);
      tipBg.setAttribute("width",W); tipBg.setAttribute("height",H+8);
      t1.setAttribute("x",tx); t1.setAttribute("y",ty); t1.textContent=dose;
      t2.setAttribute("x",tx); t2.setAttribute("y",ty+16);
      t2.textContent=mins+" → "+bgv+" mg/dL";
      t3.setAttribute("x",tx); t3.setAttribute("y",ty+30);
      t3.textContent=ci?"90% CI ["+ci+"]":"";
      tip.style.display="";
    });
    el.addEventListener("mouseout",function(){
      resetHighlight();
      tip.style.display="none";
    });
  });
})();
</script>''')

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

    # Pre-compute x positions and timestamps for JS tooltip
    ts_list  = df["timestamp"].tolist()
    ts_local = [pd.Timestamp(t).to_pydatetime().astimezone().strftime("%H:%M") for t in ts_list]
    xs_js    = json.dumps([round(x(i), 1) for i in range(n)])
    bg_js    = json.dumps([round(float(v), 1) for v in bg_vals])
    ts_js    = json.dumps(ts_local)

    parts: list[str] = [
        f'<svg id="bb-hist" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="background:#0d1117;display:block;cursor:crosshair">'
    ]

    # Gridlines
    for gv in _axis_ticks(ymin, ymax, n=5):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{gy+4:.1f}" fill="#6e7681" font-size="10" '
                     f'text-anchor="end">{gv:.0f}</text>')

    # Target bands
    def _band(lo, hi, fill, opacity):
        yhi, ylo = y(hi), y(lo)
        parts.append(f'<rect x="{ml}" y="{yhi:.1f}" width="{pw}" '
                     f'height="{ylo-yhi:.1f}" fill="{fill}" opacity="{opacity}"/>')
    _band(CLINICAL_LO, CLINICAL_HI, _CLINICAL_FILL, "0.45")
    _band(PERSONAL_LO,  PERSONAL_HI,  _PERSONAL_FILL,  "0.5")

    # Time axis labels every ~2h
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
    obs_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bg_vals) if not mask[i])
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

    # Cursor line + tooltip elements
    parts.append(f'<line id="bb-hist-cursor" x1="0" y1="{mt}" x2="0" y2="{mt+ph}" '
                 f'stroke="#e6edf3" stroke-width="1" opacity="0.4" '
                 f'stroke-dasharray="4,3" style="display:none" pointer-events="none"/>')
    parts.append(
        '<g id="bb-hist-tip" style="display:none" pointer-events="none">'
        '<rect id="bb-hist-tip-bg" rx="4" ry="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>'
        '<text id="bb-hist-t1" fill="#e6edf3" font-size="11" font-weight="600"/>'
        '<text id="bb-hist-t2" fill="#8b949e" font-size="10"/>'
        '</g>'
    )

    # Invisible wide rect to catch mouse events across the whole plot area
    parts.append(f'<rect id="bb-hist-overlay" x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                 f'fill="transparent" cursor="crosshair"/>')

    # JS: cursor line + tooltip on hover
    parts.append(f'''<script type="text/javascript">
(function(){{
  var svg=document.getElementById("bb-hist");
  var overlay=document.getElementById("bb-hist-overlay");
  var cursor=document.getElementById("bb-hist-cursor");
  var tip=document.getElementById("bb-hist-tip");
  var tipBg=document.getElementById("bb-hist-tip-bg");
  var t1=document.getElementById("bb-hist-t1");
  var t2=document.getElementById("bb-hist-t2");
  var xs={xs_js};
  var bgs={bg_js};
  var tss={ts_js};
  var ml={ml}, mt={mt}, ph={ph};

  function nearestIdx(ex){{
    var best=0,bd=1e9;
    xs.forEach(function(cx,i){{var d=Math.abs(cx-ex);if(d<bd){{bd=d;best=i;}}}});
    return best;
  }}
  overlay.addEventListener("mousemove",function(e){{
    var ex=e.offsetX; // offsetX is SVG-space since SVG has fixed pixel width
    var idx=nearestIdx(ex);
    var cx=xs[idx];
    cursor.setAttribute("x1",cx); cursor.setAttribute("x2",cx);
    cursor.style.display="";
    var bgv=bgs[idx].toFixed(1), ts=tss[idx];
    var tx=cx+8, ty=mt+20;
    if(tx+120>{W}-20){{ tx=cx-128; }}
    tipBg.setAttribute("x",tx-4); tipBg.setAttribute("y",ty-14);
    tipBg.setAttribute("width",120); tipBg.setAttribute("height",34);
    t1.setAttribute("x",tx); t1.setAttribute("y",ty); t1.textContent=ts;
    t2.setAttribute("x",tx); t2.setAttribute("y",ty+14);
    t2.textContent=bgv+" mg/dL";
    tip.style.display="";
  }});
  overlay.addEventListener("mouseleave",function(){{
    cursor.style.display="none";
    tip.style.display="none";
  }});
}})();
</script>''')

    parts.append('</svg>')
    return "".join(parts)
