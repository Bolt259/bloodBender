#!/usr/bin/env python
"""
bloodBender local dashboard — single-file, stdlib-only web server.

Displays the latest inference result written by the cron job to
``bloodBath/logs/inference_latest.json``. Reads the file fresh on every page
load, auto-refreshes the browser every 60s, and flags stale data (>10 min old).

FastAPI/Jinja2 are NOT available in the ``.#inference`` Nix shell, so this uses
only the Python standard library (``http.server`` + inline HTML). No CDN, no JS
charting libraries — fully self-contained and offline.

Run via the launcher:  ./bloodTwin/dashboard/serve.sh
Or directly inside the inference shell:
    PYTHONPATH=. python bloodTwin/dashboard/server.py
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Repo root = two levels up from this file (bloodTwin/dashboard/server.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LATEST_JSON = os.path.join(REPO_ROOT, "bloodBath", "logs", "inference_latest.json")
RAW_LSTM_GLOB = os.path.join(REPO_ROOT, "bloodBath", "bloodBank", "raw", "lstm", "pump_1539514_*.csv")

HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))

STALE_SECONDS = 90 * 60  # Tandem API upload lag is 15-40 min; alert only if truly stale


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _fmt(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _parse_ts(raw):
    """Parse the ISO timestamp from the inference JSON into an aware datetime."""
    try:
        dt = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load():
    """Load the latest inference JSON. Returns (data, error_message)."""
    try:
        mtime = os.path.getmtime(LATEST_JSON)
        with open(LATEST_JSON, "r") as f:
            data = json.load(f)
        data["_file_mtime"] = mtime
        return data, None
    except FileNotFoundError:
        return None, f"Inference output not found at {LATEST_JSON}. Has the cron job run yet?"
    except json.JSONDecodeError as e:
        return None, f"Inference output is not valid JSON (mid-write?): {e}"
    except Exception as e:  # pragma: no cover - defensive
        return None, f"Could not read inference output: {e}"


def _last_sync_time():
    """Return the mtime of the newest pump raw CSV as an aware UTC datetime."""
    files = glob.glob(RAW_LSTM_GLOB)
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    return datetime.fromtimestamp(os.path.getmtime(newest), tz=timezone.utc)


def _conformal_cell(ci, horizon):
    pair = (ci or {}).get(str(horizon)) or (ci or {}).get(horizon)
    if not pair or len(pair) != 2:
        return "n/a"
    return f"[{_fmt(pair[0])} – {_fmt(pair[1])}]"


def _point_at(fan, idx):
    """Point prediction from the nominal (dose 0) scenario at a 5-min step index."""
    if not fan:
        return None
    nominal = fan[0]
    for s in fan:
        if abs(float(s.get("dose_u", 0.0))) < 1e-9:
            nominal = s
            break
    preds = nominal.get("predicted_bg") or []
    if idx < len(preds):
        return preds[idx]
    return None


# ── SVG scenario fan ────────────────────────────────────────────────────────

def _render_fan_svg(fan):
    """Render the scenario fan as a self-contained SVG line chart.

    One line per dose scenario (nominal highlighted), x = 5-min steps (0..60min),
    y = predicted BG. Dose labels are placed at the right end of each line.
    """
    if not fan:
        return "<p>No scenarios.</p>"

    W, H = 960, 520
    ml, mr, mt, mb = 58, 80, 24, 44   # mr enlarged for dose labels
    pw, ph = W - ml - mr, H - mt - mb

    all_vals = []
    for s in fan:
        for seq in (s.get("predicted_bg"), s.get("lower"), s.get("upper")):
            if seq:
                all_vals.extend(float(v) for v in seq)
    if not all_vals:
        return "<p>No prediction values.</p>"

    ymin, ymax = min(all_vals), max(all_vals)
    ymin = min(ymin, 70.0)
    ymax = max(ymax, 180.0)
    pad = (ymax - ymin) * 0.05 or 1.0
    ymin -= pad
    ymax += pad
    nsteps = max(len(s.get("predicted_bg") or []) for s in fan)
    nsteps = max(nsteps, 1)

    def x(i):
        return ml + (pw * i / max(nsteps - 1, 1))

    def y(v):
        return mt + ph * (1 - (float(v) - ymin) / (ymax - ymin))

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" '
             f'style="background:#0e1116;border:1px solid #2a2f3a;border-radius:8px">']

    # Horizontal gridlines + y labels.
    for gv in _axis_ticks(ymin, ymax):
        gy = y(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" '
                     f'stroke="#222833" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" fill="#8a93a6" font-size="12" '
                     f'text-anchor="end">{gv:.0f}</text>')

    # Target band 70–180 mg/dL.
    yt_hi, yt_lo = y(180.0), y(70.0)
    parts.append(f'<rect x="{ml}" y="{yt_hi:.1f}" width="{pw}" height="{yt_lo - yt_hi:.1f}" '
                 f'fill="#1b3a2a" opacity="0.35"/>')
    # Target band edge labels
    parts.append(f'<text x="{ml + pw + 6}" y="{yt_hi + 4:.1f}" fill="#3a7a5a" font-size="10">180</text>')
    parts.append(f'<text x="{ml + pw + 6}" y="{yt_lo + 4:.1f}" fill="#3a7a5a" font-size="10">70</text>')

    # Vertical gridlines + X axis labels (every 10 min).
    for i in range(0, nsteps, 2):
        gx = x(i)
        parts.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt + ph}" '
                     f'stroke="#1e2530" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{H - mb + 22}" fill="#8a93a6" font-size="12" '
                     f'text-anchor="middle">{i * 5}m</text>')

    # Nominal scenario index.
    nominal_idx = 0
    for i, s in enumerate(fan):
        if abs(float(s.get("dose_u", 0.0))) < 1e-9:
            nominal_idx = i
            break

    # Sort scenarios by final BG so labels stack predictably (highest dose → lowest BG).
    def _final_bg(s):
        p = s.get("predicted_bg") or []
        return float(p[-1]) if p else 0.0

    sorted_fan = sorted(fan, key=_final_bg, reverse=True)

    # Deduplicate label y-positions to avoid overlap (min 13px apart).
    label_ys: list[float] = []

    def _nudge_y(raw_y):
        MIN_GAP = 13
        candidate = raw_y
        for prev in sorted(label_ys):
            if abs(candidate - prev) < MIN_GAP:
                candidate = prev + MIN_GAP
        label_ys.append(candidate)
        return candidate

    for s in sorted_fan:
        preds = s.get("predicted_bg") or []
        lo    = s.get("lower") or []
        up    = s.get("upper") or []
        if not preds:
            continue
        dose = float(s.get("dose_u", 0.0))
        is_nominal = abs(dose) < 1e-9
        pts = " ".join(f"{x(j):.1f},{y(v):.1f}" for j, v in enumerate(preds))

        if is_nominal:
            stroke, width, opacity = "#5ea0ff", "2.5", "1"
            label_col = "#5ea0ff"
            dose_lbl = "0 U"
        else:
            stroke, width, opacity = "#5a8494", "1.2", "0.7"
            label_col = "#8ab4c4"
            dose_lbl = f"{dose:.2g} U"

        parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                     f'stroke-width="{width}" opacity="{opacity}"/>')

        # Visible data point dots.
        dot_r = "3.5" if is_nominal else "2.5"
        dot_fill = stroke
        for j, v in enumerate(preds):
            parts.append(f'<circle cx="{x(j):.1f}" cy="{y(float(v)):.1f}" r="{dot_r}" '
                         f'fill="{dot_fill}" opacity="{opacity}" pointer-events="none"/>')

        # Invisible hit-target circles at each data point for hover tooltips.
        for j, v in enumerate(preds):
            lo_v = float(lo[j]) if j < len(lo) else None
            up_v = float(up[j]) if j < len(up) else None
            ci = f"{lo_v:.0f}–{up_v:.0f}" if lo_v is not None else ""
            cx, cy = x(j), y(float(v))
            tip = f"{dose_lbl}|+{j*5}m|{float(v):.0f}|{ci}"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent" '
                f'class="bb-pt" cursor="crosshair" '
                f'data-tip="{html.escape(tip)}" data-sx="{cx:.1f}" data-sy="{cy:.1f}"/>'
            )

        # Label at right end of line.
        lx = x(len(preds) - 1) + 7
        raw_ly = y(float(preds[-1])) + 4
        ly = _nudge_y(raw_ly)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{label_col}" '
                     f'font-size="11" font-weight="{"600" if is_nominal else "400"}">'
                     f'{html.escape(dose_lbl)}</text>')

    # Tooltip overlay (hidden until hover).
    parts.append(
        '<g id="bb-tip" style="display:none" pointer-events="none">'
        '<rect id="bb-tip-bg" rx="4" ry="4" fill="#1a2030" stroke="#3a4a60" stroke-width="1"/>'
        '<text id="bb-t1" fill="#e6e9ef" font-size="12" font-weight="600"/>'
        '<text id="bb-t2" fill="#aab2c4" font-size="11"/>'
        '<text id="bb-t3" fill="#6a8a9a" font-size="10"/>'
        '</g>'
    )

    # Inline JS: show/hide tooltip on circle hover, positioned in SVG space.
    parts.append('''<script type="text/javascript">
(function(){
  var tip=document.getElementById("bb-tip"),
      bg=document.getElementById("bb-tip-bg"),
      t1=document.getElementById("bb-t1"),
      t2=document.getElementById("bb-t2"),
      t3=document.getElementById("bb-t3");
  function show(e){
    var d=e.target.getAttribute("data-tip").split("|"),
        sx=parseFloat(e.target.getAttribute("data-sx")),
        sy=parseFloat(e.target.getAttribute("data-sy")),
        dose=d[0], mins=d[1], bg_v=d[2], ci=d[3],
        W=152, H=ci?50:36, pad=6,
        tx=sx+14, ty=sy-H-4;
    if(tx+W>880){tx=sx-W-6;}
    if(ty<10){ty=sy+10;}
    bg.setAttribute("x",tx-pad); bg.setAttribute("y",ty-14);
    bg.setAttribute("width",W); bg.setAttribute("height",H);
    t1.setAttribute("x",tx); t1.setAttribute("y",ty);
    t1.textContent=dose;
    t2.setAttribute("x",tx); t2.setAttribute("y",ty+15);
    t2.textContent=mins+" → "+bg_v+" mg/dL";
    t3.setAttribute("x",tx); t3.setAttribute("y",ty+28);
    t3.textContent=ci?"90% CI: ["+ci+"]":"";
    tip.style.display="";
  }
  function hide(){tip.style.display="none";}
  document.querySelectorAll(".bb-pt").forEach(function(el){
    el.addEventListener("mouseover",show);
    el.addEventListener("mouseout",hide);
  });
})();
</script>''')

    parts.append('</svg>')
    return "".join(parts)


def _axis_ticks(lo, hi, n=6):
    step = (hi - lo) / n
    return [lo + step * k for k in range(n + 1)]


# ── Fan table ─────────────────────────────────────────────────────────────────

def _render_fan_table(fan):
    rows = []
    for s in fan:
        dose = _fmt(s.get("dose_u"), 2)
        p = s.get("predicted_bg") or []
        lo = s.get("lower") or []
        up = s.get("upper") or []

        def cell(idx):
            if idx < len(p):
                band = ""
                if idx < len(lo) and idx < len(up):
                    band = f'<span class="band">[{_fmt(lo[idx])}–{_fmt(up[idx])}]</span>'
                return f"{_fmt(p[idx])}<br>{band}"
            return "—"

        is_nominal = abs(float(s.get("dose_u", 0.0))) < 1e-9
        cls = ' class="nominal"' if is_nominal else ""
        rows.append(
            f"<tr{cls}><td>{dose}</td><td>{cell(0)}</td><td>{cell(5)}</td>"
            f"<td>{cell(11)}</td></tr>"
        )
    return (
        '<table class="fan"><thead><tr>'
        '<th>Dose (U)</th><th>5 min</th><th>30 min</th><th>60 min</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


# ── Page ────────────────────────────────────────────────────────────────────

def render_page():
    data, err = _load()
    now = datetime.now(timezone.utc)

    if err:
        body = f'<div class="warn">{html.escape(err)}</div>'
        return _wrap(body)

    ts = _parse_ts(data.get("timestamp"))
    age_sec = (now - ts).total_seconds() if ts else None
    stale = age_sec is not None and age_sec > STALE_SECONDS

    ci = data.get("conformal_intervals") or {}
    fan = data.get("fan") or []
    fms = data.get("failure_modes") or []
    top_fm = fms[0] if fms else {}
    # Field is "probability" in the real output (task doc said "prob"); handle both.
    top_prob = top_fm.get("probability", top_fm.get("prob"))

    p30 = _point_at(fan, 5)
    p60 = _point_at(fan, 11)

    stale_banner = ""
    if stale:
        mins = int(age_sec // 60)
        stale_banner = (
            f'<div class="warn">STALE DATA — last inference was {mins} min ago '
            f'(threshold {STALE_SECONDS // 60} min). Cron job may be stopped.</div>'
        )

    ts_disp = ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if ts else "unknown"
    age_disp = f"{int(age_sec)}s ago" if age_sec is not None else "unknown"
    now_local = now.astimezone()
    server_time_disp = (
        f"{now_local.strftime('%H:%M:%S %Z')} "
        f"({now.strftime('%H:%M:%S UTC')})"
    )
    sync_ts = _last_sync_time()
    if sync_ts:
        sync_local = sync_ts.astimezone()
        sync_age = int((now - sync_ts).total_seconds())
        if sync_age < 3600:
            sync_age_disp = f"{sync_age // 60}m ago"
        else:
            sync_age_disp = f"{sync_age // 3600}h {(sync_age % 3600) // 60}m ago"
        sync_disp = f"{sync_local.strftime('%H:%M %Z')} ({sync_age_disp})"
    else:
        sync_disp = "unknown"

    cards = f"""
    <div class="grid">
      <div class="card big"><div class="label">Current BG</div>
        <div class="value">{_fmt(data.get('current_bg'))} <span class="unit">mg/dL</span></div></div>
      <div class="card"><div class="label">30 min point</div>
        <div class="value">{_fmt(p30)} <span class="unit">mg/dL</span></div>
        <div class="sub">90% CI {_conformal_cell(ci, 30)}</div></div>
      <div class="card"><div class="label">60 min point</div>
        <div class="value">{_fmt(p60)} <span class="unit">mg/dL</span></div>
        <div class="sub">90% CI {_conformal_cell(ci, 60)}</div></div>
      <div class="card"><div class="label">Top failure mode</div>
        <div class="value">{html.escape(str(top_fm.get('state', 'n/a')))}</div>
        <div class="sub">p = {_fmt(top_prob, 2) if top_prob is not None else 'n/a'}</div></div>
      <div class="card"><div class="label">Effective ISF</div>
        <div class="value">{_fmt(data.get('isf'))} <span class="unit">mg/dL/U</span></div>
        <div class="sub">est. correction {_fmt(data.get('ecd_u'), 2)} U</div></div>
      <div class="card"><div class="label">Meal probability</div>
        <div class="value">{_fmt(data.get('meal_probability'), 2)}</div>
        <div class="sub">trigger: {html.escape(str(data.get('model_trigger', 'n/a')))}</div></div>
    </div>
    """

    body = f"""
    {stale_banner}
    <div class="meta">
      Last inference: <b>{html.escape(ts_disp)}</b> ({age_disp}) &middot;
      Last synced: <b>{html.escape(sync_disp)}</b> &middot;
      server: {html.escape(server_time_disp)} &middot;
      corrected in horizon: {data.get('corrected_in_horizon')}
    </div>
    {cards}
    <h2>Scenario fan</h2>
    {_render_fan_svg(fan)}
    <p class="legend"><span class="sw-nom"></span> nominal (0 U) &nbsp;
      <span class="sw-alt"></span> dose scenarios &nbsp;
      <span class="sw-band"></span> target band 70–180</p>
    {_render_fan_table(fan)}
    """
    return _wrap(body)


def _wrap(body):
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>bloodBender dashboard</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0a0c10;
         color:#e6e9ef; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:28px 0 10px; color:#aab2c4; }}
  .meta {{ color:#8a93a6; font-size:13px; margin:8px 0 18px; }}
  .warn {{ background:#3a1c1c; border:1px solid #7a2e2e; color:#ffb4b4;
          padding:12px 16px; border-radius:8px; margin:0 0 16px; font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
  .card {{ background:#12161d; border:1px solid #232a36; border-radius:10px; padding:14px 16px; }}
  .card.big {{ border-color:#2f4a6b; }}
  .label {{ color:#8a93a6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .value {{ font-size:26px; font-weight:700; margin-top:4px; }}
  .unit {{ font-size:13px; color:#8a93a6; font-weight:400; }}
  .sub {{ color:#8a93a6; font-size:12px; margin-top:4px; }}
  table.fan {{ border-collapse:collapse; margin-top:12px; font-size:13px; width:100%; max-width:720px; }}
  table.fan th, table.fan td {{ border:1px solid #232a36; padding:6px 10px; text-align:center; }}
  table.fan th {{ background:#12161d; color:#aab2c4; }}
  table.fan tr.nominal {{ background:#152335; }}
  .band {{ color:#8a93a6; font-size:11px; }}
  .legend {{ font-size:12px; color:#8a93a6; margin-top:10px; }}
  .legend span {{ display:inline-block; width:14px; height:4px; margin-right:4px; vertical-align:middle; }}
  .sw-nom {{ background:#5ea0ff; }} .sw-alt {{ background:#5a6472; }}
  .sw-band {{ background:#1b3a2a; height:12px !important; }}
</style></head>
<body>
<h1>bloodBender — single-patient T1D inference</h1>
{body}
<div class="meta" style="margin-top:28px">Auto-refresh 60s &middot; reads {html.escape(LATEST_JSON)}</div>
</body></html>"""


# ── HTTP ──────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self.send_error(404, "Not found")
            return
        try:
            page = render_page().encode("utf-8")
        except Exception as e:  # pragma: no cover - defensive
            self.send_error(500, f"render error: {e}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, fmt, *args):  # quieter logging
        pass


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"bloodBender dashboard serving on http://{HOST}:{PORT}  (reading {LATEST_JSON})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
