#!/usr/bin/env python
"""
bloodBender local dashboard — stdlib http.server + Jinja2 templates.

Reads inference_latest.json (written by the 5-min cron) and the merged
BG history CSV every page load.  Auto-refreshes every 60s.

Run:  PYTHONPATH=. python bloodTwin/dashboard/server.py
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jinja2
import pandas as pd

from bloodTwin.dashboard.charts import render_fan_svg, render_history_svg, render_diagnostic_svg

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[2]
LATEST_JSON   = REPO_ROOT / "bloodBath" / "logs" / "inference_latest.json"
ARCHIVE_DIR   = REPO_ROOT / "bloodBath" / "logs" / "inference_archive"
MERGED_DIR    = REPO_ROOT / "bloodBath" / "bloodBank" / "merged"
RAW_LSTM_DIR  = REPO_ROOT / "bloodBath" / "bloodBank" / "raw" / "lstm"
TEMPLATES     = Path(__file__).parent / "templates"
ARCHIVE_MAX   = 96   # keep 8h @ 5-min intervals

HOST  = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT  = int(os.environ.get("DASHBOARD_PORT", "8765"))
PUMP  = os.environ.get("PUMP", "1539514")

STALE_SECONDS = 90 * 60   # Tandem API lag is 15–40 min; only warn if truly broken

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES)),
    autoescape=False,        # SVG injected via | safe; data is all internal
    undefined=jinja2.Undefined,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _maybe_archive(data: dict) -> None:
    """Save a timestamped copy of the current inference JSON for diagnostics."""
    ts = data.get("timestamp", "")
    if not ts:
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ts.replace(":", "").replace("+", "").replace("-", "")[:15]
    dest = ARCHIVE_DIR / f"inference_{safe}.json"
    if not dest.exists():
        try:
            dest.write_text(json.dumps(data))
            # Prune oldest beyond ARCHIVE_MAX
            files = sorted(ARCHIVE_DIR.glob("inference_*.json"))
            for old in files[:-ARCHIVE_MAX]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


def _load_archive() -> list[dict]:
    if not ARCHIVE_DIR.exists():
        return []
    entries = []
    for f in sorted(ARCHIVE_DIR.glob("inference_*.json"))[-ARCHIVE_MAX:]:
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            pass
    return entries


def _load_inference() -> tuple[dict, str | None]:
    try:
        with open(LATEST_JSON) as f:
            return json.load(f), None
    except FileNotFoundError:
        return {}, f"No inference output at {LATEST_JSON}. Has the cron run yet?"
    except json.JSONDecodeError as e:
        return {}, f"Inference JSON invalid (mid-write?): {e}"


def _load_history_df() -> pd.DataFrame | None:
    """Load last 300 rows from the newest test CSV only — fast, avoids reading train."""
    test_dir = MERGED_DIR / f"pump_{PUMP}" / "test"
    if not test_dir.exists():
        return None
    csvs = sorted(test_dir.glob(f"lstm_test_*.csv"))
    if not csvs:
        return None
    # Read only the newest test file, take its last 300 rows.
    df = pd.read_csv(csvs[-1], comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.iloc[-300:].reset_index(drop=True)


def _last_sync_mtime() -> datetime | None:
    files = list(RAW_LSTM_DIR.glob(f"pump_{PUMP}_*.csv"))
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    return datetime.fromtimestamp(os.path.getmtime(newest), tz=timezone.utc)


def _fmt(v, nd: int = 1) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "n/a"


def _ci_str(ci: dict, horizon: int) -> str:
    pair = (ci or {}).get(str(horizon)) or (ci or {}).get(horizon)
    if not pair or len(pair) != 2:
        return "n/a"
    return f"[{_fmt(pair[0])} – {_fmt(pair[1])}]"


def _bg_class(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f < 70:   return "bg-low"
    if f > 180:  return "bg-high"
    if f > 130:  return "bg-ok"
    return "bg-tgt"


def _fan_rows(fan: list) -> list[dict]:
    rows = []
    def cell(seq, idx):
        return _fmt(seq[idx]) if seq and idx < len(seq) else "—"
    def ci_cell(lo, up, idx):
        if lo and up and idx < len(lo) and idx < len(up):
            return f"[{_fmt(lo[idx])} – {_fmt(up[idx])}]"
        return ""
    for s in fan:
        p  = s.get("predicted_bg") or []
        lo = s.get("lower") or []
        up = s.get("upper") or []
        dose = float(s.get("dose_u", 0.0))
        rows.append({
            "dose":    _fmt(dose, 2),
            "nominal": abs(dose) < 1e-9,
            "p5":   cell(p, 0),  "ci5":  ci_cell(lo, up, 0),
            "p30":  cell(p, 5),  "ci30": ci_cell(lo, up, 5),
            "p60":  cell(p, 11), "ci60": ci_cell(lo, up, 11),
        })
    return rows


def _point_at(fan: list, idx: int) -> str:
    nominal = next((s for s in fan if abs(float(s.get("dose_u", 1))) < 1e-9), fan[0] if fan else {})
    preds = nominal.get("predicted_bg") or []
    return _fmt(preds[idx]) if idx < len(preds) else "n/a"


# ── Page renderer ─────────────────────────────────────────────────────────────

def render_page() -> str:
    data, err = _load_inference()
    now       = datetime.now(timezone.utc)

    if err:
        tmpl = _env.from_string(
            '<html><body style="background:#0d1117;color:#e6edf3;font-family:sans-serif;padding:24px">'
            '<h2 style="color:#f85149">Dashboard error</h2><p>{{ err }}</p></body></html>'
        )
        return tmpl.render(err=err)

    # ── Timestamps ──────────────────────────────────────────────────────────
    ts_raw = data.get("timestamp")
    ts     = None
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    age_sec   = (now - ts).total_seconds() if ts else None
    stale     = age_sec is not None and age_sec > STALE_SECONDS
    ts_disp   = ts.astimezone().strftime("%Y-%m-%d %H:%M %Z") if ts else "unknown"
    age_disp  = (
        f"{int(age_sec // 60)}m ago" if age_sec is not None and age_sec < 3600
        else f"{int(age_sec // 3600)}h {int((age_sec % 3600) // 60)}m ago"
        if age_sec is not None else "unknown"
    )

    sync_ts   = _last_sync_mtime()
    if sync_ts:
        sync_age = int((now - sync_ts).total_seconds())
        sync_age_disp = (f"{sync_age // 60}m ago" if sync_age < 3600
                         else f"{sync_age // 3600}h {(sync_age % 3600) // 60}m ago")
        sync_disp = f"{sync_ts.astimezone().strftime('%H:%M %Z')} ({sync_age_disp})"
    else:
        sync_disp = "unknown"

    now_local   = now.astimezone()
    server_time = (f"{now_local.strftime('%H:%M:%S %Z')} "
                   f"({now.strftime('%H:%M:%S UTC')})")

    # ── Inference fields ────────────────────────────────────────────────────
    ci  = data.get("conformal_intervals") or {}
    fan = data.get("fan") or []
    fms = data.get("failure_modes") or []
    top = fms[0] if fms else {}
    top_prob = top.get("probability", top.get("prob"))

    current_bg = _fmt(data.get("current_bg"))
    p30        = _point_at(fan, 5)
    p60        = _point_at(fan, 11)

    # ── Archive + Charts ────────────────────────────────────────────────────
    if data:
        _maybe_archive(data)
    archive     = _load_archive()
    history_df  = _load_history_df()
    fan_svg     = render_fan_svg(fan)
    history_svg = render_history_svg(history_df)
    diag_svg    = render_diagnostic_svg(history_df, archive)

    tmpl = _env.get_template("index.html.j2")
    return tmpl.render(
        stale        = stale,
        ts_disp      = ts_disp,
        age_disp     = age_disp,
        sync_disp    = sync_disp,
        server_time  = server_time,
        corrected    = data.get("corrected_in_horizon"),
        current_bg   = current_bg,
        bg_class     = _bg_class(data.get("current_bg")),
        p30          = p30,
        p30_class    = _bg_class(p30),
        p60          = p60,
        p60_class    = _bg_class(p60),
        ci30         = _ci_str(ci, 30),
        ci60         = _ci_str(ci, 60),
        top_state    = top.get("state", "n/a"),
        top_prob     = _fmt(top_prob, 2) if top_prob is not None else "n/a",
        isf          = _fmt(data.get("isf")),
        ecd          = _fmt(data.get("ecd_u"), 2),
        iob          = _fmt(data.get("iob", 0.0), 2),
        meal_prob    = _fmt(data.get("meal_probability"), 2),
        meal_prob_disp = _fmt(data.get("meal_probability"), 2),
        model_trigger  = str(data.get("model_trigger", ""))[:60],
        fan_rows     = _fan_rows(fan),
        fan_svg      = fan_svg,
        history_svg  = history_svg,
        diag_svg     = diag_svg,
        archive_count = len(archive),
        json_path    = str(LATEST_JSON),
    )


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = render_page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silence per-request logs


if __name__ == "__main__":
    print(f"bloodBender dashboard → http://{HOST}:{PORT}/")
    with ThreadingHTTPServer((HOST, PORT), Handler) as srv:
        srv.serve_forever()
