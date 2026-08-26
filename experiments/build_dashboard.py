"""Build the standalone, locally-hosted results dashboard.

Reads only committed run artifacts -- ``multiseed_raw.csv`` and a pilot
``result.json`` -- and emits a single self-contained ``index.html``. No CDN, no
network calls, no server-side code: open the file directly or serve the folder
with any static server.

Why a build step rather than fetching the CSV in the browser: a page opened over
``file://`` cannot ``fetch`` a sibling file (the browser treats it as a
cross-origin request), so the data is inlined at build time. That also makes the
output a single portable file you can copy onto a presentation machine.

Only *fully complete* seeds are included, so the dashboard never reports a mean
over an uneven number of runs while a sweep is still in progress.

Usage
-----
    python experiments/build_dashboard.py
    python experiments/build_dashboard.py --serve          # build then serve
    python experiments/build_dashboard.py --port 8020 --serve
"""

from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "dashboard_template.html"
DEFAULT_RAW = ROOT / "results" / "ember_multiseed" / "multiseed_raw.csv"
DEFAULT_PILOT = ROOT / "results" / "ember_pilot_v6" / "result.json"
DEFAULT_OUT = ROOT / "results" / "dashboard" / "index.html"


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test; falls back to a normal approximation without scipy."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"t": float("nan"), "p": float("nan"), "df": float("nan")}
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), a.size, b.size
    se = sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": float("nan"), "p": float("nan"), "df": float("nan")}
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        p = float(erfc(abs(t) / sqrt(2)))
    return {"t": round(float(t), 3), "p": round(p, 5), "df": round(float(df), 2)}


def collect(raw_csv: Path, pilot_json: Path) -> dict:
    df = pd.read_csv(raw_csv)
    n_cells = df.groupby("seed").size().max()
    complete = sorted(int(s) for s, g in df.groupby("seed") if len(g) == n_cells)
    if not complete:
        raise SystemExit(f"No complete seeds yet in {raw_csv}")
    df = df[df.seed.isin(complete)]

    policies = []
    for (cad, sel), s in df.groupby(["cadence", "data_selection"]):
        policies.append(
            dict(
                cadence=cad,
                selection=sel,
                evasion=round(s.mean_evasion_tail.mean(), 4),
                evasion_sd=round(float(s.mean_evasion_tail.std(ddof=1)), 4),
                asr=round(s.mean_attack_success_tail.mean(), 4),
                osc=round(s.oscillation_index.mean(), 4),
                osc_sd=round(float(s.oscillation_index.std(ddof=1)), 4),
                cost=round(s.total_retrain_seconds.mean(), 1),
                cost_sd=round(float(s.total_retrain_seconds.std(ddof=1)), 1),
                retrains=round(s.retrain_count.mean(), 1),
            )
        )

    def marginal(col: str) -> list[dict]:
        return [
            dict(
                level=lvl,
                evasion=round(s.mean_evasion_tail.mean(), 4),
                evasion_sd=round(float(s.mean_evasion_tail.std(ddof=1)), 4),
                osc=round(s.oscillation_index.mean(), 4),
                osc_sd=round(float(s.oscillation_index.std(ddof=1)), 4),
                cost=round(s.total_retrain_seconds.mean(), 1),
                cost_sd=round(float(s.total_retrain_seconds.std(ddof=1)), 1),
            )
            for lvl, s in df.groupby(col)
        ]

    order = sorted(policies, key=lambda p: p["evasion"])
    best, worst = order[0], order[-1]

    def series(p: dict) -> np.ndarray:
        m = (df.cadence == p["cadence"]) & (df.data_selection == p["selection"])
        return df[m].mean_evasion_tail.values

    pilot = json.loads(pilot_json.read_text(encoding="utf-8"))
    return dict(
        seeds=complete,
        n_seeds=len(complete),
        n_runs=int(len(df)),
        policies=policies,
        marg_cadence=marginal("cadence"),
        marg_selection=marginal("data_selection"),
        best={k: best[k] for k in ("cadence", "selection", "evasion", "cost")},
        worst={k: worst[k] for k in ("cadence", "selection", "evasion", "cost")},
        welch=_welch(series(best), series(worst)),
        rounds=[
            dict(
                r=x["round"],
                ev=round(x["evasion_rate"], 4),
                asr=round(x["attack_success_rate"], 4),
                q=round(x["mean_queries"], 2),
                acc=round(x["clean_accuracy"], 4),
            )
            for x in pilot["rounds"]
        ],
    )


# Standalone wrapper. The template is authored as bare page content (it is also
# published as a hosted artifact, where the skeleton is supplied); for local use
# it needs a real document, a CSS reset, and a theme control -- there is no host
# stamping data-theme on the root element here.
SKELETON = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;padding:0}
  img,svg{max-width:100%}
</style>
__CONTENT__
<style>
  #themebtn{position:fixed;top:14px;right:16px;z-index:80;font-family:var(--f-mono);
    font-size:11px;padding:6px 11px;border-radius:7px;cursor:pointer;
    background:var(--panel);color:var(--ink-2);border:1px solid var(--line);box-shadow:var(--shadow)}
  #themebtn:hover{color:var(--ink)}
  @media print{#themebtn,.rail{display:none} .shell{grid-template-columns:1fr}
    section{break-inside:avoid} main{max-width:none;padding:0}}
</style>
<button id="themebtn" type="button" aria-label="Switch colour theme">theme</button>
<script>
(function(){
  var KEY="coevomal-theme", root=document.documentElement, btn=document.getElementById("themebtn");
  try{ var saved=localStorage.getItem(KEY); if(saved) root.setAttribute("data-theme",saved); }catch(e){}
  function current(){
    var a=root.getAttribute("data-theme");
    if(a) return a;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function label(){ btn.textContent = current()==="dark" ? "light mode" : "dark mode"; }
  label();
  btn.addEventListener("click",function(){
    var next = current()==="dark" ? "light" : "dark";
    root.setAttribute("data-theme",next);
    try{ localStorage.setItem(KEY,next); }catch(e){}
    label();
  });
})();
</script>
</body>
</html>
"""


def build(raw_csv: Path, pilot_json: Path, out: Path) -> Path:
    data = collect(raw_csv, pilot_json)
    content = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    # Move the <title>/<link>/<style> preamble into <head>, keep the rest in <body>.
    marker = '<div class="shell">'
    head_part, body_part = content.split(marker, 1)
    page = SKELETON.replace(
        "__CONTENT__", head_part + "</head>\n<body>\n" + marker + body_part
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--serve", action="store_true", help="serve after building")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    out = build(args.raw, args.pilot, args.out)
    size = out.stat().st_size / 1024
    data = collect(args.raw, args.pilot)
    print(f"built {out}  ({size:.0f} KB, self-contained)")
    print(f"  seeds {data['seeds']}  ·  {data['n_runs']} runs  ·  "
          f"{len(data['policies'])} policies")
    print(f"  open directly:  {out.as_uri()}")

    if args.serve:
        import functools
        import http.server
        import socketserver

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(out.parent)
        )
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
            print(f"\nserving on http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nstopped")


if __name__ == "__main__":
    main()
