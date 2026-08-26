"""Turn a multi-seed sweep into publication-ready tables and findings.

Reads ``multiseed_raw.csv`` (one row per policy x seed) and emits:

* a Markdown results table of every policy with mean +/- std,
* per-axis marginal means (cadence alone, data-selection alone), which is how
  the factorial design attributes an effect to a *single* variable,
* Welch t-tests between the best and worst policy on the primary outcome, so
  the headline claim carries a p-value rather than an eyeballed gap,
* a plain-language findings block.

Usage:
    python experiments/analyze_results.py --raw results/ember_multiseed/multiseed_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = "mean_evasion_tail"          # settled evasion rate: lower is better
COST = "total_retrain_seconds"


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t, two-sided p)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = a.size, b.size
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    try:
        from scipy import stats

        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        # Normal approximation when scipy is unavailable.
        from math import erfc, sqrt

        p = erfc(abs(t) / sqrt(2))
    return float(t), float(p)


def _fmt(m: float, s: float, nd: int = 3) -> str:
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def analyze(raw_csv: Path, out_md: Path | None = None) -> str:
    df = pd.read_csv(raw_csv)
    n_seeds = df["seed"].nunique()
    lines: list[str] = []
    add = lines.append

    add(f"# Retraining-policy comparison ({n_seeds} seeds)\n")
    add(f"Source: `{raw_csv}`  ·  {len(df)} runs "
        f"({df['policy'].nunique()} policies x {n_seeds} seeds)\n")

    # ---- main table ---------------------------------------------------------
    add("\n## Per-policy results (mean ± std over seeds)\n")
    add("| Cadence | Data selection | Settled evasion ↓ | Attack success ↓ | "
        "Oscillation ↓ | Retrains | Cost (s) |")
    add("|---|---|---|---|---|---|---|")
    g = df.groupby(["cadence", "data_selection"])
    rows = []
    for (cad, sel), sub in g:
        row = {
            "cadence": cad, "data_selection": sel,
            PRIMARY: sub[PRIMARY].mean(), f"{PRIMARY}_std": sub[PRIMARY].std(ddof=1),
            "asr": sub.get("mean_attack_success_tail", pd.Series([np.nan])).mean(),
            "asr_std": sub.get("mean_attack_success_tail", pd.Series([np.nan])).std(ddof=1),
            "osc": sub["oscillation_index"].mean(), "osc_std": sub["oscillation_index"].std(ddof=1),
            "retrains": sub["retrain_count"].mean(),
            COST: sub[COST].mean(), f"{COST}_std": sub[COST].std(ddof=1),
        }
        rows.append(row)
        add(f"| {cad} | {sel} | {_fmt(row[PRIMARY], row[f'{PRIMARY}_std'])} | "
            f"{_fmt(row['asr'], row['asr_std'])} | {_fmt(row['osc'], row['osc_std'])} | "
            f"{row['retrains']:.1f} | {_fmt(row[COST], row[f'{COST}_std'], 1)} |")
    tab = pd.DataFrame(rows)

    # ---- marginal effects (the factorial payoff) ----------------------------
    add("\n## Marginal effect of each policy axis\n")
    add("Averaging over the other axis isolates one variable at a time -- the "
        "point of running a factorial design rather than one-off comparisons.\n")
    for axis in ["cadence", "data_selection"]:
        add(f"\n**{axis}**\n")
        add("| Level | Settled evasion ↓ | Oscillation ↓ | Cost (s) |")
        add("|---|---|---|---|")
        for level, sub in df.groupby(axis):
            add(f"| {level} | {_fmt(sub[PRIMARY].mean(), sub[PRIMARY].std(ddof=1))} | "
                f"{_fmt(sub['oscillation_index'].mean(), sub['oscillation_index'].std(ddof=1))} | "
                f"{_fmt(sub[COST].mean(), sub[COST].std(ddof=1), 1)} |")

    # ---- significance -------------------------------------------------------
    order = tab.sort_values(PRIMARY)
    best = order.iloc[0]
    worst = order.iloc[-1]
    b = df[(df.cadence == best.cadence) & (df.data_selection == best.data_selection)][PRIMARY].values
    w = df[(df.cadence == worst.cadence) & (df.data_selection == worst.data_selection)][PRIMARY].values
    t, p = _welch_t(b, w)
    add("\n## Significance of the headline gap\n")
    add(f"- Most robust: **{best.cadence} / {best.data_selection}** "
        f"(settled evasion {best[PRIMARY]:.3f})")
    add(f"- Least robust: **{worst.cadence} / {worst.data_selection}** "
        f"(settled evasion {worst[PRIMARY]:.3f})")
    add(f"- Welch t-test: t = {t:.3f}, p = {p:.4g} "
        f"({'significant' if np.isfinite(p) and p < 0.05 else 'not significant'} at α=0.05)")

    # ---- efficiency frontier -----------------------------------------------
    add("\n## Robustness-per-cost\n")
    add("| Cadence | Data selection | Settled evasion | Cost (s) | Evasion avoided per retrain-second |")
    add("|---|---|---|---|---|")
    for _, r in order.iterrows():
        eff = (1.0 - r[PRIMARY]) / r[COST] if r[COST] > 0 else float("inf")
        add(f"| {r.cadence} | {r.data_selection} | {r[PRIMARY]:.3f} | {r[COST]:.1f} | {eff:.4f} |")

    text = "\n".join(lines)
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    # The report uses non-ASCII (+/- and arrows), which the default Windows
    # console codepage cannot encode -- printing it would raise
    # UnicodeEncodeError and lose the whole report. The file is always written
    # as UTF-8; this only makes the echo to stdout safe.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="path to multiseed_raw.csv")
    ap.add_argument("--out", default=None, help="write Markdown here")
    a = ap.parse_args()
    md = analyze(Path(a.raw), Path(a.out) if a.out else None)
    print(md)


if __name__ == "__main__":
    main()
