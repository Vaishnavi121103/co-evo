"""Test whether fine-tuning is genuinely less *stable*, not merely worse.

Reporting that one arm's settled evasion "ranges from 0.29 to 0.999 while the
other sits between 0.038 and 0.043" is an eyeballed observation. This turns it
into a hypothesis test.

The comparison has to be made on **cell-centred residuals**, not on the raw
values. Each policy cell has its own mean, and the arms differ in mean by a
large margin; feeding raw numbers to a variance test would let that difference
in location leak into the dispersion estimate. Subtracting each cell's own mean
leaves only seed-to-seed variation within an otherwise identical
configuration -- which is exactly the run-to-run instability at issue.

Reported:

* **Levene / Brown-Forsythe** (median-centred, robust to non-normality, which
  matters because the residuals here are not normal).
* **Bartlett**, which is more powerful but assumes normality, as a
  cross-check rather than as the headline.
* An **F-test of variances** with an exact confidence interval on the ratio,
  since the effect size (how many times more variable) is more informative
  than the p-value.
* The **coefficient of variation** per arm, which is the scale-free version of
  the claim and immune to the arms' different means.

Usage
-----
    python experiments/variance_test.py
    python experiments/variance_test.py --out docs/variance.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DV = "mean_evasion_tail"
KEYS = ["cadence", "data_selection"]


def _complete(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    per = df.groupby("seed").size()
    return df[df.seed.isin(per[per == per.max()].index)]


def residuals(df: pd.DataFrame) -> np.ndarray:
    """Seed-to-seed deviations within each policy cell."""
    out = []
    for _, g in df.groupby(KEYS):
        out.append(g[DV].values - g[DV].mean())
    return np.concatenate(out)


def report(out: Path | None) -> str:
    seeds = [0, 1, 2]
    scr = _complete(RES / "ember_multiseed" / "multiseed_raw.csv")
    scr = scr[scr.seed.isin(seeds)]
    ft = _complete(RES / "ember_mode_costmatched" / "multiseed_raw.csv")
    ft = ft[ft.seed.isin(seeds)]

    a, b = residuals(scr), residuals(ft)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    # cells contribute (n_seeds - 1) df each
    dfa, dfb = na - scr.groupby(KEYS).ngroups, nb - ft.groupby(KEYS).ngroups

    lev_W, lev_p = stats.levene(a, b, center="median")
    bart_W, bart_p = stats.bartlett(a, b)
    F = vb / va
    p_F = 2 * min(stats.f.sf(F, dfb, dfa), stats.f.cdf(F, dfb, dfa))
    lo = F / stats.f.ppf(0.975, dfb, dfa)
    hi = F / stats.f.ppf(0.025, dfb, dfa)

    L: list[str] = []
    add = L.append
    add("# Is fine-tuning less stable, or just worse?\n")
    add(f"Both arms over seeds {seeds}, {scr.groupby(KEYS).ngroups} policy cells "
        "each. Tests run on **cell-centred residuals**, so the arms' very "
        "different means cannot leak into the dispersion estimate; what is left "
        "is seed-to-seed variation within an otherwise identical "
        "configuration.\n")

    add("\n## Dispersion\n")
    add("| Arm | Mean | SD of residuals | Variance | CV | Range |")
    add("|---|---|---|---|---|---|")
    for name, d, r in (("refit from scratch", scr, a), ("fine-tune", ft, b)):
        add(f"| {name} | {d[DV].mean():.4f} | {r.std(ddof=1):.4f} | "
            f"{r.var(ddof=1):.6f} | {r.std(ddof=1)/d[DV].mean():.3f} | "
            f"{d[DV].min():.4f}–{d[DV].max():.4f} |")

    add("\n## Tests of equal variance\n")
    add("| Test | Statistic | p | Verdict |")
    add("|---|---|---|---|")
    add(f"| Levene (Brown-Forsythe, median-centred) | W = {lev_W:.3f} | "
        f"{lev_p:.3g} | {'variances differ' if lev_p < .05 else 'no difference'} |")
    add(f"| Bartlett (assumes normality) | K² = {bart_W:.3f} | {bart_p:.3g} | "
        f"{'variances differ' if bart_p < .05 else 'no difference'} |")
    add(f"| F-test of variances | F({dfb},{dfa}) = {F:.2f} | {p_F:.3g} | "
        f"{'variances differ' if p_F < .05 else 'no difference'} |")

    add(f"\nFine-tuning's run-to-run variance is **{F:.0f}x** that of refitting "
        f"(95% CI {lo:.0f}–{hi:.0f}x). Levene is the test to quote: the "
        "residuals are not normal, and Bartlett is sensitive to that.")

    add("\n## Why this matters for the claim\n")
    add("The two arms differ in mean *and* in dispersion, and the second is a "
        "separate finding. A policy whose settled evasion lands anywhere "
        f"between {ft[DV].min():.3f} and {ft[DV].max():.3f} depending only on "
        "the seed is not simply a worse policy with a known cost -- it is "
        "unpredictable, and a defender cannot budget against it. Reporting a "
        "mean alone would hide exactly that. Refitting from scratch, by "
        f"contrast, lands within {scr[DV].min():.3f}–{scr[DV].max():.3f} "
        "across the same seeds.")

    text = "\n".join(L)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    print(report(a.out))


if __name__ == "__main__":
    main()
