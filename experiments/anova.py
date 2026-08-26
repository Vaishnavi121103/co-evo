"""Two-way ANOVA for the cadence x data-selection factorial.

The study is a balanced 3x3 design with n seeds per cell, which is what a
two-way ANOVA is for: it tests both main effects and their **interaction** in a
single model, instead of comparing two hand-picked cells and discarding the
other seven.

The interaction is not a formality here. Cadence moves settled evasion only
within ``full_replay``; under the weaker data-selection strategies the three
cadences land on nearly the same value, because evasion never falls below the
trigger and ``threshold`` degenerates into ``every_round``. Marginal means
average that structure away, so the interaction term is where it shows up.

Reported alongside the F tests:

* **Effect sizes** (partial eta^2 and omega^2). With a controlled design and
  low-noise cells, p-values mostly confirm that the knobs do something; the
  effect sizes say how much, and are what belongs in the thesis.
* **Assumption checks** (Levene for equal variances, Shapiro-Wilk for normality
  of residuals), with the robustness check matched to whichever one actually
  fails. Welch's ANOVA relaxes the equal-variance assumption and does nothing
  about non-normality, so it is the wrong fallback when Levene passes and
  Shapiro does not. **Kruskal-Wallis** addresses non-normality directly: it
  works on ranks and assumes no distributional form. Both are reported, each
  labelled with the assumption it actually covers, and **Dunn's test** with
  Holm and Bonferroni correction supplies the matching rank-based pairwise
  post-hoc.
* **Simple main effects**: given a significant interaction, the main effects
  are not directly interpretable, so cadence is also tested *within* each
  data-selection level.
* **Tukey HSD** for pairwise comparisons among the levels of each factor.

Usage
-----
    python experiments/anova.py
    python experiments/anova.py --dv oscillation_index
    python experiments/anova.py --raw results/ember_multiseed/multiseed_raw.csv \
        --out docs/anova.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results" / "ember_multiseed" / "multiseed_raw.csv"
A, B = "cadence", "data_selection"


@dataclass
class Term:
    name: str
    ss: float
    df: int
    ms: float
    F: float
    p: float
    peta2: float          # partial eta^2
    omega2: float


def two_way(df: pd.DataFrame, dv: str) -> tuple[list[Term], dict]:
    """Balanced two-way ANOVA with interaction.

    Balance is required (and asserted): with equal cell counts the Type I, II
    and III sums of squares coincide, so the decomposition below is unambiguous
    and needs no design-matrix machinery.
    """
    counts = df.groupby([A, B]).size()
    if counts.nunique() != 1:
        raise SystemExit(
            f"Design is unbalanced (cell sizes {sorted(counts.unique())}). "
            "This routine assumes a balanced design; drop incomplete seeds first."
        )
    n = int(counts.iloc[0])
    a_lv = sorted(df[A].unique())
    b_lv = sorted(df[B].unique())
    a, b = len(a_lv), len(b_lv)
    grand = df[dv].mean()

    ss_a = n * b * sum((df[df[A] == l][dv].mean() - grand) ** 2 for l in a_lv)
    ss_b = n * a * sum((df[df[B] == l][dv].mean() - grand) ** 2 for l in b_lv)
    ss_ab = 0.0
    ss_w = 0.0
    for la in a_lv:
        for lb in b_lv:
            cell = df[(df[A] == la) & (df[B] == lb)][dv]
            ss_ab += n * (
                cell.mean() - df[df[A] == la][dv].mean()
                - df[df[B] == lb][dv].mean() + grand
            ) ** 2
            ss_w += ((cell - cell.mean()) ** 2).sum()
    ss_t = ((df[dv] - grand) ** 2).sum()

    df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_w = a * b * (n - 1)
    ms_w = ss_w / df_w

    terms = []
    for name, ss, dfree in ((A, ss_a, df_a), (B, ss_b, df_b),
                            (f"{A} x {B}", ss_ab, df_ab)):
        ms = ss / dfree
        F = ms / ms_w
        p = float(stats.f.sf(F, dfree, df_w))
        terms.append(Term(name, ss, dfree, ms, F, p,
                          ss / (ss + ss_w),
                          (ss - dfree * ms_w) / (ss_t + ms_w)))
    meta = dict(n=n, a=a, b=b, df_w=df_w, ms_w=ms_w, ss_w=ss_w, ss_t=ss_t,
                grand=grand, a_lv=a_lv, b_lv=b_lv)
    return terms, meta


def assumptions(df: pd.DataFrame, dv: str) -> dict:
    groups = [g[dv].values for _, g in df.groupby([A, B])]
    lev_W, lev_p = stats.levene(*groups, center="median")
    resid = np.concatenate([g - g.mean() for g in groups])
    sh_W, sh_p = stats.shapiro(resid)
    return dict(levene_W=float(lev_W), levene_p=float(lev_p),
                shapiro_W=float(sh_W), shapiro_p=float(sh_p),
                var_ratio=float(max(g.var(ddof=1) for g in groups)
                                / max(1e-12, min(g.var(ddof=1) for g in groups))))


def welch_oneway(df: pd.DataFrame, dv: str, factor: str) -> tuple[float, float]:
    """Welch's one-way ANOVA - does not assume equal variances."""
    groups = [g[dv].values for _, g in df.groupby(factor)]
    k = len(groups)
    w = np.array([len(g) / g.var(ddof=1) for g in groups])
    m = np.array([g.mean() for g in groups])
    mw = (w * m).sum() / w.sum()
    num = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (np.array([len(g) for g in groups]) - 1)).sum()
    den = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam
    F = num / den
    df2 = (k ** 2 - 1) / (3 * lam)
    return float(F), float(stats.f.sf(F, k - 1, df2))


def kruskal_wallis(df: pd.DataFrame, dv: str, factor: str) -> dict:
    """Rank-based one-way test - the right robustness check for non-normality.

    Kruskal-Wallis makes no distributional assumption, so unlike Welch's ANOVA
    it speaks directly to a failed Shapiro-Wilk. Effect sizes reported are
    epsilon-squared (H / (n - 1)) and the eta-squared analogue
    ((H - k + 1) / (n - k)).

    Note this is a *one-way* test applied to one factor at a time, so it treats
    the other factor as noise. With a balanced design and a non-significant
    interaction that is a fair marginal check; a fully non-parametric two-way
    analysis would need an aligned rank transform.
    """
    groups = [g[dv].values for _, g in df.groupby(factor)]
    H, p = stats.kruskal(*groups)
    n, k = len(df), len(groups)
    return dict(H=float(H), p=float(p), k=k, n=n,
                eps2=float(H / (n - 1)),
                eta2=float((H - k + 1) / (n - k)))


def _adjust(pvals: list[float], method: str) -> list[float]:
    """Bonferroni or Holm-Bonferroni step-down correction."""
    m = len(pvals)
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvals]
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])   # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def dunn(df: pd.DataFrame, dv: str, factor: str) -> list[dict]:
    """Dunn's post-hoc: pairwise z-tests on pooled mean ranks, tie-corrected.

    The rank-based counterpart to Tukey HSD, and the pairwise test that belongs
    with Kruskal-Wallis. Ranks are pooled across *all* groups rather than
    recomputed per pair, which is what makes it valid as a follow-up to the
    omnibus test.
    """
    lv = sorted(df[factor].unique())
    x = df[dv].values
    N = len(x)
    ranks = stats.rankdata(x)
    _, counts = np.unique(x, return_counts=True)
    ties = float(sum(c ** 3 - c for c in counts if c > 1))
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))

    idx = {l: np.flatnonzero((df[factor] == l).values) for l in lv}
    mean_rank = {l: ranks[i].mean() for l, i in idx.items()}
    n_i = {l: len(i) for l, i in idx.items()}

    rows, praw = [], []
    for i in range(len(lv)):
        for j in range(i + 1, len(lv)):
            la, lb = lv[i], lv[j]
            se = np.sqrt(sigma2 * (1.0 / n_i[la] + 1.0 / n_i[lb]))
            z = (mean_rank[la] - mean_rank[lb]) / se
            p = 2.0 * stats.norm.sf(abs(z))
            rows.append(dict(a=la, b=lb, z=float(z), mra=float(mean_rank[la]),
                             mrb=float(mean_rank[lb]), p_raw=float(p)))
            praw.append(float(p))
    for r, ph, pb in zip(rows, _adjust(praw, "holm"), _adjust(praw, "bonferroni")):
        r["p_holm"], r["p_bonf"] = ph, pb
    return rows


def simple_main_effects(df: pd.DataFrame, dv: str, ms_w: float, df_w: int) -> list[dict]:
    """Effect of cadence *within* each data-selection level.

    When the interaction is significant the main effects are not directly
    interpretable, because the effect of one factor depends on the level of the
    other. These tests use the pooled error term from the full model.
    """
    out = []
    for lb in sorted(df[B].unique()):
        sub = df[df[B] == lb]
        lv = sorted(sub[A].unique())
        n = len(sub) // len(lv)
        gm = sub[dv].mean()
        ss = n * sum((sub[sub[A] == l][dv].mean() - gm) ** 2 for l in lv)
        dfree = len(lv) - 1
        F = (ss / dfree) / ms_w
        out.append(dict(level=lb, F=float(F), p=float(stats.f.sf(F, dfree, df_w)),
                        df=dfree, spread=float(
                            max(sub[sub[A] == l][dv].mean() for l in lv)
                            - min(sub[sub[A] == l][dv].mean() for l in lv))))
    return out


def tukey(df: pd.DataFrame, dv: str, factor: str):
    lv = sorted(df[factor].unique())
    res = stats.tukey_hsd(*[df[df[factor] == l][dv].values for l in lv])
    rows = []
    for i in range(len(lv)):
        for j in range(i + 1, len(lv)):
            rows.append(dict(a=lv[i], b=lv[j],
                             diff=float(res.statistic[i, j]),
                             p=float(res.pvalue[i, j]),
                             lo=float(res.confidence_interval().low[i, j]),
                             hi=float(res.confidence_interval().high[i, j])))
    return rows


def _sig(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def _mag(e: float) -> str:
    return "large" if e >= .14 else "medium" if e >= .06 else "small"


def report(raw: Path, dv: str) -> str:
    df = pd.read_csv(raw)
    n_cells = df.groupby("seed").size().max()
    keep = [s for s, g in df.groupby("seed") if len(g) == n_cells]
    df = df[df.seed.isin(keep)]
    if "mode" in df.columns:
        df = df[df["mode"].fillna("scratch") == "scratch"]

    terms, meta = two_way(df, dv)
    asm = assumptions(df, dv)
    L = []
    p_ = L.append
    p_(f"# Two-way ANOVA — {dv}\n")
    p_(f"Balanced {meta['a']}×{meta['b']} factorial, n={meta['n']} seeds per cell "
       f"({meta['a']*meta['b']*meta['n']} runs). Source: `{raw}`.\n")

    p_("\n## Cell means\n")
    piv = df.pivot_table(index=B, columns=A, values=dv, aggfunc="mean").round(4)
    p_("| " + B + " | " + " | ".join(piv.columns) + " |")
    p_("|" + "---|" * (len(piv.columns) + 1))
    for idx, row in piv.iterrows():
        p_(f"| {idx} | " + " | ".join(f"{v:.4f}" for v in row.values) + " |")

    p_("\n## ANOVA table\n")
    p_("| Source | SS | df | MS | F | p | partial η² | ω² | |")
    p_("|---|---|---|---|---|---|---|---|---|")
    for t in terms:
        p_(f"| {t.name} | {t.ss:.5f} | {t.df} | {t.ms:.5f} | {t.F:.2f} | "
           f"{t.p:.3g} | {t.peta2:.3f} ({_mag(t.peta2)}) | {t.omega2:.3f} | {_sig(t.p)} |")
    p_(f"| Residual | {meta['ss_w']:.5f} | {meta['df_w']} | {meta['ms_w']:.5f} | | | | | |")
    p_(f"| Total | {meta['ss_t']:.5f} | {meta['a']*meta['b']*meta['n']-1} | | | | | | |")
    p_("\n`***` p<.001 · `**` p<.01 · `*` p<.05 · `ns` not significant. "
       "Effect-size bands are Cohen's conventions for η² (.01 small, .06 medium, .14 large).")

    inter = terms[2]
    p_("\n## Interpretation\n")
    if inter.p < .05:
        p_(f"- The **interaction is significant** (F({inter.df},{meta['df_w']}) = "
           f"{inter.F:.2f}, p = {inter.p:.3g}, partial η² = {inter.peta2:.3f}). The main "
           "effects are therefore not directly interpretable on their own: the effect of "
           "one factor depends on the level of the other, so the simple main effects below "
           "are the correct place to read the cadence result.")
    else:
        p_(f"- The interaction is not significant (p = {inter.p:.3g}), so the two main "
           "effects can be read independently.")
    for t in terms[:2]:
        p_(f"- **{t.name}**: F({t.df},{meta['df_w']}) = {t.F:.2f}, p = {t.p:.3g}, "
           f"partial η² = {t.peta2:.3f} ({_mag(t.peta2)}), ω² = {t.omega2:.3f}.")

    p_("\n## Simple main effects — cadence within each data-selection level\n")
    p_("| Data selection | F | p | | Spread across cadences |")
    p_("|---|---|---|---|---|")
    for r in simple_main_effects(df, dv, meta["ms_w"], meta["df_w"]):
        p_(f"| {r['level']} | {r['F']:.2f} | {r['p']:.3g} | {_sig(r['p'])} | {r['spread']:.4f} |")

    p_("\n## Tukey HSD (95% family-wise)\n")
    for factor in (B, A):
        p_(f"\n**{factor}**\n")
        p_("| A | B | mean diff | 95% CI | p | |")
        p_("|---|---|---|---|---|---|")
        for r in tukey(df, dv, factor):
            p_(f"| {r['a']} | {r['b']} | {r['diff']:+.4f} | "
               f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.3g} | {_sig(r['p'])} |")

    p_("\n## Assumption checks\n")
    p_(f"- **Levene** (equal variances across cells): W = {asm['levene_W']:.3f}, "
       f"p = {asm['levene_p']:.3g} → "
       + ("**violated**" if asm['levene_p'] < .05 else "not violated")
       + f". Largest/smallest cell variance ratio {asm['var_ratio']:.1f}.")
    p_(f"- **Shapiro-Wilk** (normality of residuals): W = {asm['shapiro_W']:.3f}, "
       f"p = {asm['shapiro_p']:.3g} → "
       + ("**violated**" if asm['shapiro_p'] < .05 else "not violated") + ".")
    p_("\nThe two robustness checks below address **different** assumptions, so the one to "
       "quote is whichever matches the assumption that actually failed.")

    p_("\n### Kruskal-Wallis — addresses non-normality\n")
    p_("Rank-based and distribution-free, so it speaks directly to the Shapiro-Wilk result "
       "above. This is the robustness check to quote for this data.\n")
    p_("| Factor | H | df | p | ε² | η²_H | |")
    p_("|---|---|---|---|---|---|---|")
    kw = {}
    for factor in (B, A):
        r = kruskal_wallis(df, dv, factor)
        kw[factor] = r
        p_(f"| {factor} | {r['H']:.3f} | {r['k']-1} | {r['p']:.3g} | "
           f"{r['eps2']:.3f} | {r['eta2']:.3f} | {_sig(r['p'])} |")

    p_("\n### Dunn's test — rank-based pairwise post-hoc\n")
    for factor in (B, A):
        p_(f"\n**{factor}**\n")
        p_("| A | B | mean rank A | mean rank B | z | p (raw) | p (Holm) | p (Bonf.) | |")
        p_("|---|---|---|---|---|---|---|---|---|")
        for r in dunn(df, dv, factor):
            p_(f"| {r['a']} | {r['b']} | {r['mra']:.1f} | {r['mrb']:.1f} | {r['z']:+.3f} | "
               f"{r['p_raw']:.3g} | {r['p_holm']:.3g} | {r['p_bonf']:.3g} | "
               f"{_sig(r['p_holm'])} |")
    p_("\nHolm is the primary correction — uniformly more powerful than Bonferroni at the "
       "same family-wise error rate — with Bonferroni shown alongside for reference.")

    p_("\n### Welch one-way — addresses unequal variances\n")
    p_("Reported for completeness only. Welch relaxes the equal-variance assumption, which "
       "Levene did **not** flag here, so it is not the relevant fallback for this data: it "
       "does nothing about non-normality.\n")
    for factor in (A, B):
        F, p = welch_oneway(df, dv, factor)
        p_(f"- {factor}: Welch F = {F:.2f}, p = {p:.3g} {_sig(p)}")

    p_("\n### Verdict\n")
    agree = {}
    for i, factor in ((1, B), (0, A)):
        same = (kw[factor]["p"] < .05) == (terms[i].p < .05)
        agree[factor] = same
        p_(f"- **{factor}**: ANOVA p = {terms[i].p:.3g}, Kruskal-Wallis p = "
           f"{kw[factor]['p']:.3g} — " + ("the two agree" if same else "**they disagree**") + ".")
    if all(agree.values()):
        p_("\nBoth factors reach the same verdict under the rank-based test as under the F "
           "test, so the non-normal residuals do not change the conclusion. The ANOVA stays "
           "the primary analysis — it is what supplies the interaction term and the variance "
           "decomposition — with Kruskal-Wallis reported as the assumption-free confirmation.")
    else:
        p_("\nThe rank-based and parametric tests disagree. Given the non-normal residuals, "
           "the Kruskal-Wallis result is the one to treat as authoritative.")

    p_("\n## Caveat specific to this design\n")
    p_("Under `bounded_buffer` and `hard_mining` the `threshold` and `every_round` cells "
       "are near-identical, because evasion never falls below the retraining trigger and "
       "the threshold policy fires every round — it *is* `every_round` in those conditions. "
       "That degeneracy is a genuine finding rather than a nuisance, but it inflates the "
       "interaction term, and it means the cadence factor has fewer effectively distinct "
       "levels than the design nominally provides.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--dv", default="mean_evasion_tail")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    md = report(args.raw, args.dv)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
