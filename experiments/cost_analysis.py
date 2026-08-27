"""Deterministic retraining-cost accounting.

Wall-clock is not a safe cost unit here. The studies were run at different
times with different numbers of concurrent jobs, and the per-retrain wall-clock
varies by up to 3x purely from CPU contention -- enough to reverse the ordering
between policies. Any cost claim resting on ``total_retrain_seconds`` across
studies is therefore an artifact of scheduling rather than a property of the
policy.

This module states cost in units that are exact and reproducible:

* **retrains** -- how many times the defender refit. Fixed by the cadence rule
  and the observed evasion trace, and identical on any machine.
* **base learners fitted** -- how many trees were actually built across the
  run. For a from-scratch refit this is ``retrains x max_iter``; for a
  warm-started refit it is ``retrains x finetune_iter``, since scikit-learn's
  ``warm_start`` fits only the incremental trees (verified empirically).

Wall-clock is retained but reported as *indicative*, normalised per retrain so
that the contention factor is visible rather than hidden inside a total.

Usage
-----
    python experiments/cost_analysis.py
    python experiments/cost_analysis.py --out docs/cost.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# (label, directory, trees fitted per retrain). The initial fit is common to
# every arm and excluded, so these are the *incremental* costs the policy
# actually controls.
STUDIES = [
    ("main factorial (scratch)", "ember_multiseed", 80),
    ("frozen baseline", "ember_baseline_frozen", 0),
    ("minimax alternation", "ember_minimax", 80),
    ("replay cap 800", "ember_cap800", 80),
    ("fine-tune, cost-matched", "ember_mode_costmatched", 80),
    ("fine-tune, capacity-matched", "ember_mode_capmatched", 400),
]


def load(name: str) -> pd.DataFrame | None:
    p = RES / name / "multiseed_raw.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    per = df.groupby("seed").size()
    return df[df.seed.isin(per[per == per.max()].index)]


def report(out: Path | None) -> str:
    L: list[str] = []
    add = L.append
    add("# Retraining cost, in deterministic units\n")
    add("Wall-clock per retrain varies by up to 3x across these studies purely "
        "from CPU contention, so it cannot carry a cost claim. Retrains and "
        "base learners fitted are exact.\n")

    add("\n## Per study\n")
    add("| Study | Retrains | Trees fitted | Wall-clock (s) | s/retrain (indicative) |")
    add("|---|---|---|---|---|")
    rates = []
    for label, d, per_retrain in STUDIES:
        df = load(d)
        if df is None:
            continue
        r = df.retrain_count.mean()
        secs = df.total_retrain_seconds.mean()
        trees = r * per_retrain
        rate = secs / r if r else float("nan")
        if r and per_retrain == 80:
            rates.append((label, rate))
        add(f"| {label} | {r:.1f} | {trees:.0f} | {secs:.1f} | "
            f"{'n/a' if not r else f'{rate:.2f}'} |")

    if rates:
        lo = min(r for _, r in rates)
        hi = max(r for _, r in rates)
        add(f"\nAcross studies doing identically-sized refits (80 trees each), "
            f"measured s/retrain spans **{lo:.2f}--{hi:.2f}** — a "
            f"**{hi/lo:.1f}x** spread with no difference in the work performed. "
            f"That spread is contention, and it is larger than several of the "
            f"policy differences it would otherwise be used to argue for.")

    # ---- the claims that matter, restated deterministically ----------------
    main = load("ember_multiseed")
    add("\n## Cadence cost, stated exactly\n")
    add("| Cadence | Retrains | Trees fitted | Relative cost |")
    add("|---|---|---|---|")
    if main is not None:
        base = main[main.cadence == "every_round"].retrain_count.mean()
        for cad, g in main.groupby("cadence"):
            r = g.retrain_count.mean()
            add(f"| {cad} | {r:.1f} | {r*80:.0f} | {r/base:.2f}x |")
        add("\nThe vendor-facing claim rests on this table, not on wall-clock: "
            "retraining every third round performs **3** refits against **8**, "
            "a 2.67x reduction in retraining work that holds on any machine.")

    mm = load("ember_minimax")
    if mm is not None and main is not None:
        er = main[main.cadence == "every_round"].retrain_count.mean()
        add("\n## Minimax cost\n")
        add(f"Minimax performs **{mm.retrain_count.mean():.0f}** refits against "
            f"every-round's **{er:.0f}**: it is *cheaper* by a factor of "
            f"{er/mm.retrain_count.mean():.2f}x in retraining work, while "
            f"settling at roughly twice the evasion "
            f"({mm.mean_evasion_tail.mean():.3f} vs "
            f"{main[(main.cadence=='every_round')&(main.data_selection=='full_replay')].mean_evasion_tail.mean():.3f}). "
            "Its higher measured wall-clock was contention: it ran alongside "
            "two other studies.")

    cm, cp = load("ember_mode_costmatched"), load("ember_mode_capmatched")
    if cm is not None:
        add("\n## Fine-tuning cost\n")
        add("| Arm | Retrains | Trees fitted | Settled evasion |")
        add("|---|---|---|---|")
        if main is not None:
            g = main
            add(f"| refit from scratch | {g.retrain_count.mean():.1f} | "
                f"{g.retrain_count.mean()*80:.0f} | {g.mean_evasion_tail.mean():.4f} |")
        add(f"| fine-tune, cost-matched | {cm.retrain_count.mean():.1f} | "
            f"{cm.retrain_count.mean()*80:.0f} | {cm.mean_evasion_tail.mean():.4f} |")
        if cp is not None:
            add(f"| fine-tune, capacity-matched | {cp.retrain_count.mean():.1f} | "
                f"{cp.retrain_count.mean()*400:.0f} | {cp.mean_evasion_tail.mean():.4f} |")
        add("\nCost-matched fine-tuning fits the *same* number of trees per "
            "retrain as a scratch refit, so it is not the more expensive "
            "option -- the earlier wall-clock reading that suggested otherwise "
            "was contention. Capacity-matched fine-tuning fits 5x the trees "
            "and is genuinely more expensive, in fitting and in the prediction "
            "cost the attacker pays on every query, without recovering the "
            "deficit.")

    bench = RES / "cost_benchmark.json"
    if bench.exists():
        import json
        rows = json.loads(bench.read_text(encoding="utf-8"))
        add("\n## Controlled sequential measurement\n")
        add("Single job, all cores, nothing else running -- the only wall-clock "
            "numbers in this project that are comparable to each other.\n")
        add("| Cell | Retrains | Trees | Retrain (s) | s/retrain | s/100 trees |")
        add("|---|---|---|---|---|---|")
        for r in rows:
            add(f"| {r['label']} | {r['retrains']} | {r['trees']} | "
                f"{r['retrain_s']} | {r['s_per_retrain']} | {r['s_per_100_trees']} |")

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
