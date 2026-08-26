"""Multi-seed factorial sweep -> publishable mean +/- std tables.

Wraps the factorial policy sweep in an outer seed loop so every policy cell is
reported as a **mean +/- standard deviation** over N seeds, which is what a
dissertation / paper needs (a single seed is not evidence). Every seed shares
the identical data subsample per seed value, and all policies within a seed see
the same attacker initialisation stream, so differences remain attributable to
the policy variable.

Outputs
-------
* ``multiseed_raw.csv``     -- one row per (policy, seed) with full summary.
* ``multiseed_summary.csv`` -- one row per policy: mean & std of each metric.
* ``frontier.png``          -- robustness-vs-cost frontier on the seed means.

Usage
-----
    python experiments/run_multiseed.py --config configs/ember_factorial.yaml \
        --seeds 5 --out results/ember_multiseed
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from coevomal.config import ExperimentConfig
from coevomal.orchestrator import CoEvolutionOrchestrator

CADENCES = ["every_round", "every_n", "threshold"]
DATA_SELECTION = ["full_replay", "hard_mining", "bounded_buffer"]
ATTACKERS = ["dqn"]

# Metrics we aggregate across seeds.
METRICS = [
    "oscillation_index",
    "mean_evasion_tail",
    "final_evasion_rate",
    "max_evasion_rate",
    "rounds_to_convergence",
    "total_retrain_seconds",
    "retrain_count",
]


def run_multiseed(
    base: ExperimentConfig, out_dir: Path, seeds: int = 5, quiet: bool = True
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(ATTACKERS, CADENCES, DATA_SELECTION))
    raw_rows: list[dict] = []

    for s in range(seeds):
        for atk, cadence, selection in combos:
            name = f"{atk}__{cadence}__{selection}__seed{s}"
            cfg = base.replace(
                **{
                    "name": name,
                    "seed": s,
                    "dataset.seed": s,
                    "attacker.name": atk,
                    "attacker.seed": s,
                    "defender.seed": s,
                    "retrain.cadence": cadence,
                    "retrain.data_selection": selection,
                }
            )
            print(f"[seed {s}] {atk}/{cadence}/{selection} ...", flush=True)
            orch = CoEvolutionOrchestrator(cfg, verbose=not quiet)
            orch.run()
            summ = orch.result.summary(cfg.convergence_window, cfg.convergence_tol)
            row = {
                "policy": f"{atk}__{cadence}__{selection}",
                "seed": s,
                "attacker": atk,
                "cadence": cadence,
                "data_selection": selection,
            }
            for m in METRICS:
                v = summ.get(m)
                # rounds_to_convergence is None on divergence -> NaN for stats.
                row[m] = float("nan") if v is None else float(v)
            raw_rows.append(row)

    # ---- raw ---------------------------------------------------------------
    raw_path = out_dir / "multiseed_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"wrote {raw_path}")

    # ---- aggregate mean/std per policy -------------------------------------
    summary_rows: list[dict] = []
    for atk, cadence, selection in combos:
        policy = f"{atk}__{cadence}__{selection}"
        cells = [r for r in raw_rows if r["policy"] == policy]
        agg = {"policy": policy, "attacker": atk, "cadence": cadence,
               "data_selection": selection, "n_seeds": len(cells)}
        for m in METRICS:
            vals = np.array([r[m] for r in cells], dtype=float)
            agg[f"{m}_mean"] = float(np.nanmean(vals))
            agg[f"{m}_std"] = float(np.nanstd(vals))
        # convenience for the frontier plot
        agg["mean_evasion_tail"] = agg["mean_evasion_tail_mean"]
        agg["total_retrain_seconds"] = agg["total_retrain_seconds_mean"]
        summary_rows.append(agg)

    summ_path = out_dir / "multiseed_summary.csv"
    with summ_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {summ_path}")

    try:
        from coevomal.evaluation.plots import plot_frontier

        plot_frontier(summary_rows, out_dir / "frontier.png")
        print(f"wrote {out_dir / 'frontier.png'}")
    except Exception as exc:
        print(f"[warn] frontier plot failed: {exc}")

    return summ_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/ember_factorial.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", type=str, default="results/multiseed")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    base = ExperimentConfig.from_yaml(args.config)
    run_multiseed(base, out_dir=Path(args.out), seeds=args.seeds, quiet=not args.verbose)


if __name__ == "__main__":
    main()
