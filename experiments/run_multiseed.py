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
import time
from pathlib import Path

import numpy as np

from coevomal.config import ExperimentConfig
from coevomal.orchestrator import CoEvolutionOrchestrator

CADENCES = ["every_round", "every_n", "threshold"]
DATA_SELECTION = ["full_replay", "hard_mining", "bounded_buffer"]
MODES = ["scratch"]
# Attacker axis. Defaults to whatever the base config specifies -- hardcoding
# it here would silently override the configured attacker and mislabel every
# result row. Pass --attackers to sweep the attacker axis explicitly.
DEFAULT_ATTACKERS: list[str] | None = None

# Metrics we aggregate across seeds.
METRICS = [
    "oscillation_index",
    "mean_attack_success_tail",
    "final_attack_success_rate",
    "mean_evasion_tail",
    "final_evasion_rate",
    "max_evasion_rate",
    "rounds_to_convergence",
    "total_retrain_seconds",
    "total_trees_fitted",
    "retrain_count",
]


def _policy_key(attacker: str, cadence: str, selection: str, mode: str) -> str:
    """Stable identifier for one factorial cell.

    ``scratch`` is omitted from the key so that keys written before the mode
    axis existed still match, and a resumed sweep does not silently recompute
    every completed cell.
    """
    key = f"{attacker}__{cadence}__{selection}"
    return key if mode == "scratch" else f"{key}__{mode}"


def _load_existing(raw_path: Path) -> list[dict]:
    """Read previously completed rows so a re-run resumes instead of redoing."""
    if not raw_path.exists():
        return []
    rows: list[dict] = []
    with raw_path.open("r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if k in ("policy", "attacker", "cadence", "data_selection"):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def run_multiseed(
    base: ExperimentConfig,
    out_dir: Path,
    seeds: int = 5,
    quiet: bool = True,
    seed_start: int = 0,
    attackers: list[str] | None = None,
    cadences: list[str] | None = None,
    selections: list[str] | None = None,
    modes: list[str] | None = None,
) -> Path:
    """Run the sweep for seeds ``[seed_start, seeds)``, resuming if possible.

    Results are checkpointed after every cell, and any (policy, seed) already
    present in ``multiseed_raw.csv`` is skipped -- so the study can be run in
    stages and re-invoked to extend it without recomputing finished work.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attackers = attackers or DEFAULT_ATTACKERS or [base.attacker.name]
    cadences = cadences or CADENCES
    selections = selections or DATA_SELECTION
    modes = modes or MODES
    combos = list(itertools.product(attackers, cadences, selections, modes))

    # Echo the settings that silently invalidate a sweep if they are wrong:
    # the wrong attacker runs the wrong experiment, early stopping leaves
    # policies on unequal horizons, a trigger above peak evasion turns the
    # threshold policy into the frozen baseline, and an unbound replay cap
    # makes the three data-selection strategies collect identical data.
    print(f"sweep: attackers={attackers} cadences={cadences}", flush=True)
    print(f"       data_selection={selections} modes={modes}", flush=True)
    print(
        f"       rounds={base.rounds} early_stop={base.early_stop} "
        f"trigger={base.retrain.trigger_threshold} "
        f"replay_cap={base.retrain.buffer_size} "
        f"dataset={base.dataset.name} n_train={base.dataset.n_train}",
        flush=True,
    )
    if base.early_stop:
        print(
            "WARNING: early_stop is on -- policies will run for different "
            "numbers of rounds and the cost comparison will not be "
            "like-for-like. Set early_stop: false for a factorial.",
            flush=True,
        )
    raw_path = out_dir / "multiseed_raw.csv"
    raw_rows: list[dict] = _load_existing(raw_path)
    done = {(r["policy"], int(r["seed"])) for r in raw_rows}
    if done:
        print(f"resuming: {len(done)} cells already complete in {raw_path}", flush=True)

    def _flush_raw() -> None:
        """Persist after every cell so a long sweep is never all-or-nothing.

        Uses the *union* of keys across rows: a resumed run may carry rows
        written before METRICS gained a column, and csv.DictWriter raises on
        any key absent from its fieldnames. Taking the union (and filling
        gaps with NaN) means a staged study survives changes to the metric
        set between stages instead of losing the completed work.
        """
        if not raw_rows:
            return
        fieldnames: list[str] = []
        for r in raw_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with raw_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(raw_rows)

    for s in range(seed_start, seeds):
        for atk, cadence, selection, mode in combos:
            policy = _policy_key(atk, cadence, selection, mode)
            if (policy, s) in done:
                print(f"[seed {s}] {policy} -- already done, skipping", flush=True)
                continue
            name = f"{policy}__seed{s}"
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
                    "retrain.mode": mode,
                }
            )
            print(f"[seed {s}] {atk}/{cadence}/{selection} ...", flush=True)
            t_cell = time.perf_counter()
            orch = CoEvolutionOrchestrator(cfg, verbose=not quiet)
            orch.run()
            print(f"    done in {time.perf_counter() - t_cell:.0f}s", flush=True)
            summ = orch.result.summary(cfg.convergence_window, cfg.convergence_tol)
            row = {
                "policy": policy,
                "seed": s,
                "attacker": atk,
                "cadence": cadence,
                "data_selection": selection,
                "mode": mode,
            }
            for m in METRICS:
                v = summ.get(m)
                # rounds_to_convergence is None on divergence -> NaN for stats.
                row[m] = float("nan") if v is None else float(v)
            raw_rows.append(row)
            _flush_raw()   # checkpoint after every cell
        _flush_raw()
        print(f"[seed {s}] complete -> {raw_path} ({len(raw_rows)} rows)", flush=True)

    _flush_raw()
    print(f"wrote {raw_path}")

    # ---- aggregate mean/std per policy -------------------------------------
    summary_rows: list[dict] = []
    for atk, cadence, selection, mode in combos:
        policy = _policy_key(atk, cadence, selection, mode)
        cells = [r for r in raw_rows if r["policy"] == policy]
        if not cells:
            continue
        agg = {"policy": policy, "attacker": atk, "cadence": cadence,
               "data_selection": selection, "mode": mode, "n_seeds": len(cells)}
        for m in METRICS:
            vals = np.array(
                [r.get(m, float("nan")) for r in cells], dtype=float
            )
            if vals.size == 0 or np.all(np.isnan(vals)):
                agg[f"{m}_mean"] = float("nan")
                agg[f"{m}_std"] = float("nan")
            else:
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
    p.add_argument("--seeds", type=int, default=5, help="run seeds [seed-start, seeds)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", type=str, default="results/multiseed")
    p.add_argument("--attackers", type=str, default=None,
                   help="comma-separated attacker axis; defaults to the config's")
    p.add_argument("--cadences", type=str, default=None,
                   help="comma-separated cadence axis (e.g. never for the frozen baseline)")
    p.add_argument("--selections", type=str, default=None,
                   help="comma-separated data-selection axis")
    p.add_argument("--modes", type=str, default=None,
                   help="comma-separated retrain-mode axis (scratch,finetune)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    base = ExperimentConfig.from_yaml(args.config)
    split = lambda v: v.split(",") if v else None
    run_multiseed(base, out_dir=Path(args.out), seeds=args.seeds,
                  quiet=not args.verbose, seed_start=args.seed_start,
                  attackers=split(args.attackers), cadences=split(args.cadences),
                  selections=split(args.selections), modes=split(args.modes))


if __name__ == "__main__":
    main()
